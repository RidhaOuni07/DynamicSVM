"""
Unit tests for IncrementalDecrementalCOSVM (IDCOSVM).

Three groups: the original smoke tests, one regression test per P0 defect
of the mathematical review (indefinite kernel at eta = 1, learn_one from an
empty buffer, drained counters, the single active-set tolerance, the
_kkt_residuals schema and small eta), and the mathematical properties -
equivalence with a cold COSVM after every insertion and deletion,
duplicate-row symmetry and warm-start independence.

IDCoSVM rebuilds and re-solves in O(N^3) per update, so every buffer here
stays at N <= 60 and the insert/delete sweep uses two seeds.
"""

import numpy as np
import pytest

from dynamicsvm.kernels import resolve_gamma
from dynamicsvm.models._cosvm_core import compute_rho
from dynamicsvm.models.cosvm import COSVM
from dynamicsvm.stream.idcosvm import IncrementalDecrementalCOSVM, IDCOSVM
from dynamicsvm.utils.kkt import oneclass_kkt_report


def test_idcosvm_fit_and_incremental_stream():
    np.random.seed(42)
    # Generate elongated target distribution
    X_init = np.random.randn(35, 2) * np.array([[2.5, 0.4]])

    model = IncrementalDecrementalCOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale")
    model.fit(X_init)

    assert model.is_fitted
    assert len(model.X_buffer_) == 35
    assert len(model.alpha_) == 35
    assert np.all(model.alpha_ >= 0)
    assert np.isclose(np.sum(model.alpha_), 1.0, atol=1e-12)

    # Stream in 15 new instances
    X_stream = np.random.randn(15, 2) * np.array([[2.5, 0.4]])
    model.learn_batch(X_stream)

    assert len(model.X_buffer_) == 50
    assert len(model.alpha_) == 50
    assert model.n_samples_seen_ == 50
    assert np.all(model.alpha_ >= 0)
    assert np.isclose(np.sum(model.alpha_), 1.0, atol=1e-12)

    state = model.get_model_state()
    assert state["buffer_size"] == 50
    assert state["n_samples_seen"] == 50


def test_idcosvm_decremental_unlearning_specific_indices():
    np.random.seed(42)
    X = np.random.randn(30, 2) * np.array([[3.0, 0.5]])

    model = IDCOSVM(nu=0.15, eta=0.5, kernel="rbf", gamma="scale")
    model.fit(X)

    init_len = len(model.X_buffer_)
    assert init_len == 30

    # Unlearn first instance
    model.unlearn_one(0)
    assert len(model.X_buffer_) == 29
    assert len(model.alpha_) == 29
    assert model.n_unlearned_ == 1
    assert np.isclose(np.sum(model.alpha_), 1.0, atol=1e-12)

    # Unlearn multiple instances
    model.unlearn_many([5, 10, 15])
    assert len(model.X_buffer_) == 26
    assert len(model.alpha_) == 26
    assert model.n_unlearned_ == 4
    assert np.isclose(np.sum(model.alpha_), 1.0, atol=1e-12)


def test_idcosvm_unlearn_sample_by_value():
    np.random.seed(42)
    X = np.random.randn(25, 3)

    model = IDCOSVM(nu=0.1, kernel="linear")
    model.fit(X)

    target_sample = X[7].copy()
    model.unlearn_sample(target_sample)

    assert len(model.X_buffer_) == 24
    assert model.n_unlearned_ == 1

    # Attempting to unlearn already removed sample should raise ValueError
    with pytest.raises(ValueError):
        model.unlearn_sample(np.array([100.0, 100.0, 100.0]))


def test_idcosvm_pruning_strategies():
    np.random.seed(42)
    X_init = np.random.randn(30, 2)

    # Test non_sv_first pruning strategy
    model_non_sv = IncrementalDecrementalCOSVM(
        nu=0.1, eta=0.5, max_samples=25, pruning_strategy="non_sv_first"
    )
    model_non_sv.fit(X_init)
    assert len(model_non_sv.X_buffer_) == 25

    # Test mahalanobis pruning strategy
    model_mah = IncrementalDecrementalCOSVM(
        nu=0.1, eta=0.5, max_samples=20, pruning_strategy="mahalanobis"
    )
    model_mah.fit(X_init)
    assert len(model_mah.X_buffer_) == 20

    # Stream in 10 more points - buffer should stay bounded at 20
    X_extra = np.random.randn(10, 2)
    model_mah.learn_many(X_extra)
    assert len(model_mah.X_buffer_) == 20


def test_idcosvm_classification_accuracy():
    np.random.seed(42)
    theta = np.pi / 4
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    # Target distribution
    X_train_base = np.random.randn(60, 2) * np.array([[3.5, 0.4]])
    X_train = np.dot(X_train_base, rot.T)

    model = IDCOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale")
    model.fit(X_train[:40])

    # Incrementally stream the remaining 20 samples
    for i in range(40, 60):
        model.learn_one(X_train[i])

    # Inliers test
    X_test_inliers = np.dot(np.random.randn(25, 2) * np.array([[3.5, 0.4]]), rot.T)
    y_pred_inliers = model.predict(X_test_inliers)
    assert np.mean(y_pred_inliers == 1) >= 0.55

    # Outliers test far off
    X_test_outliers = np.dot(np.random.randn(25, 2) * np.array([[0.5, 3.5]]), rot.T) + np.array([8.0, -8.0])
    y_pred_outliers = model.predict(X_test_outliers)
    assert np.mean(y_pred_outliers == -1) >= 0.85


# =====================================================================
# Helpers for the regression and property tests
# =====================================================================

# One fixed probe set per regime, drawn from this seed, so that
# decision-function agreement is always measured on the same points.
_PROBE_SEED = 7


def _config(**overrides):
    """
    A COSVM-family configuration with a NUMERIC gamma.

    Every cross-model comparison below pins gamma to a float on purpose:
    COSVM resolves gamma='scale' from its own batch, so a string gamma
    would give the cold reference a different kernel from the streamed
    model and the comparison would measure the bandwidth rule instead of
    the dual solution.
    """
    config = dict(nu=0.2, eta=0.4, kernel="rbf", gamma=0.5)
    config.update(overrides)
    return config


def _kkt_report(alpha, G, C):
    """
    Scale-aware optimality report of a one-class dual point.

    eps = 1e-12 * C is the solver's snap level, well below the
    sv_tol * C that classifies the active sets, so the free set the gap
    is taken over is the numerically exact one.
    """
    return oneclass_kkt_report(alpha, G, C, eps=1e-12 * C)


def _assert_cold_cosvm_agreement(model, X_retained, probe, config, bounds):
    """
    Assert a streamed IDCoSVM matches a cold COSVM on the retained buffer.

    Feasibility and optimality of the streamed point are checked first
    (sum(alpha) = 1 to 1e-12, no box violation, KKT gap below
    1e-10 * max_k G_kk), then the cross-model distances - dual
    coefficients, offset rho relative to max(1, |rho|), the decision
    function on `probe`, and the dual objective relative to max_k G_kk -
    against `bounds` = (alpha, rho, f).

    An alpha bound of None means alpha is NOT identified in that regime:
    when G is singular (duplicated rows, or a linear kernel with
    n_features < n_samples) the dual optimum is a face, not a point, so
    two solvers may return different alphas with the same objective. The
    identified quantities - objective, rho and the decision function -
    then carry the comparison.

    Returns the achieved values so a sweep can report its maxima.
    """
    G = model.K_tilde_full_
    C = 1.0 / (config["nu"] * len(X_retained))
    scale = float(np.max(np.diag(G)))

    report = _kkt_report(model.alpha_, G, C)
    assert report["sum_alpha_minus_1"] <= 1e-12
    assert report["box_violation"] == 0.0
    assert report["kkt_gap"] <= 1e-10 * scale

    ref = COSVM(**config).fit(X_retained)
    ref_report = _kkt_report(ref.alpha_, ref.K_tilde_, ref.C_)

    achieved = {
        "d_alpha": float(np.max(np.abs(model.alpha_ - ref.alpha_))),
        "d_rho": abs(model.offset_ - ref.rho_) / max(1.0, abs(ref.rho_)),
        "d_f": float(
            np.max(
                np.abs(
                    model.decision_function(probe) - ref.decision_function(probe)
                )
            )
        ),
        "d_objective": abs(
            report["dual_objective"] - ref_report["dual_objective"]
        )
        / scale,
        "sum_alpha_minus_1": report["sum_alpha_minus_1"],
        "kkt_gap_relative": report["kkt_gap"] / scale,
    }

    bound_alpha, bound_rho, bound_f = bounds
    if bound_alpha is not None:
        assert achieved["d_alpha"] <= bound_alpha
    assert achieved["d_rho"] <= bound_rho
    assert achieved["d_f"] <= bound_f
    # The optimal value is identified in every regime (measured <= 2.4e-16
    # relative to max_k G_kk over both sweep seeds), including the ones
    # where alpha is not.
    assert achieved["d_objective"] <= 1e-14

    return achieved


# =====================================================================
# Regression tests - one per P0 defect of the mathematical review
# =====================================================================


def test_idcosvm_eta1_rejects_indefinite_kernel():
    """(i) The eta = 1 short-circuit certifies G = K is PSD.

    Review defect 1: at eta = 1 the shared geometry returned K unchecked,
    so IDCoSVM minimised a non-convex quadratic (lambda_min(G) = -15.49 on
    this data) where COSVM raised. Both models must now refuse it, and the
    PSD path must stay bit-identical.
    """
    X = np.random.RandomState(0).randn(25, 2)
    indefinite = dict(nu=0.1, eta=1.0, kernel="sigmoid", gamma=0.5, coef0=-1.0)

    for estimator in (COSVM, IDCOSVM):
        with pytest.raises(ValueError, match="not positive semidefinite"):
            estimator(**indefinite).fit(X)

    model = IDCOSVM(nu=0.1, eta=1.0, kernel="rbf", gamma=0.5).fit(X)
    assert np.array_equal(model.K_tilde_full_, model.K_full_)


def test_idcosvm_learn_one_from_empty_resolves_gamma_once():
    """(ii) learn_one on an empty buffer is a cold start, not a re-fit.

    Review defect 2: it delegated to fit, which froze gamma='scale' on the
    1/n_features fallback for the life of the model. The bandwidth must now
    be the one resolved from the first buffer of size >= 2.

    The cold reference is fitted with THAT numeric gamma: a 'scale' value
    resolved from 2 rows cannot equal one resolved from 21, and
    re-resolving at every N would break the frozen-gamma contract
    documented at kernels.py:5-11. The assertion is therefore 'equal
    kernels give equal duals'.
    """
    X = np.random.RandomState(3).randn(21, 2) * 5.0
    model = IDCOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale")

    with pytest.warns(UserWarning, match="fewer than 2 samples"):
        model.learn_one(X[0])

    for row in X[1:]:
        model.learn_one(row)

    gamma = model.kernel_.gamma
    assert gamma != pytest.approx(1.0 / X.shape[1])
    assert gamma == resolve_gamma("scale", X=X[:2])

    ref = COSVM(nu=0.1, eta=0.4, kernel="rbf", gamma=gamma).fit(X)
    assert np.max(np.abs(model.alpha_ - ref.alpha_)) <= 1e-9


def test_idcosvm_counters_survive_a_drain_and_refill():
    """(iii) Lifetime counters are history, not buffer state.

    Review defect 2 (second half): the empty-buffer path reset
    n_samples_seen_ / n_unlearned_, so a drained model forgot it had ever
    seen data.
    """
    X = np.random.RandomState(0).randn(6, 2)
    model = IDCOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma=0.5).fit(X)

    model.unlearn_many(range(6))
    model.learn_one(X[0])

    assert model.n_samples_seen_ == 7
    assert model.n_unlearned_ == 6


def test_idcosvm_has_one_active_set_tolerance():
    """(iv) S_set_ and compute_rho share the single tolerance sv_tol * C.

    Review defect 3: the sets were cut at an absolute 1e-5 while rho was
    averaged over the 1e-8 * C free set, which was worth 4.40e-03 relative
    in rho. With one tolerance that discrepancy is structurally
    unreachable, so rho is reproduced bit-for-bit.
    """
    X = np.random.RandomState(0).randn(40, 2)
    model = IDCOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma=0.5).fit(X)

    C = 1.0 / (0.1 * 40)
    alpha_tol = model.sv_tol * C
    free_set = {
        i
        for i, alpha_i in enumerate(model.alpha_)
        if alpha_tol < alpha_i < C - alpha_tol
    }

    assert set(model.S_set_) == free_set
    assert (
        compute_rho(model.K_tilde_full_, model.alpha_, C, alpha_tol=alpha_tol)
        == model.offset_
    )


def test_idcosvm_kkt_diagnostics_schema_is_stable():
    """(v) One residual schema, and the certificate the solver accepted on.

    Review defect: _kkt_residuals returned two incompatible key sets
    depending on whether the buffer was empty, and get_kkt_diagnostics
    recomputed an unscaled summary of its own.
    """
    X = np.random.RandomState(0).randn(20, 2)
    fitted = IDCOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma=0.5).fit(X)

    drained = IDCOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma=0.5).fit(X)
    drained.unlearn_many(range(20))

    assert set(fitted._kkt_residuals()) == set(drained._kkt_residuals())

    diagnostics = fitted.get_kkt_diagnostics()
    expected = {
        "kkt_gap",
        "kkt_gap_relative",
        "sum_alpha_minus_1",
        "box_violation",
        "max_stationarity_error",
        "max_lower_kkt_violation",
        "max_upper_kkt_violation",
        "sum_alpha_error",
    }
    assert expected <= set(diagnostics)


@pytest.mark.parametrize("eta", [1e-7, 1e-8, 1e-9])
def test_idcosvm_fits_at_small_eta(eta):
    """(vi) Small eta is a tolerance scale, not a mathematical boundary.

    Review defect: |G| grows like 1/eta while the solver's feasibility
    tolerance was absolute, so these three etas raised RuntimeError. All
    three are pinned because the pre-fix failure was non-monotonic.
    """
    X = np.random.RandomState(0).randn(25, 2)
    model = IDCOSVM(nu=0.1, eta=eta, kernel="rbf", gamma=0.5).fit(X)

    C = 1.0 / (0.1 * 25)
    report = _kkt_report(model.alpha_, model.K_tilde_full_, C)

    assert report["sum_alpha_minus_1"] <= 1e-12
    assert report["box_violation"] == 0.0


# =====================================================================
# Insert / delete equivalence with a cold COSVM
# =====================================================================


def _sweep_pool(kind, seed):
    """
    Row pool and fixed 30-point probe set for one structural regime.
    """
    rng = np.random.RandomState(seed)

    if kind == "gauss2":
        pool = rng.randn(36, 2) * np.array([[2.0, 0.5]])
    elif kind == "dup3":
        # Every distinct row appears three times, so the Gram matrix has
        # exactly repeated rows and columns.
        pool = np.repeat(rng.randn(12, 2) * np.array([[2.0, 0.5]]), 3, axis=0)
    elif kind == "rank3":
        # N up to 24 rows in d = 3: the linear Gram has rank <= 3.
        pool = rng.randn(30, 3)
    elif kind == "wide":
        # d >> N.
        pool = rng.randn(16, 200)
    else:
        raise AssertionError(f"Unknown pool kind {kind!r}")

    probe = np.random.RandomState(_PROBE_SEED).randn(30, pool.shape[1])

    return pool, probe


def _run_edit_sweep(config, kind, bounds, seed=0, n_init=12, n_ops=10):
    """
    Apply a seeded insert/delete sequence and check every intermediate model.

    After EVERY single operation the retained buffer is re-fitted from
    scratch by a cold COSVM and the two models are compared, so the
    equivalence is a property of each state rather than of the end point.
    The buffer is kept inside 8 <= N <= 30: COSVM needs N >= 2 and
    IDCoSVM is O(N^3) per update.
    """
    pool, probe = _sweep_pool(kind, seed)
    rng = np.random.RandomState(seed + 1)

    model = IDCOSVM(**config).fit(pool[:n_init])
    retained = list(range(n_init))
    next_row = n_init

    worst = {}

    for _ in range(n_ops):
        n_samples = len(retained)
        can_insert = next_row < len(pool) and n_samples < 30
        can_delete = n_samples > 8

        if can_insert and (not can_delete or rng.rand() < 0.6):
            model.learn_one(pool[next_row])
            retained.append(next_row)
            next_row += 1
        else:
            victim = rng.randint(n_samples)
            model.unlearn_one(victim)
            retained.pop(victim)

        achieved = _assert_cold_cosvm_agreement(
            model, pool[retained], probe, config, bounds
        )

        for key, value in achieved.items():
            worst[key] = max(worst.get(key, 0.0), value)

    return worst


# Per-regime bounds on (max|delta alpha|, |delta rho| / max(1, |rho|),
# max|delta f| on the probe set), measured over both sweep seeds and
# tightened to the achieved level, one order of magnitude of cushion.
#
# With a strictly positive definite G the streamed and the cold dual are
# bit-identical: max|delta alpha| is 0.0 in nine of the fourteen regimes
# and 3.8e-16 in the worst (nu = 0.9).
#
# The three regimes with alpha = None have a SINGULAR G - exactly repeated
# rows, or a linear kernel whose Gram has rank min(d, N) = d < N - so the
# dual optimum is a face and alpha is not identified: the two solvers land
# up to 2.1e-01 apart in alpha while agreeing to 1.2e-12 on rho, 1.2e-11
# on the decision function and 5.1e-17 on the objective. That is dual
# non-uniqueness, not disagreement, and it is the reason this sweep
# asserts the function rather than the coefficients there.
_SWEEP_BOUNDS = {
    "eta_1": (1e-14, 1e-14, 1e-13),
    "eta_0.9": (1e-14, 1e-14, 1e-13),
    "eta_0.4": (1e-14, 1e-14, 1e-13),
    "eta_1e-2": (1e-14, 1e-14, 1e-13),
    "eta_1e-4": (1e-14, 1e-14, 1e-13),
    "nu_0.05": (1e-14, 1e-14, 1e-13),
    "nu_0.2": (1e-14, 1e-14, 1e-13),
    "nu_0.9": (1e-14, 1e-14, 1e-13),
    "linear": (None, 1e-11, 1e-10),
    "rbf": (1e-14, 1e-14, 1e-13),
    "poly": (1e-12, 1e-13, 1e-12),
    "duplicated_rows": (None, 1e-11, 1e-10),
    "rank_deficient_linear": (None, 1e-11, 1e-10),
    "d_greater_than_n": (1e-14, 1e-14, 1e-13),
}

_SWEEP_CASES = [
    # eta sweep at nu = 0.2, rbf
    ("eta_1", _config(eta=1.0), "gauss2"),
    ("eta_0.9", _config(eta=0.9), "gauss2"),
    ("eta_0.4", _config(eta=0.4), "gauss2"),
    ("eta_1e-2", _config(eta=1e-2), "gauss2"),
    ("eta_1e-4", _config(eta=1e-4), "gauss2"),
    # nu sweep at eta = 0.4, rbf
    ("nu_0.05", _config(nu=0.05), "gauss2"),
    ("nu_0.2", _config(nu=0.2), "gauss2"),
    ("nu_0.9", _config(nu=0.9), "gauss2"),
    # kernel sweep at eta = 0.4, nu = 0.2
    ("linear", _config(kernel="linear"), "gauss2"),
    ("rbf", _config(kernel="rbf"), "gauss2"),
    ("poly", _config(kernel="poly", degree=3, coef0=1.0), "gauss2"),
    # structural regimes
    ("duplicated_rows", _config(), "dup3"),
    ("rank_deficient_linear", _config(kernel="linear"), "rank3"),
    ("d_greater_than_n", _config(kernel="linear"), "wide"),
]


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(
    "case, config, kind",
    _SWEEP_CASES,
    ids=[case for case, _, _ in _SWEEP_CASES],
)
def test_idcosvm_insert_delete_equals_cold_cosvm(case, config, kind, seed):
    worst = _run_edit_sweep(config, kind, _SWEEP_BOUNDS[case], seed=seed)

    # Printed so the bounds above can be re-measured with -s.
    print(
        f"[sweep {case} seed={seed}] "
        + " ".join(f"{key}={worst[key]:.3e}" for key in sorted(worst))
    )


# =====================================================================
# Positive properties
# =====================================================================


def test_idcosvm_duplicate_rows_get_equal_alpha():
    """
    Identical rows are interchangeable in the dual, so they share alpha.

    Only approximately, and the tolerance says why: repeated rows make G
    singular, so any redistribution of alpha inside a triple is also
    optimal and the symmetric point is one optimum among infinitely many.
    The solver lands 1.6e-06 from it on this draw (1e-05 over seeds 0-3,
    i.e. 4e-04 relative to alpha_i ~ 0.026) - looser than the 1.6e-08 the
    review measured on its own draw, and recorded in the FEAT-005 findings
    rather than hidden. Feasibility, by contrast, is exact to 1e-16.
    """
    base = np.random.RandomState(0).randn(10, 2) * np.array([[2.0, 0.5]])
    X = np.repeat(base, 3, axis=0)

    model = IDCOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma=0.5).fit(X)

    triples = model.alpha_.reshape(10, 3)
    for triple in triples:
        np.testing.assert_allclose(triple, triple.mean(), rtol=0.0, atol=5e-6)

    assert abs(float(np.sum(model.alpha_)) - 1.0) <= 1e-12


def test_idcosvm_is_warm_start_independent():
    """
    The dual point depends on the buffer, not on how it was assembled.

    Two routes to the same 30 rows (one batch fit, one small fit plus
    learn_one) and one random insertion permutation all have to land on the
    same solution; the permuted model is compared after mapping its buffer
    back to the canonical row order, with the decision function on a fixed
    probe set as the order-independent cross-check.
    """
    rng = np.random.RandomState(0)
    X = rng.randn(30, 2) * np.array([[2.0, 0.5]])
    probe = np.random.RandomState(_PROBE_SEED).randn(30, 2)
    config = _config(nu=0.1)

    batch = IDCOSVM(**config).fit(X)

    streamed = IDCOSVM(**config).fit(X[:10])
    for row in X[10:]:
        streamed.learn_one(row)

    assert np.max(np.abs(streamed.alpha_ - batch.alpha_)) <= 1e-12

    permutation = rng.permutation(30)
    permuted = IDCOSVM(**config).fit(X[permutation[:10]])
    for index in permutation[10:]:
        permuted.learn_one(X[index])

    # permuted.alpha_[k] belongs to row permutation[k] of X.
    alpha_canonical = np.empty(30)
    alpha_canonical[permutation] = permuted.alpha_

    assert np.max(np.abs(alpha_canonical - batch.alpha_)) <= 1e-12

    f_batch = batch.decision_function(probe)
    np.testing.assert_allclose(
        permuted.decision_function(probe),
        f_batch,
        rtol=0.0,
        atol=1e-12 * max(1.0, float(np.abs(f_batch).max())),
    )

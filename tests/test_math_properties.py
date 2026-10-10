"""
Mathematical property tests (optimality, invariance, calibration).

One file, sectioned per area of the math-rigor plan. Tests that need
scikit-learn as a reference start with ``pytest.importorskip("sklearn")``.
"""

import warnings

import numpy as np
import pytest

from dynamicsvm.kernels import (
    PolynomialKernel,
    RBFKernel,
    SigmoidKernel,
    get_kernel,
    resolve_gamma,
)
from dynamicsvm.models.cosvm import COSVM
from dynamicsvm.models.covariance_svc import CovarianceAwareSVC
from dynamicsvm.models.multiclass import MultiClassSVC
from dynamicsvm.models.svc import SVC
from dynamicsvm.solvers.lasvm import LASVM
from dynamicsvm.solvers.pegasos import PegasosSolver
from dynamicsvm.solvers.qp import QPSolver
from dynamicsvm.solvers.smo import SMOSolver
from dynamicsvm.encoders import IdentityEncoder
from dynamicsvm.stream.adiabatic_svm import AdiabaticSVM
from dynamicsvm.stream.covariance import LatentCovarianceDriftDetector, OnlineCovarianceTracker
from dynamicsvm.stream.drift import DDMDriftDetector
from dynamicsvm.stream.drift_covariance_svm import DriftAwareCovarianceSVM
from dynamicsvm.stream.icosvm import ICOSVM, _sensitivity_step
from dynamicsvm.stream.idcosvm import IDCOSVM
from dynamicsvm.utils.calibration import fit_platt_sigmoid
from dynamicsvm.utils.kkt import intercept_interval, kkt_gap, kkt_report, oneclass_kkt_report
from dynamicsvm.utils.validation import check_array


# =====================================================================
# Area A: kernels, validation, binary batch models and solvers
# =====================================================================


def _moons_scaled():
    """E1 setup: moons with clearly non-unit variance (gamma='scale' != 1)."""
    pytest.importorskip("sklearn")
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=120, noise=0.25, random_state=0)
    return X * np.array([3.0, 0.5]) + 2.0, y


def _sk_alpha(sk_model, n):
    a = np.zeros(n)
    a[sk_model.support_] = np.abs(sk_model.dual_coef_.ravel())
    return a


def test_kernel_gamma_validation():
    rng = np.random.RandomState(0)
    X = rng.randn(30, 3) * 2.0 + 1.0
    d = X.shape[1]

    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            RBFKernel(gamma=bad)
        with pytest.raises(ValueError):
            get_kernel("rbf", gamma=bad)
    with pytest.raises(ValueError):
        RBFKernel(gamma="median")
    with pytest.raises(ValueError):
        get_kernel("poly", gamma="bogus", X=X)
    with pytest.raises(ValueError):
        PolynomialKernel(degree=0)
    with pytest.raises(ValueError):
        PolynomialKernel(degree=2.5)

    # An unresolved string gamma is never resolved from the query batch.
    for kern in (RBFKernel(gamma="scale"), PolynomialKernel(gamma="auto"), SigmoidKernel(gamma="scale")):
        with pytest.raises(ValueError, match="unresolved"):
            kern(X)

    k = get_kernel("rbf", X=X)
    assert k.gamma == pytest.approx(1.0 / (d * X.var()), rel=1e-12)
    assert get_kernel("rbf", gamma="auto", X=X).gamma == pytest.approx(1.0 / d)
    # Kernel instances with a string gamma are resolved on a copy.
    inst = RBFKernel(gamma="scale")
    resolved = get_kernel(inst, X=X)
    assert resolved is not inst and isinstance(inst.gamma, str)
    assert resolved.gamma == pytest.approx(k.gamma)

    # D2: 'scale' from a single row is not identifiable -> warning + 1/d.
    with pytest.warns(UserWarning):
        g1 = resolve_gamma("scale", X=X[:1])
    assert g1 == pytest.approx(1.0 / d)


def test_check_array_1d_with_n_features():
    v = np.arange(4.0)
    assert check_array(v, n_features=4).shape == (1, 4)
    assert check_array(v, n_features=1).shape == (4, 1)
    assert check_array(v).shape == (4, 1)  # legacy fit behaviour
    with pytest.raises(ValueError):
        check_array(v, n_features=3)
    with pytest.raises(ValueError):
        check_array(np.zeros((5, 3)), n_features=4)


def _assert_batch_invariant(model, X):
    df_batch = np.asarray(model.decision_function(X))
    rows = [np.asarray(model.decision_function(X[i : i + 1])) for i in range(len(X))]
    df_rows = np.vstack(rows) if df_batch.ndim == 2 else np.concatenate(rows)
    np.testing.assert_allclose(df_batch, df_rows, rtol=0.0, atol=1e-10)
    # A 1-D row is one sample of n_features_in_ features.
    np.testing.assert_array_equal(model.predict(X[0]), model.predict(X[:1]))


def test_batch_invariance_svc_multiclass_covsvc():
    pytest.importorskip("sklearn")
    from sklearn.datasets import make_blobs

    X, y = _moons_scaled()

    svc = SVC(C=1.0, kernel="rbf", gamma="scale", random_state=0).fit(X, y)
    _assert_batch_invariant(svc, X)

    cov_svc = CovarianceAwareSVC(C=1.0, eta=0.6, kernel="rbf", gamma="scale", random_state=0).fit(X, y)
    _assert_batch_invariant(cov_svc, X)

    X3, y3 = make_blobs(n_samples=90, centers=3, cluster_std=1.5, random_state=0)
    X3 = X3 * np.array([3.0, 0.5]) + 2.0
    expected_gamma = 1.0 / (X3.shape[1] * X3.var())
    for strategy in ("ovr", "ovo"):
        mc = MultiClassSVC(C=1.0, kernel="rbf", strategy=strategy, gamma="scale", random_state=0).fit(X3, y3)
        _assert_batch_invariant(mc, X3)
        # gamma resolved once on the full X and shared by every sub-problem
        for est in mc.estimators_:
            assert est.kernel_.gamma == pytest.approx(expected_gamma, rel=1e-12)


def test_covsvc_train_decision_consistency():
    X, y = _moons_scaled()
    model = CovarianceAwareSVC(C=1.0, eta=0.6, kernel="rbf", gamma="scale", random_state=0).fit(X, y)

    Q = model._compute_Q(model.K_train_, model.H_)
    expected = Q @ (model.alpha_full_ * model.y_encoded_) + model.intercept_
    np.testing.assert_allclose(model.decision_function(X), expected, rtol=0.0, atol=1e-8)

    # H is PSD by construction (centred Mahalanobis Gram) and H 1 = 0.
    assert np.linalg.eigvalsh(model.H_).min() > -1e-10 * max(1.0, np.abs(model.H_).max())
    np.testing.assert_allclose(model.H_ @ np.ones(len(X)), 0.0, atol=1e-8)


def test_smo_qp_match_sklearn_precomputed():
    pytest.importorskip("sklearn")
    from sklearn.datasets import make_moons
    from sklearn.svm import SVC as SkSVC

    X, y01 = make_moons(n_samples=120, noise=0.25, random_state=0)
    y = np.where(y01 == 1, 1.0, -1.0)
    C = 2.0
    K = RBFKernel(gamma=1.0)(X)

    sk = SkSVC(C=C, kernel="precomputed", tol=1e-8).fit(K, y)
    dual_ref = kkt_report(_sk_alpha(sk, len(y)), K, y, C)["dual_objective"]

    # Audit E2: QP at the old SVC default had kkt_gap 1.3e-2; require >= 100x tighter.
    for solver in (SMOSolver(C=C, tol=1e-5, max_iter=10000, random_state=0), QPSolver(C=C)):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # must converge without warnings
            alpha, b, _ = solver.solve(K, y)
        report = kkt_report(alpha, K, y, C)
        assert report["sum_alpha_y"] < 1e-8
        assert report["box_violation"] == 0.0
        assert report["kkt_gap"] < 1e-4
        assert abs(report["dual_objective"] - dual_ref) <= 1e-6 * abs(dual_ref)
        assert solver.kkt_gap_ == pytest.approx(report["kkt_gap"])


def test_smo_intercept_no_free_svs():
    pytest.importorskip("sklearn")
    from sklearn.datasets import make_blobs

    # Audit E2b: tiny C, overlapping blobs -> every alpha at a bound.
    X, y01 = make_blobs(n_samples=60, centers=[[0, 0], [1, 0]], cluster_std=1.0, random_state=1)
    y = np.where(y01 == 1, 1.0, -1.0)
    K = X @ X.T
    C = 0.001
    alpha, b, _ = SMOSolver(C=C, tol=1e-6, random_state=0).solve(K, y)

    assert np.all((alpha == 0.0) | (alpha == C)), "expected no free SVs in this setup"
    L, U = intercept_interval(alpha, K, y, C, eps=1e-8 * C)
    assert np.isfinite(L) and np.isfinite(U) and L <= U
    assert L - 1e-12 <= b <= U + 1e-12
    assert b == pytest.approx(0.5 * (L + U), abs=1e-12)


def test_pegasos_bias_matches_linearsvc():
    pytest.importorskip("sklearn")
    from sklearn.datasets import make_blobs
    from sklearn.svm import LinearSVC

    # Audit E3: off-origin blobs, bias matters.
    X, y01 = make_blobs(n_samples=200, centers=[[0, 0], [2.0, 2.0]], cluster_std=1.0, random_state=3)
    X = X + np.array([5.0, 5.0])
    y = np.where(y01 == 1, 1.0, -1.0)
    C, s = 1.0, 1.0

    def primal(w, b):
        # Regularised-bias objective shared by Pegasos (D7) and liblinear
        return 0.5 * (w @ w + (b / s) ** 2) + C * np.maximum(0.0, 1.0 - y * (X @ w + b)).sum()

    ref = LinearSVC(loss="hinge", C=C, intercept_scaling=s, tol=1e-8, max_iter=200000).fit(X, y)
    p_ref = primal(ref.coef_.ravel(), ref.intercept_[0])

    w, b, _ = PegasosSolver(C=C, max_iter=500, intercept_scaling=s, averaging=True, random_state=0).solve(X, y)
    assert primal(w, b) <= 1.02 * p_ref


def test_platt_recovers_parameters():
    rng = np.random.RandomState(0)
    A, B = -2.0, 0.5
    f = rng.randn(4000) * 1.5
    p = 1.0 / (1.0 + np.exp(A * f + B))
    y01 = (rng.rand(4000) < p).astype(int)
    A_hat, B_hat = fit_platt_sigmoid(f, y01)
    assert abs(A_hat - A) < 0.3
    assert abs(B_hat - B) < 0.3


def test_svc_probability_calibrated():
    X, y = _moons_scaled()
    model = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=0).fit(X, y)
    assert model.probA_ is not None and model.probB_ is not None
    assert model.probA_ < 0.0  # P(positive) increases with the decision value

    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
    order = np.argsort(model.decision_function(X))
    assert np.all(np.diff(proba[order, 1]) >= -1e-15)

    # Deterministic for a fixed random_state
    again = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=0).fit(X, y)
    assert (again.probA_, again.probB_) == (model.probA_, model.probB_)


# =====================================================================
# Area B: exact incremental/decremental binary SVMs
# =====================================================================


def _moons_pm1(n=120):
    pytest.importorskip("sklearn")
    from sklearn.datasets import make_moons

    X, y01 = make_moons(n_samples=n, noise=0.25, random_state=0)
    return X, np.where(y01 == 1, 1.0, -1.0)


def _sk_precomputed(K, y, C):
    from sklearn.svm import SVC as SkSVC

    return SkSVC(C=C, kernel="precomputed", tol=1e-8).fit(K, y)


def _assert_adiabatic_optimal(model, C, X_eval=None):
    y = np.asarray(model.y_buffer_)
    K = model.K_full_
    report = kkt_report(model.alpha_, K, y, C)
    assert report["sum_alpha_y"] < 1e-8
    assert report["box_violation"] < 1e-10
    assert report["kkt_gap"] < 1e-4
    sk = _sk_precomputed(K, y, C)
    dual_ref = kkt_report(_sk_alpha(sk, len(y)), K, y, C)["dual_objective"]
    assert abs(report["dual_objective"] - dual_ref) <= 1e-6 * abs(dual_ref)
    # every buffered index in exactly one of S / E / O
    assert sorted(model.S_set_ + model.E_set_ + model.O_set_) == list(range(len(y)))
    assert model.n_smo_fallbacks_ == 0
    if X_eval is not None:
        f_ref = sk.decision_function(model.kernel_(X_eval, np.asarray(model.X_buffer_)))
        assert np.max(np.abs(model.decision_function(X_eval) - f_ref)) < 1e-3


def test_adiabatic_inserts_match_batch():
    X, y = _moons_pm1()
    perm = np.random.RandomState(0).permutation(len(X))
    X, y = X[perm], y[perm]
    C = 2.0
    model = AdiabaticSVM(C=C, kernel="rbf", gamma=1.0, tol=1e-6).fit(X[:40], y[:40])
    for i in range(40, 80):
        model.learn_one(X[i], y[i])
    assert len(model.y_buffer_) == 80
    _assert_adiabatic_optimal(model, C)


def test_adiabatic_forget_margin_and_error_sv():
    X, y = _moons_pm1()
    perm = np.random.RandomState(0).permutation(len(X))
    X, y = X[perm], y[perm]
    C = 2.0
    model = AdiabaticSVM(C=C, kernel="rbf", gamma=1.0, tol=1e-6).fit(X[:60], y[:60])
    assert model.S_set_ and model.E_set_

    # Audit E5: |sum a y| was 0.715 after removing a margin SV and 2.715 after an error SV.
    model.forget_one(model.S_set_[0])
    _assert_adiabatic_optimal(model, C, X_eval=X)
    assert model.E_set_
    model.forget_one(model.E_set_[0])
    _assert_adiabatic_optimal(model, C, X_eval=X)
    assert model.get_model_state()["n_smo_fallbacks"] == 0


def _lasvm_moons():
    X, y = _moons_pm1()
    model = LASVM(C=2.0, kernel="rbf", gamma=1.0, epochs=2, random_state=0).fit(X, y)
    return X, y, model


def _lasvm_working_set_gradients(model):
    a = np.asarray(model.SV_alpha_)
    y = np.asarray(model.SV_y_)
    K = model.kernel_(np.asarray(model.SV_X_))
    return a, y, K, y - K @ (a * y)


def test_lasvm_forget_feasible_and_gradients_exact():
    _, _, model = _lasvm_moons()
    # Audit E4: sum a y was 2.0 after forget_one of the largest alpha.
    model.forget_one(int(np.argmax(model.SV_alpha_)))
    a, y, K, g_true = _lasvm_working_set_gradients(model)
    assert abs(a @ y) < 1e-8
    assert np.max(np.abs(np.asarray(model.SV_g_) - g_true)) < 1e-8
    assert kkt_gap(a, K, y, model.C, eps=1e-12 * model.C) <= model.tol * (1.0 + 1e-9)


def test_lasvm_finish_close_to_sklearn():
    pytest.importorskip("sklearn")
    from sklearn.svm import SVC as SkSVC

    X, y, model = _lasvm_moons()
    assert len(model.SV_ids_) == len(set(model.SV_ids_)), "duplicate sample ids in the working set"
    C = model.C
    K = RBFKernel(gamma=1.0)(X)
    a_full = np.zeros(len(y))
    a_full[np.asarray(model.SV_ids_, dtype=int)] = model.SV_alpha_
    report = kkt_report(a_full, K, y, C)
    sk = SkSVC(C=C, kernel="precomputed", tol=1e-8).fit(K, y)
    dual_ref = kkt_report(_sk_alpha(sk, len(y)), K, y, C)["dual_objective"]
    # Audit E4: dual -50.27 vs -52.53 and full-set kkt_gap 1.46.
    assert abs(report["dual_objective"] - dual_ref) <= 1e-3 * abs(dual_ref)
    assert report["kkt_gap"] <= 10.0 * model.tol
    assert report["sum_alpha_y"] < 1e-8
    assert model.get_model_state()["n_samples_seen"] == 2 * len(y)


def test_lasvm_forget_sample_id_noop_for_pruned():
    X, _, model = _lasvm_moons()
    pruned = [k for k in range(len(X)) if k not in set(model.SV_ids_)]
    assert pruned
    before = (list(model.SV_ids_), list(model.SV_alpha_), list(model.SV_g_), model.b_)
    model.forget_sample_id(pruned[0])
    assert (list(model.SV_ids_), list(model.SV_alpha_), list(model.SV_g_), model.b_) == before

    # An id that is in the working set is removed exactly.
    target = model.SV_ids_[int(np.argmax(model.SV_alpha_))]
    model.forget_sample_id(target)
    assert target not in model.SV_ids_
    a, y, _, g_true = _lasvm_working_set_gradients(model)
    assert abs(a @ y) < 1e-8
    assert np.max(np.abs(np.asarray(model.SV_g_) - g_true)) < 1e-8


# =====================================================================
# Area C: COSVM family (shared geometry G = K A^{-1} K)
# =====================================================================


def _anisotropic_target():
    """C1 setup: std 3 along e1, 0.3 along e2."""
    return np.random.RandomState(0).randn(80, 2) * np.array([3.0, 0.3])


def _linear_toy():
    """exp_cosvm2.py data: rotated anisotropic cloud centred at (10, 10)."""
    rng = np.random.RandomState(0)
    n = 80
    rot = np.array([[np.cos(0.52), -np.sin(0.52)], [np.sin(0.52), np.cos(0.52)]])
    X = (rng.randn(n, 2) * np.array([3.0, 0.3])) @ rot.T + np.array([10.0, 10.0])
    Xc = X - X.mean(axis=0)
    sigma = Xc.T @ Xc / n  # covariance with 1/N, as in the primal
    return X, sigma


def test_cosvm_eta1_matches_ocsvm():
    pytest.importorskip("sklearn")
    from sklearn.svm import OneClassSVM

    Xt = _anisotropic_target()
    nu, gamma = 0.2, 0.2
    model = COSVM(nu=nu, eta=1.0, kernel="rbf", gamma=gamma, tol=1e-9, max_iter=5000).fit(Xt)
    ref = OneClassSVM(nu=nu, kernel="rbf", gamma=gamma, tol=1e-9).fit(Xt)
    # sklearn's alphas sum to nu*N, ours to 1.
    np.testing.assert_allclose(
        model.decision_function(Xt) * nu * len(Xt), ref.decision_function(Xt), rtol=0.0, atol=1e-4
    )


def test_cosvm_family_rejects_eta_zero():
    for cls in (COSVM, ICOSVM, IDCOSVM):
        with pytest.raises(ValueError):
            cls(eta=0.0)
        with pytest.raises(ValueError):
            cls(eta=1.5)


def test_cosvm_linear_aligns_low_variance_axis():
    X, sigma = _linear_toy()
    # Default psd_tol: the audit's version raised a spurious "not PSD" error here.
    model = COSVM(nu=0.2, eta=0.4, kernel="linear", tol=1e-9, max_iter=5000).fit(X)
    w = X.T @ model.coef_  # f(x) = w^T x - rho for a linear kernel
    low_dir = np.linalg.eigh(sigma)[1][:, 0]
    cos = abs(w @ low_dir) / np.linalg.norm(w)
    assert cos > 0.95


def test_cosvm_matches_explicit_primal_dual_linear():
    X, sigma = _linear_toy()
    eta, nu = 0.4, 0.2
    model = COSVM(nu=nu, eta=eta, kernel="linear", tol=1e-9, max_iter=5000).fit(X)

    M = eta * np.eye(2) + (1.0 - eta) * sigma
    G_ref = X @ np.linalg.solve(M, X.T)
    np.testing.assert_allclose(model.K_tilde_, G_ref, rtol=1e-6, atol=1e-9 * np.abs(G_ref).max())

    # Primal consistency: w = M^{-1} X^T alpha and f(x) = w^T x - rho.
    w_primal = np.linalg.solve(M, X.T @ model.alpha_)
    Xq = np.random.RandomState(1).randn(20, 2) * 3.0 + 10.0
    # (f is a difference of two O(rho) numbers, so the tolerance is relative to rho)
    np.testing.assert_allclose(
        model.decision_function(Xq), Xq @ w_primal - model.rho_, rtol=0.0, atol=1e-8 * abs(model.rho_)
    )


@pytest.mark.parametrize("eta", [0.7, 0.4])
def test_cosvm_nu_property_and_kkt(eta):
    Xt = _anisotropic_target()
    n = len(Xt)
    nu = 0.1
    model = COSVM(nu=nu, eta=eta, kernel="rbf", gamma="scale").fit(Xt)

    G = model.K_tilde_
    report = oneclass_kkt_report(model.alpha_, G, model.C_, rho=model.rho_, eps=1e-12 * model.C_)
    assert report["sum_alpha_minus_1"] < 1e-8
    assert report["box_violation"] == 0.0
    assert report["kkt_gap"] < 1e-5 * np.max(np.diag(G))

    # nu-property: outliers (f < 0 beyond rounding; margin SVs have f = 0) <= nu, SVs >= nu.
    f = model.decision_function(Xt)
    outlier_fraction = np.mean(f < -1e-8 * np.max(np.diag(G)))
    assert outlier_fraction <= nu + 2.0 / n
    assert np.mean(model.alpha_ > 0.0) >= nu - 1e-12
    # Training scores from the out-of-sample formula equal G alpha.
    np.testing.assert_allclose(model.score_samples(Xt), G @ model.alpha_, rtol=1e-8, atol=1e-10)


def _assert_same_model(model, ref, probe):
    np.testing.assert_allclose(model.alpha_, ref.alpha_, rtol=0.0, atol=1e-5)
    f_ref = ref.decision_function(probe)
    np.testing.assert_allclose(
        model.decision_function(probe), f_ref, rtol=0.0, atol=1e-5 * np.abs(f_ref).max()
    )


def test_icosvm_and_idcosvm_equal_batch_cosvm():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 2) * np.array([2.0, 0.5])
    probe = rng.randn(100, 2) * 2.5
    # Numeric gamma, so all three models use the same kernel.
    kw = dict(nu=0.1, eta=0.4, kernel="rbf", gamma=0.5)

    ic = ICOSVM(**kw).fit(X[:25])
    idc = IDCOSVM(**kw).fit(X[:25])
    for x in X[25:]:
        ic.learn_one(x)
        idc.learn_one(x)
    batch = COSVM(**kw).fit(X)
    for model in (ic, idc):
        _assert_same_model(model, batch, probe)
    assert ic.n_incremental_updates_ + ic.n_qp_solves_ == 25

    ic.unlearn_one(3)
    idc.unlearn_one(3)
    batch = COSVM(**kw).fit(np.delete(X, 3, axis=0))
    for model in (ic, idc):
        _assert_same_model(model, batch, probe)


def test_icosvm_sensitivity_sign():
    rng = np.random.RandomState(0)
    B = rng.randn(12, 12)
    G = B @ B.T + 0.1 * np.eye(12)
    S, c = [0, 1, 2, 3, 4], 7
    alpha = np.zeros(12)
    alpha[S] = 0.2
    rho = 0.3
    d_rho, d_alpha_S, d_g_c = _sensitivity_step(G, S, c)

    delta = 1e-3
    a_new = alpha.copy()
    a_new[c] += delta
    a_new[S] += d_alpha_S * delta
    rho_new = rho + d_rho * delta

    # margin conditions and sum(alpha) are preserved
    np.testing.assert_allclose((G @ a_new)[S] - rho_new, (G @ alpha)[S] - rho, rtol=0.0, atol=1e-10)
    assert abs(a_new.sum() - alpha.sum()) < 1e-12
    # finite difference of the rho that keeps S on the margin
    fd_rho = (np.mean((G @ a_new)[S]) - np.mean((G @ alpha)[S])) / delta
    assert fd_rho == pytest.approx(d_rho, rel=1e-8)
    # candidate gradient g_c = (G a)_c - rho
    fd_g_c = (((G @ a_new)[c] - rho_new) - ((G @ alpha)[c] - rho)) / delta
    assert fd_g_c == pytest.approx(d_g_c, rel=1e-8)


def test_icosvm_learn_one_from_empty():
    X = np.random.RandomState(0).randn(12, 2)
    model = ICOSVM(nu=0.2, eta=0.5)
    # D2: gamma='scale' from one row is not identifiable -> warning, gamma = 1/d.
    with pytest.warns(UserWarning):
        model.learn_one(X[0])
    assert model.is_fitted
    np.testing.assert_array_equal(model.alpha_, [1.0])
    assert model.kernel_.gamma == pytest.approx(0.5)
    for x in X[1:]:
        model.learn_one(x)
    assert model.n_samples_seen_ == len(X)
    batch = COSVM(nu=0.2, eta=0.5, gamma=0.5).fit(X)
    _assert_same_model(model, batch, X)


def test_icosvm_unlearn_keeps_samples_seen():
    X = np.random.RandomState(1).randn(30, 2)
    model = ICOSVM(nu=0.2, eta=0.5, gamma=0.5).fit(X[:20])
    for x in X[20:25]:
        model.learn_one(x)
    model.unlearn_one(0)
    model.unlearn_one(4)
    assert model.n_samples_seen_ == 25
    assert model.n_unlearned_ == 2
    assert len(model.X_buffer_) == 23

    window = ICOSVM(nu=0.2, eta=0.5, gamma=0.5, max_samples=10).fit(X[:10])
    for x in X[10:]:
        window.learn_one(x)
    assert len(window.X_buffer_) == 10
    assert window.n_samples_seen_ == 30
    assert window.n_unlearned_ == 20


# =====================================================================
# Area D: drift detection (calibrated surprise D_t = -log10 p, DDM)
# =====================================================================


def _stationary_tracker(s, seed=0, n_fit=200):
    rng = np.random.RandomState(seed)
    tracker = OnlineCovarianceTracker().fit(rng.randn(n_fit, 4) * s)
    return tracker, rng


@pytest.mark.parametrize("s", [0.5, 1.0, 2.0])
def test_latent_detector_false_alarm_rate(s):
    # Audit C8 (old uncalibrated D_t): 18 / 337 / 481 alarms for s = 0.5 / 1 / 2.
    tracker, rng = _stationary_tracker(s)
    detector = LatentCovarianceDriftDetector()
    n_drift = 0
    p_values = []
    for z in rng.randn(500, 4) * s:
        D_t, lambda_t = tracker.update(z.reshape(1, -1))
        assert np.isclose(D_t, -np.log10(tracker.last_p_value_))
        p_values.append(tracker.last_p_value_)
        is_drift, _ = detector.update(D_t)
        n_drift += int(is_drift)
    assert n_drift <= 10
    # Under H0 the predictive p-value is ~ U(0, 1).
    assert 0.35 <= np.mean(p_values) <= 0.65
    # The surprise score is clipped to [0, 300].
    assert 0.0 <= np.max(tracker.history_D_) <= 300.0


def test_latent_detector_detects_mean_shift():
    for s in (0.5, 1.0, 2.0):
        tracker, rng = _stationary_tracker(s)
        detector = LatentCovarianceDriftDetector()
        for z in rng.randn(100, 4) * s:
            detector.update(tracker.update(z.reshape(1, -1))[0])
        detected_at = None
        for k, z in enumerate(rng.randn(5, 4) * s + 3.0 * s):
            if detector.update(tracker.update(z.reshape(1, -1))[0])[0]:
                detected_at = k
                break
        assert detected_at is not None, s

    # Power over seeds (s = 1): a single-sample predictive test detects the
    # shift on the first shifted sample with probability ~0.85; a missed first
    # sample inflates Sigma along the shift and masks the following ones.
    hits = 0
    for seed in range(30):
        tracker, rng = _stationary_tracker(1.0, seed=100 + seed)
        detector = LatentCovarianceDriftDetector()
        for z in rng.randn(100, 4):
            detector.update(tracker.update(z.reshape(1, -1))[0])
        for z in rng.randn(5, 4) + 3.0:
            if detector.update(tracker.update(z.reshape(1, -1))[0])[0]:
                hits += 1
                break
    assert hits >= 21


def test_latent_detector_reset_is_refractory():
    detector = LatentCovarianceDriftDetector(min_instances=3)
    for _ in range(3):
        assert detector.update(10.0) == (False, False)
    assert detector.update(10.0) == (True, False)
    detector.reset()
    for _ in range(3):
        assert detector.update(10.0) == (False, False)
    assert detector.update(10.0) == (True, False)
    assert detector.drift_count_ == 2


def test_drift_aware_cov_svm_stationary():
    # Audit: 87 drift events in 140 stationary steps with the old D_t.
    rng = np.random.RandomState(0)
    X = rng.randn(200, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = DriftAwareCovarianceSVM(
        encoder=IdentityEncoder(), C=1.0, eta=0.7, kernel="rbf", max_buffer_size=80, random_state=0
    )
    model.fit(X[:60], y[:60])
    for x, yv in zip(X[60:], y[60:]):
        model.learn_one(x, yv)
    assert len(model.drift_events_) <= 3
    for event in model.drift_events_:
        assert 0.0 <= event["p_value"] <= 1e-3


def test_ddm_no_spurious_drift_after_clean_warmup():
    rng = np.random.RandomState(0)
    ddm = DDMDriftDetector()
    # Audit C9: p_min = s_min = 0 after a clean warm-up made one error a drift.
    for _ in range(50):
        assert not ddm.update(False)
    assert not ddm.update(True)

    n_drift = 0
    for err in rng.rand(2000) < 0.02:
        if ddm.update(bool(err)):
            n_drift += 1
            ddm.reset()
    assert n_drift <= 1

    detected = False
    for err in rng.rand(100) < 0.5:
        if ddm.update(bool(err)):
            detected = True
            break
    assert detected


def test_ew_covariance_unbiased():
    s = 2.0
    rng = np.random.RandomState(0)
    tracker = OnlineCovarianceTracker(lambda_base=0.95, gamma_drift=0.0).fit(rng.randn(50, 4) * s)
    diag = []
    for z in rng.randn(4000, 4) * s:
        _, lambda_t = tracker.update(z.reshape(1, -1))
        assert lambda_t == 0.95
        diag.append(np.diag(tracker.cov_))
    # Kish effective sample size of EW weights: (1 + lambda) / (1 - lambda).
    assert np.isclose(tracker.n_eff_, 1.95 / 0.05, rtol=1e-6)
    ratio = np.mean(diag[200:], axis=0) / s ** 2
    np.testing.assert_allclose(ratio, 1.0, atol=0.05)


# =====================================================================
# Integration: streaming models after fit + learn_one (HIGH-1, LOW validation)
# =====================================================================


def _streaming_models():
    from dynamicsvm.stream.windowed import SlidingWindowSVM

    return {
        "lasvm": LASVM(C=1.0, kernel="rbf", gamma="scale", random_state=0),
        "adiabatic": AdiabaticSVM(C=1.0, kernel="rbf", gamma="scale"),
        "sliding": SlidingWindowSVM(window_size=40, C=1.0, kernel="rbf", gamma="scale"),
        "drift_cov": DriftAwareCovarianceSVM(encoder=IdentityEncoder(), C=1.0, random_state=0),
        "cosvm": COSVM(nu=0.2, eta=0.5),
        "icosvm": ICOSVM(nu=0.2, eta=0.5),
        "idcosvm": IDCOSVM(nu=0.2, eta=0.5),
    }


@pytest.mark.parametrize("name", list(_streaming_models().keys()))
def test_streaming_models_batch_invariant_after_updates(name):
    # gamma='scale' is frozen at fit; streamed updates and 1-D queries must not
    # re-resolve it, and a 1-D row of length d is one sample (not d samples).
    X, y = _moons_scaled()
    X, y = X[:60], y[:60]
    model = _streaming_models()[name]
    one_class = name in ("cosvm", "icosvm", "idcosvm")
    if one_class:
        model.fit(X[:30])
        if name != "cosvm":
            for x in X[30:36]:
                model.learn_one(x)
    else:
        model.fit(X[:30], y[:30])
        for x, label in zip(X[30:36], y[30:36]):
            model.learn_one(x, label)

    f_batch = np.asarray(model.decision_function(X), dtype=float)
    f_rows = np.array([np.asarray(model.decision_function(x), dtype=float).item() for x in X])
    np.testing.assert_allclose(f_batch, f_rows, atol=1e-10, rtol=0)


def test_icosvm_learn_batch_accepts_1d_row():
    X = np.random.RandomState(0).randn(20, 2)
    model = ICOSVM(nu=0.2, eta=0.5, gamma=0.5).fit(X[:10])
    model.learn_batch(X[10])
    assert model.n_samples_seen_ == 11

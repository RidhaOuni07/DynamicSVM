"""
Incremental and Decremental Covariance-guided One-Class Support Vector Machine
(IDCOSVM).

This implementation combines:

1. The covariance-guided kernel idea of:
       Kefi-Fatteh et al. (2019),
       "A novel incremental one-class support vector machine based on
        low variance direction."

2. Incremental learning through warm-started constrained optimization.

3. Decremental learning through removal followed by constrained
   re-optimization.

4. Bounded memory with four retention HEURISTICS - FIFO, non-SV-first,
   error-first and Mahalanobis. These keep the buffer under
   ``max_samples``; none of them is adaptive in any proven sense (no
   objective bound, no influence function, no drift signal), and
   ``mahalanobis`` is anti-adaptive by construction. See
   ``select_obsolete_indices`` for the per-policy caveats before
   choosing anything other than the default ``'fifo'``.

Important
---------
Every update (learn_one, unlearn_one, pruning) is a warm-started exact
re-solve of the current finite COSVM dual: the previous alpha, made
feasible for the new box, is the starting point and the solution is
refined to the KKT conditions of the new problem. It does NOT claim to be
the exact piecewise-linear adiabatic algorithm of Cauwenberghs & Poggio,
and the cost per update is that of a batch fit (O(N^3)).

Mathematical formulation
------------------------
IDCOSVM shares the formulation of :class:`dynamicsvm.models.cosvm.COSVM`
(see ``dynamicsvm.models._cosvm_core`` for the derivation). For the N
active samples:

    min_alpha  1/2 alpha^T G alpha

subject to:

    sum(alpha_i) = 1
    0 <= alpha_i <= C

where:

    C = 1 / (nu * N),

    G = K B^{-1},   B = eta I + ((1 - eta)/N) H K,

    H = I - (1/N) 11^T,   0 < eta <= 1.

This is the unconditionally valid form: spec(H K) lies in [0, inf), so B
is invertible for every eta > 0. The symmetric form

    G = K A^{-1} K,   A = K B = eta K + ((1 - eta)/N) K H K,

is the same matrix ONLY when K is invertible, which routinely fails here
(a linear kernel with N > d, duplicated rows in the buffer, any
rank-deficient Gram matrix). What is computed is the ridged
K (A + eps I)^{-1} K with eps = reg * trace(A) / N, because the symmetric
PSD A is the factorisable object (Cholesky), not B.

G is the dual Gram matrix of the primal
min 1/2 w^T (eta I + (1 - eta) Sigma) w - rho + C sum xi with the
feature-space covariance Sigma of the active buffer.

The decision function is:

    f(x) = k(x, X)^T coef - rho,   coef = (A + eps I)^{-1} K alpha,

where rho is estimated from the KKT conditions. On the same buffer and
kernel, IDCOSVM, ICOSVM and batch COSVM solve the same problem.
"""

from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from dynamicsvm.base import BaseSVM
from dynamicsvm.kernels import Kernel, get_kernel
from dynamicsvm.models._cosvm_core import (
    COSVMGeometry,
    compute_rho,
    feasible_start,
    solve_oneclass_dual,
    validate_eta,
)
from dynamicsvm.utils.kkt import oneclass_kkt_report
from dynamicsvm.utils.validation import check_array, check_is_fitted


PruningStrategy = Literal[
    "fifo",
    "non_sv_first",
    "error_first",
    "mahalanobis",
]


# Single schema of _kkt_residuals, so its two branches cannot drift
# apart again (they used to return different key sets).
_KKT_RESIDUAL_KEYS: Tuple[str, ...] = (
    "max_stationarity_error",
    "max_lower_kkt_violation",
    "max_upper_kkt_violation",
    "sum_alpha_error",
)


def _kkt_residual_dict(
    max_stationarity_error: float,
    max_lower_kkt_violation: float,
    max_upper_kkt_violation: float,
    sum_alpha_error: float,
) -> Dict[str, float]:
    """
    Pack the four KKT residuals under ``_KKT_RESIDUAL_KEYS``, in order.
    """
    return dict(
        zip(
            _KKT_RESIDUAL_KEYS,
            (
                float(max_stationarity_error),
                float(max_lower_kkt_violation),
                float(max_upper_kkt_violation),
                float(sum_alpha_error),
            ),
        )
    )


class IncrementalDecrementalCOSVM(BaseSVM):
    """
    Incremental and Decremental Covariance-guided One-Class SVM.

    Parameters
    ----------
    nu : float, default=0.1
        One-class SVM parameter in (0, 1].

    eta : float, default=0.5
        Weight of the identity in the primal metric
        M = eta I + (1 - eta) Sigma, in (0, 1]. The dual Gram matrix is

            G = K B^{-1},   B = eta I + ((1 - eta)/N) H K

        (G = K, the standard one-class SVM, when eta = 1). The symmetric
        K A^{-1} K with A = K B is the same matrix only for invertible K;
        see ``dynamicsvm.models._cosvm_core`` for the derivation.

    kernel : {'rbf', 'linear', 'poly', 'sigmoid'} or Kernel or Callable
        Kernel specification.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient. A string gamma is resolved once, from the
        first buffer holding at least two samples, and is frozen
        afterwards (so predictions do not depend on how query samples are
        batched, see :mod:`dynamicsvm.kernels`). Starting a stream from a
        single sample cannot identify ``'scale'``, so the 1 / n_features
        fallback is used provisionally and re-resolved at the second
        sample.

    degree : int, default=3
        Polynomial kernel degree.

    coef0 : float, default=0.0
        Polynomial/sigmoid independent term.

    max_samples : int or None, default=None
        Maximum number of active samples.

    pruning_strategy : {'fifo', 'non_sv_first', 'error_first',
                         'mahalanobis'}, default='fifo'
        Retention heuristic used when the buffer exceeds
        ``max_samples``. None of the four is adaptive in a proven sense
        and ``mahalanobis`` must not be used under drift: read the
        caveat block in ``select_obsolete_indices`` before changing this
        away from ``'fifo'``.

    tol : float, default=1e-5
        Optimization tolerance. SLSQP runs at ftol = min(tol, 1e-10), so
        a value above 1e-10 does not change the solve; it can only
        tighten it. It has no effect on which samples are classified as
        support vectors - that is ``sv_tol``'s job alone.

    sv_tol : float, default=1e-8
        Active-set tolerance, applied as ``sv_tol * C``. It is the only
        tolerance that governs the S_set_ / E_set_ / O_set_ / support_
        partition, the free set rho is averaged over, and the KKT
        residual masks, so those four always agree - the class has
        exactly one active-set tolerance. Being relative to
        C = 1/(nu N), it does not degrade as N grows, unlike the
        absolute 1e-5 (the old ``tol``) it replaced. The default matches
        ``compute_rho``'s long-standing ``alpha_tol`` default, so the
        shipped behaviour of rho is unchanged. Note it is *relative* to
        C, where ``COSVM.sv_tol`` is absolute.

    max_iter : int, default=1000
        SLSQP iteration limit of every dual solve, > 0.

    reg : float, default=1e-10
        Relative ridge of A: the geometry factorises A + eps I with
        eps = reg * trace(A) / N, > 0. The induced error in
        G = K (A + eps I)^{-1} K is FIRST order in reg, not second, and
        its size grows like 1 / eta because the smallest eigenvalue of A
        scales with eta. Raising reg therefore biases G measurably at
        small eta; the default is the smallest value that still
        factorises on well-conditioned problems, and the ridge is grown
        by x10 (up to 8 times, recorded in ``n_ridge_growths_``) when it
        does not.

    psd_tol : float, default=1e-10
        Relative tolerance of the positive-semidefiniteness certificate
        of the dual Gram matrix, >= 0. At eta == 1, G is the kernel
        matrix itself, so an indefinite kernel is rejected here.

    Attributes
    ----------
    alpha_ : ndarray
        Dual coefficients for all active samples.

    offset_ : float
        One-class decision threshold rho.

    intercept_ : float
        scikit-learn-style intercept, equal to -offset_.

    support_ : ndarray
        Indices of support vectors in the active buffer.

    support_vectors_ : ndarray
        Active support vectors.

    dual_coef_ : ndarray
        Dual coefficients corresponding to support vectors.

    S_set_ : list[int]
        Margin support vectors satisfying approximately:

            0 < alpha_i < C

    E_set_ : list[int]
        Error/bound support vectors satisfying approximately:

            alpha_i = C

    O_set_ : list[int]
        Non-support vectors satisfying approximately:

            alpha_i = 0

    optimization_result_ : scipy.optimize.OptimizeResult or None
        Raw result of the LAST dual solve, carrying the solver message,
        the success flag and the certificate keys ``kkt_gap``,
        ``kkt_gap_relative`` and ``objective_scale``. It is None when the
        buffer holds a single sample (the dual is then trivially
        alpha = [1] and no solver runs) and on an empty buffer.

    n_dual_solves_ : int
        Lifetime number of dual solves, one per fit / learn_one /
        unlearn_one / pruning step. Like n_samples_seen_, it is NOT reset
        when the buffer drains.

    kkt_gap_ : float
        KKT gap of the last accepted alpha in the units of G, i.e.
        max_{alpha_i > 0} (G alpha)_i - min_{alpha_j < C} (G alpha)_j
        clipped at zero. 0.0 when no solver ran.

    kkt_gap_relative_ : float
        kkt_gap_ divided by max_k G_kk, the scale the solver's own
        acceptance gate uses (it accepts below 1e-6).

    eps_ : float
        Ridge actually added to A by the last geometry rebuild, i.e.
        eps = reg * trace(A) / N possibly grown by x10 up to 8 times.
        0.0 when eta == 1 (no ridge is needed, G = K).

    n_ridge_growths_ : int
        Number of x10 growths eps_ needed. 0 means the requested
        relative ridge sufficed.

    timestamps_ : list[int]
        Arrival counter of every active sample. It is maintained on
        every insertion and removal but is read by NO pruning policy -
        ``'fifo'`` uses buffer position, not time - so it is currently
        bookkeeping only. A genuinely time-aware retention policy is
        future work; the attribute is kept so that such a policy (and
        external callers) have the data.
    """

    def __init__(
        self,
        nu: float = 0.1,
        eta: float = 0.5,
        kernel: Union[str, Kernel, Callable] = "rbf",
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        max_samples: Optional[int] = None,
        pruning_strategy: PruningStrategy = "fifo",
        tol: float = 1e-5,
        sv_tol: float = 1e-8,
        max_iter: int = 1000,
        reg: float = 1e-10,
        psd_tol: float = 1e-10,
    ):
        super().__init__()

        if not (0.0 < nu <= 1.0):
            raise ValueError(
                f"nu must be in (0, 1], got {nu}"
            )

        validate_eta(eta)

        if max_samples is not None and max_samples <= 0:
            raise ValueError(
                f"max_samples must be positive, got {max_samples}"
            )

        valid_strategies = {
            "fifo",
            "non_sv_first",
            "error_first",
            "mahalanobis",
        }

        if pruning_strategy not in valid_strategies:
            raise ValueError(
                f"Unknown pruning_strategy={pruning_strategy!r}. "
                f"Expected one of {sorted(valid_strategies)}."
            )

        if tol <= 0:
            raise ValueError(
                f"tol must be positive, got {tol}"
            )

        if sv_tol <= 0:
            raise ValueError(
                f"sv_tol must be positive, got {sv_tol}"
            )

        if max_iter <= 0:
            raise ValueError(
                f"max_iter must be > 0, got {max_iter}"
            )

        if reg <= 0:
            raise ValueError(
                f"reg must be > 0, got {reg}"
            )

        if psd_tol < 0:
            raise ValueError(
                f"psd_tol must be >= 0, got {psd_tol}"
            )

        self.nu = float(nu)
        self.eta = float(eta)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.max_samples = (
            int(max_samples)
            if max_samples is not None
            else None
        )
        self.pruning_strategy = pruning_strategy
        self.tol = float(tol)
        self.sv_tol = float(sv_tol)
        self.max_iter = int(max_iter)
        self.reg = float(reg)
        self.psd_tol = float(psd_tol)

        # -------------------------------------------------------------
        # Dynamic state
        # -------------------------------------------------------------

        self.kernel_: Optional[Kernel] = None

        # True while a string gamma is sitting on the 1 / n_features
        # fallback because it could not be estimated from fewer than two
        # samples; _ensure_kernel re-resolves it once.
        self._gamma_provisional_: bool = False

        self.X_buffer_: List[np.ndarray] = []
        self.timestamps_: List[int] = []

        self.alpha_: np.ndarray = np.empty(0, dtype=float)

        self.offset_: float = 0.0
        self.intercept_: float = 0.0

        self.S_set_: List[int] = []
        self.E_set_: List[int] = []
        self.O_set_: List[int] = []

        self.K_full_: Optional[np.ndarray] = None
        self.K_tilde_full_: Optional[np.ndarray] = None

        # Shared COSVM geometry of the active buffer and score
        # coefficients coef_ = A^{-1} K alpha.
        self.geometry_: Optional[COSVMGeometry] = None
        self.coef_: Optional[np.ndarray] = None

        self.n_samples_seen_: int = 0
        self.n_unlearned_: int = 0

        # Numerical certificate of the last solve (see Attributes).
        self.optimization_result_: Optional[object] = None
        self.n_dual_solves_: int = 0
        self.kkt_gap_: float = 0.0
        self.kkt_gap_relative_: float = 0.0
        self.eps_: float = 0.0
        self.n_ridge_growths_: int = 0

    # =================================================================
    # Utility methods
    # =================================================================

    def _validate_X(self, X: np.ndarray, n_features: Optional[int] = None) -> np.ndarray:
        """
        Validate a two-dimensional feature matrix (see check_array for the
        1-D semantics when `n_features` is given).
        """
        X_arr = check_array(X, n_features=n_features)

        if X_arr.ndim != 2:
            raise ValueError(
                f"Expected a 2D array, got shape {X_arr.shape}"
            )

        return X_arr

    def _current_C(self) -> float:
        """
        Return the current box constraint:

            C = 1 / (nu * N)
        """
        n_samples = len(self.X_buffer_)

        if n_samples == 0:
            return np.inf

        return 1.0 / (self.nu * n_samples)

    def _reset_empty_state(self) -> None:
        """
        Reset the dynamic model state when no samples remain.
        """
        self.alpha_ = np.empty(0, dtype=float)

        self.offset_ = 0.0
        self.intercept_ = 0.0

        self.S_set_ = []
        self.E_set_ = []
        self.O_set_ = []

        self.K_full_ = None
        self.K_tilde_full_ = None

        self.geometry_ = None
        self.coef_ = None

        self.support_ = None
        self.support_vectors_ = None
        self.dual_coef_ = None
        self.n_support_ = np.array([0])

        # Per-solve certificate: no geometry and no solve exist any more.
        # n_dual_solves_ is a LIFETIME counter and deliberately survives,
        # like n_samples_seen_.
        self.optimization_result_ = None
        self.kkt_gap_ = 0.0
        self.kkt_gap_relative_ = 0.0
        self.eps_ = 0.0
        self.n_ridge_growths_ = 0

    # =================================================================
    # Covariance-guided kernel
    # =================================================================

    @staticmethod
    def _centering_matrix(n_samples: int) -> np.ndarray:
        """
        Construct the standard centering matrix:

            H = I - 1/N 11^T
        """
        if n_samples <= 0:
            return np.empty((0, 0))

        return (
            np.eye(n_samples)
            - np.ones((n_samples, n_samples)) / n_samples
        )

    def _compute_covariance_kernel_matrix(
        self,
        K: np.ndarray,
    ) -> np.ndarray:
        """
        Construct the COSVM dual Gram matrix

            G = K B^{-1},   B = eta I + ((1 - eta)/N) H K

        computed as K (A + eps I)^{-1} K with A = K B, which equals
        K A^{-1} K only for invertible K (shared with COSVM / ICOSVM,
        see ``_cosvm_core``).
        """
        return COSVMGeometry(
            K,
            self.eta,
            reg=self.reg,
            psd_tol=self.psd_tol,
        ).G

    # =================================================================
    # Kernel matrix management
    # =================================================================

    def _ensure_kernel(self) -> None:
        """
        Resolve a provisional string gamma, once.

        ``'scale'`` is not identifiable from fewer than two samples, so a
        stream started from a single observation has to fall back to
        1 / n_features. That fallback is provisional: as soon as the
        buffer holds two samples, gamma is resolved from the buffer and
        the kernel is rebuilt and frozen for the life of the model. This
        keeps the frozen-gamma contract of :mod:`dynamicsvm.kernels`
        (predictions independent of query batching) while limiting the
        fallback to a single sample instead of forever.
        """
        if (
            not self._gamma_provisional_
            or len(self.X_buffer_) < 2
        ):
            return

        self.kernel_ = get_kernel(
            self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            n_features=self.n_features_in_,
            X=np.asarray(self.X_buffer_),
        )

        self._gamma_provisional_ = False

    def _recompute_full_matrices(self) -> None:
        """
        Recompute the active-buffer kernel matrix and COSVM geometry.

        Adding or removing a sample changes N, H and A^{-1}, hence every
        entry of G, so the geometry is rebuilt (O(N^3)).
        """
        self._ensure_kernel()

        if not self.X_buffer_:
            self.K_full_ = None
            self.K_tilde_full_ = None
            self.geometry_ = None
            self.coef_ = None
            return

        X_mat = np.asarray(self.X_buffer_)

        K = np.asarray(
            self.kernel_(X_mat, X_mat),
            dtype=float,
        )

        K = 0.5 * (K + K.T)

        self.K_full_ = K

        self.geometry_ = COSVMGeometry(
            K,
            self.eta,
            reg=self.reg,
            psd_tol=self.psd_tol,
        )

        self.K_tilde_full_ = self.geometry_.G

    # =================================================================
    # KKT handling
    # =================================================================

    def _classify_kkt_sets(self) -> None:
        """
        Classify observations according to the current alpha values.

        S:
            0 < alpha_i < C

        E:
            alpha_i ~= C

        O:
            alpha_i ~= 0

        The boundaries are sv_tol * C, the same tolerance compute_rho
        averages over, so the partition and rho always see the same free
        set. The upper boundary carries one extra floor of 10 machine
        epsilons, which cannot bind for any buffer this model can hold:
        it would take C = 1/(nu N) below 2e-7, i.e. N above 4e6 even at
        nu = 1, against an O(N^3) cost per update. The early return
        keeps C = inf (empty buffer) out of the product.
        """
        n_samples = len(self.X_buffer_)

        self.S_set_ = []
        self.E_set_ = []
        self.O_set_ = []

        if n_samples == 0:
            return

        C = self._current_C()

        alpha_tol = self.sv_tol * C

        lower_tol = alpha_tol
        upper_tol = max(alpha_tol, 10.0 * np.finfo(float).eps)

        for i, alpha_i in enumerate(self.alpha_):
            if alpha_i <= lower_tol:
                self.O_set_.append(i)

            elif alpha_i >= C - upper_tol:
                self.E_set_.append(i)

            else:
                self.S_set_.append(i)

    def _compute_offset_from_kkt(self) -> float:
        """
        Estimate rho from the KKT conditions.

        For margin vectors:

            (G alpha)_i = rho

        If no strict margin vector exists, derive an admissible rho
        interval from the lower/upper KKT bounds.
        """
        n_samples = len(self.X_buffer_)

        if n_samples == 0:
            return 0.0

        # Shared with COSVM / ICOSVM: mean of (G alpha)_i over free
        # coefficients, else the midpoint of
        # [max_{alpha=C} (G alpha)_i, min_{alpha=0} (G alpha)_i]. The
        # free set is the S_set_ of _classify_kkt_sets by construction:
        # both use sv_tol * C.
        C = self._current_C()

        return compute_rho(
            self.K_tilde_full_,
            self.alpha_,
            C,
            alpha_tol=self.sv_tol * C,
        )

    def _update_kkt_state(self) -> None:
        """
        Refresh KKT partitions, offset, and public support-vector
        attributes.
        """
        self._classify_kkt_sets()

        if self.K_tilde_full_ is not None:
            self.offset_ = self._compute_offset_from_kkt()
        else:
            self.offset_ = 0.0

        self.intercept_ = -self.offset_

        self._sync_support_vectors()

    # =================================================================
    # Public support-vector state
    # =================================================================

    def _sync_support_vectors(self) -> None:
        """
        Synchronize support-vector attributes.
        """
        if not self.X_buffer_:
            self.support_vectors_ = None
            self.dual_coef_ = None
            self.support_ = None
            self.n_support_ = np.array([0])
            return

        sv_indices = sorted(
            set(self.S_set_) | set(self.E_set_)
        )

        self.support_ = np.asarray(
            sv_indices,
            dtype=int,
        )

        X_mat = np.asarray(
            self.X_buffer_
        )

        if sv_indices:
            self.support_vectors_ = X_mat[
                sv_indices
            ]

            self.dual_coef_ = self.alpha_[
                sv_indices
            ]

        else:
            self.support_vectors_ = np.empty(
                (0, X_mat.shape[1]),
                dtype=X_mat.dtype,
            )

            self.dual_coef_ = np.empty(
                0,
                dtype=float,
            )

        self.n_support_ = np.array(
            [len(sv_indices)],
            dtype=int,
        )

    # =================================================================
    # Optimization
    # =================================================================

    def _initial_feasible_alpha(
        self,
        previous_alpha: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Construct a feasible starting point.

        We need:

            sum(alpha) = 1
            0 <= alpha_i <= C

        previous_alpha is clipped to the new box and the missing mass is
        water-filled onto coefficients below C (``feasible_start``); the
        uniform point 1/N (feasible since nu <= 1) is the fallback.
        """
        n_samples = len(self.X_buffer_)

        if n_samples == 0:
            return np.empty(0)

        return feasible_start(
            previous_alpha,
            n_samples,
            self._current_C(),
        )

    def _solve_dual(
        self,
        previous_alpha: Optional[np.ndarray] = None,
    ) -> None:
        """
        Solve the current finite COSVM dual problem.

        Objective:

            1/2 alpha^T G alpha,   G = K B^{-1}

        (K A^{-1} K only for invertible K, see ``_cosvm_core``)

        subject to:

            sum(alpha) = 1
            0 <= alpha_i <= C

        The previous solution is used as a warm start whenever possible.
        """
        n_samples = len(self.X_buffer_)

        if n_samples == 0:
            self._reset_empty_state()
            return

        if self.K_tilde_full_ is None:
            raise RuntimeError(
                "K_tilde_full_ is not available."
            )

        C = self._current_C()

        x0 = self._initial_feasible_alpha(
            previous_alpha
        )

        # Shared solver: SLSQP from the feasible warm start, checked for
        # feasibility and refined to the KKT conditions. No blanket
        # renormalisation afterwards (it could push alpha above C).
        alpha, result = solve_oneclass_dual(
            self.K_tilde_full_,
            C,
            x0=x0,
            ftol=min(self.tol, 1e-10),
            max_iter=self.max_iter,
        )

        # Record the numerical certificate of this solve. result is None
        # when N == 1 (alpha = [1] exactly, no solver runs), in which
        # case the gap is zero by construction.
        self.optimization_result_ = result
        self.n_dual_solves_ += 1

        if result is None:
            self.kkt_gap_ = 0.0
            self.kkt_gap_relative_ = 0.0
        else:
            self.kkt_gap_ = float(result["kkt_gap"])
            self.kkt_gap_relative_ = float(
                result["kkt_gap_relative"]
            )

        self.eps_ = float(self.geometry_.eps_)
        self.n_ridge_growths_ = int(
            self.geometry_.n_ridge_growths_
        )

        self.alpha_ = alpha
        self.coef_ = self.geometry_.coef(alpha)

        self._update_kkt_state()

    # =================================================================
    # Fit
    # =================================================================

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> "IncrementalDecrementalCOSVM":
        """
        Fit COSVM from an initial batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : ignored
            Present for API compatibility.
        """
        X_arr = self._validate_X(X)

        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        if n_samples == 0:
            raise ValueError(
                "Cannot fit IDCOSVM on an empty dataset."
            )

        self.kernel_ = get_kernel(
            self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            n_features=n_features,
            X=X_arr,
        )

        # A string gamma resolved from a single row is the 1 / n_features
        # fallback; it is re-resolved once two samples are available.
        self._gamma_provisional_ = (
            isinstance(self.gamma, str)
            and n_samples < 2
        )

        self.X_buffer_ = [
            X_arr[i].copy()
            for i in range(n_samples)
        ]

        self.timestamps_ = list(
            range(n_samples)
        )

        self.n_samples_seen_ = n_samples
        self.n_unlearned_ = 0

        self.alpha_ = self._initial_feasible_alpha()

        self._recompute_full_matrices()

        self._solve_dual(
            previous_alpha=self.alpha_
        )

        self._enforce_max_samples()

        return self

    # =================================================================
    # Incremental learning
    # =================================================================

    def learn_one(
        self,
        x: np.ndarray,
    ) -> "IncrementalDecrementalCOSVM":
        """
        Incrementally add one observation.

        The previous solution is retained as a warm start. Because:

            C = 1/(nu*N)

        changes whenever N changes, the complete constrained problem is
        re-solved after adding the sample.

        This guarantees that the resulting alpha satisfies the current
        optimization constraints.
        """
        x_vec = np.asarray(
            x,
            dtype=float,
        ).ravel()

        if x_vec.size == 0:
            raise ValueError(
                "Cannot learn an empty sample."
            )

        # -------------------------------------------------------------
        # First observation of an empty buffer: a cold start, not a
        # re-fit (see _start_buffer_from_single).
        # -------------------------------------------------------------
        if not self.X_buffer_:
            return self._start_buffer_from_single(
                x_vec
            )

        expected_features = len(
            self.X_buffer_[0]
        )

        if x_vec.size != expected_features:
            raise ValueError(
                "Feature dimension mismatch: "
                f"expected {expected_features}, "
                f"got {x_vec.size}."
            )

        # Keep previous solution for warm start.
        previous_alpha = self.alpha_.copy()

        self.X_buffer_.append(
            x_vec.copy()
        )

        self.n_samples_seen_ += 1

        self.timestamps_.append(
            self.n_samples_seen_ - 1
        )

        # New point starts with zero coefficient.
        previous_alpha = np.append(
            previous_alpha,
            0.0,
        )

        self._recompute_full_matrices()

        self._solve_dual(
            previous_alpha=previous_alpha
        )

        self._enforce_max_samples()

        return self

    def _start_buffer_from_single(
        self,
        x_vec: np.ndarray,
    ) -> "IncrementalDecrementalCOSVM":
        """
        Cold-start the active buffer from a single observation.

        This is learn_one's empty-buffer path. It deliberately does less
        than fit: the lifetime counters n_samples_seen_ / n_unlearned_
        are continued rather than reset (a model drained by unlearning
        keeps its history), and an already resolved kernel_ is reused
        rather than re-resolved, so a frozen gamma survives a
        drain-and-refill. An explicit fit, by contrast, still resets the
        counters and re-resolves the kernel - that is what a re-fit
        means.
        """
        x_row = self._validate_X(
            x_vec.reshape(1, -1)
        )

        n_features = x_row.shape[1]

        expected_features = getattr(
            self,
            "n_features_in_",
            None,
        )

        if (
            expected_features is not None
            and n_features != expected_features
        ):
            raise ValueError(
                "Feature dimension mismatch: "
                f"expected {expected_features}, "
                f"got {n_features}."
            )

        self.n_features_in_ = n_features

        if self.kernel_ is None:
            self.kernel_ = get_kernel(
                self.kernel,
                gamma=self.gamma,
                degree=self.degree,
                coef0=self.coef0,
                n_features=n_features,
                X=x_row,
            )

            # A string gamma cannot be estimated from one sample, so it
            # sits on the 1 / n_features fallback until the buffer holds
            # two samples (_ensure_kernel).
            self._gamma_provisional_ = isinstance(
                self.gamma,
                str,
            )

        self.X_buffer_.append(
            x_row[0].copy()
        )

        self.n_samples_seen_ += 1

        self.timestamps_.append(
            self.n_samples_seen_ - 1
        )

        self._recompute_full_matrices()

        self._solve_dual()

        self._enforce_max_samples()

        return self

    def learn_batch(
        self,
        X: np.ndarray,
    ) -> "IncrementalDecrementalCOSVM":
        """
        Incrementally learn a sequence of observations.
        """
        X_arr = self._validate_X(X, n_features=getattr(self, "n_features_in_", None))

        for i in range(X_arr.shape[0]):
            self.learn_one(X_arr[i])

        return self

    def learn_many(
        self,
        X: np.ndarray,
    ) -> "IncrementalDecrementalCOSVM":
        """
        Alias for learn_batch.
        """
        return self.learn_batch(X)

    # =================================================================
    # Decremental learning
    # =================================================================

    def unlearn_one(
        self,
        index: int,
    ) -> "IncrementalDecrementalCOSVM":
        """
        Remove one sample from the active buffer.

        The deleted coefficient is removed and the remaining problem is
        re-solved with the new:

            C = 1/(nu*(N-1))

        constraint.

        Parameters
        ----------
        index : int
            Active buffer index.
        """
        n_samples = len(
            self.X_buffer_
        )

        if index < 0 or index >= n_samples:
            raise IndexError(
                f"Index {index} out of bounds for "
                f"buffer of size {n_samples}"
            )

        # -------------------------------------------------------------
        # Save the remaining alpha values as a warm start.
        # -------------------------------------------------------------
        previous_alpha = np.delete(
            self.alpha_,
            index,
        )

        # -------------------------------------------------------------
        # Remove data and timestamp.
        # -------------------------------------------------------------
        del self.X_buffer_[index]
        del self.timestamps_[index]

        self.n_unlearned_ += 1

        # -------------------------------------------------------------
        # Empty model.
        # -------------------------------------------------------------
        if not self.X_buffer_:
            self._reset_empty_state()
            return self

        # -------------------------------------------------------------
        # Rebuild matrices.
        # -------------------------------------------------------------
        self._recompute_full_matrices()

        # -------------------------------------------------------------
        # Re-solve the new constrained problem.
        # -------------------------------------------------------------
        self._solve_dual(
            previous_alpha=previous_alpha
        )

        return self

    def forget_one(
        self,
        index: int,
    ) -> "IncrementalDecrementalCOSVM":
        """
        Alias for unlearn_one.
        """
        return self.unlearn_one(index)

    def unlearn_sample(
        self,
        x: np.ndarray,
        tol: float = 1e-4,
    ) -> "IncrementalDecrementalCOSVM":
        """
        Find and remove the first active sample within Euclidean
        distance <= tol from x.
        """
        x_vec = np.asarray(
            x,
            dtype=float,
        ).ravel()

        if tol < 0:
            raise ValueError(
                f"tol must be non-negative, got {tol}"
            )

        for idx, buf_x in enumerate(
            self.X_buffer_
        ):
            if (
                np.linalg.norm(
                    buf_x - x_vec
                )
                <= tol
            ):
                return self.unlearn_one(idx)

        raise ValueError(
            "Sample not found in active memory buffer."
        )

    def unlearn_many(
        self,
        indices: Union[
            List[int],
            np.ndarray,
            Tuple[int, ...],
        ],
    ) -> "IncrementalDecrementalCOSVM":
        """
        Remove multiple samples.

        Indices refer to the buffer before deletion, so they are sorted
        in descending order before removal.
        """
        indices = list(
            map(int, indices)
        )

        if len(set(indices)) != len(indices):
            raise ValueError(
                "Duplicate indices are not allowed."
            )

        for idx in indices:
            if (
                idx < 0
                or idx >= len(self.X_buffer_)
            ):
                raise IndexError(
                    f"Index {idx} out of bounds."
                )

        for idx in sorted(
            indices,
            reverse=True,
        ):
            self.unlearn_one(idx)

        return self

    def forget_many(
        self,
        indices: Union[
            List[int],
            np.ndarray,
            Tuple[int, ...],
        ],
    ) -> "IncrementalDecrementalCOSVM":
        """
        Alias for unlearn_many.
        """
        return self.unlearn_many(indices)

    # =================================================================
    # Obsolescence / pruning
    # =================================================================

    def select_obsolete_indices(
        self,
        n_samples: int = 1,
        strategy: Optional[
            PruningStrategy
        ] = None,
    ) -> List[int]:
        """
        Select samples for automatic removal.

        Parameters
        ----------
        n_samples : int, default=1
            Number of indices to return, clipped to the buffer size.

        strategy : str or None, default=None
            Overrides ``pruning_strategy`` for this call.

        Returns
        -------
        list[int]
            Buffer positions proposed for removal.

        Notes
        -----
        What these four policies are, and are not. They are retention
        HEURISTICS. None of them carries an objective-increase bound, an
        influence function, a leverage score or a coreset guarantee: no
        policy here can say how much the dual objective, rho or the
        decision function will move when the selected points go. None of
        them consults ``timestamps_`` or any drift signal either. Only
        the ranking is used, and the ranking is a design choice.

        Measured behaviour, policy by policy:

        - ``'mahalanobis'`` removes the points FURTHEST from the buffer
          mean. For a one-class model those points are its support
          vectors: 8 of 8 selected points were support vectors, against
          a FIFO base rate of 3 of 8. It is also anti-adaptive by
          construction - under a (6, 6) mean shift with
          ``max_samples=30`` it retained 0% of the new regime, rejected
          100% of the new-regime test data and still accepted 74% of the
          obsolete regime, where ``'fifo'`` retained 97% of the new
          regime and accepted 70% of it. It MUST NOT be used under
          drift. (Flagging it at runtime would be a behaviour change and
          is deliberately not done; the responsibility is the caller's.)

        - ``'error_first'`` is self-defeating. The nu-property
          regenerates |E| ~ nu N' on the re-solve after every prune, so
          removing the error vectors only recreates them from the
          survivors (in-buffer rejection rose 0.24 -> 0.25 -> 0.37 over
          three prunes, with |E| back at 6), while permanently
          destroying the maximal-alpha points, which are the most
          informative ones.

        - ``'non_sv_first'`` rests on the premise that an alpha = 0
          point does not matter. The model's own geometry invalidates
          that for eta < 1: removing an O point changes N, H and K, and
          therefore every entry of A and G. Only at eta = 1 is the
          premise close to true.

        - ``'fifo'`` - the default - is the only one of the four that
          adapts to drift, simply because the oldest samples are the
          ones most likely to be obsolete.

        Selection is also batched on PRE-removal state while the removal
        and re-solve are sequential (``unlearn_obsolete`` calls this
        method once), so a prune with k > 1 ranks the second and later
        victims on a state that no longer exists.
        """
        strat = (
            strategy
            if strategy is not None
            else self.pruning_strategy
        )

        n_buffer = len(
            self.X_buffer_
        )

        if (
            n_buffer == 0
            or n_samples <= 0
        ):
            return []

        if strat not in {
            "fifo",
            "non_sv_first",
            "error_first",
            "mahalanobis",
        }:
            raise ValueError(
                f"Unknown pruning strategy: {strat}"
            )

        k = min(
            int(n_samples),
            n_buffer,
        )

        # -------------------------------------------------------------
        # FIFO
        # -------------------------------------------------------------
        if strat == "fifo":
            return list(
                range(k)
            )

        # -------------------------------------------------------------
        # Non-support vectors first.
        # -------------------------------------------------------------
        if strat == "non_sv_first":
            candidates = list(
                self.O_set_
            )

            remaining = [
                i
                for i in range(n_buffer)
                if i not in candidates
            ]

            candidates.extend(
                remaining
            )

            return candidates[:k]

        # -------------------------------------------------------------
        # Error vectors first.
        # -------------------------------------------------------------
        if strat == "error_first":
            candidates = list(
                self.E_set_
            )

            for group in (
                self.O_set_,
                self.S_set_,
            ):
                for idx in group:
                    if idx not in candidates:
                        candidates.append(idx)

            return candidates[:k]

        # -------------------------------------------------------------
        # Mahalanobis distance.
        #
        # This is feature-space Mahalanobis distance, not "kernel
        # covariance space". That distinction matters.
        # -------------------------------------------------------------
        X_mat = np.asarray(
            self.X_buffer_,
            dtype=float,
        )

        mean_x = np.mean(
            X_mat,
            axis=0,
        )

        diff = X_mat - mean_x

        n, d = X_mat.shape

        if n > 1:
            try:
                cov = np.cov(
                    X_mat,
                    rowvar=False,
                )

                cov = np.atleast_2d(
                    cov
                )

                # Adaptive small ridge.
                scale = float(
                    np.trace(cov)
                    / max(cov.shape[0], 1)
                )

                ridge = max(
                    1e-10,
                    1e-6 * max(scale, 1.0),
                )

                cov = (
                    cov
                    + ridge
                    * np.eye(d)
                )

                inv_cov = np.linalg.pinv(
                    cov,
                    rcond=1e-10,
                )

                dists = np.einsum(
                    "ij,jk,ik->i",
                    diff,
                    inv_cov,
                    diff,
                )

            except (
                np.linalg.LinAlgError,
                ValueError,
            ):
                dists = np.sum(
                    diff * diff,
                    axis=1,
                )

        else:
            dists = np.zeros(
                n,
                dtype=float,
            )

        ranked = np.argsort(
            -dists
        )

        return list(
            ranked[:k]
        )

    def unlearn_obsolete(
        self,
        n_samples: int = 1,
        strategy: Optional[
            PruningStrategy
        ] = None,
    ) -> "IncrementalDecrementalCOSVM":
        """
        Remove obsolete samples according to a pruning policy.
        """
        indices = self.select_obsolete_indices(
            n_samples=n_samples,
            strategy=strategy,
        )

        if indices:
            self.unlearn_many(
                indices
            )

        return self

    def _enforce_max_samples(self) -> None:
        """
        Enforce maximum active buffer size.
        """
        if self.max_samples is None:
            return

        excess = (
            len(self.X_buffer_)
            - self.max_samples
        )

        if excess > 0:
            self.unlearn_obsolete(
                n_samples=excess,
                strategy=self.pruning_strategy,
            )

    # =================================================================
    # Prediction
    # =================================================================

    def score_samples(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Return the raw COSVM score:

            score(x) = k(x, X)^T coef_ = k(x, X)^T A^{-1} K alpha
        """
        check_is_fitted(self)

        if not self.X_buffer_:
            raise RuntimeError(
                "Cannot score samples with an empty model."
            )

        X_arr = self._validate_X(X, n_features=getattr(self, "n_features_in_", None))

        K_test = np.asarray(
            self.kernel_(
                X_arr,
                np.asarray(self.X_buffer_),
            ),
            dtype=float,
        )

        return K_test @ self.coef_

    def decision_function(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Return:

            f(x) = score(x) - rho
        """
        return (
            self.score_samples(X)
            - self.offset_
        )

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict:

            +1 -> normal / target
            -1 -> outlier / anomaly
        """
        decision = self.decision_function(
            X
        )

        return np.where(
            decision >= 0.0,
            1,
            -1,
        )

    # =================================================================
    # Diagnostics
    # =================================================================

    def _kkt_residuals(self) -> Dict[str, float]:
        """
        Compute simple KKT diagnostics.

        With:

            score_i = (G alpha)_i   (G = K_tilde_full_)

        and decision convention:

            score_i - rho

        we expect approximately:

            alpha_i = 0:
                score_i - rho >= 0

            0 < alpha_i < C:
                score_i - rho = 0

            alpha_i = C:
                score_i - rho <= 0

        Both branches return exactly ``_KKT_RESIDUAL_KEYS``. The
        empty-buffer branch is reachable only through this private
        method: ``is_fitted`` is ``support_vectors_ is not None``
        (:mod:`dynamicsvm.base`), which goes False on a drained model, so
        ``check_is_fitted`` in ``get_kkt_diagnostics`` raises first.
        """
        if (
            not self.X_buffer_
            or self.K_tilde_full_ is None
        ):
            return _kkt_residual_dict(0.0, 0.0, 0.0, 0.0)

        # The empty-buffer branch above keeps C = inf out of this
        # product; the masks below use the one active-set tolerance of
        # the class, so they agree with S_set_ / E_set_ / O_set_.
        C = self._current_C()

        alpha_tol = self.sv_tol * C

        scores = (
            self.K_tilde_full_
            @ self.alpha_
        )

        residual = (
            scores
            - self.offset_
        )

        stationarity_errors = []

        for i, alpha_i in enumerate(
            self.alpha_
        ):
            if (
                alpha_i > alpha_tol
                and alpha_i < C - alpha_tol
            ):
                stationarity_errors.append(
                    abs(residual[i])
                )

        max_stationarity_error = (
            max(stationarity_errors)
            if stationarity_errors
            else 0.0
        )

        lower_violation = 0.0

        for i, alpha_i in enumerate(
            self.alpha_
        ):
            if alpha_i <= alpha_tol:
                lower_violation = max(
                    lower_violation,
                    -residual[i],
                )

        upper_violation = 0.0

        for i, alpha_i in enumerate(
            self.alpha_
        ):
            if alpha_i >= C - alpha_tol:
                upper_violation = max(
                    upper_violation,
                    residual[i],
                )

        return _kkt_residual_dict(
            max_stationarity_error,
            max(0.0, lower_violation),
            max(0.0, upper_violation),
            abs(float(np.sum(self.alpha_)) - 1.0),
        )

    def get_model_state(
        self,
    ) -> Dict[
        str,
        Union[
            int,
            float,
            str,
            List[int],
        ],
    ]:
        """
        Return a snapshot of the dynamic model state.
        """
        return {
            "buffer_size": len(
                self.X_buffer_
            ),
            "n_samples_seen": int(
                self.n_samples_seen_
            ),
            "n_unlearned": int(
                self.n_unlearned_
            ),
            "n_support_vectors": int(
                len(self.support_)
            )
            if self.support_ is not None
            else 0,
            "n_margin_svs": len(
                self.S_set_
            ),
            "n_error_svs": len(
                self.E_set_
            ),
            "n_non_svs": len(
                self.O_set_
            ),
            "offset": float(
                self.offset_
            ),
            "nu": float(
                self.nu
            ),
            "eta": float(
                self.eta
            ),
            "C": float(
                self._current_C()
            )
            if self.X_buffer_
            else 0.0,
            "pruning_strategy": self.pruning_strategy,
            "n_dual_solves": int(
                self.n_dual_solves_
            ),
            "kkt_gap": float(
                self.kkt_gap_
            ),
            "kkt_gap_relative": float(
                self.kkt_gap_relative_
            ),
            "eps": float(
                self.eps_
            ),
            "n_ridge_growths": int(
                self.n_ridge_growths_
            ),
        }

    def get_kkt_diagnostics(
        self,
    ) -> Dict[str, float]:
        """
        Return the optimality certificate of the current dual point.

        The headline quantities come from ``oneclass_kkt_report`` on the
        dual Gram matrix G = K_tilde_full_, i.e. the same scale-aware
        measure the solver's acceptance gate uses, with the free set
        identified by the class's single active-set tolerance
        ``sv_tol * C``.

        Returns
        -------
        dict with keys

        sum_alpha_minus_1
            |sum(alpha) - 1|, the equality-constraint residual.
        kkt_gap
            max_{alpha_i > eps} (G alpha)_i
            - min_{alpha_j < C - eps} (G alpha)_j, clipped at 0, in the
            units of G. 0 at the optimum.
        kkt_gap_relative
            The gap recorded by the solver divided by max_k G_kk, which
            is the dimensionless quantity the solver gate accepts below
            1e-6. It is the gap of the point the solver returned, so it
            can differ in the last digits from ``kkt_gap`` recomputed
            above.
        objective_scale
            max_k G_kk, the scale the relative gap is taken against.
        dual_objective
            0.5 alpha^T G alpha.
        box_violation
            max(0, -min alpha, max alpha - C).
        outlier_fraction
            Fraction of buffer points with (G alpha)_i - rho < 0.
        eps
            Ridge actually added to A by the last geometry rebuild.
        n_ridge_growths
            Number of x10 growths that ridge needed.
        n_dual_solves
            Lifetime number of dual solves.
        max_stationarity_error, max_lower_kkt_violation,
        max_upper_kkt_violation, sum_alpha_error
            The four ``_kkt_residuals`` entries, kept reachable: the
            per-set residuals of score_i - rho against the decision
            convention.
        """
        check_is_fitted(self)

        C = self._current_C()

        report = oneclass_kkt_report(
            self.alpha_,
            self.K_tilde_full_,
            C,
            rho=self.offset_,
            eps=self.sv_tol * C,
        )

        report["kkt_gap_relative"] = float(
            self.kkt_gap_relative_
        )
        report["objective_scale"] = float(
            max(
                float(
                    np.max(
                        np.diag(self.K_tilde_full_)
                    )
                ),
                np.finfo(float).tiny,
            )
        )

        report["eps"] = float(self.eps_)
        report["n_ridge_growths"] = int(
            self.n_ridge_growths_
        )
        report["n_dual_solves"] = int(
            self.n_dual_solves_
        )

        report.update(self._kkt_residuals())

        return report


# ---------------------------------------------------------------------
# Short alias
# ---------------------------------------------------------------------

IDCOSVM = IncrementalDecrementalCOSVM
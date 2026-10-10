"""
Incremental Covariance-guided One-Class Support Vector Machine (ICOSVM).

Based on:

    Takoua Kefi-Fatteh, Riadh Ksantini, Mohamed-Becha Kaaniche,
    Adel Bouhoula.

    "A novel incremental one-class support vector machine based
    on low variance direction."

    Pattern Recognition 91 (2019): 308-321.

The implementation maintains a COSVM model over a growing (optionally
windowed) stream.

Mathematical definitions
------------------------

ICOSVM shares the formulation of :class:`dynamicsvm.models.cosvm.COSVM`
(derivation in ``dynamicsvm.models._cosvm_core``). For the N buffered
samples, H = I - (1/N) 11^T and

    G = K A^{-1} K,   A = eta K + ((1 - eta)/N) K H K,   0 < eta <= 1.

The dual optimization problem is:

    minimize
        1/2 alpha^T G alpha

    subject to

        0 <= alpha_i <= C

        sum_i alpha_i = 1

where

        C = 1 / (nu N),

and the decision function is f(x) = k(x, X)^T A^{-1} K alpha - rho.

The KKT gradient is

    g_i = (G alpha)_i - rho,

with the three standard sets

    S:  0 < alpha_i < C   (g_i = 0)
    E:  alpha_i = C       (g_i <= 0)
    O:  alpha_i = 0       (g_i >= 0)

What an update does
-------------------
Every learn_one / unlearn_one is a warm-started exact re-solve of the
current finite COSVM dual. Adding or removing a sample changes N, hence
C = 1/(nu N), H and A^{-1}, and therefore every entry of G: the previous
alpha is in general not KKT for the new problem, and the single-candidate
Cauwenberghs-Poggio path does not apply. The update therefore

1. rebuilds K, G and the Cholesky factor of A (O(N^3));
2. tries a KKT predictor: the previous alpha extended by alpha_c = 0,
   followed by one sensitivity step on the candidate c over the current
   margin set S, using the bordered system

       [ 0     1^T  ] [ beta_0 ]      [ 1    ]
       [ 1     G_SS ] [ beta_S ] =  - [ G_Sc ],

   d alpha_S / d alpha_c = beta_S, d rho / d alpha_c = -beta_0 and
   d g_c / d alpha_c = G_cc + beta_0 + G_cS beta_S. The predictor is
   accepted only if the complete KKT check of the NEW problem passes and
   no coefficient leaves the box (no clipping);
3. otherwise re-solves the dual exactly (SLSQP from the feasible warm
   start, refined to the KKT conditions).

Because G changes globally with N, the predictor is rarely accepted for
eta < 1 (for eta = 1, G = K does not change for the old samples and it is
accepted more often); ``n_incremental_updates_`` and ``n_qp_solves_``
report how often each path ran. After a re-solve the model equals batch
COSVM on the same buffer and kernel to solver precision; after an accepted
predictor it satisfies the KKT conditions of that problem to ``kkt_tol``.
The cost of an update is that of a batch fit (O(N^3)), not O(N_s^2).
"""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from scipy.linalg import cho_solve

from dynamicsvm.base import BaseSVM
from dynamicsvm.kernels import Kernel, get_kernel
from dynamicsvm.models._cosvm_core import (
    COSVMGeometry,
    compute_rho,
    feasible_start,
    solve_oneclass_dual,
    validate_eta,
)
from dynamicsvm.utils.validation import check_array, check_is_fitted


def _sensitivity_step(
    G: np.ndarray,
    S: Union[List[int], np.ndarray],
    c: int,
) -> Optional[Tuple[float, np.ndarray, float]]:
    """
    Linear response of a one-class KKT state to increasing alpha_c.

    Keeping (G alpha)_s - rho = 0 for s in S and sum(alpha) = 1 while
    alpha_c grows requires

        G_SS beta_S + G_Sc + beta_0 1 = 0,   1^T beta_S = -1,

    i.e. beta = -M^{-1} [1; G_Sc] with M = [[0, 1^T], [1, G_SS]].

    Returns
    -------
    (d_rho, d_alpha_S, d_g_c) or None
        d_rho = -beta_0, d_alpha_S = beta_S and
        d_g_c = G_cc + beta_0 + G_cS beta_S, all per unit increase of
        alpha_c. None when M is singular.
    """
    G = np.asarray(G, dtype=float)
    s_idx = np.asarray(S, dtype=int)
    n_s = s_idx.size

    M = np.zeros((n_s + 1, n_s + 1), dtype=float)
    M[0, 1:] = 1.0
    M[1:, 0] = 1.0
    M[1:, 1:] = G[np.ix_(s_idx, s_idx)]

    rhs = np.concatenate(([1.0], G[s_idx, c]))

    try:
        beta = -np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(beta)):
        return None

    beta_0 = float(beta[0])
    beta_S = beta[1:]

    d_rho = -beta_0
    d_g_c = float(
        G[c, c]
        + beta_0
        + G[c, s_idx] @ beta_S
    )

    return d_rho, beta_S, d_g_c


class ICOSVM(BaseSVM):
    """
    Incremental Covariance-guided One-Class SVM.

    Parameters
    ----------
    nu : float, default=0.1
        Upper bound on the fraction of training errors and lower bound
        on the fraction of support vectors.

    eta : float, default=0.5
        Weight of the identity in the primal metric
        M = eta I + (1 - eta) Sigma, in (0, 1].

            eta = 1:
                Standard One-Class SVM.

            0 < eta < 1:
                Covariance-guided COSVM.

    kernel : {'rbf', 'linear', 'poly', 'sigmoid'} or Kernel or Callable
        Kernel specification.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient (resolved once, from the data given to fit).

    degree : int, default=3
        Polynomial kernel degree.

    coef0 : float, default=0.0
        Polynomial/sigmoid kernel coefficient.

    max_samples : int or None, default=None
        Maximum number of samples retained in the stream window.

    tol : float, default=1e-5
        Numerical tolerance. SLSQP runs at ftol = min(tol, 1e-10).

    max_iter : int, default=1000
        Maximum number of SLSQP iterations of the exact re-solve.

    sv_tol : float, default=1e-7
        Support-vector classification tolerance.

    kkt_tol : float, default=1e-5
        KKT tolerance used to accept the predictor step.

    Attributes
    ----------
    alpha_ : ndarray
        Full dual coefficient vector.

    coef_ : ndarray
        Score coefficients A^{-1} K alpha; score(x) = k(x, X)^T coef_.

    support_ : ndarray
        Support-vector indices.

    support_vectors_ : ndarray
        Current support vectors.

    dual_coef_ : ndarray
        Non-zero dual coefficients.

    rho_ : float
        Decision-function offset.

    offset_ : float
        Alias for rho_.

    intercept_ : float
        SVM-style intercept (-rho).

    S_set_ : list
        Margin support-vector indices.

    E_set_ : list
        Error-vector indices.

    O_set_ : list
        Non-support-vector indices.

    K_full_ : ndarray
        Current kernel Gram matrix.

    K_tilde_full_ : ndarray
        Current dual Gram matrix G = K A^{-1} K.

    geometry_ : COSVMGeometry
        Factorised geometry of the current buffer.

    R_ : ndarray or None
        Inverse of the bordered margin matrix [[0, 1^T], [1, G_SS]]
        (diagnostic).

    n_samples_seen_ : int
        Number of samples observed by the stream (not decreased by
        unlearning).

    n_unlearned_ : int
        Number of samples removed (unlearn_one or the sliding window).

    n_incremental_updates_ : int
        learn_one calls resolved by the accepted KKT predictor.

    n_qp_solves_ : int
        learn_one / unlearn_one calls resolved by an exact re-solve.
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
        tol: float = 1e-5,
        max_iter: int = 1000,
        sv_tol: float = 1e-7,
        kkt_tol: float = 1e-5,
    ):
        super().__init__()

        if not (0.0 < nu <= 1.0):
            raise ValueError(
                f"nu must be in (0, 1], got {nu}"
            )

        validate_eta(eta)

        if tol <= 0:
            raise ValueError(
                f"tol must be > 0, got {tol}"
            )

        if max_iter <= 0:
            raise ValueError(
                f"max_iter must be > 0, got {max_iter}"
            )

        if sv_tol <= 0:
            raise ValueError(
                f"sv_tol must be > 0, got {sv_tol}"
            )

        if kkt_tol <= 0:
            raise ValueError(
                f"kkt_tol must be > 0, got {kkt_tol}"
            )

        if max_samples is not None:
            if max_samples < 1:
                raise ValueError(
                    "max_samples must be >= 1."
                )
            max_samples = int(max_samples)

        self.nu = float(nu)
        self.eta = float(eta)

        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)

        self.max_samples = max_samples

        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self.sv_tol = float(sv_tol)
        self.kkt_tol = float(kkt_tol)

        # --------------------------------------------------------------
        # Model state
        # --------------------------------------------------------------

        self.kernel_: Optional[Kernel] = None

        self.X_buffer_: List[np.ndarray] = []

        self.alpha_: np.ndarray = np.empty(
            0,
            dtype=float,
        )
        self.coef_: np.ndarray = np.empty(
            0,
            dtype=float,
        )

        self.rho_: float = 0.0
        self.offset_: float = 0.0
        self.intercept_: float = 0.0

        self.R_: Optional[np.ndarray] = None

        self.S_set_: List[int] = []
        self.E_set_: List[int] = []
        self.O_set_: List[int] = []

        self.K_full_: Optional[np.ndarray] = None
        self.K_tilde_full_: Optional[np.ndarray] = None
        self.geometry_: Optional[COSVMGeometry] = None

        self.n_samples_seen_: int = 0
        self.n_unlearned_: int = 0
        self.n_incremental_updates_: int = 0
        self.n_qp_solves_: int = 0

        self.C_: float = 1.0

        self.optimization_result_ = None

    # ==================================================================
    # Kernel / covariance matrix
    # ==================================================================

    @staticmethod
    def _centering_matrix(
        n_samples: int,
    ) -> np.ndarray:
        """
        Return

            H = I - (1/N)11^T.
        """
        return (
            np.eye(n_samples)
            - np.ones(
                (n_samples, n_samples)
            ) / n_samples
        )

    def _compute_covariance_kernel_matrix(
        self,
        K: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the COSVM dual Gram matrix

            G = K A^{-1} K,   A = eta K + ((1 - eta)/N) K H K

        (shared with COSVM / IDCOSVM, see ``_cosvm_core``).
        """
        return COSVMGeometry(K, self.eta).G

    def _recompute_full_matrices(
        self,
    ) -> None:
        """
        Recompute the kernel matrix and the COSVM geometry.

        This is necessary because adding/removing a sample changes
        N, H and A^{-1}, and therefore every entry of G.
        """
        X_mat = np.asarray(
            self.X_buffer_,
            dtype=float,
        )

        self.K_full_ = np.asarray(
            self.kernel_(
                X_mat,
                X_mat,
            ),
            dtype=float,
        )

        self.K_full_ = 0.5 * (
            self.K_full_
            + self.K_full_.T
        )

        self.geometry_ = COSVMGeometry(
            self.K_full_,
            self.eta,
        )

        self.K_tilde_full_ = self.geometry_.G

    # ==================================================================
    # Batch QP solver
    # ==================================================================

    def _solve_batch_qp(
        self,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solve the current COSVM dual problem exactly.

        SLSQP (ftol = min(tol, 1e-10)) starts from x0, or from the current
        alpha_ when it has the right length, made feasible for the current
        box by clipping and water-filling; the result is refined to the
        KKT conditions (``_cosvm_core.solve_oneclass_dual``).
        """
        n_samples = len(
            self.X_buffer_
        )

        C = 1.0 / (
            self.nu * n_samples
        )

        self.C_ = C

        if x0 is None and self.alpha_.size == n_samples:
            x0 = self.alpha_

        start = feasible_start(
            x0,
            n_samples,
            C,
        )

        alpha, result = solve_oneclass_dual(
            self.K_tilde_full_,
            C,
            x0=start,
            ftol=min(self.tol, 1e-10),
            max_iter=self.max_iter,
        )

        self.optimization_result_ = result

        self.alpha_ = alpha
        self.coef_ = self.geometry_.coef(alpha)

        return alpha

    # ==================================================================
    # KKT partition
    # ==================================================================

    def _update_sets(
        self,
    ) -> None:
        """
        Rebuild S/E/O sets from alpha.

        S:
            0 < alpha_i < C

        E:
            alpha_i = C

        O:
            alpha_i = 0
        """
        n_samples = len(
            self.alpha_
        )

        self.S_set_ = []
        self.E_set_ = []
        self.O_set_ = []

        for i in range(n_samples):
            a = self.alpha_[i]

            if (
                a > self.sv_tol
                and a < self.C_ - self.sv_tol
            ):
                self.S_set_.append(i)

            elif (
                a >= self.C_ - self.sv_tol
            ):
                self.E_set_.append(i)

            else:
                self.O_set_.append(i)

    # ==================================================================
    # R matrix
    # ==================================================================

    def _build_R_matrix(
        self,
    ) -> None:
        """
        Build the inverse of the bordered margin matrix:

            R =
                [ 0      1_S^T ]
                [ 1_S    G_SS  ]^{-1}

        Diagnostic only; the predictor solves the system directly
        (``_sensitivity_step``). A pseudoinverse is used only as a
        numerical fallback.
        """
        if not self.S_set_:
            self.R_ = None
            return

        s_idx = np.asarray(
            self.S_set_,
            dtype=int,
        )

        n_s = len(
            s_idx
        )

        M = np.zeros(
            (n_s + 1, n_s + 1),
            dtype=float,
        )

        M[0, 1:] = 1.0
        M[1:, 0] = 1.0

        M[1:, 1:] = (
            self.K_tilde_full_[
                np.ix_(
                    s_idx,
                    s_idx,
                )
            ]
        )

        # Symmetrize before inversion.
        M = 0.5 * (
            M + M.T
        )

        try:
            self.R_ = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            self.R_ = np.linalg.pinv(
                M,
                rcond=1e-10,
            )

    # ==================================================================
    # Rho
    # ==================================================================

    def _compute_rho(
        self,
    ) -> None:
        """
        Compute rho from the KKT conditions (shared with COSVM/IDCOSVM).

        For free support vectors (G alpha)_i = rho; without free support
        vectors the midpoint of the KKT interval is used.
        """
        rho = compute_rho(
            self.K_tilde_full_,
            self.alpha_,
            self.C_,
        )

        self.rho_ = rho
        self.offset_ = rho
        self.intercept_ = -rho

    # ==================================================================
    # Initialization
    # ==================================================================

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> "ICOSVM":
        """
        Initialize ICOSVM from an initial batch (n_samples >= 1).

        The initial model is solved using the exact batch COSVM dual.
        With a single row and gamma='scale' the kernel falls back to
        gamma = 1/n_features with a UserWarning (the variance of one
        vector is not defined).
        """
        validate_eta(self.eta)

        X_arr = check_array(X)

        n_samples, n_features = (
            X_arr.shape
        )
        self.n_features_in_ = n_features

        if n_samples < 1:
            raise ValueError(
                "ICOSVM requires at least one "
                "initial training sample."
            )

        # --------------------------------------------------------------
        # Kernel initialization
        # --------------------------------------------------------------

        self.kernel_ = get_kernel(
            self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            n_features=n_features,
            X=X_arr,
        )

        # --------------------------------------------------------------
        # Buffer and counters
        # --------------------------------------------------------------

        self.X_buffer_ = [
            X_arr[i].copy()
            for i in range(n_samples)
        ]

        self.n_samples_seen_ = n_samples
        self.n_unlearned_ = 0
        self.n_incremental_updates_ = 0
        self.n_qp_solves_ = 0

        # --------------------------------------------------------------
        # Kernel matrices
        # --------------------------------------------------------------

        self._recompute_full_matrices()

        self.C_ = 1.0 / (
            self.nu * n_samples
        )

        # --------------------------------------------------------------
        # Exact initial COSVM solution (uniform start, like COSVM.fit)
        # --------------------------------------------------------------

        self.alpha_ = np.full(
            n_samples,
            1.0 / n_samples,
        )

        self._solve_batch_qp()

        # --------------------------------------------------------------
        # KKT state
        # --------------------------------------------------------------

        self._update_sets()
        self._compute_rho()
        self._build_R_matrix()
        self._sync_support_vectors()

        return self

    # ==================================================================
    # KKT predictor
    # ==================================================================

    def _incremental_candidate_update(
        self,
        candidate_index: int,
    ) -> bool:
        """
        KKT predictor for a newly added candidate c (alpha_c = 0).

        rho is recomputed for the new G; if the candidate violates its
        O-set condition (g_c < 0), alpha_c is increased along the
        sensitivity direction of ``_sensitivity_step`` until g_c = 0 or
        alpha_c = C. A step that would move any alpha_S outside [0, C]
        is rejected (no clipping).

        Returns
        -------
        success : bool
            True if a candidate state was produced. It is only a
            prediction: the caller accepts it only if the complete KKT
            check of the new problem passes.
        """
        c = candidate_index

        C = self.C_

        self._compute_rho()
        self._update_sets()

        g_c = float(
            self.K_tilde_full_[c]
            @ self.alpha_
            - self.rho_
        )

        # Candidate already satisfies the O-set KKT condition.
        if g_c >= -self.kkt_tol:
            return True

        s_idx = np.asarray(
            [s for s in self.S_set_ if s != c],
            dtype=int,
        )

        if s_idx.size == 0:
            return False

        step = _sensitivity_step(
            self.K_tilde_full_,
            s_idx,
            c,
        )

        if step is None:
            return False

        d_rho, d_alpha_S, d_g_c = step

        # g_c must increase towards 0 as alpha_c grows.
        if d_g_c <= 1e-12:
            return False

        delta = min(
            -g_c / d_g_c,
            C,
        )

        alpha_new = self.alpha_.copy()

        alpha_new[c] = delta
        alpha_new[s_idx] += d_alpha_S * delta

        # Reject on any bound violation (no clipping).
        if (
            np.any(alpha_new[s_idx] < 0.0)
            or np.any(alpha_new[s_idx] > C)
        ):
            return False

        self.alpha_ = alpha_new

        self.rho_ = float(
            self.rho_ + d_rho * delta
        )
        self.offset_ = self.rho_
        self.intercept_ = -self.rho_

        self._update_sets()

        return True

    # ==================================================================
    # KKT verification
    # ==================================================================

    def _check_kkt(
        self,
    ) -> bool:
        """
        Check primal feasibility and approximate KKT conditions.

        Conditions:

            sum(alpha) = 1

            0 <= alpha_i <= C

        and with

            g_i = (G alpha)_i - rho

        we require approximately:

            alpha_i = 0:
                g_i >= 0

            0 < alpha_i < C:
                g_i = 0

            alpha_i = C:
                g_i <= 0.
        """
        if self.alpha_.size == 0:
            return False

        alpha = self.alpha_
        C = self.C_

        # --------------------------------------------------------------
        # Primal feasibility
        # --------------------------------------------------------------

        if (
            abs(
                np.sum(alpha) - 1.0
            )
            > self.kkt_tol
        ):
            return False

        if (
            np.min(alpha)
            < -self.kkt_tol
        ):
            return False

        if (
            np.max(alpha - C)
            > self.kkt_tol
        ):
            return False

        # --------------------------------------------------------------
        # Stationarity / KKT
        # --------------------------------------------------------------

        g = (
            self.K_tilde_full_
            @ alpha
            - self.rho_
        )

        zero_mask = (
            alpha <= self.sv_tol
        )

        free_mask = (
            (alpha > self.sv_tol)
            & (alpha < C - self.sv_tol)
        )

        C_mask = (
            alpha >= C - self.sv_tol
        )

        if np.any(
            g[zero_mask]
            < -self.kkt_tol
        ):
            return False

        if np.any(
            np.abs(
                g[free_mask]
            )
            > self.kkt_tol
        ):
            return False

        if np.any(
            g[C_mask]
            > self.kkt_tol
        ):
            return False

        return True

    # ==================================================================
    # Full correction
    # ==================================================================

    def _correct_with_batch_qp(
        self,
    ) -> None:
        """
        Re-solve the current complete COSVM dual (the exact update path).

        The current alpha_ (previous solution extended by alpha_c = 0, or
        with the removed entry deleted) is the warm start; it is made
        feasible for the new box inside ``_solve_batch_qp``.
        """
        self._solve_batch_qp()

        self.n_qp_solves_ += 1

        self._update_sets()
        self._compute_rho()
        self._build_R_matrix()

    # ==================================================================
    # Learn one
    # ==================================================================

    def learn_one(
        self,
        x: np.ndarray,
    ) -> "ICOSVM":
        """
        Incorporate one new observation.

        The geometry is rebuilt for the new buffer, the KKT predictor is
        tried and accepted only if the new problem's KKT conditions hold;
        otherwise the dual is re-solved exactly from the warm start.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            New stream observation.

        Returns
        -------
        self
        """
        x_vec = np.asarray(
            x,
            dtype=float,
        ).reshape(-1)

        # --------------------------------------------------------------
        # First observation (empty model): fit on the single row.
        # --------------------------------------------------------------

        if not self.X_buffer_:
            self.fit(
                x_vec.reshape(
                    1,
                    -1,
                )
            )
            self._enforce_max_samples()
            return self

        # --------------------------------------------------------------
        # Check dimensionality.
        # --------------------------------------------------------------

        n_features = (
            self.X_buffer_[0].shape[0]
        )

        if x_vec.shape[0] != n_features:
            raise ValueError(
                "Incoming sample has "
                f"{x_vec.shape[0]} features, "
                f"expected {n_features}."
            )

        # --------------------------------------------------------------
        # Add sample.
        # --------------------------------------------------------------

        self.X_buffer_.append(
            x_vec.copy()
        )

        self.n_samples_seen_ += 1

        n_samples = len(
            self.X_buffer_
        )

        c = n_samples - 1

        # C changes with N:
        #
        # C = 1 / (nu N)
        #
        self.C_ = 1.0 / (
            self.nu * n_samples
        )

        # --------------------------------------------------------------
        # Rebuild K, G and A^{-1} (H and N changed).
        # --------------------------------------------------------------

        self._recompute_full_matrices()

        # --------------------------------------------------------------
        # Extend alpha with candidate alpha_c = 0.
        # --------------------------------------------------------------

        alpha_extended = np.append(
            self.alpha_,
            0.0,
        )

        self.alpha_ = alpha_extended.copy()

        # --------------------------------------------------------------
        # KKT predictor; only meaningful if the old coefficients are
        # still inside the new (smaller) box.
        # --------------------------------------------------------------

        accepted = False

        if np.max(alpha_extended) <= self.C_:
            if (
                self._incremental_candidate_update(c)
                and self._check_kkt()
            ):
                # rho from the shared rule, then re-check.
                self._compute_rho()
                accepted = self._check_kkt()

        if accepted:
            self.n_incremental_updates_ += 1
            self.coef_ = self.geometry_.coef(self.alpha_)
            self._update_sets()
            self._build_R_matrix()

        else:
            # Warm start: previous solution extended by alpha_c = 0.
            self.alpha_ = alpha_extended
            self._correct_with_batch_qp()

        self._sync_support_vectors()

        # --------------------------------------------------------------
        # Sliding-window enforcement.
        # --------------------------------------------------------------

        self._enforce_max_samples()

        return self

    # ==================================================================
    # Batch incremental learning
    # ==================================================================

    def learn_batch(
        self,
        X: np.ndarray,
    ) -> "ICOSVM":
        """
        Sequentially learn a batch of observations.
        """
        X_arr = check_array(X, n_features=getattr(self, "n_features_in_", None))

        for i in range(
            X_arr.shape[0]
        ):
            self.learn_one(
                X_arr[i]
            )

        return self

    # ==================================================================
    # Decremental learning
    # ==================================================================

    def unlearn_one(
        self,
        index: int,
    ) -> "ICOSVM":
        """
        Remove one sample from the current buffer.

        The remaining problem (new N, C, H and G) is re-solved exactly
        from the remaining coefficients as a warm start.
        ``n_samples_seen_`` is not changed (it counts observed samples);
        ``n_unlearned_`` is incremented.

        Parameters
        ----------
        index : int
            Index in the current buffer.

        Returns
        -------
        self
        """
        if not self.X_buffer_:
            raise ValueError(
                "ICOSVM has no samples."
            )

        n_samples = len(
            self.X_buffer_
        )

        if (
            index < 0
            or index >= n_samples
        ):
            raise IndexError(
                f"Index {index} out of range "
                f"for buffer size {n_samples}."
            )

        # --------------------------------------------------------------
        # Remove sample.
        # --------------------------------------------------------------

        del self.X_buffer_[index]

        self.n_unlearned_ += 1

        if not self.X_buffer_:
            self.alpha_ = np.empty(
                0,
                dtype=float,
            )
            self.coef_ = np.empty(
                0,
                dtype=float,
            )

            self.S_set_ = []
            self.E_set_ = []
            self.O_set_ = []

            self.K_full_ = None
            self.K_tilde_full_ = None
            self.geometry_ = None

            self.R_ = None

            self.rho_ = 0.0
            self.offset_ = 0.0
            self.intercept_ = 0.0

            self._sync_support_vectors()

            return self

        # --------------------------------------------------------------
        # Recompute complete matrices.
        # --------------------------------------------------------------

        remaining_alpha = np.delete(
            self.alpha_,
            index,
        )

        self.alpha_ = remaining_alpha

        self._recompute_full_matrices()

        n_samples = len(
            self.X_buffer_
        )

        self.C_ = 1.0 / (
            self.nu * n_samples
        )

        # --------------------------------------------------------------
        # G changes globally, so re-solve the complete dual.
        # --------------------------------------------------------------

        # Warm start: the remaining coefficients (self.alpha_).
        self._correct_with_batch_qp()

        self._sync_support_vectors()

        return self

    # ==================================================================
    # Sliding window
    # ==================================================================

    def _enforce_max_samples(
        self,
    ) -> None:
        """
        Keep only the most recent max_samples observations.
        """
        if self.max_samples is None:
            return

        while (
            len(self.X_buffer_)
            > self.max_samples
        ):
            self.unlearn_one(0)

    # ==================================================================
    # Support-vector synchronization
    # ==================================================================

    def _sync_support_vectors(
        self,
    ) -> None:
        """
        Synchronize public SVM attributes.
        """
        if not self.X_buffer_:
            self.support_ = np.empty(
                0,
                dtype=int,
            )

            self.support_vectors_ = None

            self.dual_coef_ = np.empty(
                0,
                dtype=float,
            )

            self.n_support_ = np.array(
                [0],
                dtype=int,
            )

            return

        sv_indices = np.flatnonzero(
            self.alpha_
            > self.sv_tol
        )

        X_mat = np.asarray(
            self.X_buffer_,
            dtype=float,
        )

        self.support_ = sv_indices

        self.support_vectors_ = (
            X_mat[sv_indices]
        )

        self.dual_coef_ = (
            self.alpha_[sv_indices]
        )

        self.n_support_ = np.array(
            [sv_indices.size],
            dtype=int,
        )

    # ==================================================================
    # Test-time kernels
    # ==================================================================

    def _test_kernel(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Raw kernel k(X, X_buffer).
        """
        check_is_fitted(self)

        X_arr = check_array(X, n_features=getattr(self, "n_features_in_", None))

        X_train = np.asarray(
            self.X_buffer_,
            dtype=float,
        )

        return np.asarray(
            self.kernel_(
                X_arr,
                X_train,
            ),
            dtype=float,
        )

    def _transform_test_kernel(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Dual cross-Gram between test and buffered samples,

            G(X, X_buffer) = k(X, X_buffer) A^{-1} K,

        whose rows on buffered points are rows of G.
        """
        K_test = self._test_kernel(X)

        if self.geometry_.cho_ is None:
            return K_test

        return K_test @ cho_solve(
            self.geometry_.cho_,
            self.K_full_,
        )

    # ==================================================================
    # Scoring
    # ==================================================================

    def score_samples(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Compute raw COSVM/ICOSVM scores:

            score(x) = k(x, X)^T coef_ = k(x, X)^T A^{-1} K alpha.
        """
        return (
            self._test_kernel(X)
            @ self.coef_
        )

    def decision_function(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Compute:

            f(x) = score(x) - rho.
        """
        return (
            self.score_samples(X)
            - self.rho_
        )

    # ==================================================================
    # Prediction
    # ==================================================================

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Return:

            +1 : normal / target
            -1 : anomaly / outlier
        """
        decision = (
            self.decision_function(X)
        )

        return np.where(
            decision >= 0.0,
            1,
            -1,
        )

    # ==================================================================
    # Diagnostics
    # ==================================================================

    def kkt_residuals(
        self,
    ) -> dict:
        """
        Return numerical KKT diagnostics.

        Returns
        -------
        dict
            Contains:

                sum_alpha_error
                min_alpha
                max_alpha
                max_zero_violation
                max_free_violation
                max_C_violation
                is_kkt
                n_incremental_updates
                n_qp_solves
        """
        check_is_fitted(self)

        alpha = self.alpha_
        C = self.C_

        g = (
            self.K_tilde_full_
            @ alpha
            - self.rho_
        )

        zero_mask = (
            alpha <= self.sv_tol
        )

        free_mask = (
            (alpha > self.sv_tol)
            & (alpha < C - self.sv_tol)
        )

        C_mask = (
            alpha >= C - self.sv_tol
        )

        zero_violation = (
            max(
                0.0,
                float(
                    -np.min(
                        g[zero_mask]
                    )
                ),
            )
            if np.any(zero_mask)
            else 0.0
        )

        free_violation = (
            float(
                np.max(
                    np.abs(
                        g[free_mask]
                    )
                )
            )
            if np.any(free_mask)
            else 0.0
        )

        C_violation = (
            max(
                0.0,
                float(
                    np.max(
                        g[C_mask]
                    )
                ),
            )
            if np.any(C_mask)
            else 0.0
        )

        return {
            "sum_alpha_error": abs(
                float(
                    np.sum(alpha)
                ) - 1.0
            ),
            "min_alpha": float(
                np.min(alpha)
            ),
            "max_alpha": float(
                np.max(alpha)
            ),
            "max_zero_violation": zero_violation,
            "max_free_violation": free_violation,
            "max_C_violation": C_violation,
            "is_kkt": self._check_kkt(),
            "n_incremental_updates": int(
                self.n_incremental_updates_
            ),
            "n_qp_solves": int(
                self.n_qp_solves_
            ),
        }

    @property
    def n_support(
        self,
    ) -> int:
        """
        Number of current support vectors.
        """
        check_is_fitted(self)

        return int(
            len(self.support_)
        )

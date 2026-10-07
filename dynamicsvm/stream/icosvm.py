"""
Incremental Covariance-guided One-Class Support Vector Machine (ICOSVM).

Based on:

    Takoua Kefi-Fatteh, Riadh Ksantini, Mohamed-Becha Kaaniche,
    Adel Bouhoula.

    "A novel incremental one-class support vector machine based
    on low variance direction."

    Pattern Recognition 91 (2019): 308-321.

The implementation maintains a COSVM model over a growing stream.

Important mathematical definitions
-----------------------------------

For N training samples:

    H = I - (1/N) 11^T

and

    K_tilde = eta K + (1 - eta) K H K^T.

The dual optimization problem is:

    minimize
        1/2 alpha^T K_tilde alpha

    subject to

        0 <= alpha_i <= C

        sum_i alpha_i = 1

where

        C = 1 / (nu N).

The KKT gradient is

    g_i = sum_j K_tilde[i,j] alpha_j - rho.

The three standard sets are:

    S:
        0 < alpha_i < C

    E:
        alpha_i = C

    O:
        alpha_i = 0

For a candidate sample c entering through an incremental KKT update,
the sensitivity system is built from the free support vectors S:

    [ 0       1^T ] [d rho / d alpha_c]   [1]
    [ 1       K_SS] [d alpha_S / d alpha_c] = -[K_Sc]

The implementation uses the equivalent augmented matrix formulation
and verifies the resulting state numerically.

Notes
-----
Because the covariance-guided kernel itself depends on the complete
dataset, adding a sample changes H and therefore changes K_tilde
globally. Consequently, an exact incremental COSVM implementation
must account for this global covariance update.

This implementation therefore uses KKT updates as the primary
incremental mechanism and performs a full QP correction only when
the resulting state is no longer sufficiently feasible/KKT-consistent.
This provides a robust research implementation while preserving the
incremental architecture.
"""

from typing import Callable, List, Optional, Union

import numpy as np
from scipy.optimize import minimize

from dynamicsvm.base import BaseSVM
from dynamicsvm.kernels import Kernel, get_kernel
from dynamicsvm.utils.validation import check_array, check_is_fitted


class ICOSVM(BaseSVM):
    """
    Incremental Covariance-guided One-Class SVM.

    Parameters
    ----------
    nu : float, default=0.1
        Upper bound on the fraction of training errors and lower bound
        on the fraction of support vectors.

    eta : float, default=0.5
        Trade-off between the standard kernel matrix and the
        covariance-guided component.

            eta = 1:
                Standard One-Class SVM.

            eta = 0:
                Pure covariance-guided component.

            0 < eta < 1:
                Combined COSVM model.

    kernel : {'rbf', 'linear', 'poly', 'sigmoid'} or Kernel or Callable
        Kernel specification.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient.

    degree : int, default=3
        Polynomial kernel degree.

    coef0 : float, default=0.0
        Polynomial/sigmoid kernel coefficient.

    max_samples : int or None, default=None
        Maximum number of samples retained in the stream window.

    tol : float, default=1e-5
        Numerical tolerance.

    max_iter : int, default=1000
        Maximum number of iterations for the fallback QP solver.

    sv_tol : float, default=1e-7
        Support-vector classification tolerance.

    kkt_tol : float, default=1e-5
        KKT feasibility tolerance.

    Attributes
    ----------
    alpha_ : ndarray
        Full dual coefficient vector.

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
        Current covariance-guided Gram matrix.

    R_ : ndarray or None
        Inverse KKT sensitivity matrix.

    n_samples_seen_ : int
        Number of samples observed by the stream.
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

        if not (0.0 <= eta <= 1.0):
            raise ValueError(
                f"eta must be in [0, 1], got {eta}"
            )

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

        self.rho_: float = 0.0
        self.offset_: float = 0.0
        self.intercept_: float = 0.0

        self.R_: Optional[np.ndarray] = None

        self.S_set_: List[int] = []
        self.E_set_: List[int] = []
        self.O_set_: List[int] = []

        self.K_full_: Optional[np.ndarray] = None
        self.K_tilde_full_: Optional[np.ndarray] = None

        self.n_samples_seen_: int = 0

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
        Compute

            K_tilde
                = eta K
                + (1 - eta) K H K^T.
        """
        K = np.asarray(
            K,
            dtype=float,
        )

        n_samples = K.shape[0]

        if n_samples <= 1:
            return K.copy()

        H = self._centering_matrix(
            n_samples
        )

        covariance_component = (
            K @ H @ K.T
        )

        K_tilde = (
            self.eta * K
            + (1.0 - self.eta)
            * covariance_component
        )

        return 0.5 * (
            K_tilde + K_tilde.T
        )

    def _recompute_full_matrices(
        self,
    ) -> None:
        """
        Recompute the complete kernel and covariance-guided matrices.

        This is necessary because adding/removing a sample changes:

            H = I - 11^T/N

        and therefore changes the covariance-guided matrix globally.
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

        self.K_tilde_full_ = (
            self._compute_covariance_kernel_matrix(
                self.K_full_
            )
        )

    # ==================================================================
    # Batch QP solver
    # ==================================================================

    def _solve_batch_qp(
        self,
    ) -> np.ndarray:
        """
        Solve the current COSVM dual problem exactly using SLSQP.

        This method is used for initialization and as a numerical
        correction when the incremental KKT state becomes unreliable.
        """
        n_samples = len(
            self.X_buffer_
        )

        C = 1.0 / (
            self.nu * n_samples
        )

        self.C_ = C

        K_tilde = self.K_tilde_full_

        def objective(
            alpha: np.ndarray,
        ) -> float:
            return 0.5 * float(
                alpha @ K_tilde @ alpha
            )

        def gradient(
            alpha: np.ndarray,
        ) -> np.ndarray:
            return K_tilde @ alpha

        constraint = {
            "type": "eq",
            "fun": lambda alpha: (
                np.sum(alpha) - 1.0
            ),
            "jac": lambda alpha: (
                np.ones_like(alpha)
            ),
        }

        bounds = [
            (0.0, C)
            for _ in range(n_samples)
        ]

        # --------------------------------------------------------------
        # Build a feasible starting point.
        # --------------------------------------------------------------

        if (
            self.alpha_.size == n_samples
            and np.sum(self.alpha_) > 0
        ):
            x0 = np.clip(
                self.alpha_,
                0.0,
                C,
            )

            # Normalization can violate upper bounds.
            # Therefore only use the previous state if it is already
            # approximately feasible.
            if (
                abs(np.sum(x0) - 1.0)
                <= self.kkt_tol
                and np.max(x0)
                <= C + self.kkt_tol
            ):
                # Repair tiny sum error without blindly normalizing.
                correction = (
                    1.0 - np.sum(x0)
                )

                if abs(correction) > 0:
                    free = np.flatnonzero(
                        x0 < C - self.kkt_tol
                    )

                    if free.size > 0:
                        x0[free[0]] += correction

                if (
                    np.min(x0) >= -self.kkt_tol
                    and np.max(x0)
                    <= C + self.kkt_tol
                ):
                    pass
                else:
                    x0 = np.full(
                        n_samples,
                        1.0 / n_samples,
                    )
            else:
                x0 = np.full(
                    n_samples,
                    1.0 / n_samples,
                )
        else:
            x0 = np.full(
                n_samples,
                1.0 / n_samples,
            )

        result = minimize(
            fun=objective,
            x0=x0,
            jac=gradient,
            bounds=bounds,
            constraints=[constraint],
            method="SLSQP",
            options={
                "maxiter": self.max_iter,
                "ftol": self.tol,
                "disp": False,
            },
        )

        self.optimization_result_ = result

        if not result.success:
            raise RuntimeError(
                "ICOSVM batch QP failed: "
                f"{result.message}"
            )

        alpha = np.asarray(
            result.x,
            dtype=float,
        )

        # --------------------------------------------------------------
        # Numerical cleanup without blindly renormalizing.
        # --------------------------------------------------------------

        alpha[
            np.abs(alpha) < self.sv_tol
        ] = 0.0

        if np.min(alpha) < -self.kkt_tol:
            raise RuntimeError(
                "QP returned alpha < 0."
            )

        if np.max(alpha - C) > self.kkt_tol:
            raise RuntimeError(
                "QP returned alpha > C."
            )

        if (
            abs(np.sum(alpha) - 1.0)
            > self.kkt_tol
        ):
            raise RuntimeError(
                "QP returned an infeasible "
                "sum(alpha) constraint."
            )

        alpha = np.clip(
            alpha,
            0.0,
            C,
        )

        self.alpha_ = alpha

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
        Build the inverse sensitivity matrix:

            R =
                [ 0      1_S^T ]
                [ 1_S    K_SS  ]^{-1}

        A pseudoinverse is used only as a numerical fallback.
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
        Compute rho from the KKT conditions.

        For free support vectors:

            Ktilde_i alpha - rho = 0.

        If no free support vectors exist, use the KKT interval.
        """
        scores = (
            self.K_tilde_full_
            @ self.alpha_
        )

        alpha = self.alpha_
        C = self.C_

        free = (
            (alpha > self.sv_tol)
            & (alpha < C - self.sv_tol)
        )

        free_idx = np.flatnonzero(
            free
        )

        if free_idx.size > 0:
            rho = float(
                np.mean(
                    scores[free_idx]
                )
            )

        else:
            zero_idx = np.flatnonzero(
                alpha <= self.sv_tol
            )

            C_idx = np.flatnonzero(
                alpha >= C - self.sv_tol
            )

            lower = (
                np.max(
                    scores[C_idx]
                )
                if C_idx.size > 0
                else -np.inf
            )

            upper = (
                np.min(
                    scores[zero_idx]
                )
                if zero_idx.size > 0
                else np.inf
            )

            if (
                np.isfinite(lower)
                and np.isfinite(upper)
            ):
                rho = 0.5 * (
                    lower + upper
                )

            elif np.isfinite(lower):
                rho = float(lower)

            elif np.isfinite(upper):
                rho = float(upper)

            else:
                rho = float(
                    np.mean(scores)
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
        Initialize ICOSVM from an initial batch.

        The initial model is solved using the exact batch COSVM dual.
        """
        X_arr = check_array(X)

        n_samples, n_features = (
            X_arr.shape
        )

        if n_samples < 2:
            raise ValueError(
                "ICOSVM requires at least two "
                "initial training samples."
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
        # Buffer
        # --------------------------------------------------------------

        self.X_buffer_ = [
            X_arr[i].copy()
            for i in range(n_samples)
        ]

        self.n_samples_seen_ = n_samples

        # --------------------------------------------------------------
        # Kernel matrices
        # --------------------------------------------------------------

        self._recompute_full_matrices()

        self.C_ = 1.0 / (
            self.nu * n_samples
        )

        # --------------------------------------------------------------
        # Exact initial COSVM solution
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
    # Incremental KKT update
    # ==================================================================

    def _incremental_candidate_update(
        self,
        candidate_index: int,
    ) -> bool:
        """
        Attempt an incremental KKT update for a newly added candidate.

        Returns
        -------
        success : bool
            True if the candidate was incorporated while preserving
            numerical feasibility.

        Notes
        -----
        This is the sensitivity step. It should be regarded as the
        fast path. The complete model is later verified.
        """
        c = candidate_index

        n_samples = len(
            self.X_buffer_
        )

        C = self.C_

        # --------------------------------------------------------------
        # Current candidate starts at alpha_c = 0.
        # --------------------------------------------------------------

        alpha_old = self.alpha_.copy()
        rho_old = self.rho_

        g_c = float(
            self.K_tilde_full_[c]
            @ self.alpha_
            - self.rho_
        )

        # Candidate already satisfies the O-set KKT condition.
        if g_c >= -self.kkt_tol:
            self._update_sets()
            return True

        # --------------------------------------------------------------
        # There must be at least one free support vector to use the
        # standard sensitivity system.
        # --------------------------------------------------------------

        if not self.S_set_:
            return False

        self._build_R_matrix()

        if self.R_ is None:
            return False

        s_idx = np.asarray(
            self.S_set_,
            dtype=int,
        )

        k_sc = (
            self.K_tilde_full_[
                s_idx,
                c,
            ]
        )

        # --------------------------------------------------------------
        # Sensitivity vector.
        #
        # beta = -R [1, K_Sc]^T
        #
        # beta[0]:
        #     sensitivity of rho
        #
        # beta[1:]:
        #     sensitivities of alpha_S.
        # --------------------------------------------------------------

        rhs = np.concatenate(
            (
                np.array([1.0]),
                k_sc,
            )
        )

        beta = -(
            self.R_ @ rhs
        )

        beta_rho = float(
            beta[0]
        )

        beta_s = beta[1:]

        # --------------------------------------------------------------
        # Candidate gradient derivative.
        # --------------------------------------------------------------

        gamma = float(
            self.K_tilde_full_[c, c]
            - beta_rho
            + np.dot(
                k_sc,
                beta_s,
            )
        )

        if abs(gamma) < 1e-12:
            return False

        # We need alpha_c to increase from zero until either:
        #
        # 1. g_c reaches zero, or
        # 2. alpha_c reaches C.
        #
        delta_to_margin = (
            -g_c / gamma
        )

        if delta_to_margin <= 0:
            return False

        delta_to_C = C

        delta = min(
            delta_to_margin,
            delta_to_C,
        )

        # --------------------------------------------------------------
        # Apply candidate update.
        # --------------------------------------------------------------

        alpha_new = (
            self.alpha_.copy()
        )

        alpha_new[c] += delta

        alpha_new[s_idx] += (
            beta_s * delta
        )

        rho_new = (
            self.rho_
            + beta_rho * delta
        )

        # --------------------------------------------------------------
        # Feasibility checks.
        # --------------------------------------------------------------

        if (
            np.min(alpha_new)
            < -self.kkt_tol
        ):
            self.alpha_ = alpha_old
            self.rho_ = rho_old
            return False

        if (
            np.max(
                alpha_new - C
            )
            > self.kkt_tol
        ):
            self.alpha_ = alpha_old
            self.rho_ = rho_old
            return False

        if (
            abs(
                np.sum(alpha_new)
                - 1.0
            )
            > self.kkt_tol
        ):
            self.alpha_ = alpha_old
            self.rho_ = rho_old
            return False

        # --------------------------------------------------------------
        # Commit.
        # --------------------------------------------------------------

        self.alpha_ = np.clip(
            alpha_new,
            0.0,
            C,
        )

        self.rho_ = float(
            rho_new
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

            g_i = Ktilde_i alpha - rho

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
        Re-solve the current complete COSVM QP.

        This is the numerical correction path used when the incremental
        sensitivity update does not satisfy the complete KKT system.
        """
        self._solve_batch_qp()

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
        Incrementally incorporate one new observation.

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
        # First observation.
        # --------------------------------------------------------------

        if not self.X_buffer_:
            self.fit(
                x_vec.reshape(
                    1,
                    -1,
                )
            )
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
        # Recompute the complete covariance-guided kernel.
        #
        # This is required because H changes from N to N+1.
        # --------------------------------------------------------------

        self._recompute_full_matrices()

        # --------------------------------------------------------------
        # Extend alpha with candidate alpha_c = 0.
        # --------------------------------------------------------------

        self.alpha_ = np.append(
            self.alpha_,
            0.0,
        )

        # --------------------------------------------------------------
        # The previous alpha vector may no longer satisfy the new
        # C because C decreases as N grows.
        #
        # If any old coefficient is now above C, the incremental
        # sensitivity path is invalid.
        # --------------------------------------------------------------

        if np.max(
            self.alpha_ - self.C_
        ) > self.kkt_tol:
            self._correct_with_batch_qp()
            self._sync_support_vectors()
            self._enforce_max_samples()
            return self

        # --------------------------------------------------------------
        # Attempt sensitivity update.
        # --------------------------------------------------------------

        success = (
            self._incremental_candidate_update(
                c
            )
        )

        # --------------------------------------------------------------
        # Always verify the complete KKT state.
        # --------------------------------------------------------------

        if not success or not self._check_kkt():
            self._correct_with_batch_qp()

        else:
            self._compute_rho()
            self._build_R_matrix()

            # The rho recomputation is deliberately performed after
            # the sensitivity update to eliminate accumulated
            # numerical drift.

            if not self._check_kkt():
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
        X_arr = check_array(X)

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

        self.n_samples_seen_ = max(
            0,
            self.n_samples_seen_ - 1,
        )

        if not self.X_buffer_:
            self.alpha_ = np.empty(
                0,
                dtype=float,
            )

            self.S_set_ = []
            self.E_set_ = []
            self.O_set_ = []

            self.K_full_ = None
            self.K_tilde_full_ = None

            self.R_ = None

            self.rho_ = 0.0
            self.offset_ = 0.0
            self.intercept_ = 0.0

            self._sync_support_vectors()

            return self

        # --------------------------------------------------------------
        # Recompute complete matrices.
        # --------------------------------------------------------------

        self.alpha_ = np.delete(
            self.alpha_,
            index,
        )

        self._recompute_full_matrices()

        n_samples = len(
            self.X_buffer_
        )

        self.C_ = 1.0 / (
            self.nu * n_samples
        )

        # --------------------------------------------------------------
        # Decremental covariance changes globally, so solve/correct
        # the complete QP.
        # --------------------------------------------------------------

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
    # Test-time modified kernel
    # ==================================================================

    def _transform_test_kernel(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Compute

            Ktilde_test
                = eta K_test
                + (1 - eta) K_test H K^T.
        """
        check_is_fitted(self)

        X_arr = check_array(X)

        X_train = np.asarray(
            self.X_buffer_,
            dtype=float,
        )

        K_test = np.asarray(
            self.kernel_(
                X_arr,
                X_train,
            ),
            dtype=float,
        )

        n_samples = (
            X_train.shape[0]
        )

        if (
            self.eta == 1.0
            or n_samples <= 1
        ):
            return K_test

        # K_test H
        K_test_centered = (
            K_test
            - np.mean(
                K_test,
                axis=1,
                keepdims=True,
            )
        )

        # K_test H K_train^T
        covariance_component = (
            K_test_centered
            @ self.K_full_.T
        )

        return (
            self.eta * K_test
            + (1.0 - self.eta)
            * covariance_component
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

            score(x)
                = sum_i alpha_i Ktilde(x, x_i).
        """
        check_is_fitted(self)

        K_tilde_test = (
            self._transform_test_kernel(X)
        )

        return (
            K_tilde_test
            @ self.alpha_
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

        Useful for validating the incremental algorithm.

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
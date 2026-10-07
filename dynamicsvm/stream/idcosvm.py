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

4. Adaptive memory management with FIFO, non-SV-first, error-first,
   and Mahalanobis pruning strategies.

Important
---------
This implementation uses KKT-aware warm starts and exact constrained
re-optimization of the current finite problem. It does NOT claim to be
the exact piecewise-linear adiabatic algorithm of Cauwenberghs & Poggio.

Mathematical formulation
------------------------
For N active samples:

    min_alpha  1/2 alpha^T K_tilde alpha

subject to:

    sum(alpha_i) = 1
    0 <= alpha_i <= C

where:

    C = 1 / (nu * N)

and the covariance-guided kernel is constructed from the centered
training Gram matrix:

    K_c = H K H

    H = I - (1/N) 11^T

    K_tilde = eta K + (1 - eta) K_c

The decision function is:

    f(x) = k_tilde(x, X)^T alpha - rho

where rho is estimated from the KKT conditions.
"""

from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from dynamicsvm.base import BaseSVM
from dynamicsvm.kernels import Kernel, get_kernel
from dynamicsvm.utils.validation import check_array, check_is_fitted


PruningStrategy = Literal[
    "fifo",
    "non_sv_first",
    "error_first",
    "mahalanobis",
]


class IncrementalDecrementalCOSVM(BaseSVM):
    """
    Incremental and Decremental Covariance-guided One-Class SVM.

    Parameters
    ----------
    nu : float, default=0.1
        One-class SVM parameter in (0, 1].

    eta : float, default=0.5
        Trade-off between the original kernel and covariance-guided
        centered kernel:

            K_tilde = eta K + (1 - eta) K_c

        where K_c = H K H.

    kernel : {'rbf', 'linear', 'poly', 'sigmoid'} or Kernel or Callable
        Kernel specification.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient.

    degree : int, default=3
        Polynomial kernel degree.

    coef0 : float, default=0.0
        Polynomial/sigmoid independent term.

    max_samples : int or None, default=None
        Maximum number of active samples.

    pruning_strategy : {'fifo', 'non_sv_first', 'error_first',
                         'mahalanobis'}, default='fifo'
        Automatic memory pruning strategy.

    tol : float, default=1e-5
        Numerical tolerance.

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

        # -------------------------------------------------------------
        # Dynamic state
        # -------------------------------------------------------------

        self.kernel_: Optional[Kernel] = None

        self.X_buffer_: List[np.ndarray] = []
        self.timestamps_: List[int] = []

        self.alpha_: np.ndarray = np.empty(0, dtype=float)

        self.offset_: float = 0.0
        self.intercept_: float = 0.0

        self.R_: Optional[np.ndarray] = None

        self.S_set_: List[int] = []
        self.E_set_: List[int] = []
        self.O_set_: List[int] = []

        self.K_full_: Optional[np.ndarray] = None
        self.K_tilde_full_: Optional[np.ndarray] = None

        # Centering statistics used by the covariance kernel.
        self.K_train_mean_: Optional[np.ndarray] = None
        self.K_global_mean_: Optional[float] = None

        self.n_samples_seen_: int = 0
        self.n_unlearned_: int = 0

    # =================================================================
    # Utility methods
    # =================================================================

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        """
        Validate a two-dimensional feature matrix.
        """
        X_arr = check_array(X)

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

        self.R_ = None

        self.S_set_ = []
        self.E_set_ = []
        self.O_set_ = []

        self.K_full_ = None
        self.K_tilde_full_ = None

        self.K_train_mean_ = None
        self.K_global_mean_ = None

        self.support_ = None
        self.support_vectors_ = None
        self.dual_coef_ = None
        self.n_support_ = np.array([0])

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
        Construct the covariance-guided training kernel.

        We use the centered Gram matrix:

            K_c = H K H

        and:

            K_tilde = eta K + (1 - eta) K_c

        The final matrix is symmetrized numerically.
        """
        K = np.asarray(K, dtype=float)

        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError(
                "Training kernel matrix must be square."
            )

        n_samples = K.shape[0]

        if n_samples == 0:
            return K.copy()

        if n_samples == 1:
            return K.copy()

        H = self._centering_matrix(n_samples)

        K_centered = H @ K @ H

        K_tilde = (
            self.eta * K
            + (1.0 - self.eta) * K_centered
        )

        return 0.5 * (K_tilde + K_tilde.T)

    def _compute_cross_covariance_kernel(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Construct the covariance-guided cross-kernel between test
        samples and the active training samples.

        Given:

            K_test = k(X_test, X_train)

        the corresponding centered cross-kernel is:

            K_test,c =
                K_test
                - row_mean(K_test)
                - training_column_mean
                + global_training_mean

        This is the rectangular analogue of:

            H K H
        """
        if self.kernel_ is None:
            raise RuntimeError("Kernel has not been initialized.")

        if self.K_full_ is None:
            raise RuntimeError(
                "Training kernel matrix is unavailable."
            )

        X_train = np.asarray(self.X_buffer_)

        K_test = np.asarray(
            self.kernel_(X_test, X_train),
            dtype=float,
        )

        if self.eta >= 1.0 or X_train.shape[0] <= 1:
            return K_test

        # Mean over training samples for every test point.
        test_row_mean = np.mean(
            K_test,
            axis=1,
            keepdims=True,
        )

        # Mean over test? No.
        # We need the mean of each training kernel column.
        train_column_mean = np.mean(
            self.K_full_,
            axis=0,
            keepdims=True,
        )

        global_mean = float(
            np.mean(self.K_full_)
        )

        K_centered = (
            K_test
            - test_row_mean
            - train_column_mean
            + global_mean
        )

        return (
            self.eta * K_test
            + (1.0 - self.eta) * K_centered
        )

    # =================================================================
    # Kernel matrix management
    # =================================================================

    def _recompute_full_matrices(self) -> None:
        """
        Recompute the complete active-buffer kernel matrices.

        This is intentionally explicit and robust. It can later be
        replaced by rank-one/rank-two updates for a more optimized
        streaming implementation.
        """
        if not self.X_buffer_:
            self.K_full_ = None
            self.K_tilde_full_ = None
            self.K_train_mean_ = None
            self.K_global_mean_ = None
            return

        X_mat = np.asarray(self.X_buffer_)

        K = np.asarray(
            self.kernel_(X_mat, X_mat),
            dtype=float,
        )

        K = 0.5 * (K + K.T)

        self.K_full_ = K

        self.K_tilde_full_ = (
            self._compute_covariance_kernel_matrix(K)
        )

        self.K_train_mean_ = np.mean(
            K,
            axis=0,
        )

        self.K_global_mean_ = float(
            np.mean(K)
        )

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
        """
        n_samples = len(self.X_buffer_)

        self.S_set_ = []
        self.E_set_ = []
        self.O_set_ = []

        if n_samples == 0:
            return

        C = self._current_C()

        lower_tol = self.tol
        upper_tol = max(self.tol, 10.0 * np.finfo(float).eps)

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

            K_tilde_i alpha = rho

        If no strict margin vector exists, derive an admissible rho
        interval from the lower/upper KKT bounds.
        """
        n_samples = len(self.X_buffer_)

        if n_samples == 0:
            return 0.0

        scores = self.K_tilde_full_ @ self.alpha_

        C = self._current_C()

        # -------------------------------------------------------------
        # Best case: strict margin support vectors.
        # -------------------------------------------------------------
        margin_indices = [
            i
            for i in range(n_samples)
            if (
                self.alpha_[i] > self.tol
                and self.alpha_[i] < C - self.tol
            )
        ]

        if margin_indices:
            return float(
                np.mean(scores[margin_indices])
            )

        # -------------------------------------------------------------
        # No strict margin vectors.
        #
        # For the minimization convention:
        #
        # alpha_i = 0  -> score_i >= rho
        # alpha_i = C  -> score_i <= rho
        #
        # Therefore:
        #
        # max_{alpha=C} score_i <= rho
        # <= min_{alpha=0} score_i
        # -------------------------------------------------------------

        lower_candidates = []
        upper_candidates = []

        for i in range(n_samples):
            if self.alpha_[i] >= C - self.tol:
                lower_candidates.append(scores[i])

            elif self.alpha_[i] <= self.tol:
                upper_candidates.append(scores[i])

        if lower_candidates and upper_candidates:
            lower = float(np.max(lower_candidates))
            upper = float(np.min(upper_candidates))

            # If numerical errors produce an inverted interval,
            # use the midpoint anyway.
            return 0.5 * (lower + upper)

        if lower_candidates:
            return float(np.max(lower_candidates))

        if upper_candidates:
            return float(np.min(upper_candidates))

        return float(np.mean(scores))

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

        self._init_R_matrix()
        self._sync_support_vectors()

    def _init_R_matrix(self) -> None:
        """
        Construct the KKT sensitivity matrix for the current margin
        support vectors:

            R =
                [ 0      1^T ]
                [ 1      K_SS ]

        and store its inverse/pseudoinverse.

        This matrix is useful for KKT sensitivity calculations, but
        the current implementation does not pretend that using R alone
        constitutes the complete adiabatic path algorithm.
        """
        if (
            self.K_tilde_full_ is None
            or not self.S_set_
        ):
            self.R_ = None
            return

        s_idx = np.asarray(
            self.S_set_,
            dtype=int,
        )

        n_s = len(s_idx)

        M = np.zeros(
            (n_s + 1, n_s + 1),
            dtype=float,
        )

        M[0, 1:] = 1.0
        M[1:, 0] = 1.0

        M[1:, 1:] = self.K_tilde_full_[
            np.ix_(s_idx, s_idx)
        ]

        # Symmetrize numerical noise.
        M = 0.5 * (M + M.T)

        try:
            cond = np.linalg.cond(M)

            if np.isfinite(cond) and cond < 1e12:
                self.R_ = np.linalg.inv(M)
            else:
                self.R_ = np.linalg.pinv(
                    M,
                    rcond=1e-10,
                )

        except np.linalg.LinAlgError:
            self.R_ = np.linalg.pinv(
                M,
                rcond=1e-10,
            )

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
        """
        n_samples = len(self.X_buffer_)

        if n_samples == 0:
            return np.empty(0)

        C = self._current_C()

        if previous_alpha is not None:
            previous_alpha = np.asarray(
                previous_alpha,
                dtype=float,
            )

            if (
                previous_alpha.shape == (n_samples,)
                and np.all(np.isfinite(previous_alpha))
            ):
                x0 = np.clip(
                    previous_alpha,
                    0.0,
                    C,
                )

                total = float(np.sum(x0))

                if abs(total - 1.0) <= self.tol:
                    return x0

        # Uniform distribution is feasible only if:
        #
        # 1/N <= 1/(nu N)
        #
        # which is equivalent to nu <= 1.
        #
        # Therefore it is always feasible under our parameter range.
        return np.full(
            n_samples,
            1.0 / n_samples,
            dtype=float,
        )

    def _solve_dual(
        self,
        previous_alpha: Optional[np.ndarray] = None,
    ) -> None:
        """
        Solve the current finite COSVM dual problem.

        Objective:

            1/2 alpha^T K_tilde alpha

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

        K_tilde = self.K_tilde_full_

        def objective(alpha: np.ndarray) -> float:
            return 0.5 * float(
                alpha @ K_tilde @ alpha
            )

        def gradient(alpha: np.ndarray) -> np.ndarray:
            return K_tilde @ alpha

        constraints = [
            {
                "type": "eq",
                "fun": lambda alpha: (
                    np.sum(alpha) - 1.0
                ),
                "jac": lambda alpha: (
                    np.ones_like(alpha)
                ),
            }
        ]

        bounds = [
            (0.0, C)
            for _ in range(n_samples)
        ]

        x0 = self._initial_feasible_alpha(
            previous_alpha
        )

        result = minimize(
            objective,
            x0,
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={
                "maxiter": 500,
                "ftol": min(
                    self.tol,
                    1e-9,
                ),
                "disp": False,
            },
        )

        if not result.success:
            # Retry from a uniform feasible point.
            uniform = np.full(
                n_samples,
                1.0 / n_samples,
            )

            result = minimize(
                objective,
                uniform,
                jac=gradient,
                bounds=bounds,
                constraints=constraints,
                method="SLSQP",
                options={
                    "maxiter": 1000,
                    "ftol": min(
                        self.tol,
                        1e-9,
                    ),
                    "disp": False,
                },
            )

        if not result.success:
            raise RuntimeError(
                "COSVM optimization failed: "
                f"{result.message}"
            )

        alpha = np.asarray(
            result.x,
            dtype=float,
        )

        # -------------------------------------------------------------
        # Numerical cleanup.
        #
        # Do NOT blindly normalize alpha after optimization because
        # that could violate the upper bounds. Instead, clip only tiny
        # numerical violations and verify feasibility.
        # -------------------------------------------------------------

        alpha[
            np.abs(alpha) <= self.tol * 0.1
        ] = 0.0

        alpha[
            np.abs(alpha - C) <= self.tol * 0.1
        ] = C

        if np.any(alpha < -self.tol):
            raise RuntimeError(
                "Optimizer returned negative dual coefficients."
            )

        if np.any(alpha > C + self.tol):
            raise RuntimeError(
                "Optimizer returned dual coefficients above C."
            )

        sum_alpha = float(
            np.sum(alpha)
        )

        if abs(sum_alpha - 1.0) > 10.0 * self.tol:
            raise RuntimeError(
                "Optimizer returned an infeasible solution: "
                f"sum(alpha)={sum_alpha:.12g}"
            )

        # Tiny residual caused by floating-point arithmetic.
        if abs(sum_alpha - 1.0) > 0.0:
            alpha *= 1.0 / sum_alpha

        self.alpha_ = alpha

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
        # First observation.
        # -------------------------------------------------------------
        if not self.X_buffer_:
            return self.fit(
                x_vec.reshape(1, -1)
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

    def learn_batch(
        self,
        X: np.ndarray,
    ) -> "IncrementalDecrementalCOSVM":
        """
        Incrementally learn a sequence of observations.
        """
        X_arr = self._validate_X(X)

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

            score(x) = k_tilde(x, X)^T alpha
        """
        check_is_fitted(self)

        if not self.X_buffer_:
            raise RuntimeError(
                "Cannot score samples with an empty model."
            )

        X_arr = self._validate_X(X)

        K_tilde_test = (
            self._compute_cross_covariance_kernel(
                X_arr
            )
        )

        return K_tilde_test @ self.alpha_

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

            score_i = (K_tilde alpha)_i

        and decision convention:

            score_i - rho

        we expect approximately:

            alpha_i = 0:
                score_i - rho >= 0

            0 < alpha_i < C:
                score_i - rho = 0

            alpha_i = C:
                score_i - rho <= 0
        """
        if (
            not self.X_buffer_
            or self.K_tilde_full_ is None
        ):
            return {
                "max_stationarity_error": 0.0,
                "min_lower_bound_margin": 0.0,
                "max_upper_bound_violation": 0.0,
                "sum_alpha_error": 0.0,
            }

        C = self._current_C()

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
                alpha_i > self.tol
                and alpha_i < C - self.tol
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
            if alpha_i <= self.tol:
                lower_violation = max(
                    lower_violation,
                    -residual[i],
                )

        upper_violation = 0.0

        for i, alpha_i in enumerate(
            self.alpha_
        ):
            if alpha_i >= C - self.tol:
                upper_violation = max(
                    upper_violation,
                    residual[i],
                )

        return {
            "max_stationarity_error": float(
                max_stationarity_error
            ),
            "max_lower_kkt_violation": float(
                max(0.0, lower_violation)
            ),
            "max_upper_kkt_violation": float(
                max(0.0, upper_violation)
            ),
            "sum_alpha_error": float(
                abs(
                    np.sum(self.alpha_)
                    - 1.0
                )
            ),
        }

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
        }

    def get_kkt_diagnostics(
        self,
    ) -> Dict[str, float]:
        """
        Public KKT diagnostic method.
        """
        check_is_fitted(self)

        return self._kkt_residuals()


# ---------------------------------------------------------------------
# Short alias
# ---------------------------------------------------------------------

IDCOSVM = IncrementalDecrementalCOSVM
"""
Covariance-guided One-Class Support Vector Machine (COSVM).

Based on the COSVM formulation introduced in:

    Naimul Mefraz Khan, Riadh Ksantini, Shafiq Ahmad, and
    Nabil Guan,
    "Covariance-guided One-Class Support Vector Machine",
    Pattern Recognition, 47(6), 2165-2177, 2014.

The implementation also follows the covariance-guided / low-variance
direction motivation developed further in:

    Takoua Kefi-Fatteh, Riadh Ksantini, Mohamed-Becha Kaaniche,
    Adel Bouhoula,
    "A novel incremental one-class support vector machine based
    on low variance direction",
    Pattern Recognition, 91, 308-321, 2019.

Mathematical formulation
------------------------

Let K be the kernel Gram matrix:

    K_ij = k(x_i, x_j)

and let

    H = I - (1/N) 11^T

be the centering matrix.

The covariance-guided kernel matrix is

    K_tilde = eta K + (1 - eta) K H K^T

with

    0 <= eta <= 1.

The COSVM dual is solved as:

    minimize
        1/2 alpha^T K_tilde alpha

    subject to
        0 <= alpha_i <= C
        sum(alpha_i) = 1

where

    C = 1 / (nu N).

The decision function is

    f(x) = sum_i alpha_i K_tilde(x, x_i) - rho.

Positive values correspond to normal samples and negative values
correspond to anomalies.
"""

from typing import Callable, Optional, Union

import numpy as np
from scipy.optimize import minimize

from dynamicsvm.base import BaseSVM
from dynamicsvm.kernels import Kernel, get_kernel
from dynamicsvm.utils.validation import check_array, check_is_fitted


class COSVM(BaseSVM):
    """
    Covariance-guided One-Class Support Vector Machine.

    Parameters
    ----------
    nu : float, default=0.1
        Controls the upper bound on the fraction of training errors
        and the lower bound on the fraction of support vectors.

        Must satisfy:

            0 < nu <= 1

    eta : float, default=0.5
        Trade-off between the standard kernel geometry and the
        covariance-guided component.

            eta = 1:
                Standard One-Class SVM kernel.

            eta = 0:
                Pure covariance-guided component.

            0 < eta < 1:
                Combination of both.

    kernel : {'rbf', 'linear', 'poly', 'sigmoid'} or Kernel or Callable
        Kernel function.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for RBF, polynomial and sigmoid kernels.

    degree : int, default=3
        Degree of the polynomial kernel.

    coef0 : float, default=0.0
        Independent term in polynomial and sigmoid kernels.

    tol : float, default=1e-5
        Optimization tolerance.

    max_iter : int, default=1000
        Maximum number of SLSQP iterations.

    psd_tol : float, default=1e-10
        Numerical tolerance used when checking the modified Gram
        matrix for positive semidefiniteness.

    sv_tol : float, default=1e-7
        Tolerance for identifying support vectors.

    Attributes
    ----------
    alpha_ : ndarray
        Complete vector of dual coefficients.

    support_ : ndarray
        Indices of support vectors.

    support_vectors_ : ndarray
        Training support vectors.

    dual_coef_ : ndarray
        Non-zero dual coefficients.

    rho_ : float
        Decision-function offset.

    offset_ : float
        Alias for rho_.

    intercept_ : float
        Negative rho, following the common SVM convention.

    K_train_ : ndarray
        Original kernel Gram matrix.

    K_tilde_ : ndarray
        Covariance-guided Gram matrix.

    C_ : float
        Upper bound on each alpha_i.
    """

    def __init__(
        self,
        nu: float = 0.1,
        eta: float = 0.5,
        kernel: Union[str, Kernel, Callable] = "rbf",
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        tol: float = 1e-5,
        max_iter: int = 1000,
        psd_tol: float = 1e-10,
        sv_tol: float = 1e-7,
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

        if psd_tol < 0:
            raise ValueError(
                f"psd_tol must be >= 0, got {psd_tol}"
            )

        if sv_tol <= 0:
            raise ValueError(
                f"sv_tol must be > 0, got {sv_tol}"
            )

        self.nu = float(nu)
        self.eta = float(eta)

        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)

        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self.psd_tol = float(psd_tol)
        self.sv_tol = float(sv_tol)

        # Fitted attributes
        self.kernel_: Optional[Kernel] = None

        self.X_train_: Optional[np.ndarray] = None

        self.K_train_: Optional[np.ndarray] = None
        self.K_tilde_: Optional[np.ndarray] = None

        self.alpha_: Optional[np.ndarray] = None

        self.support_: Optional[np.ndarray] = None
        self.support_vectors_: Optional[np.ndarray] = None
        self.dual_coef_: Optional[np.ndarray] = None

        self.C_: float = 1.0

        self.rho_: float = 0.0
        self.offset_: float = 0.0
        self.intercept_: float = 0.0

        self.n_support_: Optional[np.ndarray] = None

        self.optimization_result_ = None

    # ------------------------------------------------------------------
    # Kernel / covariance construction
    # ------------------------------------------------------------------

    @staticmethod
    def _centering_matrix(n_samples: int) -> np.ndarray:
        """
        Construct

            H = I - (1/N) 11^T.
        """
        if n_samples <= 0:
            raise ValueError(
                "n_samples must be strictly positive."
            )

        return (
            np.eye(n_samples)
            - np.ones((n_samples, n_samples)) / n_samples
        )

    def _compute_covariance_kernel_matrix(
        self,
        K: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the COSVM modified Gram matrix:

            K_tilde = eta K + (1 - eta) K H K^T

        where

            H = I - (1/N) 11^T.

        Since K is symmetric for a standard kernel Gram matrix,
        K^T = K, but K^T is kept explicitly for mathematical clarity.
        """
        K = np.asarray(K, dtype=float)

        if K.ndim != 2:
            raise ValueError(
                "K must be a two-dimensional matrix."
            )

        n_samples, n_features = K.shape

        if n_samples != n_features:
            raise ValueError(
                "Training Gram matrix must be square."
            )

        if n_samples == 1:
            return K.copy()

        # H = I - 11^T / N
        H = self._centering_matrix(n_samples)

        # Covariance-guided component:
        #
        #     K H K^T
        #
        covariance_component = K @ H @ K.T

        K_tilde = (
            self.eta * K
            + (1.0 - self.eta) * covariance_component
        )

        # Numerical symmetrization.
        K_tilde = 0.5 * (
            K_tilde + K_tilde.T
        )

        return K_tilde

    def _stabilize_psd_matrix(
        self,
        K: np.ndarray,
    ) -> np.ndarray:
        """
        Numerically stabilize a theoretically PSD matrix.

        Small negative eigenvalues can appear because of floating-point
        arithmetic. They are clipped to zero if they are within psd_tol.

        A genuinely negative eigenvalue larger than the tolerance is
        considered an error.
        """
        K = 0.5 * (K + K.T)

        eigenvalues = np.linalg.eigvalsh(K)
        min_eigenvalue = float(np.min(eigenvalues))

        if min_eigenvalue < -self.psd_tol:
            raise ValueError(
                "The COSVM modified kernel matrix is not positive "
                "semidefinite. "
                f"Minimum eigenvalue: {min_eigenvalue:.3e}"
            )

        if min_eigenvalue < 0.0:
            # Numerical correction only.
            K = K - min_eigenvalue * np.eye(K.shape[0])

        return K

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> "COSVM":
        """
        Fit COSVM on normal/target training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : ignored
            Present only for API compatibility.

        Returns
        -------
        self
            Fitted COSVM estimator.
        """
        X_arr = check_array(X)

        n_samples, n_features = X_arr.shape

        if n_samples < 2:
            raise ValueError(
                "COSVM requires at least two training samples."
            )

        self.X_train_ = X_arr.copy()

        # --------------------------------------------------------------
        # Kernel
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
        # nu constraint
        # --------------------------------------------------------------

        self.C_ = 1.0 / (
            self.nu * n_samples
        )

        # --------------------------------------------------------------
        # Original kernel matrix
        # --------------------------------------------------------------

        self.K_train_ = np.asarray(
            self.kernel_(X_arr, X_arr),
            dtype=float,
        )

        self.K_train_ = 0.5 * (
            self.K_train_ + self.K_train_.T
        )

        # --------------------------------------------------------------
        # COSVM modified kernel
        # --------------------------------------------------------------

        self.K_tilde_ = (
            self._compute_covariance_kernel_matrix(
                self.K_train_
            )
        )

        self.K_tilde_ = (
            self._stabilize_psd_matrix(
                self.K_tilde_
            )
        )

        # --------------------------------------------------------------
        # Dual optimization
        #
        # min 1/2 alpha^T K_tilde alpha
        #
        # s.t.
        #       0 <= alpha_i <= C
        #       sum(alpha_i) = 1
        # --------------------------------------------------------------

        def objective(alpha: np.ndarray) -> float:
            return 0.5 * float(
                alpha @ self.K_tilde_ @ alpha
            )

        def gradient(alpha: np.ndarray) -> np.ndarray:
            return self.K_tilde_ @ alpha

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
            (0.0, self.C_)
            for _ in range(n_samples)
        ]

        # Uniform solution:
        #
        # alpha_i = 1/N
        #
        # This is feasible because:
        #
        # 1/N <= 1/(nu N)
        #
        # for nu <= 1.
        x0 = np.full(
            n_samples,
            1.0 / n_samples,
            dtype=float,
        )

        result = minimize(
            fun=objective,
            x0=x0,
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={
                "maxiter": self.max_iter,
                "ftol": self.tol,
                "disp": False,
            },
        )

        self.optimization_result_ = result

        # --------------------------------------------------------------
        # Never silently use an unsuccessful optimization result.
        # --------------------------------------------------------------

        if not result.success:
            raise RuntimeError(
                "COSVM optimization failed: "
                f"{result.message}"
            )

        alpha = np.asarray(
            result.x,
            dtype=float,
        )

        # --------------------------------------------------------------
        # Numerical cleanup
        # --------------------------------------------------------------

        alpha[np.abs(alpha) < self.sv_tol] = 0.0

        # Check feasibility BEFORE doing anything else.
        sum_error = abs(
            np.sum(alpha) - 1.0
        )

        lower_violation = np.min(alpha)

        upper_violation = np.max(
            alpha - self.C_
        )

        feasibility_tol = max(
            self.tol * 10.0,
            1e-8,
        )

        if sum_error > feasibility_tol:
            raise RuntimeError(
                "COSVM optimizer returned an infeasible solution: "
                f"|sum(alpha)-1| = {sum_error:.3e}"
            )

        if lower_violation < -feasibility_tol:
            raise RuntimeError(
                "COSVM optimizer returned alpha < 0: "
                f"minimum alpha = {lower_violation:.3e}"
            )

        if upper_violation > feasibility_tol:
            raise RuntimeError(
                "COSVM optimizer returned alpha > C: "
                f"maximum violation = {upper_violation:.3e}"
            )

        # Tiny numerical violations are clipped only after feasibility
        # has been checked.
        alpha = np.clip(
            alpha,
            0.0,
            self.C_,
        )

        self.alpha_ = alpha

        # --------------------------------------------------------------
        # Support vectors
        # --------------------------------------------------------------

        sv_mask = (
            self.alpha_ > self.sv_tol
        )

        self.support_ = np.flatnonzero(
            sv_mask
        )

        self.support_vectors_ = (
            X_arr[self.support_]
        )

        self.dual_coef_ = (
            self.alpha_[self.support_]
        )

        self.n_support_ = np.array(
            [len(self.support_)],
            dtype=int,
        )

        # --------------------------------------------------------------
        # Estimate rho using free support vectors
        #
        # For 0 < alpha_i < C:
        #
        #     sum_j alpha_j Ktilde_ij - rho = 0
        #
        # therefore:
        #
        #     rho = sum_j alpha_j Ktilde_ij.
        # --------------------------------------------------------------

        self._compute_rho()

        return self

    # ------------------------------------------------------------------
    # Rho / threshold
    # ------------------------------------------------------------------

    def _compute_rho(self) -> None:
        """
        Compute the decision-function offset rho.

        Prefer free support vectors:

            0 < alpha_i < C.

        If there are no free support vectors, use the KKT interval
        implied by the lower- and upper-bound coefficients.
        """
        check_is_fitted(self)

        alpha = self.alpha_
        K_tilde = self.K_tilde_

        scores_train = (
            K_tilde @ alpha
        )

        alpha_tol = max(
            self.sv_tol,
            self.tol * 10.0,
        )

        # Free support vectors:
        #
        #     0 < alpha_i < C
        #
        free_mask = (
            (alpha > alpha_tol)
            & (alpha < self.C_ - alpha_tol)
        )

        free_indices = np.flatnonzero(
            free_mask
        )

        if free_indices.size > 0:
            rho = float(
                np.mean(
                    scores_train[
                        free_indices
                    ]
                )
            )

        else:
            # ----------------------------------------------------------
            # No free support vectors.
            #
            # KKT conditions imply:
            #
            # alpha_i = 0  => score_i >= rho
            # alpha_i = C  => score_i <= rho
            #
            # Therefore rho lies between:
            #
            # max(score_i : alpha_i = C)
            #
            # and
            #
            # min(score_i : alpha_i = 0).
            # ----------------------------------------------------------

            upper_bound_mask = (
                alpha <= alpha_tol
            )

            lower_bound_mask = (
                alpha >= self.C_ - alpha_tol
            )

            upper_values = scores_train[
                upper_bound_mask
            ]

            lower_values = scores_train[
                lower_bound_mask
            ]

            if (
                lower_values.size > 0
                and upper_values.size > 0
            ):
                rho_lower = float(
                    np.max(lower_values)
                )

                rho_upper = float(
                    np.min(upper_values)
                )

                rho = 0.5 * (
                    rho_lower + rho_upper
                )

            elif lower_values.size > 0:
                rho = float(
                    np.mean(lower_values)
                )

            elif upper_values.size > 0:
                rho = float(
                    np.mean(upper_values)
                )

            else:
                # Extremely unlikely fallback.
                rho = float(
                    np.mean(scores_train)
                )

        self.rho_ = rho

        # Keep compatibility with the original implementation.
        self.offset_ = rho

        # SVM-style intercept convention.
        self.intercept_ = -rho

    # ------------------------------------------------------------------
    # Modified test kernel
    # ------------------------------------------------------------------

    def _transform_test_kernel(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the covariance-guided kernel between test samples
        and training samples.

        For X_test:

            K_test = k(X_test, X_train)

        the modified kernel is:

            Ktilde_test
                = eta K_test
                + (1 - eta) K_test H K_train^T

        where

            H = I - (1/N)11^T.
        """
        check_is_fitted(self)

        X_test_arr = check_array(
            X_test
        )

        K_test = np.asarray(
            self.kernel_(
                X_test_arr,
                self.X_train_,
            ),
            dtype=float,
        )

        # Standard OCSVM case.
        if self.eta == 1.0:
            return K_test

        n_samples = (
            self.X_train_.shape[0]
        )

        if n_samples <= 1:
            return K_test

        # --------------------------------------------------------------
        # K_test H
        #
        # H = I - 11^T/N
        #
        # Therefore:
        #
        # K_test H
        # = K_test - row_mean(K_test)
        # --------------------------------------------------------------

        K_test_centered = (
            K_test
            - np.mean(
                K_test,
                axis=1,
                keepdims=True,
            )
        )

        # --------------------------------------------------------------
        # K_test H K_train^T
        # --------------------------------------------------------------

        covariance_component = (
            K_test_centered
            @ self.K_train_.T
        )

        K_tilde_test = (
            self.eta * K_test
            + (1.0 - self.eta)
            * covariance_component
        )

        return K_tilde_test

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_samples(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the raw COSVM score before subtracting rho.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Raw score:

                sum_i alpha_i Ktilde(x, x_i)
        """
        check_is_fitted(self)

        X_arr = check_array(X)

        K_tilde_test = (
            self._transform_test_kernel(
                X_arr
            )
        )

        return (
            K_tilde_test @ self.alpha_
        )

    def decision_function(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the signed COSVM decision function.

            f(x)
              = sum_i alpha_i Ktilde(x, x_i)
                - rho

        Positive values indicate normal samples.

        Negative values indicate anomalies.
        """
        return (
            self.score_samples(X)
            - self.rho_
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict normal/anomalous labels.

        Returns
        -------
        y_pred : ndarray
            +1 for normal samples.

            -1 for anomalous samples.
        """
        decision = (
            self.decision_function(X)
        )

        return np.where(
            decision >= 0.0,
            1,
            -1,
        )

    # ------------------------------------------------------------------
    # Eta optimization
    # ------------------------------------------------------------------

    def optimize_eta(
        self,
        X: np.ndarray,
        target_fol: Optional[float] = None,
        eta_step: float = 0.05,
    ) -> float:
        """
        Search for eta using the training outlier fraction.

        The search starts from eta=1 and moves toward eta=0.

        Parameters
        ----------
        X : array-like
            Training data.

        target_fol : float or None, default=None
            Desired training outlier fraction.

            If None, nu is used.

        eta_step : float, default=0.05
            Decrement applied to eta.

        Returns
        -------
        best_eta : float
            Selected eta.

        Notes
        -----
        This is a practical eta-selection routine. It should not be
        interpreted as a literal reproduction of every experimental
        detail in the original COSVM paper.
        """
        X_arr = check_array(X)

        if not (0.0 < eta_step <= 1.0):
            raise ValueError(
                "eta_step must be in (0, 1]."
            )

        if target_fol is None:
            target_fol = self.nu

        if not (0.0 <= target_fol <= 1.0):
            raise ValueError(
                "target_fol must be in [0, 1]."
            )

        original_eta = self.eta

        best_eta = 1.0
        best_distance = np.inf

        eta_values = np.arange(
            1.0,
            -eta_step / 2.0,
            -eta_step,
        )

        for eta in eta_values:
            eta = float(
                np.clip(
                    eta,
                    0.0,
                    1.0,
                )
            )

            self.eta = eta

            try:
                self.fit(X_arr)
            except RuntimeError:
                # Restore the original parameter before propagating
                # the optimization error.
                self.eta = original_eta
                raise

            predictions = self.predict(
                X_arr
            )

            fol = float(
                np.mean(
                    predictions == -1
                )
            )

            distance = abs(
                fol - target_fol
            )

            if distance < best_distance:
                best_distance = distance
                best_eta = eta

            # Stop once the desired outlier fraction has been reached
            # or exceeded.
            if fol >= target_fol:
                best_eta = eta
                break

        # Fit once more using selected eta.
        self.eta = float(best_eta)
        self.fit(X_arr)

        return self.eta

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_support(self) -> int:
        """
        Total number of support vectors.
        """
        check_is_fitted(self)

        return int(
            len(self.support_)
        )
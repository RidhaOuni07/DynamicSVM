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

With the kernel feature map phi, Phi = [phi(x_1), ..., phi(x_N)],
K = Phi^T Phi, H = I - (1/N) 11^T and the feature-space covariance

    Sigma = (1/N) Phi H Phi^T,

the COSVM primal is

    minimize    1/2 w^T (eta I + (1 - eta) Sigma) w - rho + C sum_i xi_i
    subject to  w^T phi(x_i) >= rho - xi_i,   xi_i >= 0,

with 0 < eta <= 1 and C = 1 / (nu N). Penalising w^T Sigma w pushes w
towards the low-variance directions of the target class.

Its dual is

    minimize    1/2 alpha^T G alpha
    subject to  0 <= alpha_i <= C,   sum(alpha_i) = 1,

    G = K A^{-1} K,   A = eta K + ((1 - eta)/N) K H K,

and the decision function is

    f(x) = k(x, X)^T A^{-1} K alpha - rho = k(x, X)^T coef - rho.

On training rows K coef = G alpha, so the KKT conditions and the decision
function use the same scores. eta = 1 is exactly the standard one-class
SVM (G = K). The derivation and the numerical treatment (Cholesky of
A + eps I) are in ``dynamicsvm.models._cosvm_core``.

Positive values correspond to normal samples and negative values
correspond to anomalies.
"""

from typing import Callable, Optional, Union

import numpy as np
from scipy.linalg import cho_solve

from dynamicsvm.base import BaseSVM
from dynamicsvm.kernels import Kernel, get_kernel
from dynamicsvm.models._cosvm_core import (
    COSVMGeometry,
    compute_rho,
    solve_oneclass_dual,
    stabilize_psd,
    validate_eta,
)
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
        Weight of the identity in the primal metric
        M = eta I + (1 - eta) Sigma.

            eta = 1:
                Standard One-Class SVM.

            0 < eta < 1:
                Covariance-guided COSVM; smaller eta penalises the
                high-variance directions of the target class more.

        eta = 0 is rejected: the out-of-sample score is then
        ill-defined (A is singular).

    kernel : {'rbf', 'linear', 'poly', 'sigmoid'} or Kernel or Callable
        Kernel function.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for RBF, polynomial and sigmoid kernels.

    degree : int, default=3
        Degree of the polynomial kernel.

    coef0 : float, default=0.0
        Independent term in polynomial and sigmoid kernels.

    tol : float, default=1e-5
        Optimization tolerance. SLSQP runs at ftol = min(tol, 1e-10).

    max_iter : int, default=1000
        Maximum number of SLSQP iterations.

    psd_tol : float, default=1e-10
        Relative tolerance used when checking the dual Gram matrix G for
        positive semidefiniteness: eigenvalues down to
        -psd_tol * max(1, max |eig(G)|) are treated as rounding.

    sv_tol : float, default=1e-7
        Tolerance for identifying support vectors.

    reg : float, default=1e-10
        Relative ridge of A: A + eps I with eps = reg * trace(A) / N.

    Attributes
    ----------
    alpha_ : ndarray
        Complete vector of dual coefficients.

    coef_ : ndarray
        Score coefficients A^{-1} K alpha; score(x) = k(x, X)^T coef_.

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
        Dual Gram matrix G = K A^{-1} K.

    geometry_ : COSVMGeometry
        Factorised geometry of the training set.

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
        reg: float = 1e-10,
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

        if psd_tol < 0:
            raise ValueError(
                f"psd_tol must be >= 0, got {psd_tol}"
            )

        if sv_tol <= 0:
            raise ValueError(
                f"sv_tol must be > 0, got {sv_tol}"
            )

        if reg <= 0:
            raise ValueError(
                f"reg must be > 0, got {reg}"
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
        self.reg = float(reg)

        # Fitted attributes
        self.kernel_: Optional[Kernel] = None

        self.X_train_: Optional[np.ndarray] = None

        self.K_train_: Optional[np.ndarray] = None
        self.K_tilde_: Optional[np.ndarray] = None
        self.geometry_: Optional[COSVMGeometry] = None

        self.alpha_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None

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
        Compute the COSVM dual Gram matrix

            G = K A^{-1} K,   A = eta K + ((1 - eta)/N) K H K,

        (G = K when eta == 1). See ``_cosvm_core.COSVMGeometry``.
        """
        return COSVMGeometry(
            K,
            self.eta,
            reg=self.reg,
        ).G

    def _stabilize_psd_matrix(
        self,
        K: np.ndarray,
    ) -> np.ndarray:
        """
        Numerically stabilize a theoretically PSD matrix.

        Negative eigenvalues within the relative tolerance
        psd_tol * max(1, max |eig|) are rounding and are shifted to zero;
        a more negative eigenvalue raises ValueError.
        """
        return stabilize_psd(
            K,
            self.psd_tol,
        )

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
        validate_eta(self.eta)

        X_arr = check_array(X)

        n_samples, n_features = X_arr.shape

        if n_samples < 2:
            raise ValueError(
                "COSVM requires at least two training samples."
            )

        self.X_train_ = X_arr.copy()
        self.n_features_in_ = n_features

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
        # COSVM dual Gram G = K A^{-1} K
        # --------------------------------------------------------------

        self.geometry_ = COSVMGeometry(
            self.K_train_,
            self.eta,
            reg=self.reg,
        )

        self.K_tilde_ = self._stabilize_psd_matrix(
            self.geometry_.G
        )

        # --------------------------------------------------------------
        # Dual optimization
        #
        # min 1/2 alpha^T G alpha
        #
        # s.t.
        #       0 <= alpha_i <= C
        #       sum(alpha_i) = 1
        # --------------------------------------------------------------

        alpha, result = solve_oneclass_dual(
            self.K_tilde_,
            self.C_,
            x0=None,
            ftol=min(self.tol, 1e-10),
            max_iter=self.max_iter,
        )

        self.optimization_result_ = result

        self.alpha_ = alpha
        self.coef_ = self.geometry_.coef(alpha)

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
        # rho from the KKT conditions (free support vectors satisfy
        # (G alpha)_i = rho).
        # --------------------------------------------------------------

        self._compute_rho()

        return self

    # ------------------------------------------------------------------
    # Rho / threshold
    # ------------------------------------------------------------------

    def _compute_rho(self) -> None:
        """
        Compute the decision-function offset rho.

        Prefer free support vectors (0 < alpha_i < C, where
        (G alpha)_i = rho). If there are none, use the midpoint of the
        KKT interval [max_{alpha_i = C} (G alpha)_i,
        min_{alpha_i = 0} (G alpha)_i].
        """
        check_is_fitted(self)

        rho = compute_rho(
            self.K_tilde_,
            self.alpha_,
            self.C_,
        )

        self.rho_ = rho

        # Keep compatibility with the original implementation.
        self.offset_ = rho

        # SVM-style intercept convention.
        self.intercept_ = -rho

    # ------------------------------------------------------------------
    # Test kernels
    # ------------------------------------------------------------------

    def _test_kernel(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Raw kernel k(X_test, X_train).
        """
        check_is_fitted(self)

        X_test_arr = check_array(
            X_test,
            n_features=getattr(self, "n_features_in_", None),
        )

        return np.asarray(
            self.kernel_(
                X_test_arr,
                self.X_train_,
            ),
            dtype=float,
        )

    def _transform_test_kernel(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Dual cross-Gram between test and training samples,

            G(X_test, X_train) = k(X_test, X_train) A^{-1} K,

        the rectangular counterpart of G (its rows on training points are
        rows of G). score(x) = G(x, X_train) alpha = k(x, X_train) coef_.
        """
        K_test = self._test_kernel(X_test)

        if self.geometry_.cho_ is None:
            return K_test

        return K_test @ cho_solve(
            self.geometry_.cho_,
            self.K_train_,
        )

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

                k(x, X_train)^T coef_ = k(x, X_train)^T A^{-1} K alpha
        """
        return (
            self._test_kernel(X) @ self.coef_
        )

    def decision_function(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the signed COSVM decision function.

            f(x) = k(x, X_train)^T coef_ - rho

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

        The search starts from eta=1 and moves toward eta=0 (eta = 0
        itself is excluded, see the class docstring).

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

        # eta must stay in (0, 1].
        eta_values = eta_values[
            eta_values > 1e-12
        ]

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

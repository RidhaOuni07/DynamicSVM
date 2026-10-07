"""
Covariance-guided One-Class Support Vector Machine (COSVM).
Based on:
    Takoua Kefi-Fatteh, Riadh Ksantini, Mohamed-Becha Kaaniche, Adel Bouhoula.
    "A novel incremental one-class support vector machine based on low variance direction."
    Pattern Recognition 91 (2019): 308-321.
    and
    Naimul Mefraz Khan et al. "Covariance-guided One-Class Support Vector Machine."
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize

from dynamicsvm.base import BaseSVM
from dynamicsvm.kernels import Kernel, get_kernel
from dynamicsvm.utils.validation import check_array, check_is_fitted


class COSVM(BaseSVM):
    r"""
    Covariance-guided One-Class Support Vector Machine (COSVM).

    Improves upon standard One-Class SVM by controlling the direction of the separating
    hyperplane through incorporation of empirical covariance spread in kernel feature space,
    penalizing high-variance directions and emphasizing low-variance directions.

    Dual Optimization Problem:
    $$\min_{\alpha} \frac{1}{2} \alpha^T \tilde{K} \alpha - b \left(1 - \sum_{i=1}^N \alpha_i\right)$$
    $$\text{s.t. } 0 \le \alpha_i \le C, \quad \sum_{i=1}^N \alpha_i = 1$$
    where:
    $$\tilde{K} = \eta K + (1 - \eta) K \left(I - \frac{1}{N} \mathbf{1}\mathbf{1}^T\right) K^T$$
    and $C = \frac{1}{\nu N}$ with $\nu \in (0, 1]$.

    Parameters
    ----------
    nu : float, default=0.1
        An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
        Should be in the interval (0, 1].
    eta : float, default=0.5
        Tradeoff parameter in [0, 1].
        - eta = 1.0 reduces to standard One-Class SVM (OSVM).
        - eta = 0.0 emphasizes purely the low-variance kernel covariance direction.
        - 0 < eta < 1 balances kernel geometry with empirical variance structure.
    kernel : {'rbf', 'linear', 'poly', 'sigmoid'} or Kernel or Callable, default='rbf'
        Kernel function.
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    degree : int, default=3
        Degree of the polynomial kernel function.
    coef0 : float, default=0.0
        Independent term in polynomial and sigmoid kernel functions.
    tol : float, default=1e-5
        Tolerance for optimization solver stopping criterion.
    max_iter : int, default=1000
        Maximum iterations for quadratic solver.
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
    ):
        super().__init__()
        if not (0.0 < nu <= 1.0):
            raise ValueError(f"nu must be in (0, 1], got {nu}")
        self.nu = float(nu)
        self.eta = float(np.clip(eta, 0.0, 1.0))
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self.kernel_: Optional[Kernel] = None
        self.X_train_: Optional[np.ndarray] = None
        self.K_train_: Optional[np.ndarray] = None
        self.K_tilde_: Optional[np.ndarray] = None
        self.C_: float = 1.0
        self.offset_: float = 0.0

    def _compute_covariance_kernel_matrix(self, K: np.ndarray) -> np.ndarray:
        """
        Compute the covariance-augmented Gram matrix:
        K_tilde = eta * K + (1 - eta) * K * H * K^T
        where H = I - 1/N * 1*1^T.
        """
        N = K.shape[0]
        if N <= 1:
            return K.copy()

        # Center matrix H = I - 1/N * J
        # K * H = K - mean(K, axis=1, keepdims=True)
        K_mean = np.mean(K, axis=1, keepdims=True)
        KH = K - K_mean
        # KH * K^T
        KHK = np.dot(KH, K.T)

        K_tilde = self.eta * K + (1.0 - self.eta) * KHK
        # Ensure symmetry and numerical stability
        K_tilde = 0.5 * (K_tilde + K_tilde.T)
        return K_tilde

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "COSVM":
        """
        Fit the Covariance-guided One-Class SVM model on training data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors of the target/normal class.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : COSVM
            Fitted estimator.
        """
        X_arr = check_array(X)
        n_samples, n_features = X_arr.shape
        self.X_train_ = X_arr.copy()

        # Initialize kernel
        self.kernel_ = get_kernel(
            self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            n_features=n_features,
            X=X_arr,
        )

        self.C_ = 1.0 / (self.nu * n_samples)

        # Compute standard Gram matrix K and covariance-augmented K_tilde
        self.K_train_ = self.kernel_(X_arr, X_arr)
        self.K_tilde_ = self._compute_covariance_kernel_matrix(self.K_train_)

        # Solve dual quadratic program:
        # min 0.5 * alpha^T K_tilde alpha
        # s.t. 0 <= alpha_i <= C
        #      sum(alpha_i) = 1
        def objective(alpha: np.ndarray) -> float:
            return 0.5 * float(np.dot(alpha, np.dot(self.K_tilde_, alpha)))

        def gradient(alpha: np.ndarray) -> np.ndarray:
            return np.dot(self.K_tilde_, alpha)

        eq_constraint = {
            "type": "eq",
            "fun": lambda alpha: np.sum(alpha) - 1.0,
            "jac": lambda alpha: np.ones_like(alpha),
        }

        bounds = [(0.0, self.C_) for _ in range(n_samples)]
        # Uniform initial feasible point
        x0 = np.full(n_samples, 1.0 / n_samples)

        res = minimize(
            fun=objective,
            x0=x0,
            jac=gradient,
            bounds=bounds,
            constraints=[eq_constraint],
            method="SLSQP",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        alpha = np.maximum(0.0, res.x)
        # Re-normalize to satisfy sum(alpha) = 1
        alpha_sum = np.sum(alpha)
        if alpha_sum > 0:
            alpha = alpha / alpha_sum

        alpha[alpha < 1e-6] = 0.0

        # Identify Support Vectors
        sv_mask = alpha > 1e-6
        self.support_ = np.where(sv_mask)[0]
        self.support_vectors_ = X_arr[sv_mask]
        self.dual_coef_ = alpha[sv_mask]
        self.n_support_ = np.array([len(self.support_)])

        # Compute threshold offset b:
        # For margin SVs (0 < alpha_i < C), g_i = sum_j K_tilde_ij * alpha_j - b = 0 => b = (K_tilde * alpha)_i
        margin_mask = (alpha > 1e-5) & (alpha < (self.C_ - 1e-5))
        margin_idx = np.where(margin_mask)[0]

        K_tilde_alpha = np.dot(self.K_tilde_, alpha)
        if len(margin_idx) > 0:
            self.offset_ = float(np.mean(K_tilde_alpha[margin_idx]))
        elif len(self.support_) > 0:
            self.offset_ = float(np.mean(K_tilde_alpha[self.support_]))
        else:
            self.offset_ = 0.0

        self.intercept_ = -self.offset_
        return self

    def _transform_test_kernel(self, X_test: np.ndarray) -> np.ndarray:
        """
        Compute the modified kernel evaluation vector for test data X_test:
        k_tilde(x) = eta * K(x, X) + (1 - eta) * K(x, X) * H * K(X, X)^T
        """
        check_is_fitted(self)
        X_test_arr = check_array(X_test)
        K_test = self.kernel_(X_test_arr, self.X_train_)  # shape (n_test, n_train)

        if self.eta >= 1.0 or self.X_train_.shape[0] <= 1:
            return K_test

        # Center operator H on training kernel
        K_mean = np.mean(self.K_train_, axis=1, keepdims=True)
        KH = self.K_train_ - K_mean  # shape (n_train, n_train)

        # K_test * KH^T (or K_test * (I - 1/N J) * K_train^T)
        # Note: K_test * H = K_test - mean(K_test, axis=1)
        K_test_H = K_test - np.mean(K_test, axis=1, keepdims=True)
        K_test_cov = np.dot(K_test_H, self.K_train_.T)

        return self.eta * K_test + (1.0 - self.eta) * K_test_cov

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute raw scoring function (decision value before offset) for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        X_arr = check_array(X)
        K_tilde_test = self._transform_test_kernel(X_arr)

        full_alpha = np.zeros(self.X_train_.shape[0])
        full_alpha[self.support_] = self.dual_coef_

        scores = np.dot(K_tilde_test, full_alpha)
        return scores

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Signed distance to the separating hyperplane.
        Positive values indicate normal/target class (+1), negative values indicate outliers (-1).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        dec : ndarray of shape (n_samples,)
            f(x) = score(x) - offset
        """
        return self.score_samples(X) - self.offset_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            +1 for normal/target samples, -1 for outliers/anomalies.
        """
        dec = self.decision_function(X)
        return np.where(dec >= 0.0, 1, -1)

    def optimize_eta(
        self,
        X: np.ndarray,
        target_fol: float = 0.1,
        eta_step: float = 0.05,
    ) -> float:
        """
        Optimize the tradeoff parameter eta using fraction of outliers (f_OL) stopping criterion
        as described in Section 4.1 of Kefi-Fatteh et al. (2019).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training dataset.
        target_fol : float, default=0.1
            Predefined target fraction of outliers threshold (f_OL <= nu).
        eta_step : float, default=0.05
            Step size for decrementing eta from 1.0 downwards.

        Returns
        -------
        best_eta : float
            Optimal eta parameter value.
        """
        X_arr = check_array(X)
        n_samples = X_arr.shape[0]

        best_eta = 1.0
        eta_curr = 1.0
        while eta_curr >= 0.0:
            self.eta = eta_curr
            self.fit(X_arr)
            preds = self.predict(X_arr)
            fol = np.mean(preds == -1)

            best_eta = eta_curr
            if fol >= target_fol:
                break
            eta_curr -= eta_step

        self.eta = float(best_eta)
        self.fit(X_arr)
        return self.eta

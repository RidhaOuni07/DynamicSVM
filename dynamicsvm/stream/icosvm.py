"""
Incremental Covariance-guided One-Class Support Vector Machine (ICOSVM).
Based on:
    Takoua Kefi-Fatteh, Riadh Ksantini, Mohamed-Becha Kaaniche, Adel Bouhoula.
    "A novel incremental one-class support vector machine based on low variance direction."
    Pattern Recognition 91 (2019): 308-321.
"""

from typing import Callable, List, Optional, Set, Tuple, Union
import numpy as np

from dynamicsvm.base import BaseSVM
from dynamicsvm.kernels import Kernel, get_kernel
from dynamicsvm.models.cosvm import COSVM
from dynamicsvm.utils.validation import check_array, check_is_fitted


class ICOSVM(BaseSVM):
    r"""
    Incremental Covariance-guided One-Class Support Vector Machine (ICOSVM).

    Maintains Karush-Kuhn-Tucker (KKT) equilibrium dynamically across incoming streaming samples
    while retaining the low-variance direction covariance guidance in kernel space:

    $$g_i = \sum_{j} \tilde{K}_{ij} \alpha_j - b$$
    Partition into subsets:
    - $S$: Margin Support Vectors ($0 < \alpha_i < C$, $g_i = 0$)
    - $E$: Error Support Vectors ($\alpha_i = C$, $g_i \le 0$)
    - $O$: Non-Support Vectors ($\alpha_i = 0$, $g_i \ge 0$)

    Sensitivity & Inverse Matrix:
    $$R = \begin{bmatrix} 0 & \mathbf{1}_S^T \\ \mathbf{1}_S & \tilde{K}_{S,S} \end{bmatrix}^{-1}$$
    $$\beta = -R \begin{bmatrix} 1 \\ \tilde{K}_{S,c} \end{bmatrix}$$
    $$\gamma_i = \tilde{K}_{i,c} + \sum_{j \in S} \tilde{K}_{ij} \beta_j$$

    Parameters
    ----------
    nu : float, default=0.1
        Fraction of outliers bound, in (0, 1].
    eta : float, default=0.5
        Kernel vs Covariance tradeoff parameter in [0, 1].
    kernel : {'rbf', 'linear', 'poly', 'sigmoid'} or Kernel or Callable, default='rbf'
        Kernel function.
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    max_samples : Optional[int], default=None
        Maximum buffer size for sliding window stream tracking.
    tol : float, default=1e-5
        Tolerance for numerical checks.
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
    ):
        super().__init__()
        self.nu = float(nu)
        self.eta = float(np.clip(eta, 0.0, 1.0))
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.tol = float(tol)

        # Internal online state
        self.kernel_: Optional[Kernel] = None
        self.X_buffer_: List[np.ndarray] = []
        self.alpha_: np.ndarray = np.array([])
        self.offset_: float = 0.0
        self.R_: Optional[np.ndarray] = None
        self.S_set_: List[int] = []  # Margin SV indices
        self.E_set_: List[int] = []  # Error SV indices
        self.O_set_: List[int] = []  # Non SV indices
        self.K_full_: Optional[np.ndarray] = None
        self.K_tilde_full_: Optional[np.ndarray] = None
        self.n_samples_seen_: int = 0

    def _compute_covariance_kernel_matrix(self, K: np.ndarray) -> np.ndarray:
        N = K.shape[0]
        if N <= 1:
            return K.copy()
        K_mean = np.mean(K, axis=1, keepdims=True)
        KH = K - K_mean
        KHK = np.dot(KH, K.T)
        K_tilde = self.eta * K + (1.0 - self.eta) * KHK
        return 0.5 * (K_tilde + K_tilde.T)

    def _init_R_matrix(self) -> None:
        """Initialize inverse equilibrium matrix R for margin SVs."""
        if not self.S_set_:
            self.R_ = np.array([[-1e6]])
            return
        n_s = len(self.S_set_)
        M = np.zeros((n_s + 1, n_s + 1))
        M[0, 1:] = 1.0
        M[1:, 0] = 1.0
        s_idx = np.array(self.S_set_)
        M[1:, 1:] = self.K_tilde_full_[np.ix_(s_idx, s_idx)] + 1e-7 * np.eye(n_s)
        try:
            self.R_ = np.linalg.pinv(M)
        except Exception:
            self.R_ = np.linalg.lstsq(M, np.eye(n_s + 1), rcond=None)[0]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ICOSVM":
        """
        Initialize the incremental model with an initial training dataset X.
        """
        X_arr = check_array(X)
        n_samples, n_features = X_arr.shape

        self.kernel_ = get_kernel(
            self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            n_features=n_features,
            X=X_arr,
        )

        batch_model = COSVM(
            nu=self.nu,
            eta=self.eta,
            kernel=self.kernel_,
            tol=self.tol,
        )
        batch_model.fit(X_arr)

        self.X_buffer_ = [X_arr[i].copy() for i in range(n_samples)]
        self.n_samples_seen_ = n_samples

        full_alpha = np.zeros(n_samples)
        if batch_model.support_ is not None and batch_model.dual_coef_ is not None:
            full_alpha[batch_model.support_] = batch_model.dual_coef_

        self.alpha_ = full_alpha
        self.offset_ = batch_model.offset_
        self.intercept_ = -self.offset_

        C_val = 1.0 / (self.nu * n_samples)
        self.K_full_ = batch_model.K_train_.copy()
        self.K_tilde_full_ = batch_model.K_tilde_.copy()

        # Classify into S, E, O sets
        self.S_set_ = []
        self.E_set_ = []
        self.O_set_ = []

        for i in range(n_samples):
            a_i = self.alpha_[i]
            if a_i > self.tol and a_i < (C_val - self.tol):
                self.S_set_.append(i)
            elif a_i >= (C_val - self.tol):
                self.E_set_.append(i)
            else:
                self.O_set_.append(i)

        if not self.S_set_ and self.E_set_:
            # Promote first error SV to S for numerical baseline if empty
            self.S_set_.append(self.E_set_.pop(0))

        self._init_R_matrix()
        self._sync_support_vectors()
        return self

    def _sync_support_vectors(self) -> None:
        """Synchronize support vectors and dual coefficients."""
        if not self.X_buffer_:
            self.support_vectors_ = None
            self.dual_coef_ = None
            self.support_ = None
            self.n_support_ = np.array([0])
            return

        sv_indices = sorted(self.S_set_ + self.E_set_)
        self.support_ = np.array(sv_indices, dtype=int)
        X_mat = np.array(self.X_buffer_)
        self.support_vectors_ = X_mat[sv_indices] if len(sv_indices) > 0 else np.empty((0, X_mat.shape[1]))
        self.dual_coef_ = self.alpha_[sv_indices] if len(sv_indices) > 0 else np.empty(0)
        self.n_support_ = np.array([len(sv_indices)])

    def _recompute_full_matrices(self) -> None:
        """Recompute Gram matrix and covariance augmentation when buffer changes."""
        X_mat = np.array(self.X_buffer_)
        self.K_full_ = self.kernel_(X_mat, X_mat)
        self.K_tilde_full_ = self._compute_covariance_kernel_matrix(self.K_full_)

    def learn_one(self, x: np.ndarray) -> "ICOSVM":
        """
        Incrementally learn one incoming sample x.

        Parameters
        ----------
        x : array-like of shape (n_features,) or (1, n_features)

        Returns
        -------
        self : ICOSVM
        """
        x_vec = np.asarray(x).ravel()
        if not self.X_buffer_:
            return self.fit(x_vec.reshape(1, -1))

        self.n_samples_seen_ += 1
        self.X_buffer_.append(x_vec.copy())
        c_idx = len(self.X_buffer_) - 1
        n_samples = len(self.X_buffer_)
        C_val = 1.0 / (self.nu * n_samples)

        # Update Kernel Matrices
        self._recompute_full_matrices()

        # Append candidate to alpha array with initial weight 0
        self.alpha_ = np.append(self.alpha_, 0.0)

        # Compute initial gradient/margin slope g_c = sum_j K_tilde[c, j] * alpha[j] - offset
        g_c = float(np.dot(self.K_tilde_full_[c_idx], self.alpha_) - self.offset_)

        # If sample satisfies KKT (g_c >= 0 with alpha_c = 0), it is a non-support vector
        if g_c >= -self.tol:
            self.O_set_.append(c_idx)
            self._sync_support_vectors()
            self._enforce_max_samples()
            return self

        # Adiabatic update: find maximum feasible alpha_c increment
        # Compute beta and gamma
        if not self.S_set_:
            self.S_set_.append(c_idx)
            self._init_R_matrix()
            self._reoptimize_step()
            self._sync_support_vectors()
            self._enforce_max_samples()
            return self

        s_idx = np.array(self.S_set_)
        k_s_c = self.K_tilde_full_[s_idx, c_idx]
        v = np.concatenate([[1.0], k_s_c])
        beta = -np.dot(self.R_, v)  # shape: [beta_b, beta_s1, beta_s2, ...]
        beta_b = beta[0]
        beta_s = beta[1:]

        gamma_c = float(self.K_tilde_full_[c_idx, c_idx] + np.dot(k_s_c, beta_s) + beta_b)
        if abs(gamma_c) < 1e-9:
            gamma_c = 1e-9

        # Maximum increments
        delta_S = -g_c / gamma_c if gamma_c > 0 else C_val
        delta_E = C_val - self.alpha_[c_idx]

        delta_alpha = min(max(0.0, delta_S), delta_E)

        # Apply updates
        self.alpha_[c_idx] += delta_alpha
        for idx_pos, s_pos in enumerate(self.S_set_):
            self.alpha_[s_pos] += beta_s[idx_pos] * delta_alpha
        self.offset_ -= beta_b * delta_alpha
        self.intercept_ = -self.offset_

        # Assign candidate to appropriate set
        if abs(self.alpha_[c_idx] - C_val) <= self.tol:
            self.E_set_.append(c_idx)
        elif self.alpha_[c_idx] > self.tol:
            self.S_set_.append(c_idx)
            self._init_R_matrix()
        else:
            self.O_set_.append(c_idx)

        # Re-verify and maintain partition equilibrium
        self._reoptimize_step()
        self._sync_support_vectors()
        self._enforce_max_samples()
        return self

    def _reoptimize_step(self) -> None:
        """
        Fast QP stabilization step ensuring strict sum(alpha)=1, box bounds, and exact KKT equilibrium.
        """
        from scipy.optimize import minimize

        n_samples = len(self.X_buffer_)
        C_val = 1.0 / (self.nu * n_samples)

        def objective(alpha: np.ndarray) -> float:
            return 0.5 * float(np.dot(alpha, np.dot(self.K_tilde_full_, alpha)))

        def gradient(alpha: np.ndarray) -> np.ndarray:
            return np.dot(self.K_tilde_full_, alpha)

        eq_constraint = {
            "type": "eq",
            "fun": lambda alpha: np.sum(alpha) - 1.0,
            "jac": lambda alpha: np.ones_like(alpha),
        }

        bounds = [(0.0, C_val) for _ in range(n_samples)]
        x0 = np.clip(self.alpha_, 0.0, C_val)
        if np.sum(x0) <= 0:
            x0 = np.full(n_samples, 1.0 / n_samples)
        else:
            x0 = x0 / np.sum(x0)

        res = minimize(
            fun=objective,
            x0=x0,
            jac=gradient,
            bounds=bounds,
            constraints=[eq_constraint],
            method="SLSQP",
            options={"maxiter": 200, "ftol": self.tol},
        )

        alpha = np.maximum(0.0, res.x)
        if np.sum(alpha) > 0:
            alpha = alpha / np.sum(alpha)
        alpha[alpha < 1e-6] = 0.0
        self.alpha_ = alpha

        # Update S, E, O partitions
        self.S_set_ = []
        self.E_set_ = []
        self.O_set_ = []
        for i in range(n_samples):
            a_i = self.alpha_[i]
            if a_i > self.tol and a_i < (C_val - self.tol):
                self.S_set_.append(i)
            elif a_i >= (C_val - self.tol):
                self.E_set_.append(i)
            else:
                self.O_set_.append(i)

        if not self.S_set_ and (self.E_set_ or self.O_set_):
            candidates = self.E_set_ if self.E_set_ else self.O_set_
            self.S_set_.append(candidates.pop(0))

        self._init_R_matrix()
        K_tilde_alpha = np.dot(self.K_tilde_full_, self.alpha_)
        if self.S_set_:
            self.offset_ = float(np.mean(K_tilde_alpha[self.S_set_]))
        else:
            self.offset_ = float(np.mean(K_tilde_alpha))
        self.intercept_ = -self.offset_

    def unlearn_one(self, index: int) -> "ICOSVM":
        """
        Decremental unlearning of a sample at buffer index.
        """
        if index < 0 or index >= len(self.X_buffer_):
            raise IndexError(f"Index {index} out of range for buffer size {len(self.X_buffer_)}")

        del self.X_buffer_[index]
        self.alpha_ = np.delete(self.alpha_, index)
        if len(self.X_buffer_) == 0:
            self.alpha_ = np.array([])
            self.S_set_ = []
            self.E_set_ = []
            self.O_set_ = []
            self._sync_support_vectors()
            return self

        self._recompute_full_matrices()
        self._reoptimize_step()
        self._sync_support_vectors()
        return self

    def _enforce_max_samples(self) -> None:
        """Drop oldest samples if buffer exceeds max_samples."""
        if self.max_samples is not None:
            while len(self.X_buffer_) > self.max_samples:
                self.unlearn_one(0)

    def learn_batch(self, X: np.ndarray) -> "ICOSVM":
        """
        Incrementally learn a batch of incoming samples sequentially.
        """
        X_arr = check_array(X)
        for i in range(X_arr.shape[0]):
            self.learn_one(X_arr[i])
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision scoring values for samples in X.
        """
        check_is_fitted(self)
        X_arr = check_array(X)
        X_train_mat = np.array(self.X_buffer_)

        K_test = self.kernel_(X_arr, X_train_mat)
        if self.eta < 1.0 and X_train_mat.shape[0] > 1:
            K_test_H = K_test - np.mean(K_test, axis=1, keepdims=True)
            K_test_cov = np.dot(K_test_H, self.K_full_.T)
            K_tilde_test = self.eta * K_test + (1.0 - self.eta) * K_test_cov
        else:
            K_tilde_test = K_test

        return np.dot(K_tilde_test, self.alpha_)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Signed distance to the separating hyperplane.
        """
        return self.score_samples(X) - self.offset_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict +1 for inliers and -1 for outliers.
        """
        dec = self.decision_function(X)
        return np.where(dec >= 0.0, 1, -1)

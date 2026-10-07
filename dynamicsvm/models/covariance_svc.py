"""
Covariance-Aware Support Vector Classification (CovarianceAwareSVC).
Integrates empirical latent covariance geometry with kernel optimization.
"""

from typing import Callable, Optional, Union
import numpy as np

from dynamicsvm.base import BaseSVM
from dynamicsvm.kernels import Kernel, get_kernel
from dynamicsvm.solvers.smo import SMOSolver
from dynamicsvm.solvers.qp import QPSolver
from dynamicsvm.utils.validation import check_X_y, check_array, check_is_fitted


class CovarianceAwareSVC(BaseSVM):
    r"""
    Covariance-Aware Support Vector Classifier.
    Augments standard kernel Gram matrix $K$ with empirical latent covariance geometry:
    
    $$Q = \eta K + (1 - \eta) \Delta$$
    $$\Delta = K H K^T, \quad H = \frac{1}{d} Z (\Sigma + \epsilon I)^{-1} Z^T$$
    
    where $Z$ represents latent representations, $\Sigma$ is the active latent covariance,
    and $\eta \in [0, 1]$ balances standard kernel curvature with latent covariance geometry.
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter.
    eta : float, default=0.7
        Weight on standard kernel Gram matrix vs covariance component (1.0 = standard SVM).
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'} or Kernel, default='rbf'
        Base kernel function.
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly', 'sigmoid'.
    solver : {'smo', 'qp'}, default='smo'
        Dual optimization solver.
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    max_iter : int, default=1000
        Maximum iterations for solver.
    eps : float, default=1e-4
        Covariance regularization parameter.
    random_state : int, optional
        Seed for solver initialization.
    """

    def __init__(
        self,
        C: float = 1.0,
        eta: float = 0.7,
        kernel: Union[str, Kernel, Callable] = "rbf",
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        solver: str = "smo",
        tol: float = 1e-3,
        max_iter: int = 1000,
        eps: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.C = float(C)
        self.eta = float(np.clip(eta, 0.0, 1.0))
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.solver = solver
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.eps = float(eps)
        self.random_state = random_state

        self.cov_: Optional[np.ndarray] = None
        self.inv_cov_: Optional[np.ndarray] = None
        self.H_: Optional[np.ndarray] = None
        self.K_train_: Optional[np.ndarray] = None
        self.Z_train_: Optional[np.ndarray] = None
        self.y_encoded_: Optional[np.ndarray] = None
        self.alpha_full_: Optional[np.ndarray] = None

    def _init_kernel(self) -> Kernel:
        return get_kernel(
            self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
        )

    def _compute_H(self, Z: np.ndarray, cov: Optional[np.ndarray]) -> np.ndarray:
        n_samples, n_features = Z.shape
        if cov is None:
            cov = np.cov(Z, rowvar=False) if n_samples > 1 else np.eye(n_features)
            if cov.ndim == 0:
                cov = np.array([[cov]])

        cov_reg = cov + self.eps * np.eye(n_features)
        try:
            inv_cov = np.linalg.pinv(cov_reg)
        except np.linalg.LinAlgError:
            inv_cov = np.eye(n_features) / self.eps

        self.cov_ = cov
        self.inv_cov_ = inv_cov
        # H = (1 / d) * Z * inv_cov * Z^T
        H = np.dot(np.dot(Z, inv_cov), Z.T) / max(1, n_features)
        return H

    def _compute_Q(
        self,
        K: np.ndarray,
        H: np.ndarray,
    ) -> np.ndarray:
        """
        Compute covariance-guided Gram matrix Q = eta * K + (1 - eta) * Delta.
        """
        if self.eta >= 1.0:
            return K

        Delta = np.dot(np.dot(K, H), K)
        # Normalize Delta scale to match K for numerical consistency
        norm_K = np.linalg.norm(K, "fro")
        norm_Delta = np.linalg.norm(Delta, "fro")
        if norm_Delta > 1e-12 and norm_K > 1e-12:
            Delta = Delta * (norm_K / norm_Delta)

        Q = self.eta * K + (1.0 - self.eta) * Delta
        # Ensure positive semi-definiteness
        Q = 0.5 * (Q + Q.T)
        return Q

    def fit(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        cov: Optional[np.ndarray] = None,
        warm_start_alpha: Optional[np.ndarray] = None,
        warm_start_b: float = 0.0,
    ) -> "CovarianceAwareSVC":
        """
        Fit the Covariance-Aware SVM model in latent space Z.
        """
        Z_arr, y_arr = check_X_y(Z, y)
        self.classes_ = np.unique(y_arr)

        if len(self.classes_) != 2:
            raise ValueError(
                f"CovarianceAwareSVC supports binary classification. Found {len(self.classes_)} classes."
            )

        # Encode labels to {-1.0, +1.0}
        y_encoded = np.where(y_arr == self.classes_[1], 1.0, -1.0)
        self.y_encoded_ = y_encoded
        self.Z_train_ = Z_arr.copy()

        self.kernel_ = self._init_kernel()
        K = self.kernel_(Z_arr, Z_arr)
        self.K_train_ = K

        self.H_ = self._compute_H(Z_arr, cov)
        Q = self._compute_Q(K, self.H_)

        # Select solver
        if self.solver == "smo":
            solver_inst = SMOSolver(
                C=self.C,
                tol=self.tol,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        elif self.solver == "qp":
            solver_inst = QPSolver(
                C=self.C,
                tol=self.tol,
                max_iter=self.max_iter,
            )
        else:
            raise ValueError(f"Unknown solver: '{self.solver}'")

        alpha, b, n_iter = solver_inst.solve(
            Q,
            y_encoded,
            initial_alpha=warm_start_alpha,
            initial_b=warm_start_b,
        )

        self.alpha_full_ = alpha.copy()
        sv_threshold = 1e-5
        sv_mask = alpha > sv_threshold

        self.support_ = np.where(sv_mask)[0]
        self.support_vectors_ = Z_arr[sv_mask]
        self.alpha_ = alpha[sv_mask]
        self.dual_coef_ = (alpha[sv_mask] * y_encoded[sv_mask]).reshape(1, -1)
        self.intercept_ = float(b)
        self.n_iter_ = n_iter

        sv_y = y_encoded[sv_mask]
        n_sv_neg = int(np.sum(sv_y == -1.0))
        n_sv_pos = int(np.sum(sv_y == 1.0))
        self.n_support_ = np.array([n_sv_neg, n_sv_pos], dtype=int)

        return self

    def decision_function(self, Z: np.ndarray) -> np.ndarray:
        """
        Evaluate decision function in latent space:
        f(Z) = sum_{j} alpha_j y_j Q(Z_test, Z_train) + b
        """
        check_is_fitted(self)
        Z_arr = check_array(Z)

        if len(self.support_) == 0 or self.Z_train_ is None:
            return np.full(Z_arr.shape[0], self.intercept_)

        # Base kernel between test samples and training samples
        K_test_train = self.kernel_(Z_arr, self.Z_train_)

        if self.eta >= 1.0 or self.H_ is None or self.K_train_ is None:
            Q_test = K_test_train
        else:
            Delta_test = np.dot(np.dot(K_test_train, self.H_), self.K_train_)
            norm_K = np.linalg.norm(self.K_train_, "fro")
            norm_Delta = np.linalg.norm(np.dot(np.dot(self.K_train_, self.H_), self.K_train_), "fro")
            if norm_Delta > 1e-12 and norm_K > 1e-12:
                Delta_test = Delta_test * (norm_K / norm_Delta)
            Q_test = self.eta * K_test_train + (1.0 - self.eta) * Delta_test

        # Compute f(Z) = Q_test * (alpha * y) + b
        ay_full = self.alpha_full_ * self.y_encoded_
        decision = np.dot(Q_test, ay_full) + self.intercept_
        return decision

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Predict class labels for latent samples Z."""
        decision = self.decision_function(Z)
        pred_idx = np.where(decision >= 0, 1, 0)
        return self.classes_[pred_idx]

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        """Compute class probabilities via logistic sigmoid mapping."""
        df = self.decision_function(Z)
        prob_pos = 1.0 / (1.0 + np.exp(-np.clip(df, -50.0, 50.0)))
        prob_neg = 1.0 - prob_pos
        return np.column_stack([prob_neg, prob_pos])

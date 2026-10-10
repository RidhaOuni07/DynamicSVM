"""
Covariance-Aware Support Vector Classification (CovarianceAwareSVC).
Integrates empirical latent covariance geometry with kernel optimization.
"""

from typing import Callable, Optional, Union
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular

from dynamicsvm.base import BaseSVM
from dynamicsvm.kernels import Kernel, get_kernel
from dynamicsvm.solvers.smo import SMOSolver
from dynamicsvm.solvers.qp import QPSolver
from dynamicsvm.utils.validation import check_X_y, check_array, check_is_fitted


class CovarianceAwareSVC(BaseSVM):
    r"""
    Covariance-Aware Support Vector Classifier.
    Augments standard kernel Gram matrix $K$ with empirical latent covariance geometry:
    
    $$Q = \eta K + (1 - \eta) \, s \, \Delta$$
    $$\Delta = K H K, \quad H = \frac{1}{d} Z_c (\Sigma + \epsilon I)^{-1} Z_c^T,
      \quad s = \frac{\|K\|_F}{\|K H K\|_F}$$
    
    where $Z$ represents latent representations, $Z_c = Z - \mu$ is $Z$ centred
    with the mean $\mu$ that belongs to the covariance $\Sigma$ (the column mean
    of $Z$ unless ``fit(..., mean=...)`` is given), and $\eta \in [0, 1]$
    balances standard kernel curvature with latent covariance geometry.
    $H$ is a centred Mahalanobis Gram matrix, so $H$, $\Delta$ and $Q$ are
    positive semi-definite by construction.

    The scale $s$ is a *fitted constant* (stored as ``delta_scale_``): it makes
    $\|s\Delta\|_F = \|K\|_F$ on the training set so that $\eta$ weighs two
    terms of the same magnitude. It is computed once at ``fit`` and reused
    unchanged for prediction, so train and test use the same kernel
    $Q(x, x') = \eta k(x, x') + (1-\eta) s \, k(x, Z)^T H k(Z, x')$.
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter.
    eta : float, default=0.7
        Weight on standard kernel Gram matrix vs covariance component (1.0 = standard SVM).
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'} or Kernel, default='rbf'
        Base kernel function.
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly', 'sigmoid'. A string is resolved
        once from the training matrix in ``fit``.
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
        self.mean_: Optional[np.ndarray] = None
        self.inv_cov_: Optional[np.ndarray] = None
        self.H_: Optional[np.ndarray] = None
        self.K_train_: Optional[np.ndarray] = None
        self.Z_train_: Optional[np.ndarray] = None
        self.y_encoded_: Optional[np.ndarray] = None
        self.alpha_full_: Optional[np.ndarray] = None
        self.ay_full_: Optional[np.ndarray] = None
        self.v_: Optional[np.ndarray] = None
        self.delta_scale_: float = 1.0

    def _init_kernel(self, X: np.ndarray) -> Kernel:
        # A string gamma is resolved once here, from the training matrix.
        return get_kernel(
            self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            X=X,
            n_features=X.shape[1],
        )

    def _compute_H(
        self,
        Z: np.ndarray,
        cov: Optional[np.ndarray],
        mean: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        H = (1/d) Z_c (Sigma + eps I)^{-1} Z_c^T with Z_c = Z - mean.

        Computed as W^T W / d with L L^T = Sigma + eps I (Cholesky) and
        W = L^{-1} Z_c^T, so H is PSD by construction (no pseudo-inverse).
        """
        n_samples, n_features = Z.shape
        if cov is None:
            cov = np.cov(Z, rowvar=False) if n_samples > 1 else np.eye(n_features)
        cov = np.atleast_2d(np.asarray(cov, dtype=float))
        if mean is None:
            mean = np.mean(Z, axis=0)
        mean = np.asarray(mean, dtype=float).ravel()

        cov_reg = 0.5 * (cov + cov.T) + self.eps * np.eye(n_features)
        L = cholesky(cov_reg, lower=True)

        Z_c = Z - mean
        W = solve_triangular(L, Z_c.T, lower=True)

        self.cov_ = cov
        self.mean_ = mean
        self.inv_cov_ = cho_solve((L, True), np.eye(n_features))
        H = np.dot(W.T, W) / max(1, n_features)
        return H

    def _compute_Q(
        self,
        K: np.ndarray,
        H: np.ndarray,
    ) -> np.ndarray:
        """
        Compute covariance-guided Gram matrix Q = eta * K + (1 - eta) * s * Delta,
        Delta = K H K, s = ||K||_F / ||Delta||_F (stored as ``delta_scale_``).
        """
        if self.eta >= 1.0:
            self.delta_scale_ = 1.0
            return K

        Delta = np.dot(np.dot(K, H), K)
        # Fitted scale constant s making ||s Delta||_F = ||K||_F
        norm_K = np.linalg.norm(K, "fro")
        norm_Delta = np.linalg.norm(Delta, "fro")
        if norm_Delta > 1e-12 and norm_K > 1e-12:
            self.delta_scale_ = float(norm_K / norm_Delta)
        else:
            self.delta_scale_ = 1.0

        Q = self.eta * K + (1.0 - self.eta) * self.delta_scale_ * Delta
        # Symmetrise against round-off (K and K H K are PSD, so Q is PSD)
        Q = 0.5 * (Q + Q.T)
        return Q

    def fit(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        cov: Optional[np.ndarray] = None,
        warm_start_alpha: Optional[np.ndarray] = None,
        warm_start_b: float = 0.0,
        mean: Optional[np.ndarray] = None,
    ) -> "CovarianceAwareSVC":
        """
        Fit the Covariance-Aware SVM model in latent space Z.

        Parameters
        ----------
        cov : np.ndarray of shape (d, d), optional
            Latent covariance Sigma (default: sample covariance of Z).
        mean : np.ndarray of shape (d,), optional
            Mean that belongs to ``cov``, used to centre Z in H
            (default: column mean of Z).
        """
        Z_arr, y_arr = check_X_y(Z, y)
        self.classes_ = np.unique(y_arr)
        self.n_features_in_ = Z_arr.shape[1]

        if len(self.classes_) != 2:
            raise ValueError(
                f"CovarianceAwareSVC supports binary classification. Found {len(self.classes_)} classes."
            )

        # Encode labels to {-1.0, +1.0}
        y_encoded = np.where(y_arr == self.classes_[1], 1.0, -1.0)
        self.y_encoded_ = y_encoded
        self.Z_train_ = Z_arr.copy()

        self.kernel_ = self._init_kernel(Z_arr)
        K = self.kernel_(Z_arr, Z_arr)
        self.K_train_ = K

        self.H_ = self._compute_H(Z_arr, cov, mean=mean)
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
        self.kkt_gap_ = float(solver_inst.kkt_gap_)

        self.alpha_full_ = alpha.copy()
        # Support vectors: every alpha > 0 (the solvers snap round-off to the bounds)
        sv_mask = alpha > 0.0

        self.support_ = np.where(sv_mask)[0]
        self.support_vectors_ = Z_arr[sv_mask]
        self.alpha_ = alpha[sv_mask]
        self.dual_coef_ = (alpha[sv_mask] * y_encoded[sv_mask]).reshape(1, -1)
        self.intercept_ = float(b)
        self.n_iter_ = n_iter

        # Cached prediction coefficients:
        # f(z) = eta k(z,Z) ay + (1-eta) s k(z,Z) H K ay + b = eta k ay + (1-eta) s k v + b
        self.ay_full_ = alpha * y_encoded
        if self.eta < 1.0:
            self.v_ = np.dot(self.H_, np.dot(K, self.ay_full_))
        else:
            self.v_ = None

        sv_y = y_encoded[sv_mask]
        n_sv_neg = int(np.sum(sv_y == -1.0))
        n_sv_pos = int(np.sum(sv_y == 1.0))
        self.n_support_ = np.array([n_sv_neg, n_sv_pos], dtype=int)

        return self

    def decision_function(self, Z: np.ndarray) -> np.ndarray:
        """
        Evaluate decision function in latent space:
        f(Z) = sum_{j} alpha_j y_j Q(Z_test, Z_train_j) + b
             = eta K_test (alpha*y) + (1 - eta) s K_test v + b,  v = H K (alpha*y).
        Cost O(n_test * N) per call.
        """
        check_is_fitted(self)
        Z_arr = check_array(Z, n_features=getattr(self, "n_features_in_", None))

        if len(self.support_) == 0 or self.Z_train_ is None:
            return np.full(Z_arr.shape[0], self.intercept_)

        # Base kernel between test samples and training samples
        K_test_train = self.kernel_(Z_arr, self.Z_train_)

        if self.eta >= 1.0 or self.v_ is None:
            decision = np.dot(K_test_train, self.ay_full_) + self.intercept_
        else:
            decision = (
                self.eta * np.dot(K_test_train, self.ay_full_)
                + (1.0 - self.eta) * self.delta_scale_ * np.dot(K_test_train, self.v_)
                + self.intercept_
            )
        return decision

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Predict class labels for latent samples Z."""
        decision = self.decision_function(Z)
        pred_idx = np.where(decision >= 0, 1, 0)
        return self.classes_[pred_idx]

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        """
        Uncalibrated logistic of the margin, 1 / (1 + exp(-f(z))), columns
        ordered as ``classes_``. Monotone in f but not a calibrated
        probability estimate.
        """
        df = self.decision_function(Z)
        prob_pos = 1.0 / (1.0 + np.exp(-np.clip(df, -50.0, 50.0)))
        prob_neg = 1.0 - prob_pos
        return np.column_stack([prob_neg, prob_pos])

"""
Support Vector Classifier (SVC) implementation with kernel support and dual solvers.
"""

from typing import Callable, Optional, Union
import numpy as np

from dynamicsvm.base import BaseSVM
from dynamicsvm.kernels import Kernel, get_kernel
from dynamicsvm.solvers.smo import SMOSolver
from dynamicsvm.solvers.qp import QPSolver
from dynamicsvm.utils.validation import check_X_y, check_array, check_is_fitted


class SVC(BaseSVM):
    """
    C-Support Vector Classification.
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is inversely
        proportional to C. Must be strictly positive.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'} or callable, default='rbf'
        Specifies the kernel type to be used in the algorithm.
    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    coef0 : float, default=0.0
        Independent term in kernel function.
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    max_iter : int, default=1000
        Hard limit on iterations within solver.
    solver : {'smo', 'qp'}, default='smo'
        Optimization solver to use. 'smo' is Platt's Sequential Minimal Optimization.
        'qp' uses general Quadratic Programming via scipy SLSQP.
    random_state : int, optional
        Seed for the random number generator.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: Union[str, Kernel, Callable] = "rbf",
        degree: int = 3,
        gamma: Union[str, float] = "scale",
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 1000,
        solver: str = "smo",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.C = float(C)
        self.kernel = kernel
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.solver = solver
        self.random_state = random_state

        if self.C <= 0:
            raise ValueError("C must be strictly positive.")

    def _init_kernel(self) -> Kernel:
        return get_kernel(
            self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVC":
        """
        Fit the SVM model according to the given training data.
        """
        X_arr, y_arr = check_X_y(X, y)
        self.classes_ = np.unique(y_arr)

        if len(self.classes_) != 2:
            raise ValueError(
                f"SVC only directly supports binary classification. Found {len(self.classes_)} classes. "
                f"For multi-class, use MultiClassSVC."
            )

        # Map classes to {-1, +1}
        # Class 0 -> -1.0, Class 1 -> +1.0
        y_encoded = np.where(y_arr == self.classes_[1], 1.0, -1.0)

        # Instantiate kernel
        self.kernel_ = self._init_kernel()

        # Compute Gram matrix
        K = self.kernel_(X_arr, X_arr)

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
            raise ValueError(f"Unknown solver: '{self.solver}'. Choose 'smo' or 'qp'.")

        alpha, b, n_iter = solver_inst.solve(K, y_encoded)

        # Identify support vectors (alpha > threshold)
        sv_threshold = 1e-5
        sv_mask = alpha > sv_threshold

        self.support_ = np.where(sv_mask)[0]
        self.support_vectors_ = X_arr[sv_mask]
        self.alpha_ = alpha[sv_mask]
        self.dual_coef_ = (alpha[sv_mask] * y_encoded[sv_mask]).reshape(1, -1)
        self.intercept_ = float(b)
        self.n_iter_ = n_iter

        # Count support vectors per class
        sv_y = y_encoded[sv_mask]
        n_sv_neg = int(np.sum(sv_y == -1.0))
        n_sv_pos = int(np.sum(sv_y == 1.0))
        self.n_support_ = np.array([n_sv_neg, n_sv_pos], dtype=int)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the decision function for the samples in X:
        f(X) = sum_{i in SV} alpha_i y_i K(x_i, X) + b
        """
        check_is_fitted(self)
        X_arr = check_array(X)

        if len(self.support_) == 0:
            # Trivial degenerate case: constant prediction
            return np.full(X_arr.shape[0], self.intercept_)

        # Gram matrix between X and support vectors: (n_test_samples, n_support_vectors)
        K_test = self.kernel_(X_arr, self.support_vectors_)

        # dual_coef_ is (1, n_sv), K_test is (n_samples, n_sv)
        decision = np.dot(K_test, self.dual_coef_.ravel()) + self.intercept_
        return decision

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on samples in X.
        """
        decision = self.decision_function(X)
        pred_idx = np.where(decision >= 0, 1, 0)
        return self.classes_[pred_idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute probabilities using logistic sigmoid scaling over decision function values.
        """
        df = self.decision_function(X)
        # Standard sigmoid mapping
        prob_pos = 1.0 / (1.0 + np.exp(-np.clip(df, -50.0, 50.0)))
        prob_neg = 1.0 - prob_pos
        return np.column_stack([prob_neg, prob_pos])

"""
Support Vector Classifier (SVC) implementation with kernel support and dual solvers.
"""

from typing import Callable, Optional, Union
import numpy as np

from dynamicsvm.base import BaseSVM
from dynamicsvm.kernels import Kernel, get_kernel
from dynamicsvm.solvers.smo import SMOSolver
from dynamicsvm.solvers.qp import QPSolver
from dynamicsvm.utils.calibration import fit_platt_sigmoid, platt_proba
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
        Tolerance for stopping criterion (SMO KKT tolerance; for 'qp' the
        KKT-gap acceptance threshold, SLSQP itself runs at ftol=1e-10).
    max_iter : int, default=1000
        Hard limit on iterations within solver.
    solver : {'smo', 'qp'}, default='smo'
        Optimization solver to use. 'smo' is Platt's Sequential Minimal Optimization.
        'qp' uses general Quadratic Programming via scipy SLSQP.
    random_state : int, optional
        Seed for the random number generator.
    probability : bool, default=False
        If True, ``fit`` additionally runs a deterministic stratified 3-fold
        internal cross-validation (seeded by ``random_state``) and fits Platt's
        sigmoid P(y=1|f) = 1/(1+exp(A f + B)) on the out-of-fold decision
        values (Lin, Lin & Weng, 2007). ``predict_proba`` then uses (A, B).

    Notes
    -----
    A string ``gamma`` is resolved once from the training matrix in ``fit``
    (scikit-learn semantics), so predictions do not depend on how the query
    samples are batched.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen in ``fit``.
    kkt_gap_ : float
        Maximal-violating-pair KKT gap of the solver's dual solution.
    probA_, probB_ : float or None
        Platt parameters (only when ``probability=True``).
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
        probability: bool = False,
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
        self.probability = bool(probability)
        self.probA_: Optional[float] = None
        self.probB_: Optional[float] = None

        if self.C <= 0:
            raise ValueError("C must be strictly positive.")

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVC":
        """
        Fit the SVM model according to the given training data.
        """
        X_arr, y_arr = check_X_y(X, y)
        self.classes_ = np.unique(y_arr)
        self.n_features_in_ = X_arr.shape[1]

        if len(self.classes_) != 2:
            raise ValueError(
                f"SVC only directly supports binary classification. Found {len(self.classes_)} classes. "
                f"For multi-class, use MultiClassSVC."
            )

        # Map classes to {-1, +1}
        # Class 0 -> -1.0, Class 1 -> +1.0
        y_encoded = np.where(y_arr == self.classes_[1], 1.0, -1.0)

        # Instantiate kernel
        self.kernel_ = self._init_kernel(X_arr)

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
        self.kkt_gap_ = float(solver_inst.kkt_gap_)

        # Support vectors: every alpha > 0 (the solvers snap round-off to the bounds)
        sv_mask = alpha > 0.0

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

        self.probA_ = None
        self.probB_ = None
        if self.probability:
            self._fit_platt(X_arr, y_arr, y_encoded)

        return self

    def _fit_platt(self, X_arr: np.ndarray, y_arr: np.ndarray, y_encoded: np.ndarray) -> None:
        """Fit Platt's (A, B) on stratified 3-fold out-of-fold decision values."""
        n_folds = 3
        rng = np.random.RandomState(self.random_state if self.random_state is not None else 0)
        folds = np.empty(len(y_encoded), dtype=int)
        for label in (-1.0, 1.0):
            idx = np.where(y_encoded == label)[0]
            if len(idx) < n_folds:
                raise ValueError(
                    f"probability=True needs at least {n_folds} samples per class for the internal CV."
                )
            perm = rng.permutation(idx)
            folds[perm] = np.arange(len(perm)) % n_folds

        decision_oof = np.empty(len(y_encoded), dtype=float)
        for k in range(n_folds):
            test_mask = folds == k
            sub = SVC(
                C=self.C,
                kernel=self.kernel_,  # gamma already resolved on the full training set
                degree=self.degree,
                gamma=self.gamma,
                coef0=self.coef0,
                tol=self.tol,
                max_iter=self.max_iter,
                solver=self.solver,
                random_state=self.random_state,
                probability=False,
            )
            sub.fit(X_arr[~test_mask], y_arr[~test_mask])
            decision_oof[test_mask] = sub.decision_function(X_arr[test_mask])

        y01 = (y_encoded > 0).astype(int)
        self.probA_, self.probB_ = fit_platt_sigmoid(decision_oof, y01)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the decision function for the samples in X:
        f(X) = sum_{i in SV} alpha_i y_i K(x_i, X) + b
        """
        check_is_fitted(self)
        X_arr = check_array(X, n_features=getattr(self, "n_features_in_", None))

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
        Class probabilities, columns ordered as ``classes_``.

        With ``probability=True`` these are Platt-calibrated:
        P(classes_[1] | x) = 1 / (1 + exp(A f(x) + B)). Otherwise this is an
        *uncalibrated* logistic of the margin, 1 / (1 + exp(-f(x))): monotone
        in f, but not a probability estimate in any statistical sense.
        """
        df = self.decision_function(X)
        if self.probA_ is not None and self.probB_ is not None:
            prob_pos = platt_proba(df, self.probA_, self.probB_)
        else:
            prob_pos = 1.0 / (1.0 + np.exp(-np.clip(df, -50.0, 50.0)))
        prob_neg = 1.0 - prob_pos
        return np.column_stack([prob_neg, prob_pos])

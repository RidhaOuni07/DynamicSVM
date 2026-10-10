"""
Multi-class classification strategies (One-vs-Rest and One-vs-One) for SVM.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from dynamicsvm.base import BaseClassifier
from dynamicsvm.kernels import Kernel, get_kernel, resolve_gamma
from dynamicsvm.models.svc import SVC
from dynamicsvm.utils.validation import check_X_y, check_array, check_is_fitted


class MultiClassSVC(BaseClassifier):
    """
    Multi-class Support Vector Classifier using One-vs-Rest (OvR) or One-vs-One (OvO).
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter.
    kernel : str, Kernel or callable, default='rbf'
        Kernel function.
    strategy : {'ovr', 'ovo'}, default='ovr'
        Multi-class decomposition strategy.
    **kwargs :
        Additional arguments passed to binary SVC estimators.

    Notes
    -----
    A string ``gamma`` is resolved once on the full training matrix and shared
    by every binary sub-problem (as in libsvm).

    OvR predicts ``argmax_k f_k(x)``. The binary machines are trained
    independently, so their margins are not calibrated against each other;
    the comparison is the standard heuristic, not a probabilistic rule.

    OvO predicts by majority vote over the K(K-1)/2 pairwise machines. Ties
    go to the class with the lowest index in ``classes_`` (``np.argmax``
    returns the first maximum), as in libsvm.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: Union[str, Kernel, Callable] = "rbf",
        strategy: str = "ovr",
        **kwargs,
    ):
        self.C = C
        self.kernel = kernel
        self.strategy = strategy.lower()
        self.kwargs = kwargs
        self.estimators_: List[SVC] = []
        self.classes_: Optional[np.ndarray] = None
        self.pairs_: Optional[List[Tuple[int, int]]] = None

        if self.strategy not in ("ovr", "ovo"):
            raise ValueError(f"Unknown strategy '{self.strategy}'. Choose 'ovr' or 'ovo'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiClassSVC":
        X_arr, y_arr = check_X_y(X, y)
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("Training data must contain at least 2 distinct classes.")

        self.n_features_in_ = X_arr.shape[1]

        # Resolve a string gamma ONCE on the full training matrix and give the
        # same float to every binary sub-problem (libsvm does the same for OvO);
        # otherwise each OvO pair would get its own data-dependent kernel.
        sub_kwargs = dict(self.kwargs)
        kernel = self.kernel
        if isinstance(kernel, str):
            gamma = sub_kwargs.get("gamma", "scale")
            if isinstance(gamma, str):
                sub_kwargs["gamma"] = resolve_gamma(gamma, X=X_arr)
        elif isinstance(kernel, Kernel):
            kernel = get_kernel(kernel, X=X_arr, n_features=X_arr.shape[1])

        self.estimators_ = []

        if self.strategy == "ovr":
            # Train one binary SVC for each class vs all other classes
            for cls in self.classes_:
                y_binary = np.where(y_arr == cls, 1, 0)
                estimator = SVC(C=self.C, kernel=kernel, **sub_kwargs)
                estimator.fit(X_arr, y_binary)
                self.estimators_.append(estimator)

        elif self.strategy == "ovo":
            # Train one binary SVC for every pair of classes
            self.pairs_ = []
            for i in range(n_classes):
                for j in range(i + 1, n_classes):
                    cls_i = self.classes_[i]
                    cls_j = self.classes_[j]
                    mask = (y_arr == cls_i) | (y_arr == cls_j)
                    X_pair = X_arr[mask]
                    y_pair = y_arr[mask]

                    estimator = SVC(C=self.C, kernel=kernel, **sub_kwargs)
                    estimator.fit(X_pair, y_pair)
                    self.estimators_.append(estimator)
                    self.pairs_.append((i, j))

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate decision functions across all sub-estimators.
        """
        check_is_fitted(self, attributes=["classes_"])
        X_arr = check_array(X, n_features=getattr(self, "n_features_in_", None))

        if self.strategy == "ovr":
            scores = np.column_stack([est.decision_function(X_arr) for est in self.estimators_])
            return scores
        else:
            # Pairwise decision functions
            return np.column_stack([est.decision_function(X_arr) for est in self.estimators_])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        """
        check_is_fitted(self, attributes=["classes_"])
        X_arr = check_array(X, n_features=getattr(self, "n_features_in_", None))
        n_samples = X_arr.shape[0]

        if self.strategy == "ovr":
            scores = self.decision_function(X_arr)
            best_idx = np.argmax(scores, axis=1)
            return self.classes_[best_idx]

        elif self.strategy == "ovo":
            # Majority voting
            votes = np.zeros((n_samples, len(self.classes_)), dtype=int)
            for estimator in self.estimators_:
                preds = estimator.predict(X_arr)
                for sample_idx, pred_cls in enumerate(preds):
                    cls_idx = np.where(self.classes_ == pred_cls)[0][0]
                    votes[sample_idx, cls_idx] += 1

            best_idx = np.argmax(votes, axis=1)
            return self.classes_[best_idx]

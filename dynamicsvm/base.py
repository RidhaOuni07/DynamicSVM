"""
Base interfaces and classes for Support Vector Machines.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseEstimator(ABC):
    """Base class for all estimators in DynamicSVM."""

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = {}
        for key in getattr(self, "__dict__", {}):
            if not key.startswith("_"):
                params[key] = getattr(self, key)
        return params

    def set_params(self, **params: Any) -> "BaseEstimator":
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self


class BaseClassifier(BaseEstimator):
    """Base class for classifiers in DynamicSVM."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier":
        """Fit the model according to the given training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        """
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


class BaseSVM(BaseClassifier):
    """
    Base class for Support Vector Machine models.
    Provides common properties for support vectors, dual coefficients, and intercept.
    """

    def __init__(self):
        self.support_vectors_: Optional[np.ndarray] = None
        self.dual_coef_: Optional[np.ndarray] = None
        self.support_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.classes_: Optional[np.ndarray] = None
        self.n_support_: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        """Check if the estimator has been fitted."""
        return self.support_vectors_ is not None

    @abstractmethod
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the decision function for the samples in X.
        """
        pass

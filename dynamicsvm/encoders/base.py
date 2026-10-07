"""
Base classes for representation learning encoders in DynamicSVM.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseEncoder(ABC):
    """
    Abstract base class for representation learning encoders.
    Maps high-dimensional input features X into a stable latent space Z.
    """

    def __init__(self):
        self.is_frozen_: bool = False
        self.latent_dim_: Optional[int] = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseEncoder":
        """
        Fit the encoder representation on training data X.
        """
        pass

    @abstractmethod
    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input samples X into latent representations Z.
        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Alias for encode."""
        return self.encode(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the encoder and return latent representations."""
        return self.fit(X, y).encode(X)

    def freeze(self) -> "BaseEncoder":
        """
        Freeze the encoder parameters to ensure a stable coordinate system
        for downstream streaming SVM and drift monitoring.
        """
        self.is_frozen_ = True
        return self

    def unfreeze(self) -> "BaseEncoder":
        """Unfreeze the encoder parameters."""
        self.is_frozen_ = False
        return self

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """
        Reconstruct original features X from latent representations Z.
        Default implementation raises NotImplementedError if the encoder is non-inverting.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement decode/reconstruction.")

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Encode then decode."""
        return self.decode(self.encode(X))

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute squared reconstruction error ||X - reconstruct(X)||^2 per sample."""
        X_rec = self.reconstruct(X)
        return np.sum((X - X_rec) ** 2, axis=1)

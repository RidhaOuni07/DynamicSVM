"""
Kernel definitions and computations for Support Vector Machines.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union
import numpy as np


class Kernel(ABC):
    """Abstract base class for SVM Kernels."""

    @abstractmethod
    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the Gram matrix between X and Y.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples_X, n_features)
        Y : np.ndarray of shape (n_samples_Y, n_features), optional
            If None, Y is assumed to be X.

        Returns
        -------
        K : np.ndarray of shape (n_samples_X, n_samples_Y)
        """
        pass

    def diag(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the diagonal elements K(x_i, x_i) efficiently.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        diag_K : np.ndarray of shape (n_samples,)
        """
        return np.diag(self(X, X))


class LinearKernel(Kernel):
    """
    Linear Kernel: K(x, y) = <x, y> + c
    """

    def __init__(self, offset: float = 0.0):
        self.offset = float(offset)

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if Y is None:
            Y = X
        else:
            Y = np.asarray(Y, dtype=float)
        return np.dot(X, Y.T) + self.offset

    def diag(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.sum(X * X, axis=1) + self.offset

    def __repr__(self) -> str:
        return f"LinearKernel(offset={self.offset})"


class RBFKernel(Kernel):
    """
    Radial Basis Function (Gaussian) Kernel:
    K(x, y) = exp(-gamma * ||x - y||^2)
    """

    def __init__(self, gamma: Union[float, str] = "scale"):
        self.gamma = gamma

    def _resolve_gamma(self, X: np.ndarray) -> float:
        if isinstance(self.gamma, str):
            n_features = X.shape[1] if X.ndim > 1 else 1
            if self.gamma == "scale":
                X_var = np.var(X)
                return 1.0 / (n_features * X_var) if X_var != 0 else 1.0
            elif self.gamma == "auto":
                return 1.0 / n_features
            else:
                raise ValueError(f"Unknown gamma strategy: {self.gamma}")
        return float(self.gamma)

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        gamma_val = self._resolve_gamma(X)

        if Y is None:
            # Squared Euclidean distances: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 <x_i, x_j>
            X_norm_sq = np.sum(X ** 2, axis=1, keepdims=True)
            dist_sq = X_norm_sq + X_norm_sq.T - 2.0 * np.dot(X, X.T)
        else:
            Y = np.asarray(Y, dtype=float)
            X_norm_sq = np.sum(X ** 2, axis=1, keepdims=True)
            Y_norm_sq = np.sum(Y ** 2, axis=1, keepdims=True)
            dist_sq = X_norm_sq + Y_norm_sq.T - 2.0 * np.dot(X, Y.T)

        # Numerical stability: clamp distances to >= 0
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.exp(-gamma_val * dist_sq)

    def diag(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.ones(X.shape[0], dtype=float)

    def __repr__(self) -> str:
        return f"RBFKernel(gamma={self.gamma!r})"


class PolynomialKernel(Kernel):
    """
    Polynomial Kernel:
    K(x, y) = (gamma * <x, y> + coef0) ^ degree
    """

    def __init__(self, degree: int = 3, gamma: Union[float, str] = "scale", coef0: float = 1.0):
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)

    def _resolve_gamma(self, X: np.ndarray) -> float:
        if isinstance(self.gamma, str):
            n_features = X.shape[1] if X.ndim > 1 else 1
            if self.gamma == "scale":
                X_var = np.var(X)
                return 1.0 / (n_features * X_var) if X_var != 0 else 1.0
            elif self.gamma == "auto":
                return 1.0 / n_features
            else:
                raise ValueError(f"Unknown gamma strategy: {self.gamma}")
        return float(self.gamma)

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        gamma_val = self._resolve_gamma(X)
        if Y is None:
            Y = X
        else:
            Y = np.asarray(Y, dtype=float)
        return (gamma_val * np.dot(X, Y.T) + self.coef0) ** self.degree

    def __repr__(self) -> str:
        return f"PolynomialKernel(degree={self.degree}, gamma={self.gamma!r}, coef0={self.coef0})"


class SigmoidKernel(Kernel):
    """
    Sigmoid (Hyperbolic Tangent) Kernel:
    K(x, y) = tanh(gamma * <x, y> + coef0)
    """

    def __init__(self, gamma: Union[float, str] = "scale", coef0: float = 0.0):
        self.gamma = gamma
        self.coef0 = float(coef0)

    def _resolve_gamma(self, X: np.ndarray) -> float:
        if isinstance(self.gamma, str):
            n_features = X.shape[1] if X.ndim > 1 else 1
            if self.gamma == "scale":
                X_var = np.var(X)
                return 1.0 / (n_features * X_var) if X_var != 0 else 1.0
            elif self.gamma == "auto":
                return 1.0 / n_features
            else:
                raise ValueError(f"Unknown gamma strategy: {self.gamma}")
        return float(self.gamma)

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        gamma_val = self._resolve_gamma(X)
        if Y is None:
            Y = X
        else:
            Y = np.asarray(Y, dtype=float)
        return np.tanh(gamma_val * np.dot(X, Y.T) + self.coef0)

    def __repr__(self) -> str:
        return f"SigmoidKernel(gamma={self.gamma!r}, coef0={self.coef0})"


class CustomKernel(Kernel):
    """Wrapper for user-defined kernel functions."""

    def __init__(self, func: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.func = func

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if Y is None:
            Y = X
        else:
            Y = np.asarray(Y, dtype=float)
        return self.func(X, Y)

    def __repr__(self) -> str:
        return f"CustomKernel(func={self.func.__name__ if hasattr(self.func, '__name__') else self.func})"


def get_kernel(kernel: Union[str, Kernel, Callable], **kwargs) -> Kernel:
    """
    Factory function to obtain a Kernel instance.
    
    Parameters
    ----------
    kernel : str, Kernel instance, or callable
        'linear', 'rbf', 'poly', 'polynomial', 'sigmoid', or custom callable.
    **kwargs :
        Additional hyperparameters for the kernel (e.g. gamma, degree, coef0).
    """
    if isinstance(kernel, Kernel):
        return kernel

    if callable(kernel) and not isinstance(kernel, type):
        return CustomKernel(kernel)

    if isinstance(kernel, str):
        k_str = kernel.lower()
        gamma = kwargs.get("gamma", "scale")
        if isinstance(gamma, str) and "X" in kwargs and kwargs["X"] is not None:
            X_ref = np.asarray(kwargs["X"])
            n_features = X_ref.shape[1] if X_ref.ndim > 1 else 1
            if gamma == "scale":
                X_var = float(np.var(X_ref))
                gamma = 1.0 / (n_features * X_var) if X_var != 0.0 else 1.0
            elif gamma == "auto":
                gamma = 1.0 / n_features

        if k_str == "linear":
            return LinearKernel(offset=kwargs.get("offset", 0.0))
        elif k_str in ("rbf", "gaussian"):
            return RBFKernel(gamma=gamma)
        elif k_str in ("poly", "polynomial"):
            return PolynomialKernel(
                degree=kwargs.get("degree", 3),
                gamma=gamma,
                coef0=kwargs.get("coef0", 1.0),
            )
        elif k_str == "sigmoid":
            return SigmoidKernel(
                gamma=gamma,
                coef0=kwargs.get("coef0", 0.0),
            )
        else:
            raise ValueError(f"Unknown kernel name: '{kernel}'. Choose from 'linear', 'rbf', 'poly', 'sigmoid', or provide a Kernel instance/callable.")

    raise TypeError(f"Invalid kernel type: {type(kernel)}. Expected str, Kernel, or callable.")

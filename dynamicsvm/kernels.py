"""
Kernel definitions and computations for Support Vector Machines.

String gamma values (``'scale'`` / ``'auto'``) follow scikit-learn semantics
and are resolved exactly once, from the *training* matrix, by
:func:`get_kernel` (``get_kernel(..., X=X_train)``). A kernel object never
resolves gamma from the batch it is evaluated on: calling an RBF, polynomial
or sigmoid kernel whose gamma is still a string raises ``ValueError``. This
makes predictions independent of how the query samples are batched.
"""

import copy
import numbers
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union
import numpy as np


_GAMMA_STRINGS = ("scale", "auto")


def resolve_gamma(
    gamma: Union[float, str],
    X: Optional[np.ndarray] = None,
    n_features: Optional[int] = None,
) -> float:
    """
    Resolve a kernel coefficient to a strictly positive float.

    Parameters
    ----------
    gamma : {'scale', 'auto'} or float
        ``'scale'``: 1 / (n_features * X.var()) (1.0 if the variance is 0).
        ``'auto'``: 1 / n_features. A numeric gamma must be finite and > 0.
    X : np.ndarray of shape (n_samples, n_features), optional
        Training matrix used for ``'scale'``. With fewer than two rows (or no
        X at all) the variance of the data is not identifiable from the
        sample, so ``'scale'`` emits a ``UserWarning`` and falls back to the
        ``'auto'`` value 1 / n_features (equal to ``'scale'`` for
        standardised data).
    n_features : int, optional
        Number of features; taken from ``X.shape[1]`` when X is given.

    Returns
    -------
    gamma : float
    """
    if isinstance(gamma, str):
        if gamma not in _GAMMA_STRINGS:
            raise ValueError(
                f"Unknown gamma strategy: {gamma!r}. Use 'scale', 'auto' or a positive float."
            )
        X_ref = None
        if X is not None:
            X_ref = np.asarray(X, dtype=float)
            if X_ref.ndim == 1:
                # A 1-D vector is one sample unless the model has a single feature.
                X_ref = X_ref.reshape(-1, 1) if n_features == 1 else X_ref.reshape(1, -1)
            n_features = X_ref.shape[1]
        if n_features is None or int(n_features) < 1:
            raise ValueError(f"gamma={gamma!r} needs X or n_features to be resolved.")
        n_features = int(n_features)

        if gamma == "auto":
            return 1.0 / n_features

        # gamma == 'scale'
        if X_ref is None or X_ref.shape[0] < 2:
            warnings.warn(
                "gamma='scale' cannot be estimated from fewer than 2 samples; "
                f"using gamma = 1/n_features = {1.0 / n_features:.6g} (the 'auto' value).",
                UserWarning,
                stacklevel=3,
            )
            return 1.0 / n_features
        X_var = float(np.var(X_ref))
        return 1.0 / (n_features * X_var) if X_var != 0.0 else 1.0

    return _validate_numeric_gamma(gamma)


def _validate_numeric_gamma(gamma) -> float:
    if isinstance(gamma, bool) or not isinstance(gamma, numbers.Real):
        raise ValueError(f"gamma must be 'scale', 'auto' or a positive float, got {gamma!r}.")
    value = float(gamma)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"gamma must be finite and strictly positive, got {gamma!r}.")
    return value


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


class _GammaKernelMixin:
    """
    Shared gamma handling for kernels with a ``gamma`` coefficient.

    A string gamma is kept as a placeholder until :func:`get_kernel` resolves
    it from the training matrix; evaluating the kernel before that raises.
    """

    def _set_gamma(self, gamma: Union[float, str]) -> None:
        if isinstance(gamma, str):
            if gamma not in _GAMMA_STRINGS:
                raise ValueError(
                    f"Unknown gamma strategy: {gamma!r}. Use 'scale', 'auto' or a positive float."
                )
            self.gamma = gamma
        else:
            self.gamma = _validate_numeric_gamma(gamma)

    def _gamma_value(self) -> float:
        if isinstance(self.gamma, str):
            raise ValueError("gamma is unresolved; pass X= to get_kernel or use a numeric gamma")
        return float(self.gamma)

    def _resolve_gamma(self, X: Optional[np.ndarray] = None) -> float:
        """Backward-compatible alias of ``_gamma_value``; X is ignored on purpose."""
        return self._gamma_value()

    def resolved(self, X: Optional[np.ndarray] = None, n_features: Optional[int] = None):
        """Return a shallow copy with a string gamma resolved from X / n_features."""
        if not isinstance(self.gamma, str):
            return self
        new = copy.copy(self)
        new.gamma = resolve_gamma(self.gamma, X=X, n_features=n_features)
        return new


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


class RBFKernel(_GammaKernelMixin, Kernel):
    """
    Radial Basis Function (Gaussian) Kernel:
    K(x, y) = exp(-gamma * ||x - y||^2), gamma > 0 (positive definite).

    A string gamma must be resolved with ``get_kernel(..., X=X_train)``
    before the kernel is evaluated.
    """

    def __init__(self, gamma: Union[float, str] = "scale"):
        self._set_gamma(gamma)

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        gamma_val = self._gamma_value()

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


class PolynomialKernel(_GammaKernelMixin, Kernel):
    """
    Polynomial Kernel:
    K(x, y) = (gamma * <x, y> + coef0) ^ degree

    Positive semi-definite (a valid Mercer kernel) for integer degree >= 1,
    gamma > 0 and coef0 >= 0. With coef0 < 0 the Gram matrix can be
    indefinite. A string gamma must be resolved with
    ``get_kernel(..., X=X_train)`` before the kernel is evaluated.
    """

    def __init__(self, degree: int = 3, gamma: Union[float, str] = "scale", coef0: float = 1.0):
        is_integer = isinstance(degree, numbers.Integral) or (
            isinstance(degree, numbers.Real) and float(degree).is_integer()
        )
        if isinstance(degree, bool) or not is_integer or int(degree) < 1:
            raise ValueError(f"degree must be an integer >= 1, got {degree!r}.")
        self.degree = int(degree)
        self._set_gamma(gamma)
        self.coef0 = float(coef0)

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        gamma_val = self._gamma_value()
        if Y is None:
            Y = X
        else:
            Y = np.asarray(Y, dtype=float)
        return (gamma_val * np.dot(X, Y.T) + self.coef0) ** self.degree

    def __repr__(self) -> str:
        return f"PolynomialKernel(degree={self.degree}, gamma={self.gamma!r}, coef0={self.coef0})"


class SigmoidKernel(_GammaKernelMixin, Kernel):
    """
    Sigmoid (Hyperbolic Tangent) Kernel:
    K(x, y) = tanh(gamma * <x, y> + coef0)

    Not positive semi-definite in general (for any gamma > 0 there are inputs
    with an indefinite Gram matrix), so the dual is not always convex. SMO
    handles non-positive curvature (eta <= 0) explicitly, but SLSQP-based
    solvers can stop at non-global stationary points. A string gamma must be
    resolved with ``get_kernel(..., X=X_train)`` before the kernel is
    evaluated.
    """

    def __init__(self, gamma: Union[float, str] = "scale", coef0: float = 0.0):
        self._set_gamma(gamma)
        self.coef0 = float(coef0)

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        gamma_val = self._gamma_value()
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
        ``X`` (training matrix) and/or ``n_features`` resolve a string gamma
        once via :func:`resolve_gamma`; the returned kernel then stores a
        float gamma. Without them a string gamma is kept unresolved and the
        kernel raises ``ValueError`` when evaluated.
    """
    X_ref = kwargs.get("X", None)
    n_features = kwargs.get("n_features", None)
    can_resolve = X_ref is not None or n_features is not None

    if isinstance(kernel, Kernel):
        if isinstance(kernel, _GammaKernelMixin) and isinstance(kernel.gamma, str) and can_resolve:
            return kernel.resolved(X=X_ref, n_features=n_features)
        return kernel

    if callable(kernel) and not isinstance(kernel, type):
        return CustomKernel(kernel)

    if isinstance(kernel, str):
        k_str = kernel.lower()
        gamma = kwargs.get("gamma", "scale")
        if gamma is None:
            gamma = "scale"
        if k_str in ("rbf", "gaussian", "poly", "polynomial", "sigmoid"):
            if isinstance(gamma, str) and can_resolve:
                gamma = resolve_gamma(gamma, X=X_ref, n_features=n_features)

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

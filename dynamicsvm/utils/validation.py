"""
Input validation utilities.
"""

from typing import Optional, Tuple
import numpy as np


def check_array(
    X: np.ndarray,
    ensure_2d: bool = True,
    dtype: type = np.float64,
    n_features: Optional[int] = None,
) -> np.ndarray:
    """
    Validate and convert input to a 2D numpy array.

    Parameters
    ----------
    X : array-like
    ensure_2d : bool, default=True
    dtype : type, default=np.float64
    n_features : int, optional
        Number of features the estimator was fitted on. When given, a 1-D
        vector of length ``n_features`` is read as ONE sample, shape
        ``(1, n_features)`` (``n_features == 1`` keeps ``(n, 1)``); any other
        length, or a 2-D array with a different column count, raises
        ``ValueError``. When None, a 1-D input becomes ``(n, 1)`` (legacy
        behaviour used by ``fit``).
    """
    arr = np.asarray(X, dtype=dtype)
    if ensure_2d:
        if arr.ndim == 1:
            if n_features is None or int(n_features) == 1:
                arr = arr.reshape(-1, 1)
            elif arr.shape[0] == int(n_features):
                arr = arr.reshape(1, -1)
            else:
                raise ValueError(
                    f"1-D input of length {arr.shape[0]} is incompatible with "
                    f"n_features={int(n_features)}; pass a 2-D array of shape (n_samples, n_features)."
                )
        elif arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}")
        if n_features is not None and arr.shape[1] != int(n_features):
            raise ValueError(
                f"X has {arr.shape[1]} features, but the estimator expects {int(n_features)}."
            )
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Input array contains NaN or infinity.")
    return arr


def check_X_y(
    X: np.ndarray,
    y: np.ndarray,
    ensure_2d: bool = True,
    dtype: type = np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate input features X and target labels y."""
    X_arr = check_array(X, ensure_2d=ensure_2d, dtype=dtype)
    y_arr = np.asarray(y)

    if y_arr.ndim != 1:
        y_arr = y_arr.ravel()

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: X has {X_arr.shape[0]}, y has {y_arr.shape[0]}"
        )

    if len(y_arr) == 0:
        raise ValueError("Empty training data provided.")

    return X_arr, y_arr


def check_is_fitted(estimator, attributes=None, msg=None):
    """Check if the estimator has been fitted."""
    if msg is None:
        msg = f"This {estimator.__class__.__name__} instance is not fitted yet. Call 'fit' first."

    if attributes is None:
        attributes = ["support_vectors_"]

    for attr in attributes:
        if not hasattr(estimator, attr) or getattr(estimator, attr) is None:
            raise RuntimeError(msg)

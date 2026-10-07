"""
Input validation utilities.
"""

from typing import Tuple
import numpy as np


def check_array(
    X: np.ndarray,
    ensure_2d: bool = True,
    dtype: type = np.float64,
) -> np.ndarray:
    """Validate and convert input to 2D numpy array."""
    arr = np.asarray(X, dtype=dtype)
    if ensure_2d:
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}")
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

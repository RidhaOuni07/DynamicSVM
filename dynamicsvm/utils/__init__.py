"""
Validation and utility functions for DynamicSVM.
"""

from dynamicsvm.utils.validation import check_X_y, check_array, check_is_fitted
from dynamicsvm.utils.metrics import accuracy_score, confusion_matrix, hinge_loss

__all__ = [
    "check_X_y",
    "check_array",
    "check_is_fitted",
    "accuracy_score",
    "confusion_matrix",
    "hinge_loss",
]

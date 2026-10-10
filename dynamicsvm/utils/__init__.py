"""
Validation and utility functions for DynamicSVM.
"""

from dynamicsvm.utils.validation import check_X_y, check_array, check_is_fitted
from dynamicsvm.utils.metrics import accuracy_score, confusion_matrix, hinge_loss
from dynamicsvm.utils.kkt import (
    compute_intercept,
    dual_objective,
    intercept_interval,
    kkt_gap,
    kkt_report,
    oneclass_kkt_report,
)
from dynamicsvm.utils.calibration import fit_platt_sigmoid, platt_proba

__all__ = [
    "check_X_y",
    "check_array",
    "check_is_fitted",
    "accuracy_score",
    "confusion_matrix",
    "hinge_loss",
    "compute_intercept",
    "dual_objective",
    "intercept_interval",
    "kkt_gap",
    "kkt_report",
    "oneclass_kkt_report",
    "fit_platt_sigmoid",
    "platt_proba",
]

"""
Evaluation metrics for classification and SVM performance.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate classification accuracy."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(y_t) == 0:
        return 0.0
    return float(np.mean(y_t == y_p))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute confusion matrix to evaluate classification accuracy."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_t, y_p]))
    n_labels = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    cm = np.zeros((n_labels, n_labels), dtype=int)
    for t, p in zip(y_t, y_p):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def hinge_loss(y_true: np.ndarray, pred_decision: np.ndarray) -> float:
    """
    Compute the average hinge loss: max(0, 1 - y_true * f(x))
    Assumes y_true in {-1, +1}.
    """
    y_t = np.asarray(y_true, dtype=float)
    df = np.asarray(pred_decision, dtype=float)
    losses = np.maximum(0.0, 1.0 - y_t * df)
    return float(np.mean(losses))

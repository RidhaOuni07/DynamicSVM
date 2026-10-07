"""
Unit tests for MultiClassSVC.
"""

import numpy as np
from dynamicsvm.models.multiclass import MultiClassSVC
from dynamicsvm.utils.metrics import accuracy_score


def test_multiclass_ovr_and_ovo():
    # 3-class 2D dataset
    np.random.seed(42)
    c1 = np.random.randn(15, 2) + np.array([0.0, 4.0])
    c2 = np.random.randn(15, 2) + np.array([-4.0, -2.0])
    c3 = np.random.randn(15, 2) + np.array([4.0, -2.0])

    X = np.vstack([c1, c2, c3])
    y = np.array([0] * 15 + [1] * 15 + [2] * 15)

    # 1. One-vs-Rest
    ovr = MultiClassSVC(C=1.0, kernel="rbf", strategy="ovr", random_state=42)
    ovr.fit(X, y)
    preds_ovr = ovr.predict(X)
    assert accuracy_score(y, preds_ovr) >= 0.95

    # 2. One-vs-One
    ovo = MultiClassSVC(C=1.0, kernel="rbf", strategy="ovo", random_state=42)
    ovo.fit(X, y)
    preds_ovo = ovo.predict(X)
    assert accuracy_score(y, preds_ovo) >= 0.95

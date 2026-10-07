"""
Unit tests for SVC (Support Vector Classifier).
"""

import numpy as np
import pytest
from dynamicsvm.models.svc import SVC
from dynamicsvm.utils.metrics import accuracy_score


def test_svc_linear_kernel():
    # Linearly separable XOR-like split or shifted clusters
    np.random.seed(42)
    X_pos = np.random.randn(20, 2) + np.array([2.0, 2.0])
    X_neg = np.random.randn(20, 2) + np.array([-2.0, -2.0])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 20 + [-1] * 20)

    clf = SVC(C=1.0, kernel="linear", solver="smo", random_state=42)
    clf.fit(X, y)

    assert clf.is_fitted
    assert len(clf.support_vectors_) > 0
    assert clf.score(X, y) >= 0.95


def test_svc_rbf_kernel_nonlinear():
    # Concentric circles dataset
    np.random.seed(42)
    # Inner circle: class 0
    theta_inner = np.linspace(0, 2 * np.pi, 25)
    r_inner = 0.5 + 0.1 * np.random.randn(25)
    X_inner = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])

    # Outer circle: class 1
    theta_outer = np.linspace(0, 2 * np.pi, 25)
    r_outer = 2.0 + 0.1 * np.random.randn(25)
    X_outer = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])

    X = np.vstack([X_inner, X_outer])
    y = np.array([0] * 25 + [1] * 25)

    clf = SVC(C=10.0, kernel="rbf", gamma=1.0, solver="smo", random_state=42)
    clf.fit(X, y)

    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    assert acc == 1.0


def test_svc_qp_solver():
    np.random.seed(42)
    X = np.array([[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]])
    y = np.array([1, 1, -1, -1])

    clf_qp = SVC(C=1.0, kernel="linear", solver="qp")
    clf_qp.fit(X, y)
    assert clf_qp.score(X, y) == 1.0


def test_svc_invalid_targets():
    X = np.ones((5, 2))
    y = np.array([1, 2, 3, 4, 5])
    clf = SVC()
    with pytest.raises(ValueError):
        clf.fit(X, y)

"""
Unit tests for LASVM (Online and Active Learning Support Vector Machine with Support Vector Filtering).
"""

import numpy as np
import pytest

from dynamicsvm.solvers.lasvm import LASVM, LASVMClassifier
from dynamicsvm.models.svc import SVC


def test_lasvm_online_fit():
    np.random.seed(42)
    # Linearly separable clusters
    X_pos = np.random.randn(25, 2) + np.array([2.0, 2.0])
    X_neg = np.random.randn(25, 2) + np.array([-2.0, -2.0])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 25 + [-1] * 25)

    model = LASVM(C=1.0, kernel="linear", epochs=2, selection_strategy="online")
    model.fit(X, y)

    assert model.is_fitted
    preds = model.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.95

    state = model.get_model_state()
    assert state["n_support_vectors"] > 0
    assert state["n_samples_seen"] == 100


def test_lasvm_random_and_active_selection():
    np.random.seed(42)
    X = np.random.randn(40, 2)
    y = np.where(X[:, 0] * X[:, 1] > 0, 1, -1)

    # Test random selection
    model_rand = LASVM(C=1.0, kernel="rbf", gamma=0.5, epochs=2, selection_strategy="random", random_state=42)
    model_rand.fit(X, y)
    acc_rand = np.mean(model_rand.predict(X) == y)
    assert acc_rand >= 0.80

    # Test active selection
    model_active = LASVM(C=1.0, kernel="rbf", gamma=0.5, epochs=2, selection_strategy="active", random_state=42)
    model_active.fit(X, y)
    acc_active = np.mean(model_active.predict(X) == y)
    assert acc_active >= 0.80


def test_lasvm_streaming_learn_one():
    np.random.seed(42)
    X_init = np.array([[2.0, 2.0], [-2.0, -2.0]])
    y_init = np.array([1, -1])

    model = LASVM(C=1.0, kernel="rbf", gamma=0.5)
    model.fit(X_init, y_init)

    # Stream new points
    stream_X = np.array([
        [2.5, 2.5],
        [-2.5, -2.5],
        [1.8, 2.1],
        [-1.9, -2.2]
    ])
    stream_y = np.array([1, -1, 1, -1])

    for xi, yi in zip(stream_X, stream_y):
        model.learn_one(xi, yi)

    preds = model.predict(stream_X)
    assert np.all(preds == stream_y)


def test_lasvm_decision_and_proba():
    np.random.seed(42)
    X = np.random.randn(30, 2)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

    model = LASVM(C=1.0, kernel="linear")
    model.fit(X, y)

    df = model.decision_function(X)
    assert df.shape == (30,)

    proba = model.predict_proba(X)
    assert proba.shape == (30, 2)
    assert np.allclose(np.sum(proba, axis=1), 1.0)

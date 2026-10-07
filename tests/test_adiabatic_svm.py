"""
Unit tests for AdiabaticSVM (Exact Incremental and Decremental SVM via Adiabatic Path Following).
"""

import numpy as np
import pytest

from dynamicsvm.stream.adiabatic_svm import AdiabaticSVM, ExactIncrementalDecrementalSVM, IncrementalDecrementalSVC
from dynamicsvm.models.svc import SVC


def test_adiabatic_svm_batch_fit():
    np.random.seed(42)
    # Linearly separable clusters
    X_pos = np.random.randn(20, 2) + np.array([2.0, 2.0])
    X_neg = np.random.randn(20, 2) + np.array([-2.0, -2.0])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 20 + [-1] * 20)

    model = AdiabaticSVM(C=1.0, kernel="linear")
    model.fit(X, y)

    assert model.is_fitted
    preds = model.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.95

    # Check partitions S, E, O
    state = model.get_model_state()
    assert state["n_samples_buffered"] == 40
    assert state["n_margin_sv"] + state["n_error_sv"] + state["n_non_sv"] == 40
    assert model.support_vectors_ is not None
    assert len(model.support_vectors_) == (state["n_margin_sv"] + state["n_error_sv"])


def test_adiabatic_svm_incremental_learning():
    np.random.seed(42)
    # Initial seed
    X_init = np.array([
        [2.0, 2.0], [2.5, 2.5],
        [-2.0, -2.0], [-2.5, -2.5]
    ])
    y_init = np.array([1, 1, -1, -1])

    model = AdiabaticSVM(C=1.0, kernel="rbf", gamma=0.5)
    model.fit(X_init, y_init)

    # Stream new points one by one
    stream_X = np.array([
        [3.0, 3.0],
        [-3.0, -3.0],
        [1.5, 1.8],
        [-1.6, -1.7]
    ])
    stream_y = np.array([1, -1, 1, -1])

    for xi, yi in zip(stream_X, stream_y):
        model.learn_one(xi, yi)

    assert model.n_samples_seen_ == 8
    preds = model.predict(stream_X)
    assert np.all(preds == stream_y)


def test_adiabatic_svm_decremental_unlearning():
    np.random.seed(42)
    X = np.array([
        [2.0, 2.0], [2.2, 2.1], [3.0, 3.0],
        [-2.0, -2.0], [-2.2, -2.1], [-3.0, -3.0]
    ])
    y = np.array([1, 1, 1, -1, -1, -1])

    model = AdiabaticSVM(C=1.0, kernel="linear")
    model.fit(X, y)

    initial_state = model.get_model_state()
    assert initial_state["n_samples_buffered"] == 6

    # Unlearn a non-support vector / first sample
    model.forget_one(0)
    state_after_1 = model.get_model_state()
    assert state_after_1["n_samples_buffered"] == 5
    assert state_after_1["n_unlearned"] == 1

    # Unlearn by sample vector matching
    model.unlearn_sample(np.array([-3.0, -3.0]), y=-1)
    state_after_2 = model.get_model_state()
    assert state_after_2["n_samples_buffered"] == 4
    assert state_after_2["n_unlearned"] == 2


def test_adiabatic_svm_max_samples_pruning():
    np.random.seed(42)
    model = AdiabaticSVM(C=1.0, kernel="linear", max_samples=6, pruning_strategy="fifo")

    # Fit 4 samples
    X_init = np.array([[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]])
    y_init = np.array([1, 1, -1, -1])
    model.fit(X_init, y_init)

    # Stream 6 more samples
    for i in range(6):
        xi = np.array([1.0 + 0.1 * i, 1.0 + 0.1 * i]) if i % 2 == 0 else np.array([-1.0 - 0.1 * i, -1.0 - 0.1 * i])
        yi = 1 if i % 2 == 0 else -1
        model.learn_one(xi, yi)

    state = model.get_model_state()
    assert state["n_samples_buffered"] == 6
    assert state["n_samples_seen"] == 10
    assert state["n_unlearned"] == 4


def test_adiabatic_svm_decision_and_proba():
    np.random.seed(42)
    X = np.random.randn(30, 2)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

    model = AdiabaticSVM(C=1.0, kernel="rbf")
    model.fit(X, y)

    df = model.decision_function(X)
    assert df.shape == (30,)

    proba = model.predict_proba(X)
    assert proba.shape == (30, 2)
    assert np.allclose(np.sum(proba, axis=1), 1.0)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

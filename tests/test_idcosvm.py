"""
Unit tests for IncrementalDecrementalCOSVM (IDCOSVM).
"""

import numpy as np
import pytest

from dynamicsvm.stream.idcosvm import IncrementalDecrementalCOSVM, IDCOSVM


def test_idcosvm_fit_and_incremental_stream():
    np.random.seed(42)
    # Generate elongated target distribution
    X_init = np.random.randn(35, 2) * np.array([[2.5, 0.4]])

    model = IncrementalDecrementalCOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale")
    model.fit(X_init)

    assert model.is_fitted
    assert len(model.X_buffer_) == 35
    assert len(model.alpha_) == 35
    assert np.all(model.alpha_ >= 0)
    assert np.isclose(np.sum(model.alpha_), 1.0, atol=1e-3)

    # Stream in 15 new instances
    X_stream = np.random.randn(15, 2) * np.array([[2.5, 0.4]])
    model.learn_batch(X_stream)

    assert len(model.X_buffer_) == 50
    assert len(model.alpha_) == 50
    assert model.n_samples_seen_ == 50
    assert np.all(model.alpha_ >= 0)
    assert np.isclose(np.sum(model.alpha_), 1.0, atol=1e-3)

    state = model.get_model_state()
    assert state["buffer_size"] == 50
    assert state["n_samples_seen"] == 50


def test_idcosvm_decremental_unlearning_specific_indices():
    np.random.seed(42)
    X = np.random.randn(30, 2) * np.array([[3.0, 0.5]])

    model = IDCOSVM(nu=0.15, eta=0.5, kernel="rbf", gamma="scale")
    model.fit(X)

    init_len = len(model.X_buffer_)
    assert init_len == 30

    # Unlearn first instance
    model.unlearn_one(0)
    assert len(model.X_buffer_) == 29
    assert len(model.alpha_) == 29
    assert model.n_unlearned_ == 1
    assert np.isclose(np.sum(model.alpha_), 1.0, atol=1e-3)

    # Unlearn multiple instances
    model.unlearn_many([5, 10, 15])
    assert len(model.X_buffer_) == 26
    assert len(model.alpha_) == 26
    assert model.n_unlearned_ == 4
    assert np.isclose(np.sum(model.alpha_), 1.0, atol=1e-3)


def test_idcosvm_unlearn_sample_by_value():
    np.random.seed(42)
    X = np.random.randn(25, 3)

    model = IDCOSVM(nu=0.1, kernel="linear")
    model.fit(X)

    target_sample = X[7].copy()
    model.unlearn_sample(target_sample)

    assert len(model.X_buffer_) == 24
    assert model.n_unlearned_ == 1

    # Attempting to unlearn already removed sample should raise ValueError
    with pytest.raises(ValueError):
        model.unlearn_sample(np.array([100.0, 100.0, 100.0]))


def test_idcosvm_pruning_strategies():
    np.random.seed(42)
    X_init = np.random.randn(30, 2)

    # Test non_sv_first pruning strategy
    model_non_sv = IncrementalDecrementalCOSVM(
        nu=0.1, eta=0.5, max_samples=25, pruning_strategy="non_sv_first"
    )
    model_non_sv.fit(X_init)
    assert len(model_non_sv.X_buffer_) == 25

    # Test mahalanobis pruning strategy
    model_mah = IncrementalDecrementalCOSVM(
        nu=0.1, eta=0.5, max_samples=20, pruning_strategy="mahalanobis"
    )
    model_mah.fit(X_init)
    assert len(model_mah.X_buffer_) == 20

    # Stream in 10 more points - buffer should stay bounded at 20
    X_extra = np.random.randn(10, 2)
    model_mah.learn_many(X_extra)
    assert len(model_mah.X_buffer_) == 20


def test_idcosvm_classification_accuracy():
    np.random.seed(42)
    theta = np.pi / 4
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    # Target distribution
    X_train_base = np.random.randn(60, 2) * np.array([[3.5, 0.4]])
    X_train = np.dot(X_train_base, rot.T)

    model = IDCOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale")
    model.fit(X_train[:40])

    # Incrementally stream the remaining 20 samples
    for i in range(40, 60):
        model.learn_one(X_train[i])

    # Inliers test
    X_test_inliers = np.dot(np.random.randn(25, 2) * np.array([[3.5, 0.4]]), rot.T)
    y_pred_inliers = model.predict(X_test_inliers)
    assert np.mean(y_pred_inliers == 1) >= 0.55

    # Outliers test far off
    X_test_outliers = np.dot(np.random.randn(25, 2) * np.array([[0.5, 3.5]]), rot.T) + np.array([8.0, -8.0])
    y_pred_outliers = model.predict(X_test_outliers)
    assert np.mean(y_pred_outliers == -1) >= 0.85

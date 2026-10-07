"""
Unit tests for Covariance-guided One-Class Support Vector Machine (COSVM).
"""

import numpy as np
import pytest

from dynamicsvm.models.cosvm import COSVM


def test_cosvm_basic_fit_and_predict():
    np.random.seed(42)
    # Generate elongated 2D Gaussian dataset
    X_train = np.random.randn(80, 2) * np.array([[3.0, 0.5]])

    model = COSVM(nu=0.1, eta=0.5, kernel="rbf", gamma="scale")
    model.fit(X_train)

    assert model.is_fitted
    assert model.support_vectors_ is not None
    assert len(model.support_vectors_) > 0
    assert len(model.dual_coef_) == len(model.support_vectors_)
    assert np.all(model.dual_coef_ >= 0)

    # Inliers prediction
    y_train_pred = model.predict(X_train)
    # Most points should be classified as inliers (+1)
    inlier_fraction = np.mean(y_train_pred == 1)
    assert inlier_fraction >= 0.80

    # Anomalies prediction far from the origin
    X_anomalies = np.array([
        [15.0, 15.0],
        [-15.0, -15.0],
        [0.0, 10.0],
    ])
    y_anom_pred = model.predict(X_anomalies)
    assert np.all(y_anom_pred == -1)

    dec = model.decision_function(X_anomalies)
    assert np.all(dec < 0.0)


def test_cosvm_eta_variation():
    np.random.seed(42)
    X = np.random.randn(50, 4)

    # Test pure kernel (eta=1.0)
    cosvm_std = COSVM(nu=0.15, eta=1.0, kernel="linear").fit(X)
    # Test covariance-augmented (eta=0.3)
    cosvm_cov = COSVM(nu=0.15, eta=0.3, kernel="linear").fit(X)

    assert cosvm_std.is_fitted
    assert cosvm_cov.is_fitted

    scores_std = cosvm_std.score_samples(X[:5])
    scores_cov = cosvm_cov.score_samples(X[:5])
    assert scores_std.shape == (5,)
    assert scores_cov.shape == (5,)


def test_cosvm_optimize_eta():
    np.random.seed(42)
    X = np.random.randn(60, 2) * np.array([[2.5, 0.7]])

    model = COSVM(nu=0.2, kernel="rbf")
    best_eta = model.optimize_eta(X, target_fol=0.1, eta_step=0.1)

    assert 0.0 <= best_eta <= 1.0
    assert model.eta == best_eta
    assert model.is_fitted


def test_cosvm_invalid_nu():
    with pytest.raises(ValueError):
        COSVM(nu=0.0)
    with pytest.raises(ValueError):
        COSVM(nu=1.5)

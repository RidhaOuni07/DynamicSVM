"""
Unit tests for DriftAwareCovarianceSVM end-to-end architecture.
"""

import numpy as np
import pytest

from dynamicsvm.encoders import AutoEncoder, PCAEncoder, IdentityEncoder
from dynamicsvm.stream.drift_covariance_svm import DriftAwareCovarianceSVM


def test_drift_aware_covariance_svm_basic():
    rng = np.random.RandomState(42)
    # Initial stationary dataset
    X_pos = rng.randn(30, 4) + np.array([2.0, 2.0, 0.0, 0.0])
    X_neg = rng.randn(30, 4) + np.array([-2.0, -2.0, 0.0, 0.0])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 30 + [0] * 30)

    model = DriftAwareCovarianceSVM(
        encoder=PCAEncoder(n_components=2),
        C=1.0,
        eta=0.7,
        max_buffer_size=80,
        min_buffer_size=10,
        drift_threshold=2.5,
        random_state=42,
    )
    model.fit(X, y)

    assert model.is_fitted
    assert model.encoder.is_frozen_
    assert len(model._buffer_X) == 60

    # Test initial predictions
    preds = model.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.90

    # Stream new points
    new_x = np.array([2.5, 2.5, 0.1, -0.1])
    model.learn_one(new_x, 1)
    assert len(model._buffer_X) == 61

    state = model.get_model_state()
    assert state["mu_t"] is not None
    assert state["Sigma_t"] is not None
    assert "drift_state_t" in state


def test_drift_adaptation_and_decremental_removal():
    rng = np.random.RandomState(42)
    # Initial regime: centered at +2 and -2
    X_init_pos = rng.randn(20, 2) + 2.0
    X_init_neg = rng.randn(20, 2) - 2.0
    X_init = np.vstack([X_init_pos, X_init_neg])
    y_init = np.array([1] * 20 + [0] * 20)

    model = DriftAwareCovarianceSVM(
        encoder=IdentityEncoder(),
        C=1.0,
        eta=0.6,
        max_buffer_size=40,
        min_buffer_size=10,
        drift_threshold=1.5,
        warning_threshold=0.8,
        retire_ratio_on_drift=0.4,
        random_state=42,
    )
    model.fit(X_init, y_init)
    assert len(model._buffer_X) == 40

    # Introduce sudden distribution shift (drift) in incoming stream
    X_drift_pos = rng.randn(25, 2) + np.array([10.0, 10.0])
    X_drift_neg = rng.randn(25, 2) + np.array([5.0, 5.0])
    X_drift = np.vstack([X_drift_pos, X_drift_neg])
    y_drift = np.array([1] * 25 + [0] * 25)

    # Ingest streaming drifted data
    model.learn_many(X_drift, y_drift)

    # Bounded buffer size check
    assert len(model._buffer_X) <= model.max_buffer_size

    # Check drift was detected and logged
    assert model.drift_detector_.drift_count_ > 0
    assert len(model.drift_events_) > 0

    # Predictions on new regime should be accurate
    preds = model.predict(X_drift)
    acc = np.mean(preds == y_drift)
    assert acc >= 0.85

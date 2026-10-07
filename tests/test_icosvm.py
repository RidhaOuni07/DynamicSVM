"""
Unit tests for Incremental Covariance-guided One-Class Support Vector Machine (ICOSVM).
"""

import numpy as np
import pytest

from dynamicsvm.stream.icosvm import ICOSVM


def test_icosvm_fit_and_learn_one():
    np.random.seed(42)
    X_init = np.random.randn(40, 2) * np.array([[2.0, 0.5]])

    icosvm = ICOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale")
    icosvm.fit(X_init)

    assert icosvm.is_fitted
    assert len(icosvm.X_buffer_) == 40
    assert len(icosvm.alpha_) == 40

    # Stream in 10 additional points
    X_stream = np.random.randn(10, 2) * np.array([[2.0, 0.5]])
    for i in range(10):
        icosvm.learn_one(X_stream[i])

    assert len(icosvm.X_buffer_) == 50
    assert len(icosvm.alpha_) == 50
    assert icosvm.support_vectors_ is not None

    # Test predictions
    preds = icosvm.predict(X_init[:5])
    assert preds.shape == (5,)


def test_icosvm_learn_batch():
    np.random.seed(42)
    X_init = np.random.randn(30, 3)
    X_batch = np.random.randn(15, 3)

    icosvm = ICOSVM(nu=0.15, eta=0.6, kernel="linear")
    icosvm.fit(X_init)
    icosvm.learn_batch(X_batch)

    assert len(icosvm.X_buffer_) == 45
    assert len(icosvm.alpha_) == 45


def test_icosvm_unlearn_and_sliding_window():
    np.random.seed(42)
    X = np.random.randn(30, 2)

    # Max samples = 25 enforces a sliding window
    icosvm = ICOSVM(nu=0.1, eta=0.5, max_samples=25)
    icosvm.fit(X[:20])

    assert len(icosvm.X_buffer_) == 20

    # Add 10 points
    icosvm.learn_batch(X[20:30])
    assert len(icosvm.X_buffer_) <= 25

    # Manual unlearn
    prev_len = len(icosvm.X_buffer_)
    icosvm.unlearn_one(0)
    assert len(icosvm.X_buffer_) == prev_len - 1

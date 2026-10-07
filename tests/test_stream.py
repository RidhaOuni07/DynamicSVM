"""
Unit tests for stream, sliding window, and drift detection modules.
"""

import numpy as np
from dynamicsvm.stream.windowed import SlidingWindowSVM
from dynamicsvm.stream.drift import DDMDriftDetector, DriftAwareSVM


def test_sliding_window_incremental_and_decremental():
    # Stream of points
    np.random.seed(42)
    X_init = np.array([[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]])
    y_init = np.array([1, 1, -1, -1])

    sw_svm = SlidingWindowSVM(window_size=6, kernel="linear", C=1.0)
    sw_svm.fit(X_init, y_init)

    assert len(sw_svm._buffer_X) == 4

    # Incremental update
    sw_svm.learn_one(np.array([3.0, 3.0]), 1)
    assert len(sw_svm._buffer_X) == 5

    # Decremental update
    sw_svm.forget_one(0)
    assert len(sw_svm._buffer_X) == 4


def test_drift_detector():
    detector = DDMDriftDetector(min_instances=10, warning_level=1.5, drift_level=2.5)

    # Stationary low error phase
    for _ in range(20):
        detector.update(error=False)

    assert not detector.in_drift_zone

    # Sudden high error stream (drift)
    drift_seen = False
    for _ in range(30):
        if detector.update(error=True):
            drift_seen = True
            break

    assert drift_seen

"""
Unit tests for Online Covariance Tracking, Latent Drift Detection, and Obsolescence Scoring.
"""

import numpy as np
import pytest

from dynamicsvm.stream.covariance import (
    LatentCovarianceDriftDetector,
    ObsolescenceScorer,
    OnlineCovarianceTracker,
)


def test_online_covariance_tracker():
    rng = np.random.RandomState(42)
    # Regime 1: Mean [0, 0], Std [1, 1]
    Z_regime1 = rng.randn(50, 2)
    tracker = OnlineCovarianceTracker(lambda_base=0.90, lambda_min=0.40)
    tracker.fit(Z_regime1)

    assert tracker.mean_ is not None
    assert tracker.cov_ is not None
    assert tracker.cov_.shape == (2, 2)
    np.testing.assert_allclose(tracker.mean_, np.mean(Z_regime1, axis=0), atol=1e-5)

    # Regime 2: Mean [5, 5] (strong drift)
    Z_regime2 = rng.randn(10, 2) + 5.0
    D_t, lambda_t = tracker.update(Z_regime2)
    
    assert D_t > 0.0
    # Drift should lower lambda below lambda_base
    assert lambda_t < tracker.lambda_base
    assert tracker.n_updates_ == 2

    # Mahalanobis distance test
    m_dist = tracker.mahalanobis_distance(tracker.mean_.reshape(1, -1))
    np.testing.assert_allclose(m_dist, [0.0], atol=1e-3)


def test_latent_covariance_drift_detector():
    detector = LatentCovarianceDriftDetector(
        warning_threshold=2.0,
        drift_threshold=5.0,
        min_instances=3,
    )

    # Warm-up instances
    is_drift, is_warn = detector.update(10.0)
    assert not is_drift and not is_warn
    detector.update(10.0)
    detector.update(10.0)

    # Now past min_instances
    is_drift, is_warn = detector.update(1.0)
    assert not is_drift and not is_warn

    # Trigger warning
    is_drift, is_warn = detector.update(3.0)
    assert is_warn and not is_drift
    assert detector.in_warning_zone_

    # Trigger drift
    is_drift, is_warn = detector.update(6.0)
    assert is_drift and not is_warn
    assert detector.in_drift_zone_
    assert detector.drift_count_ == 1


def test_obsolescence_scorer():
    rng = np.random.RandomState(42)
    Z = rng.randn(10, 2)
    arrival_times = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    tracker = OnlineCovarianceTracker()
    tracker.fit(Z)

    scorer = ObsolescenceScorer(
        lambda_age=0.4,
        lambda_temporal=0.3,
        lambda_dist=0.3,
        sv_protection_factor=0.5,
    )

    # Sample 0 is oldest; sample 9 is newest
    is_sv = np.array([False] * 10)
    scores = scorer.score_samples(
        Z_stored=Z,
        arrival_times=arrival_times,
        current_time=10,
        tracker=tracker,
        is_support_vector=is_sv,
    )

    assert scores.shape == (10,)
    # Sample 0 should have higher age/temporal obsolescence than sample 9
    assert scores[0] > scores[9]

    # Test support vector protection
    is_sv_protected = np.array([True] + [False] * 9)
    scores_protected = scorer.score_samples(
        Z_stored=Z,
        arrival_times=arrival_times,
        current_time=10,
        tracker=tracker,
        is_support_vector=is_sv_protected,
    )
    assert scores_protected[0] < scores[0]

    # Test top-k obsolescence selection
    obsolete_indices = scorer.select_obsolete_indices(scores, n_to_remove=2)
    assert len(obsolete_indices) == 2

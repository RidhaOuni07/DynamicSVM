"""
Unit tests for CovarianceAwareSVC.
"""

import numpy as np
import pytest

from dynamicsvm.models.covariance_svc import CovarianceAwareSVC


def test_covariance_aware_svc_fit_and_predict():
    rng = np.random.RandomState(42)
    # Binary dataset
    X_pos = rng.randn(25, 2) + np.array([2.0, 2.0])
    X_neg = rng.randn(25, 2) + np.array([-2.0, -2.0])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 25 + [-1] * 25)

    clf = CovarianceAwareSVC(C=1.0, eta=0.6, kernel="rbf", gamma="scale", solver="smo")
    clf.fit(X, y)

    assert clf.is_fitted
    assert len(clf.support_vectors_) > 0
    assert clf.dual_coef_ is not None

    preds = clf.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.90

    proba = clf.predict_proba(X)
    assert proba.shape == (50, 2)
    np.testing.assert_allclose(np.sum(proba, axis=1), np.ones(50), atol=1e-5)


def test_covariance_aware_svc_qp_solver():
    rng = np.random.RandomState(42)
    X_pos = rng.randn(15, 2) + np.array([1.5, 1.5])
    X_neg = rng.randn(15, 2) + np.array([-1.5, -1.5])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 15 + [0] * 15)

    clf = CovarianceAwareSVC(C=1.0, eta=0.5, kernel="linear", solver="qp")
    clf.fit(X, y)

    assert clf.is_fitted
    preds = clf.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.90

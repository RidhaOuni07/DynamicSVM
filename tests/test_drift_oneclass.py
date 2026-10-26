import numpy as np
import pytest

from dynamicsvm.encoders import AutoEncoder
from dynamicsvm.encoders.base import BaseEncoder
from dynamicsvm.models.cosvm import COSVM
from dynamicsvm.stream.drift import DriftDetector
from dynamicsvm.stream.drift_oneclass import DriftAwareOneClassSVM
from dynamicsvm.stream.icosvm import ICOSVM
from dynamicsvm.stream.idcosvm import IncrementalDecrementalCOSVM


class _ThirdObservationDriftDetector(DriftDetector):
    def __init__(self):
        self.reset()

    def update(self, error: bool) -> bool:
        self.n_updates += 1
        return self.n_updates == 3

    def reset(self) -> None:
        self.n_updates = 0


class _CountingEncoder(BaseEncoder):
    def __init__(self):
        super().__init__()
        self.fit_calls = 0
        self.mean_ = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.latent_dim_ = 2
        self.fit_calls += 1
        return self

    def encode(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, :2] - self.mean_[:2]


class _NeverDriftDetector(DriftDetector):
    def update(self, error: bool) -> bool:
        return False

    def reset(self) -> None:
        pass


@pytest.mark.parametrize(
    "estimator",
    [
        COSVM(nu=0.1, eta=0.5, kernel="rbf", gamma=0.5),
        ICOSVM(nu=0.1, eta=0.5, kernel="rbf", gamma=0.5),
        IncrementalDecrementalCOSVM(nu=0.1, eta=0.5, kernel="rbf", gamma=0.5),
    ],
)
def test_drift_aware_wrapper_resets_each_cosvm_variant(estimator):
    rng = np.random.RandomState(42)
    X_initial = rng.normal(size=(12, 2))
    X_shifted = rng.normal(loc=4.0, size=(3, 2))
    model = DriftAwareOneClassSVM(
        estimator,
        window_size=8,
        drift_detector=_ThirdObservationDriftDetector(),
    ).fit(X_initial)

    drift_flags = [model.learn_one(x) for x in X_shifted]

    assert drift_flags == [False, False, True]
    assert model.n_drifts_detected_ == 1
    assert model.drift_events_[0]["recent_window_size"] == 3
    adapted_buffer = (
        model._active_samples
        if isinstance(estimator, COSVM)
        else model.base_estimator.X_buffer_
    )
    np.testing.assert_allclose(np.asarray(adapted_buffer), X_shifted)
    assert model.predict(X_shifted).shape == (3,)
    assert model.decision_function(X_shifted).shape == (3,)


def test_drift_aware_wrapper_rejects_invalid_estimator_and_window():
    with pytest.raises(TypeError, match="base_estimator"):
        DriftAwareOneClassSVM(object())
    with pytest.raises(ValueError, match="window_size"):
        DriftAwareOneClassSVM(COSVM(), window_size=1)


def test_default_ddm_detects_shift_from_initial_anomaly_baseline():
    from dynamicsvm.stream.drift import DDMDriftDetector

    rng = np.random.RandomState(0)
    X_initial = rng.normal(size=(40, 2))
    X_shifted = rng.normal(loc=5.0, scale=0.2, size=(30, 2))
    model = DriftAwareOneClassSVM(
        COSVM(nu=0.1, eta=0.5, kernel="rbf", gamma=0.5),
        window_size=30,
        drift_detector=DDMDriftDetector(min_instances=5, min_errors=3),
    ).fit(X_initial)

    model.learn_many(X_shifted)

    assert model.n_drifts_detected_ > 0
    assert len(model.drift_events_) == model.n_drifts_detected_
    assert len(model._active_samples) <= model.window_size


@pytest.mark.parametrize(
    "estimator_type",
    [COSVM, ICOSVM, IncrementalDecrementalCOSVM],
)
def test_autoencoder_stacks_with_each_cosvm_variant(estimator_type):
    rng = np.random.RandomState(7)
    X_initial = rng.normal(size=(16, 3))
    encoder = AutoEncoder(
        latent_dim=2,
        hidden_dims=(4,),
        max_iter=5,
        batch_size=8,
        random_state=7,
    )
    model = DriftAwareOneClassSVM(
        estimator_type(nu=0.1, eta=0.5, kernel="rbf", gamma=0.5),
        window_size=10,
        encoder=encoder,
    ).fit(X_initial)

    assert not model.encoder.is_frozen_
    assert model.encoder.latent_dim_ == 2
    assert model.predict(X_initial[:4]).shape == (4,)
    assert model.decision_function(X_initial[:4]).shape == (4,)

    model.learn_one(rng.normal(size=3))
    assert model.predict(X_initial[:2]).shape == (2,)


@pytest.mark.parametrize(
    "estimator_type",
    [COSVM, ICOSVM, IncrementalDecrementalCOSVM],
)
def test_encoder_is_refit_on_active_raw_window_after_accepted_sample(estimator_type):
    rng = np.random.RandomState(11)
    X_initial = rng.normal(size=(20, 3))
    encoder = _CountingEncoder()
    model = DriftAwareOneClassSVM(
        estimator_type(nu=0.1, eta=0.5, kernel="rbf", gamma=0.5),
        window_size=12,
        drift_detector=_NeverDriftDetector(),
        encoder=encoder,
    ).fit(X_initial)

    encoded_active = encoder.encode(np.asarray(model._active_raw_samples))
    accepted = np.flatnonzero(model.base_estimator.predict(encoded_active) == 1)
    assert len(accepted) > 0
    sample = np.asarray(model._active_raw_samples)[accepted[0]]
    fit_calls_before = encoder.fit_calls

    assert model.learn_one(sample) is False
    assert encoder.fit_calls == fit_calls_before + 1
    assert not encoder.is_frozen_
    assert len(model._active_raw_samples) == 12
    assert model.base_estimator.n_features_in_ == 2


def test_confirmed_drift_refits_encoder_from_recent_raw_samples():
    rng = np.random.RandomState(12)
    X_initial = rng.normal(size=(20, 3))
    X_stream = rng.normal(loc=3.0, size=(3, 3))
    encoder = _CountingEncoder()
    model = DriftAwareOneClassSVM(
        COSVM(nu=0.1, eta=0.5, kernel="rbf", gamma=0.5),
        window_size=12,
        drift_detector=_ThirdObservationDriftDetector(),
        encoder=encoder,
    ).fit(X_initial)

    model.learn_many(X_stream[:2])
    fit_calls_before_drift = encoder.fit_calls
    assert model.learn_one(X_stream[2]) is True

    assert encoder.fit_calls == fit_calls_before_drift + 1
    assert len(model._active_raw_samples) == 3
    np.testing.assert_allclose(np.asarray(model._active_raw_samples), X_stream)
    np.testing.assert_allclose(encoder.mean_, np.mean(X_stream, axis=0))

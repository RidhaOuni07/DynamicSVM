"""Drift adaptation for batch and streaming one-class SVM estimators."""

from collections import deque
from typing import Deque, Dict, List, Optional, Union

import numpy as np

from dynamicsvm.encoders.base import BaseEncoder
from dynamicsvm.models.cosvm import COSVM
from dynamicsvm.stream.drift import DDMDriftDetector, DriftDetector
from dynamicsvm.stream.icosvm import ICOSVM
from dynamicsvm.stream.idcosvm import IncrementalDecrementalCOSVM
from dynamicsvm.utils.validation import check_array, check_is_fitted


OneClassEstimator = Union[COSVM, ICOSVM, IncrementalDecrementalCOSVM]


class DriftAwareOneClassSVM:
    """Wrap a COSVM-family estimator with anomaly-rate drift detection.

    A point rejected by the current one-class model is treated as an online
    error for the drift detector and is not added to the model unless a drift
    is confirmed. On confirmed drift, the estimator is refit on the recent
    input window, allowing a persistent new normal regime to become the new
    reference distribution.

    This is a detector-and-reset wrapper, not a new COSVM optimization
    algorithm. Its drift signal is the one-class anomaly decision, so isolated
    outliers and genuine distribution changes are not intrinsically
    distinguishable.

    Parameters
    ----------
    base_estimator : COSVM, ICOSVM, or IncrementalDecrementalCOSVM
        One-class model to adapt. Fit this wrapper before calling ``learn_one``.
    encoder : BaseEncoder, optional
        Optional adaptive representation encoder. It is fitted on the active
        raw-input window, then that entire window is re-encoded and the
        one-class model is refitted after each accepted sample or drift event.
        This preserves consistent latent coordinates but uses batch refits.
    window_size : int, default=200
        Maximum retained sample count. Also determines the recent input
        window used to refit after a detected drift.
    drift_detector : DriftDetector, optional
        Binary-error drift detector. Defaults to ``DDMDriftDetector``.
    """

    def __init__(
        self,
        base_estimator: OneClassEstimator,
        window_size: int = 200,
        drift_detector: Optional[DriftDetector] = None,
        encoder: Optional[BaseEncoder] = None,
    ):
        if not isinstance(
            base_estimator,
            (COSVM, ICOSVM, IncrementalDecrementalCOSVM),
        ):
            raise TypeError(
                "base_estimator must be COSVM, ICOSVM, or "
                "IncrementalDecrementalCOSVM."
            )
        if window_size < 2:
            raise ValueError("window_size must be at least 2.")

        self.base_estimator = base_estimator
        self.window_size = int(window_size)
        self.drift_detector = drift_detector or DDMDriftDetector()
        self.encoder = encoder
        self.n_drifts_detected_: int = 0
        self.n_samples_seen_: int = 0
        self.drift_events_: List[Dict[str, int]] = []
        self._active_raw_samples: Deque[np.ndarray] = deque(maxlen=self.window_size)
        self._active_samples: Deque[np.ndarray] = deque(maxlen=self.window_size)
        self._recent_stream: Deque[np.ndarray] = deque(maxlen=self.window_size)
        self.n_features_in_: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        return self.n_features_in_ is not None

    def fit(self, X: np.ndarray) -> "DriftAwareOneClassSVM":
        X_arr = check_array(X)
        if len(X_arr) < 2:
            raise ValueError("At least two samples are required to initialize the wrapper.")

        self.n_features_in_ = X_arr.shape[1]
        if self.encoder is None:
            X_model = X_arr
        else:
            self.encoder.unfreeze()
            self.encoder.fit(X_arr[-self.window_size :])
            X_model = self._encode(X_arr)

        if len(X_model) < 2:
            raise ValueError("The encoder must produce at least two training samples.")

        initial_samples = X_model[-self.window_size :]
        initial_raw_samples = X_arr[-self.window_size :]
        self._active_raw_samples = deque(
            (sample.copy() for sample in initial_raw_samples),
            maxlen=self.window_size,
        )
        self._active_samples = deque(
            (sample.copy() for sample in initial_samples),
            maxlen=self.window_size,
        )
        self._recent_stream = deque(maxlen=self.window_size)
        self.base_estimator.fit(np.asarray(self._active_samples))
        self._sync_active_samples()
        self._prime_drift_detector()
        self.n_drifts_detected_ = 0
        self.n_samples_seen_ = len(X_arr)
        self.drift_events_ = []
        return self

    def _encode(self, X: np.ndarray) -> np.ndarray:
        X_arr = check_array(X, n_features=self.n_features_in_)
        if self.encoder is None:
            return X_arr

        Z = check_array(self.encoder.encode(X_arr))
        if len(Z) != len(X_arr):
            raise ValueError("The encoder must preserve the number of input samples.")
        return Z

    def _refit_encoder_and_estimator(self) -> None:
        raw_samples = np.asarray(self._active_raw_samples)
        if len(raw_samples) < 2:
            raise RuntimeError(
                "At least two active raw samples are required to refit the encoder."
            )
        if self.encoder is None:
            encoded_samples = raw_samples
        else:
            self.encoder.unfreeze()
            self.encoder.fit(raw_samples)
            encoded_samples = self._encode(raw_samples)
        self.base_estimator.fit(encoded_samples)
        self._sync_active_samples()

    def _fit_recent_window(self) -> None:
        raw_samples = np.asarray(self._recent_stream)
        if len(raw_samples) < 2:
            raise RuntimeError(
                "A confirmed drift requires at least two recent samples to refit."
            )
        self._active_raw_samples = deque(
            (sample.copy() for sample in raw_samples),
            maxlen=self.window_size,
        )
        self._refit_encoder_and_estimator()

    def _sync_active_samples(self) -> None:
        if isinstance(self.base_estimator, COSVM):
            samples = self.base_estimator.X_train_
        else:
            samples = self.base_estimator.X_buffer_
        self._active_samples = deque(
            (sample.copy() for sample in samples),
            maxlen=self.window_size,
        )

    def _prime_drift_detector(self) -> None:
        self.drift_detector.reset()
        if not isinstance(self.drift_detector, DDMDriftDetector):
            return
        initial_errors = self.base_estimator.predict(
            np.asarray(self._active_samples)
        ) == -1
        for error in initial_errors:
            self.drift_detector.update(bool(error))

    def _learn_inlier(self, x: np.ndarray) -> None:
        if self.encoder is not None:
            self._active_raw_samples.append(x.copy())
            self._refit_encoder_and_estimator()
            return

        if isinstance(self.base_estimator, COSVM):
            self._active_raw_samples.append(x.copy())
            self._active_samples.append(x.copy())
            self.base_estimator.fit(np.asarray(self._active_samples))
            return

        self.base_estimator.learn_one(x)
        if len(self.base_estimator.X_buffer_) > self.window_size:
            self.base_estimator.unlearn_one(0)
        self._active_raw_samples = deque(
            (sample.copy() for sample in self.base_estimator.X_buffer_),
            maxlen=self.window_size,
        )
        self._active_samples = deque(
            (sample.copy() for sample in self.base_estimator.X_buffer_),
            maxlen=self.window_size,
        )

    def learn_one(self, x: np.ndarray) -> bool:
        """Process one unlabeled sample and return whether drift was detected."""
        check_is_fitted(self, attributes=["n_features_in_"])
        x_arr = check_array(np.asarray(x, dtype=float).reshape(1, -1), n_features=self.n_features_in_)
        raw_sample = x_arr[0]
        latent_sample = self._encode(x_arr)
        is_anomaly = bool(self.base_estimator.predict(latent_sample)[0] == -1)
        drift_detected = self.drift_detector.update(is_anomaly)

        self.n_samples_seen_ += 1
        self._recent_stream.append(raw_sample.copy())

        if drift_detected:
            old_buffer_size = len(self._active_samples)
            self.n_drifts_detected_ += 1
            self.drift_events_.append(
                {
                    "sample_index": self.n_samples_seen_ - 1,
                    "old_buffer_size": old_buffer_size,
                    "recent_window_size": len(self._recent_stream),
                }
            )
            self._fit_recent_window()
            self._prime_drift_detector()
            return True

        if not is_anomaly:
            self._learn_inlier(raw_sample)
        return False

    def learn_many(self, X: np.ndarray) -> "DriftAwareOneClassSVM":
        """Process samples in order; returns this wrapper."""
        check_is_fitted(self, attributes=["n_features_in_"])
        X_arr = check_array(X, n_features=self.n_features_in_)
        for sample in X_arr:
            self.learn_one(sample)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["n_features_in_"])
        return self.base_estimator.predict(self._encode(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["n_features_in_"])
        return self.base_estimator.decision_function(self._encode(X))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["n_features_in_"])
        return self.base_estimator.score_samples(self._encode(X))

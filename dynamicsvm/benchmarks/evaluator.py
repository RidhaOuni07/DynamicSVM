"""Evaluation helpers for one-class anomaly detectors."""

import time
from typing import Any, Dict

import numpy as np


class AnomalyBenchmarkEvaluator:
    """Evaluate a one-class model on normal and anomalous test samples."""

    @staticmethod
    def evaluate(
        model: Any,
        X_train_normal: np.ndarray,
        X_test_normal: np.ndarray,
        X_test_anomalies: np.ndarray,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        model.fit(X_train_normal)
        fit_time_ms = (time.perf_counter() - start) * 1000.0

        pred_normal = model.predict(X_test_normal)
        target_detection_rate = float(np.mean(pred_normal == 1))
        pred_anomalies = model.predict(X_test_anomalies)
        anomaly_detection_rate = float(np.mean(pred_anomalies == -1))
        support_vectors = getattr(model, "support_vectors_", None)
        n_support_vectors = len(support_vectors) if support_vectors is not None else 0

        return {
            "target_detection_rate": target_detection_rate,
            "anomaly_detection_rate": anomaly_detection_rate,
            "balanced_accuracy": 0.5
            * (target_detection_rate + anomaly_detection_rate),
            "fit_time_ms": fit_time_ms,
            "n_support_vectors": n_support_vectors,
        }

"""
Evaluation protocols and metrics for streaming, batch, and anomaly detection benchmarks.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from dynamicsvm.utils.metrics import accuracy_score, confusion_matrix


class PrequentialEvaluator:
    r"""
    Prequential (Test-Then-Train) Streaming Evaluation Protocol.

    For each incoming streaming instance $(x_t, y_t)$:
    1. Test: Predict label $\hat{y}_t = \text{predict}(x_t)$ and test margin.
    2. Record correctness $I(\hat{y}_t == y_t)$ and update sliding/cumulative metrics.
    3. Train: Incrementally update model via `learn_one(x_t, y_t)`.
    4. Record execution latency, memory footprint, and support vector sparsity.

    Parameters
    ----------
    window_size : int, default=50
        Size of sliding window for rolling accuracy calculation.
    track_sv_history : bool, default=True
        Whether to record support vector counts at each streaming step.
    """

    def __init__(self, window_size: int = 50, track_sv_history: bool = True):
        self.window_size = int(window_size)
        self.track_sv_history = track_sv_history

    def evaluate(
        self,
        model: Any,
        X_stream: np.ndarray,
        y_stream: np.ndarray,
        initial_fit_samples: int = 0,
    ) -> Dict[str, Any]:
        """
        Run prequential evaluation over a stream of samples.

        Parameters
        ----------
        model : Any
            Streaming SVM model supporting `learn_one(x, y)` and `predict(x)`.
        X_stream : np.ndarray of shape (n_samples, n_features)
        y_stream : np.ndarray of shape (n_samples,)
        initial_fit_samples : int, default=0
            Number of initial samples to fit in batch before streaming starts.

        Returns
        -------
        results : Dict[str, Any]
            Dictionary containing cumulative accuracy, sliding window accuracy curve,
            streaming latency, throughput, and support vector dynamics.
        """
        n_samples = len(X_stream)
        stream_start_idx = 0

        if initial_fit_samples > 0:
            fit_count = min(initial_fit_samples, n_samples)
            if hasattr(model, "fit"):
                model.fit(X_stream[:fit_count], y_stream[:fit_count])
            else:
                for i in range(fit_count):
                    model.learn_one(X_stream[i], y_stream[i])
            stream_start_idx = fit_count

        predictions = []
        ground_truth = []
        step_latencies = []
        rolling_accuracies = []
        sv_counts = []

        total_correct = 0
        window_history: List[int] = []

        t_start_total = time.perf_counter()

        for t in range(stream_start_idx, n_samples):
            xt = X_stream[t]
            yt = y_stream[t]

            # 1. Test phase (latency measured)
            t0 = time.perf_counter()
            pred = model.predict(xt.reshape(1, -1))[0]
            
            # 2. Train phase
            model.learn_one(xt, yt)
            t1 = time.perf_counter()

            step_latencies.append(t1 - t0)

            is_correct = int(pred == yt)
            total_correct += is_correct
            predictions.append(pred)
            ground_truth.append(yt)

            window_history.append(is_correct)
            if len(window_history) > self.window_size:
                window_history.pop(0)

            rolling_acc = float(np.mean(window_history))
            rolling_accuracies.append(rolling_acc)

            # 3. Track SV state
            if self.track_sv_history:
                if hasattr(model, "support_vectors_") and model.support_vectors_ is not None:
                    sv_counts.append(len(model.support_vectors_))
                elif hasattr(model, "get_model_state"):
                    state = model.get_model_state()
                    sv_counts.append(state.get("n_support_vectors", state.get("n_sv", 0)))
                elif hasattr(model, "_buffer_X"):
                    sv_counts.append(len(model._buffer_X))
                else:
                    sv_counts.append(0)

        t_end_total = time.perf_counter()
        total_streamed = len(predictions)
        cum_acc = float(total_correct / max(1, total_streamed))
        avg_latency_ms = float(np.mean(step_latencies) * 1000.0) if step_latencies else 0.0
        throughput = float(total_streamed / max(1e-6, t_end_total - t_start_total))

        return {
            "n_samples_evaluated": total_streamed,
            "cumulative_accuracy": cum_acc,
            "rolling_accuracy_curve": np.array(rolling_accuracies),
            "mean_rolling_accuracy": float(np.mean(rolling_accuracies)) if rolling_accuracies else cum_acc,
            "final_rolling_accuracy": rolling_accuracies[-1] if rolling_accuracies else cum_acc,
            "mean_step_latency_ms": avg_latency_ms,
            "p95_step_latency_ms": float(np.percentile(step_latencies, 95) * 1000.0) if step_latencies else 0.0,
            "throughput_samples_per_sec": throughput,
            "sv_counts_curve": np.array(sv_counts),
            "final_sv_count": sv_counts[-1] if sv_counts else 0,
            "predictions": np.array(predictions),
            "ground_truth": np.array(ground_truth),
        }


class BatchBenchmarkEvaluator:
    """Benchmark evaluator for batch models and optimization solvers."""

    @staticmethod
    def evaluate(
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate a batch model across fit time, predict time, accuracy, and SV counts.
        """
        t0_fit = time.perf_counter()
        model.fit(X_train, y_train)
        t1_fit = time.perf_counter()
        fit_time_ms = (t1_fit - t0_fit) * 1000.0

        t0_pred = time.perf_counter()
        y_pred = model.predict(X_test)
        t1_pred = time.perf_counter()
        predict_time_ms = (t1_pred - t0_pred) * 1000.0

        acc_test = accuracy_score(y_test, y_pred)
        acc_train = accuracy_score(y_train, model.predict(X_train))

        n_sv = len(model.support_vectors_) if hasattr(model, "support_vectors_") and model.support_vectors_ is not None else 0
        sv_ratio = float(n_sv / len(X_train)) if len(X_train) > 0 else 0.0

        return {
            "train_accuracy": acc_train,
            "test_accuracy": acc_test,
            "fit_time_ms": fit_time_ms,
            "predict_time_ms": predict_time_ms,
            "n_support_vectors": n_sv,
            "sv_ratio": sv_ratio,
        }


class AnomalyBenchmarkEvaluator:
    """Benchmark evaluator for One-Class SVMs and Anomaly Detectors."""

    @staticmethod
    def evaluate(
        model: Any,
        X_train_normal: np.ndarray,
        X_test_normal: np.ndarray,
        X_test_anomalies: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate one-class anomaly detector on normal test targets and anomalies.
        """
        t0_fit = time.perf_counter()
        model.fit(X_train_normal)
        t1_fit = time.perf_counter()
        fit_time_ms = (t1_fit - t0_fit) * 1000.0

        # Normal test (+1 target)
        pred_normal = model.predict(X_test_normal)
        target_accuracy = float(np.mean(pred_normal == 1))  # True Positive Rate

        # Anomaly test (-1 target)
        pred_anomalies = model.predict(X_test_anomalies)
        anomaly_accuracy = float(np.mean(pred_anomalies == -1))  # True Negative Rate

        # Balanced Accuracy
        balanced_accuracy = 0.5 * (target_accuracy + anomaly_accuracy)

        n_sv = len(model.support_vectors_) if hasattr(model, "support_vectors_") and model.support_vectors_ is not None else 0

        return {
            "target_detection_rate": target_accuracy,
            "anomaly_detection_rate": anomaly_accuracy,
            "balanced_accuracy": balanced_accuracy,
            "fit_time_ms": fit_time_ms,
            "n_support_vectors": n_sv,
        }

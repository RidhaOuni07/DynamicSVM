"""
Comparative benchmark suites and runners for DynamicSVM.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from dynamicsvm.benchmarks.datasets import (
    make_sea_stream,
    make_rotating_hyperplane_stream,
    make_gaussian_drift_stream,
    make_anisotropic_anomaly_dataset,
    make_benchmark_moons,
    make_benchmark_circles,
)
from dynamicsvm.benchmarks.evaluator import (
    PrequentialEvaluator,
    BatchBenchmarkEvaluator,
    AnomalyBenchmarkEvaluator,
)


def benchmark_streaming_suite(
    models: Dict[str, Any],
    dataset_name: str = "sea",
    n_samples: int = 600,
    window_size: int = 50,
    initial_fit_samples: int = 30,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run prequential comparative streaming benchmark across multiple models.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary mapping model names to model instances.
    dataset_name : {'sea', 'hyperplane', 'gaussian_drift'}
    n_samples : int, default=600
    window_size : int, default=50
    initial_fit_samples : int, default=30
    random_state : int, default=42

    Returns
    -------
    results : Dict[str, Any]
        Comparison dictionary containing per-model metrics and ranking summaries.
    """
    if dataset_name == "sea":
        X, y, drift_points = make_sea_stream(n_samples=n_samples, random_state=random_state)
    elif dataset_name == "hyperplane":
        X, y = make_rotating_hyperplane_stream(n_samples=n_samples, random_state=random_state)
        drift_points = []
    elif dataset_name == "gaussian_drift":
        X, y, drift_points = make_gaussian_drift_stream(n_samples=n_samples, random_state=random_state)
    else:
        raise ValueError(f"Unknown streaming dataset {dataset_name}")

    evaluator = PrequentialEvaluator(window_size=window_size)
    model_results = {}

    for name, model in models.items():
        res = evaluator.evaluate(
            model=model,
            X_stream=X,
            y_stream=y,
            initial_fit_samples=initial_fit_samples,
        )
        model_results[name] = res

    # Summary table
    summary = []
    for name, res in model_results.items():
        summary.append({
            "model": name,
            "cumulative_accuracy": res["cumulative_accuracy"],
            "final_rolling_accuracy": res["final_rolling_accuracy"],
            "mean_step_latency_ms": res["mean_step_latency_ms"],
            "throughput_samples_per_sec": res["throughput_samples_per_sec"],
            "final_sv_count": res["final_sv_count"],
        })

    # Sort by cumulative accuracy descending
    summary.sort(key=lambda s: s["cumulative_accuracy"], reverse=True)

    return {
        "dataset": dataset_name,
        "n_samples": n_samples,
        "drift_points": drift_points,
        "models": model_results,
        "summary": summary,
    }


def benchmark_anomaly_suite(
    models: Dict[str, Any],
    n_normal: int = 150,
    n_anomalies: int = 30,
    ratio: float = 10.0,
    dimension: int = 2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run comparative anomaly detection benchmark on anisotropic low-variance directions.

    Parameters
    ----------
    models : Dict[str, Any]
    n_normal : int, default=150
    n_anomalies : int, default=30
    ratio : float, default=10.0
    dimension : int, default=2
    random_state : int, default=42

    Returns
    -------
    results : Dict[str, Any]
    """
    X_train_norm, X_test_norm, X_test_anom = make_anisotropic_anomaly_dataset(
        n_normal=n_normal,
        n_anomalies=n_anomalies,
        ratio=ratio,
        dimension=dimension,
        random_state=random_state,
    )

    evaluator = AnomalyBenchmarkEvaluator()
    model_results = {}

    for name, model in models.items():
        res = evaluator.evaluate(
            model=model,
            X_train_normal=X_train_norm,
            X_test_normal=X_test_norm,
            X_test_anomalies=X_test_anom,
        )
        model_results[name] = res

    summary = []
    for name, res in model_results.items():
        summary.append({
            "model": name,
            "target_detection_rate": res["target_detection_rate"],
            "anomaly_detection_rate": res["anomaly_detection_rate"],
            "balanced_accuracy": res["balanced_accuracy"],
            "fit_time_ms": res["fit_time_ms"],
            "n_support_vectors": res["n_support_vectors"],
        })

    summary.sort(key=lambda s: s["balanced_accuracy"], reverse=True)

    return {
        "dataset": "anisotropic_low_variance",
        "ratio": ratio,
        "dimension": dimension,
        "models": model_results,
        "summary": summary,
    }


def benchmark_solvers_suite(
    solvers: Dict[str, Any],
    dataset_name: str = "moons",
    n_samples: int = 200,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compare optimization solvers (SMO, QP, Pegasos, LASVM, CovarianceSVC) on static benchmarks.
    """
    if dataset_name == "moons":
        X, y = make_benchmark_moons(n_samples=n_samples, random_state=random_state)
    elif dataset_name == "circles":
        X, y = make_benchmark_circles(n_samples=n_samples, random_state=random_state)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    # Split 75% train / 25% test
    n_train = int(len(X) * 0.75)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    evaluator = BatchBenchmarkEvaluator()
    solver_results = {}

    for name, model in solvers.items():
        res = evaluator.evaluate(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        solver_results[name] = res

    summary = []
    for name, res in solver_results.items():
        summary.append({
            "solver": name,
            "train_accuracy": res["train_accuracy"],
            "test_accuracy": res["test_accuracy"],
            "fit_time_ms": res["fit_time_ms"],
            "predict_time_ms": res["predict_time_ms"],
            "n_support_vectors": res["n_support_vectors"],
            "sv_ratio": res["sv_ratio"],
        })

    summary.sort(key=lambda s: s["test_accuracy"], reverse=True)

    return {
        "dataset": dataset_name,
        "n_samples": n_samples,
        "solvers": solver_results,
        "summary": summary,
    }

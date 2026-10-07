"""
Unit and integration tests for DynamicSVM Benchmarking Suite.
"""

import numpy as np
import pytest

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
from dynamicsvm.benchmarks.runner import (
    benchmark_streaming_suite,
    benchmark_anomaly_suite,
    benchmark_solvers_suite,
)
from dynamicsvm.models.svc import SVC
from dynamicsvm.models.covariance_svc import CovarianceAwareSVC
from dynamicsvm.models.cosvm import COSVM
from dynamicsvm.stream.icosvm import ICOSVM
from dynamicsvm.stream.idcosvm import IncrementalDecrementalCOSVM
from dynamicsvm.stream.adiabatic_svm import AdiabaticSVM
from dynamicsvm.solvers.lasvm import LASVM
from dynamicsvm.solvers.pegasos import PegasosSolver
from dynamicsvm.stream.windowed import SlidingWindowSVM
from dynamicsvm.stream.drift_covariance_svm import DriftAwareCovarianceSVM
from dynamicsvm.encoders.autoencoder import PCAEncoder


def test_make_sea_stream():
    n_samples = 400
    X, y, drift_points = make_sea_stream(n_samples=n_samples, noise=0.0, random_state=42)

    assert X.shape == (n_samples, 3)
    assert y.shape == (n_samples,)
    assert set(np.unique(y)).issubset({-1.0, 1.0})
    assert len(drift_points) == 3
    assert drift_points == [100, 200, 300]


def test_make_rotating_hyperplane_stream():
    n_samples = 300
    X, y = make_rotating_hyperplane_stream(n_samples=n_samples, n_features=4, drift_speed=0.01, random_state=42)

    assert X.shape == (n_samples, 4)
    assert y.shape == (n_samples,)
    assert set(np.unique(y)).issubset({-1.0, 1.0})


def test_make_gaussian_drift_stream():
    n_samples = 300
    X, y, drift_points = make_gaussian_drift_stream(n_samples=n_samples, n_features=5, n_drifts=2, random_state=42)

    assert X.shape == (n_samples, 5)
    assert y.shape == (n_samples,)
    assert len(drift_points) == 2


def test_make_anisotropic_anomaly_dataset():
    n_normal = 100
    n_anomalies = 20
    X_train_norm, X_test_norm, X_test_anom = make_anisotropic_anomaly_dataset(
        n_normal=n_normal,
        n_anomalies=n_anomalies,
        ratio=8.0,
        dimension=2,
        random_state=42,
    )

    assert X_train_norm.shape == (100, 2)
    assert X_test_norm.shape == (50, 2)
    assert X_test_anom.shape == (20, 2)


def test_make_benchmark_shapes():
    X_moons, y_moons = make_benchmark_moons(n_samples=100, random_state=42)
    assert X_moons.shape == (100, 2)
    assert len(y_moons) == 100

    X_circ, y_circ = make_benchmark_circles(n_samples=100, random_state=42)
    assert X_circ.shape == (100, 2)
    assert len(y_circ) == 100


def test_prequential_evaluator_streaming():
    X, y, _ = make_sea_stream(n_samples=120, noise=0.0, random_state=42)

    evaluator = PrequentialEvaluator(window_size=20)
    model = SlidingWindowSVM(window_size=30, kernel="linear", C=1.0)

    res = evaluator.evaluate(model, X, y, initial_fit_samples=20)

    assert res["n_samples_evaluated"] == 100
    assert 0.0 <= res["cumulative_accuracy"] <= 1.0
    assert len(res["rolling_accuracy_curve"]) == 100
    assert res["throughput_samples_per_sec"] > 0
    assert res["mean_step_latency_ms"] >= 0


def test_batch_benchmark_evaluator():
    X, y = make_benchmark_moons(n_samples=120, noise=0.05, random_state=42)
    n_train = 80
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    model = SVC(C=2.0, kernel="rbf", gamma=1.0, solver="smo")
    evaluator = BatchBenchmarkEvaluator()

    res = evaluator.evaluate(model, X_train, y_train, X_test, y_test)

    assert res["train_accuracy"] >= 0.85
    assert res["test_accuracy"] >= 0.85
    assert res["fit_time_ms"] > 0
    assert res["n_support_vectors"] > 0


def test_anomaly_benchmark_evaluator_cosvm_vs_baseline():
    X_train_norm, X_test_norm, X_test_anom = make_anisotropic_anomaly_dataset(
        n_normal=120,
        n_anomalies=25,
        ratio=10.0,
        random_state=42,
    )

    evaluator = AnomalyBenchmarkEvaluator()

    # Low-variance covariance-guided COSVM
    cosvm = COSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale")
    res_cosvm = evaluator.evaluate(cosvm, X_train_norm, X_test_norm, X_test_anom)

    assert res_cosvm["target_detection_rate"] >= 0.50
    assert res_cosvm["anomaly_detection_rate"] >= 0.80
    assert res_cosvm["balanced_accuracy"] >= 0.65


def test_benchmark_streaming_suite_runner():
    models = {
        "SlidingWindow": SlidingWindowSVM(window_size=25, kernel="rbf", gamma=0.5, C=1.0),
        "AdiabaticSVM": AdiabaticSVM(C=1.0, kernel="rbf", gamma=0.5, max_samples=30),
        "LASVM": LASVM(C=1.0, kernel="rbf", gamma=0.5, epochs=1, random_state=42),
    }

    results = benchmark_streaming_suite(
        models=models,
        dataset_name="sea",
        n_samples=100,
        window_size=20,
        initial_fit_samples=20,
        random_state=42,
    )

    assert results["dataset"] == "sea"
    assert len(results["summary"]) == 3
    assert all("cumulative_accuracy" in s for s in results["summary"])


def test_benchmark_anomaly_suite_runner():
    models = {
        "COSVM": COSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale"),
        "ICOSVM": ICOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale", max_samples=100),
        "IDCOSVM": IncrementalDecrementalCOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale"),
    }

    results = benchmark_anomaly_suite(
        models=models,
        n_normal=80,
        n_anomalies=20,
        ratio=8.0,
        random_state=42,
    )

    assert results["dataset"] == "anisotropic_low_variance"
    assert len(results["summary"]) == 3
    for s in results["summary"]:
        assert s["anomaly_detection_rate"] >= 0.70


def test_benchmark_solvers_suite_runner():
    solvers = {
        "SMO": SVC(C=1.0, kernel="rbf", gamma=1.0, solver="smo"),
        "QP": SVC(C=1.0, kernel="rbf", gamma=1.0, solver="qp"),
        "LASVM": LASVM(C=1.0, kernel="rbf", gamma=1.0, epochs=2, random_state=42),
    }

    results = benchmark_solvers_suite(
        solvers=solvers,
        dataset_name="moons",
        n_samples=100,
        random_state=42,
    )

    assert results["dataset"] == "moons"
    assert len(results["summary"]) == 3
    for s in results["summary"]:
        assert s["test_accuracy"] >= 0.80

import numpy as np

from dynamicsvm.benchmarks import (
    AnomalyBenchmarkEvaluator,
    make_anisotropic_anomaly_dataset,
    make_benchmark_circles,
    make_benchmark_moons,
)
from dynamicsvm.models.cosvm import COSVM


def test_oneclass_dataset_generators_return_expected_shapes():
    train, normal, anomalies = make_anisotropic_anomaly_dataset(
        n_normal=40,
        n_anomalies=8,
        dimension=3,
        random_state=4,
    )
    assert train.shape == (40, 3)
    assert normal.shape == (20, 3)
    assert anomalies.shape == (8, 3)

    moons, moon_labels = make_benchmark_moons(n_samples=30, random_state=4)
    circles, circle_labels = make_benchmark_circles(n_samples=30, random_state=4)
    assert moons.shape == circles.shape == (30, 2)
    assert set(np.unique(moon_labels)) == set(np.unique(circle_labels)) == {-1.0, 1.0}


def test_anomaly_benchmark_evaluator_accepts_cosvm():
    rng = np.random.RandomState(4)
    X_train = rng.normal(size=(40, 2))
    X_test_normal = rng.normal(size=(12, 2))
    X_test_anomalies = rng.normal(loc=5.0, size=(8, 2))

    result = AnomalyBenchmarkEvaluator.evaluate(
        COSVM(nu=0.1, eta=0.5, kernel="rbf", gamma=0.5),
        X_train,
        X_test_normal,
        X_test_anomalies,
    )

    assert 0.0 <= result["balanced_accuracy"] <= 1.0
    assert result["fit_time_ms"] >= 0.0
    assert result["n_support_vectors"] > 0

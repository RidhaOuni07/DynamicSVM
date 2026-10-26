"""Synthetic datasets for COSVM-family anomaly-detection experiments."""

from typing import Optional, Tuple

import numpy as np


def make_anisotropic_anomaly_dataset(
    n_normal: int = 150,
    n_anomalies: int = 30,
    ratio: float = 10.0,
    dimension: int = 2,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate normal samples and low-variance-direction anomalies."""
    rng = np.random.RandomState(random_state)
    theta = np.pi / 4.0

    if dimension == 2:
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )
    else:
        rotation, _ = np.linalg.qr(rng.randn(dimension, dimension))
    scales = np.ones(dimension)
    scales[0] = ratio

    train = (rng.randn(n_normal, dimension) * scales) @ rotation.T
    n_test_normal = max(20, n_normal // 2)
    test_normal = (rng.randn(n_test_normal, dimension) * scales) @ rotation.T

    anomaly = rng.randn(n_anomalies, dimension)
    anomaly[:, 0] = rng.uniform(
        -ratio * 0.5, ratio * 0.5, size=n_anomalies
    )
    anomaly[:, 1] = rng.choice([-1.0, 1.0], size=n_anomalies) * rng.uniform(
        3.0, 6.0, size=n_anomalies
    )
    test_anomalies = anomaly @ rotation.T
    return train, test_normal, test_anomalies


def make_benchmark_moons(
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two interleaving half-circles for one-class experiments."""
    rng = np.random.RandomState(random_state)
    n_per_class = n_samples // 2

    theta = np.linspace(0.0, np.pi, n_per_class)
    first = np.column_stack(
        (
            np.cos(theta) + rng.randn(n_per_class) * noise,
            np.sin(theta) + rng.randn(n_per_class) * noise,
        )
    )
    second = np.column_stack(
        (
            1.0 - np.cos(theta) + rng.randn(n_per_class) * noise,
            0.5 - np.sin(theta) + rng.randn(n_per_class) * noise,
        )
    )
    X = np.vstack((first, second))
    y = np.concatenate(
        (np.ones(n_per_class, dtype=float), -np.ones(n_per_class, dtype=float))
    )
    permutation = rng.permutation(len(X))
    return X[permutation], y[permutation]


def make_benchmark_circles(
    n_samples: int = 200,
    noise: float = 0.05,
    factor: float = 0.5,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate concentric circles for one-class experiments."""
    rng = np.random.RandomState(random_state)
    n_per_class = n_samples // 2

    theta_outer = rng.uniform(0.0, 2.0 * np.pi, n_per_class)
    outer = np.column_stack(
        (
            np.cos(theta_outer) + rng.randn(n_per_class) * noise,
            np.sin(theta_outer) + rng.randn(n_per_class) * noise,
        )
    )
    theta_inner = rng.uniform(0.0, 2.0 * np.pi, n_per_class)
    inner = np.column_stack(
        (
            factor * np.cos(theta_inner) + rng.randn(n_per_class) * noise,
            factor * np.sin(theta_inner) + rng.randn(n_per_class) * noise,
        )
    )
    X = np.vstack((outer, inner))
    y = np.concatenate(
        (np.ones(n_per_class, dtype=float), -np.ones(n_per_class, dtype=float))
    )
    permutation = rng.permutation(len(X))
    return X[permutation], y[permutation]

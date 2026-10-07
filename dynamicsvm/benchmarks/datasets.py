"""
Benchmark dataset generators for static, streaming, concept drift, and anomaly detection evaluations.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def make_sea_stream(
    n_samples: int = 1000,
    noise: float = 0.05,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    r"""
    Generate the classic SEA Concept Drift benchmark stream.

    The SEA generator consists of 3 continuous features $x_1, x_2, x_3 \in [0, 10]$,
    where $x_3$ is an irrelevant noise attribute. The classification threshold
    $x_1 + x_2 > \theta$ changes abruptly across 4 concepts:
    - Concept 1: $\theta = 8.0$
    - Concept 2: $\theta = 9.0$
    - Concept 3: $\theta = 7.0$
    - Concept 4: $\theta = 9.5$

    Parameters
    ----------
    n_samples : int, default=1000
        Total number of streaming instances to generate.
    noise : float, default=0.05
        Fraction of labels to randomly flip (label noise).
    random_state : Optional[int], default=42
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 3)
        Stream feature vectors.
    y : np.ndarray of shape (n_samples,)
        Binary class labels {-1, 1}.
    drift_points : List[int]
        Indices where abrupt concept drift occurs.
    """
    rng = np.random.RandomState(random_state)
    thresholds = [8.0, 9.0, 7.0, 9.5]
    n_concepts = len(thresholds)
    samples_per_concept = n_samples // n_concepts
    remainder = n_samples % n_concepts

    X_list = []
    y_list = []
    drift_points = []
    curr_idx = 0

    for i, theta in enumerate(thresholds):
        if i > 0:
            drift_points.append(curr_idx)
        count = samples_per_concept + (remainder if i == n_concepts - 1 else 0)
        curr_idx += count

        # Generate features in [0, 10]
        x_block = rng.uniform(0.0, 10.0, size=(count, 3))
        sum_feat = x_block[:, 0] + x_block[:, 1]
        y_block = np.where(sum_feat > theta, 1.0, -1.0)

        # Apply label noise
        if noise > 0:
            noise_mask = rng.rand(count) < noise
            y_block[noise_mask] = -y_block[noise_mask]

        X_list.append(x_block)
        y_list.append(y_block)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y, drift_points


def make_rotating_hyperplane_stream(
    n_samples: int = 1000,
    n_features: int = 4,
    drift_speed: float = 0.005,
    noise: float = 0.02,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Rotating Hyperplane benchmark stream with gradual / continuous concept drift.

    A hyperplane $w(t)^T x = 0$ continuously rotates at angular velocity `drift_speed`.

    Parameters
    ----------
    n_samples : int, default=1000
    n_features : int, default=4
    drift_speed : float, default=0.005
    noise : float, default=0.02
    random_state : Optional[int], default=42

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,) in {-1, 1}
    """
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, n_features))
    y = np.zeros(n_samples)

    w_base = rng.randn(n_features)
    w_base /= np.linalg.norm(w_base)

    for t in range(n_samples):
        # Rotate weight vector continuously
        angle = t * drift_speed
        w_t = w_base.copy()
        w_t[0] = np.cos(angle)
        w_t[1] = np.sin(angle)
        w_t /= np.linalg.norm(w_t)

        score = np.dot(X[t], w_t)
        label = 1.0 if score >= 0 else -1.0

        if noise > 0 and rng.rand() < noise:
            label = -label
        y[t] = label

    return X, y


def make_gaussian_drift_stream(
    n_samples: int = 600,
    n_features: int = 6,
    n_drifts: int = 2,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Generate a multi-dimensional Gaussian streaming dataset with mean & covariance shifts.

    Parameters
    ----------
    n_samples : int, default=600
    n_features : int, default=6
    n_drifts : int, default=2
    random_state : Optional[int], default=42

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,) in {-1, 1}
    drift_points : List[int]
    """
    rng = np.random.RandomState(random_state)
    n_regimes = n_drifts + 1
    samples_per_regime = n_samples // n_regimes
    remainder = n_samples % n_regimes

    X_list = []
    y_list = []
    drift_points = []
    curr_idx = 0

    for r in range(n_regimes):
        if r > 0:
            drift_points.append(curr_idx)
        count = samples_per_regime + (remainder if r == n_regimes - 1 else 0)
        curr_idx += count
        n_pos = count // 2
        n_neg = count - n_pos

        # Dynamic mean shifts per regime
        mu_pos = rng.randn(n_features) * 2.0 + np.ones(n_features) * (r * 2.0)
        mu_neg = rng.randn(n_features) * 2.0 - np.ones(n_features) * (r * 2.0)

        # Scale variance
        scale = 0.7 + 0.3 * r
        X_pos = rng.randn(n_pos, n_features) * scale + mu_pos
        X_neg = rng.randn(n_neg, n_features) * scale + mu_neg

        X_block = np.vstack([X_pos, X_neg])
        y_block = np.array([1.0] * n_pos + [-1.0] * n_neg)

        # Shuffle regime block
        perm = rng.permutation(count)
        X_list.append(X_block[perm])
        y_list.append(y_block[perm])

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y, drift_points


def make_anisotropic_anomaly_dataset(
    n_normal: int = 150,
    n_anomalies: int = 30,
    ratio: float = 10.0,
    dimension: int = 2,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate an anisotropic distribution benchmark for low-variance anomaly detection (COSVM/ICOSVM).

    Normal data is distributed along a high-variance principal axis and low-variance orthogonal axis.
    Anomalies are placed along the low-variance orthogonal manifold where standard Euclidean metrics
    fail without covariance guidance.

    Parameters
    ----------
    n_normal : int, default=150
    n_anomalies : int, default=30
    ratio : float, default=10.0
        Ratio of high variance to low variance.
    dimension : int, default=2
    random_state : Optional[int], default=42

    Returns
    -------
    X_train_normal : np.ndarray of shape (n_normal, dimension)
    X_test_normal : np.ndarray of shape (n_normal // 2, dimension)
    X_test_anomalies : np.ndarray of shape (n_anomalies, dimension)
    """
    rng = np.random.RandomState(random_state)
    theta = np.pi / 4.0

    if dimension == 2:
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        scales = np.array([ratio, 1.0])
    else:
        # Random orthogonal rotation matrix
        A = rng.randn(dimension, dimension)
        rot, _ = np.linalg.qr(A)
        scales = np.ones(dimension)
        scales[0] = ratio
        scales[1:] = 1.0

    # Training normal
    Z_train = rng.randn(n_normal, dimension) * scales
    X_train_normal = np.dot(Z_train, rot.T)

    # Test normal
    n_test_norm = max(20, n_normal // 2)
    Z_test_norm = rng.randn(n_test_norm, dimension) * scales
    X_test_normal = np.dot(Z_test_norm, rot.T)

    # Test anomalies (amplified along low-variance directions)
    Z_anom = rng.randn(n_anomalies, dimension)
    Z_anom[:, 0] = rng.uniform(-ratio * 0.5, ratio * 0.5, size=n_anomalies)  # Normal along major axis
    Z_anom[:, 1] = rng.choice([-1.0, 1.0], size=n_anomalies) * rng.uniform(3.0, 6.0, size=n_anomalies)  # Outlier on minor axis
    X_test_anomalies = np.dot(Z_anom, rot.T)

    return X_train_normal, X_test_normal, X_test_anomalies


def make_benchmark_moons(
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 2D interleaving half circles (Two Moons) benchmark."""
    rng = np.random.RandomState(random_state)
    n_per_class = n_samples // 2

    # Moon 1
    t1 = np.linspace(0, np.pi, n_per_class)
    x1 = np.cos(t1) + rng.randn(n_per_class) * noise
    y1 = np.sin(t1) + rng.randn(n_per_class) * noise
    X1 = np.column_stack([x1, y1])

    # Moon 2
    t2 = np.linspace(0, np.pi, n_per_class)
    x2 = 1.0 - np.cos(t2) + rng.randn(n_per_class) * noise
    y2 = 1.0 - np.sin(t2) - 0.5 + rng.randn(n_per_class) * noise
    X2 = np.column_stack([x2, y2])

    X = np.vstack([X1, X2])
    y = np.array([1.0] * n_per_class + [-1.0] * n_per_class)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def make_benchmark_circles(
    n_samples: int = 200,
    noise: float = 0.05,
    factor: float = 0.5,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Concentric Circles non-linear benchmark."""
    rng = np.random.RandomState(random_state)
    n_per_class = n_samples // 2

    # Outer circle
    t_out = rng.uniform(0, 2 * np.pi, n_per_class)
    x_out = np.cos(t_out) + rng.randn(n_per_class) * noise
    y_out = np.sin(t_out) + rng.randn(n_per_class) * noise
    X_out = np.column_stack([x_out, y_out])

    # Inner circle
    t_in = rng.uniform(0, 2 * np.pi, n_per_class)
    x_in = np.cos(t_in) * factor + rng.randn(n_per_class) * noise
    y_in = np.sin(t_in) * factor + rng.randn(n_per_class) * noise
    X_in = np.column_stack([x_in, y_in])

    X = np.vstack([X_out, X_in])
    y = np.array([1.0] * n_per_class + [-1.0] * n_per_class)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]

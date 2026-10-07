"""
Online Covariance Tracking, Latent Space Drift Monitoring, and Obsolescence Scoring.
Based on the Drift-Aware Covariance-Aware Incremental SVM research architecture.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from dynamicsvm.utils.validation import check_array


class OnlineCovarianceTracker:
    r"""
    Maintains and updates online latent mean $\mu_t$ and covariance $\Sigma_t$
    with drift-dependent adaptive exponential forgetting:
    
    $$\Sigma_t = \lambda(D_t) \Sigma_{t-1} + (1 - \lambda(D_t)) \Sigma_{B_t}$$
    $$\mu_t = \lambda(D_t) \mu_{t-1} + (1 - \lambda(D_t)) \mu_{B_t}$$
    
    where $D_t = \lambda_\mu \|\mu_{B_t} - \mu_t\|_2^2 + \lambda_\Sigma \|\Sigma_{B_t} - \Sigma_t\|_F^2$.
    
    Parameters
    ----------
    lambda_base : float, default=0.95
        Baseline forgetting factor $\lambda_0 \in (0, 1)$ under stationary conditions.
    lambda_min : float, default=0.50
        Minimum forgetting factor when strong drift occurs (allows rapid adaptation).
    lambda_mu : float, default=1.0
        Weight for mean discrepancy in the drift statistic $D_t$.
    lambda_sigma : float, default=1.0
        Weight for covariance Frobenius discrepancy in the drift statistic $D_t$.
    gamma_drift : float, default=0.5
        Sensitivity parameter $\gamma_D$ for adaptive forgetting $\lambda(D_t) = \max(\lambda_{\min}, \lambda_0 e^{-\gamma_D D_t})$.
    eps : float, default=1e-4
        Regularization constant added to covariance diagonal for invertibility.
    """

    def __init__(
        self,
        lambda_base: float = 0.95,
        lambda_min: float = 0.50,
        lambda_mu: float = 1.0,
        lambda_sigma: float = 1.0,
        gamma_drift: float = 0.5,
        eps: float = 1e-4,
    ):
        self.lambda_base = float(lambda_base)
        self.lambda_min = float(lambda_min)
        self.lambda_mu = float(lambda_mu)
        self.lambda_sigma = float(lambda_sigma)
        self.gamma_drift = float(gamma_drift)
        self.eps = float(eps)

        self.mean_: Optional[np.ndarray] = None
        self.cov_: Optional[np.ndarray] = None
        self.inv_cov_: Optional[np.ndarray] = None
        self.n_updates_: int = 0
        self.last_drift_statistic_: float = 0.0
        self.last_lambda_: float = self.lambda_base
        self.history_D_: List[float] = []

    def fit(self, Z: np.ndarray) -> "OnlineCovarianceTracker":
        """
        Initialize mean and covariance statistics from initial latent batch Z.
        """
        Z_arr = check_array(Z)
        n_samples, n_features = Z_arr.shape

        self.mean_ = np.mean(Z_arr, axis=0)
        Z_centered = Z_arr - self.mean_
        if n_samples > 1:
            self.cov_ = np.dot(Z_centered.T, Z_centered) / (n_samples - 1)
        else:
            self.cov_ = np.eye(n_features)

        self._update_inv_cov()
        self.n_updates_ = 1
        self.last_drift_statistic_ = 0.0
        self.last_lambda_ = self.lambda_base
        self.history_D_ = [0.0]
        return self

    def _update_inv_cov(self) -> None:
        """Compute regularized inverse covariance (Sigma + eps * I)^{-1}."""
        if self.cov_ is None:
            return
        d = self.cov_.shape[0]
        cov_reg = self.cov_ + self.eps * np.eye(d)
        try:
            self.inv_cov_ = np.linalg.pinv(cov_reg)
        except np.linalg.LinAlgError:
            self.inv_cov_ = np.eye(d) / self.eps

    def compute_drift_statistic(self, Z_batch: np.ndarray) -> float:
        """
        Compute drift discrepancy statistic D_t between current state and incoming batch:
        D_t = lambda_mu * ||mu_B - mu_t||^2 + lambda_sigma * ||Sigma_B - Sigma_t||_F^2
        """
        if self.mean_ is None or self.cov_ is None:
            return 0.0

        Z_arr = check_array(Z_batch)
        mu_B = np.mean(Z_arr, axis=0)
        diff_mu = mu_B - self.mean_
        mu_dist_sq = float(np.sum(diff_mu ** 2))

        if Z_arr.shape[0] > 1:
            Z_cent = Z_arr - mu_B
            cov_B = np.dot(Z_cent.T, Z_cent) / (Z_arr.shape[0] - 1)
        else:
            cov_B = self.cov_.copy()

        diff_cov = cov_B - self.cov_
        cov_dist_sq = float(np.sum(diff_cov ** 2))  # Squared Frobenius norm

        D_t = self.lambda_mu * mu_dist_sq + self.lambda_sigma * cov_dist_sq
        return float(D_t)

    def compute_adaptive_lambda(self, D_t: float) -> float:
        """
        Compute drift-adaptive forgetting factor lambda(D_t).
        Higher drift -> lower lambda -> faster adaptation.
        """
        decay = np.exp(-self.gamma_drift * max(0.0, D_t))
        adaptive_lambda = self.lambda_base * decay
        return float(np.clip(adaptive_lambda, self.lambda_min, self.lambda_base))

    def update(self, Z_batch: np.ndarray) -> Tuple[float, float]:
        """
        Update latent mean and covariance with new incoming batch Z_batch.
        
        Returns
        -------
        D_t : float
            Computed drift discrepancy statistic.
        lambda_t : float
            Adaptive forgetting factor used for this update.
        """
        Z_arr = check_array(Z_batch)
        if self.mean_ is None or self.cov_ is None:
            self.fit(Z_arr)
            return 0.0, self.lambda_base

        D_t = self.compute_drift_statistic(Z_arr)
        lambda_t = self.compute_adaptive_lambda(D_t)

        mu_B = np.mean(Z_arr, axis=0)
        if Z_arr.shape[0] > 1:
            Z_cent = Z_arr - mu_B
            cov_B = np.dot(Z_cent.T, Z_cent) / (Z_arr.shape[0] - 1)
        else:
            # Single sample: outer product rank-1 update
            diff = mu_B - self.mean_
            cov_B = np.outer(diff, diff)

        # Exponential adaptation
        self.mean_ = lambda_t * self.mean_ + (1.0 - lambda_t) * mu_B
        self.cov_ = lambda_t * self.cov_ + (1.0 - lambda_t) * cov_B
        self._update_inv_cov()

        self.n_updates_ += 1
        self.last_drift_statistic_ = D_t
        self.last_lambda_ = lambda_t
        self.history_D_.append(D_t)

        return D_t, lambda_t

    def mahalanobis_distance(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance d_M(z, mu_t, Sigma_t) for each row in Z:
        d_M(z) = sqrt( (z - mu)^T (Sigma + eps*I)^{-1} (z - mu) )
        """
        if self.mean_ is None or self.inv_cov_ is None:
            raise RuntimeError("OnlineCovarianceTracker is not initialized.")
        Z_arr = check_array(Z)
        diff = Z_arr - self.mean_
        # (N, d) dot (d, d) -> (N, d), elementwise product with diff -> sum across axis 1
        dist_sq = np.sum(np.dot(diff, self.inv_cov_) * diff, axis=1)
        return np.sqrt(np.maximum(0.0, dist_sq))


class LatentCovarianceDriftDetector:
    """
    Concept & Latent Distribution Drift Detector operating on latent covariance statistics.
    Evaluates drift discrepancy statistic D_t against statistical warning and drift thresholds.
    
    Parameters
    ----------
    warning_threshold : float, default=1.5
        Threshold above which a drift warning is triggered.
    drift_threshold : float, default=3.0
        Threshold above which confirmed distribution drift is triggered.
    min_instances : int, default=5
        Warm-up observations required before triggering drift alerts.
    """

    def __init__(
        self,
        warning_threshold: float = 1.5,
        drift_threshold: float = 3.0,
        min_instances: int = 5,
    ):
        self.warning_threshold = float(warning_threshold)
        self.drift_threshold = float(drift_threshold)
        self.min_instances = int(min_instances)

        self.n_checks_: int = 0
        self.in_warning_zone_: bool = False
        self.in_drift_zone_: bool = False
        self.drift_count_: int = 0
        self.history_status_: List[str] = []

    def update(self, D_t: float) -> Tuple[bool, bool]:
        """
        Update detector state with drift statistic D_t.
        
        Returns
        -------
        is_drift : bool
            True if significant distribution drift is confirmed.
        is_warning : bool
            True if warning level is reached.
        """
        self.n_checks_ += 1

        if self.n_checks_ <= self.min_instances:
            self.history_status_.append("WARMUP")
            return False, False

        if D_t >= self.drift_threshold:
            self.in_drift_zone_ = True
            self.in_warning_zone_ = False
            self.drift_count_ += 1
            self.history_status_.append("DRIFT")
            return True, False
        elif D_t >= self.warning_threshold:
            self.in_warning_zone_ = True
            self.in_drift_zone_ = False
            self.history_status_.append("WARNING")
            return False, True
        else:
            self.in_warning_zone_ = False
            self.in_drift_zone_ = False
            self.history_status_.append("STABLE")
            return False, False

    def reset(self) -> None:
        """Reset internal warning/drift states."""
        self.in_warning_zone_ = False
        self.in_drift_zone_ = False


class ObsolescenceScorer:
    r"""
    Computes sample obsolescence score $O_i$ for decremental unlearning / forgetting:
    
    $$O_i = \lambda_A A_i + \lambda_T T_i + \lambda_D D_i$$
    
    where:
    - $A_i$: Normalized sample age $\frac{t_{\text{current}} - t_i}{t_{\max} + 1}$
    - $T_i$: Temporal decay distance $1 - \exp(-\beta (t_{\text{current}} - t_i))$
    - $D_i$: Distributional inconsistency (Mahalanobis distance to current latent regime)
    
    Parameters
    ----------
    lambda_age : float, default=0.3
        Weight for sample chronological age.
    lambda_temporal : float, default=0.3
        Weight for temporal decay distance from current regime.
    lambda_dist : float, default=0.4
        Weight for latent distributional inconsistency.
    beta : float, default=0.05
        Decay constant for temporal distance.
    sv_protection_factor : float, default=0.5
        Reduction multiplier on obsolescence score for active Support Vectors,
        ensuring old non-support vectors are pruned before critical margin vectors.
    """

    def __init__(
        self,
        lambda_age: float = 0.3,
        lambda_temporal: float = 0.3,
        lambda_dist: float = 0.4,
        beta: float = 0.05,
        sv_protection_factor: float = 0.5,
    ):
        self.lambda_age = float(lambda_age)
        self.lambda_temporal = float(lambda_temporal)
        self.lambda_dist = float(lambda_dist)
        self.beta = float(beta)
        self.sv_protection_factor = float(sv_protection_factor)

    def score_samples(
        self,
        Z_stored: np.ndarray,
        arrival_times: np.ndarray,
        current_time: int,
        tracker: OnlineCovarianceTracker,
        is_support_vector: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate obsolescence scores for all stored samples in memory.
        
        Returns
        -------
        scores : np.ndarray of shape (n_stored_samples,)
            Higher score indicates higher obsolescence (prime candidate for removal).
        """
        Z_arr = check_array(Z_stored)
        n_samples = Z_arr.shape[0]
        if n_samples == 0:
            return np.array([])

        times = np.asarray(arrival_times, dtype=float)
        delta_t = np.maximum(0.0, current_time - times)

        # 1. Age score A_i in [0, 1]
        max_age = max(1.0, float(np.max(delta_t)))
        A = delta_t / (max_age + 1e-8)

        # 2. Temporal distance T_i in [0, 1]
        T = 1.0 - np.exp(-self.beta * delta_t)

        # 3. Distributional inconsistency D_i via Mahalanobis distance
        m_dist = tracker.mahalanobis_distance(Z_arr)
        max_dist = max(1.0, float(np.max(m_dist)))
        D = m_dist / (max_dist + 1e-8)

        scores = self.lambda_age * A + self.lambda_temporal * T + self.lambda_dist * D

        # Apply support vector protection
        if is_support_vector is not None:
            sv_mask = np.asarray(is_support_vector, dtype=bool)
            scores[sv_mask] *= self.sv_protection_factor

        return scores

    def select_obsolete_indices(
        self,
        scores: np.ndarray,
        n_to_remove: int,
    ) -> np.ndarray:
        """
        Select indices with highest obsolescence scores.
        """
        if len(scores) == 0 or n_to_remove <= 0:
            return np.array([], dtype=int)
        k = min(len(scores), n_to_remove)
        # Top-k highest scores
        return np.argsort(scores)[-k:]

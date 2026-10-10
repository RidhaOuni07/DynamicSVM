"""
Online Covariance Tracking, Latent Space Drift Monitoring, and Obsolescence Scoring.
Based on the Drift-Aware Covariance-Aware Incremental SVM research architecture.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from dynamicsvm.utils.validation import check_array

# Upper clip of the surprise score D_t = -log10 p (p-values below 1e-300).
_MAX_SURPRISE = 300.0


class OnlineCovarianceTracker:
    r"""
    Online latent mean $\mu_t$ and covariance $\Sigma_t$ with exponential
    forgetting, plus a calibrated drift statistic.

    Exponentially weighted (EW) estimates. Each incoming row $z$ is merged with
    weight $1 - \lambda$ (older rows are down-weighted by $\lambda$):

    $$\delta = z - \mu, \quad \mu \leftarrow \mu + (1 - \lambda)\,\delta, \quad
    V \leftarrow \lambda \left(V + (1 - \lambda)\,\delta \delta^T\right)$$

    $V$ is the weighted (biased) covariance $\sum_i w_i (z_i - \mu)(z_i - \mu)^T$
    with normalised weights $w_i$ (Finch 2009 recursion). Kish's effective sample
    size $n_{\text{eff}} = 1 / \sum_i w_i^2$ follows

    $$n_{\text{eff}} \leftarrow 1 / \left(\lambda^2 / n_{\text{eff}} + (1 - \lambda)^2\right)$$

    and, for i.i.d. rows, $E[V] = (1 - 1/n_{\text{eff}})\,\Sigma$, so the exposed
    estimate $\Sigma_t = V\, n_{\text{eff}} / (n_{\text{eff}} - 1)$ is unbiased.
    ``fit`` starts from the sample mean and the unbiased sample covariance with
    $n_{\text{eff}} = n$. For a constant $\lambda$, $n_{\text{eff}} \to (1+\lambda)/(1-\lambda)$
    (39 for $\lambda = 0.95$).

    Drift statistic. For an incoming batch $B$ of $n_B$ rows with mean $\bar z$,
    evaluated against the state *before* the update:

    $$T^2 = \frac{(\bar z - \mu)^T (\Sigma + \epsilon I)^{-1} (\bar z - \mu)}{1/n_B + 1/n_{\text{eff}}}$$

    - $n_B < d + 1$ (single samples): Hotelling calibration
      $F = T^2 (n_{\text{eff}} - d) / (d (n_{\text{eff}} - 1)) \sim F(d, n_{\text{eff}} - d)$
      when $n_{\text{eff}} > d + 1$, else $T^2 \sim \chi^2_d$.
    - $n_B \ge d + 1$: add the covariance likelihood-ratio term
      $\mathrm{LRT} = (n_B - 1)[\mathrm{tr}(\Sigma^{-1} S_B) - \log\det(\Sigma^{-1} S_B) - d]$
      with $S_B$ the unbiased batch covariance, and use
      $T^2 + \mathrm{LRT} \sim \chi^2_{d + d(d+1)/2}$ (asymptotic, $\Sigma$ treated as known).

    The returned score is the surprise $D_t = -\log_{10} p$, clipped to
    $[0, 300]$; $D_t = 2$ means $p = 0.01$, $D_t = 3$ means $p = 0.001$. The
    calibration is exact for equally weighted Gaussian rows and approximate for
    EW weights (non-integer $n_{\text{eff}}$).

    Forgetting factor. $\lambda(D_t) = \lambda_0$ when $D_t \le$ ``adapt_threshold``,
    else $\max(\lambda_{\min}, \lambda_0 e^{-\gamma_D (D_t - \text{adapt\_threshold})})$.
    Ordinary stationary fluctuations therefore never shrink $n_{\text{eff}}$, which
    prevents a self-reinforcing alarm -> small $\lambda$ -> noisy $\Sigma$ -> alarm loop.

    Parameters
    ----------
    lambda_base : float, default=0.95
        Baseline forgetting factor $\lambda_0 \in (0, 1)$ under stationary conditions.
    lambda_min : float, default=0.50
        Minimum forgetting factor when strong drift occurs (allows rapid adaptation).
    lambda_mu : float, default=1.0
        Unused; kept for backward compatibility with the former uncalibrated
        statistic $\lambda_\mu \|\mu_B - \mu\|^2 + \lambda_\Sigma \|\Sigma_B - \Sigma\|_F^2$.
    lambda_sigma : float, default=1.0
        Unused; kept for backward compatibility (see ``lambda_mu``).
    gamma_drift : float, default=0.5
        Sensitivity $\gamma_D$ of the forgetting factor to the surprise above
        ``adapt_threshold``. ``gamma_drift=0`` gives a constant $\lambda_0$.
    eps : float, default=1e-4
        Regularization constant added to covariance diagonal for invertibility.
    adapt_threshold : float, default=2.0
        Surprise level ($-\log_{10} p$) above which $\lambda$ is lowered.
    """

    def __init__(
        self,
        lambda_base: float = 0.95,
        lambda_min: float = 0.50,
        lambda_mu: float = 1.0,
        lambda_sigma: float = 1.0,
        gamma_drift: float = 0.5,
        eps: float = 1e-4,
        adapt_threshold: float = 2.0,
    ):
        self.lambda_base = float(lambda_base)
        self.lambda_min = float(lambda_min)
        self.lambda_mu = float(lambda_mu)
        self.lambda_sigma = float(lambda_sigma)
        self.gamma_drift = float(gamma_drift)
        self.eps = float(eps)
        self.adapt_threshold = float(adapt_threshold)

        self.mean_: Optional[np.ndarray] = None
        self.cov_: Optional[np.ndarray] = None
        self.inv_cov_: Optional[np.ndarray] = None
        self.n_eff_: float = 0.0
        self.n_updates_: int = 0
        self.last_drift_statistic_: float = 0.0
        self.last_p_value_: float = 1.0
        self.last_lambda_: float = self.lambda_base
        self.history_D_: List[float] = []
        # Weighted (biased) covariance V; cov_ = V * n_eff / (n_eff - 1)
        self._weighted_cov: Optional[np.ndarray] = None
        self._cho = None

    def fit(self, Z: np.ndarray) -> "OnlineCovarianceTracker":
        """
        Initialize mean and covariance statistics from initial latent batch Z
        (sample mean, unbiased sample covariance, n_eff_ = n).
        """
        Z_arr = check_array(Z)
        n_samples, n_features = Z_arr.shape

        self.mean_ = np.mean(Z_arr, axis=0)
        Z_centered = Z_arr - self.mean_
        if n_samples > 1:
            self._weighted_cov = np.dot(Z_centered.T, Z_centered) / n_samples
            self.n_eff_ = float(n_samples)
        else:
            # One row carries no covariance information: identity prior
            # (cov_ = I), counted as one extra pseudo-sample (n_eff_ = 2).
            self._weighted_cov = 0.5 * np.eye(n_features)
            self.n_eff_ = 2.0
        self._sync_cov()

        self.n_updates_ = 1
        self.last_drift_statistic_ = 0.0
        self.last_p_value_ = 1.0
        self.last_lambda_ = self.lambda_base
        self.history_D_ = [0.0]
        return self

    def _sync_cov(self) -> None:
        """Set cov_ = V * n_eff / (n_eff - 1) (unbiased) and refresh the inverse."""
        if self.n_eff_ > 1.0:
            self.cov_ = self._weighted_cov * (self.n_eff_ / (self.n_eff_ - 1.0))
        else:
            self.cov_ = self._weighted_cov.copy()
        self._update_inv_cov()

    def _update_inv_cov(self) -> None:
        """Cholesky factor and inverse of the regularized covariance Sigma + eps * I."""
        if self.cov_ is None:
            return
        d = self.cov_.shape[0]
        cov_reg = 0.5 * (self.cov_ + self.cov_.T) + self.eps * np.eye(d)
        try:
            self._cho = cho_factor(cov_reg, lower=True)
            self.inv_cov_ = cho_solve(self._cho, np.eye(d))
        except (LinAlgError, ValueError):
            self._cho = None
            self.inv_cov_ = np.linalg.pinv(cov_reg)

    def _solve(self, B: np.ndarray) -> np.ndarray:
        """Return (Sigma + eps I)^{-1} B."""
        if self._cho is not None:
            return cho_solve(self._cho, B)
        return np.dot(self.inv_cov_, B)

    def _drift_log_p(self, Z_arr: np.ndarray) -> float:
        """Natural log of the p-value of the batch Z_arr under the current state."""
        n_B, d = Z_arr.shape
        z_bar = np.mean(Z_arr, axis=0)
        diff = z_bar - self.mean_
        n_eff = max(self.n_eff_, 1.0)
        maha = float(np.dot(diff, self._solve(diff)))
        T2 = max(0.0, maha) / (1.0 / n_B + 1.0 / n_eff)

        if n_B < d + 1:
            if n_eff > d + 1:
                F = T2 * (n_eff - d) / (d * (n_eff - 1.0))
                return float(stats.f.logsf(F, d, n_eff - d))
            return float(stats.chi2.logsf(T2, d))

        Z_cent = Z_arr - z_bar
        S_B = np.dot(Z_cent.T, Z_cent) / (n_B - 1)
        M = self._solve(S_B)
        sign, logdet = np.linalg.slogdet(M)
        if sign <= 0:
            # Degenerate batch covariance (collinear rows): maximal surprise.
            return -np.inf
        lrt = (n_B - 1) * (float(np.trace(M)) - logdet - d)
        df = d + d * (d + 1) // 2
        return float(stats.chi2.logsf(T2 + max(0.0, lrt), df))

    def compute_drift_statistic(self, Z_batch: np.ndarray) -> float:
        """
        Calibrated drift surprise D_t = -log10 p of the batch under the current
        state (Hotelling predictive test for n_B < d + 1, plus the covariance
        likelihood-ratio term for n_B >= d + 1; see the class docstring).
        Stores the p-value in ``last_p_value_``.
        """
        if self.mean_ is None or self.cov_ is None:
            return 0.0

        Z_arr = check_array(Z_batch, n_features=self.mean_.shape[0])
        log_p = self._drift_log_p(Z_arr)
        if np.isnan(log_p):
            log_p = -np.inf
        D_t = float(np.clip(-log_p / np.log(10.0), 0.0, _MAX_SURPRISE))
        self.last_p_value_ = float(10.0 ** (-D_t))
        return D_t

    def compute_adaptive_lambda(self, D_t: float) -> float:
        """
        Drift-adaptive forgetting factor: lambda_base while D_t <= adapt_threshold,
        else max(lambda_min, lambda_base * exp(-gamma_drift * (D_t - adapt_threshold))).
        """
        if D_t <= self.adapt_threshold:
            return self.lambda_base
        decay = np.exp(-self.gamma_drift * (D_t - self.adapt_threshold))
        return float(max(self.lambda_min, self.lambda_base * decay))

    def update(self, Z_batch: np.ndarray) -> Tuple[float, float]:
        """
        Score Z_batch against the current state, then merge its rows one by one
        with the EW recursion of the class docstring.

        Returns
        -------
        D_t : float
            Drift surprise -log10 p of the batch (before the update).
        lambda_t : float
            Adaptive forgetting factor used for this update.
        """
        Z_arr = check_array(Z_batch, n_features=None if self.mean_ is None else self.mean_.shape[0])
        if self.mean_ is None or self.cov_ is None:
            self.fit(Z_arr)
            return 0.0, self.lambda_base

        D_t = self.compute_drift_statistic(Z_arr)
        lambda_t = self.compute_adaptive_lambda(D_t)

        mean = self.mean_.copy()
        V = self._weighted_cov.copy()
        n_eff = self.n_eff_
        w = 1.0 - lambda_t
        for z in Z_arr:
            delta = z - mean
            mean = mean + w * delta
            V = lambda_t * (V + w * np.outer(delta, delta))
            n_eff = 1.0 / (lambda_t ** 2 / n_eff + w ** 2)
        self.mean_ = mean
        self._weighted_cov = V
        self.n_eff_ = float(n_eff)
        self._sync_cov()

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
        Z_arr = check_array(Z, n_features=self.mean_.shape[0])
        diff = Z_arr - self.mean_
        # (N, d) dot (d, d) -> (N, d), elementwise product with diff -> sum across axis 1
        dist_sq = np.sum(np.dot(diff, self.inv_cov_) * diff, axis=1)
        return np.sqrt(np.maximum(0.0, dist_sq))


class LatentCovarianceDriftDetector:
    """
    Threshold detector for the latent drift surprise D_t.

    The input is the calibrated score D_t = -log10 p produced by
    ``OnlineCovarianceTracker.compute_drift_statistic`` (p = p-value of the
    incoming sample/batch under the current latent Gaussian state). A threshold
    tau therefore corresponds to a per-check false-alarm probability 10^{-tau}
    under stationarity, independently of the latent scale.

    Parameters
    ----------
    warning_threshold : float, default=2.0
        Warning when D_t >= warning_threshold (default: p < 0.01).
    drift_threshold : float, default=3.0
        Confirmed drift when D_t >= drift_threshold (default: p < 0.001).
    min_instances : int, default=5
        Warm-up checks without alerts after construction and after every
        ``reset()`` (refractory period after a confirmed drift).
    """

    def __init__(
        self,
        warning_threshold: float = 2.0,
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
        """
        Reset warning/drift states and restart the warm-up, so the next
        ``min_instances`` checks raise no alert (``drift_count_`` is kept).
        """
        self.in_warning_zone_ = False
        self.in_drift_zone_ = False
        self.n_checks_ = 0


class ObsolescenceScorer:
    r"""
    Computes sample obsolescence score $O_i$ for decremental unlearning / forgetting:
    
    $$O_i = \lambda_A A_i + \lambda_T T_i + \lambda_D D_i$$
    
    where:
    - $A_i$: Normalized sample age $\frac{t_{\text{current}} - t_i}{t_{\max} + 1}$
    - $T_i$: Temporal decay distance $1 - \exp(-\beta (t_{\text{current}} - t_i))$
    - $D_i$: Distributional inconsistency (Mahalanobis distance to current latent regime)

    This score is a heuristic ranking rule, not a derived quantity: the weights,
    the max-normalisation of $A_i$ and $D_i$ (which makes scores depend on the
    current buffer) and ``sv_protection_factor`` are design choices with no
    optimality or statistical guarantee. Only the ordering of the scores is used.
    
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

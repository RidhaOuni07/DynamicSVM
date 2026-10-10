"""
Drift-Aware Covariance-Aware Incremental/Decremental Support Vector Machine.
Full implementation of the research architecture from the PhD summary paper:
Autoencoder -> Latent-space monitoring -> Covariance-aware SVM -> Incremental/Decremental adaptation.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from dynamicsvm.encoders.base import BaseEncoder
from dynamicsvm.encoders.autoencoder import AutoEncoder, IdentityEncoder, PCAEncoder
from dynamicsvm.models.covariance_svc import CovarianceAwareSVC
from dynamicsvm.stream.base import StreamingSVM
from dynamicsvm.stream.covariance import (
    LatentCovarianceDriftDetector,
    ObsolescenceScorer,
    OnlineCovarianceTracker,
)
from dynamicsvm.utils.validation import check_X_y, check_array, check_is_fitted


class DriftAwareCovarianceSVM(StreamingSVM):
    r"""
    Drift-Aware Covariance-Aware Incremental/Decremental Support Vector Machine.
    
    Architecture:
    1. Pretrained/Frozen Representation Encoder: maps $x \to z$ to provide a stable coordinate system.
    2. Online Latent Mean & Covariance Tracker: tracks $\mu_t, \Sigma_t$ with unbiased
       exponentially weighted estimates and drift-adaptive forgetting.
    3. Latent Drift Detector: each new $z_t$ is scored by the calibrated surprise
       $D_t = -\log_{10} p_t$, $p_t$ the Hotelling predictive p-value of $z_t$ under
       $(\mu_{t-1}, \Sigma_{t-1}, n_{\text{eff}})$ (see ``OnlineCovarianceTracker``).
       After a confirmed drift the detector is reset, which imposes a refractory
       warm-up of ``min_instances`` checks.
    4. Decremental Unlearning via Obsolescence Scoring: $O_i = \lambda_A A_i + \lambda_T T_i + \lambda_D D_i$
       (heuristic ranking, see ``ObsolescenceScorer``).
    5. Covariance-Aware SVM Decision Boundary: $Q_t = \eta K_t + (1 - \eta) s K_t H_t K_t$
       (see ``CovarianceAwareSVC``).

    Every ``learn_one`` step refits ``CovarianceAwareSVC`` exactly on the current
    buffer (batch re-solve); "incremental/decremental" refers to buffer insertion
    and removal, not to an incremental update of the dual solution.
    
    Parameters
    ----------
    encoder : BaseEncoder, optional
        Feature encoder mapping inputs X into latent space Z. If None, AutoEncoder is used.
    C : float, default=1.0
        SVM regularization parameter.
    eta : float, default=0.7
        Weight between kernel Gram matrix and covariance component in CovarianceAwareSVC.
    kernel : str or Kernel, default='rbf'
        Kernel function for SVM in latent space.
    gamma : str or float, default='scale'
        Kernel coefficient.
    solver : {'smo', 'qp'}, default='smo'
        Optimization solver.
    max_buffer_size : int, default=200
        Maximum number of retained training samples in active memory buffer.
    min_buffer_size : int, default=10
        Minimum samples needed before training SVM classifier.
    drift_threshold : float, default=3.0
        Drift is confirmed when $D_t = -\log_{10} p_t \ge$ drift_threshold
        (default: p < 0.001 per step under stationarity).
    warning_threshold : float, default=2.0
        Drift warning when $D_t \ge$ warning_threshold (default: p < 0.01).
    lambda_base : float, default=0.95
        Baseline forgetting factor $\lambda_0$ for covariance tracker.
    lambda_min : float, default=0.50
        Minimum forgetting factor during heavy drift.
    lambda_age : float, default=0.3
        Weight for sample age in obsolescence score.
    lambda_temporal : float, default=0.3
        Weight for temporal distance in obsolescence score.
    lambda_dist : float, default=0.4
        Weight for latent Mahalanobis distance in obsolescence score.
    retire_ratio_on_drift : float, default=0.30
        Fraction of oldest/most obsolete samples to decrementally purge upon confirmed drift.
    random_state : int, optional
        Random seed.
    """

    def __init__(
        self,
        encoder: Optional[BaseEncoder] = None,
        C: float = 1.0,
        eta: float = 0.7,
        kernel: str = "rbf",
        gamma: Union[str, float] = "scale",
        solver: str = "smo",
        max_buffer_size: int = 200,
        min_buffer_size: int = 10,
        drift_threshold: float = 3.0,
        warning_threshold: float = 2.0,
        lambda_base: float = 0.95,
        lambda_min: float = 0.50,
        lambda_age: float = 0.3,
        lambda_temporal: float = 0.3,
        lambda_dist: float = 0.4,
        retire_ratio_on_drift: float = 0.30,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = encoder or PCAEncoder(n_components=4)
        self.C = float(C)
        self.eta = float(eta)
        self.kernel = kernel
        self.gamma = gamma
        self.solver = solver
        self.max_buffer_size = int(max_buffer_size)
        self.min_buffer_size = int(min_buffer_size)
        self.drift_threshold = float(drift_threshold)
        self.warning_threshold = float(warning_threshold)
        self.lambda_base = float(lambda_base)
        self.lambda_min = float(lambda_min)
        self.lambda_age = float(lambda_age)
        self.lambda_temporal = float(lambda_temporal)
        self.lambda_dist = float(lambda_dist)
        self.retire_ratio_on_drift = float(retire_ratio_on_drift)
        self.random_state = random_state

        # Memory buffer storing (x, z, y, timestamp)
        self._buffer_X: List[np.ndarray] = []
        self._buffer_Z: List[np.ndarray] = []
        self._buffer_y: List[Union[int, float, str]] = []
        self._buffer_timestamps: List[int] = []

        self.current_time_step_: int = 0
        self.covariance_tracker_: Optional[OnlineCovarianceTracker] = None
        self.drift_detector_: Optional[LatentCovarianceDriftDetector] = None
        self.obsolescence_scorer_: Optional[ObsolescenceScorer] = None
        self.base_svm_: Optional[CovarianceAwareSVC] = None

        self.drift_events_: List[Dict[str, Any]] = []
        self.history_D_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DriftAwareCovarianceSVM":
        """
        Initial model training:
        1. Fit and freeze the representation encoder.
        2. Encode inputs X into latent representations Z.
        3. Initialize online covariance tracker and drift detector.
        4. Populate initial memory buffer.
        5. Fit initial CovarianceAwareSVC.
        """
        X_arr, y_arr = check_X_y(X, y)
        self.classes_ = np.unique(y_arr)
        # Raw input feature count (before encoding)
        self.n_features_in_ = X_arr.shape[1]

        if len(self.classes_) != 2:
            raise ValueError(f"Binary classification required. Found classes {self.classes_}")

        # 1. Representation Learning & Freezing
        if not getattr(self.encoder, "is_frozen_", False):
            self.encoder.fit(X_arr)
            self.encoder.freeze()

        # 2. Encode to latent space
        Z = self.encoder.encode(X_arr)

        # 3. Initialize tracking & scoring components
        self.covariance_tracker_ = OnlineCovarianceTracker(
            lambda_base=self.lambda_base,
            lambda_min=self.lambda_min,
        )
        self.covariance_tracker_.fit(Z)

        self.drift_detector_ = LatentCovarianceDriftDetector(
            warning_threshold=self.warning_threshold,
            drift_threshold=self.drift_threshold,
        )
        # The tracker is already estimated from the whole initial batch, so the
        # calibrated D_t is valid from the first streamed sample: skip the
        # initial warm-up (it still applies as a refractory period after each
        # confirmed drift, see learn_one). Otherwise a shift arriving with the
        # first samples is absorbed by the tracker (lambda -> lambda_min) before
        # any check is allowed to fire.
        self.drift_detector_.n_checks_ = self.drift_detector_.min_instances

        self.obsolescence_scorer_ = ObsolescenceScorer(
            lambda_age=self.lambda_age,
            lambda_temporal=self.lambda_temporal,
            lambda_dist=self.lambda_dist,
        )

        # 4. Initialize memory buffer
        self._buffer_X = [x.copy() for x in X_arr]
        self._buffer_Z = [z.copy() for z in Z]
        self._buffer_y = list(y_arr)
        self._buffer_timestamps = list(range(len(X_arr)))
        self.current_time_step_ = len(X_arr)

        # 5. Fit CovarianceAwareSVC
        self._refit_svm()

        return self

    def _refit_svm(self) -> None:
        """Refit the covariance-aware SVM on active memory buffer."""
        if len(self._buffer_Z) < self.min_buffer_size:
            return

        Z_active = np.array(self._buffer_Z)
        y_active = np.array(self._buffer_y)

        # Check binary class presence in active buffer
        active_classes = np.unique(y_active)
        if len(active_classes) < 2:
            return

        active_cov = self.covariance_tracker_.cov_ if self.covariance_tracker_ is not None else None
        # The tracker supplies only the metric Sigma^{-1}. Z is centred with the
        # buffer mean (CovarianceAwareSVC default), so H = C_N Z Sigma^{-1} Z^T C_N / d
        # satisfies H 1 = 0 and the covariance term is invariant to translations
        # of the buffer. Centring with the EW tracker mean instead would leave a
        # rank-one term in (buffer mean - tracker mean), which is non-zero after
        # pruning.
        active_mean = None

        self.base_svm_ = CovarianceAwareSVC(
            C=self.C,
            eta=self.eta,
            kernel=self.kernel,
            gamma=self.gamma,
            solver=self.solver,
            random_state=self.random_state,
        )
        self.base_svm_.fit(Z_active, y_active, cov=active_cov, mean=active_mean)

        # Sync base attributes
        self.support_vectors_ = self.base_svm_.support_vectors_
        self.dual_coef_ = self.base_svm_.dual_coef_
        self.support_ = self.base_svm_.support_
        self.intercept_ = self.base_svm_.intercept_
        self.n_support_ = self.base_svm_.n_support_

    def forget_one(self, index: int) -> "DriftAwareCovarianceSVM":
        """
        Decrementally remove a sample at index from active memory buffer.
        """
        if 0 <= index < len(self._buffer_X):
            del self._buffer_X[index]
            del self._buffer_Z[index]
            del self._buffer_y[index]
            del self._buffer_timestamps[index]
        return self

    def forget_many(self, indices: Union[np.ndarray, List[int]]) -> "DriftAwareCovarianceSVM":
        """
        Decrementally remove multiple samples from active memory.
        """
        sorted_indices = sorted(indices, reverse=True)
        for idx in sorted_indices:
            self.forget_one(idx)
        return self

    def _prune_obsolete_samples(self, n_to_remove: int) -> None:
        """Score all active samples and prune the most obsolete ones."""
        if n_to_remove <= 0 or len(self._buffer_Z) <= self.min_buffer_size:
            return

        Z_active = np.array(self._buffer_Z)
        times_active = np.array(self._buffer_timestamps)

        # Identify active support vector indices
        is_sv = np.zeros(len(self._buffer_Z), dtype=bool)
        if self.base_svm_ is not None and self.base_svm_.support_ is not None:
            valid_sv = [idx for idx in self.base_svm_.support_ if idx < len(self._buffer_Z)]
            if valid_sv:
                is_sv[valid_sv] = True

        scores = self.obsolescence_scorer_.score_samples(
            Z_stored=Z_active,
            arrival_times=times_active,
            current_time=self.current_time_step_,
            tracker=self.covariance_tracker_,
            is_support_vector=is_sv,
        )

        remove_indices = self.obsolescence_scorer_.select_obsolete_indices(scores, n_to_remove)
        self.forget_many(remove_indices)

    def learn_one(self, x: np.ndarray, y: Union[int, float, str]) -> "DriftAwareCovarianceSVM":
        """
        Incrementally learn one incoming sample (x, y):
        1. Encode x -> z using frozen encoder.
        2. Compute drift statistic D_t and update covariance tracker.
        3. Trigger decremental unlearning if drift or buffer limit exceeded.
        4. Ingest new sample (x, z, y).
        5. Update Covariance-Aware SVM.
        """
        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        z = self.encoder.encode(x_arr)[0]

        self.current_time_step_ += 1

        # Check covariance & drift
        D_t, lambda_t = self.covariance_tracker_.update(z.reshape(1, -1))
        self.history_D_.append(D_t)

        is_drift, is_warning = self.drift_detector_.update(D_t)

        if is_drift:
            # Significant drift: prune obsolete samples
            n_retire = max(1, int(len(self._buffer_Z) * self.retire_ratio_on_drift))
            self._prune_obsolete_samples(n_retire)
            self.drift_events_.append(
                {
                    "time": self.current_time_step_,
                    "D_t": D_t,
                    "p_value": self.covariance_tracker_.last_p_value_,
                    "lambda_t": lambda_t,
                    "retired_samples": n_retire,
                }
            )
            # Refractory warm-up: one regime change triggers one pruning, not
            # one per sample while the tracker catches up with the new regime.
            self.drift_detector_.reset()

        # Buffer capacity maintenance
        if len(self._buffer_X) >= self.max_buffer_size:
            self._prune_obsolete_samples(1)

        # Ingest new sample
        self._buffer_X.append(x_arr[0])
        self._buffer_Z.append(z)
        self._buffer_y.append(y)
        self._buffer_timestamps.append(self.current_time_step_)

        # Update decision boundary
        self._refit_svm()

        return self

    def learn_many(self, X: np.ndarray, y: np.ndarray) -> "DriftAwareCovarianceSVM":
        """
        Incrementally learn a batch of new incoming samples (X, y).
        """
        X_arr = check_array(X, n_features=getattr(self, "n_features_in_", None))
        y_arr = np.asarray(y)
        for xi, yi in zip(X_arr, y_arr):
            self.learn_one(xi, yi)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values for input samples X:
        x -> frozen encoder -> z -> CovarianceAwareSVC -> f(z)
        """
        check_is_fitted(self)
        X_arr = check_array(X, n_features=getattr(self, "n_features_in_", None))
        Z = self.encoder.encode(X_arr)
        return self.base_svm_.decision_function(Z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input samples X."""
        check_is_fitted(self)
        X_arr = check_array(X, n_features=getattr(self, "n_features_in_", None))
        Z = self.encoder.encode(X_arr)
        return self.base_svm_.predict(Z)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Class probabilities for input samples X: an uncalibrated logistic of
        the margin, 1 / (1 + exp(-f(x))) (see CovarianceAwareSVC.predict_proba).
        """
        check_is_fitted(self)
        X_arr = check_array(X, n_features=getattr(self, "n_features_in_", None))
        Z = self.encoder.encode(X_arr)
        return self.base_svm_.predict_proba(Z)

    def get_model_state(self) -> Dict[str, Any]:
        """
        Return the compact online model state M_t = {S_t, alpha_t, b_t, mu_t, Sigma_t, drift_state_t}.
        """
        return {
            "S_t": np.array(self._buffer_Z) if self._buffer_Z else None,
            "alpha_t": self.base_svm_.alpha_ if self.base_svm_ is not None else None,
            "b_t": self.intercept_,
            "mu_t": self.covariance_tracker_.mean_ if self.covariance_tracker_ is not None else None,
            "Sigma_t": self.covariance_tracker_.cov_ if self.covariance_tracker_ is not None else None,
            "drift_state_t": {
                "in_drift": self.drift_detector_.in_drift_zone_,
                "in_warning": self.drift_detector_.in_warning_zone_,
                "drift_count": self.drift_detector_.drift_count_,
                "last_D_t": self.covariance_tracker_.last_drift_statistic_,
                "last_lambda": self.covariance_tracker_.last_lambda_,
            },
        }

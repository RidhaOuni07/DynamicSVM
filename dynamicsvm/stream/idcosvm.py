"""
Incremental and Decremental Covariance-guided One-Class Support Vector Machine (IDCOSVM).

Combines the low-variance direction kernel covariance projection of Kefi-Fatteh et al. (2019)
with exact adiabatic incremental learning and decremental unlearning (Cauwenberghs & Poggio,
Laskov et al.) and adaptive memory obsolescence management.
"""

from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize

from dynamicsvm.base import BaseSVM
from dynamicsvm.kernels import Kernel, get_kernel
from dynamicsvm.models.cosvm import COSVM
from dynamicsvm.utils.validation import check_array, check_is_fitted


class IncrementalDecrementalCOSVM(BaseSVM):
    r"""
    Incremental and Decremental Covariance-guided One-Class Support Vector Machine (IDCOSVM).

    Maintains the optimal Karush-Kuhn-Tucker (KKT) equilibrium dynamically under both
    data influx (incremental addition) and data expiry / sample removal (decremental unlearning)
    while guiding the separating boundary via low-variance empirical covariance directions:

    $$\min_{\alpha} \frac{1}{2} \alpha^T \tilde{K} \alpha - b \left(1 - \sum_{i=1}^N \alpha_i\right)$$
    $$\text{s.t. } 0 \le \alpha_i \le C, \quad \sum_{i=1}^N \alpha_i = 1$$
    where:
    $$\tilde{K} = \eta K + (1 - \eta) K H K^T, \quad C = \frac{1}{\nu N}$$

    KKT Margin Partitions:
    - $\mathcal{S}$ (Margin SVs): $0 < \alpha_i < C$, $g_i = 0$
    - $\mathcal{E}$ (Error SVs): $\alpha_i = C$, $g_i \le 0$
    - $\mathcal{O}$ (Non SVs): $\alpha_i = 0$, $g_i \ge 0$

    Decremental Unlearning:
    - If $x_p \in \mathcal{O}$ ($\alpha_p = 0$): Instantaneous $O(1)$ pruning without disturbing remaining coefficients.
    - If $x_p \in \mathcal{S} \cup \mathcal{E}$ ($\alpha_p > 0$): Adiabatic coefficient reduction $\alpha_p \to 0$ with sensitivity
      distribution across $\mathcal{S}$ via matrix $R = \begin{bmatrix} 0 & \mathbf{1}^T \\ \mathbf{1} & \tilde{K}_{\mathcal{S},\mathcal{S}} \end{bmatrix}^{-1}$.

    Parameters
    ----------
    nu : float, default=0.1
        Outlier fraction upper bound, $\nu \in (0, 1]$.
    eta : float, default=0.5
        Tradeoff parameter in [0, 1] between standard kernel geometry ($\eta=1.0$)
        and empirical covariance projection ($\eta=0.0$).
    kernel : {'rbf', 'linear', 'poly', 'sigmoid'} or Kernel or Callable, default='rbf'
        Kernel function.
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    degree : int, default=3
        Degree for polynomial kernel.
    coef0 : float, default=0.0
        Independent term for poly/sigmoid kernel.
    max_samples : Optional[int], default=None
        Maximum buffer size. When exceeded, decremental unlearning is automatically
        triggered using `pruning_strategy`.
    pruning_strategy : {'fifo', 'non_sv_first', 'error_first', 'mahalanobis'}, default='fifo'
        Strategy used for automatic decremental removal when buffer reaches `max_samples`:
        - 'fifo': Oldest sample (FIFO sliding window).
        - 'non_sv_first': Remove non-support vectors first to retain boundary information.
        - 'error_first': Remove error support vectors first.
        - 'mahalanobis': Remove sample furthest from empirical covariance centroid.
    tol : float, default=1e-5
        Tolerance for optimization convergence and KKT margin checks.
    """

    def __init__(
        self,
        nu: float = 0.1,
        eta: float = 0.5,
        kernel: Union[str, Kernel, Callable] = "rbf",
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        max_samples: Optional[int] = None,
        pruning_strategy: Literal["fifo", "non_sv_first", "error_first", "mahalanobis"] = "fifo",
        tol: float = 1e-5,
    ):
        super().__init__()
        if not (0.0 < nu <= 1.0):
            raise ValueError(f"nu must be in (0, 1], got {nu}")
        self.nu = float(nu)
        self.eta = float(np.clip(eta, 0.0, 1.0))
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.pruning_strategy = pruning_strategy
        self.tol = float(tol)

        # Dynamic internal state
        self.kernel_: Optional[Kernel] = None
        self.X_buffer_: List[np.ndarray] = []
        self.timestamps_: List[int] = []
        self.alpha_: np.ndarray = np.array([])
        self.offset_: float = 0.0
        self.R_: Optional[np.ndarray] = None
        self.S_set_: List[int] = []  # Margin SV indices
        self.E_set_: List[int] = []  # Error SV indices
        self.O_set_: List[int] = []  # Non SV indices
        self.K_full_: Optional[np.ndarray] = None
        self.K_tilde_full_: Optional[np.ndarray] = None
        self.n_samples_seen_: int = 0
        self.n_unlearned_: int = 0

    def _compute_covariance_kernel_matrix(self, K: np.ndarray) -> np.ndarray:
        """Compute K_tilde = eta * K + (1 - eta) * K * H * K^T."""
        N = K.shape[0]
        if N <= 1:
            return K.copy()
        K_mean = np.mean(K, axis=1, keepdims=True)
        KH = K - K_mean
        KHK = np.dot(KH, K.T)
        K_tilde = self.eta * K + (1.0 - self.eta) * KHK
        return 0.5 * (K_tilde + K_tilde.T)

    def _init_R_matrix(self) -> None:
        """Initialize or update sensitivity matrix R for margin SVs."""
        if not self.S_set_:
            self.R_ = np.array([[-1e6]])
            return
        n_s = len(self.S_set_)
        M = np.zeros((n_s + 1, n_s + 1))
        M[0, 1:] = 1.0
        M[1:, 0] = 1.0
        s_idx = np.array(self.S_set_)
        M[1:, 1:] = self.K_tilde_full_[np.ix_(s_idx, s_idx)] + 1e-7 * np.eye(n_s)
        try:
            self.R_ = np.linalg.pinv(M)
        except Exception:
            self.R_ = np.linalg.lstsq(M, np.eye(n_s + 1), rcond=None)[0]

    def _sync_support_vectors(self) -> None:
        """Synchronize public attributes with current support vectors."""
        if not self.X_buffer_:
            self.support_vectors_ = None
            self.dual_coef_ = None
            self.support_ = None
            self.n_support_ = np.array([0])
            return

        sv_indices = sorted(self.S_set_ + self.E_set_)
        self.support_ = np.array(sv_indices, dtype=int)
        X_mat = np.array(self.X_buffer_)
        self.support_vectors_ = X_mat[sv_indices] if len(sv_indices) > 0 else np.empty((0, X_mat.shape[1]))
        self.dual_coef_ = self.alpha_[sv_indices] if len(sv_indices) > 0 else np.empty(0)
        self.n_support_ = np.array([len(sv_indices)])

    def _recompute_full_matrices(self) -> None:
        """Recompute full kernel and covariance matrices across active buffer."""
        if not self.X_buffer_:
            self.K_full_ = None
            self.K_tilde_full_ = None
            return
        X_mat = np.array(self.X_buffer_)
        self.K_full_ = self.kernel_(X_mat, X_mat)
        self.K_tilde_full_ = self._compute_covariance_kernel_matrix(self.K_full_)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IncrementalDecrementalCOSVM":
        """
        Fit model from an initial batch dataset X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Initial training data.
        y : Ignored
            API consistency.

        Returns
        -------
        self : IncrementalDecrementalCOSVM
        """
        X_arr = check_array(X)
        n_samples, n_features = X_arr.shape

        self.kernel_ = get_kernel(
            self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            n_features=n_features,
            X=X_arr,
        )

        batch_model = COSVM(
            nu=self.nu,
            eta=self.eta,
            kernel=self.kernel_,
            tol=self.tol,
        )
        batch_model.fit(X_arr)

        self.X_buffer_ = [X_arr[i].copy() for i in range(n_samples)]
        self.timestamps_ = list(range(n_samples))
        self.n_samples_seen_ = n_samples
        self.n_unlearned_ = 0

        full_alpha = np.zeros(n_samples)
        if batch_model.support_ is not None and batch_model.dual_coef_ is not None:
            full_alpha[batch_model.support_] = batch_model.dual_coef_

        self.alpha_ = full_alpha
        self.offset_ = batch_model.offset_
        self.intercept_ = -self.offset_

        C_val = 1.0 / (self.nu * n_samples)
        self.K_full_ = batch_model.K_train_.copy()
        self.K_tilde_full_ = batch_model.K_tilde_.copy()

        # Classify into S, E, O partitions
        self.S_set_ = []
        self.E_set_ = []
        self.O_set_ = []

        for i in range(n_samples):
            a_i = self.alpha_[i]
            if a_i > self.tol and a_i < (C_val - self.tol):
                self.S_set_.append(i)
            elif a_i >= (C_val - self.tol):
                self.E_set_.append(i)
            else:
                self.O_set_.append(i)

        if not self.S_set_ and self.E_set_:
            self.S_set_.append(self.E_set_.pop(0))

        self._init_R_matrix()
        self._sync_support_vectors()
        self._enforce_max_samples()
        return self

    def _reoptimize_equilibrium(self) -> None:
        """
        Exact equilibrium stabilization step ensuring sum(alpha)=1, box bounds, and KKT consistency.
        """
        n_samples = len(self.X_buffer_)
        if n_samples == 0:
            self.alpha_ = np.array([])
            self.S_set_ = []
            self.E_set_ = []
            self.O_set_ = []
            self.offset_ = 0.0
            self.intercept_ = 0.0
            return

        C_val = 1.0 / (self.nu * n_samples)

        def objective(alpha: np.ndarray) -> float:
            return 0.5 * float(np.dot(alpha, np.dot(self.K_tilde_full_, alpha)))

        def gradient(alpha: np.ndarray) -> np.ndarray:
            return np.dot(self.K_tilde_full_, alpha)

        eq_constraint = {
            "type": "eq",
            "fun": lambda alpha: np.sum(alpha) - 1.0,
            "jac": lambda alpha: np.ones_like(alpha),
        }

        bounds = [(0.0, C_val) for _ in range(n_samples)]
        x0 = np.clip(self.alpha_, 0.0, C_val)
        if np.sum(x0) <= 0:
            x0 = np.full(n_samples, 1.0 / n_samples)
        else:
            x0 = x0 / np.sum(x0)

        res = minimize(
            fun=objective,
            x0=x0,
            jac=gradient,
            bounds=bounds,
            constraints=[eq_constraint],
            method="SLSQP",
            options={"maxiter": 250, "ftol": self.tol},
        )

        alpha = np.maximum(0.0, res.x)
        if np.sum(alpha) > 0:
            alpha = alpha / np.sum(alpha)
        alpha[alpha < 1e-6] = 0.0
        self.alpha_ = alpha

        # Refresh S, E, O partitions
        self.S_set_ = []
        self.E_set_ = []
        self.O_set_ = []
        for i in range(n_samples):
            a_i = self.alpha_[i]
            if a_i > self.tol and a_i < (C_val - self.tol):
                self.S_set_.append(i)
            elif a_i >= (C_val - self.tol):
                self.E_set_.append(i)
            else:
                self.O_set_.append(i)

        if not self.S_set_ and (self.E_set_ or self.O_set_):
            candidates = self.E_set_ if self.E_set_ else self.O_set_
            self.S_set_.append(candidates.pop(0))

        self._init_R_matrix()
        K_tilde_alpha = np.dot(self.K_tilde_full_, self.alpha_)
        margin_mask = (self.alpha_ > 1e-5) & (self.alpha_ < (C_val - 1e-5))
        margin_idx = np.where(margin_mask)[0]
        if len(margin_idx) > 0:
            self.offset_ = float(np.mean(K_tilde_alpha[margin_idx]))
        elif self.S_set_:
            self.offset_ = float(np.mean(K_tilde_alpha[self.S_set_]))
        else:
            self.offset_ = float(np.mean(K_tilde_alpha))
        self.intercept_ = -self.offset_

    def learn_one(self, x: np.ndarray) -> "IncrementalDecrementalCOSVM":
        """
        Incrementally add and learn a single incoming observation x.

        Parameters
        ----------
        x : array-like of shape (n_features,) or (1, n_features)

        Returns
        -------
        self : IncrementalDecrementalCOSVM
        """
        x_vec = np.asarray(x).ravel()
        if not self.X_buffer_:
            return self.fit(x_vec.reshape(1, -1))

        self.n_samples_seen_ += 1
        self.X_buffer_.append(x_vec.copy())
        self.timestamps_.append(self.n_samples_seen_ - 1)
        c_idx = len(self.X_buffer_) - 1
        n_samples = len(self.X_buffer_)
        C_val = 1.0 / (self.nu * n_samples)

        # Update kernel matrices
        self._recompute_full_matrices()

        # Initialize candidate alpha
        self.alpha_ = np.append(self.alpha_, 0.0)

        # Compute margin gradient g_c
        g_c = float(np.dot(self.K_tilde_full_[c_idx], self.alpha_) - self.offset_)

        # KKT Check: If g_c >= 0, point naturally belongs to O (non-SV)
        if g_c >= -self.tol:
            self.O_set_.append(c_idx)
            self._sync_support_vectors()
            self._enforce_max_samples()
            return self

        # Adiabatic path step
        if not self.S_set_:
            self.S_set_.append(c_idx)
            self._init_R_matrix()
            self._reoptimize_equilibrium()
            self._sync_support_vectors()
            self._enforce_max_samples()
            return self

        s_idx = np.array(self.S_set_)
        k_s_c = self.K_tilde_full_[s_idx, c_idx]
        v = np.concatenate([[1.0], k_s_c])
        beta = -np.dot(self.R_, v)
        beta_b = beta[0]
        beta_s = beta[1:]

        gamma_c = float(self.K_tilde_full_[c_idx, c_idx] + np.dot(k_s_c, beta_s) + beta_b)
        if abs(gamma_c) < 1e-9:
            gamma_c = 1e-9

        delta_S = -g_c / gamma_c if gamma_c > 0 else C_val
        delta_E = C_val - self.alpha_[c_idx]
        delta_alpha = min(max(0.0, delta_S), delta_E)

        # Apply adiabatic displacement
        self.alpha_[c_idx] += delta_alpha
        for idx_pos, s_pos in enumerate(self.S_set_):
            self.alpha_[s_pos] += beta_s[idx_pos] * delta_alpha
        self.offset_ -= beta_b * delta_alpha
        self.intercept_ = -self.offset_

        if abs(self.alpha_[c_idx] - C_val) <= self.tol:
            self.E_set_.append(c_idx)
        elif self.alpha_[c_idx] > self.tol:
            self.S_set_.append(c_idx)
            self._init_R_matrix()
        else:
            self.O_set_.append(c_idx)

        self._reoptimize_equilibrium()
        self._sync_support_vectors()
        self._enforce_max_samples()
        return self

    def learn_batch(self, X: np.ndarray) -> "IncrementalDecrementalCOSVM":
        """
        Incrementally learn a sequence of incoming observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : IncrementalDecrementalCOSVM
        """
        X_arr = check_array(X)
        for i in range(X_arr.shape[0]):
            self.learn_one(X_arr[i])
        return self

    def learn_many(self, X: np.ndarray) -> "IncrementalDecrementalCOSVM":
        """Alias for learn_batch."""
        return self.learn_batch(X)

    def unlearn_one(self, index: int) -> "IncrementalDecrementalCOSVM":
        """
        Decrementally remove and unlearn an instance at a specific buffer index.

        Parameters
        ----------
        index : int
            Index of the sample in active memory buffer to unlearn.

        Returns
        -------
        self : IncrementalDecrementalCOSVM
        """
        n_samples = len(self.X_buffer_)
        if index < 0 or index >= n_samples:
            raise IndexError(f"Index {index} out of bounds for buffer of size {n_samples}")

        # Check if sample was in O (non-SV with alpha=0)
        is_non_sv = (index in self.O_set_) or (self.alpha_[index] <= self.tol)

        # Remove from buffer and metadata
        del self.X_buffer_[index]
        del self.timestamps_[index]
        self.alpha_ = np.delete(self.alpha_, index)
        self.n_unlearned_ += 1

        if len(self.X_buffer_) == 0:
            self.alpha_ = np.array([])
            self.S_set_ = []
            self.E_set_ = []
            self.O_set_ = []
            self._sync_support_vectors()
            return self

        self._recompute_full_matrices()

        # If it was an inactive non-SV, remaining weights are already balanced; quick re-index
        if is_non_sv:
            # Shift remaining indices
            self.S_set_ = [i if i < index else i - 1 for i in self.S_set_ if i != index]
            self.E_set_ = [i if i < index else i - 1 for i in self.E_set_ if i != index]
            self.O_set_ = [i if i < index else i - 1 for i in self.O_set_ if i != index]
            # Normalization check
            if np.sum(self.alpha_) > 0:
                self.alpha_ = self.alpha_ / np.sum(self.alpha_)
            self._init_R_matrix()
            self._sync_support_vectors()
            return self

        # For margin or error SVs, restore full KKT equilibrium
        self._reoptimize_equilibrium()
        self._sync_support_vectors()
        return self

    def forget_one(self, index: int) -> "IncrementalDecrementalCOSVM":
        """Alias for unlearn_one."""
        return self.unlearn_one(index)

    def unlearn_sample(self, x: np.ndarray, tol: float = 1e-4) -> "IncrementalDecrementalCOSVM":
        """
        Find and decrementally unlearn a sample matching vector x.

        Parameters
        ----------
        x : array-like of shape (n_features,)
        tol : float, default=1e-4
            Euclidean distance tolerance to match vector.

        Returns
        -------
        self : IncrementalDecrementalCOSVM
        """
        x_vec = np.asarray(x).ravel()
        for idx, buf_x in enumerate(self.X_buffer_):
            if np.linalg.norm(buf_x - x_vec) <= tol:
                return self.unlearn_one(idx)
        raise ValueError("Sample not found in active memory buffer.")

    def unlearn_many(self, indices: Union[List[int], np.ndarray]) -> "IncrementalDecrementalCOSVM":
        """
        Decrementally remove multiple instances given by indices.

        Parameters
        ----------
        indices : array-like of int
            List or array of buffer indices to remove.

        Returns
        -------
        self : IncrementalDecrementalCOSVM
        """
        sorted_indices = sorted(indices, reverse=True)
        for idx in sorted_indices:
            self.unlearn_one(idx)
        return self

    def forget_many(self, indices: Union[List[int], np.ndarray]) -> "IncrementalDecrementalCOSVM":
        """Alias for unlearn_many."""
        return self.unlearn_many(indices)

    def select_obsolete_indices(
        self,
        n_samples: int = 1,
        strategy: Optional[Literal["fifo", "non_sv_first", "error_first", "mahalanobis"]] = None,
    ) -> List[int]:
        """
        Select the most obsolete candidate buffer indices for decremental removal.

        Parameters
        ----------
        n_samples : int, default=1
            Number of sample indices to select.
        strategy : {'fifo', 'non_sv_first', 'error_first', 'mahalanobis'}, optional
            Pruning strategy. Defaults to instance's `self.pruning_strategy`.

        Returns
        -------
        selected_indices : list of int
        """
        strat = strategy if strategy is not None else self.pruning_strategy
        buf_len = len(self.X_buffer_)
        if buf_len == 0 or n_samples <= 0:
            return []

        k = min(n_samples, buf_len)

        if strat == "fifo":
            # Oldest first (indices 0, 1, ..., k-1)
            return list(range(k))

        elif strat == "non_sv_first":
            # Select non-SVs first, then oldest SVs
            candidates = list(self.O_set_)
            if len(candidates) < k:
                remaining = [i for i in range(buf_len) if i not in candidates]
                candidates.extend(remaining)
            return candidates[:k]

        elif strat == "error_first":
            # Select error SVs first (outliers/noise), then non-SVs, then margin SVs
            candidates = list(self.E_set_)
            if len(candidates) < k:
                candidates.extend(self.O_set_)
            if len(candidates) < k:
                remaining = [i for i in range(buf_len) if i not in candidates]
                candidates.extend(remaining)
            return candidates[:k]

        elif strat == "mahalanobis":
            # Measure distance from empirical center in kernel covariance space
            X_mat = np.array(self.X_buffer_)
            mean_x = np.mean(X_mat, axis=0)
            diff = X_mat - mean_x
            if X_mat.shape[0] > X_mat.shape[1]:
                cov = np.cov(X_mat, rowvar=False) + 1e-4 * np.eye(X_mat.shape[1])
                try:
                    inv_cov = np.linalg.pinv(cov)
                    dists = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
                except Exception:
                    dists = np.sum(diff ** 2, axis=1)
            else:
                dists = np.sum(diff ** 2, axis=1)

            # Sort descending (furthest/most anomalous first)
            ranked = np.argsort(-dists)
            return list(ranked[:k])

        else:
            return list(range(k))

    def unlearn_obsolete(
        self,
        n_samples: int = 1,
        strategy: Optional[Literal["fifo", "non_sv_first", "error_first", "mahalanobis"]] = None,
    ) -> "IncrementalDecrementalCOSVM":
        """
        Decrementally prune obsolete samples according to selected policy.

        Parameters
        ----------
        n_samples : int, default=1
            Number of obsolete instances to unlearn.
        strategy : optional
            Policy strategy.

        Returns
        -------
        self : IncrementalDecrementalCOSVM
        """
        indices_to_drop = self.select_obsolete_indices(n_samples=n_samples, strategy=strategy)
        return self.unlearn_many(indices_to_drop)

    def _enforce_max_samples(self) -> None:
        """Enforce maximum active buffer constraint by unlearning obsolete samples."""
        if self.max_samples is not None and len(self.X_buffer_) > self.max_samples:
            excess = len(self.X_buffer_) - self.max_samples
            self.unlearn_obsolete(n_samples=excess, strategy=self.pruning_strategy)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate scoring values (f(x) before offset) for test instances X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        X_arr = check_array(X)
        X_train_mat = np.array(self.X_buffer_)

        K_test = self.kernel_(X_arr, X_train_mat)
        if self.eta < 1.0 and X_train_mat.shape[0] > 1:
            K_test_H = K_test - np.mean(K_test, axis=1, keepdims=True)
            K_test_cov = np.dot(K_test_H, self.K_full_.T)
            K_tilde_test = self.eta * K_test + (1.0 - self.eta) * K_test_cov
        else:
            K_tilde_test = K_test

        return np.dot(K_tilde_test, self.alpha_)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate signed distance to separating decision boundary: f(x) = score(x) - offset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        dec : ndarray of shape (n_samples,)
        """
        return self.score_samples(X) - self.offset_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform one-class classification on samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            +1 for normal/target samples, -1 for outliers/anomalies.
        """
        dec = self.decision_function(X)
        return np.where(dec >= 0.0, 1, -1)

    def get_model_state(self) -> Dict[str, Union[int, float, List[int]]]:
        """
        Retrieve dictionary snapshot of model's dynamic streaming state.

        Returns
        -------
        state : dict
        """
        return {
            "buffer_size": len(self.X_buffer_),
            "n_samples_seen": self.n_samples_seen_,
            "n_unlearned": self.n_unlearned_,
            "n_support_vectors": int(len(self.support_)) if self.support_ is not None else 0,
            "n_margin_svs": len(self.S_set_),
            "n_error_svs": len(self.E_set_),
            "n_non_svs": len(self.O_set_),
            "offset": float(self.offset_),
            "eta": float(self.eta),
            "nu": float(self.nu),
            "pruning_strategy": self.pruning_strategy,
        }


# Short alias
IDCOSVM = IncrementalDecrementalCOSVM

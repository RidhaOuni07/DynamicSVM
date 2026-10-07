"""
LASVM: Online and Active Support Vector Machine with Support Vector Filtering.

Based on Antoine Bordes, Seyda Ertekin, Jason Weston, and Léon Bottou (JMLR 2005):
'Fast Kernel Classifiers with Online and Active Learning'.
"""

from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np

from dynamicsvm.base import BaseSVM
from dynamicsvm.kernels import Kernel, get_kernel
from dynamicsvm.utils.validation import check_array, check_X_y, check_is_fitted


class LASVM(BaseSVM):
    r"""
    LASVM: Fast Online & Active Support Vector Classifier with Support Vector Filtering.

    LASVM alternates between:
    1. `PROCESS(k_t)`: Online gradient initialization and selective SMO update on $\tau$-violating pairs.
    2. `REPROCESS()`: Working set directional optimization with active removal of non-support vectors ($\alpha_i = 0$).
    3. `FINISH()`: Convergence polish with strict KKT violation tolerances.
    4. Active Learning Selection: Online sample querying based on minimum margin distance $|f(x)|$.

    Mathematical Formulation:
    $$\max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$
    $$\text{s.t. } 0 \le \alpha_i \le C, \quad \sum_i \alpha_i y_i = 0$$

    Gradient tracking:
    $$g_i = y_i - \sum_{j \in \mathcal{S}} \alpha_j y_j K(x_i, x_j)$$
    $$I_{\text{up}} = \{ i \in \mathcal{S} \mid (\alpha_i < C, y_i = 1) \lor (\alpha_i > 0, y_i = -1) \}$$
    $$I_{\text{low}} = \{ i \in \mathcal{S} \mid (\alpha_i > 0, y_i = 1) \lor (\alpha_i < C, y_i = -1) \}$$

    $\tau$-violating condition:
    $$\max_{i \in I_{\text{up}}} g_i - \min_{j \in I_{\text{low}}} g_j > \tau$$

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'} or Kernel or Callable, default='rbf'
        Kernel function.
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    degree : int, default=3
        Degree for polynomial kernel.
    coef0 : float, default=0.0
        Independent term for poly/sigmoid kernel.
    tol : float, default=1e-3
        KKT violation tolerance $\tau$.
    max_iter : int, default=1000
        Maximum iterations for finishing polish.
    epochs : int, default=1
        Number of online passes over training data.
    selection_strategy : {'online', 'random', 'active'}, default='online'
        Sampling policy:
        - 'online': Process incoming samples sequentially.
        - 'random': Process samples in random order.
        - 'active': Active learning selecting samples closest to current hyperplane ($|f(x)|$).
    active_pool_size : int, default=50
        Candidate pool size for active selection queries.
    random_state : Optional[int], default=None
        Random state seed.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: Union[str, Kernel, Callable] = "rbf",
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 1000,
        epochs: int = 1,
        selection_strategy: Literal["online", "random", "active"] = "online",
        active_pool_size: int = 50,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        if C <= 0:
            raise ValueError(f"C must be strictly positive, got {C}")
        self.C = float(C)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.epochs = int(epochs)
        self.selection_strategy = selection_strategy
        self.active_pool_size = int(active_pool_size)
        self.random_state = random_state

        # Internal state
        self.kernel_: Optional[Kernel] = None
        self.SV_X_: List[np.ndarray] = []
        self.SV_y_: List[float] = []
        self.SV_alpha_: List[float] = []
        self.SV_g_: List[float] = []
        self.b_: float = 0.0
        self.n_samples_seen_: int = 0
        self.n_filtered_sv_: int = 0
        self.rng = np.random.RandomState(random_state)

    def _init_kernel(self, n_features: int, X_sample: Optional[np.ndarray] = None) -> Kernel:
        return get_kernel(
            self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            n_features=n_features,
            X=X_sample,
        )

    def _get_I_up_low(self) -> Tuple[List[int], List[int]]:
        """Identify indices in I_up and I_low sets."""
        I_up: List[int] = []
        I_low: List[int] = []
        eps = 1e-7
        for idx in range(len(self.SV_X_)):
            a = self.SV_alpha_[idx]
            y = self.SV_y_[idx]
            # I_up condition: (alpha < C and y = 1) or (alpha > 0 and y = -1)
            if (a < (self.C - eps) and y > 0) or (a > eps and y < 0):
                I_up.append(idx)
            # I_low condition: (alpha > 0 and y = 1) or (alpha < C and y = -1)
            if (a > eps and y > 0) or (a < (self.C - eps) and y < 0):
                I_low.append(idx)
        return I_up, I_low

    def _compute_b(self) -> float:
        """Calculate bias b = 0.5 * (max_{i in I_up} g_i + min_{j in I_low} g_j)."""
        I_up, I_low = self._get_I_up_low()
        if not I_up and not I_low:
            return 0.0
        max_up = max(self.SV_g_[i] for i in I_up) if I_up else 0.0
        min_low = min(self.SV_g_[j] for j in I_low) if I_low else 0.0
        if I_up and I_low:
            return float(0.5 * (max_up + min_low))
        elif I_up:
            return float(max_up)
        else:
            return float(min_low)

    def _smo_step(self, i: int, j: int) -> bool:
        """
        Execute analytic 2-variable SMO step on indices (i, j).
        """
        if i == j:
            return False

        xi, yi = self.SV_X_[i], self.SV_y_[i]
        xj, yj = self.SV_X_[j], self.SV_y_[j]
        ai, aj = self.SV_alpha_[i], self.SV_alpha_[j]
        gi, gj = self.SV_g_[i], self.SV_g_[j]

        # Kernel evaluations
        kii = float(self.kernel_(xi.reshape(1, -1), xi.reshape(1, -1))[0, 0])
        kjj = float(self.kernel_(xj.reshape(1, -1), xj.reshape(1, -1))[0, 0])
        kij = float(self.kernel_(xi.reshape(1, -1), xj.reshape(1, -1))[0, 0])
        eta = kii + kjj - 2.0 * kij

        if eta <= 1e-12:
            eta = 1e-12

        # Step size in dual variable space
        # Directional gradient difference
        step = (gi - gj) / eta

        # Compute bounding limits
        s = yi * yj
        if s > 0:
            # y_i == y_j: alpha_i + alpha_j = const
            # delta_alpha_i + delta_alpha_j = 0  => delta_alpha_i = -delta_alpha_j
            # For yi = 1: alpha_i increases, alpha_j decreases
            # lambda = min(step, C - ai, aj)
            if yi > 0:
                delta = min(max(0.0, step), self.C - ai, aj)
                new_ai = ai + delta
                new_aj = aj - delta
            else:
                delta = min(max(0.0, step), ai, self.C - aj)
                new_ai = ai - delta
                new_aj = aj + delta
        else:
            # y_i != y_j: alpha_i - alpha_j = const
            # delta_alpha_i = delta_alpha_j
            if yi > 0:
                delta = min(max(0.0, step), self.C - ai, self.C - aj)
                new_ai = ai + delta
                new_aj = aj + delta
            else:
                delta = min(max(0.0, step), ai, aj)
                new_ai = ai - delta
                new_aj = aj - delta

        diff_i = new_ai - ai
        diff_j = new_aj - aj

        if abs(diff_i) < 1e-10 and abs(diff_j) < 1e-10:
            return False

        # Apply update
        self.SV_alpha_[i] = float(np.clip(new_ai, 0.0, self.C))
        self.SV_alpha_[j] = float(np.clip(new_aj, 0.0, self.C))

        # Update gradients across working set S
        # g_k = y_k - sum_s alpha_s y_s K(x_k, x_s)
        # Delta g_k = - (Delta alpha_i * y_i * K(x_k, x_i) + Delta alpha_j * y_j * K(x_k, x_j))
        X_mat = np.array(self.SV_X_)
        k_all_i = self.kernel_(X_mat, xi.reshape(1, -1)).ravel()
        k_all_j = self.kernel_(X_mat, xj.reshape(1, -1)).ravel()

        delta_g = -(diff_i * yi * k_all_i + diff_j * yj * k_all_j)
        for idx in range(len(self.SV_g_)):
            self.SV_g_[idx] += float(delta_g[idx])

        self.b_ = self._compute_b()
        return True

    def process(self, x: np.ndarray, y: float) -> bool:
        """
        LASVM PROCESS step: Add sample (x, y) with alpha=0, compute gradient,
        and perform selective SMO step with extreme violator.
        """
        x_vec = np.asarray(x, dtype=float).ravel()
        y_val = float(y)

        # Compute initial gradient: g = y - sum_{s in S} alpha_s y_s K(x, x_s)
        if len(self.SV_X_) > 0:
            X_mat = np.array(self.SV_X_)
            k_row = self.kernel_(x_vec.reshape(1, -1), X_mat).ravel()
            alpha_arr = np.array(self.SV_alpha_)
            y_arr = np.array(self.SV_y_)
            f_val = np.sum(alpha_arr * y_arr * k_row)
            g_val = y_val - f_val
        else:
            g_val = y_val

        k_idx = len(self.SV_X_)
        self.SV_X_.append(x_vec.copy())
        self.SV_y_.append(y_val)
        self.SV_alpha_.append(0.0)
        self.SV_g_.append(float(g_val))

        I_up, I_low = self._get_I_up_low()

        if y_val > 0:
            i = k_idx
            if not I_low:
                return False
            j = min(I_low, key=lambda idx: self.SV_g_[idx])
        else:
            j = k_idx
            if not I_up:
                return False
            i = max(I_up, key=lambda idx: self.SV_g_[idx])

        # Check tau-violating condition
        if (self.SV_g_[i] - self.SV_g_[j]) > self.tol:
            return self._smo_step(i, j)
        return False

    def reprocess(self) -> bool:
        """
        LASVM REPROCESS step: Find extreme violating pair in active SV set S,
        optimize pair, and filter out non-violating zero-alpha points.
        """
        I_up, I_low = self._get_I_up_low()
        if not I_up or not I_low:
            return False

        i = max(I_up, key=lambda idx: self.SV_g_[idx])
        j = min(I_low, key=lambda idx: self.SV_g_[idx])

        if (self.SV_g_[i] - self.SV_g_[j]) > self.tol:
            success = self._smo_step(i, j)
            self._filter_support_vectors()
            return success

        self._filter_support_vectors()
        return False

    def _filter_support_vectors(self) -> None:
        """
        Support Vector Filtering: Remove points with alpha = 0 that satisfy KKT.
        """
        if len(self.SV_X_) <= 2:
            return

        b_val = self.b_
        eps = 1e-7
        keep_indices: List[int] = []

        for idx in range(len(self.SV_X_)):
            a = self.SV_alpha_[idx]
            y = self.SV_y_[idx]
            g = self.SV_g_[idx]

            if a > eps:
                keep_indices.append(idx)
            else:
                # alpha == 0
                # If y = 1 and g < b_val + tol: non-violating, can be pruned
                # If y = -1 and g > b_val - tol: non-violating, can be pruned
                is_violating = (y > 0 and g > (b_val + self.tol)) or (y < 0 and g < (b_val - self.tol))
                if is_violating:
                    keep_indices.append(idx)
                else:
                    self.n_filtered_sv_ += 1

        if len(keep_indices) < len(self.SV_X_) and len(keep_indices) >= 2:
            self.SV_X_ = [self.SV_X_[k] for k in keep_indices]
            self.SV_y_ = [self.SV_y_[k] for k in keep_indices]
            self.SV_alpha_ = [self.SV_alpha_[k] for k in keep_indices]
            self.SV_g_ = [self.SV_g_[k] for k in keep_indices]

    def finish(self) -> "LASVM":
        """
        LASVM FINISH step: Repeat reprocess until working set S satisfies KKT tolerance.
        """
        for _ in range(self.max_iter):
            I_up, I_low = self._get_I_up_low()
            if not I_up or not I_low:
                break
            i = max(I_up, key=lambda idx: self.SV_g_[idx])
            j = min(I_low, key=lambda idx: self.SV_g_[idx])
            if (self.SV_g_[i] - self.SV_g_[j]) <= self.tol:
                break
            self._smo_step(i, j)
            self._filter_support_vectors()

        # Final purge of zero alpha support vectors
        final_keep = [k for k in range(len(self.SV_X_)) if self.SV_alpha_[k] > 1e-7]
        if len(final_keep) > 0:
            self.SV_X_ = [self.SV_X_[k] for k in final_keep]
            self.SV_y_ = [self.SV_y_[k] for k in final_keep]
            self.SV_alpha_ = [self.SV_alpha_[k] for k in final_keep]
            self.SV_g_ = [self.SV_g_[k] for k in final_keep]

        self.b_ = self._compute_b()
        self._sync_support_vectors()
        return self

    def _sync_support_vectors(self) -> None:
        """Synchronize public attributes with current support vector sets."""
        if not self.SV_X_:
            self.support_vectors_ = None
            self.dual_coef_ = None
            self.support_ = None
            self.intercept_ = float(self.b_)
            self.n_support_ = np.array([0, 0], dtype=int)
            return

        X_mat = np.array(self.SV_X_)
        y_arr = np.array(self.SV_y_)
        alpha_arr = np.array(self.SV_alpha_)

        sv_mask = alpha_arr > 1e-7
        if np.any(sv_mask):
            self.support_vectors_ = X_mat[sv_mask]
            self.dual_coef_ = (alpha_arr[sv_mask] * y_arr[sv_mask]).reshape(1, -1)
            self.support_ = np.where(sv_mask)[0]
            sv_y = y_arr[sv_mask]
            n_neg = int(np.sum(sv_y == -1.0))
            n_pos = int(np.sum(sv_y == 1.0))
            self.n_support_ = np.array([n_neg, n_pos], dtype=int)
        else:
            self.support_vectors_ = np.empty((0, X_mat.shape[1]))
            self.dual_coef_ = np.empty((1, 0))
            self.support_ = np.array([], dtype=int)
            self.n_support_ = np.array([0, 0], dtype=int)

        self.intercept_ = float(self.b_)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LASVM":
        """
        Fit LASVM on training dataset with online or active learning passes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        X_arr, y_arr = check_X_y(X, y)
        self.classes_ = np.unique(y_arr)
        if len(self.classes_) != 2:
            raise ValueError(f"LASVM requires binary classification (found {len(self.classes_)} classes)")

        y_encoded = np.where(y_arr == self.classes_[1], 1.0, -1.0)
        n_samples, n_features = X_arr.shape

        self.kernel_ = self._init_kernel(n_features, X_arr)
        self.SV_X_ = []
        self.SV_y_ = []
        self.SV_alpha_ = []
        self.SV_g_ = []
        self.n_samples_seen_ = 0
        self.n_filtered_sv_ = 0

        unseen_indices = list(range(n_samples))

        for epoch in range(self.epochs):
            if self.selection_strategy == "random":
                order = self.rng.permutation(n_samples)
                for idx in order:
                    self.process(X_arr[idx], y_encoded[idx])
                    self.reprocess()
                    self.n_samples_seen_ += 1
            elif self.selection_strategy == "active":
                # Active selection: query pool based on |f(x)|
                pool = list(range(n_samples))
                while pool:
                    if len(self.SV_X_) >= 2 and len(pool) > self.active_pool_size:
                        cand_pool = self.rng.choice(pool, size=self.active_pool_size, replace=False)
                        f_scores = [abs(float(self.decision_function(X_arr[c].reshape(1, -1))[0])) for c in cand_pool]
                        best_cand = cand_pool[int(np.argmin(f_scores))]
                        pool.remove(best_cand)
                        idx = best_cand
                    else:
                        idx = pool.pop(0)

                    self.process(X_arr[idx], y_encoded[idx])
                    self.reprocess()
                    self.n_samples_seen_ += 1
            else:
                # Default 'online' sequential pass
                for idx in range(n_samples):
                    self.process(X_arr[idx], y_encoded[idx])
                    self.reprocess()
                    self.n_samples_seen_ += 1

        self.finish()
        return self

    def learn_one(self, x: np.ndarray, y: Union[int, float, str]) -> "LASVM":
        """
        Incrementally stream a new sample into LASVM.
        """
        x_vec = np.asarray(x, dtype=float).ravel()
        if not hasattr(self, "classes_") or self.classes_ is None:
            self.classes_ = np.array([-1.0, 1.0])
            self.kernel_ = self._init_kernel(len(x_vec), x_vec.reshape(1, -1))

        y_enc = 1.0 if y == self.classes_[1] else -1.0
        self.process(x_vec, y_enc)
        self.reprocess()
        self.n_samples_seen_ += 1
        self._sync_support_vectors()
        return self

    def learn_many(self, X: np.ndarray, y: np.ndarray) -> "LASVM":
        """
        Incrementally update the SVM model with a batch of new incoming instances.
        """
        for xi, yi in zip(X, y):
            self.learn_one(xi, yi)
        return self

    def forget_one(self, index: int = 0) -> "LASVM":
        """
        Remove support vector at index and restore equilibrium.
        """
        if 0 <= index < len(self.SV_X_):
            del self.SV_X_[index]
            del self.SV_y_[index]
            del self.SV_alpha_[index]
            del self.SV_g_[index]
            self.finish()
        return self

    def forget_many(self, indices: np.ndarray) -> "LASVM":
        """
        Decrementally remove/unlearn multiple instances from the model's active memory.
        """
        sorted_indices = sorted(indices, reverse=True)
        for idx in sorted_indices:
            self.forget_one(idx)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate decision function f(X) = sum_i alpha_i y_i K(x_i, X) + b.
        """
        X_arr = check_array(X)
        if not self.SV_X_:
            return np.zeros(X_arr.shape[0])

        alpha_arr = np.array(self.SV_alpha_)
        y_arr = np.array(self.SV_y_)
        sv_mask = alpha_arr > 1e-7

        if not np.any(sv_mask):
            return np.full(X_arr.shape[0], self.b_)

        sv_X = np.array(self.SV_X_)[sv_mask]
        sv_dual = alpha_arr[sv_mask] * y_arr[sv_mask]

        K_test = self.kernel_(X_arr, sv_X)
        decision = np.dot(K_test, sv_dual) + self.b_
        return decision

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform binary classification."""
        decision = self.decision_function(X)
        pred_idx = np.where(decision >= 0, 1, 0)
        return self.classes_[pred_idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities via logistic sigmoid."""
        df = self.decision_function(X)
        prob_pos = 1.0 / (1.0 + np.exp(-np.clip(df, -50.0, 50.0)))
        prob_neg = 1.0 - prob_pos
        return np.column_stack([prob_neg, prob_pos])

    def get_model_state(self) -> Dict[str, Union[int, float]]:
        """Return diagnostic metrics of the current model state."""
        return {
            "n_support_vectors": len(self.SV_X_),
            "b": float(self.b_),
            "n_samples_seen": self.n_samples_seen_,
            "n_filtered_sv": self.n_filtered_sv_,
        }


# Academic and practical alias
LASVMClassifier = LASVM

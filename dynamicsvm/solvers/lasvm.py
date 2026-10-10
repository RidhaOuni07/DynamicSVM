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

    Support vector filtering (Bordes et al. 2005): with $g_i = \max_{I_{\text{up}}} g$
    and $g_j = \min_{I_{\text{low}}} g$, a point $s$ with $\alpha_s = 0$ is removed
    only if $(y_s = +1, g_s \le g_j)$ or $(y_s = -1, g_s \ge g_i)$, i.e. it cannot
    form a $\tau$-violating pair.

    Every working-set entry carries a stable sample id (``SV_ids_``): the
    training index in ``fit`` and a running counter in ``learn_one``. ``fit``
    never inserts an id that is already in the working set (a second copy
    would let one sample carry a total weight of up to 2C), and after the
    online epochs it runs at most 10 full-set passes that PROCESS every
    training point that is $\tau$-violating against the current working set.

    Decremental unlearning (``forget_one`` / ``forget_sample_id``) first
    transfers $\alpha_p$ to partner multipliers along feasible directions
    (keeping $\sum_i \alpha_i y_i = 0$ and the cached gradients exact), then
    deletes the point, whose $\alpha_p$ is now exactly 0, and runs FINISH.

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
        self.SV_ids_: List[int] = []
        self._next_id_: int = 0
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
        # Multipliers are snapped exactly onto 0 / C in _smo_step, so a
        # round-off sized margin is enough to identify the bounds.
        eps = 1e-12 * self.C
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

        if delta <= 0.0:
            return False

        # Snap round-off neighbours of the bounds exactly onto the bounds
        snap = 1e-12 * self.C
        new_ai = float(min(max(new_ai, 0.0), self.C))
        new_aj = float(min(max(new_aj, 0.0), self.C))
        if new_ai < snap:
            new_ai = 0.0
        elif new_ai > self.C - snap:
            new_ai = self.C
        if new_aj < snap:
            new_aj = 0.0
        elif new_aj > self.C - snap:
            new_aj = self.C

        diff_i = new_ai - ai
        diff_j = new_aj - aj

        if diff_i == 0.0 and diff_j == 0.0:
            return False

        # Apply update (the gradient update below uses the applied differences,
        # so the cached g stay consistent with alpha)
        self.SV_alpha_[i] = new_ai
        self.SV_alpha_[j] = new_aj

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

    def _keep_working_set(self, keep_indices: List[int]) -> None:
        """Restrict every parallel working-set list to ``keep_indices``."""
        self.SV_X_ = [self.SV_X_[k] for k in keep_indices]
        self.SV_y_ = [self.SV_y_[k] for k in keep_indices]
        self.SV_alpha_ = [self.SV_alpha_[k] for k in keep_indices]
        self.SV_g_ = [self.SV_g_[k] for k in keep_indices]
        self.SV_ids_ = [self.SV_ids_[k] for k in keep_indices]

    def process(self, x: np.ndarray, y: float, sample_id: Optional[int] = None) -> bool:
        """
        LASVM PROCESS step: Add sample (x, y) with alpha=0, compute gradient,
        and perform selective SMO step with extreme violator.

        ``sample_id`` is the stable id stored in ``SV_ids_`` (a running
        counter is used when it is None).
        """
        x_vec = np.asarray(x, dtype=float).ravel()
        y_val = float(y)
        if sample_id is None:
            sample_id = self._next_id_
            self._next_id_ += 1

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
        self.SV_ids_.append(int(sample_id))

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
        Support Vector Filtering (Bordes et al. 2005).

        With g_i = max_{I_up} g and g_j = min_{I_low} g, an alpha = 0 point s is
        removed only if (y_s = +1 and g_s <= g_j) or (y_s = -1 and g_s >= g_i):
        such a point cannot form a tau-violating pair with any working-set
        point, and removing a zero multiplier changes neither sum(alpha y) nor
        any cached gradient.
        """
        if len(self.SV_X_) <= 2:
            return

        I_up, I_low = self._get_I_up_low()
        g_max_up = max(self.SV_g_[i] for i in I_up) if I_up else None
        g_min_low = min(self.SV_g_[j] for j in I_low) if I_low else None
        keep_indices: List[int] = []
        n_removed = 0

        for idx in range(len(self.SV_X_)):
            a = self.SV_alpha_[idx]
            y = self.SV_y_[idx]
            g = self.SV_g_[idx]

            if a > 0.0:
                keep_indices.append(idx)
                continue
            removable = (y > 0 and g_min_low is not None and g <= g_min_low) or (
                y < 0 and g_max_up is not None and g >= g_max_up
            )
            if removable:
                n_removed += 1
            else:
                keep_indices.append(idx)

        if n_removed > 0 and len(keep_indices) >= 2:
            self.n_filtered_sv_ += n_removed
            self._keep_working_set(keep_indices)

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
            if not self._smo_step(i, j):
                break  # the maximal violating pair cannot move: no further progress
            self._filter_support_vectors()

        # Final purge of zero-alpha points (alpha is exactly 0 after snapping,
        # so neither sum(alpha y) nor the cached gradients change)
        final_keep = [k for k in range(len(self.SV_X_)) if self.SV_alpha_[k] > 0.0]
        if len(final_keep) > 0:
            self._keep_working_set(final_keep)

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

        sv_mask = alpha_arr > 0.0
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
        self.n_features_in_ = n_features

        self.kernel_ = self._init_kernel(n_features, X_arr)
        self.SV_X_ = []
        self.SV_y_ = []
        self.SV_alpha_ = []
        self.SV_g_ = []
        self.SV_ids_ = []
        self._next_id_ = n_samples  # ids 0..n-1 are the training indices
        self.n_samples_seen_ = 0
        self.n_filtered_sv_ = 0

        def visit(idx: int) -> None:
            # A sample already in the working set is never inserted twice
            # (duplicates would let it carry a total weight of up to 2C).
            if int(idx) not in self.SV_ids_:
                self.process(X_arr[idx], y_encoded[idx], sample_id=int(idx))
            self.reprocess()
            self.n_samples_seen_ += 1

        for epoch in range(self.epochs):
            if self.selection_strategy == "random":
                order = self.rng.permutation(n_samples)
                for idx in order:
                    visit(idx)
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

                    visit(idx)
            else:
                # Default 'online' sequential pass
                for idx in range(n_samples):
                    visit(idx)

        self.finish()
        self._full_set_passes(X_arr, y_encoded)
        return self

    def _full_set_passes(self, X_arr: np.ndarray, y_encoded: np.ndarray, max_passes: int = 10) -> None:
        """
        Final full-set optimisation passes (does not change ``n_samples_seen_``).

        Every training point outside the working set has alpha = 0, so its
        gradient is g_k = y_k - sum_{s in S} a_s y_s K(x_k, x_s). It is
        tau-violating if (y_k = +1 and g_k - min_{I_low} g > tol) or
        (y_k = -1 and max_{I_up} g - g_k > tol). Violators are PROCESSed (by
        id) and FINISH is run again, at most ``max_passes`` times.
        """
        for _ in range(max_passes):
            if not self.SV_X_:
                return
            in_set = set(self.SV_ids_)
            outside = np.array([k for k in range(len(X_arr)) if k not in in_set], dtype=int)
            if outside.size == 0:
                return
            I_up, I_low = self._get_I_up_low()
            if not I_up or not I_low:
                return
            g_max_up = max(self.SV_g_[i] for i in I_up)
            g_min_low = min(self.SV_g_[j] for j in I_low)

            ay = np.array(self.SV_alpha_) * np.array(self.SV_y_)
            K_out = self.kernel_(X_arr[outside], np.array(self.SV_X_))
            g_out = y_encoded[outside] - K_out @ ay
            y_out = y_encoded[outside]
            violating = ((y_out > 0) & (g_out - g_min_low > self.tol)) | (
                (y_out < 0) & (g_max_up - g_out > self.tol)
            )
            if not np.any(violating):
                return
            for k in outside[violating]:
                if int(k) not in self.SV_ids_:
                    self.process(X_arr[k], y_encoded[k], sample_id=int(k))
            self.finish()

    def learn_one(self, x: np.ndarray, y: Union[int, float, str]) -> "LASVM":
        """
        Incrementally stream a new sample into LASVM.
        """
        x_vec = np.asarray(x, dtype=float).ravel()
        if not hasattr(self, "classes_") or self.classes_ is None:
            self.classes_ = np.array([-1.0, 1.0])
            self.n_features_in_ = len(x_vec)
            # Single-vector start: get_kernel freezes gamma='scale' to 1/n_features (with a warning)
            self.kernel_ = self._init_kernel(len(x_vec), x_vec.reshape(1, -1))

        y_enc = 1.0 if y == self.classes_[1] else -1.0
        self.process(x_vec, y_enc, sample_id=self._next_id_)
        self._next_id_ += 1
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
        Decrementally remove the working-set entry at position ``index``.

        ``index`` is the position in the current working set (``SV_X_``,
        ``SV_ids_``), NOT the original sample index; use
        :meth:`forget_sample_id` to remove a sample by its stable id.

        Before deletion alpha_p is transferred to partner multipliers q along
        feasible directions Delta a_p = -delta, Delta a_q = y_p y_q delta (so
        sum(alpha y) is unchanged): opposite-class q with a_q > 0 (a_q
        decreases) or same-class q with a_q < C (a_q increases). With the
        minimisation dual D, dD/d delta = y_p (g_p - g_q), so q is the
        candidate with the largest g_q if y_p = +1 and the smallest g_q if
        y_p = -1; delta = min(a_p, capacity of q). Cached gradients are kept
        exact by the same rank-2 update as in the SMO step. A partner always
        exists while a_p > 0 because sum_{i != p} a_i y_i = -a_p y_p. Once a_p
        is exactly 0 the point is deleted (no gradient term remains) and
        FINISH restores the KKT conditions on the working set.
        """
        if not (0 <= index < len(self.SV_X_)):
            return self

        p = int(index)
        y_p = self.SV_y_[p]
        X_mat = np.array(self.SV_X_)
        k_p = self.kernel_(X_mat, self.SV_X_[p].reshape(1, -1)).ravel()

        while self.SV_alpha_[p] > 0.0:
            best_q = -1
            best_cap = 0.0
            for q in range(len(self.SV_X_)):
                if q == p:
                    continue
                a_q, y_q = self.SV_alpha_[q], self.SV_y_[q]
                cap = a_q if y_q != y_p else self.C - a_q
                if cap <= 0.0:
                    continue
                if best_q < 0:
                    better = True
                elif y_p > 0:
                    better = self.SV_g_[q] > self.SV_g_[best_q]
                else:
                    better = self.SV_g_[q] < self.SV_g_[best_q]
                if better:
                    best_q, best_cap = q, cap
            if best_q < 0:
                break  # infeasible input (sum(alpha y) != 0); nothing left to transfer to

            q = best_q
            y_q = self.SV_y_[q]
            a_p_old = self.SV_alpha_[p]
            delta = min(a_p_old, best_cap)
            new_ap = 0.0 if delta == a_p_old else a_p_old - delta
            a_q_old = self.SV_alpha_[q]
            new_aq = a_q_old + y_p * y_q * delta
            if delta == best_cap:
                new_aq = 0.0 if y_q != y_p else self.C  # q reaches its bound exactly
            diff_p = new_ap - a_p_old
            diff_q = new_aq - a_q_old
            self.SV_alpha_[p] = new_ap
            self.SV_alpha_[q] = new_aq

            k_q = self.kernel_(X_mat, self.SV_X_[q].reshape(1, -1)).ravel()
            delta_g = -(diff_p * y_p * k_p + diff_q * y_q * k_q)
            for idx in range(len(self.SV_g_)):
                self.SV_g_[idx] += float(delta_g[idx])

        self.SV_alpha_[p] = 0.0
        del self.SV_X_[p]
        del self.SV_y_[p]
        del self.SV_alpha_[p]
        del self.SV_g_[p]
        del self.SV_ids_[p]
        self.finish()
        return self

    def forget_sample_id(self, sample_id: int) -> "LASVM":
        """
        Decrementally remove the sample with stable id ``sample_id``.

        Ids are training indices for samples seen in ``fit`` and running
        counter values for ``learn_one``. If the id is no longer in the
        working set it was filtered out with alpha = 0; such an inactive
        point does not affect the optimum, so this is an exact no-op.
        """
        try:
            position = self.SV_ids_.index(int(sample_id))
        except ValueError:
            return self
        return self.forget_one(position)

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
        X_arr = check_array(X, n_features=getattr(self, "n_features_in_", None))
        if not self.SV_X_:
            return np.zeros(X_arr.shape[0])

        alpha_arr = np.array(self.SV_alpha_)
        y_arr = np.array(self.SV_y_)
        sv_mask = alpha_arr > 0.0

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
        """
        Uncalibrated logistic of the margin, 1 / (1 + exp(-f(x))), columns
        ordered as ``classes_``. Monotone in f but not a calibrated
        probability estimate.
        """
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

"""
Adiabatic Path Following Incremental and Decremental Support Vector Machine.

Implements the exact Kuhn-Tucker relations and continuous adiabatic parameter migration
derived by Cauwenberghs & Poggio (NeurIPS 2000 / 2001) and Laskov et al. (JMLR 2006),
for both insertion (alpha_c: 0 -> final value) and removal (alpha_p: current value -> 0).
The sensitivity matrix R is recomputed at every event (O(|S|^3)); no rank-one updates
are used. SMO is only a counted last-resort fallback (``n_smo_fallbacks_``), always
started from a feasible point.
"""

from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np

from dynamicsvm.base import BaseSVM
from dynamicsvm.kernels import Kernel, get_kernel
from dynamicsvm.solvers.smo import SMOSolver
from dynamicsvm.stream.base import StreamingSVM
from dynamicsvm.utils.kkt import compute_intercept
from dynamicsvm.utils.validation import check_array, check_X_y, check_is_fitted

# Sensitivities |beta|, |gamma| below this are treated as zero when locating events.
_SENS_EPS = 1e-12


class AdiabaticSVM(StreamingSVM):
    r"""
    Exact Incremental and Decremental Support Vector Classifier via Adiabatic Path Following.

    Maintains exact Karush-Kuhn-Tucker (KKT) equilibrium dynamically upon sample insertion
    or deletion by continuously tracking the sensitivity matrix $R$ and migrating Lagrange
    multipliers $\alpha$ and bias $b$ across the three partition sets:
    - $\mathcal{S}$ (Margin SVs): $0 < \alpha_i < C$, $g_i = 0$
    - $\mathcal{E}$ (Error SVs): $\alpha_i = C$, $g_i \le 0$
    - $\mathcal{O}$ (Non SVs): $\alpha_i = 0$, $g_i \ge 0$

    where the margin function for sample $i$ is:
    $$g_i = y_i f(x_i) - 1 = y_i \left(\sum_{j \in \mathcal{S} \cup \mathcal{E}} y_j \alpha_j K(x_i, x_j) + b\right) - 1$$

    Equilibrium Sensitivity System:
    $$\begin{bmatrix} 0 & \mathbf{y}_{\mathcal{S}}^T \\ \mathbf{y}_{\mathcal{S}} & \mathbf{Q}_{\mathcal{S}\mathcal{S}} \end{bmatrix}
      \begin{bmatrix} \Delta b \\ \Delta \boldsymbol{\alpha}_{\mathcal{S}} \end{bmatrix} =
      -\begin{bmatrix} y_c \\ \mathbf{Q}_{\mathcal{S}, c} \end{bmatrix} \Delta \alpha_c$$

    $$R = \begin{bmatrix} 0 & \mathbf{y}_{\mathcal{S}}^T \\ \mathbf{y}_{\mathcal{S}} & \mathbf{Q}_{\mathcal{S}\mathcal{S}} \end{bmatrix}^{-1}, \quad
      \boldsymbol{\beta} = -R \begin{bmatrix} y_c \\ \mathbf{Q}_{\mathcal{S}, c} \end{bmatrix}, \quad
      \gamma_i = Q_{ic} + y_i \beta_0 + \sum_{j \in \mathcal{S}} Q_{ij} \beta_j$$

    Incremental step (``learn_one``): with $\Delta\alpha_c = t \ge 0$,
    $\Delta\alpha_{\mathcal{S}} = \beta_{\mathcal{S}} t$, $\Delta b = \beta_0 t$,
    $\Delta g_i = \gamma_i t$. The step stops at the first event (c reaches C,
    $g_c$ reaches 0, an S point hits 0 or C, an E/O point reaches $g = 0$),
    the sets are updated and R is recomputed. If $\mathcal{S}$ is empty, a
    bias-only step $\Delta b = y_c t$ is taken until an O point with
    $y_i = -y_c$ or an E point with $y_i = y_c$ reaches $g = 0$ (it joins S),
    or $g_c$ reaches 0.

    Decremental step (``forget_one``): p is removed from its set,
    $\mathcal{S}' = \mathcal{S} \setminus \{p\}$ and
    $\beta = -R' [y_p; Q_{\mathcal{S}',p}]$. With $\Delta\alpha_p = -\delta$:
    $\Delta\alpha_s = -\beta_s\delta$, $\Delta b = -\beta_0\delta$,
    $\Delta g_i = -\gamma_i\delta$. Events: $\alpha_p$ reaches 0, an S' point
    hits 0 or C, an E point ($\gamma_i < 0$) or O point ($\gamma_i > 0$)
    reaches $g = 0$. With $\mathcal{S}'$ empty a bias-only step
    $\Delta b = -y_p t$ brings in an O point with $y_i = y_p$ or an E point
    with $y_i = -y_p$. p is deleted only once $\alpha_p = 0$ exactly, so
    $\sum_i \alpha_i y_i = 0$ and every other $g_i$ are unchanged by the
    deletion. If the remaining buffer has a single class, $\alpha = 0$ and
    $b = \pm 1$ (that class).

    After every set change, $(b, \alpha_{\mathcal{S}})$ is re-solved from the
    margin system ($g_{\mathcal{S}} = 0$, $\sum_i \alpha_i y_i = 0$) so round-off
    does not accumulate along the path.

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
    max_samples : Optional[int], default=None
        Maximum buffer size. When exceeded, decremental unlearning is automatically
        triggered using `pruning_strategy`.
    pruning_strategy : {'fifo', 'non_sv_first', 'error_first'}, default='fifo'
        Strategy used for automatic decremental removal when buffer reaches `max_samples`.
    tol : float, default=1e-4
        Convergence tolerance of the batch SMO solves (``fit`` and fallbacks),
        which run at ``min(tol, 1e-6)``.
    max_path_steps : int, default=500
        Maximum adiabatic migration steps per sample insertion/deletion. On
        exhaustion a feasible SMO warm start is run and counted in
        ``n_smo_fallbacks_``.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: Union[str, Kernel, Callable] = "rbf",
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        max_samples: Optional[int] = None,
        pruning_strategy: Literal["fifo", "non_sv_first", "error_first"] = "fifo",
        tol: float = 1e-4,
        max_path_steps: int = 500,
    ):
        super().__init__()
        if C <= 0:
            raise ValueError(f"C must be strictly positive, got {C}")
        self.C = float(C)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.pruning_strategy = pruning_strategy
        self.tol = float(tol)
        self.max_path_steps = int(max_path_steps)

        # Dynamic internal state
        self.kernel_: Optional[Kernel] = None
        self.X_buffer_: List[np.ndarray] = []
        self.y_buffer_: List[float] = []
        self.alpha_: np.ndarray = np.array([])
        self.b_: float = 0.0
        self.R_: Optional[np.ndarray] = None
        self.S_set_: List[int] = []  # Margin SVs: 0 < alpha_i < C, g_i = 0
        self.E_set_: List[int] = []  # Error SVs: alpha_i = C, g_i <= 0
        self.O_set_: List[int] = []  # Non SVs: alpha_i = 0, g_i >= 0
        self.K_full_: Optional[np.ndarray] = None
        self.n_samples_seen_: int = 0
        self.n_unlearned_: int = 0
        self.n_smo_fallbacks_: int = 0

    def _init_kernel(self, n_features: int, X_sample: Optional[np.ndarray] = None) -> Kernel:
        return get_kernel(
            self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            n_features=n_features,
            X=X_sample,
        )

    def _y_arr(self) -> np.ndarray:
        return np.asarray(self.y_buffer_, dtype=float)

    def _margin_matrix(self) -> np.ndarray:
        """Bordered margin matrix M = [[0, y_S^T], [y_S, Q_SS]] for the current S."""
        s_idx = np.asarray(self.S_set_, dtype=int)
        n_s = s_idx.size
        y_s = self._y_arr()[s_idx]
        M = np.zeros((n_s + 1, n_s + 1))
        M[0, 1:] = y_s
        M[1:, 0] = y_s
        M[1:, 1:] = np.outer(y_s, y_s) * self.K_full_[np.ix_(s_idx, s_idx)]
        return M

    def _compute_R_matrix(self) -> None:
        """Compute the sensitivity matrix R = M^{-1} for the current margin set S (no ridge)."""
        if not self.S_set_:
            self.R_ = np.array([[0.0]])
            return
        self.R_ = np.linalg.pinv(self._margin_matrix())

    def _project_margin_set(self) -> None:
        """
        Re-solve (b, alpha_S) from g_S = 0 and sum(alpha y) = 0 with every
        alpha outside S held fixed. Removes the round-off accumulated along
        the path; skipped if the system is inconsistent (singular M) or the
        solution leaves the box by more than round-off.
        """
        if not self.S_set_:
            return
        s_idx = np.asarray(self.S_set_, dtype=int)
        y = self._y_arr()
        ay_out = self.alpha_ * y
        ay_out[s_idx] = 0.0
        rhs = np.concatenate([[-np.sum(ay_out)], 1.0 - y[s_idx] * (self.K_full_[s_idx] @ ay_out)])
        x = self.R_ @ rhs
        M = self._margin_matrix()
        if np.linalg.norm(M @ x - rhs) > 1e-9 * (1.0 + np.linalg.norm(rhs)):
            return
        a_s = x[1:]
        band = 1e-12 * self.C
        if np.any(a_s < -band) or np.any(a_s > self.C + band):
            return
        self.alpha_[s_idx] = a_s
        self.b_ = float(x[0])

    def _refresh_margin_set(self) -> None:
        """Recompute R for the current S and re-solve (b, alpha_S) exactly."""
        self._compute_R_matrix()
        self._project_margin_set()

    def _sensitivities(self, j: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Sensitivities for driving multiplier j (requires a non-empty S and R_).

        Returns beta_0 = d b / d alpha_j, beta_S = d alpha_S / d alpha_j and
        gamma (length n) with gamma_i = d g_i / d alpha_j = Q_ij + y_i beta_0
        + sum_{s in S} Q_is beta_s (meaningful for i not in S).
        """
        y = self._y_arr()
        s_idx = np.asarray(self.S_set_, dtype=int)
        y_s = y[s_idx]
        y_j = y[j]
        v = np.concatenate([[y_j], y_s * y_j * self.K_full_[s_idx, j]])
        beta = -self.R_ @ v
        beta_0 = float(beta[0])
        beta_s = beta[1:]
        gamma = y * y_j * self.K_full_[:, j] + y * beta_0 + y * (self.K_full_[:, s_idx] @ (y_s * beta_s))
        return beta_0, beta_s, gamma

    def _set_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.y_buffer_)
        in_E = np.zeros(n, dtype=bool)
        in_O = np.zeros(n, dtype=bool)
        in_E[self.E_set_] = True
        in_O[self.O_set_] = True
        return in_E, in_O

    def _assign_partitions(self) -> None:
        """
        Assign S/E/O from a batch solution and make it exactly consistent.

        Multipliers within eps_a = 1e-8 C of a bound are snapped onto it (O to
        exactly 0, E to exactly C); the residual r = sum(alpha y) is then moved
        onto S (alpha_S -= y_S r / |S|). If that is impossible (S empty or a
        bound would break), the solver's values are kept and classified by
        their exact bounds. b is the mean of y_s - (K (alpha y))_s over S (the
        KKT interval midpoint if S is empty), then (b, alpha_S) are re-solved
        from the margin system.
        """
        y = self._y_arr()
        C = self.C
        a = np.asarray(self.alpha_, dtype=float).copy()
        eps_a = 1e-8 * C
        at_0 = a <= eps_a
        at_C = a >= C - eps_a
        free = ~(at_0 | at_C)
        a_new = a.copy()
        a_new[at_0] = 0.0
        a_new[at_C] = C
        r = float(a_new @ y)
        ok = True
        if r != 0.0:
            s_idx = np.flatnonzero(free)
            if s_idx.size == 0:
                ok = False
            else:
                a_new[s_idx] -= y[s_idx] * r / s_idx.size
                ok = bool(np.all(a_new[s_idx] > 0.0) and np.all(a_new[s_idx] < C))
        if not ok:
            a_new = a
            at_0 = a <= 0.0
            at_C = a >= C
            free = ~(at_0 | at_C)

        self.alpha_ = a_new
        self.S_set_ = [int(i) for i in np.flatnonzero(free)]
        self.E_set_ = [int(i) for i in np.flatnonzero(at_C)]
        self.O_set_ = [int(i) for i in np.flatnonzero(at_0)]
        if self.S_set_:
            s_idx = np.asarray(self.S_set_, dtype=int)
            self.b_ = float(np.mean(y[s_idx] - self.K_full_[s_idx] @ (self.alpha_ * y)))
        else:
            self.b_ = compute_intercept(self.alpha_, self.K_full_, y, C, eps=1e-8 * C, fallback_b=self.b_)
        self._refresh_margin_set()

    def _set_single_class_state(self) -> None:
        """Single-class buffer: alpha = 0, b = +/-1 (that class), every point in O (g_i = 0)."""
        n = len(self.y_buffer_)
        self.alpha_ = np.zeros(n)
        self.b_ = float(self.y_buffer_[0]) if n else 0.0
        self.S_set_ = []
        self.E_set_ = []
        self.O_set_ = list(range(n))
        self._compute_R_matrix()

    def _transfer_alpha(self, p: int) -> None:
        """
        Drive alpha_p to exactly 0 along feasible directions (Delta a_p = -delta,
        Delta a_q = y_p y_q delta), keeping sum(alpha y) = 0. Partner q: largest
        g^L_q = y_q - (K (alpha y))_q if y_p = +1, smallest if y_p = -1 (same rule
        as LASVM.forget_one). Used only before an SMO fallback.
        """
        y = self._y_arr()
        y_p = y[p]
        while self.alpha_[p] > 0.0:
            g_l = y - self.K_full_ @ (self.alpha_ * y)
            cap = np.where(y != y_p, self.alpha_, self.C - self.alpha_)
            cand = cap > 0.0
            cand[p] = False
            if not np.any(cand):
                break
            idx = np.flatnonzero(cand)
            q = int(idx[np.argmax(g_l[idx])] if y_p > 0 else idx[np.argmin(g_l[idx])])
            delta = min(self.alpha_[p], cap[q])
            if delta == cap[q]:
                self.alpha_[q] = 0.0 if y[q] != y_p else self.C
            else:
                self.alpha_[q] += y_p * y[q] * delta
            self.alpha_[p] = 0.0 if delta == self.alpha_[p] else self.alpha_[p] - delta
        self.alpha_[p] = 0.0

    def _assert_box(self) -> None:
        band = 1e-10 * max(1.0, self.C)
        assert self.alpha_.size == 0 or (
            np.min(self.alpha_) >= -band and np.max(self.alpha_) <= self.C + band
        ), "AdiabaticSVM: alpha left the box [0, C]"

    def _compute_margin_values(self) -> np.ndarray:
        """Compute g_i = y_i f(x_i) - 1 for all active samples."""
        n = len(self.X_buffer_)
        if n == 0:
            return np.array([])
        y_arr = np.array(self.y_buffer_)
        f_vals = np.dot(self.K_full_, self.alpha_ * y_arr) + self.b_
        g = y_arr * f_vals - 1.0
        return g

    def _sync_support_vectors(self) -> None:
        """Synchronize public attributes with current support vector sets."""
        if not self.X_buffer_:
            self.support_vectors_ = None
            self.dual_coef_ = None
            self.support_ = None
            self.intercept_ = float(self.b_)
            self.n_support_ = np.array([0, 0], dtype=int)
            return

        sv_indices = sorted(self.S_set_ + self.E_set_)
        self.support_ = np.array(sv_indices, dtype=int)
        X_mat = np.array(self.X_buffer_)
        y_arr = np.array(self.y_buffer_)

        if len(sv_indices) > 0:
            self.support_vectors_ = X_mat[sv_indices]
            self.dual_coef_ = (self.alpha_[sv_indices] * y_arr[sv_indices]).reshape(1, -1)
            sv_y = y_arr[sv_indices]
            n_neg = int(np.sum(sv_y == -1.0))
            n_pos = int(np.sum(sv_y == 1.0))
            self.n_support_ = np.array([n_neg, n_pos], dtype=int)
        else:
            self.support_vectors_ = np.empty((0, X_mat.shape[1]))
            self.dual_coef_ = np.empty((1, 0))
            self.n_support_ = np.array([0, 0], dtype=int)

        self.intercept_ = float(self.b_)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdiabaticSVM":
        """
        Fit model from an initial batch dataset X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Initial training data.
        y : array-like of shape (n_samples,)
            Target binary class labels.

        Returns
        -------
        self : AdiabaticSVM
        """
        X_arr, y_arr = check_X_y(X, y)
        self.classes_ = np.unique(y_arr)
        if len(self.classes_) != 2:
            raise ValueError(f"AdiabaticSVM requires exactly 2 classes, found {len(self.classes_)}")

        y_encoded = np.where(y_arr == self.classes_[1], 1.0, -1.0)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        self.kernel_ = self._init_kernel(n_features, X_arr)
        self.K_full_ = self.kernel_(X_arr, X_arr)

        # Batch solve to find starting equilibrium
        solver = SMOSolver(C=self.C, tol=min(self.tol, 1e-6), max_iter=10000)
        alpha, b, _ = solver.solve(self.K_full_, y_encoded)

        self.X_buffer_ = [X_arr[i].copy() for i in range(n_samples)]
        self.y_buffer_ = [float(y_encoded[i]) for i in range(n_samples)]
        self.alpha_ = alpha.copy()
        self.b_ = float(b)
        self.n_samples_seen_ = n_samples
        self.n_unlearned_ = 0
        self.n_smo_fallbacks_ = 0

        # Classify into S, E, O partitions (snapped, sum(alpha y) restored)
        self._assign_partitions()
        self._sync_support_vectors()
        self._enforce_max_samples()
        return self

    def _recompute_equilibrium_smo(self) -> None:
        """
        Last-resort fallback: feasible SMO warm start from the current alpha
        (sum(alpha y) = 0 must hold), then partition assignment. Counted in
        ``n_smo_fallbacks_``.
        """
        n_samples = len(self.X_buffer_)
        if n_samples == 0:
            return
        y_arr = self._y_arr()
        if len(np.unique(y_arr)) < 2:
            self._set_single_class_state()
            return

        self.n_smo_fallbacks_ += 1
        solver = SMOSolver(C=self.C, tol=min(self.tol, 1e-6), max_iter=10000)
        alpha, b, _ = solver.solve(
            self.K_full_,
            y_arr,
            initial_alpha=self.alpha_,
            initial_b=self.b_,
        )
        self.alpha_ = alpha
        self.b_ = float(b)
        self._assign_partitions()

    def learn_one(self, x: np.ndarray, y: Union[int, float, str]) -> "AdiabaticSVM":
        """
        Incrementally add and learn a single incoming observation (x, y)
        using exact Cauwenberghs & Poggio adiabatic migration.

        Parameters
        ----------
        x : array-like of shape (n_features,)
        y : class label

        Returns
        -------
        self : AdiabaticSVM
        """
        x_vec = np.asarray(x, dtype=float).ravel()
        if not self.X_buffer_:
            # First sample encountered
            if getattr(self, "classes_", None) is None:
                self.classes_ = np.array([-1.0, 1.0])
            y_enc = 1.0 if y == self.classes_[1] else -1.0
            self.X_buffer_ = [x_vec.copy()]
            self.y_buffer_ = [y_enc]
            self.alpha_ = np.array([0.0])
            self.b_ = y_enc  # single-class state: g_0 = y_0 b - 1 = 0, KKT holds with alpha = 0
            self.n_features_in_ = len(x_vec)
            self.kernel_ = self._init_kernel(len(x_vec), x_vec.reshape(1, -1))
            self.K_full_ = np.array([[float(self.kernel_(x_vec.reshape(1, -1), x_vec.reshape(1, -1))[0, 0])]])
            self.O_set_ = [0]
            self.S_set_ = []
            self.E_set_ = []
            self.n_samples_seen_ = 1
            self._sync_support_vectors()
            return self

        y_enc = 1.0 if y == self.classes_[1] else -1.0
        c_idx = len(self.X_buffer_)
        self.X_buffer_.append(x_vec.copy())
        self.y_buffer_.append(y_enc)
        self.n_samples_seen_ += 1

        # Expand kernel matrix
        X_mat = np.array(self.X_buffer_[:-1])
        k_new_col = self.kernel_(X_mat, x_vec.reshape(1, -1)).ravel()
        k_self = float(self.kernel_(x_vec.reshape(1, -1), x_vec.reshape(1, -1))[0, 0])

        n_prev = len(self.X_buffer_) - 1
        new_K = np.zeros((n_prev + 1, n_prev + 1))
        new_K[:n_prev, :n_prev] = self.K_full_
        new_K[:n_prev, n_prev] = k_new_col
        new_K[n_prev, :n_prev] = k_new_col
        new_K[n_prev, n_prev] = k_self
        self.K_full_ = new_K

        self.alpha_ = np.append(self.alpha_, 0.0)

        # Margin function evaluation g_c = y_c f(x_c) - 1
        g_c = float(self._compute_margin_values()[c_idx])

        # KKT check: g_c >= 0 means alpha_c = 0 is already optimal (c belongs to O)
        if g_c >= -1e-12:
            self.O_set_.append(c_idx)
            self._sync_support_vectors()
            self._enforce_max_samples()
            return self

        if not self._incremental_path(c_idx):
            # Path exhausted: place c by its alpha and repair from the (feasible) current point
            a_c = self.alpha_[c_idx]
            if a_c == 0.0:
                self.O_set_.append(c_idx)
            elif a_c == self.C:
                self.E_set_.append(c_idx)
            else:
                self.S_set_.append(c_idx)
            self._recompute_equilibrium_smo()

        self._assert_box()
        self._sync_support_vectors()
        self._enforce_max_samples()
        return self

    def _apply_transition(self, event_type: str, event_idx: int) -> None:
        """Move event_idx between S/E/O (snapping its alpha exactly) and refresh R."""
        if event_type == "S_to_O":
            self.alpha_[event_idx] = 0.0
            self.S_set_.remove(event_idx)
            self.O_set_.append(event_idx)
        elif event_type == "S_to_E":
            self.alpha_[event_idx] = self.C
            self.S_set_.remove(event_idx)
            self.E_set_.append(event_idx)
        elif event_type == "E_to_S":
            self.E_set_.remove(event_idx)
            self.S_set_.append(event_idx)
        elif event_type == "O_to_S":
            self.O_set_.remove(event_idx)
            self.S_set_.append(event_idx)
        self._refresh_margin_set()

    def _incremental_path(self, c: int) -> bool:
        """
        Cauwenberghs-Poggio insertion of c (not in any set, alpha_c = 0, g_c < 0).

        Returns True when c has been placed in S, E or O with every KKT
        condition satisfied; False if ``max_path_steps`` was exhausted.
        Zero-length steps are applied as immediate set transitions.
        """
        C = self.C
        y = self._y_arr()
        y_c = y[c]
        for _ in range(self.max_path_steps):
            g = self._compute_margin_values()
            in_E, in_O = self._set_masks()

            if not self.S_set_:
                # Bias-only step: Delta b = y_c t, Delta g_i = y_i y_c t (alpha fixed)
                cands: List[Tuple[float, str, int]] = [(max(0.0, -g[c]), "c_margin", c)]
                for i in np.flatnonzero(in_O & (y == -y_c)):
                    cands.append((max(0.0, g[i]), "O_to_S", int(i)))
                for i in np.flatnonzero(in_E & (y == y_c)):
                    cands.append((max(0.0, -g[i]), "E_to_S", int(i)))
                t, event_type, event_idx = min(cands, key=lambda e: e[0])
                self.b_ += y_c * t
                if event_type == "c_margin":
                    if self.alpha_[c] == 0.0:
                        self.O_set_.append(c)
                    else:
                        self.S_set_.append(c)
                        self._refresh_margin_set()
                    return True
                self._apply_transition(event_type, event_idx)
                continue

            beta_0, beta_s, gamma = self._sensitivities(c)
            cands = [(C - self.alpha_[c], "c_to_E", c)]
            if gamma[c] > _SENS_EPS:
                cands.append((max(0.0, -g[c] / gamma[c]), "c_to_S", c))
            for pos, s in enumerate(self.S_set_):
                b_s = beta_s[pos]
                if b_s > _SENS_EPS:
                    cands.append((max(0.0, (C - self.alpha_[s]) / b_s), "S_to_E", s))
                elif b_s < -_SENS_EPS:
                    cands.append((max(0.0, self.alpha_[s] / -b_s), "S_to_O", s))
            for i in np.flatnonzero(in_E & (gamma > _SENS_EPS)):
                cands.append((max(0.0, -g[i] / gamma[i]), "E_to_S", int(i)))
            for i in np.flatnonzero(in_O & (gamma < -_SENS_EPS)):
                cands.append((max(0.0, -g[i] / gamma[i]), "O_to_S", int(i)))

            t, event_type, event_idx = min(cands, key=lambda e: e[0])
            s_idx = np.asarray(self.S_set_, dtype=int)
            self.alpha_[c] += t
            self.alpha_[s_idx] += beta_s * t
            self.b_ += beta_0 * t

            if event_type == "c_to_E":
                self.alpha_[c] = C
                self.E_set_.append(c)
                self._project_margin_set()
                return True
            if event_type == "c_to_S":
                self.S_set_.append(c)
                self._refresh_margin_set()
                return True
            self._apply_transition(event_type, event_idx)
        return False

    def forget_one(self, index: int = 0) -> "AdiabaticSVM":
        """
        Decrementally remove/unlearn an instance from the model's active memory
        using exact Cauwenberghs & Poggio adiabatic parameter unlearning.

        Parameters
        ----------
        index : int
            Index of the sample in active memory buffer to unlearn.

        Returns
        -------
        self : AdiabaticSVM
        """
        n_samples = len(self.X_buffer_)
        if index < 0 or index >= n_samples:
            raise IndexError(f"Index {index} out of bounds for buffer of size {n_samples}")

        p = int(index)
        # p leaves its set; E points are never moved into S artificially
        for part in (self.S_set_, self.E_set_, self.O_set_):
            if p in part:
                part.remove(p)

        needs_repair = False
        if self.alpha_[p] > 0.0:
            if not self._decremental_path(p):
                # Path exhausted: make alpha feasible with alpha_p = 0, repair after deletion
                self._transfer_alpha(p)
                needs_repair = True

        # alpha_p == 0 exactly: deleting p changes neither sum(alpha y) nor any other g_i
        del self.X_buffer_[p]
        del self.y_buffer_[p]
        self.alpha_ = np.delete(self.alpha_, p)
        self.K_full_ = np.delete(np.delete(self.K_full_, p, axis=0), p, axis=1)
        self.n_unlearned_ += 1

        if len(self.X_buffer_) == 0:
            self.S_set_ = []
            self.E_set_ = []
            self.O_set_ = []
            self._compute_R_matrix()
            self._sync_support_vectors()
            return self

        # Remap partition indices after deletion
        self.S_set_ = [i if i < p else i - 1 for i in self.S_set_]
        self.E_set_ = [i if i < p else i - 1 for i in self.E_set_]
        self.O_set_ = [i if i < p else i - 1 for i in self.O_set_]

        if len(np.unique(self.y_buffer_)) < 2:
            self._set_single_class_state()
        elif needs_repair:
            self._recompute_equilibrium_smo()
        else:
            self._refresh_margin_set()

        self._assert_box()
        self._sync_support_vectors()
        return self

    def _decremental_path(self, p: int) -> bool:
        """
        Cauwenberghs-Poggio removal: drive alpha_p (p already removed from its
        set) to exactly 0 while keeping every other KKT condition.

        Returns True when alpha_p == 0; False if ``max_path_steps`` was exhausted
        or no event exists (degenerate input).
        """
        C = self.C
        y = self._y_arr()
        y_p = y[p]
        self._refresh_margin_set()  # R' for S' = S \ {p}
        for _ in range(self.max_path_steps):
            if self.alpha_[p] == 0.0:
                return True
            g = self._compute_margin_values()
            in_E, in_O = self._set_masks()

            if not self.S_set_:
                # Bias-only step: Delta b = -y_p t, Delta g_i = -y_i y_p t (alpha fixed).
                # Joiners are exactly the points that can absorb alpha_p afterwards.
                cands: List[Tuple[float, str, int]] = []
                for i in np.flatnonzero(in_O & (y == y_p)):
                    cands.append((max(0.0, g[i]), "O_to_S", int(i)))
                for i in np.flatnonzero(in_E & (y == -y_p)):
                    cands.append((max(0.0, -g[i]), "E_to_S", int(i)))
                if not cands:
                    return False
                t, event_type, event_idx = min(cands, key=lambda e: e[0])
                self.b_ -= y_p * t
                self._apply_transition(event_type, event_idx)
                continue

            beta_0, beta_s, gamma = self._sensitivities(p)
            # Delta alpha_p = -delta: Delta alpha_S = -beta_S delta, Delta b = -beta_0 delta,
            # Delta g_i = -gamma_i delta
            cands = [(self.alpha_[p], "p_zero", p)]
            for pos, s in enumerate(self.S_set_):
                b_s = beta_s[pos]
                if b_s > _SENS_EPS:
                    cands.append((max(0.0, self.alpha_[s] / b_s), "S_to_O", s))
                elif b_s < -_SENS_EPS:
                    cands.append((max(0.0, (C - self.alpha_[s]) / -b_s), "S_to_E", s))
            for i in np.flatnonzero(in_E & (gamma < -_SENS_EPS)):
                cands.append((max(0.0, g[i] / gamma[i]), "E_to_S", int(i)))
            for i in np.flatnonzero(in_O & (gamma > _SENS_EPS)):
                cands.append((max(0.0, g[i] / gamma[i]), "O_to_S", int(i)))

            delta, event_type, event_idx = min(cands, key=lambda e: e[0])
            s_idx = np.asarray(self.S_set_, dtype=int)
            self.alpha_[p] -= delta
            self.alpha_[s_idx] -= beta_s * delta
            self.b_ -= beta_0 * delta

            if event_type == "p_zero":
                self.alpha_[p] = 0.0
                self._project_margin_set()
                return True
            self._apply_transition(event_type, event_idx)
        return self.alpha_[p] == 0.0

    def unlearn_sample(self, x: np.ndarray, y: Optional[Union[int, float, str]] = None) -> "AdiabaticSVM":
        """
        Unlearn the instance closest in Euclidean distance to vector x.
        """
        if not self.X_buffer_:
            return self
        x_vec = np.asarray(x, dtype=float).ravel()
        X_mat = np.array(self.X_buffer_)
        dists = np.linalg.norm(X_mat - x_vec, axis=1)
        if y is not None:
            y_enc = 1.0 if y == self.classes_[1] else -1.0
            for idx in np.argsort(dists):
                if self.y_buffer_[idx] == y_enc:
                    return self.forget_one(idx)
        closest_idx = int(np.argmin(dists))
        return self.forget_one(closest_idx)

    def _enforce_max_samples(self) -> None:
        """Evict samples if buffer exceeds max_samples."""
        if self.max_samples is None:
            return
        while len(self.X_buffer_) > self.max_samples:
            idx_to_remove = self._select_sample_to_prune()
            self.forget_one(idx_to_remove)

    def _select_sample_to_prune(self) -> int:
        """Select index to remove based on configured pruning_strategy."""
        if self.pruning_strategy == "non_sv_first" and self.O_set_:
            return self.O_set_[0]
        elif self.pruning_strategy == "error_first" and self.E_set_:
            return self.E_set_[0]
        # Default FIFO
        return 0

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the decision function f(X) = sum_i alpha_i y_i K(x_i, X) + b.
        """
        check_is_fitted(self)
        X_arr = check_array(X, n_features=getattr(self, "n_features_in_", None))
        if not self.X_buffer_ or self.support_vectors_ is None or len(self.support_vectors_) == 0:
            return np.full(X_arr.shape[0], self.b_)

        K_test = self.kernel_(X_arr, self.support_vectors_)
        decision = np.dot(K_test, self.dual_coef_.ravel()) + self.b_
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

    def get_model_state(self) -> Dict[str, Union[int, float, List[int]]]:
        """Return diagnostic metrics of the current model state."""
        return {
            "n_samples_buffered": len(self.X_buffer_),
            "n_margin_sv": len(self.S_set_),
            "n_error_sv": len(self.E_set_),
            "n_non_sv": len(self.O_set_),
            "b": float(self.b_),
            "n_samples_seen": self.n_samples_seen_,
            "n_unlearned": self.n_unlearned_,
            "n_smo_fallbacks": self.n_smo_fallbacks_,
        }


# Alias for academic and practical naming conventions
IncrementalDecrementalSVC = AdiabaticSVM
ExactIncrementalDecrementalSVM = AdiabaticSVM

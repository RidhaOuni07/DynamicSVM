"""
Adiabatic Path Following Incremental and Decremental Support Vector Machine.

Implements the exact Kuhn-Tucker relations and continuous adiabatic parameter migration
derived by Cauwenberghs & Poggio (NeurIPS 2000 / 2001) and Laskov et al. (JMLR 2006).
"""

from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np

from dynamicsvm.base import BaseSVM
from dynamicsvm.kernels import Kernel, get_kernel
from dynamicsvm.solvers.smo import SMOSolver
from dynamicsvm.stream.base import StreamingSVM
from dynamicsvm.utils.validation import check_array, check_X_y, check_is_fitted


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
        Tolerance for optimization convergence and KKT margin checks.
    max_path_steps : int, default=500
        Maximum adiabatic migration steps per sample insertion/deletion.
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

    def _init_kernel(self, n_features: int, X_sample: Optional[np.ndarray] = None) -> Kernel:
        return get_kernel(
            self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            n_features=n_features,
            X=X_sample,
        )

    def _compute_R_matrix(self) -> None:
        """Compute the sensitivity matrix R = M^{-1} for current margin set S."""
        n_s = len(self.S_set_)
        if n_s == 0:
            self.R_ = np.array([[0.0]])
            return

        M = np.zeros((n_s + 1, n_s + 1))
        y_s = np.array([self.y_buffer_[i] for i in self.S_set_])
        M[0, 1:] = y_s
        M[1:, 0] = y_s

        s_idx = np.array(self.S_set_)
        # Q_ss = y_i y_j K(x_i, x_j)
        K_ss = self.K_full_[np.ix_(s_idx, s_idx)]
        Q_ss = np.outer(y_s, y_s) * K_ss
        M[1:, 1:] = Q_ss + 1e-9 * np.eye(n_s)

        try:
            self.R_ = np.linalg.pinv(M)
        except Exception:
            self.R_ = np.linalg.lstsq(M, np.eye(n_s + 1), rcond=None)[0]

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

        self.kernel_ = self._init_kernel(n_features, X_arr)
        self.K_full_ = self.kernel_(X_arr, X_arr)

        # Batch solve to find starting equilibrium
        solver = SMOSolver(C=self.C, tol=self.tol, max_iter=2000)
        alpha, b, _ = solver.solve(self.K_full_, y_encoded)

        self.X_buffer_ = [X_arr[i].copy() for i in range(n_samples)]
        self.y_buffer_ = [float(y_encoded[i]) for i in range(n_samples)]
        self.alpha_ = alpha.copy()
        self.b_ = float(b)
        self.n_samples_seen_ = n_samples
        self.n_unlearned_ = 0

        # Classify into S, E, O partitions
        self.S_set_ = []
        self.E_set_ = []
        self.O_set_ = []

        for i in range(n_samples):
            a_i = self.alpha_[i]
            if a_i > self.tol and a_i < (self.C - self.tol):
                self.S_set_.append(i)
            elif a_i >= (self.C - self.tol):
                self.E_set_.append(i)
            else:
                self.O_set_.append(i)

        self._compute_R_matrix()
        self._sync_support_vectors()
        self._enforce_max_samples()
        return self

    def _recompute_equilibrium_smo(self) -> None:
        """Re-synchronize exact equilibrium if edge cases destabilize matrix R."""
        n_samples = len(self.X_buffer_)
        if n_samples < 2:
            return
        y_arr = np.array(self.y_buffer_)
        unique_classes = np.unique(y_arr)
        if len(unique_classes) < 2:
            return

        solver = SMOSolver(C=self.C, tol=self.tol, max_iter=500)
        alpha, b, _ = solver.solve(
            self.K_full_,
            y_arr,
            initial_alpha=self.alpha_,
            initial_b=self.b_,
        )
        self.alpha_ = alpha
        self.b_ = float(b)

        self.S_set_ = []
        self.E_set_ = []
        self.O_set_ = []
        for i in range(n_samples):
            a_i = self.alpha_[i]
            if a_i > self.tol and a_i < (self.C - self.tol):
                self.S_set_.append(i)
            elif a_i >= (self.C - self.tol):
                self.E_set_.append(i)
            else:
                self.O_set_.append(i)

        self._compute_R_matrix()

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
            self.classes_ = np.array([-1.0, 1.0]) if not hasattr(self, "classes_") else self.classes_
            y_enc = 1.0 if y == self.classes_[1] else -1.0
            self.X_buffer_ = [x_vec.copy()]
            self.y_buffer_ = [y_enc]
            self.alpha_ = np.array([0.0])
            self.b_ = 0.0
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
        y_arr = np.array(self.y_buffer_)
        f_c = float(np.dot(self.K_full_[c_idx, :c_idx], self.alpha_[:c_idx] * y_arr[:c_idx]) + self.b_)
        g_c = y_enc * f_c - 1.0

        # KKT Check: If g_c >= 0, point satisfies KKT with alpha_c = 0 (belongs to O)
        if g_c >= -self.tol:
            self.O_set_.append(c_idx)
            self._sync_support_vectors()
            self._enforce_max_samples()
            return self

        # Check if both classes exist in active buffer
        unique_classes = np.unique(self.y_buffer_)
        if len(unique_classes) < 2 or not self.S_set_:
            # Fall back to quick SMO equilibrium to initialize dual variables cleanly
            self._recompute_equilibrium_smo()
            self._sync_support_vectors()
            self._enforce_max_samples()
            return self

        # Adiabatic path following: increase alpha_c towards C while preserving sum(y_i alpha_i) = 0 and g_S = 0
        path_step = 0
        while path_step < self.max_path_steps:
            path_step += 1
            if not self.S_set_:
                self._recompute_equilibrium_smo()
                break

            s_idx = np.array(self.S_set_)
            y_s = y_arr[s_idx]
            q_s_c = y_s * y_enc * self.K_full_[s_idx, c_idx]
            v = np.concatenate([[y_enc], q_s_c])
            beta = -np.dot(self.R_, v)
            beta_0 = beta[0]  # Rate of change of b: Delta b / Delta alpha_c
            beta_s = beta[1:] # Rate of change of alpha_S: Delta alpha_s / Delta alpha_c

            # Rate of change of g_c: gamma_c
            gamma_c = float(self.K_full_[c_idx, c_idx] + y_enc * beta_0 + np.dot(q_s_c, beta_s))

            # Rate of change of g_i for all remaining samples
            gamma = np.zeros(len(self.X_buffer_))
            for i in range(len(self.X_buffer_)):
                if i not in self.S_set_:
                    q_i_s = y_arr[i] * y_s * self.K_full_[i, s_idx]
                    q_i_c = y_arr[i] * y_enc * self.K_full_[i, c_idx]
                    gamma[i] = q_i_c + y_arr[i] * beta_0 + np.dot(q_i_s, beta_s)

            # Evaluate maximum possible step Delta alpha_c before a boundary transition occurs
            step_candidates: List[Tuple[float, str, int]] = []

            # 1. c reaches upper bound C
            step_c_bound = self.C - self.alpha_[c_idx]
            if step_c_bound > 1e-12:
                step_candidates.append((step_c_bound, "c_to_E", c_idx))

            # 2. c reaches margin g_c = 0
            if gamma_c > 1e-9:
                curr_g_c = y_enc * (np.dot(self.K_full_[c_idx], self.alpha_ * y_arr) + self.b_) - 1.0
                if curr_g_c < -1e-9:
                    step_c_margin = -curr_g_c / gamma_c
                    if step_c_margin > 1e-12:
                        step_candidates.append((step_c_margin, "c_to_S", c_idx))

            # 3. An SV in S hits 0 or C
            for idx_pos, s_pos in enumerate(self.S_set_):
                b_val = beta_s[idx_pos]
                if b_val < -1e-9:
                    step_s_zero = -self.alpha_[s_pos] / b_val
                    if step_s_zero > 1e-12:
                        step_candidates.append((step_s_zero, "S_to_O", s_pos))
                elif b_val > 1e-9:
                    step_s_c = (self.C - self.alpha_[s_pos]) / b_val
                    if step_s_c > 1e-12:
                        step_candidates.append((step_s_c, "S_to_E", s_pos))

            # 4. A point in E hits margin g_i = 0
            curr_g = self._compute_margin_values()
            for e_pos in self.E_set_:
                if gamma[e_pos] > 1e-9 and curr_g[e_pos] < -1e-9:
                    step_e_s = -curr_g[e_pos] / gamma[e_pos]
                    if step_e_s > 1e-12:
                        step_candidates.append((step_e_s, "E_to_S", e_pos))

            # 5. A point in O hits margin g_i = 0
            for o_pos in self.O_set_:
                if o_pos != c_idx and gamma[o_pos] < -1e-9 and curr_g[o_pos] > 1e-9:
                    step_o_s = -curr_g[o_pos] / gamma[o_pos]
                    if step_o_s > 1e-12:
                        step_candidates.append((step_o_s, "O_to_S", o_pos))

            if not step_candidates:
                self._recompute_equilibrium_smo()
                break

            # Find minimum allowable step
            step_candidates.sort(key=lambda t: t[0])
            min_step, event_type, event_idx = step_candidates[0]

            if min_step <= 1e-12:
                # Numerical micro-stall; stabilize via SMO
                self._recompute_equilibrium_smo()
                break

            # Apply adiabatic displacement
            self.alpha_[c_idx] += min_step
            for idx_pos, s_pos in enumerate(self.S_set_):
                self.alpha_[s_pos] += beta_s[idx_pos] * min_step
            self.b_ += beta_0 * min_step

            # Execute partition transition
            if event_type == "c_to_E":
                self.alpha_[c_idx] = self.C
                self.E_set_.append(c_idx)
                break
            elif event_type == "c_to_S":
                self.S_set_.append(c_idx)
                self._compute_R_matrix()
                break
            elif event_type == "S_to_O":
                self.alpha_[event_idx] = 0.0
                self.S_set_.remove(event_idx)
                self.O_set_.append(event_idx)
                self._compute_R_matrix()
            elif event_type == "S_to_E":
                self.alpha_[event_idx] = self.C
                self.S_set_.remove(event_idx)
                self.E_set_.append(event_idx)
                self._compute_R_matrix()
            elif event_type == "E_to_S":
                self.E_set_.remove(event_idx)
                self.S_set_.append(event_idx)
                self._compute_R_matrix()
            elif event_type == "O_to_S":
                self.O_set_.remove(event_idx)
                self.S_set_.append(event_idx)
                self._compute_R_matrix()

        # Enforce bounds and clip numerical drift
        self.alpha_ = np.clip(self.alpha_, 0.0, self.C)
        self._sync_support_vectors()
        self._enforce_max_samples()
        return self

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

        # If sample is a non-support vector (alpha_p = 0), removal leaves KKT equilibrium intact
        is_non_sv = (index in self.O_set_) or (self.alpha_[index] <= self.tol)

        if not is_non_sv and n_samples > 2:
            # Adiabatically drive alpha_p down to 0
            y_arr = np.array(self.y_buffer_)
            p_idx = index

            # If p is in E, first migrate it into S if possible
            if p_idx in self.E_set_:
                self.E_set_.remove(p_idx)
                self.S_set_.append(p_idx)
                self._compute_R_matrix()

            path_step = 0
            while self.alpha_[p_idx] > self.tol and path_step < self.max_path_steps:
                path_step += 1
                if not self.S_set_:
                    break

                s_idx = np.array(self.S_set_)
                y_s = y_arr[s_idx]
                y_p = y_arr[p_idx]
                q_s_p = y_s * y_p * self.K_full_[s_idx, p_idx]
                v = np.concatenate([[y_p], q_s_p])
                beta = -np.dot(self.R_, v)
                beta_0 = beta[0]
                beta_s = beta[1:]

                # Decrease alpha_p: step Delta alpha_p < 0
                max_step_down = self.alpha_[p_idx]
                step_candidates: List[Tuple[float, str, int]] = [(max_step_down, "p_zero", p_idx)]

                for idx_pos, s_pos in enumerate(self.S_set_):
                    if s_pos != p_idx:
                        b_val = beta_s[idx_pos]
                        # We are subtracting (Delta alpha_p is negative)
                        # alpha_s_new = alpha_s + beta_s * (-delta)
                        if b_val > 1e-9:
                            step_zero = self.alpha_[s_pos] / b_val
                            if step_zero > 1e-12:
                                step_candidates.append((step_zero, "S_to_O", s_pos))
                        elif b_val < -1e-9:
                            step_c = (self.C - self.alpha_[s_pos]) / (-b_val)
                            if step_c > 1e-12:
                                step_candidates.append((step_c, "S_to_E", s_pos))

                step_candidates.sort(key=lambda t: t[0])
                delta_p, event_type, event_idx = step_candidates[0]

                if delta_p <= 1e-12:
                    break

                self.alpha_[p_idx] = max(0.0, self.alpha_[p_idx] - delta_p)
                for idx_pos, s_pos in enumerate(self.S_set_):
                    if s_pos != p_idx:
                        self.alpha_[s_pos] -= beta_s[idx_pos] * delta_p
                self.b_ -= beta_0 * delta_p

                if event_type == "p_zero":
                    self.alpha_[p_idx] = 0.0
                    break
                elif event_type == "S_to_O":
                    self.alpha_[event_idx] = 0.0
                    self.S_set_.remove(event_idx)
                    self.O_set_.append(event_idx)
                    self._compute_R_matrix()
                elif event_type == "S_to_E":
                    self.alpha_[event_idx] = self.C
                    self.S_set_.remove(event_idx)
                    self.E_set_.append(event_idx)
                    self._compute_R_matrix()

        # Delete sample from buffer and metadata
        del self.X_buffer_[index]
        del self.y_buffer_[index]
        self.alpha_ = np.delete(self.alpha_, index)
        self.K_full_ = np.delete(np.delete(self.K_full_, index, axis=0), index, axis=1)
        self.n_unlearned_ += 1

        if len(self.X_buffer_) == 0:
            self.S_set_ = []
            self.E_set_ = []
            self.O_set_ = []
            self._sync_support_vectors()
            return self

        # Remap partition indices after deletion
        self.S_set_ = [i if i < index else i - 1 for i in self.S_set_ if i != index]
        self.E_set_ = [i if i < index else i - 1 for i in self.E_set_ if i != index]
        self.O_set_ = [i if i < index else i - 1 for i in self.O_set_ if i != index]

        # Verify equilibrium
        if not is_non_sv:
            self._recompute_equilibrium_smo()
        else:
            self._compute_R_matrix()

        self._sync_support_vectors()
        return self

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
        X_arr = check_array(X)
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
        """Compute class probabilities via logistic sigmoid."""
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
        }


# Alias for academic and practical naming conventions
IncrementalDecrementalSVC = AdiabaticSVM
ExactIncrementalDecrementalSVM = AdiabaticSVM

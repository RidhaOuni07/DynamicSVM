"""
Sequential Minimal Optimization (SMO) algorithm for Support Vector Machines.
Based on John Platt's 1998 SMO paper and subsequent enhancements.
"""

from typing import Optional, Tuple
import numpy as np


class SMOSolver:
    """
    SMO Solver for binary dual SVM quadratic optimization:
    
    min_alpha 0.5 * sum_ij alpha_i alpha_j y_i y_j K(x_i, x_j) - sum_i alpha_i
    s.t. 0 <= alpha_i <= C
         sum_i alpha_i y_i = 0
    """

    def __init__(
        self,
        C: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 1000,
        eps: float = 1e-5,
        random_state: Optional[int] = None,
    ):
        self.C = float(C)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.eps = float(eps)
        self.rng = np.random.RandomState(random_state)

    def solve(
        self,
        K: np.ndarray,
        y: np.ndarray,
        initial_alpha: Optional[np.ndarray] = None,
        initial_b: float = 0.0,
    ) -> Tuple[np.ndarray, float, int]:
        """
        Solve the dual SVM optimization problem.

        Parameters
        ----------
        K : np.ndarray of shape (n_samples, n_samples)
            The precomputed Gram matrix K(X, X).
        y : np.ndarray of shape (n_samples,)
            Binary class labels encoded as -1.0 and +1.0.
        initial_alpha : np.ndarray, optional
            Warm-start dual coefficients.
        initial_b : float, optional
            Warm-start intercept.

        Returns
        -------
        alpha : np.ndarray of shape (n_samples,)
            Lagrange multipliers.
        b : float
            Intercept term.
        n_iters : int
            Number of outer iterations performed.
        """
        n_samples = K.shape[0]
        y = np.asarray(y, dtype=float)

        if initial_alpha is not None:
            alpha = np.asarray(initial_alpha, dtype=float).copy()
        else:
            alpha = np.zeros(n_samples, dtype=float)

        # Use a list wrapper so b can be mutated in place across helper methods
        b_container = [float(initial_b)]

        # Compute initial error cache E_i = u(x_i) - y_i = sum_j alpha_j y_j K(j, i) + b - y_i
        u = np.dot(K, alpha * y) + b_container[0]
        errors = u - y

        num_changed = 0
        examine_all = True
        n_iters = 0

        while (num_changed > 0 or examine_all) and (n_iters < self.max_iter):
            num_changed = 0
            if examine_all:
                for i in range(n_samples):
                    num_changed += self._examine_example(i, alpha, b_container, errors, K, y, n_samples)
            else:
                # Loop over non-bound examples (0 < alpha < C)
                non_bound_indices = np.where((alpha > self.eps) & (alpha < self.C - self.eps))[0]
                for i in non_bound_indices:
                    num_changed += self._examine_example(i, alpha, b_container, errors, K, y, n_samples)

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            n_iters += 1

        # Final refined intercept calculation using margin support vectors
        b = self._compute_intercept(alpha, K, y, fallback_b=b_container[0])

        return alpha, b, n_iters

    def _examine_example(
        self,
        i2: int,
        alpha: np.ndarray,
        b_container: list,
        errors: np.ndarray,
        K: np.ndarray,
        y: np.ndarray,
        n_samples: int,
    ) -> int:
        y2 = y[i2]
        alph2 = alpha[i2]
        E2 = errors[i2]
        r2 = E2 * y2

        # Check KKT violation with tolerance
        if (r2 < -self.tol and alph2 < self.C - self.eps) or (r2 > self.tol and alph2 > self.eps):
            # First heuristic: choose i1 with max |E1 - E2| among non-bound examples
            non_bound = np.where((alpha > self.eps) & (alpha < self.C - self.eps))[0]
            if len(non_bound) > 1:
                if E2 >= 0:
                    i1 = int(non_bound[np.argmin(errors[non_bound])])
                else:
                    i1 = int(non_bound[np.argmax(errors[non_bound])])

                if self._take_step(i1, i2, alpha, b_container, errors, K, y):
                    return 1

            # Second heuristic: loop through all non-bound examples starting from random offset
            if len(non_bound) > 0:
                start_offset = self.rng.randint(0, len(non_bound))
                for k in range(len(non_bound)):
                    i1 = non_bound[(start_offset + k) % len(non_bound)]
                    if i1 != i2 and self._take_step(i1, i2, alpha, b_container, errors, K, y):
                        return 1

            # Third heuristic: loop through all examples starting from random offset
            start_offset = self.rng.randint(0, n_samples)
            for k in range(n_samples):
                i1 = (start_offset + k) % n_samples
                if i1 != i2 and self._take_step(i1, i2, alpha, b_container, errors, K, y):
                    return 1

        return 0

    def _take_step(
        self,
        i1: int,
        i2: int,
        alpha: np.ndarray,
        b_container: list,
        errors: np.ndarray,
        K: np.ndarray,
        y: np.ndarray,
    ) -> bool:
        if i1 == i2:
            return False

        b_val = b_container[0]
        alph1 = alpha[i1]
        alph2 = alpha[i2]
        y1 = y[i1]
        y2 = y[i2]
        E1 = errors[i1]
        E2 = errors[i2]
        s = y1 * y2

        # Compute bounds L and H for alpha2
        if y1 != y2:
            L = max(0.0, alph2 - alph1)
            H = min(self.C, self.C + alph2 - alph1)
        else:
            L = max(0.0, alph1 + alph2 - self.C)
            H = min(self.C, alph1 + alph2)

        if abs(L - H) < 1e-12:
            return False

        k11 = K[i1, i1]
        k12 = K[i1, i2]
        k22 = K[i2, i2]
        eta = k11 + k22 - 2.0 * k12

        if eta > 0:
            a2 = alph2 + y2 * (E1 - E2) / eta
            # Clip a2 into [L, H]
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            # Objective function evaluation at bounds L and H
            c1 = alph1 + s * alph2
            a1_L = c1 - s * L
            obj_L = self._eval_sub_objective(a1_L, L, i1, i2, K, y, alpha)
            a1_H = c1 - s * H
            obj_H = self._eval_sub_objective(a1_H, H, i1, i2, K, y, alpha)

            if obj_L < obj_H - self.eps:
                a2 = L
            elif obj_L > obj_H + self.eps:
                a2 = H
            else:
                a2 = alph2

        if abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps):
            return False

        a1 = alph1 + s * (alph2 - a2)

        # Update threshold b
        b1 = b_val - E1 - y1 * (a1 - alph1) * k11 - y2 * (a2 - alph2) * k12
        b2 = b_val - E2 - y1 * (a1 - alph1) * k12 - y2 * (a2 - alph2) * k22

        if 0 < a1 < self.C:
            b_new = b1
        elif 0 < a2 < self.C:
            b_new = b2
        else:
            b_new = (b1 + b2) / 2.0

        delta_a1 = a1 - alph1
        delta_a2 = a2 - alph2
        delta_b = b_new - b_val

        # Update error cache for all samples
        errors += y1 * delta_a1 * K[i1, :] + y2 * delta_a2 * K[i2, :] + delta_b

        alpha[i1] = a1
        alpha[i2] = a2
        b_container[0] = b_new
        return True

    def _eval_sub_objective(
        self,
        a1: float,
        a2: float,
        i1: int,
        i2: int,
        K: np.ndarray,
        y: np.ndarray,
        alpha: np.ndarray,
    ) -> float:
        """Evaluate dual objective restricted to changes in a1 and a2."""
        k11 = K[i1, i1]
        k22 = K[i2, i2]
        k12 = K[i1, i2]
        y1, y2 = y[i1], y[i2]

        other_idx = np.ones(len(alpha), dtype=bool)
        other_idx[i1] = False
        other_idx[i2] = False

        ay_other = alpha[other_idx] * y[other_idx]
        v1 = np.dot(K[i1, other_idx], ay_other)
        v2 = np.dot(K[i2, other_idx], ay_other)

        obj = (
            0.5 * (a1**2 * k11 + a2**2 * k22)
            + (a1 * a2 * y1 * y2 * k12)
            + a1 * y1 * v1
            + a2 * y2 * v2
            - (a1 + a2)
        )
        return float(obj)

    def _compute_intercept(
        self,
        alpha: np.ndarray,
        K: np.ndarray,
        y: np.ndarray,
        fallback_b: float = 0.0,
    ) -> float:
        """Compute exact intercept b using margin support vectors (0 < alpha_i < C)."""
        ay = alpha * y
        margin_sv = np.where((alpha > self.eps) & (alpha < self.C - self.eps))[0]
        if len(margin_sv) > 0:
            b_values = y[margin_sv] - np.dot(K[margin_sv, :], ay)
            return float(np.mean(b_values))

        sv = np.where(alpha > self.eps)[0]
        if len(sv) > 0:
            b_values = y[sv] - np.dot(K[sv, :], ay)
            return float(np.mean(b_values))

        return float(fallback_b)

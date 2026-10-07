"""
Quadratic Programming (QP) solver for dual SVM using scipy.optimize.
Useful as an exact reference solver for PhD benchmarks and small-to-medium datasets.
"""

from typing import Optional, Tuple
import numpy as np
from scipy.optimize import minimize


class QPSolver:
    """
    Solves the dual quadratic programming problem for SVM:
    
    min_alpha 0.5 * alpha^T Q alpha - 1^T alpha
    s.t.      0 <= alpha_i <= C
              y^T alpha = 0
    where Q_ij = y_i y_j K(x_i, x_j).
    """

    def __init__(self, C: float = 1.0, tol: float = 1e-5, max_iter: int = 1000):
        self.C = float(C)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def solve(
        self,
        K: np.ndarray,
        y: np.ndarray,
        initial_alpha: Optional[np.ndarray] = None,
        initial_b: float = 0.0,
    ) -> Tuple[np.ndarray, float, int]:
        """
        Solve dual problem using SLSQP.
        """
        n_samples = K.shape[0]
        y = np.asarray(y, dtype=float)

        # Matrix Q = y * K * y
        Q = np.outer(y, y) * K

        def objective(alpha):
            return 0.5 * np.dot(alpha, np.dot(Q, alpha)) - np.sum(alpha)

        def gradient(alpha):
            return np.dot(Q, alpha) - np.ones_like(alpha)

        # Equality constraint: y^T alpha = 0
        eq_constraint = {
            "type": "eq",
            "fun": lambda alpha: np.dot(y, alpha),
            "jac": lambda alpha: y,
        }

        # Box bounds: 0 <= alpha_i <= C
        bounds = [(0.0, self.C) for _ in range(n_samples)]

        if initial_alpha is not None:
            x0 = np.clip(initial_alpha, 0.0, self.C)
        else:
            x0 = np.zeros(n_samples)

        res = minimize(
            fun=objective,
            x0=x0,
            jac=gradient,
            bounds=bounds,
            constraints=[eq_constraint],
            method="SLSQP",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        alpha = np.maximum(0.0, res.x)
        alpha[alpha < 1e-7] = 0.0

        # Compute intercept b from margin support vectors (0 < alpha_i < C)
        margin_idx = np.where((alpha > 1e-5) & (alpha < self.C - 1e-5))[0]
        if len(margin_idx) > 0:
            # y_i * (sum_j alpha_j y_j K_ji + b) = 1 => b = y_i - sum_j alpha_j y_j K_ji
            ay = alpha * y
            b_values = y[margin_idx] - np.dot(K[margin_idx, :], ay)
            b = float(np.mean(b_values))
        else:
            sv_idx = np.where(alpha > 1e-5)[0]
            if len(sv_idx) > 0:
                ay = alpha * y
                b_values = y[sv_idx] - np.dot(K[sv_idx, :], ay)
                b = float(np.mean(b_values))
            else:
                b = float(initial_b)

        return alpha, b, int(res.nit) if hasattr(res, "nit") else 1

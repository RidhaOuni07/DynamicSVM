"""
Quadratic Programming (QP) solver for dual SVM using scipy.optimize.
Useful as a reference solver for small-to-medium datasets (dense SLSQP is
O(n^3) per iteration).
"""

import warnings
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from dynamicsvm.utils.kkt import compute_intercept, kkt_gap


class QPSolver:
    """
    Solves the dual quadratic programming problem for SVM:
    
    min_alpha 0.5 * alpha^T Q alpha - 1^T alpha
    s.t.      0 <= alpha_i <= C
              y^T alpha = 0
    where Q_ij = y_i y_j K(x_i, x_j).

    Parameters
    ----------
    C : float, default=1.0
        Box constraint.
    tol : float, default=1e-5
        Acceptance threshold on the maximal-violating-pair KKT gap. A
        solution with a larger gap (or an SLSQP failure) is still returned but
        triggers a ``RuntimeWarning``.
    max_iter : int, default=1000
        Maximum SLSQP iterations.
    ftol : float, default=1e-10
        SLSQP objective tolerance, independent of ``tol``.

    Attributes
    ----------
    result_ : scipy.optimize.OptimizeResult
        Raw SLSQP result of the last solve.
    kkt_gap_ : float
        KKT gap of the returned alpha.
    """

    def __init__(self, C: float = 1.0, tol: float = 1e-5, max_iter: int = 1000, ftol: float = 1e-10):
        self.C = float(C)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.ftol = float(ftol)
        self.result_ = None
        self.kkt_gap_: Optional[float] = None

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
        C = self.C

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
        bounds = [(0.0, C) for _ in range(n_samples)]

        if initial_alpha is not None:
            x0 = np.clip(np.asarray(initial_alpha, dtype=float), 0.0, C)
        else:
            x0 = np.zeros(n_samples)

        res = minimize(
            fun=objective,
            x0=x0,
            jac=gradient,
            bounds=bounds,
            constraints=[eq_constraint],
            method="SLSQP",
            options={"maxiter": self.max_iter, "ftol": self.ftol},
        )
        self.result_ = res

        alpha = np.clip(res.x, 0.0, C)

        # Snap multipliers within round-off of a bound onto the bound
        snap = 1e-8 * C
        alpha[alpha < snap] = 0.0
        alpha[alpha > C - snap] = C

        # Restore sum(alpha * y) = 0 exactly by shifting the residual onto the
        # free multipliers (sum_free y_i^2 r / n_free = r), if that stays in the box.
        residual = float(np.dot(alpha, y))
        free = (alpha > 0.0) & (alpha < C)
        if residual != 0.0 and np.any(free):
            shifted = alpha[free] - y[free] * residual / np.count_nonzero(free)
            if np.all(shifted >= 0.0) and np.all(shifted <= C):
                alpha[free] = shifted

        b = compute_intercept(alpha, K, y, C, eps=snap, fallback_b=initial_b)
        self.kkt_gap_ = kkt_gap(alpha, K, y, C)

        if not res.success:
            warnings.warn(
                f"QPSolver: SLSQP did not converge ({res.message}); KKT gap = {self.kkt_gap_:.3e}.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif self.kkt_gap_ > self.tol:
            warnings.warn(
                f"QPSolver: KKT gap {self.kkt_gap_:.3e} exceeds tol={self.tol:.1e}; "
                f"consider a larger max_iter or smaller ftol.",
                RuntimeWarning,
                stacklevel=2,
            )

        return alpha, b, int(res.nit) if hasattr(res, "nit") else 1

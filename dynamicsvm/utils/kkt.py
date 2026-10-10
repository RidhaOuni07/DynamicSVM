"""
Optimality diagnostics for SVM duals.

Binary C-SVM dual (the convention used by every binary solver here):

    min_a  D(a) = 0.5 a^T Q a - 1^T a,   Q_ij = y_i y_j K_ij
    s.t.   0 <= a_i <= C,   sum_i a_i y_i = 0,
    f(x) = sum_i a_i y_i K(x_i, x) + b.

One-class dual (COSVM family):

    min_a  0.5 a^T G a   s.t.   0 <= a_i <= C,   sum_i a_i = 1,
    f(x) = score(x) - rho.

The same functions are used by solvers (acceptance checks) and by tests, so
the measured optimality quantity is identical everywhere.
"""

from typing import Dict, Optional
import numpy as np


def dual_objective(alpha: np.ndarray, K: np.ndarray, y: np.ndarray) -> float:
    """Binary dual objective D(a) = 0.5 a^T (yy^T * K) a - sum(a)."""
    alpha = np.asarray(alpha, dtype=float)
    y = np.asarray(y, dtype=float)
    ay = alpha * y
    return float(0.5 * ay @ (np.asarray(K, dtype=float) @ ay) - np.sum(alpha))


def kkt_gap(
    alpha: np.ndarray,
    K: np.ndarray,
    y: np.ndarray,
    C: float,
    eps: float = 1e-8,
) -> float:
    """
    libsvm maximal-violating-pair gap m(a) - M(a), clipped at 0.

    With G = Q a - 1 (gradient of D):
    m(a) = max_{i in I_up} -y_i G_i,  M(a) = min_{i in I_low} -y_i G_i,
    I_up  = {a_i < C, y_i = +1} u {a_i > 0, y_i = -1},
    I_low = {a_i < C, y_i = -1} u {a_i > 0, y_i = +1}.
    a is optimal iff the gap is <= 0 (Keerthi et al. 2001; Fan et al. 2005).
    """
    alpha = np.asarray(alpha, dtype=float)
    y = np.asarray(y, dtype=float)
    G = y * (np.asarray(K, dtype=float) @ (alpha * y)) - 1.0
    v = -y * G
    up = ((alpha < C - eps) & (y > 0)) | ((alpha > eps) & (y < 0))
    low = ((alpha < C - eps) & (y < 0)) | ((alpha > eps) & (y > 0))
    if not np.any(up) or not np.any(low):
        return 0.0
    return float(max(0.0, np.max(v[up]) - np.min(v[low])))


def compute_intercept(
    alpha: np.ndarray,
    K: np.ndarray,
    y: np.ndarray,
    C: float,
    eps: float = 1e-8,
    fallback_b: float = 0.0,
) -> float:
    """
    Intercept b consistent with the KKT conditions.

    With e_i = y_i - (K (a * y))_i:
    - free SVs (eps < a_i < C - eps): b = mean(e_i) over free SVs;
    - otherwise b is the midpoint of the admissible interval [L, U],
      L = max{e_i : (a_i = 0, y_i = +1) or (a_i = C, y_i = -1)},
      U = min{e_i : (a_i = 0, y_i = -1) or (a_i = C, y_i = +1)};
      if only one side is non-empty that bound is returned.
    """
    alpha = np.asarray(alpha, dtype=float)
    y = np.asarray(y, dtype=float)
    e = y - np.asarray(K, dtype=float) @ (alpha * y)

    free = (alpha > eps) & (alpha < C - eps)
    if np.any(free):
        return float(np.mean(e[free]))

    at_zero = alpha <= eps
    at_C = alpha >= C - eps
    lower_mask = (at_zero & (y > 0)) | (at_C & (y < 0))
    upper_mask = (at_zero & (y < 0)) | (at_C & (y > 0))
    has_L = bool(np.any(lower_mask))
    has_U = bool(np.any(upper_mask))
    if has_L and has_U:
        return float(0.5 * (np.max(e[lower_mask]) + np.min(e[upper_mask])))
    if has_L:
        return float(np.max(e[lower_mask]))
    if has_U:
        return float(np.min(e[upper_mask]))
    return float(fallback_b)


def intercept_interval(
    alpha: np.ndarray,
    K: np.ndarray,
    y: np.ndarray,
    C: float,
    eps: float = 1e-8,
):
    """Return (L, U) of the admissible intercept interval (+-inf for empty sides)."""
    alpha = np.asarray(alpha, dtype=float)
    y = np.asarray(y, dtype=float)
    e = y - np.asarray(K, dtype=float) @ (alpha * y)
    at_zero = alpha <= eps
    at_C = alpha >= C - eps
    lower_mask = (at_zero & (y > 0)) | (at_C & (y < 0))
    upper_mask = (at_zero & (y < 0)) | (at_C & (y > 0))
    L = float(np.max(e[lower_mask])) if np.any(lower_mask) else -np.inf
    U = float(np.min(e[upper_mask])) if np.any(upper_mask) else np.inf
    return L, U


def kkt_report(
    alpha: np.ndarray,
    K: np.ndarray,
    y: np.ndarray,
    C: float,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Optimality summary of a binary C-SVM dual point.

    Returns
    -------
    dict with keys
        sum_alpha_y    : |sum_i a_i y_i| (equality-constraint violation)
        kkt_gap        : maximal-violating-pair gap (0 at the optimum)
        dual_objective : D(a)
        box_violation  : max(0, -min a, max a - C)
    """
    alpha = np.asarray(alpha, dtype=float)
    y = np.asarray(y, dtype=float)
    return {
        "sum_alpha_y": float(abs(np.dot(alpha, y))),
        "kkt_gap": kkt_gap(alpha, K, y, C, eps=eps),
        "dual_objective": dual_objective(alpha, K, y),
        "box_violation": float(max(0.0, -np.min(alpha), np.max(alpha) - C)) if alpha.size else 0.0,
    }


def oneclass_kkt_report(
    alpha: np.ndarray,
    G: np.ndarray,
    C: float,
    rho: Optional[float] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Optimality summary of a one-class dual point (sum(a) = 1, 0 <= a <= C).

    The KKT conditions say there is rho with (G a)_i <= rho for a_i > 0 and
    (G a)_j >= rho for a_j < C, so the gap
    max_{a_i > eps} (G a)_i - min_{a_j < C - eps} (G a)_j (clipped at 0)
    is 0 at the optimum.

    Returns
    -------
    dict with keys
        sum_alpha_minus_1 : |sum a - 1|
        kkt_gap           : see above
        dual_objective    : 0.5 a^T G a
        box_violation     : max(0, -min a, max a - C)
        outlier_fraction  : fraction of training points with (G a)_i - rho < 0
                            (only when rho is given)
    """
    alpha = np.asarray(alpha, dtype=float)
    Ga = np.asarray(G, dtype=float) @ alpha
    pos = alpha > eps
    below_C = alpha < C - eps
    if np.any(pos) and np.any(below_C):
        gap = float(max(0.0, np.max(Ga[pos]) - np.min(Ga[below_C])))
    else:
        gap = 0.0
    report = {
        "sum_alpha_minus_1": float(abs(np.sum(alpha) - 1.0)),
        "kkt_gap": gap,
        "dual_objective": float(0.5 * alpha @ Ga),
        "box_violation": float(max(0.0, -np.min(alpha), np.max(alpha) - C)) if alpha.size else 0.0,
    }
    if rho is not None:
        report["outlier_fraction"] = float(np.mean(Ga - float(rho) < 0.0))
    return report

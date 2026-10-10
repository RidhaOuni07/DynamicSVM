"""
Platt scaling of SVM decision values.

P(y = 1 | f) = 1 / (1 + exp(A f + B)), with (A, B) fitted by maximum
likelihood on out-of-sample decision values using the Newton method with
backtracking line search of Lin, Lin & Weng (2007), "A note on Platt's
probabilistic outputs for support vector machines", Machine Learning 68.
Targets are smoothed as in Platt (1999): t+ = (N+ + 1)/(N+ + 2),
t- = 1/(N- + 2).
"""

from typing import Tuple
import numpy as np


def _platt_objective(z: np.ndarray, t: np.ndarray) -> float:
    # sum_i t_i z_i + log(1 + exp(-z_i)), written stably with logaddexp
    return float(np.sum(t * z + np.logaddexp(0.0, -z)))


def fit_platt_sigmoid(
    decision_values: np.ndarray,
    y01: np.ndarray,
    max_iter: int = 100,
    min_step: float = 1e-10,
    sigma: float = 1e-12,
) -> Tuple[float, float]:
    """
    Fit Platt's sigmoid parameters.

    Parameters
    ----------
    decision_values : np.ndarray of shape (n_samples,)
        Out-of-sample decision values f_i.
    y01 : np.ndarray of shape (n_samples,)
        Labels in {0, 1} (1 = positive class).

    Returns
    -------
    A, B : float
        P(y = 1 | f) = 1 / (1 + exp(A f + B)).
    """
    f = np.asarray(decision_values, dtype=float).ravel()
    y = np.asarray(y01).ravel()
    if f.shape[0] != y.shape[0]:
        raise ValueError("decision_values and y01 must have the same length.")
    pos = y == 1
    prior1 = float(np.sum(pos))
    prior0 = float(len(y) - prior1)

    hi_target = (prior1 + 1.0) / (prior1 + 2.0)
    lo_target = 1.0 / (prior0 + 2.0)
    t = np.where(pos, hi_target, lo_target)

    A = 0.0
    B = float(np.log((prior0 + 1.0) / (prior1 + 1.0)))
    fval = _platt_objective(f * A + B, t)

    for _ in range(max_iter):
        z = f * A + B
        p = 1.0 / (1.0 + np.exp(np.clip(z, -500.0, 500.0)))  # P(y=1 | f)
        q = 1.0 - p
        d2 = p * q
        h11 = sigma + np.sum(f * f * d2)
        h22 = sigma + np.sum(d2)
        h21 = np.sum(f * d2)
        d1 = t - p
        g1 = np.sum(f * d1)
        g2 = np.sum(d1)

        if abs(g1) < 1e-5 and abs(g2) < 1e-5:
            break

        det = h11 * h22 - h21 * h21
        dA = -(h22 * g1 - h21 * g2) / det
        dB = -(-h21 * g1 + h11 * g2) / det
        gd = g1 * dA + g2 * dB

        step = 1.0
        while step >= min_step:
            new_A = A + step * dA
            new_B = B + step * dB
            new_f = _platt_objective(f * new_A + new_B, t)
            if new_f < fval + 1e-4 * step * gd:
                A, B, fval = new_A, new_B, new_f
                break
            step *= 0.5

        if step < min_step:
            break

    return float(A), float(B)


def platt_proba(f: np.ndarray, A: float, B: float) -> np.ndarray:
    """Return P(y = 1 | f) = 1 / (1 + exp(A f + B))."""
    z = np.asarray(f, dtype=float) * A + B
    return 1.0 / (1.0 + np.exp(np.clip(z, -500.0, 500.0)))

"""
Shared geometry, dual solver and offset for the COSVM family.

COSVM, ICOSVM and IDCOSVM all use the functions in this module, so the three
models solve the same finite problem and produce the same decision function
on the same buffer.

Primal -> dual derivation
-------------------------
Let phi be the kernel feature map, Phi = [phi(x_1), ..., phi(x_N)],
K = Phi^T Phi, H = I - (1/N) 1 1^T and

    Sigma = (1/N) Phi H Phi^T

the (biased) empirical covariance operator in feature space. The COSVM
primal (Khan et al. 2014; Kefi-Fatteh et al. 2019) is

    min_{w, rho, xi}  1/2 w^T M w - rho + C sum_i xi_i
    s.t.              w^T phi(x_i) >= rho - xi_i,   xi_i >= 0,

    M = eta I + (1 - eta) Sigma,   0 < eta <= 1,   C = 1 / (nu N).

M is positive definite for eta > 0, so the problem is strictly convex in w.
Stationarity of the Lagrangian gives

    M w = Phi alpha,   sum_i alpha_i = 1,   0 <= alpha_i <= C,

hence w = M^{-1} Phi alpha and the dual is

    min_alpha  1/2 alpha^T G alpha
    s.t.       sum_i alpha_i = 1,   0 <= alpha_i <= C,

    G = Phi^T M^{-1} Phi.

Kernelisation. M maps span(Phi) into itself, so M^{-1} Phi = Phi Z with
(eta I + ((1 - eta)/N) H K) Z = I. Therefore

    G = K Z = K (eta I + ((1 - eta)/N) H K)^{-1} = K A^{-1} K,
    A = eta K + ((1 - eta)/N) K H K          (symmetric PSD),

and the out-of-sample score is

    w^T phi(x) = phi(x)^T Phi Z alpha = k(x)^T A^{-1} K alpha = k(x)^T coef,
    coef = A^{-1} K alpha,

so f(x) = k(x)^T coef - rho. On training rows K coef = G alpha, i.e. the
training scores used by the KKT conditions and the test scores are the same
function. For a linear kernel G = X M^{-1} X^T exactly.

Numerics
--------
A is singular whenever K is (for example a linear kernel with N > d), so it
is regularised as A + eps I with eps = reg * trace(A) / N and factored once
by Cholesky. For a linear kernel K (A + eps I)^{-1} K -> X M^{-1} X^T as
eps -> 0. eta = 1 short-circuits to G = K, coef = alpha (exactly the
standard one-class SVM). eta = 0 is rejected: A is then singular along
K^{-1} 1, which leaves the training dual unchanged but makes out-of-sample
scores scale like 1/eps.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.optimize import minimize

from dynamicsvm.utils.kkt import oneclass_kkt_report


def validate_eta(eta: float) -> float:
    """Return eta as a float, raising unless 0 < eta <= 1."""
    eta = float(eta)
    if not (0.0 < eta <= 1.0):
        raise ValueError(
            f"eta must be in (0, 1], got {eta}. eta = 0 makes the "
            "out-of-sample COSVM score ill-defined (A is singular)."
        )
    return eta


class COSVMGeometry:
    """
    Dual Gram matrix G = K A^{-1} K and score coefficients of a COSVM.

    Parameters
    ----------
    K : ndarray of shape (N, N)
        Kernel Gram matrix of the training buffer.

    eta : float
        Trade-off in (0, 1].

    reg : float, default=1e-10
        Relative ridge: A + eps I with eps = reg * trace(A) / N.

    Attributes
    ----------
    K : ndarray
        Symmetrised kernel matrix.

    G : ndarray
        Dual Gram matrix (symmetric PSD).

    eps_ : float
        Ridge actually added to A (0.0 when eta == 1).
    """

    def __init__(
        self,
        K: np.ndarray,
        eta: float,
        reg: float = 1e-10,
    ):
        K = np.asarray(K, dtype=float)

        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError(
                "Training Gram matrix must be square."
            )

        if reg <= 0:
            raise ValueError(
                f"reg must be > 0, got {reg}"
            )

        self.K = 0.5 * (K + K.T)
        self.eta = validate_eta(eta)
        self.reg = float(reg)
        self.n_samples = self.K.shape[0]

        self.cho_ = None
        self.eps_ = 0.0

        if self.eta == 1.0:
            self.G = self.K.copy()
            return

        N = self.n_samples

        # K H = K - row means of K;  K H K = (K H) K.
        KH = self.K - np.mean(
            self.K,
            axis=1,
            keepdims=True,
        )

        A = (
            self.eta * self.K
            + ((1.0 - self.eta) / N) * (KH @ self.K)
        )
        A = 0.5 * (A + A.T)

        eps = self.reg * max(
            float(np.trace(A)) / N,
            np.finfo(float).tiny,
        )

        # Rounding in A can exceed a very small ridge for large N; grow
        # the ridge until the Cholesky factorisation exists.
        identity = np.eye(N)
        for _ in range(8):
            try:
                self.cho_ = cho_factor(
                    A + eps * identity,
                    lower=True,
                )
                break
            except LinAlgError:
                eps *= 10.0
        else:
            raise LinAlgError(
                "COSVM: A + eps I is not positive definite."
            )

        self.eps_ = float(eps)

        G = self.K @ cho_solve(self.cho_, self.K)
        self.G = 0.5 * (G + G.T)

    def coef(
        self,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """
        Score coefficients coef = A^{-1} K alpha (alpha itself when eta == 1).

        The score of x is k(x, X_train)^T coef.
        """
        alpha = np.asarray(alpha, dtype=float)

        if self.cho_ is None:
            return alpha.copy()

        return cho_solve(
            self.cho_,
            self.K @ alpha,
        )


def stabilize_psd(
    G: np.ndarray,
    psd_tol: float,
) -> np.ndarray:
    """
    Symmetrise G and remove rounding-level negative curvature.

    The tolerance is relative: tol = psd_tol * max(1, max |eig(G)|). An
    eigenvalue below -tol raises ValueError; a smaller negative minimum is
    shifted to zero.
    """
    G = 0.5 * (np.asarray(G, dtype=float) + np.asarray(G, dtype=float).T)

    if G.size == 0:
        return G

    eigenvalues = np.linalg.eigvalsh(G)
    min_eigenvalue = float(eigenvalues[0])

    tol = float(psd_tol) * max(
        1.0,
        float(np.max(np.abs(eigenvalues))),
    )

    if min_eigenvalue < -tol:
        raise ValueError(
            "The COSVM dual Gram matrix is not positive semidefinite. "
            f"Minimum eigenvalue: {min_eigenvalue:.3e} "
            f"(tolerance {tol:.3e})"
        )

    if min_eigenvalue < 0.0:
        G = G - min_eigenvalue * np.eye(G.shape[0])

    return G


def feasible_start(
    x0: Optional[np.ndarray],
    n_samples: int,
    C: float,
) -> np.ndarray:
    """
    Feasible point of {sum a = 1, 0 <= a <= C} close to x0.

    x0 is clipped to [0, C]; a total above 1 is scaled down (stays in the
    box), a deficit is water-filled onto the entries below C. Without a
    usable x0 the uniform point 1/N is returned (feasible since nu <= 1).
    """
    uniform = np.full(
        n_samples,
        1.0 / n_samples,
        dtype=float,
    )

    if x0 is None:
        return uniform

    x = np.asarray(x0, dtype=float).copy()

    if x.shape != (n_samples,) or not np.all(np.isfinite(x)):
        return uniform

    x = np.clip(x, 0.0, C)

    total = float(np.sum(x))
    if total > 1.0:
        x *= 1.0 / total

    deficit = 1.0 - float(np.sum(x))

    for _ in range(n_samples + 1):
        if deficit <= 1e-15:
            break

        room = C - x
        free = room > 0.0

        if not np.any(free):
            break

        share = deficit / int(np.sum(free))
        add = np.where(free, np.minimum(room, share), 0.0)

        x += add
        deficit -= float(np.sum(add))

    return np.clip(x, 0.0, C)


def _restore_sum(
    alpha: np.ndarray,
    C: float,
) -> np.ndarray:
    """
    Put the rounding residual 1 - sum(alpha) on free coefficients when that
    keeps them inside the box (otherwise alpha is returned unchanged).
    """
    residual = 1.0 - float(np.sum(alpha))

    if residual == 0.0:
        return alpha

    free = np.flatnonzero(
        (alpha > 0.0) & (alpha < C)
    )

    if free.size == 0:
        return alpha

    shifted = alpha.copy()
    shifted[free] += residual / free.size

    if np.all(shifted[free] >= 0.0) and np.all(shifted[free] <= C):
        return shifted

    return alpha


def _kkt_gap(
    alpha: np.ndarray,
    G: np.ndarray,
    C: float,
) -> float:
    return oneclass_kkt_report(alpha, G, C, eps=1e-12 * C)["kkt_gap"]


def _refine_pairwise(
    G: np.ndarray,
    alpha: np.ndarray,
    C: float,
    target_gap: float,
    max_steps: int,
) -> np.ndarray:
    """
    Maximal-violating-pair steps (one-class SMO, as in libsvm) from alpha.

    With g = G a, mass t moves from j = argmax_{a_j > 0} g_j to
    i = argmin_{a_i < C} g_i; t minimises the objective along that feasible
    direction, clipped to the box. Every step keeps sum a = 1 and
    0 <= a <= C exactly; the loop stops when g_j - g_i <= target_gap.
    """
    a = alpha.copy()
    g = G @ a
    diag = np.diag(G)

    for _ in range(max_steps):
        up = a < C
        down = a > 0.0

        if not np.any(up) or not np.any(down):
            break

        i = int(np.flatnonzero(up)[np.argmin(g[up])])
        j = int(np.flatnonzero(down)[np.argmax(g[down])])

        violation = g[j] - g[i]

        if violation <= target_gap or i == j:
            break

        curvature = diag[i] + diag[j] - 2.0 * G[i, j]
        t_max = min(C - a[i], a[j])

        if curvature > 0.0:
            t = min(violation / curvature, t_max)
        else:
            t = t_max

        if t <= 0.0:
            break

        a[i] = C if t == C - a[i] else a[i] + t
        a[j] = 0.0 if t == a[j] else a[j] - t

        g += t * (G[:, i] - G[:, j])

    return a


def _polish_active_set(
    G: np.ndarray,
    alpha: np.ndarray,
    C: float,
) -> np.ndarray:
    """
    Exact solve of the KKT system on the partition found by SLSQP.

    With O = {a = 0}, E = {a = C}, S = the rest, the optimum for this
    partition solves

        G_SS a_S + C G_SE 1 - rho 1 = 0,   1^T a_S = 1 - C |E|.

    The refined point is returned only if it is feasible and has a smaller
    KKT gap than alpha; otherwise alpha is returned unchanged.
    """
    S = np.flatnonzero((alpha > 0.0) & (alpha < C))
    E = np.flatnonzero(alpha >= C)

    if S.size == 0:
        return alpha

    n_s = S.size
    M = np.zeros((n_s + 1, n_s + 1), dtype=float)
    M[:n_s, :n_s] = G[np.ix_(S, S)]
    M[:n_s, n_s] = -1.0
    M[n_s, :n_s] = 1.0

    rhs = np.empty(n_s + 1, dtype=float)
    rhs[:n_s] = -C * np.sum(G[np.ix_(S, E)], axis=1) if E.size else 0.0
    rhs[n_s] = 1.0 - C * E.size

    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return alpha

    a_S = sol[:n_s]

    if not np.all(np.isfinite(a_S)) or np.any(a_S < 0.0) or np.any(a_S > C):
        return alpha

    refined = alpha.copy()
    refined[S] = a_S
    refined = _restore_sum(refined, C)

    if _kkt_gap(refined, G, C) < _kkt_gap(alpha, G, C):
        return refined

    return alpha


def solve_oneclass_dual(
    G: np.ndarray,
    C: float,
    x0: Optional[np.ndarray] = None,
    ftol: float = 1e-10,
    max_iter: int = 1000,
    feas_tol: float = 1e-8,
) -> Tuple[np.ndarray, object]:
    """
    Solve min 1/2 a^T G a  s.t.  sum a = 1, 0 <= a <= C  with SLSQP.

    Parameters
    ----------
    G : ndarray of shape (N, N)
        Symmetric PSD dual Gram matrix.

    C : float
        Box bound, must satisfy N C >= 1.

    x0 : ndarray or None
        Warm start; made feasible by :func:`feasible_start`.

    ftol : float, default=1e-10
        SLSQP precision goal.

    max_iter : int, default=1000
        SLSQP iteration limit.

    feas_tol : float, default=1e-8
        Maximum constraint violation accepted from SLSQP before snapping.

    Returns
    -------
    alpha : ndarray
        Feasible dual solution (snapped to the bounds within 1e-12 C).

    result : scipy.optimize.OptimizeResult or None
        Raw SLSQP result (None when N == 1).

    Raises
    ------
    RuntimeError
        If SLSQP returns an infeasible point, or fails and the refined
        point still has a relative KKT gap above 1e-6 (a successful SLSQP
        run with such a gap only warns).

    Notes
    -----
    The SLSQP point is refined by maximal-violating-pair steps and an
    exact solve of the KKT system on the identified partition, so the
    returned alpha does not depend on the warm start beyond rounding.
    """
    G = np.asarray(G, dtype=float)
    n_samples = G.shape[0]

    if n_samples == 0:
        raise ValueError("Cannot solve an empty one-class dual.")

    if n_samples * C < 1.0 - 1e-12:
        raise ValueError(
            "Infeasible one-class dual: N * C < 1."
        )

    if n_samples == 1:
        return np.array([1.0]), None

    start = feasible_start(x0, n_samples, C)

    def objective(alpha: np.ndarray) -> float:
        return 0.5 * float(alpha @ G @ alpha)

    def gradient(alpha: np.ndarray) -> np.ndarray:
        return G @ alpha

    constraints = [
        {
            "type": "eq",
            "fun": lambda alpha: np.sum(alpha) - 1.0,
            "jac": lambda alpha: np.ones_like(alpha),
        }
    ]

    bounds = [(0.0, C)] * n_samples

    result = minimize(
        fun=objective,
        x0=start,
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={
            "maxiter": int(max_iter),
            "ftol": float(ftol),
            "disp": False,
        },
    )

    alpha = np.asarray(result.x, dtype=float)

    # ------------------------------------------------------------------
    # Feasibility is checked BEFORE any cleanup.
    # ------------------------------------------------------------------

    sum_error = abs(float(np.sum(alpha)) - 1.0)
    lower_violation = float(-np.min(alpha))
    upper_violation = float(np.max(alpha) - C)

    if (
        not np.all(np.isfinite(alpha))
        or sum_error > feas_tol
        or lower_violation > feas_tol
        or upper_violation > feas_tol
    ):
        raise RuntimeError(
            "COSVM dual solver returned an infeasible point: "
            f"|sum(alpha)-1|={sum_error:.3e}, "
            f"min(alpha)={-lower_violation:.3e}, "
            f"max(alpha)-C={upper_violation:.3e} "
            f"({result.message})"
        )

    # Snap rounding-level bound violations and near-bound values.
    snap = 1e-12 * C
    alpha = np.clip(alpha, 0.0, C)
    alpha[alpha < snap] = 0.0
    alpha[alpha > C - snap] = C
    alpha = _restore_sum(alpha, C)

    # ------------------------------------------------------------------
    # Refinement. SLSQP stops on a relative objective decrease, which on
    # an ill-conditioned G leaves alpha off along low-curvature
    # directions (so two warm starts give visibly different alphas).
    # Feasible pairwise steps drive the KKT gap down, then the KKT system
    # of the identified partition is solved exactly.
    # ------------------------------------------------------------------

    # |(G a)_i| <= max_k G_kk for every feasible a (G PSD, sum a = 1), so
    # the diagonal is the natural scale of the scores and of the gap.
    scale = max(
        float(np.max(np.diag(G))),
        np.finfo(float).tiny,
    )

    alpha = _refine_pairwise(
        G,
        alpha,
        C,
        target_gap=1e-12 * scale,
        max_steps=50 * n_samples,
    )
    alpha = _polish_active_set(G, alpha, C)

    gap = _kkt_gap(alpha, G, C)

    if gap > 1e-6 * scale:
        message = (
            "COSVM dual optimisation: relative KKT gap "
            f"{gap / scale:.3e} after refinement (SLSQP: {result.message})"
        )

        if not result.success:
            raise RuntimeError(message)

        warnings.warn(
            message,
            RuntimeWarning,
            stacklevel=2,
        )

    return alpha, result


def compute_rho(
    G: np.ndarray,
    alpha: np.ndarray,
    C: float,
    alpha_tol: Optional[float] = None,
) -> float:
    """
    Offset rho from the one-class KKT conditions.

    With s = G alpha: free coefficients (alpha_tol < a_i < C - alpha_tol)
    satisfy s_i = rho, so rho is their mean. Without free coefficients rho
    is the midpoint of the admissible interval
    [max_{a_i = C} s_i, min_{a_i = 0} s_i] (one-sided bound if one side is
    empty).

    alpha_tol defaults to 1e-8 * C.
    """
    alpha = np.asarray(alpha, dtype=float)
    scores = np.asarray(G, dtype=float) @ alpha

    if alpha_tol is None:
        alpha_tol = 1e-8 * C

    free = (alpha > alpha_tol) & (alpha < C - alpha_tol)

    if np.any(free):
        return float(np.mean(scores[free]))

    at_C = alpha >= C - alpha_tol
    at_zero = alpha <= alpha_tol

    lower = float(np.max(scores[at_C])) if np.any(at_C) else -np.inf
    upper = float(np.min(scores[at_zero])) if np.any(at_zero) else np.inf

    if np.isfinite(lower) and np.isfinite(upper):
        return 0.5 * (lower + upper)

    if np.isfinite(lower):
        return lower

    if np.isfinite(upper):
        return upper

    return float(np.mean(scores))

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
B Z = I, where

    B = eta I + ((1 - eta)/N) H K.

The primary form of the dual Gram matrix is therefore

    G = K Z = K B^{-1},

which is valid unconditionally, because B is invertible for every
eta > 0: H K is similar to the PSD matrix K^{1/2} H K^{1/2} on range(K)
(and acts as zero on the complement), so spec(H K) lies in [0, inf) and
spec(B) in [eta, inf). No assumption on K is needed.

The familiar symmetric form

    G = K A^{-1} K,   A = K B = eta K + ((1 - eta)/N) K H K,

is equivalent to K B^{-1} ONLY when K is invertible, since A^{-1} =
B^{-1} K^{-1} requires K^{-1}. That precondition routinely fails here --
a linear kernel with N > d, duplicated rows, or any rank-deficient Gram
matrix -- and then K A^{-1} K is undefined while K B^{-1} is not. Treat
K A^{-1} K as shorthand for K B^{-1}, not as the definition.

What this module actually COMPUTES is the ridged symmetric form

    G = K (A + eps I)^{-1} K,   eps = reg * trace(A) / N,

and the symmetric A is still the right object to compute: A is symmetric
PSD, so A + eps I admits a Cholesky factorisation (one O(N^3) factor,
reused for coef), G comes out symmetric by construction, and the ridge is
relative to the scale of A. The non-symmetric K B^{-1} would need an
explicit symmetrisation and a general LU solve. As eps -> 0 the computed
G converges to K B^{-1} on range(K), with an error that is first order in
eps (see Numerics).

With that in hand, the out-of-sample score is

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
K^{-1} 1, which makes out-of-sample scores scale like 1/eps.

The training dual is not eta-invariant as eta -> 0 either: alpha only
SATURATES below about eta = 1e-4 (measured max|delta alpha| = 0.167
between eta = 0.5 and eta = 1e-4), so a small eta solves a materially
different problem rather than the same one. The ridge error in G is
likewise first order in reg and grows like 1/eta, because the smallest
eigenvalue of A scales with eta.
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
    Dual Gram matrix G = K B^{-1} and score coefficients of a COSVM.

    B = eta I + ((1 - eta)/N) H K is always invertible for eta > 0; the
    symmetric form K A^{-1} K with A = K B is the same matrix only when K
    is invertible, and what is computed is the ridged K (A + eps I)^{-1} K
    (see the module docstring for the derivation).

    Parameters
    ----------
    K : ndarray of shape (N, N)
        Kernel Gram matrix of the training buffer.

    eta : float
        Trade-off in (0, 1].

    reg : float, default=1e-10
        Relative ridge: A + eps I with eps = reg * trace(A) / N.

    psd_tol : float, default=1e-10
        Relative tolerance of the positive-semidefiniteness certificate,
        >= 0. At eta == 1 the dual Gram matrix is K itself, so K must be
        PSD: a Cholesky factorisation of K + psd_tol * max(1, max_k |K_kk|) I
        is attempted and, if it fails, ``stabilize_psd`` decides between
        rounding-level negative curvature (shifted to zero) and a genuinely
        indefinite kernel (ValueError). For eta < 1 the Cholesky of
        A + eps I already certifies G >= 0 and psd_tol is unused.

    Attributes
    ----------
    K : ndarray
        Symmetrised kernel matrix.

    G : ndarray
        Dual Gram matrix (symmetric PSD). At eta == 1 with a PSD kernel it
        is a bit-identical copy of K, which preserves the exact reduction
        to the standard one-class SVM.

    eps_ : float
        Ridge actually added to A (0.0 when eta == 1).

    n_ridge_growths_ : int
        Number of times eps had to be multiplied by 10 before the
        Cholesky factorisation of A + eps I existed. 0 when the first
        factorisation succeeded, and 0 on the eta == 1 path (no ridge).
        A non-zero value means rounding in A exceeded the requested
        relative ridge, so eps_ is 10 ** n_ridge_growths_ times
        reg * trace(A) / N.
    """

    def __init__(
        self,
        K: np.ndarray,
        eta: float,
        reg: float = 1e-10,
        psd_tol: float = 1e-10,
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

        if psd_tol < 0:
            raise ValueError(
                f"psd_tol must be >= 0, got {psd_tol}"
            )

        self.K = 0.5 * (K + K.T)
        self.eta = validate_eta(eta)
        self.reg = float(reg)
        self.psd_tol = float(psd_tol)
        self.n_samples = self.K.shape[0]

        self.cho_ = None
        self.eps_ = 0.0
        self.n_ridge_growths_ = 0

        if self.eta == 1.0:
            # G = K, so the dual is convex only if K is PSD. A jittered
            # Cholesky is a sufficient certificate and is cheaper than an
            # eigendecomposition; it succeeded on every PSD kernel tried
            # (rank-deficient linear, duplicated rows, d >> N) and failed
            # on an indefinite sigmoid kernel.
            if self.n_samples == 0:
                self.G = self.K.copy()
                return

            jitter = self.psd_tol * max(
                1.0,
                float(np.max(np.abs(np.diag(self.K)))),
            )

            try:
                cho_factor(
                    self.K + jitter * np.eye(self.n_samples),
                    lower=True,
                )
            except LinAlgError:
                # Either rounding-level negative curvature (shifted to
                # zero) or a genuinely indefinite kernel (ValueError).
                self.G = stabilize_psd(self.K, self.psd_tol)
            else:
                # Certificate passed: keep G bit-identical to K, which is
                # what makes eta = 1 reduce exactly to the one-class SVM
                # and keeps K @ coef == G @ alpha.
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
        for growths in range(8):
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
        self.n_ridge_growths_ = int(growths)

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
        Raw SLSQP result (None when N == 1), carrying three extra keys
        that certify the returned point:

        ``kkt_gap``
            max_i violation of the one-class KKT conditions of the
            RETURNED alpha, in the units of G.
        ``objective_scale``
            scale = max(max_k G_kk, tiny), the natural scale of the
            scores because |(G a)_i| <= max_k G_kk for every feasible a,
            and the constant the solve is normalised by (see Notes).
        ``kkt_gap_relative``
            ``kkt_gap / objective_scale``, dimensionless; this is the
            quantity the final gate tests, and it accepts below 1e-6.

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

    The problem is solved on the normalised Gram matrix
    G_n = G / max(max_k G_kk, tiny). Scaling a quadratic by a positive
    constant does not move its argmin, so the returned alpha is unchanged
    in meaning and no caller has to rescale anything (``compute_rho`` is
    deliberately called on the UNNORMALISED G, and must stay that way).
    The reason to normalise is that ftol is absolute on the objective
    while |G| grows like 1/eta: at eta = 1e-8, max_k G_kk is already
    ~8e+06, so an unnormalised objective of size O(1/eta) makes
    ftol = 1e-10 unreachable and SLSQP stops at a point that fails the
    absolute feasibility gate. On G_n the objective is O(1) at every eta.
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

    # |(G a)_i| <= max_k G_kk for every feasible a (G PSD, sum a = 1), so
    # the diagonal is the natural scale of the scores and of the gap. The
    # whole solve runs on G / scale: the argmin is identical, but the
    # objective is O(1) at every eta, which is what makes the absolute
    # ftol and the absolute feasibility gate meaningful (|G| ~ 1/eta).
    scale = max(
        float(np.max(np.diag(G))),
        np.finfo(float).tiny,
    )
    G_n = G / scale

    def objective(alpha: np.ndarray) -> float:
        return 0.5 * float(alpha @ G_n @ alpha)

    def gradient(alpha: np.ndarray) -> np.ndarray:
        return G_n @ alpha

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

    # On G_n the scale is exactly 1, so the gap and its target are
    # dimensionless; they are reported below in the units of G.
    alpha = _refine_pairwise(
        G_n,
        alpha,
        C,
        target_gap=1e-12,
        max_steps=50 * n_samples,
    )
    alpha = _polish_active_set(G_n, alpha, C)

    gap_n = _kkt_gap(alpha, G_n, C)

    if gap_n > 1e-6:
        message = (
            "COSVM dual optimisation: relative KKT gap "
            f"{gap_n:.3e} after refinement (SLSQP: {result.message})"
        )

        if not result.success:
            raise RuntimeError(message)

        warnings.warn(
            message,
            RuntimeWarning,
            stacklevel=2,
        )

    # OptimizeResult is a dict subclass, so attaching the certificate is
    # purely additive: callers unpacking the 2-tuple are unaffected.
    result["kkt_gap"] = float(gap_n * scale)
    result["objective_scale"] = float(scale)
    result["kkt_gap_relative"] = float(gap_n)

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

"""
Pegasos (Primal Estimated sub-GrAdient SOlver for SVM) algorithm.
Efficient primal stochastic subgradient descent method for linear SVM.
"""

from typing import Optional, Tuple
import numpy as np


class PegasosSolver:
    r"""
    Pegasos Primal Stochastic Subgradient Solver for Linear SVM
    (Shalev-Shwartz, Singer, Srebro & Cotter, 2011).

    With ``fit_intercept=True`` the bias is handled as a *regularised*
    augmented coordinate (Pegasos paper, Sec. 6; identical to liblinear's
    ``intercept_scaling``): x~ = [x, s], w~ = [w, b / s] with
    s = ``intercept_scaling``, and the solver minimises

        P(w, b) = (lambda / 2) (||w||^2 + (b / s)^2)
                  + (1 / n) sum_i max(0, 1 - y_i (<w, x_i> + b)),

    lambda = 1 / (C n). Multiplying by C n gives
    0.5 (||w||^2 + (b/s)^2) + C sum_i hinge_i, the objective of
    ``sklearn.svm.LinearSVC(loss='hinge', intercept_scaling=s)``. Every
    coordinate of w~ is lambda-strongly convex, so the 1/(lambda t) step size
    and the projection onto the ball ||w~|| <= 1/sqrt(lambda) are valid for
    the bias too. With ``fit_intercept=False`` the objective is the plain
    (lambda / 2)||w||^2 + mean hinge with no bias.

    With ``averaging=True`` the returned iterate is the suffix average of
    w~_t over the last half of the T steps, which attains the O(1/T) rate for
    strongly convex objectives (Rakhlin, Shamir & Sridharan, 2012); otherwise
    the last iterate is returned.

    Parameters
    ----------
    C : float, default=1.0
    max_iter : int, default=1000
        Number of epochs (passes over a random permutation of the data).
    batch_size : int, default=1
    random_state : int, optional
    fit_intercept : bool, default=True
    intercept_scaling : float, default=1.0
        Value s of the constant augmented feature (s > 0).
    averaging : bool, default=True
        Return the suffix average over the last half of the steps.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        batch_size: int = 1,
        random_state: Optional[int] = None,
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        averaging: bool = True,
    ):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.rng = np.random.RandomState(random_state)
        self.fit_intercept = fit_intercept
        self.intercept_scaling = float(intercept_scaling)
        self.averaging = bool(averaging)

        if self.intercept_scaling <= 0:
            raise ValueError("intercept_scaling must be strictly positive.")

    def solve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_w: Optional[np.ndarray] = None,
        initial_b: float = 0.0,
    ) -> Tuple[np.ndarray, float, int]:
        """
        Solve primal problem via stochastic subgradient descent.

        Returns
        -------
        w : np.ndarray of shape (n_features,)
        b : float
            0.0 when ``fit_intercept=False``.
        n_iter : int
            Number of epochs run.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        lam = 1.0 / (self.C * n_samples)
        s = self.intercept_scaling

        if self.fit_intercept:
            # Augmented design: x~ = [x, s]
            X_work = np.hstack([X, np.full((n_samples, 1), s)])
        else:
            X_work = X
        dim = X_work.shape[1]

        w = np.zeros(dim, dtype=float)
        if initial_w is not None:
            w[:n_features] = np.asarray(initial_w, dtype=float).ravel()
        if self.fit_intercept:
            # w~_{d+1} = b / s
            w[n_features] = float(initial_b) / s

        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        total_steps = self.max_iter * n_batches
        average_start = total_steps // 2  # suffix average over steps t > T/2
        w_sum = np.zeros(dim, dtype=float)
        n_averaged = 0
        radius = 1.0 / np.sqrt(lam)

        t = 0
        for epoch in range(self.max_iter):
            indices = self.rng.permutation(n_samples)
            for start_idx in range(0, n_samples, self.batch_size):
                t += 1
                eta = 1.0 / (lam * t)
                batch_indices = indices[start_idx : start_idx + self.batch_size]

                X_b = X_work[batch_indices]
                y_b = y[batch_indices]

                # Margins: y * <w~, x~>
                margins = y_b * np.dot(X_b, w)
                violators = margins < 1.0

                # Subgradient of (lambda/2)||w~||^2 + mean batch hinge
                grad = lam * w
                if np.any(violators):
                    grad = grad - np.dot(y_b[violators], X_b[violators]) / len(batch_indices)

                w = w - eta * grad

                # Projection onto the ball ||w~|| <= 1/sqrt(lambda)
                w_norm = np.linalg.norm(w)
                if w_norm > radius:
                    w *= radius / w_norm

                if self.averaging and t > average_start:
                    w_sum += w
                    n_averaged += 1

        if self.averaging and n_averaged > 0:
            w = w_sum / n_averaged

        if self.fit_intercept:
            return w[:n_features].copy(), float(w[n_features] * s), self.max_iter
        return w, 0.0, self.max_iter

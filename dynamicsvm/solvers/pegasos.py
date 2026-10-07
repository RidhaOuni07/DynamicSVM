"""
Pegasos (Primal Estimated sub-GrAdient SOlver for SVM) algorithm.
Efficient primal stochastic subgradient descent method for linear SVM.
"""

from typing import Optional, Tuple
import numpy as np


class PegasosSolver:
    """
    Pegasos Primal Stochastic Subgradient Solver for Linear SVM:
    min_w (lambda / 2) ||w||^2 + (1 / n) sum_i max(0, 1 - y_i (<w, x_i> + b))
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        batch_size: int = 1,
        random_state: Optional[int] = None,
        fit_intercept: bool = True,
    ):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.rng = np.random.RandomState(random_state)
        self.fit_intercept = fit_intercept

    def solve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_w: Optional[np.ndarray] = None,
        initial_b: float = 0.0,
    ) -> Tuple[np.ndarray, float, int]:
        """
        Solve primal problem via stochastic subgradient descent.
        """
        n_samples, n_features = X.shape
        lam = 1.0 / (self.C * n_samples)

        if initial_w is not None:
            w = np.asarray(initial_w, dtype=float).copy()
        else:
            w = np.zeros(n_features, dtype=float)

        b = float(initial_b)

        t = 0
        for epoch in range(self.max_iter):
            indices = self.rng.permutation(n_samples)
            for start_idx in range(0, n_samples, self.batch_size):
                t += 1
                eta = 1.0 / (lam * t)
                batch_indices = indices[start_idx : start_idx + self.batch_size]

                X_b = X[batch_indices]
                y_b = y[batch_indices]

                # Margins: y * (<w, x> + b)
                margins = y_b * (np.dot(X_b, w) + (b if self.fit_intercept else 0.0))
                violators = margins < 1.0

                # Gradient step
                grad_w = lam * w
                grad_b = 0.0

                if np.any(violators):
                    X_viol = X_b[violators]
                    y_viol = y_b[violators]
                    grad_w -= np.dot(y_viol, X_viol) / len(batch_indices)
                    if self.fit_intercept:
                        grad_b -= np.sum(y_viol) / len(batch_indices)

                w -= eta * grad_w
                if self.fit_intercept:
                    b -= eta * grad_b

                # Projection step: w = min(1, 1 / (sqrt(lambda) * ||w||)) * w
                w_norm = np.linalg.norm(w)
                if w_norm > 0:
                    scale = min(1.0, 1.0 / (np.sqrt(lam) * w_norm))
                    w *= scale

        return w, b, self.max_iter

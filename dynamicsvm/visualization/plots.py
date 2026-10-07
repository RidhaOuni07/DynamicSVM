"""
Decision boundary and margin plotting functions.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from dynamicsvm.base import BaseSVM


def plot_decision_boundary(
    model: BaseSVM,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "SVM Decision Boundary",
    show_support_vectors: bool = True,
    ax: Optional[plt.Axes] = None,
    grid_resolution: int = 200,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plot 2D decision boundary, margins, and support vectors.
    
    Parameters
    ----------
    model : BaseSVM
        Fitted SVM model.
    X : np.ndarray of shape (n_samples, 2)
        2D feature matrix.
    y : np.ndarray of shape (n_samples,)
        Class labels.
    title : str
        Title of the plot.
    show_support_vectors : bool
        Whether to highlight support vectors.
    ax : matplotlib.axes.Axes, optional
        Target axis. If None, a new figure is created.
    grid_resolution : int
        Resolution of contour meshgrid.
    save_path : str, optional
        File path to save the generated figure.
    """
    if X.shape[1] != 2:
        raise ValueError(f"plot_decision_boundary requires 2D features, got {X.shape[1]} features.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Determine mesh bounds with padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    if hasattr(model, "decision_function"):
        Z = model.decision_function(grid_points)
        if Z.ndim > 1 and Z.shape[1] > 1:
            Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)

        # Plot filled contour for decision region and lines for margins (-1, 0, 1)
        ax.contourf(xx, yy, Z > 0, alpha=0.2, cmap="coolwarm")
        contours = ax.contour(
            xx,
            yy,
            Z,
            levels=[-1.0, 0.0, 1.0],
            linestyles=["--", "-", "--"],
            colors=["black", "black", "black"],
            linewidths=[1.2, 2.0, 1.2],
        )
        ax.clabel(contours, inline=True, fontsize=9, fmt={-1.0: "-1 (margin)", 0.0: "0 (boundary)", 1.0: "+1 (margin)"})
    else:
        Z = model.predict(grid_points)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")

    # Scatter training data
    classes = np.unique(y)
    colors = ["blue" if cls == classes[0] else "red" for cls in y]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=colors, edgecolors="k", s=45, alpha=0.85, label="Data Points")

    # Highlight support vectors
    if show_support_vectors and hasattr(model, "support_vectors_") and model.support_vectors_ is not None:
        sv = model.support_vectors_
        ax.scatter(
            sv[:, 0],
            sv[:, 1],
            s=120,
            facecolors="none",
            edgecolors="gold",
            linewidths=2.0,
            label=f"Support Vectors ({len(sv)})",
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature 1", fontsize=10)
    ax.set_ylabel("Feature 2", fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax

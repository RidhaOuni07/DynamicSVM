"""
Unit tests for SVM optimization solvers (SMO, QP, Pegasos).
"""

import numpy as np
from dynamicsvm.kernels import LinearKernel, RBFKernel
from dynamicsvm.solvers.smo import SMOSolver
from dynamicsvm.solvers.qp import QPSolver
from dynamicsvm.solvers.pegasos import PegasosSolver


def test_smo_and_qp_linearly_separable():
    # Simple linearly separable 2D dataset
    X = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 0.0],
        [-1.0, -1.0],
        [-2.0, 0.0],
        [-2.0, -2.0],
    ])
    y = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])

    kernel = LinearKernel()
    K = kernel(X, X)

    # 1. SMO Solver
    smo = SMOSolver(C=10.0, tol=1e-4)
    alpha_smo, b_smo, _ = smo.solve(K, y)

    # 2. QP Solver
    qp = QPSolver(C=10.0, tol=1e-5)
    alpha_qp, b_qp, _ = qp.solve(K, y)

    # Both must satisfy box constraints 0 <= alpha <= C
    assert np.all(alpha_smo >= -1e-5)
    assert np.all(alpha_smo <= 10.0 + 1e-5)
    assert np.all(alpha_qp >= -1e-5)
    assert np.all(alpha_qp <= 10.0 + 1e-5)

    # Check that predictions match
    decision_smo = np.dot(K, alpha_smo * y) + b_smo
    decision_qp = np.dot(K, alpha_qp * y) + b_qp

    preds_smo = np.where(decision_smo >= 0, 1.0, -1.0)
    preds_qp = np.where(decision_qp >= 0, 1.0, -1.0)

    np.testing.assert_array_equal(preds_smo, y)
    np.testing.assert_array_equal(preds_qp, y)


def test_pegasos_linear():
    X = np.array([
        [2.0, 2.0],
        [3.0, 3.0],
        [-2.0, -2.0],
        [-3.0, -3.0],
    ])
    y = np.array([1.0, 1.0, -1.0, -1.0])

    pegasos = PegasosSolver(C=1.0, max_iter=200, random_state=42)
    w, b, _ = pegasos.solve(X, y)

    preds = np.where(np.dot(X, w) + b >= 0, 1.0, -1.0)
    np.testing.assert_array_equal(preds, y)

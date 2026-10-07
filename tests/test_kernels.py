"""
Unit tests for SVM kernels.
"""

import numpy as np
import pytest
from dynamicsvm.kernels import (
    LinearKernel,
    RBFKernel,
    PolynomialKernel,
    SigmoidKernel,
    CustomKernel,
    get_kernel,
)


def test_linear_kernel():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    kernel = LinearKernel(offset=1.0)
    K = kernel(X)
    expected = np.dot(X, X.T) + 1.0
    np.testing.assert_allclose(K, expected)
    np.testing.assert_allclose(kernel.diag(X), np.diag(expected))


def test_rbf_kernel():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    kernel = RBFKernel(gamma=0.5)
    K = kernel(X)
    # K[0, 0] = 1, K[1, 1] = 1, K[0, 1] = exp(-0.5 * 2) = exp(-1)
    expected = np.array([[1.0, np.exp(-1.0)], [np.exp(-1.0), 1.0]])
    np.testing.assert_allclose(K, expected, rtol=1e-5)
    np.testing.assert_allclose(kernel.diag(X), [1.0, 1.0])


def test_poly_kernel():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    kernel = PolynomialKernel(degree=2, gamma=1.0, coef0=1.0)
    K = kernel(X)
    # <x0, x0> = 1 -> (1+1)^2 = 4
    # <x0, x1> = 0 -> (0+1)^2 = 1
    expected = np.array([[4.0, 1.0], [1.0, 4.0]])
    np.testing.assert_allclose(K, expected)


def test_sigmoid_kernel():
    X = np.array([[0.0, 0.0]])
    kernel = SigmoidKernel(gamma=1.0, coef0=0.0)
    K = kernel(X)
    np.testing.assert_allclose(K, [[0.0]])


def test_custom_kernel():
    def my_kernel(X, Y):
        return np.ones((X.shape[0], Y.shape[0]))

    kernel = get_kernel(my_kernel)
    X = np.zeros((3, 2))
    K = kernel(X)
    assert K.shape == (3, 3)
    assert np.all(K == 1.0)


def test_get_kernel_factory():
    assert isinstance(get_kernel("linear"), LinearKernel)
    assert isinstance(get_kernel("rbf"), RBFKernel)
    assert isinstance(get_kernel("poly", degree=4), PolynomialKernel)
    assert isinstance(get_kernel("sigmoid"), SigmoidKernel)
    with pytest.raises(ValueError):
        get_kernel("unknown_kernel")

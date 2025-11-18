"""A few linear algebra tests."""

import numpy as np
from pyimcom.lakernel import CholKernel, _assign_subvector, _extract_subvector


def test_incr():
    """Test function for 'fixing' a matrix."""

    N = 6
    A = np.zeros((N, N))
    d = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d[i, j] = 2 * np.pi * (i - j) / N
    for k in range(1, N // 2 + 1):
        A += np.cos(k * d) / k / N

    A = A - 1e-3 * np.identity(N)
    AA = A + 1e-4 * np.identity(N)
    L = CholKernel._cholesky_wrapper(AA, np.diag_indices(N), A)
    w, v = np.linalg.eigh(L @ L.T)
    assert np.abs(w[0] - 1e-4) < 1e-7


def test_subvec():
    """Test function for subvectors."""

    x = _extract_subvector(np.arange(7).astype(np.float64) - 2, np.arange(1, 4))
    assert x.size == 3
    assert np.abs(x[-1] - 1) < 1e-8

    u = np.arange(10).astype(np.float64)
    v = 2 * u
    _assign_subvector(u, v, np.arange(3, 6))
    assert np.amax(u - np.array([0.0, 1.0, 2.0, 0.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0])) < 1e-6

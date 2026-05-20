"""A few linear algebra tests."""

import numpy as np
from pyimcom.lakernel import CholKernel, EigenKernel, _assign_subvector, _extract_subvector


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


class EmptyClass:
    """Just a data container."""

    pass


def test_eigen():
    """Test function for EigenKernel."""

    N = 6
    A = np.zeros((N, N))
    d = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d[i, j] = 2 * np.pi * (i - j) / N
    for k in range(1, N // 2 + 1):
        A += np.cos(k * d) / k / N
    mBhalf = np.zeros((1, 16, N))
    for i in range(N):
        for j in range(16):
            _d = 2 * np.pi * (i - 0.4 * j) / N
            for k in range(1, N // 2 + 1):
                mBhalf[0, j, i] += np.cos(k * _d) / k / N
    C = A[0, 0]

    # make dummy configuration + stamp
    cfg = EmptyClass()
    cfg.n2f = 4
    cfg.n_out = 1
    cfg.kappaC_arr = [1e-2]
    cfg.uctarget = 1e-4
    cfg.sigmamax = 0.5
    blk = EmptyClass()
    blk.cfg = cfg
    outst = EmptyClass()
    outst.blk = blk
    outst.inpix_cumsum = np.array(
        [
            N,
        ]
    )
    outst.sysmata = A
    outst.mhalfb = mBhalf
    outst.outovlc = np.array(
        [
            C,
        ]
    )

    e = EigenKernel(outst)
    e()

    assert np.all(e.outst.UC >= 0)
    for j in range(16):
        if j % 5 == 0:
            assert e.outst.UC.ravel()[j] < 1.0e-4
        else:
            assert 0.05 < e.outst.UC.ravel()[j] < 0.2
        assert 0.6 < e.outst.Sigma.ravel()[j] < 1.0
        assert 0.002 < e.outst.kappa.ravel()[j] < 0.004


def test_eigen2():
    """Test function for EigenKernel."""

    N = 6
    A = np.zeros((N, N))
    d = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d[i, j] = 2 * np.pi * (i - j) / N
    for k in range(1, N // 2 + 1):
        A += np.cos(k * d) / k / N
    mBhalf = np.zeros((1, 16, N))
    for i in range(N):
        for j in range(16):
            _d = 2 * np.pi * (i - 0.4 * j) / N
            for k in range(1, N // 2 + 1):
                mBhalf[0, j, i] += np.cos(k * _d) / k / N
    C = A[0, 0]

    # make dummy configuration + stamp
    cfg = EmptyClass()
    cfg.n2f = 4
    cfg.n_out = 1
    cfg.kappaC_arr = [1e-4, 1e-3, 1e-2]
    cfg.uctarget = 1e-4
    cfg.sigmamax = 1.0
    blk = EmptyClass()
    blk.cfg = cfg
    outst = EmptyClass()
    outst.blk = blk
    outst.inpix_cumsum = np.array(
        [
            N,
        ]
    )
    outst.sysmata = A
    outst.mhalfb = mBhalf
    outst.outovlc = np.array(
        [
            C,
        ]
    )

    e = EigenKernel(outst)
    e()

    assert np.all(e.outst.UC >= 0)
    print(e.outst.UC)
    print(e.outst.Sigma)
    print(e.outst.kappa)

    for j in range(16):
        if j % 5 == 0:
            assert e.outst.UC.ravel()[j] < 1.0e-4
            assert 5e-4 < e.outst.kappa.ravel()[j] < 1.5e-3
        else:
            assert 0.05 < e.outst.UC.ravel()[j] < 0.2
            assert 5e-6 < e.outst.kappa.ravel()[j] < 1.5e-5
        assert 0.6 < e.outst.Sigma.ravel()[j] < 1.0

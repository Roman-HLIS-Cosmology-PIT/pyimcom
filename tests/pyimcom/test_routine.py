"""This test compares the C vs. Python numerical functions."""

import furry_parakeet.pyimcom_croutines as pcr
import numpy as np
from pyimcom.routine import gridD5512C, iD5512C, iD5512C_sym, lakernel1, lsolve_sps


def test_interp():
    """Tests interpolation."""

    # build function to interpolate
    nx = 32
    ny = 64
    N = 10
    npts = N**2
    infunc = np.sin(np.linspace(0, 200, 2 * nx * ny)).reshape((2, ny, nx))

    # and the output sample locations
    x_, _ = np.modf(np.arange(npts) / np.sqrt(5))
    x_ *= 40
    y_, _ = np.modf(np.arange(npts) * 2 / np.sqrt(5))
    y_ *= 40

    # now try interpolation both ways
    fhatout1 = np.zeros((2, npts))
    fhatout2 = np.zeros((2, npts))
    iD5512C(infunc, x_, y_, fhatout1)
    pcr.iD5512C(infunc, x_, y_, fhatout2)
    assert np.amax(np.abs(fhatout1)) > 0.98
    assert np.amax(np.abs(fhatout1 - fhatout2)) < 1e-9

    # another interpolation, this time symmetrical
    for i in range(1, N):
        for j in range(i):
            x_[i * N + j] = x_[j * N + i]
            y_[i * N + j] = y_[j * N + i]
    fhatout1 = np.zeros((2, npts))
    fhatout2 = np.zeros((2, npts))
    iD5512C_sym(infunc, x_, y_, fhatout1)
    iD5512C(infunc, x_, y_, fhatout2)
    assert np.amax(np.abs(fhatout1)) > 0.98
    assert np.amax(np.abs(fhatout1 - fhatout2)) < 1e-9

    # compare symmetrical against furry-parakeet version
    fhatout2 = np.zeros((2, npts))
    pcr.iD5512C(infunc, x_, y_, fhatout2)
    assert np.amax(np.abs(fhatout1 - fhatout2)) < 1e-9

    # compare grid version
    npi = 3
    nxo = 12
    nyo = 20
    xpos = np.zeros((npi, nxo))
    ypos = np.zeros((npi, nyo))
    for i in range(npi):
        xpos[i, :] = np.linspace(2 + i, nx - 2 - i, nxo)
        ypos[i, :] = np.linspace(2 + i, ny - 2 - i, nyo)
    fhatout1 = np.zeros((npi, nxo * nyo))
    fhatout2 = np.zeros((npi, nxo * nyo))
    gridD5512C(infunc[0, :, :], xpos, ypos, fhatout1)
    pcr.gridD5512C(infunc[0, :, :], xpos, ypos, fhatout2)
    assert np.amax(np.abs(fhatout1)) > 0.98
    assert np.amax(np.abs(fhatout1 - fhatout2)) < 1e-9


def test_kernel():
    """
    Test case for the kernel.

    This is nothing fancy.  The test interpolateion is a grid with Gaussian PSF.
    """

    sigma = 4.0

    # test grid: interpolate an m1xm1 image from n1xn1
    m1 = 25
    n1 = 33
    n = n1 * n1
    m = m1 * m1

    x = np.zeros((n,))
    y = np.zeros((n,))
    for i in range(n1):
        y[n1 * i : n1 * i + n1] = i
        x[i::n1] = i
    xout = np.zeros((m,))
    yout = np.zeros((m,))
    for i in range(m1):
        yout[m1 * i : m1 * i + m1] = 5 + 0.25 * i
        xout[i::m1] = 5 + 0.25 * i

    A = np.zeros((n, n))
    mBhalf = np.zeros((m, n))
    C = 1.0
    for i in range(n):
        for j in range(n):
            A[i, j] = np.exp(-1.0 / sigma**2 * ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2))
        for a in range(m):
            mBhalf[a, i] = np.exp(-1.0 / sigma**2 * ((x[i] - xout[a]) ** 2 + (y[i] - yout[a]) ** 2))

    # rescale everything
    A *= 0.7
    mBhalf *= 0.7
    C *= 0.7

    # get versions for kernel
    lam, Q = np.linalg.eigh(A)
    mPhalf = mBhalf @ Q

    # convergence parameters
    targetleak = 1e-8
    kCmin = 1e-16
    kCmax = 1e16
    nbis = 53
    smax = 0.5

    # make output arrays
    kappa1 = np.zeros((m,))
    Sigma1 = np.zeros((m,))
    UC1 = np.zeros((m,))
    T1 = np.zeros((m, n))
    kappa2 = np.zeros((m,))
    Sigma2 = np.zeros((m,))
    UC2 = np.zeros((m,))
    T2 = np.zeros((m, n))

    # now the computation
    lakernel1(lam, Q, mPhalf, C, targetleak, kCmin, kCmax, nbis, kappa1, Sigma1, UC1, T1, smax)
    pcr.lakernel1(lam, Q, mPhalf, C, targetleak, kCmin, kCmax, nbis, kappa2, Sigma2, UC2, T2, smax)

    assert np.amax(kappa1) < 3.5e-7
    assert np.amin(kappa1) > 2.5e-7
    assert np.amax(np.abs(kappa1 - kappa2)) < 1.0e-12

    assert np.amax(Sigma1) < 0.38
    assert np.amin(Sigma1) > 0.34
    assert np.amax(np.abs(Sigma1 - Sigma2)) < 1.0e-7

    assert np.amax(UC1) < 1.1e-8
    assert np.amin(UC1) > 9e-9
    assert np.amax(np.abs(UC1 - UC2)) < 1.0e-14

    assert np.amax(np.abs(T1)) > 0.077
    assert np.amax(np.abs(T1)) < 0.079
    assert np.amax(np.abs(T1 - T2)) < 1e-8

    # now use this as a test of lsolve_sps
    A_ = A + np.identity(n)
    b_ = mBhalf[0, :]
    x_ref = np.linalg.solve(A_, b_)
    x_ = np.zeros_like(x_ref)
    lsolve_sps(n, A_, x_, b_)

    assert np.amax(np.abs(x_ref)) > 0.070
    assert np.amax(np.abs(x_ref)) < 0.075
    assert np.amax(np.abs(x_ - x_ref)) < 1e-10

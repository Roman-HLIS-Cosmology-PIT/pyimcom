import numpy as np
from pyimcom.splitpsf.imsubtract import fftconvolve_multi
from scipy.signal import fftconvolve


def test_fftconvolve_multi():
    """Simple test."""

    # set up test functions
    u = np.linspace(0, 2047, 2048)
    x_, y_ = np.meshgrid(u[:256], u[:256])
    arr1 = x_ / (1.0 + 0.5 * np.cos(0.01 * y_**2))
    x_, y_ = np.meshgrid(u, u)
    arr2 = np.sin(x_ / 10.0) * np.exp(-0.01 * (y_ - 1400.0) ** 2)
    out1 = fftconvolve(arr1, arr2, mode="valid")
    out2 = np.zeros_like(out1)

    # this configuration was chosen to ensure that the horizontal bands
    # would be used.
    fftconvolve_multi(arr1, arr2, out2, mode="valid", verbose=True)

    print(np.amax(np.abs(out1)))
    print(np.amax(np.abs(out2)))
    print(np.amax(np.abs(out1 - out2)))
    assert np.amax(np.abs(out1 - out2)) < 1e-9 * np.amax(np.abs(out1))

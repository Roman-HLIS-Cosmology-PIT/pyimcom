"""Specialized PSF test functions."""

import numpy as np
from pyimcom.psfutil import OutPSF, PSFGrp

def test_simple_airy():
    """Test function for simple Airy disc."""

    # Setup
    # Note that 25*4 = 100
    PSFGrp.setup(npixpsf=25, oversamp=4, dtheta=0.1)

    # No obscuration
    im = OutPSF.psf_simple_airy(100, 4.0)
    fwhm = OutPSF.get_psf_fwhm(im)
    print(im[50, 50])
    print(np.sum(im))
    print(fwhm)
    assert np.abs(im[50, 50] - 0.045421877940855226) < 0.001
    assert np.abs(np.sum(im) - 0.9853733474017817) < 0.001
    assert np.abs(fwhm - 4.116308537118992) < 0.001

    # With obscuration
    im = OutPSF.psf_simple_airy(100, 4.0, obsc=0.5)
    fwhm = OutPSF.get_psf_fwhm(im)
    print(im[50, 50])
    print(np.sum(im))
    print(fwhm)
    assert np.abs(im[50, 50] - 0.03339794632862726) < 0.001
    assert np.abs(np.sum(im) - 0.970256598273068) < 0.001
    assert np.abs(fwhm - 3.647095402134021) < 0.001


def test_interpolators():
    """Simple test for interpolation weights."""

    w = np.zeros(10)
    OutPSF.iD5512C_getw(w, 0.5)
    w[5] -= 1.
    assert np.all(np.abs(w) < 1e-8)

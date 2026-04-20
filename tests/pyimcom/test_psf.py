"""Specialized PSF test functions."""

import matplotlib.pyplot as plt
import numpy as np
from pyimcom.psfutil import OutPSF, PSFGrp, PSFInterpolator


def test_simple_airy(tmp_path, monkeypatch):
    """Test function for simple Airy disc."""

    # This allows us to do visualizations in a test.
    # new place to save figures, in sequence
    _counter = [0]

    def save_instead_of_show(count=_counter):
        plt.savefig(tmp_path / f"plot_output_psf_{count[0]%1000:d}.png")
        count[0] += 1

    monkeypatch.setattr(plt, "show", save_instead_of_show)

    # Setup
    # Note that 25*4 = 100
    PSFGrp.setup(npixpsf=25, oversamp=4, dtheta=0.1)

    # No obscuration
    im = OutPSF.psf_simple_airy(100, 4.0)
    fwhm = OutPSF.get_psf_fwhm(im, visualize=True)
    print(im[50, 50])
    print(np.sum(im))
    print(fwhm)
    assert np.abs(im[50, 50] - 0.045421877940855226) < 0.001
    assert np.abs(np.sum(im) - 0.9853733474017817) < 0.001
    assert np.abs(fwhm - 4.116308537118992) < 0.001
    sig = OutPSF.get_psf_inv_width(im)
    assert 1.63 < sig < 1.66
    print(">>", sig, _counter, str(tmp_path))

    # visualize the PSF
    theta = np.pi / 6.0
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    yxco = 1.4 * np.einsum("ab,bij->aij", rot, np.mgrid[:100, :100] / 10.0 - 4.5)
    PSFGrp.visualize_psf(OutPSF.psf_simple_airy(25, 4.0), yxco, 12.5, 12.5)

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
    w[5] -= 1.0
    assert np.all(np.abs(w) < 1e-8)


def test_psf_interpolator():
    """Test toggling the PSF interpolator to ensure no errors."""

    PSFInterpolator.set_G4460()
    PSFInterpolator.unset_G4460()

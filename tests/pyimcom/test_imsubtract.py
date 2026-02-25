import numpy as np
from astropy.io import fits
from pyimcom.splitpsf.imsubtract import fftconvolve_multi, run_imsubtract_all
from pyimcom.splitpsf.splitpsf import split_psf_to_fits
from scipy.signal import fftconvolve

PSF_FILE = "https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/psf_test.fits"


def test_psfsplit(tmp_path):
    """PSF split test from remote file.

    Parameters
    ----------
    tmp_path : str
        Directory in which to run the test.

    Returns
    -------
    None

    """

    outfile = str(tmp_path) + "/split.fits"
    split_psf_to_fits(PSF_FILE, "missing_wcs_{:d}.fits", {}, outfile)

    maxG_expected = [
        0.008040595,
        0.007834151,
        0.0076568737,
        0.008104268,
        0.007863815,
        0.007660740,
        0.008082002,
        0.007931602,
        0.0077521205,
        0.007960151,
        0.007802289,
        0.007691010,
        0.007846788,
        0.007757318,
        0.0077059544,
        0.007690219,
        0.007671312,
        0.007695832,
    ]

    with fits.open(outfile) as f:
        print(f.info())

        # check header
        assert f[0].header["NSCA"] == 18
        assert f[0].header["OVSAMP"] == 6
        assert f[0].header["GSSKIP"] == 18
        assert f[0].header["KERSKIP"] == 36
        assert f[0].header["INWCS01"] == "/dev/null"

        # short range PSFs
        for i in range(1, 19):
            assert f[i].header["SCA"] == i
            assert f[i].header["NAXIS1"] == 384
            assert f[i].header["NAXIS2"] == 384
            assert f[i].header["NAXIS3"] == 4
            assert np.abs(np.amax(f[i].data) - maxG_expected[i - 1]) < 2e-7

            assert f[i + 18].header["SCA"] == i
            assert f[i + 18].header["NAXIS1"] == 384
            assert f[i + 18].header["NAXIS2"] == 384
            assert f[i + 18].header["NAXIS3"] == 4

            assert np.amax(f[i].data - f[i + 18].data) > 0.0024
            assert np.amax(f[i].data - f[i + 18].data) < 0.0028
            assert np.amin(f[i].data - f[i + 18].data) > -6e-4
            assert np.amin(f[i].data - f[i + 18].data) < -4e-4

            assert np.amax(np.abs(f[i + 36].data)) < 4e-5


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


def test_run_imsubtract_all_with_config(setup):
    """Test that run_imsubtract_all loads the config and looks for files in the right place.

    The setup fixture creates the config file and directory structure but doesn't
    populate the cache with the .fits files that run_imsubtract_all looks for.
    This test verifies the function can load the config and process correctly
    (finding zero files, as expected).
    """
    cfg = setup
    import pathlib

    cfg_file = str(pathlib.Path(cfg.inlayercache).parent.parent / "cfg.txt")

    # Verify the config file exists
    import os

    assert os.path.exists(cfg_file), f"Config file not found at {cfg_file}"

    # Call run_imsubtract_all with max_imgs=None and max_workers=1
    # It should successfully load the config, find zero images (cache is empty),
    # and complete without error
    run_imsubtract_all(cfg_file, workers=1, max_imgs=None, display="/dev/null")

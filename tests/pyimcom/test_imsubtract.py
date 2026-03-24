import os
import re
import subprocess
import urllib.request

import numpy as np
from astropy.io import fits
from pyimcom.splitpsf.imsubtract import fftconvolve_multi
from pyimcom.splitpsf.imsubtract_wrapper import run_imsubtract_all
from pyimcom.splitpsf.splitpsf import split_psf_to_fits
from pyimcom.splitpsf.splitpsf_wrapper import split_psf_single
from scipy.signal import fftconvolve

PSF_FILE = "https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/psf_test.fits"
IMSUBTRACT_CONFIG = "https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/imsubtract/test_imsubtract_config.json"
IMSUBTRACT_INPUT_PATH = "https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/imsubtract"


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


def test_psfsplit_single(tmp_path):
    """PSF split test using split_psf_single !! (from remote file).

    Parameters
    ----------
    tmp_path : str
        Directory in which to run the test.

    Returns
    -------
    None

    """

    outfile = str(tmp_path) + "/split2.fits"
    TEST_FILES = [PSF_FILE, "missing_wcs_{:d}.fits", outfile]
    split_psf_single(None, None, None, None, {}, TEST_FILES=TEST_FILES)

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


def test_run_imsubtract_all(tmp_path, config_file=IMSUBTRACT_CONFIG):
    """
    Test the run_imsubtract_all function.
    This test runs the imsubtract pipeline on a small set of images specified in the config file,
    and checks that the output files are created and have the expected properties.
    """

    tmp_dir = str(tmp_path)
    tmp_imsub = tmp_dir + "/temp_imsubtract"
    # make temp_imsub directory
    os.makedirs(tmp_imsub, exist_ok=True)
    os.makedirs(tmp_imsub + "/blocks", exist_ok=True)

    if config_file.startswith("http"):
        urllib.request.urlretrieve(config_file, tmp_imsub + "/test_imsubtract_config.json")
        config_file = tmp_imsub + "/test_imsubtract_config.json"

    # read cache files into tmp_imsub
    cache_files = [
        "r1_00013912_17.fits.gz",
        "r1_00000670_12.fits.gz",
        "r1_00013912_17_wcs.asdf",
        "r1_00000670_12_wcs.asdf",
    ]
    for filename in cache_files:
        cf_url = IMSUBTRACT_INPUT_PATH + "/cache/" + filename
        cf_local = os.path.join(tmp_imsub, filename)
        urllib.request.urlretrieve(cf_url, cf_local)
        if cf_local[-3:] == ".gz":
            subprocess.run(["gunzip", cf_local])  # files on wiki were gzipped
    block_files = [
        "im3x2-H1_32_00.fits",
        "im3x2-H1_35_02.fits",
        "im3x2-H1_36_02.fits",
        "im3x2-H1_37_02.fits",
    ]
    for filename in block_files:
        cf_url = IMSUBTRACT_INPUT_PATH + "/blocks/" + filename
        cf_local = os.path.join(tmp_imsub, filename)
        urllib.request.urlretrieve(cf_url, cf_local)

    # psf files
    tmp_psf = tmp_imsub + "/r1.psf"
    os.makedirs(tmp_psf, exist_ok=True)
    for filename in ["psf_13912.fits", "psf_670.fits"]:
        cf_url = IMSUBTRACT_INPUT_PATH + "/cache/r1.psf/" + filename
        cf_local = os.path.join(tmp_psf, filename)
        urllib.request.urlretrieve(cf_url, cf_local)

    with open(config_file, "r") as f:
        cfg_text = f.read()
    cfg_text = cfg_text.replace("$TMPDIR", tmp_dir)
    cfg_text = cfg_text.replace("$CACHE", tmp_imsub + "/r1")
    cfg_text = cfg_text.replace(
        "https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/imsubtract/blocks/im3x2-H1",
        tmp_imsub + "/blocks/im3x2-H1"
    )
    with open(config_file, "w") as f:
        f.write(cfg_text)

    # Diagnostic output
    print(cfg_text)
    print(os.walk(tmp_imsub))

    run_imsubtract_all(config_file, workers=2, max_imgs=2, display="/dev/null")

    # Check for outputs:
    expected_files = [f"{tmp_imsub}/r1_00013912_17_subI.fits", f"{tmp_imsub}/r1_00000670_12_subI.fits"]
    for fname in expected_files:
        assert os.path.isfile(fname), f"Expected output file {fname} not found."

    # Check that the output files have the expected properties
    for fname in expected_files:
        m = re.search(r"r1_(\d{8})_(\d{2})_subI\.fits", fname)
        obsid = int(m.group(1))
        scaid = int(m.group(2))
        print("Testing file:", fname, "with obsid:", obsid, "and scaid:", scaid)

        # Fetch the files
        with fits.open(fname) as f:
            image_subtracted = f[0].data[1, :, :]
        with fits.open(f"{tmp_imsub}/r1_{obsid:08d}_{scaid:02d}.fits.gz") as f:
            image_original = f[0].data[1, :, :]

        diff = image_subtracted - image_original

        # Assert some properties of the output:
        assert np.count_nonzero(diff) > 0, f"Output file {fname} is identical to the original image."
        assert np.sum(image_subtracted) < np.sum(
            image_original
        ), f"Output file {fname} has a larger sum than the original image."
        assert np.mean(diff) < 0, "Mean diff should be less than zero."

        # Compare diff cutout with the expected cutout
        cutout_url = f"{IMSUBTRACT_INPUT_PATH}/{obsid:08d}_{scaid:02d}_diff_cutout.fits"
        cutout_local = os.path.join(tmp_imsub, f"{obsid:08d}_{scaid:02d}_diff_cutout.fits")
        urllib.request.urlretrieve(cutout_url, cutout_local)
        with fits.open(cutout_local) as f:
            expected_diff_cutout = f[0].data
            ymin, ymax = f[0].header["YSTART"], f[0].header["YSTOP"]
            xmin, xmax = f[0].header["XSTART"], f[0].header["XSTOP"]

        diff_cutout = diff[ymin:ymax, xmin:xmax]
        assert np.allclose(
            diff_cutout, expected_diff_cutout, atol=1e-6
        ), f"Diff cutout for {fname} does not match expected values."

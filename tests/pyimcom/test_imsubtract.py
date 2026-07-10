import os
import re
import shutil
import subprocess
import urllib.request

import numpy as np
import pytest
from astropy.io import fits
from pyimcom.config import Config
from pyimcom.splitpsf.imsubtract import fftconvolve_multi, get_wcs, pltshow, reinterp, run_imsubtract
from pyimcom.splitpsf.imsubtract_wrapper import run_imsubtract_all, run_imsubtract_single
from pyimcom.splitpsf.splitpsf import SplitPSF, split_psf_to_fits
from pyimcom.splitpsf.splitpsf_wrapper import split_psf_single
from scipy.signal import fftconvolve

PSF_FILE = "https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/psf_test.fits"
IMSUBTRACT_CONFIG = "https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/imsubtract/test_imsubtract_config.json"
IMSUBTRACT_INPUT_PATH = "https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/imsubtract"


def test_reinterp():
    """Test interpolation function."""

    n = 40
    u = 2
    v = 5
    s_ = np.linspace(0, 2.0 * np.pi * (1 - 1 / n), n)
    s2_ = np.linspace(2.0 * np.pi * 1.5 / n, 2.0 * np.pi * (1 - 2.5 / n), n // 2 - 1)
    x, y = np.meshgrid(s_, s_)
    x2, y2 = np.meshgrid(s2_, s2_)
    arr = np.cos(u * x + v * y)
    arr2 = reinterp(arr)
    arr2_alt = np.sum(arr[1:-1, 1:-1].reshape((n // 2 - 1, 2, n // 2 - 1, 2)), axis=(-3, -1))
    target = 4 * np.cos(u * x2 + v * y2)
    er2_alt = np.amax(np.abs(arr2_alt - target))
    er2 = np.amax(np.abs(arr2 - target))
    assert er2 < 0.04
    assert er2 < 0.2 * er2_alt


def test_altwcs(tmp_path):
    """Test reading from SCIWCS."""

    fn = str(tmp_path) + "/altwcs.fits"

    # Make the image (including WCS header)
    im = fits.ImageHDU(np.zeros((4088, 4088), dtype=np.int16), name="SCIWCS")
    im.header["CRPIX1"] = 2030
    im.header["CDELT1"] = -0.0001
    im.header["CTYPE1"] = "RA---TAN"
    im.header["CRVAL1"] = 32.0
    im.header["CUNIT1"] = "deg"
    im.header["CRPIX2"] = 2030
    im.header["CDELT2"] = 0.0001
    im.header["CTYPE2"] = "DEC--TAN"
    im.header["CRVAL2"] = 15.0
    im.header["CUNIT2"] = "deg"
    im.header["EQUINOX"] = 2000.0
    fits.HDUList([fits.PrimaryHDU(), im]).writeto(fn)

    # get the WCS and clear the old file
    mywcs = get_wcs(fn)
    os.remove(fn)

    # now tests on the WCS we got
    out = mywcs.all_pix2world(np.array([[2029, 1979], [2029, 2029], [2029, 2079]]), 0)
    assert np.allclose(out, np.array([[32.0, 14.995], [32.0, 15.0], [32.0, 15.005]]))


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

    # test defaulting to fftconvolve
    x1 = fftconvolve(arr1, arr2, mode="same")
    x2 = np.zeros_like(x1)
    fftconvolve_multi(arr1, arr2, x2, mode="same")
    assert np.allclose(x1, x2)

    # another test defaulting to fftconvolve
    x1 = fftconvolve(arr1, arr2, mode="valid")
    x2 = np.zeros_like(x1)
    fftconvolve_multi(arr1, arr2, x2, mode="valid", nb=24)
    assert np.allclose(x1, x2)


def _run_imsubtract_all(tmp_path, config_file, test2x2=False):
    """
    Test the run_imsubtract_all function.
    This test runs the imsubtract pipeline on a small set of images specified in the config file,
    and checks that the output files are created and have the expected properties.
    """

    tmp_dir = str(tmp_path)
    tmp_imsub = tmp_dir + "/temp_imsubtract"
    tmp_mmap = tmp_dir + "/temp_mmap"
    tmp_figs = tmp_dir + "/temp_figs"
    # make temp_imsub directory
    os.makedirs(tmp_imsub, exist_ok=True)
    os.makedirs(tmp_imsub + "/blocks", exist_ok=True)
    os.makedirs(tmp_mmap, exist_ok=True)
    os.makedirs(tmp_figs, exist_ok=True)

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
        cf_local = os.path.join(tmp_imsub + "/blocks", filename)
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
        tmp_imsub + "/blocks/im3x2-H1",
    )
    with open(config_file, "w") as f:
        f.write(cfg_text)

    # Diagnostic output
    print(cfg_text)
    print(os.walk(tmp_imsub))

    # single run (this ensures that the codecov tracking works since it doesn't follow subprocesses)
    run_imsubtract_single(
        Config(config_file),
        17,
        13912,
        tmp_imsub,
        "r1_00013912_17.fits",
        display="/dev/null",
        max_layers=1,
        mmap=tmp_mmap,
        bin2x2=test2x2,
    )
    with fits.open(f"{tmp_imsub}/r1_00013912_17_subI.fits") as f:
        single_run = np.copy(f[0].data[0, :, :])
    # alt single run with wcs_shortcut turned off
    run_imsubtract_single(
        Config(config_file),
        17,
        13912,
        tmp_imsub,
        "r1_00013912_17.fits",
        display="/dev/null",
        wcs_shortcut=False,
        max_layers=1,
        mmap=tmp_mmap,
        bin2x2=test2x2,
    )
    with fits.open(f"{tmp_imsub}/r1_00013912_17_subI.fits") as f:
        single_run_alt = np.copy(f[0].data[0, :, :])
    p99_single = np.percentile(np.abs(single_run), 99)
    p99_diff = np.percentile(np.abs(single_run - single_run_alt), 99)
    assert p99_diff < 0.01 * p99_single  # require the 2 versions to have only a small difference

    # full multi run
    # I set the number of workers to 1 (which is kind of silly) to stay within the footprint of
    # the free GitHub runner during tests.
    run_imsubtract_all(config_file, workers=1, max_imgs=2, display="/dev/null", mmap=tmp_mmap, bin2x2=test2x2)

    # Check for outputs:
    expected_files = [f"{tmp_imsub}/r1_00013912_17_subI.fits", f"{tmp_imsub}/r1_00000670_12_subI.fits"]
    for fname in expected_files:
        assert os.path.isfile(fname), f"Expected output file {fname} not found."

    # Check that the output files have the expected properties
    ik = 0
    for fname in expected_files:
        m = re.search(r"r1_(\d{8})_(\d{2})_subI\.fits", fname)
        obsid = int(m.group(1))
        scaid = int(m.group(2))
        print("Testing file:", fname, "with obsid:", obsid, "and scaid:", scaid)

        # Fetch the files
        with fits.open(fname) as f:
            image_subtracted = f[0].data[1, :, :]
        with fits.open(f"{tmp_imsub}/r1_{obsid:08d}_{scaid:02d}.fits") as f:
            image_original = f[0].data[1, :, :]

        diff = image_subtracted - image_original

        # Assert some properties of the output:
        assert np.count_nonzero(diff) > 0, f"Output file {fname} is identical to the original image."
        assert np.sum(image_subtracted) < np.sum(
            image_original
        ), f"Output file {fname} has a larger sum than the original image."
        assert np.mean(diff) < 0, "Mean diff should be less than zero."

        # Compare diff cutout with the expected cutout
        cutout_url = f"{IMSUBTRACT_INPUT_PATH}/{obsid:d}_{scaid:d}_diff_cutout.fits"
        cutout_local = os.path.join(tmp_imsub, f"{obsid:d}_{scaid:d}_diff_cutout.fits")
        urllib.request.urlretrieve(cutout_url, cutout_local)
        with fits.open(cutout_local) as f:
            expected_diff_cutout = f[0].data
            ymin, ymax = f[0].header["YSTART"], f[0].header["YSTOP"]
            xmin, xmax = f[0].header["XSTART"], f[0].header["XSTOP"]
            print(f[0].header)

        diff_cutout = diff[ymin:ymax, xmin:xmax]
        atol = np.array([1.1e-3, 2.0e-2])[ik] if test2x2 else 1.0e-6
        assert np.allclose(
            diff_cutout, expected_diff_cutout, atol=atol
        ), f"Diff cutout for {fname} does not match expected values."
        if test2x2:
            print(np.amax(np.abs(diff_cutout[8:-8, 8:-8] - expected_diff_cutout[8:-8, 8:-8])))
            assert np.allclose(
                diff_cutout[8:-8, 8:-8], expected_diff_cutout[8:-8, 8:-8], atol=np.array([9.0e-4, 1.2e-2])[ik]
            ), f"Diff cutout for {fname} does not match expected values."
        ik += 1  # noqa: SIM113

    # compare "single" to "all" case
    with fits.open(f"{tmp_imsub}/r1_00013912_17_subI.fits") as f:
        assert np.allclose(f[0].data[0, :, :], single_run, rtol=1.0e-6, atol=1.0e-6)
        del single_run
        multi_run = np.copy(f[0].data[0, :, :])

    # original wrapper
    os.remove(str(tmp_imsub) + "/r1_00013912_17_subI.fits")
    run_imsubtract(config_file, display=f"{tmp_figs}/win", scanum=17, max_layers=1, bin2x2=test2x2)
    with fits.open(f"{tmp_imsub}/r1_00013912_17_subI.fits") as f:
        assert np.allclose(f[0].data[0, :, :], multi_run, rtol=1.0e-6, atol=1.0e-6)
    fname = f"{tmp_figs}/win_13912_17_35_02.png"
    assert os.path.isfile(fname), f"Expected output file {fname} not found."

    # remove files from this test to save space
    for fl in [
        "r1_00000670_12.fits",
        "r1_00013912_17.fits",
        "r1_00000670_12_subI.fits",
        "r1_00013912_17_subI.fits",
    ]:
        os.remove(str(tmp_imsub) + "/" + fl)
    shutil.rmtree(tmp_imsub)
    shutil.rmtree(tmp_mmap)
    shutil.rmtree(tmp_figs)


def test_run_imsubtract_all2(tmp_path):
    """Test with 2x2 bin version."""
    _run_imsubtract_all(tmp_path, IMSUBTRACT_CONFIG, test2x2=True)


def test_run_imsubtract_all(tmp_path):
    """Basic version of test."""
    _run_imsubtract_all(tmp_path, IMSUBTRACT_CONFIG)


def test_staticmethods():
    """Unit tests for staticmethods in splitpsf.py."""

    # test for Truncate_2D_integratedBlackman
    # 11x11 block, transition width of 3 on each side
    arr = SplitPSF.Truncate_2D_integratedBlackman(11, 3)
    u = 0.0770055
    assert np.allclose(
        arr[6, :],
        np.array([u, 0.5, 1 - u, 1, 1, 1, 1, 1, 1 - u, 0.5, u]),
        rtol=0,
        atol=1.0e-6,
    )
    assert np.allclose(arr[6, :], arr[:, 6], rtol=0, atol=1.0e-6)
    for x in [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]:
        assert np.abs(x - u**2) < 1.0e-6


def test_pltshow():
    """Test the other options for pltshow."""

    class _Plt:
        """Test class to count calls."""

        def __init__(self):
            self.calls = 0

        def show(self):
            """Mock show method."""
            self.calls += 1

    plot = _Plt()
    assert plot.calls == 0
    pltshow(plot, None)
    assert plot.calls == 1
    pltshow(plot, "/dev/null")
    assert plot.calls == 1


def test_imsubtract_exceptions(tmp_path):
    """Test that the correct exceptions are raised in imsubtract."""

    tmp_dir = str(tmp_path)
    tmp_imsub = tmp_dir + "/temp_imsubtract"
    tmp_mmap = tmp_dir + "/temp_mmap"
    tmp_figs = tmp_dir + "/temp_figs"
    # make temp_imsub directory
    os.makedirs(tmp_imsub, exist_ok=True)
    os.makedirs(tmp_imsub + "/blocks", exist_ok=True)
    os.makedirs(tmp_mmap, exist_ok=True)
    os.makedirs(tmp_figs, exist_ok=True)

    urllib.request.urlretrieve(IMSUBTRACT_CONFIG, tmp_imsub + "/test_imsubtract_config.json")
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

    # psf files
    tmp_psf = tmp_imsub + "/r1.psf"
    os.makedirs(tmp_psf, exist_ok=True)
    for filename in ["psf_13912.fits", "psf_670.fits"]:
        cf_url = IMSUBTRACT_INPUT_PATH + "/cache/r1.psf/" + filename
        cf_local = os.path.join(tmp_psf, filename)
        urllib.request.urlretrieve(cf_url, cf_local)

        # now mess with these files to trigger exceptions
        with fits.open(cf_local, mode="update") as f:
            print(f[0].header["OVSAMP"])
            f[0].header["OVSAMP"] = 7

    with open(config_file, "r") as f:
        cfg_text = f.read()
    cfg_text = cfg_text.replace("$TMPDIR", tmp_dir)
    cfg_text = cfg_text.replace("$CACHE", tmp_imsub + "/r1")
    cfg_text = cfg_text.replace(
        "https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/imsubtract/blocks/im3x2-H1",
        tmp_imsub + "/blocks/im3x2-H1",
    )
    with open(config_file, "w") as f:
        f.write(cfg_text)

    # single run (this ensures that the codecov tracking works since it doesn't follow subprocesses)
    with pytest.raises(ValueError, match=r"oversamp, oversamp=7"):
        run_imsubtract_single(
            Config(config_file),
            17,
            13912,
            tmp_imsub,
            "r1_00013912_17.fits",
            display="/dev/null",
            max_layers=1,
            mmap=tmp_mmap,
            bin2x2=True,
        )

    # this will raise a different exception
    for filename in ["psf_13912.fits", "psf_670.fits"]:
        cf_local = os.path.join(tmp_psf, filename)
        with fits.open(cf_local, mode="update") as f:
            print(f[0].header["OVSAMP"])
            f[0].header["OVSAMP"] = 3
    with pytest.raises(ValueError, match=r"oversamp=3 is odd, not consistent with bin2x2"):
        run_imsubtract_single(
            Config(config_file),
            17,
            13912,
            tmp_imsub,
            "r1_00013912_17.fits",
            display="/dev/null",
            max_layers=1,
            mmap=tmp_mmap,
            bin2x2=True,
        )

    # remove files from this test to save space
    shutil.rmtree(tmp_imsub)
    shutil.rmtree(tmp_mmap)
    shutil.rmtree(tmp_figs)

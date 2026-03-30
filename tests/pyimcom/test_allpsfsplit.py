"""Test function for split_psf_all."""

import os
import re
import urllib.request

import numpy as np
from astropy.io import fits
from pyimcom.config import Config
from pyimcom.splitpsf.splitpsf_wrapper import split_psf_all

CFGSETUP = """
{
    "OBSFILE": "$OBS/Roman_WAS_obseq_11_1_23.fits",
    "INDATA": [
        "/__not_needed",
        "L2_2506"
    ],
    "FILTER": 2,
    "INPSF": [
        "$INPSF",
        "L2_2506",
        6
    ],
    "EXTRAINPUT": [
        "whitenoise10"
    ],
    "CTR": [
        9.55,
        -44.1
    ],
    "BLOCK": 36,
    "OUTSIZE": [
        80,
        32,
        0.0390625
    ],
    "FADE": 0,
    "PAD": 2,
    "PADSIDES": "all",
    "STOP": 0,
    "OUTMAPS": "USTN",
    "OUT": "/__not_needed",
    "TEMPFILE": "$TMPDIR",
    "INLAYERCACHE": "$CACHE",
    "NOUT": 1,
    "OUTPSF": "GAUSSIAN",
    "EXTRASMOOTH": 0.8493218002880191,
    "NPIXPSF": 48,
    "PSFSPLIT": [
        5.25,
        8.75,
        0.01
    ],
    "PSFCIRC": false,
    "PSFNORM": false,
    "AMPPEN": [
        0.0,
        0.0
    ],
    "FLATPEN": 0.0,
    "INPAD": 0.6,
    "LAKERNEL": "Iterative",
    "ITERRTOL": 0.0015,
    "ITERMAX": 30,
    "KAPPAC": [
        0.0
    ],
    "UCMIN": 1e-06,
    "SMAX": 0.5
}
"""


def test_allpsfsplit(tmp_path):
    """Tests psf-splitting multiple files."""

    # Set up the configuration file for our environment
    os.makedirs(str(tmp_path / "cache"), exist_ok=True)
    os.makedirs(str(tmp_path / "psf"), exist_ok=True)
    cfg = re.sub(r"\$TMPDIR", str(tmp_path) + "/", CFGSETUP)
    cfg = re.sub(r"\$CACHE", str(tmp_path) + "/cache/testfiles", cfg)
    cfg = re.sub(r"\$INPSF", str(tmp_path) + "/psf", cfg)
    cfg = re.sub(r"\$OBS", str(tmp_path), cfg)
    with open(tmp_path / "cfg_psftest.json", "w") as f:
        f.write(cfg)

    # Download files
    for fl in [670, 13912]:
        urllib.request.urlretrieve(
            f"https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/allpsf/testpsf_{fl:d}.fits",
            str(tmp_path) + f"/psf/psf_polyfit_{fl:d}.fits",
        )
    urllib.request.urlretrieve(
        "https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/Roman_WAS_obseq_11_1_23.fits",
        str(tmp_path) + "/Roman_WAS_obseq_11_1_23.fits",
    )

    # run split PSF
    split_psf_all(Config(cfg), 2)  # 2 workers

    # check we have everything
    for fl in [13912, 670]:
        outfile = str(tmp_path) + f"/cache/testfiles.psf/psf_{fl:d}.fits"
        assert os.path.isfile(outfile)

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
            assert f[i].header["NAXIS1"] == 432
            assert f[i].header["NAXIS2"] == 432
            assert f[i].header["NAXIS3"] == 4

            assert f[i + 18].header["SCA"] == i
            assert f[i + 18].header["NAXIS1"] == 112
            assert f[i + 18].header["NAXIS2"] == 112
            assert f[i + 18].header["NAXIS3"] == 4

            assert f[i + 36].header["SCA"] == i
            assert f[i + 36].header["NAXIS1"] == 432
            assert f[i + 36].header["NAXIS2"] == 432
            assert f[i + 36].header["NAXIS3"] == 4

            assert 0.85 < np.sum(f[i + 18].data[0, :, :]) < 0.91
            e = np.sum(f[i].data[0, :, :]) - np.sum(f[i + 18].data[0, :, :]) - np.sum(f[i + 36].data[0, :, :])
            assert -0.0009 < e < -0.0004

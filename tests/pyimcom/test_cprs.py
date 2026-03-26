"""Tests for parallel compression utilities."""

import re
import urllib.request

import numpy as np
from astropy.io import fits
from pyimcom.compress.compressutils import CompressedOutput, ReadFile
from pyimcom.compress.compressutils_wrapper import compress_all_blocks, compress_one_block

EXAMPLE_FILE = (
    "https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/compressiontest_F_02_11.fits"
)

# The methods to compress the indicated layers.
LAYER_PARS_DIC = {
    "gsstar": {"VMIN": -1.0 / 64.0, "VMAX": 7.0 / 64.0, "BITKEEP": 20, "DIFF": True, "SOFTBIAS": -1},
    "nstar": {"VMIN": -1500.0, "VMAX": 10500.0, "BITKEEP": 20, "DIFF": True, "SOFTBIAS": -1},
    "gsext": {"VMIN": -1.0 / 64.0, "VMAX": 7.0 / 64.0, "BITKEEP": 20, "DIFF": True, "SOFTBIAS": -1},
    "whitenoise": {"VMIN": -8.0, "VMAX": 8.0, "BITKEEP": 14, "DIFF": True, "SOFTBIAS": -1},
    "1fnoise": {"VMIN": -32.0, "VMAX": 32.0, "BITKEEP": 14, "DIFF": True, "SOFTBIAS": -1},
}


def runcprs(tmp_path, allfiles=False):
    """
    Test compression of a file.

    Parameters
    ----------
    tmp_path : str
        Directory in which to run the test.
    allfiles : boolean
        Run all files?

    Returns
    -------
    None

    """

    # Download the test file to `floc`
    tmp_dir = str(tmp_path)
    floc = tmp_dir + "/test_F_02_11.fits"
    urllib.request.urlretrieve(EXAMPLE_FILE, floc)

    # Get the configuration and overwrite the output stem
    with CompressedOutput(floc) as f:
        cfg = f.cfg
    print("old stem -->", cfg.outstem)
    cfg.outstem = floc[:-11]
    print("new stem -->", cfg.outstem)

    # get block index
    m = re.match(r"_(\d+)_(\d+)", floc[-11:])
    ibx = int(m.group(1))
    iby = int(m.group(2))
    print("blocks", ibx, iby)

    if allfiles:
        compress_all_blocks(cfg, LAYER_PARS_DIC, 4)  # 4 workers

    else:
        # and now compress
        fcprs = compress_one_block(cfg, LAYER_PARS_DIC, ibx, iby)
        assert fcprs == floc[:-5] + ".cpr.fits.gz"

        # the next one doesn't exist yet
        fcprs = compress_one_block(cfg, LAYER_PARS_DIC, ibx, iby + 1)
        assert fcprs is None

    # check for the right number of HDUs
    fout = floc[:-5] + ".cpr.fits.gz"
    with fits.open(fout) as f:
        assert len(f) == 20

    # new check the decompression
    with fits.open(floc) as fi, ReadFile(fout) as fo:
        diff = fi[0].data[0, :, :, :] - fo[0].data[0, :, :, :]
    maxdiff = np.amax(np.abs(diff), axis=(1, 2))
    assert np.all(maxdiff <= np.array([1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.2e-7, 0.011, 1.2e-7, 0.004, 0.001]))


def test_singlecprs(tmp_path):
    """
    Test compression of a file.

    Parameters
    ----------
    tmp_path : str
        Directory in which to run the test.

    Returns
    -------
    None

    """

    runcprs(tmp_path)


def test_allcprs(tmp_path):
    """
    Test compression of all files.

    Parameters
    ----------
    tmp_path : str
        Directory in which to run the test.

    Returns
    -------
    None

    """

    runcprs(tmp_path, allfiles=True)

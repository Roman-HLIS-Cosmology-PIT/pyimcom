"""Test function for the main driver in splitpsf."""

import os
from urllib.request import urlretrieve

import numpy as np
from astropy.io import fits
from astropy.table import Table
from pyimcom.splitpsf import splitpsf

PSF_FILE = "https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/psf_test.fits"

myCfg_format = """
{
    "OBSFILE": "$TMPDIR/obs.fits",
    "INDATA": [
        "$TMPDIR/in",
        "L2_2506"
    ],
    "CTR": [
        60.0504,
        -3.8
    ],
    "LONPOLE": 240.0,
    "OUTSIZE": [
        4,
        25,
        0.04
    ],
    "BLOCK": 2,
    "FILTER": 1,
    "LAKERNEL": "Cholesky",
    "KAPPAC": [
         5e-4
    ],
    "INPSF": [
        "$TMPDIR/psf",
        "L2_2506",
        6
    ],
    "EXTRAINPUT": [
        "gsstar14",
        "gsext14,seed=100,shear=-.01:0.017320508075688773",
        "cstar14",
        "nstar14,2e5,100,256",
        "whitenoise1",
        "1fnoise2",
        "gsext14,n=0.5,hlr=0.1,shape=0.2:0.1,shear=0.05:-0.12",
        "gstrstar14"
    ],
    "PSFSPLIT": [
        4.0,
        9.0,
        0.01
    ],
    "PADSIDES": "all",
    "OUTMAPS": "USTKN",
    "OUT": "$TMPDIR/out/testout_F",
    "INPAD": 0.8,
    "NPIXPSF": 42,
    "FADE": 1,
    "PAD": 0,
    "NOUT": 1,
    "OUTPSF": "GAUSSIAN",
    "EXTRASMOOTH": 0.9265328730414752,
    "INLAYERCACHE": "$TMPDIR/cache/in"
}
"""


def splitpsf_driver(tmp_path, splitzeta):
    """Main test function."""

    # first, get the configuration file.
    with open(tmp_path / "cfg.txt", "w") as f:
        f.write(myCfg_format.replace("$TMPDIR", str(tmp_path)))

    # make and clear the cache directory
    cachedir = tmp_path / "cache"
    cachedir.mkdir(parents=True, exist_ok=True)
    files = os.listdir(cachedir)
    for file in files:
        if file[-5:] == ".fits":
            print("removing", os.path.join(cachedir, file))
            os.remove(os.path.join(cachedir, file))

    # and the PSF directory
    psfdir = tmp_path / "psf"
    psfdir.mkdir(parents=True, exist_ok=True)
    files = os.listdir(psfdir)
    for file in files:
        if file[-5:] == ".fits":
            print("removing", os.path.join(psfdir, file))
            os.remove(os.path.join(psfdir, file))

    # and now download the PSF
    urlretrieve(PSF_FILE, str(tmp_path) + "/psf/psf_polyfit_34.fits")
    urlretrieve(PSF_FILE, str(tmp_path) + "/psf/psf_polyfit_60.fits")

    # now make the observation file
    obs = []
    for j in range(80):
        jj = 10 + j % 16
        date = 61541 + 0.01 * j
        exptime = 139.8
        ra = 60.0 + 0.01 * jj + 0.2 * (j / 16)
        dec = -4.0 + 0.1 * jj
        pa = 20.0
        filter = "F184" if j > 12 else "H158"
        obs.append((date, exptime, ra, dec, pa, filter))
    data = np.rec.array(
        obs, formats="float64,float64,float64,float64,float64,S4", names="date,exptime,ra,dec,pa,filter"
    )
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(data=Table(data))]).writeto(
        tmp_path / "obs.fits", overwrite=True
    )

    # run PSF splitting
    splitpsf.main(str(tmp_path / "cfg.txt"))

    # check that we have the files we should have
    assert os.path.exists(str(tmp_path) + "/cache/in.psf/psf_34.fits")
    assert os.path.exists(str(tmp_path) + "/cache/in.psf/psf_60.fits")

    # study contents of the files
    with fits.open(str(tmp_path) + "/cache/in.psf/psf_60.fits") as f:
        print(f.info())
        print(f[0].header)

        for sca in range(1, 19):
            s_s = np.sum(f[sca].data)
            m_s = np.amax(f[sca].data)
            assert 0.95 < s_s < 1.0
            assert 0.007 < m_s < 0.009
            assert np.shape(f[sca].data) == (4, 384, 384)

            s_s = np.sum(f[sca + 18].data[0, :, :])
            m_s = np.amax(f[sca + 18].data[0, :, :])
            assert 0.85 < s_s < 0.9
            assert 0.005 < m_s < 0.007
            assert np.shape(f[sca + 18].data) == (4, 120, 120)

            s_s = np.sum(f[sca + 36].data)
            m_s = np.amax(np.abs(f[sca + 36].data))
            assert 0.09 < s_s < 0.11
            assert 3e-5 < m_s < 8e-5
            print(sca, s_s, m_s)
            assert np.shape(f[sca + 36].data) == (4, 384, 384)


def test_splitpsf_driver_withsplitzeta(tmp_path):
    """Test with splitzeta."""
    splitpsf_driver(tmp_path, True)


def test_splitpsf_driver_nosplitzeta(tmp_path):
    """Test with splitzeta."""
    splitpsf_driver(tmp_path, False)

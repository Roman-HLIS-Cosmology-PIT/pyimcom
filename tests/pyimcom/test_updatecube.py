"""Test for the updatecube function."""

import json
import os

import numpy as np
from astropy.io import fits
from pyimcom.splitpsf.update_cube import update

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


def test_update(tmp_path):
    """Tests update_cube.py; makes dummy files."""

    tmp_path = str(tmp_path)

    # make cache directory
    os.makedirs(f"{tmp_path}/cache", exist_ok=True)

    # first, get the configuration file.
    with open(tmp_path + "/cfg.txt", "w") as f:
        f.write(myCfg_format.replace("$TMPDIR", tmp_path))
    with open(tmp_path + "/cache/in_oldcfg.json", "w") as f:
        f.write(myCfg_format.replace("$TMPDIR", tmp_path))

    # which files we want -- this is arbitrary since `update`
    # doesn't know where these are, it just moves them around, so make dummies
    filelist = [(1400, 12), (191403, 7), (12345678, 18)]
    for obsidsca in filelist:
        im = np.zeros((4, 4), dtype=np.int32)
        im[0, 0] = obsid = obsidsca[0]
        im[0, 1] = sca = obsidsca[1]
        fits.PrimaryHDU(im).writeto(f"{tmp_path}/cache/in_{obsid:08d}_{sca:02d}.fits", overwrite=True)
        im2 = np.copy(im)
        im2[-1, -1] = 256  # label the new file with this corner of the array
        fits.PrimaryHDU(im2).writeto(f"{tmp_path}/cache/in_{obsid:08d}_{sca:02d}_subI.fits", overwrite=True)

    # now run the update
    update(tmp_path + "/cfg.txt", proceed=True)

    # check whether we have the right files
    for obsidsca in filelist:
        obsid = obsidsca[0]
        sca = obsidsca[1]
        with fits.open(f"{tmp_path}/cache/in_{obsid:08d}_{sca:02d}.fits") as f:
            assert f[0].data[0, 0] == obsid
            assert f[0].data[0, 1] == sca
            assert f[0].data[-1, -1] == 256
        with fits.open(f"{tmp_path}/cache/in_{obsid:08d}_{sca:02d}_00iter.fits") as f:
            assert f[0].data[0, 0] == obsid
            assert f[0].data[0, 1] == sca
            assert f[0].data[-1, -1] == 0

    assert np.loadtxt(f"{tmp_path}/cache/in_iter.txt").ravel()[0] == 1
    with open(f"{tmp_path}/cache/in_oldcfg.json", "r") as f:
        cfg = json.load(f)
    assert cfg["CONFIG0"]["INPSF"][-1] == 6

"""Exception testing."""

import pytest
from pyimcom.splitpsf import splitpsf

myCfg2_format = """
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
    "OUTPSF": "AIRYOBSC",
    "EXTRASMOOTH": 0.9265328730414752,
    "$KEY": "$TMPDIR/cache/in"
}
"""


def test_mainexcept(tmp_path):
    """Test for the right exceptions getting raised when we give bad values."""

    # first, get the configuration file with INLAYERCACHE
    with open(tmp_path / "cfg.txt", "w") as f:
        f.write(myCfg2_format.replace("$TMPDIR", str(tmp_path)).replace("$KEY", "INLAYERCACHE"))

    with pytest.raises(ValueError, match=r"Gaussian"):
        splitpsf.main(str(tmp_path / "cfg.txt"))

    # now, get the configuration file with no INLAYERCACHE
    with open(tmp_path / "cfg2.txt", "w") as f:
        f.write(myCfg2_format.replace("$TMPDIR", str(tmp_path)))

    with pytest.raises(KeyError, match=r"INLAYERCACHE"):
        splitpsf.main(str(tmp_path / "cfg2.txt"))

    # now, get a nonexistent path
    with open(tmp_path / "cfg3.txt", "w") as f:
        f.write(
            myCfg2_format.replace("$TMPDIR", str(tmp_path) + "/doesntexist/doesntexist/doesntexist")
            .replace("$KEY", "INLAYERCACHE")
            .replace("AIRYOBSC", "GAUSSIAN")
        )

    with pytest.raises(OSError):
        splitpsf.main(str(tmp_path / "cfg3.txt"))

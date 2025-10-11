import io
import sys

from pyimcom.config import Config


def _fpcheck(a, b, tol):
    """Checks that |a-b|<tol."""
    return a > b - tol and a < b + tol


def test_interface(tmp_path):
    """
    Test for the command line interface.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Path for temporary files.

    Returns
    -------
    None

    Notes
    -----
    This test will need to be updated if you add to the configuration module.

    """

    tdir = str(tmp_path)  # temporary directory

    # Build the script.

    # SECTION I
    # OBSFILE
    script = tdir + "/obsfile.fits\n"  # doesn't need to exist for this case -- just checking the formatting
    # INDATA
    script += tdir + "/L2 L2_2506\n"
    # FILTER
    script += "1\n"
    # INPSF
    script += tdir + "/psf6 L2_2506 6\n"
    # PSFSPLIT
    script += "5.5 9.5 0.01\n"

    # SECTION II
    # PMASK
    script += "\n"
    # CMASK
    script += "\n"
    # EXTRAINPUT
    script += (
        "truth,0.004906087669824225 noise,Rz4S2C1 noise,Rz4S2C2 gsstar14 nstar14,2e5,86,3 "
        "1fnoise9 whitenoise10\n"
    )
    # LABNOISETHRESHOLD
    script += "\n"

    # SECTION III
    # CTR
    script += "9.55 -44.1\n"
    # LONPOLE
    script += "240.0\n"
    # BLOCK
    script += "12\n"
    # OUTSIZE
    script += "72 36 0.046296296296296294\n"

    # SECTION IV
    # FADE
    script += "1\n"
    # PAD
    script += "1\n"
    # PADSIDES
    script += "all\n"
    # STOP
    script += "\n"

    # SECTION V
    # OUTMAPS
    script += "USTN\n"
    # OUT
    script += tdir + "/test26_F\n"
    # TEMPFILE
    script += "\n"
    # INLAYERCACHE
    script += tdir + "/cache/in_F\n"

    # SECTION VI
    # NOUT
    script += "1\n"
    # OUTPSF
    script += "GAUSSIAN\n"
    # EXTRASMOOTH
    script += "0.9265328730414752\n"

    # SECTION VII
    # NPIXPSF
    script += "42\n"
    # PSFCIRC
    script += "False\n"
    # PSFNORM
    script += "False\n"
    # AMPPPEN
    script += "\n"
    # FLATPEN
    script += "\n"
    # INPAD
    script += "0.950\n"

    # SECTION VIII
    # LAKERNEL
    script += "Cholesky\n"
    # KAPPAC
    script += "5e-4\n"
    # UCMIN
    script += "\n"
    # SMAX
    script += "\n"

    # SECTION IX
    # TILESCHM
    script += "\n"
    # RERUN
    script += "\n"
    # MOSAIC
    script += "137\n"

    print(script)
    # assert 1 > 2 # <-- This was to check that the test script was actually being run. Leave commented.

    # Now read this into a configuration as if it were from the input terminal
    original_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO(script)
        cfg = Config(None)
    finally:
        sys.stdin = original_stdin

    for k in dir(cfg):
        x = str(k)
        if x[0] != "_" and hasattr(cfg, x):
            print(x, getattr(cfg, x))

    # and we can now see if the configuration was built properly
    assert cfg.Nside == 2592
    assert cfg.NsideP == 2664
    assert cfg.amp_penalty[0] == 0.0
    assert cfg.amp_penalty[1] == 0.0
    assert cfg.cfg_file is None
    assert cfg.cr_mask_rate == 0.0
    assert _fpcheck(cfg.dec, -44.1, 1e-8)
    assert _fpcheck(cfg.dtheta, 1.2860082304526749e-05, 1e-11)
    assert cfg.extrainput == [
        None,
        "truth,0.004906087669824225",
        "noise,Rz4S2C1",
        "noise,Rz4S2C2",
        "gsstar14",
        "nstar14,2e5,86,3",
        "1fnoise9",
        "whitenoise10",
    ]
    assert cfg.fade_kernel == 1
    assert cfg.informat == "L2_2506"
    assert cfg.inlayercache == tdir + "/cache/in_F"
    assert cfg.inpath == tdir + "/L2"
    assert cfg.inpsf_format == "L2_2506"
    assert cfg.inpsf_oversamp == 6
    assert cfg.inpsf_path == tdir + "/psf6"
    assert _fpcheck(cfg.instamp_pad, 4.6057299705405916e-06, 1e-14)
    assert _fpcheck(cfg.kappaC_arr[0], 5e-4, 1e-10)
    assert _fpcheck(cfg.labnoisethreshold, 3.0, 1e-6)
    assert cfg.mosaic == 137
    assert cfg.n1 == 72
    assert cfg.n1P == 74
    assert cfg.n2 == 36
    assert cfg.n2f == 38
    assert cfg.n_inframe == 8
    assert cfg.n_out == 1
    assert cfg.nblock == 12
    assert cfg.npixpsf == 42
    assert cfg.obsfile == tdir + "/obsfile.fits"
    assert cfg.outmaps == "USTN"
    assert cfg.outpsf == "GAUSSIAN"
    assert cfg.outstem == tdir + "/test26_F"
    assert cfg.pad_sides == "all"
    assert cfg.permanent_mask is None
    assert not cfg.psf_circ
    assert not cfg.psf_norm
    assert _fpcheck(cfg.psfsplit_epsilon, 0.01, 1e-8)
    assert _fpcheck(cfg.psfsplit_r1, 5.5, 2e-6)
    assert _fpcheck(cfg.psfsplit_r2, 9.5, 2e-6)
    assert _fpcheck(cfg.ra, 9.55, 1e-8)
    assert cfg.rerun == "Not_specified"
    assert _fpcheck(cfg.sigmamax, 0.5, 1e-6)
    assert _fpcheck(cfg.sigmatarget, 0.9265328730414752, 5e-7)
    assert cfg.stoptile == 0
    assert cfg.tempfile == ""
    assert cfg.tileschm == "Not_specified"
    assert _fpcheck(cfg.uctarget, 1e-6, 1e-12)
    assert cfg.use_filter == 1


# test_interface("out") # <-- comment out in production version

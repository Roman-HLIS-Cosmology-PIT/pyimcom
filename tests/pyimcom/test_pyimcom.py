"""
Small-scale test of PyIMCOM, designed for fast test of core functionality.

This does 2 blocks.
"""

import copy
import os
import pathlib

import asdf
import galsim
import gwcs
import numpy as np
import pytest
from astropy import coordinates as coord
from astropy import units as u
from astropy import wcs
from astropy.io import fits
from astropy.modeling import models
from astropy.table import Table
from furry_parakeet.pyimcom_croutines import gridD5512C
from gwcs import coordinate_frames as cf
from pyimcom.analysis import OutImage
from pyimcom.coadd import Block
from pyimcom.compress.compressutils import CompressedOutput, ReadFile
from pyimcom.config import Config
from pyimcom.diagnostics.layer_diagnostics import LayerReport
from pyimcom.diagnostics.mosaicimage import MosaicImage
from pyimcom.diagnostics.noise_diagnostics import NoiseReport
from pyimcom.diagnostics.report import ValidationReport
from pyimcom.diagnostics.stars import SimulatedStar
from pyimcom.psfutil import OutPSF
from pyimcom.truthcats import gen_truthcats_from_cfg
from pyimcom.wcsutil import _stand_alone_test
from scipy.signal import convolve

EXAMPLE_FILE = (
    "https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/compressiontest_F_02_11.fits"
)

# constants
degree = np.pi / 180.0
nside = 4088

# center position
cra = 60.0504 * degree
cdec = -3.8 * degree

# width of PSF in output pixels, area scaling
sig = 0.9265328730414752 * 0.11 / 0.04
sc = (0.04 / 0.11) ** 2

# star position
sra = 60.0508 * degree
sdec = -3.8005 * degree

# format for the configuration file.
# $TMPDIR will get replaced with the temporary directory.
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
    "OUTMAPS": "USTN",
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

# CD and CRPIX values for linear approximations to the SCA WCSs.
wcsdata = np.array(
    [
        [
            3.08219178082192e-05,
            1.22309197651664e-07,
            0,
            -3.02103718199609e-05,
            9.31141597190574e-10,
            -254.695456590193,
            819.255060728745,
        ],
        [
            3.04549902152642e-05,
            1.22309197651664e-07,
            -1.22309197651663e-07,
            -2.97211350293542e-05,
            9.05141916965698e-10,
            -302.076620500445,
            5721.07850461111,
        ],
        [
            3.02103718199609e-05,
            2.44618395303327e-07,
            -1.22309197651663e-07,
            -2.88649706457926e-05,
            8.71991576701989e-10,
            -340.491336421342,
            10368.6800480357,
        ],
        [
            3.05772994129159e-05,
            1.22309197651664e-07,
            -2.44618395303327e-07,
            -3.02103718199609e-05,
            9.23721665434799e-10,
            -4684.7680248753,
            -19.9937811750988,
        ],
        [
            3.04549902152642e-05,
            3.6692759295499e-07,
            -3.66927592954991e-07,
            -2.97211350293542e-05,
            9.05022240647057e-10,
            -4754.73767727858,
            4920.56054745611,
        ],
        [
            3.02103718199609e-05,
            4.89236790606654e-07,
            -3.66927592954991e-07,
            -2.91095890410959e-05,
            8.79231993979803e-10,
            -4894.76339878177,
            9448.63987477456,
        ],
        [
            3.04549902152642e-05,
            2.44618395303327e-07,
            -3.66927592954991e-07,
            -3.03326810176125e-05,
            9.23691746355138e-10,
            -9119.77676286723,
            -2073.79302302983,
        ],
        [
            3.02103718199609e-05,
            4.89236790606654e-07,
            -3.66927592954991e-07,
            -2.98434442270059e-05,
            9.0140203200815e-10,
            -9255.53159851301,
            2786.07620817844,
        ],
        [
            2.99657534246575e-05,
            6.11545988258318e-07,
            -6.11545988258311e-07,
            -2.92318982387476e-05,
            8.75581866261235e-10,
            -9476.57488467452,
            7313.76934905176,
        ],
        [
            3.06996086105675e-05,
            -1.22309197651664e-07,
            1.22309197651664e-07,
            -3.02103718199609e-05,
            9.27431631312686e-10,
            4156.44544809343,
            827.807471449771,
        ],
        [
            3.04549902152642e-05,
            -2.44618395303327e-07,
            1.22309197651664e-07,
            -2.95988258317025e-05,
            9.0140203200815e-10,
            4207.94795539033,
            5735.52044609665,
        ],
        [
            3.02103718199609e-05,
            -2.44618395303327e-07,
            1.22309197651664e-07,
            -2.88649706457926e-05,
            8.71991576701989e-10,
            4262.97958483445,
            10367.9787270544,
        ],
        [
            3.05772994129158e-05,
            -2.44618395303327e-07,
            2.44618395303327e-07,
            -3.00880626223092e-05,
            9.1995186139759e-10,
            8568.20762326005,
            -30.0470924938209,
        ],
        [
            3.03326810176125e-05,
            -3.6692759295499e-07,
            2.44618395303327e-07,
            -2.97211350293542e-05,
            9.0143195108781e-10,
            8671.99004281589,
            4891.17687278038,
        ],
        [
            3.02103718199609e-05,
            -3.66927592954991e-07,
            3.66927592954991e-07,
            -2.89872798434442e-05,
            8.75581866261235e-10,
            8754.5221937468,
            9476.99395181958,
        ],
        [
            3.03326810176125e-05,
            -2.44618395303327e-07,
            2.44618395303327e-07,
            -3.03326810176125e-05,
            9.2001169955691e-10,
            13087.5824390244,
            -2119.77756097561,
        ],
        [
            3.03326810176125e-05,
            -3.66927592954991e-07,
            4.89236790606654e-07,
            -2.99657534246575e-05,
            9.08762125604605e-10,
            13130.6172384276,
            2825.69171001514,
        ],
        [
            3.00880626223092e-05,
            -6.11545988258318e-07,
            4.89236790606654e-07,
            -2.92318982387476e-05,
            8.79231993979802e-10,
            13350.5118589853,
            7261.98346207507,
        ],
    ]
)


def pull_from_file(infile):
    """
    Utility to extract machine-readable data from the LaTeX as a dictionary so we can verify it.

    Parameters
    ----------
    infile : str or str-like
        LaTeX file to read from.

    Returns
    -------
    dict
        Dictionary of data blocks.

    """

    with open(infile, "r") as f:
        lines = f.readlines()
    exdata = {}
    is_read = False
    for line in lines:
        if line[:9] == "$$$START ":
            name = line.split()[1]
            is_read = True
            info = ""
            continue
        if line[:7] == "$$$END ":
            exdata[name] = info
            is_read = False
            continue
        if is_read:
            info += line + "\n"
    return exdata


def make_simple_wcs(ra, dec, pa, sca):
    """
    Makes a simple approximate WCS for an SCA.

    Parameters
    ----------
    ra : float
        RA of the WFI center in degrees.
    dec : float
        Dec of the WFI center in degrees.
    pa : float
        PA of the WFI center in degrees.
    sca : int
        SCA number, in 1 .. 18.

    Returns
    -------
    astropy.wcs.WCS
        Simple WCS approximation.

    """

    outwcs = wcs.WCS(naxis=2)
    outwcs.wcs.crpix = [wcsdata[sca - 1, -2], wcsdata[sca - 1, -1]]
    outwcs.wcs.cd = wcsdata[sca - 1, :4].reshape((2, 2))
    outwcs.wcs.ctype = ["RA---ARC", "DEC--ARC"]
    outwcs.wcs.crval = [ra, dec]
    outwcs.wcs.lonpole = pa - 180.0 if pa >= 180.0 else pa + 180.0

    return outwcs


@pytest.fixture
def setup(tmp_path):
    """
    Generates sample input files for a pyimcom run.

    Parameters
    ----------
    tmp_path : str
        Directory in which to run the test.

    Returns
    -------
    None

    """

    # first, get the configuration file.
    with open(tmp_path / "cfg.txt", "w") as f:
        f.write(myCfg_format.replace("$TMPDIR", str(tmp_path)))

    # now make the observation file
    obs = []
    for j in range(14):
        jj = 10 - abs(j)
        date = 61541 + 0.01 * jj
        exptime = 139.8
        ra = 60.0 + 0.01 * jj
        dec = -4.0 + 0.1 * jj
        if j > 10:
            ra -= 0.03
        pa = 20.0
        filter = "F184" if j < 12 else "H158"
        obs.append((date, exptime, ra, dec, pa, filter))
    data = np.rec.array(
        obs, formats="float64,float64,float64,float64,float64,S4", names="date,exptime,ra,dec,pa,filter"
    )
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(data=Table(data))]).writeto(
        tmp_path / "obs.fits", overwrite=True
    )

    # now make the PSFs
    (tmp_path / "psf").mkdir(parents=True, exist_ok=True)
    ov = 6  # oversampling factor needs to be even here
    psf = []
    for i in range(len(obs)):
        psf.append(OutPSF.psf_cplx_airy(ov * 20, ov * 1.326, sigma=ov * 0.3, features=i % 8))
        psf_cube = np.zeros((4,) + np.shape(psf[i]), dtype=np.float32)
        psf_cube[0, :, :] = psf[i]
        imfits = [fits.PrimaryHDU()]
        for _ in range(18):
            imfits.append(fits.ImageHDU(psf_cube))
        fits.HDUList(imfits).writeto(tmp_path / f"psf/psf_polyfit_{i:d}.fits", overwrite=True)
    ns_psf = np.shape(psf[-1])[0]
    ctr_psf = (ns_psf - 1) / 2.0

    # tophat kernel
    tk = np.ones(ov + 1)
    # 'wiggling' edges, see Numerical Recipes
    tk[0] -= 5.0 / 8.0
    tk[-1] -= 5.0 / 8.0
    tk[1] += 1.0 / 6.0
    tk[-2] += 1.0 / 6.0
    tk[2] -= 1.0 / 24.0
    tk[-3] -= 1.0 / 24.0

    # draw the images
    (tmp_path / "in").mkdir(parents=True, exist_ok=True)
    olog = ""
    for iobs in range(len(obs)):
        filt = data["filter"][iobs].decode("ascii")
        print(filt)
        if filt == "F184":
            for sca in range(1, 19):
                this_wcs = make_simple_wcs(data["ra"][iobs], data["dec"][iobs], data["pa"][iobs], sca)
                rapos, decpos = this_wcs.pixel_to_world_values(2043.5, 2043.5)
                olog += f"{iobs}, {sca}, {rapos}, {decpos}\n"

                mu = np.sin(cdec) * np.sin(decpos * degree) + np.cos(cdec) * np.cos(decpos * degree) * np.cos(
                    rapos * degree - cra
                )
                if mu > np.cos(0.08 * degree):
                    # need to draw this SCA
                    xstar, ystar = this_wcs.world_to_pixel_values(sra / degree, sdec / degree)
                    im = np.zeros((1, nside**2))
                    psfc = convolve(psf[iobs], np.outer(tk, tk), mode="same", method="direct")
                    gridD5512C(
                        psfc,
                        (ov * (np.linspace(0, nside - 1, nside) - xstar) + ctr_psf).reshape((1, nside)),
                        (ov * (np.linspace(0, nside - 1, nside) - ystar) + ctr_psf).reshape((1, nside)),
                        im,
                    )
                    mode = np.argmax(im)
                    mode_y = mode // nside
                    mode_x = mode % nside
                    im = im.reshape((nside, nside)).astype(np.float32)
                    print(np.sum(im))
                    print(
                        "**", iobs, sca, rapos, decpos, np.arccos(mu) / degree, xstar, ystar, mode_x, mode_y
                    )

                    # some tests on the images
                    assert np.sum(im) < 1.05
                    if np.sum(im) > 0.5:
                        assert np.hypot(xstar - mode_x, ystar - mode_y) < 1.0

                    # pipeline version of the WCS
                    distortion = models.AffineTransformation2D(this_wcs.wcs.cd, translation=[0, 0])
                    distortion.inverse = models.AffineTransformation2D(
                        np.linalg.inv(this_wcs.wcs.cd), translation=[0, 0]
                    )
                    celestial_rotation = models.RotateNative2Celestial(
                        this_wcs.wcs.crval[0], this_wcs.wcs.crval[1], this_wcs.wcs.lonpole
                    )
                    det2sky = (
                        (
                            models.Shift(-(this_wcs.wcs.crpix[0] - 1))
                            & models.Shift(-(this_wcs.wcs.crpix[1] - 1))
                        )
                        | distortion
                        | models.Pix2Sky_ARC()
                        | celestial_rotation
                    )
                    det2sky.name = "please_is_someone_actually_reading_this"
                    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix))
                    sky_frame = cf.CelestialFrame(
                        reference_frame=coord.ICRS(), name="icrs", unit=(u.deg, u.deg)
                    )
                    sca_gwcs = gwcs.WCS([(detector_frame, det2sky), (sky_frame, None)])

                    # Test for the gwcs.
                    #
                    # This part isn't actually testing pyimcom yet, but if there's a problem
                    # with the test setup, we will have trouble understanding failures later.
                    xs1, ys1 = sca_gwcs.invert(sra / degree, sdec / degree)
                    print(xs1, xstar, ys1, ystar)
                    assert np.hypot(xs1 - xstar, ys1 - ystar) < 1e-3

                    # write to file. these are minimal fields that are needed.
                    asdf.AsdfFile({"roman": {"data": im, "meta": {"wcs": sca_gwcs}}}).write_to(
                        tmp_path / f"in/sim_L2_{filt:s}_{iobs:d}_{sca:d}.asdf"
                    )

                    # WCS test
                    assert _stand_alone_test(str(tmp_path / f"in/sim_L2_{filt:s}_{iobs:d}_{sca:d}.asdf"))

                    # Also can write a FITS version to make sure we can ...
                    # hope this is useful to look at if something goes wrong
                    # fits.PrimaryHDU(im, header=this_wcs.to_header()).writeto(
                    #     tmp_path / f"in/sim_L2_{filt:s}_{iobs:d}_{sca:d}_asfits.fits", overwrite=True
                    # )

                    # now make the masks
                    mask = np.zeros((nside, nside), dtype=np.uint8)
                    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(mask, name="MASK")]).writeto(
                        tmp_path / f"in/sim_L2_{filt:s}_{iobs:d}_{sca:d}_mask.fits", overwrite=True
                    )

            # corners -- for testing
            sca = 18
            olog += "-- lower left--\n"
            rapos, decpos = this_wcs.pixel_to_world_values(0.0, 0.0)
            olog += f"{iobs}, {sca}, {rapos}, {decpos}\n"
            olog += "-- lower right--\n"
            rapos, decpos = this_wcs.pixel_to_world_values(4087.0, 0.0)
            olog += f"{iobs}, {sca}, {rapos}, {decpos}\n"
            if iobs == 2:
                assert np.hypot(rapos - 59.89309302318237, decpos + 2.9109906089005753) < 0.01
            olog += "-- top left--\n"
            rapos, decpos = this_wcs.pixel_to_world_values(0.0, 4087.0)
            olog += f"{iobs}, {sca}, {rapos}, {decpos}\n"
            if iobs == 2:
                assert np.hypot(rapos - 59.733417024909365, decpos + 2.982181679089024) < 0.01

    with open(tmp_path / "wcslog.txt", "w") as f:
        f.write(olog)

    # now make the cache directory
    cachedir = tmp_path / "cache"
    cachedir.mkdir(parents=True, exist_ok=True)
    # clear old cache if it is there
    files = os.listdir(cachedir)
    for file in files:
        if file[-5:] == ".fits":
            print("removing", os.path.join(cachedir, file))
            os.remove(os.path.join(cachedir, file))

    # ... and the output directory
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)

    # this part runs all 4 blocks ... they're pretty small!
    cfg = Config(tmp_path / "cfg.txt")
    print(cfg.to_file(None))
    # Running all 4 blocks so we can include the diagnostics in the test coverage.
    for iblk in range(4):
        Block(cfg=cfg, this_sub=iblk)
    gen_truthcats_from_cfg(cfg)

    # now try the multi-kappa kernel
    cfg2 = copy.deepcopy(cfg)
    cfg2.kappaC_arr = np.array([5e-4, 1e-3, 2e-3])
    cfg2.outstem += "_multik"
    cfg2()
    Block(cfg=cfg2, this_sub=1)

    # now try the empirical kernel
    cfg2 = copy.deepcopy(cfg)
    cfg2.linear_algebra = "Empirical"
    cfg2.no_qlt_ctrl = False
    cfg2.outstem += "_empir"
    cfg2()
    Block(cfg=cfg2, this_sub=1)

    # now try the iterative kernel
    cfg2 = copy.deepcopy(cfg)
    cfg2.linear_algebra = "Iterative"
    cfg2.iter_rtol = 1.5e-3
    cfg2.iter_max = 2  # not a good choice, just fast for testing
    cfg2.outstem += "_iter"
    cfg2()
    Block(cfg=cfg2, this_sub=1)


def test_PyIMCOM_run1(tmp_path, setup):
    """
    Examine PyIMCOM outputs.

    Parameters
    ----------
    tmp_path : str
        Directory in which to run the test.
    setup : function
        For pytest fixture.

    Returns
    -------
    None

    """

    ## Science star portion ##

    with fits.open(tmp_path / "out/testout_F_00_01.fits") as fblock:
        # position of "science" star
        w = wcs.WCS(fblock[0].header)
        # there are 2 extra axes if you pull the WCS this way
        posout = w.wcs_world2pix(np.array([[sra / degree, sdec / degree, 0, 0]]), 0).ravel()
        xs = posout[0]
        ys = posout[1]
        print(f"science layer star at ({xs},{ys})")

        # output block size
        (ny, nx) = np.shape(fblock[0].data)[-2:]
        x, y = np.meshgrid(np.linspace(0, nx - 1, nx), np.linspace(0, ny - 1, ny))

        # predicted & data images
        p = np.exp(-0.5 * ((x - xs) ** 2 + (y - ys) ** 2) / sig**2) / (2 * np.pi * sig**2 * sc)
        d = fblock[0].data[0, 0, :, :]
        SL1 = np.sum(p * d) / np.sum(p**2)
        VAR = np.sum((d - SL1 * p) ** 2) / np.sum(p**2)
        print("**", SL1, VAR)
        print(np.sum(p))

        assert np.abs(SL1 - 1) < 5e-4
        assert VAR < 1e-5

        # Compare to multi-kappa case
        with fits.open(tmp_path / "out/testout_F_multik_00_01.fits") as fblock_multik:
            d_multik = fblock_multik[0].data[0, 0, :, :]
            mean_diff = np.mean(d - d_multik)
            std_diff = np.std(d - d_multik)
            print(f"# {mean_diff}, {std_diff} from {np.std(d)}")
            assert std_diff < 5e-6
            assert np.abs(mean_diff) < 1e-6

        # Compare to empirical case
        with fits.open(tmp_path / "out/testout_F_empir_00_01.fits") as fblock_multik:
            d_multik = fblock_multik[0].data[0, 0, :, :]
            mean_diff = np.mean(d - d_multik)
            std_diff = np.std(d - d_multik)
            print(f"# {mean_diff}, {std_diff} from {np.std(d)}")
            assert std_diff < 0.91 * np.std(d)
            assert np.abs(mean_diff) < 2e-4

        # Compare to iterative case
        with fits.open(tmp_path / "out/testout_F_iter_00_01.fits") as fblock_multik:
            d_multik = fblock_multik[0].data[0, 0, :, :]
            mean_diff = np.mean(d - d_multik)
            std_diff = np.std(d - d_multik)
            print(f"# {mean_diff}, {std_diff} from {np.std(d)}")
            assert std_diff < 2.5e-3
            assert np.abs(mean_diff) < 2e-4

    ## Injected star portion ##

    with fits.open(tmp_path / "out/testout_F_TruthCat.fits") as f_inj:
        print(f_inj.info())
        # get the first star in the table --- in this case, it's the only one
        print(f_inj["TRUTH14"].data[0])
        ibx = f_inj["TRUTH14"].data["ibx"][0]
        iby = f_inj["TRUTH14"].data["iby"][0]
        xs = f_inj["TRUTH14"].data["x"][0]
        ys = f_inj["TRUTH14"].data["y"][0]
        print(ibx, iby, xs, ys)

    # for this one, we're going to test ReadFile
    # old code:
    # with fits.open(tmp_path / f"out/testout_F_{ibx:02d}_{iby:02d}.fits") as fblock:
    pth = pathlib.Path(tmp_path / f"out/testout_F_{ibx:02d}_{iby:02d}.fits")
    with ReadFile(pth) as fblock:
        # output block size
        (ny, nx) = np.shape(fblock[0].data)[-2:]
        x, y = np.meshgrid(np.linspace(0, nx - 1, nx), np.linspace(0, ny - 1, ny))

        # predicted & data images
        p = np.exp(-0.5 * ((x - xs) ** 2 + (y - ys) ** 2) / sig**2) / (2 * np.pi * sig**2 * sc)
        d = fblock[0].data[0, 1, :, :]
        SL1 = np.sum(p * d) / np.sum(p**2)
        VAR = np.sum((d - SL1 * p) ** 2) / np.sum(p**2)
        print("**", SL1, VAR)
        print(np.sum(p))

        assert np.abs(SL1 - 1) < 5e-4
        assert VAR < 1e-5

        # difference between gsstar and cstar
        diff = np.amax(np.abs(fblock[0].data[0, 1, :, :] - fblock[0].data[0, 3, :, :]))
        assert diff < 5e-4

        diff = np.amax(np.abs(fblock[0].data[0, 4, :, :] / 2e5 - fblock[0].data[0, 3, :, :]))
        assert diff < 5e-4
        dmax = np.amax(fblock[0].data[0, 4, :, :])
        assert np.abs(dmax - 35879.0) < 500.0

        # difference between gsstar and gstrstar
        diff = np.amax(np.abs(fblock[0].data[0, 1, :, :] - fblock[0].data[0, 8, :, :]))
        diff /= np.amax(np.abs(fblock[0].data[0, 1, :, :]))
        print(diff)
        assert diff < 0.66667

        # values from noise layers
        test5 = np.array([[0.7601451, 0.9042513], [0.64049757, 0.70962816]])
        test6 = np.array([[0.24921854, -0.23588116], [-0.39272013, -0.6111549]])
        assert np.amax(np.abs(fblock[0].data[0, 5, :2, :2] - test5)) < 1e-3
        assert np.amax(np.abs(fblock[0].data[0, 6, :2, :2] - test6)) < 1e-3

        # simulated Gaussian galaxy
        xc_ = 14  # center of region
        yc_ = 41
        im = fblock[0].data[0, 7, yc_ - 8 : yc_ + 9, xc_ - 8 : xc_ + 9]
        print(im)
        print(np.sum(im))
        moms = galsim.Image(im).FindAdaptiveMom()
        assert np.abs(moms.moments_centroid.x + xc_ - xs - 9.0) < 0.025  # 1 mas tolerance
        assert np.abs(moms.moments_centroid.y + yc_ - ys - 9.0) < 0.025  # 1 mas tolerance
        # rotate moments to original frame
        # (60 degree rotation)
        th_ = np.pi / 180.0 * 60.0
        e_ = (moms.observed_e1 + 1j * moms.observed_e2) * np.exp(2j * th_)
        e1 = e_.real
        e2 = e_.imag
        T = (0.04 * moms.moments_sigma) ** 2 / (1 - e1**2 - e2**2) * 2.0
        Cxx = T * (1 + e1) / 2.0
        Cxy = T * e2 / 2.0
        Cyy = T * (1 - e1) / 2.0
        assert np.abs(Cxx - 0.02242) < 3e-4  # compare to analytic moments for this object
        assert np.abs(Cxy + 0.00042) < 3e-4
        assert np.abs(Cyy - 0.01473) < 3e-4

    # Test output reader
    my_block = OutImage(pth)
    sci_image = my_block.get_coadded_layer("SCI")
    ci = np.argmax(sci_image)
    sci_image.ravel()[ci]

    assert np.shape(sci_image) == (100, 100)
    assert np.abs(sci_image.ravel()[843] - 0.18244877) < 1e-4

    # Test coverage
    coverage1 = my_block.get_mean_coverage()
    coverage2 = my_block.get_mean_coverage(padding=True)
    assert coverage1 >= 2.5
    assert coverage2 >= 2.5
    assert coverage1 <= 3.5
    assert coverage2 <= 3.5
    del coverage1, coverage2

    # Test output map reader
    outfidelity = my_block.get_output_map("FIDELITY")
    print(np.shape(outfidelity), np.amin(outfidelity), np.median(outfidelity), np.amax(outfidelity))
    assert np.amin(outfidelity) > 1.0e-7
    assert np.median(outfidelity) > 1.3e-6
    assert np.median(outfidelity) < 1.5e-6
    assert np.amax(outfidelity) < 1.0e-5
    assert np.shape(outfidelity) == (100, 100)

    ## Configuration test ##

    with fits.open(tmp_path / "out/testout_F_00_01.fits") as fblock:
        hdr = fblock["CONFIG"].header
        print(hdr)
        assert hdr["TILESCHM"] == "Not_specified"
        assert hdr["RERUN"] == "Not_specified"
        assert hdr["MOSAIC"] == -1
        assert hdr["FILTER"] == "F184"
        assert hdr["BLOCKX"] == 0
        assert hdr["BLOCKY"] == 1

    os.mkdir(tmp_path / "rpt")
    rpt = ValidationReport(
        str(tmp_path) + "/out/testout_F_00_00.fits", str(tmp_path) + "/rpt/report-F", clear_all=True
    )
    sectionlist = [MosaicImage, SimulatedStar, NoiseReport, LayerReport]
    for cls in sectionlist:
        s = cls(rpt)
        s.build()  # specify nblockmax to do just the lower corner
        rpt.addsections([s])
        del s
    rpt.compile()  # test that the LaTeX compiles!
    assert os.path.exists(str(tmp_path) + "/rpt/report-F_main.pdf")

    # test data blocks in report
    texdata = pull_from_file(str(tmp_path) + "/rpt/report-F_main.tex")
    assert texdata["MosaicImage"][:18] == "N =  2, BIN =   1\n"
    assert texdata["SimulatedStar"].split()[0] == "RMS_ELLIP_ADAPT"
    assert texdata["SimulatedStar"].split()[1][:7] == "0.00010"

    fields = texdata["NoiseReport"].split()
    print(fields)
    assert fields[0] == "LAYER00"
    assert fields[1] == "whitenoise1"
    assert np.abs(float(fields[2]) - 0.162502) < 1e-4
    assert fields[3] == "LAYER01"
    assert fields[4] == "1fnoise2"
    assert np.abs(float(fields[5]) - 2.96809) < 1e-4

    # Test updating the right side of this block
    my_block._load_or_save_hdu_list()  # load all the data into memory
    # just for the test, fool IMCOM into thinking we used auto so that it will run
    my_cfg = Config("".join(my_block.hdu_list["CONFIG"].data["text"].tolist()))
    my_cfg.pad_sides = "auto"
    config_hdu = fits.TableHDU.from_columns(
        [
            fits.Column(
                name="text",
                array=my_cfg.to_file(None).splitlines(),
                format="A512",
                ascii=True,
            )
        ]
    )
    config_hdu.header["EXTNAME"] = "CONFIG"
    my_block.hdu_list["CONFIG"] = config_hdu
    # The replacement I turned off since it isn't supposed to run without padding.
    #
    # pth2 = pathlib.Path(tmp_path / f"out/testout_F_{ibx+1:02d}_{iby:02d}.fits")
    # right_image = OutImage(pth2)
    # right_image._load_or_save_hdu_list()
    # d1 = np.copy(my_block.hdu_list["PRIMARY"].data[0, 0, :, -1])
    # my_block._update_hdu_data(right_image, "right", add_mode=False)
    # d2 = np.copy(my_block.hdu_list["PRIMARY"].data[0, 0, :, -1])
    # er = np.amax(np.abs(d1 - d2)) / np.amax(np.abs(d1))
    # print(er)
    # assert er > 1.0e-6
    # assert er < 0.5
    my_block._load_or_save_hdu_list(load_mode=False)  # close the data
    assert not hasattr(my_block, "hdu_list")


def test_compress(tmp_path):
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

    (tmp_path / "compression_test").mkdir(parents=True, exist_ok=True)

    # _original_file = str(tmp_path / "out/testout_F_00_01.fits")
    _original_file = EXAMPLE_FILE
    _compressed_file = str(tmp_path / "compression_test/compressiontest_F_02_11_compressed.fits")
    _decompressed_file = str(tmp_path / "compression_test/compressiontest_F_02_11_decompressed.fits")
    _recompressed_file = str(tmp_path / "compression_test/compressiontest_F_02_11_recompressed.fits")

    with CompressedOutput(_original_file) as f:
        print("ftype =", f.ftype)
        print("gzip =", f.gzip)
        print("cprstype =", f.cprstype)
        print(f.hdul.info())
        print(f.cfg.to_file(fname=None))

        for j in range(1, len(f.cfg.extrainput)):
            if (
                f.cfg.extrainput[j][:6].lower() == "gsstar"
                or f.cfg.extrainput[j][:5].lower() == "cstar"
                or f.cfg.extrainput[j][:8].lower() == "gstrstar"
                or f.cfg.extrainput[j][:8].lower() == "gsfdstar"
            ):
                f.compress_layer(
                    j,
                    scheme="I24B",
                    pars={
                        "VMIN": -1.0 / 64.0,
                        "VMAX": 7.0 / 64.0,
                        "BITKEEP": 20,
                        "DIFF": True,
                        "SOFTBIAS": -1,
                    },
                )
            if f.cfg.extrainput[j][:5].lower() == "nstar":
                f.compress_layer(
                    j,
                    scheme="I24B",
                    pars={"VMIN": -1500.0, "VMAX": 10500.0, "BITKEEP": 20, "DIFF": True, "SOFTBIAS": -1},
                )
            if f.cfg.extrainput[j][:5].lower() == "gsext":
                f.compress_layer(
                    j,
                    scheme="I24B",
                    pars={
                        "VMIN": -1.0 / 64.0,
                        "VMAX": 7.0 / 64.0,
                        "BITKEEP": 20,
                        "DIFF": True,
                        "SOFTBIAS": -1,
                    },
                )
            if f.cfg.extrainput[j][:8].lower() == "labnoise":
                f.compress_layer(
                    j,
                    scheme="I24B",
                    pars={"VMIN": -5, "VMAX": 5, "BITKEEP": 16, "DIFF": True, "SOFTBIAS": -1},
                )
            if f.cfg.extrainput[j][:10].lower() == "whitenoise":
                f.compress_layer(
                    j,
                    scheme="I24B",
                    pars={"VMIN": -8, "VMAX": 8, "BITKEEP": 16, "DIFF": True, "SOFTBIAS": -1},
                )
            if f.cfg.extrainput[j][:7].lower() == "1fnoise":
                f.compress_layer(
                    j,
                    scheme="I24B",
                    pars={"VMIN": -32, "VMAX": 32, "BITKEEP": 16, "DIFF": True, "SOFTBIAS": -1},
                )
        f.to_file(_compressed_file, overwrite=True)

    with CompressedOutput(_compressed_file) as g:
        print(g.hdul.info())
        g.decompress()
        print(g.hdul.info())
        # print(g.hdul["CPRESS"].data["text"])
        g.to_file(_decompressed_file, overwrite=True)

    with CompressedOutput(_decompressed_file) as h:
        h.recompress()
        h.to_file(_recompressed_file, overwrite=True)

    with ReadFile(_original_file) as original:
        for _step, _processed_file in zip(
            ["compressed", "decompressed", "recompressed"],
            [_compressed_file, _decompressed_file, _recompressed_file],
            strict=False,
        ):
            with ReadFile(_processed_file) as processed:
                # Now check each layer
                for j in range(np.shape(original[0].data)[-3]):
                    cprs_dict = CompressedOutput.get_compression_dict(processed, j)

                    atol = rtol = 1.0e-7  # defaults
                    # overwrite if needed
                    if "SCHEME" in cprs_dict and cprs_dict["SCHEME"][:3] == "I24":
                        atol = (float(cprs_dict["VMAX"]) - float(cprs_dict["VMIN"])) / 2 ** int(
                            cprs_dict["BITKEEP"]
                        )
                        rtol = 2 ** (-23)
                    print(f"{_step}: testing layer {j} with atol={atol} and rtol={rtol}")

                    np.testing.assert_allclose(
                        processed[0].data[0, j, :, :],
                        original[0].data[0, j, :, :],
                        rtol=rtol,
                        atol=atol,
                        err_msg=f"Compression test failed for {_step}",
                    )

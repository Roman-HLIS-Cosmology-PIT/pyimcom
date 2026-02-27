"""Shared pytest fixtures for pyimcom tests."""

import copy
import os

import asdf
import gwcs
import numpy as np
import pytest
from astropy import coordinates as coord
from astropy import units as u
from astropy.io import fits
from astropy.modeling import models
from astropy.table import Table
from furry_parakeet.pyimcom_croutines import gridD5512C
from gwcs import coordinate_frames as cf
from pyimcom.coadd import Block
from pyimcom.config import Config
from pyimcom.psfutil import OutPSF
from pyimcom.truthcats import gen_truthcats_from_cfg
from pyimcom.wcsutil import _stand_alone_test
from scipy.signal import convolve

# Import the configuration format from test_pyimcom
from .test_pyimcom import cdec, cra, degree, make_simple_wcs, myCfg_format, nside, sdec, sra


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

    return cfg

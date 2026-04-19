import gwcs
import numpy as np
import pytest
from astropy import coordinates as coord
from astropy.modeling import models
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from gwcs import coordinate_frames as cf
from pyimcom.wcsutil import ABasis, PyIMCOM_WCS, get_pix_area


def test_fits(tmp_path):
    """
    Test function for FITS WCS utilities.

    Parameters
    ----------
    tmp_path : str or pathlib.Path
        Directory in which to run the test.

    Returns
    -------
    None

    """

    # WCS information
    N = 192
    ovsamp = 2

    # reference point in degrees
    ra_ = 20.0
    dec_ = 40.0

    # make a simple FITS header
    hdu = fits.PrimaryHDU(np.zeros((N, N)))
    hdu.header["CRPIX1"] = 12
    hdu.header["CRPIX2"] = 24
    hdu.header["CD1_1"] = 0.1
    hdu.header["CD1_2"] = 0.0
    hdu.header["CD2_1"] = 0.0
    hdu.header["CD2_2"] = 0.1
    hdu.header["CTYPE1"] = "RA---STG"
    hdu.header["CTYPE2"] = "DEC--STG"
    hdu.header["CRVAL1"] = ra_
    hdu.header["CRVAL2"] = dec_
    w = WCS(hdu.header)

    print(w)

    Omega = get_pix_area(w, region=[10, 150, 22, 40], pad=1, ovsamp=ovsamp)

    f = str(tmp_path) + "/Om.fits"
    # fits.PrimaryHDU(Omega).writeto(f, overwrite=True)

    print(np.shape(Omega), np.amin(Omega), np.amax(Omega))
    print(Omega[::4, ::4])

    assert np.min(Omega) > 0.0097 and np.max(Omega) < 0.0102

    # now build the same thing rotated to a different place
    hdu.header["CRVAL1"] = 20.0
    hdu.header["CRVAL2"] = 89.0

    f = str(tmp_path) + "/Om.fits"
    fits.PrimaryHDU(Omega).writeto(f, overwrite=True)
    f2 = str(tmp_path) + "/im-zeros.fits"
    hdu.writeto(f2, overwrite=True)
    wr = WCS(hdu.header)
    Omega_rotated = get_pix_area(wr, region=[10, 150, 22, 40], pad=1, ovsamp=ovsamp)

    err = Omega_rotated / Omega - 1.0
    print(np.max(np.abs(err)))
    assert np.max(np.amax(err)) < 3e-6


def test_gwcs():
    """Test for initializing from a GWCS, and running from pixel -> world -> back."""

    # coordinates
    crpix = np.array([-254.695456590193, 819.255060728745])
    cd = np.array(
        [
            [3.08219178082192e-05, 1.22309197651664e-07],
            [0, -3.02103718199609e-05],
        ]
    )

    # pipeline version of the WCS
    distortion = models.AffineTransformation2D(cd, translation=[0, 0])
    distortion.inverse = models.AffineTransformation2D(np.linalg.inv(cd), translation=[0, 0])
    celestial_rotation = models.RotateNative2Celestial(9.55, -41.0, 180.0)
    shift = models.Shift(-(crpix[0] - 1)) & models.Shift(-(crpix[1] - 1))
    det2sky = shift | distortion | models.Pix2Sky_ARC() | celestial_rotation
    det2sky.name = "TestMapping1"
    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix))
    sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), name="icrs", unit=(u.deg, u.deg))
    sca_gwcs = gwcs.WCS([(detector_frame, det2sky), (sky_frame, None)])

    # PyIMCOM_WCS version of this
    pyimcom_wcs = PyIMCOM_WCS(sca_gwcs, noconvert=True)
    x = 256.0
    y = 512.0
    ra, dec = pyimcom_wcs.all_pix2world(x, y)
    ra2, dec2 = sca_gwcs.pixel_to_world_values([x], [y])
    print(ra, dec)
    assert np.hypot(ra - ra2[0], dec - dec2[0]) < 1.0e-5  # check pixel mapping
    xx, yy = pyimcom_wcs.all_world2pix(ra, dec)
    assert np.hypot(x - xx, y - yy) < 0.01  # check pixel mapping


def test_special_cases():
    """Test edge cases for WCS."""

    ABasis(6, 1000).coef_setup()  # should just pass through
    with pytest.raises(TypeError):
        PyIMCOM_WCS(0)  # can't give it an integer

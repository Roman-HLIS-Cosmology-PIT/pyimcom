import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from pyimcom.wcsutil import get_pix_area


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

    Omega = get_pix_area(w, region=[10, 50, 22, 40], pad=1, ovsamp=ovsamp)

    f = str(tmp_path) + "/Om.fits"  # noqa: F841
    # fits.PrimaryHDU(Omega).writeto(f, overwrite=True)

    print(np.shape(Omega))
    print(Omega[::4, ::4])

    assert np.min(Omega) > 0.0098 and np.max(Omega) < 0.0102

    # now build the same thing rotated to a different place
    hdu.header["CRVAL1"] = 20.0
    hdu.header["CRVAL2"] = 89.0

    f = str(tmp_path) + "/Om.fits"  # noqa: F841
    fits.PrimaryHDU(Omega).writeto(f, overwrite=True)
    Omega_rotated = get_pix_area(w, region=[10, 50, 22, 40], pad=1, ovsamp=ovsamp)

    err = Omega_rotated / Omega - 1.0
    print(np.max(np.abs(err)))
    assert np.max(np.amax(err)) < 1e-7


# test_fits("out") # <-- comment out

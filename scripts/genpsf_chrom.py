import sys

import galsim
import numpy as np
from astropy.io import fits
from roman_imsim.utils import roman_utils
from scipy.ndimage import gaussian_filter


def get_psf_fits(
    obsid, outdir, oversample_factor=8, sed_type="flat", stamp_size=512, sed=None, normalize=True
):
    """
    This function builds an oversampled PSF image for each of the 18 SCAs of a Roman
    WFI exposure and writes a multi-extension FITS file.  For each SCA, several PSFs
    are evaluated at fixed reference positions, mapped to Legendre polynomial
    coefficients (0th, d/du, d/dv, d^2/dudv). Example use is provided at bottom
    of this file.

    Parameters
    ----------
    obsid : int
        Observation ID.
    outdir : str
        Directory where the output file will be written.
    oversample_factor : int, optional
        Oversampling factor used when drawing the PSF. Default is 8.
    sed_type : {"flat", "lin", "quad", "real"}, optional
        Type of SED used for the PSF:
            - "flat":    constant photon SED
            - "lin":     SED proportional to the wavelength
            - "quad":    SED proportional to the wavelength squared
            - "real":    a user-supplied `galsim.SED` object via the `sed` argument
    stamp_size : int, optional
        Size (in pixels) of the square oversampled PSF image to draw. Default: 512.
    sed : galsim.SED, optional
        Actual SED object used only when ``sed_type="real"``.
    normalize : bool, optional
        If True, normalize the PSF flux to unity through the bandpass of each SCA’s
        WAS image. Default: True.

    Returns
    ------
    A FITS file written to::

        <outdir>/psf_polyfit_<obsid>.fits

    containing 19 HDUs (1 primary + 18 SCAs).
    """

    st_model = galsim.DeltaFunction()

    if sed_type == "flat":
        st_model = st_model * galsim.SED(lambda x: 1, "nm", "fphotons")
    if sed_type == "lin":
        st_model = st_model * galsim.SED(lambda x: x, "nm", "fphotons")
    if sed_type == "quad":
        st_model = st_model * galsim.SED(lambda x: x**2, "nm", "fphotons")
    if sed_type == "real":
        assert sed is not None
        st_model = st_model * sed

    # put some information in the Primary HDU
    mainhdu = fits.PrimaryHDU()
    mainhdu.header["CFORMAT"] = "Legendre basis"
    mainhdu.header["PORDER"] = (1, "bivariate polynomial order")
    mainhdu.header["ABSCISSA"] = ("u=(x-2044.5)/2044, v=(y-2044.5)/2044", "x,y start at 1")
    mainhdu.header["NCOEF"] = (4, "(PORDER+1)**2")
    mainhdu.header["SEQ"] = "for n=0..PORDER { for m=0..PORDER { coef P_m(u) P_n(v) }}"
    mainhdu.header["OBSID"] = obsid
    mainhdu.header["NSCA"] = 18
    mainhdu.header["OVSAMP"] = 8
    mainhdu.header["SIMRUN"] = "Theta -> OSC"
    hdulist = [mainhdu]

    # make each layer
    for sca in range(1, 19):
        util = roman_utils("was.yaml", image_name=f"Roman_WAS_simple_model_Y106_{obsid:d}_{sca:d}.fits.gz")
        if normalize:
            st_model = st_model.withFlux(1.0, util.bpass)

        out_psf = np.zeros((6, stamp_size, stamp_size))
        x_ = [1.0, 4088.0, 1.0, 4088.0, 2044.5, 500.0]
        y_ = [1.0, 1.0, 4088.0, 4088.0, 2044.5, 1000.0]
        for j in range(5):
            x_[j] = 0.9 * x_[j] + 0.1 * 2044.5
            y_[j] = 0.9 * y_[j] + 0.1 * 2044.5
            psf = util.getPSF(x_[j], y_[j], 8)
            psf_image = galsim.Convolve(st_model, galsim.Transform(psf, jac=8 * np.identity(2)))
            stamp = galsim.Image(stamp_size, stamp_size, wcs=util.wcs)
            arr = psf_image.drawImage(util.bpass, image=stamp, wcs=util.wcs, method="no_pixel")
            out_psf[j, :, :] = arr.array

        out_psf_coef = np.zeros((5, stamp_size, stamp_size))
        out_psf_coef[0, :, :] = out_psf[4, :, :]
        out_psf_coef[1, :, :] = (
            (out_psf[1, :, :] + out_psf[3, :, :] - out_psf[0, :, :] - out_psf[2, :, :])
            / 2.0
            / (4087 * 0.9)
            * 2044
        )
        out_psf_coef[2, :, :] = (
            (out_psf[2, :, :] + out_psf[3, :, :] - out_psf[0, :, :] - out_psf[1, :, :])
            / 2.0
            / (4087 * 0.9)
            * 2044
        )
        out_psf_coef[3, :, :] = (
            (out_psf[0, :, :] + out_psf[3, :, :] - out_psf[1, :, :] - out_psf[2, :, :])
            / 4.0
            / (4087 * 0.45) ** 2
            * 2044**2
        )

        # convolution
        sig_mtf = 0.3279 * 8  # 8 for oversampled pixels
        for j in range(4):
            out_psf_coef[j, :, :] = (
                0.17519 * gaussian_filter(out_psf_coef[j, :, :], 0.4522 * sig_mtf, truncate=7.0)
                + 0.53146 * gaussian_filter(out_psf_coef[j, :, :], 0.8050 * sig_mtf, truncate=7.0)
                + 0.29335 * gaussian_filter(out_psf_coef[j, :, :], 1.4329 * sig_mtf, truncate=7.0)
            )

        hdu = fits.ImageHDU(out_psf_coef[:4, 128:-128, 128:-128].astype(np.float32))
        hdu.header["OBSID"] = obsid
        hdu.header["SCA"] = sca

        hdulist.append(hdu)
    fits.HDUList(hdulist).writeto(outdir + f"/psf_polyfit_{obsid:d}.fits", overwrite=True)


# parameters
sed_type = "flat"
sed = None
eff_const = False
normalize = True
outdir = "./input_psf/psf_flatsed"
obsid = int(sys.argv[1])
stamp_size = 512

## Run psf generation
get_psf_fits(obsid, outdir, sed_type=sed_type, sed=sed, normalize=normalize, eff_const=eff_const)

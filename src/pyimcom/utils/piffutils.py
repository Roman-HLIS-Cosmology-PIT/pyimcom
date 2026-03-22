import numpy as np
import piff
from numpy.polynomial import legendre


def piff_to_legendre(psf_file, chipnum, stamp_size=128, oversamp=6, legendre_order=5, normbox=None):
    """Convert a PSF file from piff to a Legendre polynomial expansion.

    Parameters
    ----------
    psf_file : str
        The path to the PSF file.
    chipnum: int
        The sca/chip number at which to draw the PSF.
    stamp_size: int, optional
        The size of the PSF stamp. Default is 128.
    oversamp: int, optional
        The oversampling factor for the PSF stamp. Default is 6.
    legendre_order : int, optional
        Polynomial order for Legendre polynomial expansion. Default is 5.
    normbox : int, optional
        If given, normalizes the PSF to integrate to 1 in the specified box size (which may be
        different from the region size used in Piff; we envision it will be larger if the PSF has
        far wings that are not re-fit by Piff, e.g., from a physical model or fit to scattered
        light in stacked bright stars, etc.).

    Returns
    -------
    coeffs : np.ndarray of shape ((legendre order + 1)**2, stamp_size*oversamp, stamp_size*oversamp)
        The coefficients of the Legendre polynomial expansion.
    """

    # First read the psf via piff from given file
    psf = piff.read(psf_file)

    # Now, find the points at which you want to draw the PSF
    # which is given by the Gauss Legendre method. This should capture
    # the spatial variance in the PSF through the Legendre polynomials.

    quad_points, quad_weights = legendre.leggauss(legendre_order + 1)
    # transform quad_points from [-1,1] to [0, 4088]
    quad_coords = 2044.0 * quad_points + 2043.5
    # Precompute 1D Legendre basis functions at the quadrature points.
    basis_functions = np.array(
        [legendre.legval(quad_points, [0] * k + [1]) for k in range(legendre_order + 1)]
    )
    # Initialize coefficient array.
    coeffs = np.zeros(
        ((legendre_order + 1) ** 2, stamp_size * oversamp, stamp_size * oversamp), dtype=np.float32
    )
    # Now, we draw the PSF at the given points.
    for iu, x in enumerate(quad_coords):
        for iv, y in enumerate(quad_coords):
            stamp = np.zeros((stamp_size * oversamp, stamp_size * oversamp), dtype=np.float32)

            # get sub-PSFs in each region
            s = np.linspace(-0.5 + 0.5 / oversamp, 0.5 - 0.5 / oversamp, oversamp)
            for j in range(oversamp):
                for i in range(oversamp):
                    stamp[j::oversamp, i::oversamp] = psf.draw(
                        chipnum=chipnum,
                        x=x,
                        y=y,
                        center=True,
                        offset=(-s[i], -s[j]),
                        stamp_size=stamp_size,
                        sca=chipnum,
                    ).array

            # normalization
            if normbox is not None:
                stamp[:, :] /= np.sum(
                    psf.draw(chipnum=chipnum, x=x, y=y, center=True, stamp_size=normbox, sca=chipnum).array
                )

            # For each pair of Legendre orders, update the corresponding coefficient image
            idx = 0
            for v_order in range(legendre_order + 1):
                for u_order in range(legendre_order + 1):
                    # Legendre polynomial normalization. Also includes oversamp**2 because
                    # IMCOM expects the PSF to sum to 1 (so think of this as "fraction of response
                    # in each subpixel").
                    norm = (2 * u_order + 1) * (2 * v_order + 1) / 4.0 / oversamp**2
                    weight = (
                        norm
                        * quad_weights[iu]
                        * quad_weights[iv]
                        * basis_functions[u_order, iu]
                        * basis_functions[v_order, iv]
                    )
                    coeffs[idx, :, :] += weight * stamp
                    idx += 1
    return coeffs

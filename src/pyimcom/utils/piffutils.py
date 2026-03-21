import numpy as np
import piff
from numpy.polynomial import legendre


def piff_to_legendre(psf_file, chipnum, stamp_size=128, oversamp=6, legendre_order=5):
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
            s = np.linspace(-0.5 + 0.5 / oversamp, 0.5 - 0.5 / oversamp, oversamp)
            for j in range(oversamp):
                for i in range(oversamp):
                    te = psf.draw(
                        chipnum=chipnum,
                        x=x,
                        y=y,
                        center=True,
                        offset=(-s[i], -s[j]),
                        stamp_size=stamp_size,
                        sca=chipnum
                    ).array
                    print(stamp_size, oversamp, np.shape(te), np.shape(stamp), np.shape(stamp[j::oversamp, i::oversamp]))
                    stamp[j::oversamp, i::oversamp] = te
            # For each pair of Legendre orders, update the corresponding coefficient image
            idx = 0
            for v_order in range(legendre_order + 1):
                for u_order in range(legendre_order + 1):
                    norm = (2 * u_order + 1) * (2 * v_order + 1) / 4.0
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

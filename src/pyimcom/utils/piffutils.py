import numpy as np
import piff
from numpy.polynomial import legendre
from scipy.interpolate import RegularGridInterpolator

def piff_to_legendre(psf_file, chipnum, x, y, stamp_size = 48, legendre_order = 5):
    """Convert a PSF file from piff to a Legendre polynomial expansion.

    Parameters
    ----------
    psf_file : str
        The path to the PSF file.
    legendre_order : int
        Polynomial order for Legendre polynomial expansion. Default is 5.

    Returns
    -------
    coeffs : np.ndarray
        The coefficients of the Legendre polynomial expansion.
    """
    psf = piff.read(psf_file)
    image = psf.draw(chipnum = chipnum, x = x, y = y, stamp_size = stamp_size)
    image_data = image.array

    #Using Gauss-Legendre Method to sample PSF at roots of Legendre polynomials
    quad_points, quad_weights = legendre.leggauss(legendre_order + 1)
    #transform coordinates to stamp range [0, stamp_size]
    quad_coords = (quad_points + 1) * stamp_size / 2
    xx, yy = np.meshgrid(quad_coords, quad_coords)
    coords_full = np.linspace(0, stamp_size, stamp_size)
    #interpolate to get PSF values at the roots/quadrature points
    interp = RegularGridInterpolator((coords_full, coords_full), image_data, bounds_error=False, fill_value=0)
    psf_values = interp(np.column_stack([yy.ravel(), xx.ravel()])).reshape(xx.shape)
    n_basis = (legendre_order + 1) ** 2
    coeffs = np.zeros(n_basis)
    idx = 0
    #Should return in order of iterating through x first, then y, which is the same order as the basis functions are defined
    for j in range(legendre_order + 1):
        for i in range(legendre_order + 1):
            # Evaluate Legendre polynomials at quadrature points
            leg_y = legendre.legval(quad_points, [0]*i + [1])
            leg_x = legendre.legval(quad_points, [0]*j + [1])
            W = np.outer(quad_weights*leg_y, quad_weights*leg_x)
            coeffs[idx] = ((2*i + 1)*(2*j + 1) / 4.0) * np.sum(W * psf_values)
            idx += 1

    return coeffs



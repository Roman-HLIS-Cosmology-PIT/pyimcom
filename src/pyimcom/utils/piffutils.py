import numpy as np
import piff
from numpy.polynomial import legendre

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
    
    coords = np.linspace(-1, 1, stamp_size)
    xx, yy = np.meshgrid(coords, coords) 
    coeffs = legendre.legfit(np.vstack(xx.ravel(), yy.ravel()), image.ravel(), deg = [legendre_order, legendre_order])
    coeffs = coeffs.reshape((legendre_order + 1, legendre_order + 1))
    return coeffs
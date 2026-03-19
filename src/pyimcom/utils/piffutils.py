import numpy as np
import piff
from numpy.polynomial import legendre
from scipy.interpolate import RegularGridInterpolator

def piff_to_legendre(psf_file, chipnum, stamp_size = 128, oversamp = 6, legendre_order = 5):
    """Convert a PSF file from piff to a Legendre polynomial expansion.

    Parameters
    ----------
    psf_file : str
        The path to the PSF file.
    legendre_order : int
        Polynomial order for Legendre polynomial expansion. Default is 5.

    Returns
    -------
    coeffs : np.ndarray of shape ((legendre order + 1)**2, stamp_size*oversamp, stamp_size*oversamp)
        The coefficients of the Legendre polynomial expansion.
    """
    #First read the psf via piff from given file
    psf = piff.read(psf_file)
    #Generate an empty 4d array, where the first coordinate corresponds to 
    #the u grid, the second coordinate corresponds to the v grid, 
    #and the third and fourth coordinates each correspond to the psf stamp.
    stamps = np.zeros((legendre_order + 1, legendre_order + 1, stamp_size*oversamp, stamp_size*oversamp))
    

    #Now, find the points at which you want to draw the PSF
    #which is given by the Gauss Legendre method. This should capture
    #the spatial variance in the PSF through the Legendre polynomials. 

    quad_points, quad_weights = legendre.leggauss(legendre_order + 1)
    #transform quad_points from [-1,1] to [0, 4088]
    quad_coords = 2044.0 * quad_points + 2043.5
    #Now, we draw the PSF at the given points. 
    for i, x in enumerate(quad_coords):
        for j, y in enumerate(quad_coords):
            stamps[i, j, :, :] = psf.draw(chipnum = chipnum, x = x, y = y, stamp_size = stamp_size*oversamp, sca = chipnum).array
            
    coeffs = np.zeros(((legendre_order + 1)**2, stamp_size*oversamp, stamp_size*oversamp))
    basis_functions = np.array([legendre.legval(quad_points, [0]*k + [1]) for k in range(legendre_order + 1)])
    idx = 0
    for j in range(legendre_order + 1): #order in v
        for i in range(legendre_order + 1): # order in u
            w_u = quad_weights * basis_functions[i]
            w_v = quad_weights * basis_functions[j]
            W = np.outer(w_u, w_v)
            norm = (2*i + 1) * (2*j + 1) / 4.0 

            weighted_stamps = stamps * W[:, :, np.newaxis, np.newaxis]
            coeffs[idx, :, :] = norm * np.sum(weighted_stamps, axis =(0, 1))
            idx += 1

    return coeffs
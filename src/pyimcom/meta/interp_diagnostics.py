"""
Fourier space information.

"""

import numpy as np

# from astropy.io import fits
from .ginterp import MultiInterp

def interp_transfer_function(Rsearch, samp, Cov, Jac=None, epsilon=1.0e-7, umax=1.0, N=128):
    """
    Computes the Fourier domain interpolation transfer function.

    Parameters
    ----------
    Rsearch : float
        Search radius (from corners), in gridded pixels.
    samp : float
        Sampling rate of input image (samples per FWHM).
    Cov : np.ndarray of float
        Covariance matrix of extra smoothing. length 3, array-like [Cxx, Cxy, Cyy].
    Jac : np.ndarray of float or None
        The Jacobian to apply; shape = (2, 2).
    epsilon : float, optional
        Regularization parameter to prevent singular correlations.
    umax : float, optional
        Maximum value of u or v.
    N : int, optional
        Size of output array (N, N). Must be even.

    Returns
    -------
    T_tilde : np.ndarray of float
        A map of the transfer function.

    Notes
    -----
    The output is sampled on the grid of Fourier modes with u and v at::

        -umax, umax*(-1+2/N), umax*(-1+4/N) ... umax*(1-2/N)

    The zero mode is at `T_tilde[N//2, N//2]`.

    """

    # default is no shear
    if Jac is None:
        Jac = np.identity(2)

    # check if N is even
    if N%2==1:
        raise ValueError(f"interp_transfer_function: N={N} is odd.")

    N_ = int(np.ceil(N/umax + Rsearch + 1))
    if N_%2==1:
        N_+=1
    in_array = np.zeros((1, 2*N_, 2*N_))
    in_array[0, N_, N_] = 1.

    in_mask = np.zeros((2*N_, 2*N_), dtype=bool) # trivial mask

    out_array, _, _, _ = MultiInterp(in_array, in_mask, (N, N), np.array([N_, N_]) - N/4./umax*Jac@np.array([1,1]),
        Jac/(2*umax), Rsearch, samp, Cov, epsilon=epsilon)
    out_array = out_array[0, :, :] * np.abs(np.linalg.det(Jac))

    out_array_ft = np.fft.fft2(np.fft.fftshift(out_array)).real / (2.*umax)**2 # normalized FT
    return np.fft.fftshift(out_array_ft)

# def _sample():
#     t = interp_transfer_function(6.0, 4.5, np.array([0.64, 0., 0.32]), Jac = np.array([[0.5,0.],[0.,0.5]]), epsilon=1.0e-7, umax=1.0, N=256)
#     fits.PrimaryHDU(t).writeto("t.fits", overwrite=True)
# _sample()

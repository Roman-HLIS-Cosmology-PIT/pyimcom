"""First pass at noise power spectrum / homogenization.

(This isn't ready yet, but is something I worked on last year so want to put it
in the repository so I don't lose the code over time. --C.H.)
"""

import sys

import numpy as np
from astropy.io import fits
from scipy.signal import convolve


def winonly(im, n1):
    """
    Tukey window an image, fading in/out over the first n1 pixels on each side.

    Parameters
    ----------
    im : np.array
        The image array.
    n1 : int
        Number of pixels around the edge to window.

    Returns
    -------
    np.array
        The windowed array.

    """

    (ny, nx) = np.shape(im)
    imwin = np.copy(im)
    f = np.ones(nx)
    f[:n1] = np.sin(np.linspace(0.5, n1 - 0.5, n1) * np.pi / 2.0 / n1) ** 2
    f[-n1:] = f[n1 - 1 :: -1]
    imwin[:, :] *= f[None, :]
    Area = np.sum(f**2)
    f = np.ones(ny)
    f[:n1] = np.sin(np.linspace(0.5, n1 - 0.5, n1) * np.pi / 2.0 / n1) ** 2
    f[-n1:] = f[n1 - 1 :: -1]
    imwin[:, :] *= f[:, None]
    return imwin


def winpad(im, n1):
    """
    Tukey window and pad an image to a multiple of n1.

    Parameters
    ----------
    im : np.array
        The image array.
    n1 : int
        Number of pixels around the edge to window.

    Returns
    -------
    imbig : np.array
        The image, windowed and padded a multiple of `n1` on each axis.
    Area : float
        The effective area of the image (in pixels).

    """

    (ny, nx) = np.shape(im)
    imwin = np.copy(im)
    f = np.ones(nx)
    f[:n1] = np.sin(np.linspace(0.5, n1 - 0.5, n1) * np.pi / 2.0 / n1) ** 2
    f[-n1:] = f[n1 - 1 :: -1]
    imwin[:, :] *= f[None, :]
    Area = np.sum(f**2)
    f = np.ones(ny)
    f[:n1] = np.sin(np.linspace(0.5, n1 - 0.5, n1) * np.pi / 2.0 / n1) ** 2
    f[-n1:] = f[n1 - 1 :: -1]
    imwin[:, :] *= f[:, None]
    Area *= np.sum(f**2)
    ny2 = (ny // n1 + 1) * n1
    nx2 = (nx // n1 + 1) * n1
    imbig = np.zeros((ny2, nx2), dtype=im.dtype)
    imbig[:ny, :nx] = imwin
    return imbig, Area


def binpowspec(im, n1):
    """
    Power spectrum, binned into n1 x n1.

    Parameters
    ----------
    im : np.array
        The image array.
    n1 : int
        Number of pixels around the edge to window.

    Returns
    -------
    np.array
        The power spectrum, averaged into bins of wavenumber of 1/`n1` cycles per pixel
        on each axis. The shape is (`n1`, `n1`).

    Notes
    -----
    * `n1` should be even.
    * the [0,0] corner is wavenumber zero.
    * the average of the power spectrum array is normalized to be the variance of `im`

    """

    assert n1 % 2 == 0
    imbig, Area = winpad(im, 2 * n1)
    (ny2, nx2) = np.shape(imbig)
    powspec = np.abs(np.fft.fft2(imbig)) ** 2 / Area
    powspec = (powspec + np.roll(powspec, -1, axis=0)) / 2.0
    powspec = (powspec + np.roll(powspec, -1, axis=1)) / 2.0
    powspec = np.roll(powspec, (ny2 // n1 // 2, nx2 // n1 // 2), axis=(0, 1))
    return np.mean(powspec.reshape((n1, ny2 // n1, n1, nx2 // n1)), axis=(1, 3))


def recoverpower(im, n2, ps, offset=(0, 0), anc=None):
    """
    Makes an image that has the complementary local power spectrum to im.

    Parameters
    ----------
    im : np.array
        The input noise image.
    n2 : int
        The spacing at which the power spectrum is computed.
    ps : np.array
        The target power spectrum.
    offset : (int, int), optional
        The coordinates of a point on the regular `n2` x `n2` grid to be used
        in computing power spectra (this is in case you want to use images that aren't spaced
        by a multiple of `n2`).
    anc : np.array, optional
        A 3D array, where each slice ``anc[j,:,:]`` has the same power spectrum as `im`.

    Returns
    -------
    im_out : np.array
        The complementary map. This is a filtered version of `im` with the same shape.
    err : np.array
        An error map, also the same shape as `im`; indicates fractional error in setting the
        power spectrum (due to, e.g., `im` having too much power).

    Notes
    -----
    The input power spectrum ps is an (n1,n1) shaped grid with k=0 at the [0,0] corner.

    Power spectrum is binned to an n1 x n1 grid (i.e., correlation function
    is represented out to a separation of n1/2).

    The power spectrum is re-computed on a grid every n2 x n2 pixels. If the size of image im
    is not a multiple of n2, there may be some extrapolation near the upper and right sides.
    Must have n2 >= 2*n1.

    If offset is non-zero, gives the (x,y) coordinates to offset from the corner to the regular
    grid (allows the same grid to be used across images that are not a multiple of n2 pixels separated).

    An ancillary 3D image with multiple layers can be given as well. It is assumed that each
    layer has the same local power spectrum as im (this reduces sample variance).

    """

    # Read dimensions
    (n1, n1_) = np.shape(ps)
    (ny, nx) = np.shape(im)
    nyblock = (ny - offset[1]) // n2
    nxblock = (nx - offset[0]) // n2
    assert n2 >= 2 * n1
    assert n1_ == n1

    im_out = np.zeros_like(im)
    err = np.zeros_like(im)

    # Tukey edge filter
    tkf = np.sin(np.linspace(0.5, 2 * n1 - 0.5, 2 * n1) * np.pi / 2.0 / 2.0 / n1) ** 2

    # in the following loop:
    # xbmin,xbmax,ybmin,ybmax = bounding box of the region under study
    # xbmin_,xbmax_,ybmin_,ybmax_ = padded bounding box (including overlap with neighbors)
    for jy in range(nyblock):
        ybmin = jy * n2 + offset[1]
        ybmax = ybmin + n2
        if jy == 0:
            ybmin = 0
        if jy == nyblock - 1:
            ybmax = ny
        ybmin_ = max(ybmin - n1, 0)
        ybmax_ = min(ybmax + n1, ny)
        for jx in range(nxblock):
            xbmin = jx * n2 + offset[0]
            xbmax = xbmin + n2
            if jx == 0:
                xbmin = 0
            if jx == nxblock - 1:
                xbmax = nx
            xbmin_ = max(xbmin - n1, 0)
            xbmax_ = min(xbmax + n1, nx)
            im_ps = binpowspec(im[ybmin_:ybmax_, xbmin_:xbmax_], n1)
            if anc is not None:
                n_anc = np.shape(anc)[0]
                for k_ in range(n_anc):
                    im_ps += binpowspec(anc[k_, ybmin_:ybmax_, xbmin_:xbmax_], n1)
                im_ps /= n_anc + 1
            f = np.clip(im_ps / ps, 1e-2, 1)
            R = np.sqrt((1.0 - f) / f)
            # print('>>', R[0,0], R[1,0], R[0,1])
            Rr = np.zeros((2 * n1 + 1, 2 * n1 + 1))
            Rr[n1 // 2 + 1 : n1 // 2 + n1, n1 // 2 + 1 : n1 // 2 + n1] = np.real(
                np.fft.fftshift(np.fft.ifft2(R))
            )[1:, 1:]

            # get filtered image, trim if necessary
            tpad = np.copy(im[ybmin_:ybmax_, xbmin_:xbmax_])
            if jy != 0:
                tpad[: 2 * n1, :] *= tkf[:, None]
            if jy != nyblock - 1:
                tpad[-2 * n1 :, :] *= tkf[::-1, None]
            if jx != 0:
                tpad[:, : 2 * n1] *= tkf[None, :]
            if jx != nyblock - 1:
                tpad[:, -2 * n1 :] *= tkf[None, ::-1]
            block = convolve(tpad, Rr, mode="full", method="auto")[n1:-n1, n1:-n1]

            # print(jy, jx, (xbmin,xbmax,ybmin,ybmax), (xbmin_,xbmax_,ybmin_,ybmax_))
            im_out[ybmin_:ybmax_, xbmin_:xbmax_] += block

            # warnings
            thiserr = np.sum(np.clip(im_ps - ps, 0.0, None)) / np.sum(ps)
            err[ybmin_:ybmax_, xbmin_:xbmax_] = np.maximum(thiserr, err[ybmin_:ybmax_, xbmin_:xbmax_])
    return im_out, err


def recoverpower2(im, n2, ps, offset=(0, 0), anc=None, nsub=2):
    """
    Makes an image that has the complementary local power spectrum to im.

    Like ``recoverpower`, except with the power spectrum computed in sub-blocks for a smoother output.

    Parameters
    ----------
    im : np.array
        The input noise image.
    n2 : int
        The spacing at which the power spectrum is computed.
    ps : np.array
        The target power spectrum.
    offset : (int, int), optional
        The coordinates of a point on the regular `n2` x `n2` grid to be used
        in computing power spectra (this is in case you want to use images that aren't spaced
        by a multiple of `n2`).
    anc : np.array, optional
        A 3D array, where each slice ``anc[j,:,:]`` has the same power spectrum as `im`.

    Returns
    -------
    im_out : np.array
        The complementary map. This is a filtered version of `im` with the same shape.
    err : np.array
        An error map, also the same shape as `im`; indicates fractional error in setting the
        power spectrum (due to, e.g., `im` having too much power).

    See Also
    --------
    recoverpower
        Version without sub-blocks.

    """

    im_out = np.zeros_like(im)
    (ny, nx) = np.shape(im)
    err = np.zeros((ny, nx), dtype=np.float32)
    for iy in range(nsub):
        for ix in range(nsub):
            offset_ = ((offset[0] + ix * n2 // nsub) % n2, (offset[1] + iy * n2 // nsub) % n2)
            im_, err_ = recoverpower(im, n2, ps, offset=offset_, anc=anc)
            im_out += im_ / nsub**2
            err += err_ / nsub**2
    return im_out, err


def gaussps(var, sigx, n1, ovsamp=4):
    """
    Helper function to make a Gaussian power spectrum.

    Parameters
    ----------
    var : float
        The desired variance.
    sigx : float
        The 1 sigma width of the correlation function in real-space pixels.
    n1 : int
        Size of the power spectrum image.
    ovsamp : int, optional
        Oversampling factor for computing the power spectrum.

    Returns
    -------
    np.array
        A 2D power spectrum of shape (`n1`, `n1`). This is sampled with the [j,i]
        component at wavenumber (i/`n1`, j/`n1`) cycles per pixel.

    """

    ps = np.zeros((n1 * ovsamp, n1 * ovsamp))
    sigk = 1.0 / (2 * np.pi) / sigx  # width in k-space in units of the Brillouin zone
    x = np.linspace(0, 1 - 1 / (n1 * ovsamp), n1 * ovsamp) - (ovsamp - 1) / 2.0 / ovsamp / n1
    x = np.where(x >= 0.5, x - 1, x)
    kx, ky = np.meshgrid(x, x)
    ps = np.exp(-(kx**2 + ky**2) / 2.0 / sigk**2)
    ps = np.mean(ps.reshape((n1, ovsamp, n1, ovsamp)), axis=(1, 3))
    ps *= var / np.mean(ps)
    return ps


def cf(im, n1):
    """
    Correlation function of an image.

    Parameters
    ----------
    im : np.array
        The input image.
    n1 : int
        The size of the correlation function image desired.

    Returns
    -------
    np.array
        The correlation function, shape = (`n1`, `n1`). The zero lag correlation
        function is at ``[n1//2, n1//2]``.

    Notes
    -----
    If you want the zero-lag correlation in the center, choose `n1` to be odd.

    """

    (ny, nx) = np.shape(im)
    cfn = convolve(im, im[::-1, ::-1], mode="full", method="fft")[
        ny - 1 - n1 // 2 : ny - 1 - n1 // 2 + n1, nx - 1 - n1 // 2 : nx - 1 - n1 // 2 + n1
    ]
    x, y = np.meshgrid(np.linspace(-(n1 // 2), n1 - n1 // 2, n1), np.linspace(-(n1 // 2), n1 - n1 // 2, n1))
    cfd = (ny - np.abs(y)) * (nx - np.abs(x))
    return cfn / cfd


def _test(infile):
    with fits.open(infile) as f:
        # <-- Here, might need to take a different choice of layers / ultimately make this more general.
        im = f[0].data[0, -1, :, :]
        im_other = f[0].data[0, -2, :, :]
        imB = f[0].data[0, -4:-2, :, :]
    ex, area = winpad(im, 16)
    print(np.shape(im), area)
    fits.PrimaryHDU(ex).writeto("ex.fits", overwrite=True)
    print(np.mean(binpowspec(im, 24)), binpowspec(im, 24)[0, 0])
    # print(np.std(im)**2)

    A = 0.3
    S = 2.2
    ps = gaussps(A, S, 24)
    ps_ext = gaussps(A, S, 32)
    print(np.mean(ps), ps[0, 0])
    complement, err = recoverpower2(im, 100, ps_ext, offset=(0, 0), anc=imB, nsub=2)
    # print(np.mean(err,axis=(0,1)))
    print(np.percentile(err, 50))
    print(np.percentile(err, 75))
    print(np.percentile(err, 95))
    print(np.percentile(err, 99))
    fits.PrimaryHDU(complement).writeto("complement.fits", overwrite=True)
    fits.PrimaryHDU(complement + im_other).writeto("tot.fits", overwrite=True)
    fits.PrimaryHDU(err).writeto("err.fits", overwrite=True)

    # power spectrum outputs
    ps_out = np.zeros((120, 24))
    ps_out[0:24, :] = np.fft.fftshift(ps)
    ps_out[24:48, :] = np.fft.fftshift(binpowspec(complement + im_other, 24))
    ps_out[48:72, :] = np.fft.fftshift(binpowspec(im_other, 24))
    ps_out[72:96, :] = np.fft.fftshift(binpowspec(complement, 24))
    ps_out[96:120, :] = ps_out[24:48, :] - ps_out[0:24, :]
    fits.PrimaryHDU(ps_out).writeto("ps.fits", overwrite=True)

    # correlation fcn outputs
    N = 81
    cf_out = np.zeros((3 * N, N))
    cf_out[:N, :] = cf(complement + im_other, N)
    cf_out[N : 2 * N, :] = A * np.exp(
        -0.5
        * (
            np.linspace(-(N - 1) / 2, (N - 1) / 2, N)[None, :] ** 2
            + np.linspace(-(N - 1) / 2, (N - 1) / 2, N)[:, None] ** 2
        )
        / S**2
    )
    cf_out[2 * N :, :] = cf_out[:N, :] - cf_out[N : 2 * N, :]
    fits.PrimaryHDU(cf_out).writeto("cf.fits", overwrite=True)

    (ny, nx) = np.shape(im)
    c = 6
    for iy in range(c):
        y1 = iy * ny // c
        y2 = (iy + 1) * ny // c
        for ix in range(c):
            x1 = ix * nx // c
            x2 = (ix + 1) * nx // c
            print(
                "{:2d} {:2d} {:6.4f} {:6.4f} {:6.4f}".format(
                    iy,
                    ix,
                    np.std(im_other[y1:y2, x1:x2]),
                    np.std(complement[y1:y2, x1:x2]),
                    np.std(complement[y1:y2, x1:x2] + im_other[y1:y2, x1:x2]),
                )
            )


if __name__ == "__main__":
    # this is meant to be called python reconv.py [path]/prod_H_16_16_map.fits
    # the latter should have the last 4 layers being copies of the same white noise
    _test(sys.argv[1])

"""
Routines for implementing the image subtraction step in PSF wing removal.

Functions
---------
pltshow
    Helper to determine where to save a plot.
get_wcs
    Extracts the World Coordinate System from a cached file.
run_imsubtract
    Main workflow for image subtraction step.

"""

import gc
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import asdf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.wcsapi import SlicedLowLevelWCS

# import from furry_parakeet
from furry_parakeet import (
    pyimcom_croutines,
)
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.signal import fftconvolve
from scipy.signal.windows import tukey
from scipy.special import eval_legendre

# local imports
from ..config import Config, Settings
from ..diagnostics.context_figure import ReportFigContext
from ..utils import compareutils
from ..wcsutil import (
    PyIMCOM_WCS,
    get_pix_area,
)


def fftconvolve_multi(in1, in2, out, mode="full", nb=4, workers=None, verbose=False):
    """
    Convolve two N-dimensional arrays using FFT.

    This is almost a drop-in replacement for ``scipy.signal.fftconvolve``.
    The big difference is that the convolution is directly added to `out`,
    rather than being a return value.

    For the 2D `mode` = "valid" case, this splits up the data into `nb`
    blocks for the convolution. It is designed to be efficient when `in1` is
    much smaller than `in2`.

    Parameters
    ----------
    in1 : np.ndarray
        First input.
    in2 : np.ndarray
        Second input. Should have the same number of dimensions as `in1`.
    out : np.ndarray
        Location to add to the output image. Must have the right dimensions.
    mode : str, optional
        Mode; options are "full", "valid", and "same" (just as for the
        scipy functions).
    nb : int, optional
        Number of blocks to use.
    workers : int, optional
        Number of workers for the FFTs if requesting parallelism.
    verbose : bool, optional
        Whether to print the intermediate steps.

    Returns
    -------
    None

    """

    t0 = time.time()

    # if we're not using valid, or not in 2D, use standard fftconvolve
    if mode != "valid" or len(np.shape(in1)) != 2:
        out += fftconvolve(in1, in2, mode=mode, workers=workers)
        return

    # Now we know we're 2D and in valid mode. Get shapes
    (s1y, s1x) = np.shape(in1)
    (s2y, s2x) = np.shape(in2)
    Lx = abs(s1x - s2x) + 1
    Ly = abs(s1y - s2y) + 1

    # If in1 is big enough that it will break the indexing
    if s1y >= Ly // nb:
        out += fftconvolve(in1, in2, mode=mode, workers=workers)
        return

    # loop over horizontal bands
    height = (Ly + nb - 1) // nb
    if height <= s1y:
        out += fftconvolve(in1, in2, mode=mode, workers=workers)  # also return if the strip is too narrow
        return
    lenx = next_fast_len(s1x + s2x)
    leny = next_fast_len(s1y + height)
    in1_ = np.zeros((leny, lenx))
    in2_ = np.zeros((leny, lenx))
    in1_[:s1y, :s1x] = in1
    in1_ft = rfft2(in1_, workers=workers)
    del in1_
    for j in range(nb):
        gc.collect()
        ybottom = j * height
        ytop = min((j + 1) * height, Ly)
        dy = ytop - ybottom
        in2_[:, :] = 0.0
        in2_[: dy + s1y - 1, :s2x] = in2[ybottom : ytop + s1y - 1, :]
        in2_ft = rfft2(in2_, workers=workers) * in1_ft
        if verbose:
            print("y =", ybottom, ytop, "of Ly =", Ly)
        out[ybottom:ytop, :] += irfft2(in2_ft, s=(leny, lenx), workers=workers)[
            s1y - 1 : dy + s1y - 1, s1x - 1 : Lx + s1x - 1
        ]
        # B = fftconvolve(in1, in2[ybottom : ytop + s1y - 1, :], mode="valid")
        # print(np.shape(A), np.shape(B))
        # print(np.amax(np.abs(A)), np.amax(np.abs(B)), np.amax(np.abs(A-B)))
        print(f"t = {time.time()-t0:6.3f} s, shape =", (leny, lenx), "ft =", np.shape(in1_ft))
        sys.stdout.flush()

    del in2_, in1_ft, in2_ft
    gc.collect()


def pltshow(plt, display, pars={}):
    """
    Where to save a plot.

    Parameters
    ----------
    plt : matplotlib.pyplot
        The pyplot module to use for plotting.
    display : str or None
        Sends to file (if string), screen (None), or nowhere (if '/dev/null')
    pars : dict, optional
        Parameters for saving the file.
        Must be provided if a file is requested.

    Returns
    -------
    None

    Notes
    -----
    The `pars` dictionary contains the keys:
    * 'type' : str, currently only supports 'window'
    * 'obsid' : int, observation ID
    * 'sca' : int, SCA number
    * 'ix' : int, x block index
    * 'iy' : int, y block index

    """

    if display is None:
        plt.show()
        return

    if display == "/dev/null":
        return

    # if we get here, we need to save the file
    if pars["type"].lower() == "window":
        obsid = pars["obsid"]
        sca = pars["sca"]
        ix = pars["ix"]
        iy = pars["iy"]
        plt.savefig(display + f"_{obsid}_{sca}_{ix:02d}_{iy:02d}.png")


def get_wcs(cachefile):
    """
    Gets the WCS from a cached FITS file.

    If a gwcs is used, finds the attached ASDF file and reads that.

    Parameters
    ----------
    cachefile : str
        Name of the cached file.

    Returns
    -------
    pyimcom.wcsutils.PyIMCOM_WCS
        The World Coordinate System of the cached file.

    """

    with fits.open(cachefile) as hdul:
        if "WCSTYPE" in hdul[1].header and hdul[1].header["WCSTYPE"][:4].lower() == "gwcs":
            with asdf.open(cachefile[:-5] + "_wcs.asdf") as f2:
                return PyIMCOM_WCS(f2["wcs"])
        return PyIMCOM_WCS(hdul["SCIWCS"].header)


def get_wcs_from_infile(infile):
    """
    Gets the "sub-WCS" from the FITS file. Using SlicedLowLevelWCS avoids extra axes
    which create additional complications in the WCS.

    Parameters
    ----------
    infile : str
        Name of the file.

    Returns
    -------
    sub-WCS
        The World Coordinate System of the file with only the necessary axes.

    """
    g = infile[0].header
    block_wcs = SlicedLowLevelWCS(WCS(g), slices=[0, 0, slice(0, g["NAXIS2"]), slice(0, g["NAXIS1"])])

    return block_wcs


def run_imsubtract_single(
    cfgdata,
    scaid,
    obsid,
    path,
    expname,
    display=None,
    local_output=False,
    fft_workers=None,
    wcs_shortcut=True,
):
    """
    Main routine to run imsubtract on a single image.

    Parameters
    ----------
    cfgdata : Config object
        a pyimcom Config object
    scaid: int
        SCA ID for the image to be subtracted. Should be in range 1..18, inclusive.
    obsid: int
        Observation ID for the image to be subtracted.
    path: str
        Path to the directory containing the cached files.
    expname: str
        Name of the cached file (will be of the form `stem_obsid_scaid.fits').
    display : str or None, optional
        Display location for intermediate steps.
    local_output : bool, optional
        Whether to direct the file to local output instead of the cache directory.
        (This will normally be the default False; it is provided only so that if
        more than one user runs tests at the same time, they can use True to avoid
        a collision.)
    fft_workers : int, optional
        Number of workers for the FFTs if requesting parallelism.
    wcs_shortcut : bool, optional
        If set, allows interpolation methods to speed up WCS computations.

    Notes
    -----
    There are several options for `display`:

    * `display` = None : print to screen
    * `display` = '/dev/null' : don't save
    * `display` = any other string : save to ``display+f'_{obsid}_{sca}_{ix:02d}_{iy:02d}.png'``

    """
    info = cfgdata.inlayercache
    block_path = cfgdata.outstem
    ra = cfgdata.ra * (np.pi / 180)  # convert to radians
    dec = cfgdata.dec * (np.pi / 180)  # convert to radians
    lonpole = cfgdata.lonpole * (np.pi / 180)  # convert to radians
    nblock = cfgdata.nblock
    n1 = cfgdata.n1  # number of postage stamps
    n2 = cfgdata.n2  # size of single run
    postage_pad = cfgdata.postage_pad  # postage stamp padding
    dtheta_deg = cfgdata.dtheta
    blocksize_rad = n1 * n2 * dtheta_deg * (np.pi) / 180  # convert to radians

    # get information from Settings
    pix_size = Settings.pixscale_native  # native pixel scale in arcsec
    sca_nside = Settings.sca_nside  # length of sca side, in native pixels

    # inlayercache data --- changed to context manager structure
    with fits.open(path + "/" + expname) as hdul:
        # read in the input image, I
        I_img = np.memmap(
            path + "/" + expname[:-5] + "_data.npy",
            dtype=np.float32,
            mode="w+",
            shape=np.shape(hdul[0].data),
        )
        I_img[:, :, :] = hdul[0].data  # this is I
    # find number of layers
    nlayer = np.shape(I_img)[-3]
    # get wcs information from fits file (or asdf if indicated)
    sca_wcs = get_wcs(path + "/" + expname)
    # results from splitpsf
    # read in the kernel
    with fits.open(f"{info}.psf/psf_{obsid:d}.fits") as hdul2:
        K = np.copy(hdul2[sca + hdul2[0].header["KERSKIP"]].data)
        # get the number of pixels on the axis
        axis_num = K.shape[1]  # kernel pixels
        Ncoeff = K.shape[0]  # number of coefficients

        # get the oversampling factor
        oversamp = hdul2[0].header["OVSAMP"]  # number of kernel pixels / native pixels

    # SCA padding
    I_pad = int(np.ceil(axis_num / 2 / oversamp))  # native pixels
    # define the first index needed for convolution
    first_index = (oversamp + 2 * oversamp * I_pad - axis_num) // 2

    # get the kernel size
    s_in_rad = pix_size * np.pi / (180 * 3600)  # convert arcsec to radians
    ker_size = axis_num / oversamp * s_in_rad  # native pixels

    # start coordinate transformations
    # define pad
    pad = ker_size / 2  # at least half of the kernel size in native pixels
    # convert to x, y, z using wcs coords (center of SCA)
    x, y, z, p = compareutils.getfootprint(sca_wcs, pad)
    v = np.array([x, y, z])

    # convert to x', y', z'
    # define coordinates and transformation matrix
    ex = np.array([np.sin(ra), -np.cos(ra), 0])
    ey = np.array([-np.cos(ra) * np.sin(dec), -np.sin(dec) * np.sin(ra), np.cos(dec)])
    ez = np.array([-np.cos(dec) * np.cos(ra), -np.cos(dec) * np.sin(ra), -np.sin(dec)])
    T = np.array([ex, ey, ez])

    # perform transformation and define individual values
    v_p = np.matmul(T, v)
    x_p = v_p[0]
    y_p = v_p[1]
    z_p = v_p[2]

    # define the rotation matrix, coefficient, and additional vector
    rot = np.array([[-np.cos(lonpole), -np.sin(lonpole)], [np.sin(lonpole), -np.cos(lonpole)]])
    coeff = 2 / (1 - z_p)
    v_convert = np.array([x_p, y_p])

    # convert to eta and xi (block coordinates)
    block_coords = coeff * np.matmul(rot, v_convert)

    # find theta in original coordinates, convert to block coordinates
    theta = (
        2 * np.arctan(np.sqrt(p / (2 - p)))
        + blocksize_rad / np.sqrt(2)
        + np.sqrt(2) * pad
        + ker_size / np.sqrt(2)
    ) * coeff
    theta_block = theta / blocksize_rad

    # add theta to this set of coords
    block_coords = np.append(block_coords, theta)

    # convert the units of this coordinate system to blocks
    block_coords_blocks = block_coords / blocksize_rad

    # find the center of SCA relative to the bottom left of the mosaic
    SCA_coords = block_coords_blocks.copy()
    SCA_coords[:2] += nblock / 2  # take only the xi and eta directions

    # find the blocks the SCA covers
    side = np.arange(nblock) + 0.5
    xx, yy = np.meshgrid(side, side)
    distance = np.hypot(xx - SCA_coords[0], yy - SCA_coords[1])
    in_SCA = np.where(distance <= theta_block)
    block_list = np.stack((in_SCA[1], in_SCA[0]), axis=-1)

    # define the canvas to add interpolated blocks
    # size is SCA+padding on both sides scaled back to kernel pixels
    A = oversamp * (sca_nside + 2 * I_pad)

    skipblocks = set()  # blocks we know we can skip since they turned out to have no overlap
    lrbt_table = {}  # the [left, right, bottom, top] of each block

    # get pixel area map (once)
    area_np = (
        get_pix_area(sca_wcs, region=[-I_pad, sca_nside + I_pad, -I_pad, sca_nside + I_pad])
        / (pix_size * 180 / np.pi) ** 2
    ).astype(np.float32)

    # add for loop over layers (nlayers)
    for n in range(nlayer):
        H_canvas = np.zeros((A, A), dtype=np.float32)
        # define other important quantities for convolution
        Nl = int(np.floor(np.sqrt(Ncoeff + 0.5)))
        KH = np.zeros((A - axis_num + 1, A - axis_num + 1), dtype=np.float32)
        x_canvas = np.linspace(-I_pad - 0.5 + 0.5 / oversamp, sca_nside + I_pad - 0.5 + 0.5 / oversamp, A)
        u_canvas = (x_canvas - (sca_nside - 1) / 2) / (sca_nside / 2)

        # loop over the blocks in the list
        for ix, iy in block_list:
            if (ix, iy) in skipblocks:
                continue
            print("BLOCK: ", ix, iy)
            sys.stdout.flush()
            t0 = time.time()

            # open the block info
            with fits.open(block_path + f"_{ix:02d}_{iy:02d}.fits") as hdul3:
                block_wcs = get_wcs_from_infile(hdul3)
                print(f"+ block: {time.time()-t0:6.2f}")
                sys.stdout.flush()

                # determine the length of one axis of the block
                block_length = hdul3[0].header["NAXIS1"]  # length in output pixels
                overlap = n2 * postage_pad  # size of one overlap region due to postage stamp
                a1 = 2 * (2 * overlap - 1) / (block_length - 1)  # percentage of region to have
                # window function taper
                # the '-1' is due to scipy's convention on alpha that the denominator is the distance from
                # the first to the last point, so 1 less than the length
                window = tukey(block_length, alpha=a1).astype(np.float32)
                # apply window function to block data in both directions
                block = hdul3[0].data[0, n, :, :] * window[:, None] * window[None, :]
                print(f"+ windowed: {time.time()-t0:6.2f}")
                sys.stdout.flush()

            # check the window function
            if display != "/dev/null":
                print("FIG")
                with ReportFigContext(matplotlib, plt):
                    plt.plot(np.arange(len(window)), window, color="indigo")
                    plt.axvline(block_length - 1, c="mediumpurple")
                    plt.axvline(block_length - overlap - 1, c="mediumpurple")
                    plt.axvline(block_length - 2 * overlap - 1, c="mediumpurple")
                    plt.xlim(block_length - 3 * overlap, block_length + overlap)
                    plt.plot(block_length - 2, window[block_length - 2], c="darkmagenta", marker="o")
                    plt.plot(
                        block_length - 2 * overlap,
                        window[block_length - 2 * overlap],
                        c="darkmagenta",
                        marker="o",
                    )
                    plt.plot(
                        block_length - overlap, window[block_length - overlap], c="blueviolet", marker="o"
                    )
                    plt.plot(
                        block_length - overlap - 2,
                        window[block_length - overlap - 2],
                        c="blueviolet",
                        marker="o",
                    )
                    pltshow(plt, display, {"type": "window", "obsid": obsid, "sca": sca, "ix": ix, "iy": iy})

            print(f"+ figure: {time.time()-t0:6.2f}")
            sys.stdout.flush()
            gc.collect()

            if (ix, iy) in lrbt_table:
                # get bounding box if we already have it
                [left, right, bottom, top] = lrbt_table[(ix, iy)]
            else:
                # find the 'Bounding Box' in SCA coordinates
                # create mesh grid for output block
                block_arr = np.arange(block_length)
                x_out, y_out = np.meshgrid(block_arr, block_arr)
                # convert to ra and dec using block wcs
                ra_sca, dec_sca = block_wcs.pixel_to_world_values(x_out, y_out, 0)
                del x_out, y_out

                # convert into SCA coordinates
                x_in, y_in = sca_wcs.all_world2pix(ra_sca, dec_sca, 0)
                del ra_sca, dec_sca

                # get the bounding box from the max and min values
                left = int(np.floor(np.min(x_in)))
                right = int(np.ceil(np.max(x_in)))
                bottom = int(np.floor(np.min(y_in)))
                top = int(np.ceil(np.max(y_in)))
                del x_in, y_in

                # trim bounding box to ensure not extending past SCA padding
                left = np.max([left, -I_pad])
                right = np.min([right, sca_nside - 1 + I_pad])
                bottom = np.max([bottom, -I_pad])
                top = np.min([top, sca_nside - 1 + I_pad])
                lrbt_table[(ix, iy)] = [left, right, bottom, top]
            gc.collect()

            print(f"+ wcsmap: {time.time()-t0:6.2f}")
            sys.stdout.flush()

            # create the bounding box mesh grid, with ovsamp
            # determine side lengths of the box
            width = int(oversamp * (right - left + 1))
            height = int(oversamp * (top - bottom + 1))

            # check if weight, height are positive
            if width <= 0 or height <= 0:
                skipblocks.add((ix, iy))  # can skip this block for the next layer
                continue

            # two options for getting the inverse WCS mapping: one with shortcut, one without
            if wcs_shortcut:
                # create arrays for meshgrid
                x = np.linspace(left - 0.5, right + 0.5, right - left + 2)
                y = np.linspace(bottom - 0.5, top + 0.5, top - bottom + 2)
                bb_x, bb_y = np.meshgrid(x, y)
                # map bounding box from SCA to output block coordinates
                ra_map, dec_map = sca_wcs.all_pix2world(bb_x, bb_y, 0)
                del bb_x, bb_y
                x_bb_temp, y_bb_temp = block_wcs.world_to_pixel_values(ra_map, dec_map, 0)
                del ra_map, dec_map
                x_bb = np.zeros((height, width))
                y_bb = np.zeros((height, width))
                for i in range(oversamp):
                    fi = (i + 0.5) / oversamp
                    x1 = (1 - fi) * x_bb_temp[:, :-1] + fi * x_bb_temp[:, 1:]
                    y1 = (1 - fi) * y_bb_temp[:, :-1] + fi * y_bb_temp[:, 1:]
                    for j in range(oversamp):
                        fj = (j + 0.5) / oversamp
                        x_bb[j::oversamp, i::oversamp] = (1 - fj) * x1[:-1, :] + fj * x1[1:, :]
                        y_bb[j::oversamp, i::oversamp] = (1 - fj) * y1[:-1, :] + fj * y1[1:, :]
                del x_bb_temp, y_bb_temp, x1, y1
            else:
                # create arrays for meshgrid
                x = np.linspace(left - 0.5 + 0.5 / oversamp, right + 0.5 - 0.5 / oversamp, width)
                y = np.linspace(bottom - 0.5 + 0.5 / oversamp, top + 0.5 + 0.5 / oversamp, height)
                bb_x, bb_y = np.meshgrid(x, y)
                # map bounding box from SCA to output block coordinates
                ra_map, dec_map = sca_wcs.all_pix2world(bb_x, bb_y, 0)
                del bb_x, bb_y
                x_bb, y_bb = block_wcs.world_to_pixel_values(ra_map, dec_map, 0)
                del ra_map, dec_map

            print(f"+ inv wcs map: {time.time()-t0:6.2f}")
            sys.stdout.flush()

            # add padding to the block (with window applied)
            block_padded = np.pad(block, 5, mode="constant", constant_values=0)[None, :, :].astype(np.float64)
            x_bb += 5
            y_bb += 5

            # create interpolated version of block
            H = np.zeros((1, np.size(x_bb)))
            pyimcom_croutines.iG4460C(block_padded, x_bb.ravel(), y_bb.ravel(), H)
            # reshape H
            H = H.reshape(x_bb.shape)

            print(f"+ interp: {time.time()-t0:6.2f}")
            sys.stdout.flush()

            # multiply by Jacobian to H
            # get native pixel size (in units of the ideal pixel, [0.11 arcsec]^2 for Roman)
            if wcs_shortcut:
                # previous area call: get_pix_area(sca_wcs, region=[left, right + 1, bottom, top + 1])
                # note that was in steradians, this one is in ideal pixels

                # this should be faster
                native_pix = np.repeat(
                    np.repeat(
                        area_np[I_pad + bottom : I_pad + top + 1, I_pad + left : I_pad + right + 1],
                        oversamp,
                        axis=1,
                    ),
                    oversamp,
                    axis=0,
                )
            else:
                native_pix = (
                    get_pix_area(sca_wcs, region=[left, right + 1, bottom, top + 1], ovsamp=oversamp)
                    / (pix_size * 180 / np.pi) ** 2
                )

            H *= native_pix

            print(f"+ area: {time.time()-t0:6.2f}")
            sys.stdout.flush()

            # add H to H_canvas
            H_canvas[
                oversamp * (bottom + I_pad) : oversamp * (top + 1 + I_pad),
                oversamp * (left + I_pad) : oversamp * (right + 1 + I_pad),
            ] += H

        # some cleanup
        del H, native_pix

        # apply convolution to canvas
        for lu in range(Nl):
            # save first multiplication
            Hlu = H_canvas * eval_legendre(lu, u_canvas).astype(np.float32)[None, :]
            for lv in range(Nl):
                print("Convolve", lu, lv)
                sys.stdout.flush()
                fftconvolve_multi(
                    K[lu + lv * Nl, :, :],
                    Hlu * eval_legendre(lv, u_canvas).astype(np.float32)[:, None],
                    KH,
                    mode="valid",
                    nb=6,
                    workers=fft_workers,
                )

        # subtract from the input image (using less memory)
        I_img[n, :, :] -= KH[first_index:-first_index:oversamp, first_index:-first_index:oversamp]

    # write output file for each exposure
    fname = f"{info}_{obsid:08d}_{scaid:02d}_subI.fits"
    if local_output:
        fname = f"{obsid:08d}_{scaid:02d}_subI.fits"
    print("saving >>", fname)
    sys.stdout.flush()

    # this version copies HDU #1 (which contains the WCS)
    with fits.open(path + "/" + expname) as f_in:
        fits.HDUList([fits.PrimaryHDU(data=I_img), f_in[1]]).writeto(fname, overwrite=True)


def run_imsubtract_all(cfg_file, workers=4, max_imgs=None, display=None):
    """
    Main routine to run imsubtract on all images in the cache.

    Parameters
    ----------
    cfg_file: str
        Path to the config file.
    workers: int, optional
        Number of workers to use for parallel processing. Default is 4.
    max_imgs: int, optional
        If provided, does computations for a maximum number of SCAs. Most users will
        want the default of None; this is provided mainly for testing.
    display: str or None, optional
        Display location for intermediate steps. Default is None.
    """
    # load the file using Config and get information
    cfgdata = Config(cfg_file)

    cacheinfo = cfgdata.inlayercache

    # separate the path from the inlayercache info
    m = re.search(r"^(.*)\/(.*)", cacheinfo)
    if m:
        path = m.group(1)
        stem = m.group(2)

    # create empty list of exposures
    exps = []

    # find all the fits files and add them to the list
    for _, _, files in os.walk(path):
        for file in files:
            if file.startswith(stem) and file.endswith(".fits") and file[-6].isdigit():
                exps.append(file)

    print("List of exposures:", exps)

    # Run imsubtract on each exposure in parallel using ProcessPoolExecutor
    count = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for exp in exps:
            m2 = re.search(r"(\w*)_0*(\d*)_(\d*).fits", exp)
            if m2:
                obsid = int(m2.group(2))
                scaid = int(m2.group(3))
                futures.append(
                    executor.submit(
                        run_imsubtract_single,
                        cfgdata,
                        scaid,
                        obsid,
                        path,
                        exp,
                        display=display,
                        fft_workers=None,
                    )
                )

        # Wait for all futures to complete
        for future in as_completed(futures):
            count += 1
            if max_imgs is not None and count > max_imgs:
                break

            try:
                future.result()
            except Exception as e:
                print(f"Worker failed with exception {e}", flush=True)


if __name__ == "__main__":
    """Calling program is here.

    python3 -m pyimcom.splitpsf.imsubtract  <config> <sca> [<output images>]
    (uses plt.show() if output stem not specified; output image directory is relative to cache file)

    """

    start = time.time()
    # get the json file
    config_file = sys.argv[1]

    # get the SCA (0 for all of them)
    sca = int(sys.argv[2])
    if sca == 0:
        sca = None

    display = sys.argv[3] if len(sys.argv) > 3 else None

    workers = os.cpu_count()

    run_imsubtract_all(config_file, workers, max_imgs=None, display=display)

    end = time.time()
    elapsed = end - start
    print(f"finished at t = {elapsed:.2f} s")

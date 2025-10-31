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

import os
import re
import sys
import time

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


def fftconvolve_multi(in1, in2, out, mode="full", nb=4, verbose=False):
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
    verbose : bool, optional
        Whether to print the intermediate steps.

    Returns
    -------
    None

    """

    # if we're not using valid, or not in 2D, use standard fftconvolve
    if mode != "valid" or len(np.shape(in1)) != 2:
        out += fftconvolve(in1, in2, mode=mode)
        return

    # Now we know we're 2D and in valid mode. Get shapes
    (s1y, s1x) = np.shape(in1)
    (s2y, s2x) = np.shape(in2)
    Ly = abs(s1y - s2y) + 1

    # If in1 is big enough that it will break the indexing
    if s1y >= Ly // nb:
        out += fftconvolve(in1, in2, mode=mode)
        return

    # loop over horizontal bands
    height = Ly // nb
    for j in range(nb):
        ybottom = j * height
        ytop = min((j + 1) * height, Ly)
        if verbose:
            print("y =", ybottom, ytop, "of Ly =", Ly)
        out[ybottom:ytop, :] += fftconvolve(in1, in2[ybottom : s2y + ytop - Ly, :], mode="valid")


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


def run_imsubtract(config_file, display=None, scanum=None, local_output=False, max_img=None):
    """
    Main routine to run imsubtract.

    Parameters
    ----------
    config_file : str
        Location of a configuration file.
    display : str or None, optional
        Display location for intermediate steps.
    scanum : int or None, optional
        If not None, only run this SCA. Should be in range 1..18, inclusive.
        (Mostly used for parallelization.)
    local_output : bool, optional
        Whether to direct the file to local output instead of the cache directory.
        (This will normally be the default False; it is provided only so that if
        more than one user runs tests at the same time, they can use True to avoid
        a collision.)
    max_img : int, optional
        If provided, does computations for a maximum number of SCAs. Most users will
        want the default of None; this is provided mainly for testing.

    Notes
    -----
    There are several options for `display`:

    * `display` = None : print to screen
    * `display` = '/dev/null' : don't save
    * `display` = any other string : save to ``display+f'_{obsid}_{sca}_{ix:02d}_{iy:02d}.png'``

    """

    # load the file using Config and get information
    cfgdata = Config(config_file)

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

    # separate the path from the inlayercache info
    m = re.search(r"^(.*)\/(.*)", info)
    if m:
        path = m.group(1)
        exp = m.group(2)

    # create empty list of exposures
    exps = []

    # find all the fits files and add them to the list
    for _, _, files in os.walk(path):
        for file in files:
            if file.startswith(exp) and file.endswith(".fits") and file[-6].isdigit():
                exps.append(file)
    print("list of files:", exps)

    # loop over the list of observation pair files (for each SCA)
    count = 0
    for exp in exps:
        # get SCA and obsid
        m2 = re.search(r"(\w*)_0*(\d*)_(\d*).fits", exp)
        if m2:
            obsid = int(m2.group(2))
            sca = int(m2.group(3))
        if scanum is not None and scanum != sca:
            continue  # only do the given SCA
        print("OBSID: ", obsid, "SCA: ", sca)

        # inlayercache data --- changed to context manager structure
        with fits.open(path + "/" + exp) as hdul:
            # read in the input image, I
            I_img = np.copy(hdul[0].data)  # this is I
        # find number of layers
        nlayer = np.shape(I_img)[-3]
        # get wcs information from fits file (or asdf if indicated)
        sca_wcs = get_wcs(path + "/" + exp)

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

        # add for loop over layers (nlayers)
        for n in range(nlayer):
            H_canvas = np.zeros((A, A))
            # define other important quantities for convolution
            Nl = int(np.floor(np.sqrt(Ncoeff + 0.5)))
            KH = np.zeros((A - axis_num + 1, A - axis_num + 1))
            x_canvas = np.linspace(-I_pad - 0.5 + 0.5 / oversamp, sca_nside + I_pad - 0.5 + 0.5 / oversamp, A)
            u_canvas = (x_canvas - (sca_nside - 1) / 2) / (sca_nside / 2)

            # loop over the blocks in the list
            for ix, iy in block_list:
                print("BLOCK: ", ix, iy)
                sys.stdout.flush()

                # open the block info
                with fits.open(block_path + f"_{ix:02d}_{iy:02d}.fits") as hdul3:
                    block_data = np.copy(hdul3[0].data)
                    block_wcs = get_wcs_from_infile(hdul3)

                # determine the length of one axis of the block
                block_length = block_data.shape[-1]  # length in output pixels
                overlap = n2 * postage_pad  # size of one overlap region due to postage stamp
                a1 = 2 * (2 * overlap - 1) / (block_length - 1)  # percentage of region to have
                # window function taper
                # the '-1' is due to scipy's convention on alpha that the denominator is the distance from the
                # first to the last point, so 1 less than the length
                window = tukey(block_length, alpha=a1)
                # apply window function to block data in both directions
                block = block_data[0, n, :, :] * window[:, None] * window[None, :]

                # check the window function
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
                print(
                    window[block_length - 1],
                    window[block_length - 2 * overlap],
                    window[block_length - 1] + window[block_length - 2 * overlap],
                )
                print(
                    window[block_length - overlap],
                    window[block_length - overlap - 1],
                    window[block_length - overlap] + window[block_length - overlap - 1],
                )

                # find the 'Bounding Box' in SCA coordinates
                # create mesh grid for output block
                block_arr = np.arange(block_length)
                x_out, y_out = np.meshgrid(block_arr, block_arr)
                # convert to ra and dec using block wcs
                ra_sca, dec_sca = block_wcs.pixel_to_world_values(x_out, y_out, 0)

                # convert into SCA coordinates
                x_in, y_in = sca_wcs.all_world2pix(ra_sca, dec_sca, 0)

                # get the bounding box from the max and min values
                left = int(np.floor(np.min(x_in)))
                right = int(np.ceil(np.max(x_in)))
                bottom = int(np.floor(np.min(y_in)))
                top = int(np.ceil(np.max(y_in)))

                # trim bounding box to ensure not extending past SCA padding
                left = np.max([left, -I_pad])
                right = np.min([right, sca_nside - 1 + I_pad])
                bottom = np.max([bottom, -I_pad])
                top = np.min([top, sca_nside - 1 + I_pad])

                # create the bounding box mesh grid, with ovsamp
                # determine side lengths of the box
                width = int(oversamp * (right - left + 1))
                height = int(oversamp * (top - bottom + 1))

                # check if weight, height are positive
                if width <= 0 or height <= 0:
                    continue

                # create arrays for meshgrid
                x = np.linspace(left - 0.5 + 0.5 / oversamp, right + 0.5 - 0.5 / oversamp, width)
                y = np.linspace(bottom - 0.5 + 0.5 / oversamp, top + 0.5 + 0.5 / oversamp, height)
                bb_x, bb_y = np.meshgrid(x, y)

                # map bounding box from SCA to output block coordinates
                ra_map, dec_map = sca_wcs.all_pix2world(bb_x, bb_y, 0)
                x_bb, y_bb = block_wcs.world_to_pixel_values(ra_map, dec_map, 0)

                # add padding to the block (with window applied)
                block_padded = np.pad(block, 5, mode="constant", constant_values=0)[None, :, :]
                x_bb += 5
                y_bb += 5

                # create interpolated version of block
                H = np.zeros((1, np.size(x_bb)))
                pyimcom_croutines.iD5512C(block_padded, x_bb.ravel(), y_bb.ravel(), H)
                # reshape H
                H = H.reshape(x_bb.shape)

                # multiply by Jacobian to H
                # get native pixel size
                native_pix = get_pix_area(sca_wcs, region=[left, right + 1, bottom, top + 1], ovsamp=oversamp)

                H *= native_pix / (pix_size * 180 / np.pi) ** 2

                # add H to H_canvas
                H_canvas[
                    oversamp * (bottom + I_pad) : oversamp * (top + 1 + I_pad),
                    oversamp * (left + I_pad) : oversamp * (right + 1 + I_pad),
                ] += H

            # apply convolution to canvas
            for lu in range(Nl):
                for lv in range(Nl):
                    sys.stdout.flush()
                    fftconvolve_multi(
                        K[lu + lv * Nl, :, :],
                        H_canvas
                        * eval_legendre(lu, u_canvas)[None, :]
                        * eval_legendre(lv, u_canvas)[:, None],
                        KH,
                        mode="valid",
                    )

            # subtract from the input image (using less memory)
            I_img[n, :, :] -= KH[first_index:-first_index:oversamp, first_index:-first_index:oversamp]

        # save outside of the layer for loop
        # write output file for each exposure
        fname = f"{info}_{obsid:08d}_{sca:02d}_subI.fits"
        if local_output:
            fname = f"{obsid:08d}_{sca:02d}_subI.fits"
        print("saving >>", fname)
        sys.stdout.flush()
        fits.PrimaryHDU(data=I_img).writeto(fname, overwrite=True)

        # exit if we've specified a maximum number of SCAs
        count += 1
        if max_img is not None and count == max_img:
            return


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

    display = "/dev/null"
    if len(sys.argv) > 3:
        display = sys.argv[3]
    run_imsubtract(config_file, display=display, scanum=sca)  # , max_img=1)

    end = time.time()
    elapsed = end - start
    print(f"Execution time: {elapsed:.4f} seconds.")

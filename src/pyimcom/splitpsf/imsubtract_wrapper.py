import gc
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

# import from furry_parakeet
from furry_parakeet import (
    pyimcom_croutines,
)
from scipy.signal.windows import tukey
from scipy.special import eval_legendre

from ..config import Config, Settings
from ..diagnostics.context_figure import ReportFigContext
from ..utils import compareutils
from ..wcsutil import get_pix_area

# local imports
from .imsubtract import fftconvolve_multi, get_wcs, get_wcs_from_infile, pltshow


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
    max_layers=None,
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
    max_layers : int, optional
        Maximum number of layers to process. (For testing; default is None, which means no limit.)

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
        K = np.copy(hdul2[scaid + hdul2[0].header["KERSKIP"]].data)
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

    # New memory maps
    H_canvas = np.memmap(
        path + "/" + expname[:-5] + "_hcanvas.npy",
        dtype=np.float32,
        mode="w+",
        shape=(A, A),
    )
    KH = np.memmap(
        path + "/" + expname[:-5] + "_kh.npy",
        dtype=np.float32,
        mode="w+",
        shape=(A - axis_num + 1, A - axis_num + 1),
    )

    # add for loop over layers (nlayers)
    for n in range(nlayer) if max_layers is None else range(min(nlayer, max_layers)):
        print(f"Observation {obsid}, SCA {scaid}")
        print(f"Layer {n+1}", flush=True)
        H_canvas[:, :] = 0.0
        # define other important quantities for convolution
        Nl = int(np.floor(np.sqrt(Ncoeff + 0.5)))
        KH[:, :] = 0.0
        x_canvas = np.linspace(-I_pad - 0.5 + 0.5 / oversamp, sca_nside + I_pad - 0.5 + 0.5 / oversamp, A)
        u_canvas = (x_canvas - (sca_nside - 1) / 2) / (sca_nside / 2)

        # These will be overwritten if the block is not skipped
        H = None

        # loop over the blocks in the list
        block_count = 0
        for ix, iy in block_list:
            if (ix, iy) in skipblocks:
                continue

            if max_layers is not None and block_count > max_layers:  # Max blocks = 5 when testing
                print(f"Reached max_blocks={max_layers}, stopping early for testing.")
                break

            print("BLOCK: ", ix, iy)
            print(f"Block count: {block_count}/{max_layers if max_layers is not None else len(block_list)}")
            t0 = time.time()

            # open the block info
            try:
                with fits.open(block_path + f"_{ix:02d}_{iy:02d}.fits") as hdul3:
                    block_wcs = get_wcs_from_infile(hdul3)
                    print(f"+ block: {time.time()-t0:6.2f}")

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
            except FileNotFoundError:
                print(f"Block file for block ({ix}, {iy}) not found. Skipping this block.")
                skipblocks.add((ix, iy))
                continue

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
                    pltshow(
                        plt, display, {"type": "window", "obsid": obsid, "sca": scaid, "ix": ix, "iy": iy}
                    )

            print(f"+ figure: {time.time()-t0:6.2f}")
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

            # multiply by Jacobian to H
            # get native pixel size (in units of the ideal pixel, [0.11 arcsec]^2 for Roman)
            if wcs_shortcut:
                # previous area call: get_pix_area(sca_wcs, region=[left, right + 1, bottom, top + 1])
                # note that was in steradians, this one is in ideal pixels

                # this should be faster
                for j2 in range(oversamp):
                    for i2 in range(oversamp):
                        H[j2::oversamp, i2::oversamp] *= area_np[
                            I_pad + bottom : I_pad + top + 1, I_pad + left : I_pad + right + 1
                        ]
            else:
                H *= (
                    get_pix_area(sca_wcs, region=[left, right + 1, bottom, top + 1], ovsamp=oversamp)
                    / (pix_size * 180 / np.pi) ** 2
                )

            print(f"+ area: {time.time()-t0:6.2f}")

            # add H to H_canvas
            H_canvas[
                oversamp * (bottom + I_pad) : oversamp * (top + 1 + I_pad),
                oversamp * (left + I_pad) : oversamp * (right + 1 + I_pad),
            ] += H

            block_count += 1

        # some cleanup
        del H

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
        print(f"Subtracted layer {n+1}/{nlayer}, t = {time.time()-t0:6.2f}", flush=True)

    # not needed anymore
    del KH
    os.remove(path + "/" + expname[:-5] + "_kh.npy")
    del H_canvas
    os.remove(path + "/" + expname[:-5] + "_hcanvas.npy")

    # write output file for each exposure
    fname = f"{info}_{obsid:08d}_{scaid:02d}_subI.fits"
    if local_output:
        fname = os.path.join(cfgdata.tempfile, f"{obsid:08d}_{scaid:02d}_subI.fits")
    print("saving >>", fname)
    sys.stdout.flush()

    # this version copies HDU #1 (which contains the WCS)
    with fits.open(path + "/" + expname) as f_in:
        fits.HDUList([fits.PrimaryHDU(data=I_img), f_in[1]]).writeto(fname, overwrite=True)
    os.remove(path + "/" + expname[:-5] + "_data.npy")


def run_imsubtract_all(cfg_file, workers=4, max_imgs=None, display=None, local_output=False):
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
    local_output: bool, optional
        Whether to direct the file to local output instead of the cache directory.
    """
    # Additional imports
    import multiprocessing as mp
    import traceback

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

    # print("List of exposures:", exps)

    # Run imsubtract on each exposure in parallel using ProcessPoolExecutor
    count = 0
    start_method = "forkserver" if os.name.lower() == "posix" else "spawn"
    ctx = mp.get_context(start_method)
    nfail = 0

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = []
        for exp in exps:
            if max_imgs is not None and count > max_imgs:
                break
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
                        local_output=local_output,
                        max_layers=max_imgs,
                    )
                )
                count += 1

        # Wait for all futures to complete
        for future in as_completed(futures):
            try:
                future.result()
                print(f"Completed {count}/{len(futures)}", flush=True)

            except Exception as e:
                nfail += 1
                print(f"Worker failed with exception {e}", flush=True)
                traceback.print_exc()

    if nfail > 0:
        print(f"{nfail}/{len(futures)} instances of run_imsubtract_single failed.", flush=True)

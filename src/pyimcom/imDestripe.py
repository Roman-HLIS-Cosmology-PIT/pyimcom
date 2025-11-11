"""
Program to remove correlated noise stripes from RST images.

Classes
-------
Sca_img
     Class defining an SCA image object
Parameters
        Class holding the destriping parameters for a given mosaic
Cost_models
        Class holding the cost function models. Currently only quadratic is supported

Functions
---------
write_to_file
    Function to write some text to an output file
save_fits
    Save a 2D image to a FITS file with locking, retries, and atomic rename.
apply_object_mask
    Apply a bright object mask to an image.
quadratic
    Quadratic cost function f(x) = x^2
absolute
    Absolute cost function f(x) = |x|
huber_loss
    Huber loss cost function
quad_prime
    Derivative of quadratic cost function f'(x) = 2x
abs_prime
    Derivative of absolute cost function f'(x) = sign(x)
huber_prime
    Derivative of Huber loss cost function
get_scas
    Function to get a list of all SCA images and their WCSs for this mosaic
interpolate_image_bilinear
    Interpolate values from a "reference" SCA image onto a "target" SCA coordinate grid
transpose_interpolate
    Interpolate backwards from image_A to image_B space.
transpose_par
    Sum up the values of an image across rows
get_effective_gain
    retrieve the effective gain and n_eff of the image. valid only for already-interpolated images
get_ids
    Take an SCA label and parse it out to get the Obsid and SCA id strings.
save_snapshot
    Save restart state to pickle file.
get_neighbors
    Get the neighboring SCAs for each SCA in the mosaic
residual_function
    Calculate the residual image, = grad(epsilon)
residual_function_single
    Helper function to calculate residuals for a single SCA
cost_function_single
    Helper function to calculate cost for a single SCA
linear_search_general
    Perform a linear search to find the optimal step size alpha along a given direction
linear_search_quadratic
    Calculate optimal step size alpha along a given direction, for quadratic cost function
conjugate_gradient
    Perform the conjugate gradient algorithm to minimize the cost function
"""

import copy
import cProfile
import csv
import glob
import io
import os
import pickle
import pstats
import random
import re
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import numpy as np
import asdf
from astropy import wcs
from astropy.io import fits
from config import Config, Settings
from filelock import FileLock, Timeout
from memory_profiler import memory_usage
from scipy.ndimage import binary_dilation
from utils import compareutils
from wcsutil import PyIMCOM_WCS
try:
    import furry_parakeet.pyimcom_croutines as pyimcom_croutines
except ImportError:
    import pyimcom_croutines

#from Config.Settings import RomanFilters as filters
filters = Settings.RomanFilters
t0_global = time.time()  # after imports

# Module settings
testing = True
use_output_float = np.float32
tempdir = str(os.environ["TMPDIR"]) if "TMPDIR" in os.environ else os.getenv("TMPDIR") + "/"

# For test outputs: set sca=0 to not produce test outputs.
img_full_output = {"obsid": 670, "scaid": 10}

class Cost_models:
    """
    Class holding the cost function models. This is a dictionary of functions
    """

    def __init__(self, cfg):
        models = {
            "quadratic": (quadratic, quad_prime),
            "absolute": (absolute, abs_prime),
            "huber_loss": (huber_loss, huber_prime),
        }

        self.model = cfg.cost_model

        if self.model == "huber_loss":
            self.thresh = cfg.hub_thresh
            write_to_file(f"Cost model is Huber Loss with threshold: {self.thresh}")
        else:
            self.thresh = None

        self.f, self.f_prime = models[self.model]


class Sca_img:
    """
    Class defining an SCA image object.

    Parameters
    --------
        scaid :  Str
            the SCA id
        obsid : Str
            the observation id
        cfg : Config object
            built from the configuration file
        interpolated : Bool
            True if you want the interpolated version of this SCA and not the original. Default False
        add_objmask : Bool
            True if you want to apply the permanent pixel mask and a bright object mask. Default True

    Attributes
    --------
        image : 2D np array
            the SCA image (4088x4088)
        shape : Tuple
            the shape of the image
        w : WCS object
            the astropy.wcs object associated with this SCA
        obsid : Str
            observation ID of this SCA image
        scaid : Str
            SCA ID (position on focal plane) of this SCA image
        mask : 2D np array
            The full pixel mask that is used on this image. Is correct only after applying masks to image
        g_eff : 2D np array
            Effective gain in each pixel of the image
        params_subtracted : Bool
            True if parameters have been subtracted from this image.
        cfg : Config object
            the configuration object passed in at initialization

    Methods
    --------
        apply_noise
            Apply the appropriate lab noise frame to the SCA image
        apply_permanent_mask
            Apply the SCA permanent pixel mask to the image
        apply_asdf_mask
            Apply the SCA ASDF file mask to the image
        apply_all_mask
            Apply the full SCA mask to the image
        subtract_parameters
            Subtract a given set of parameters from self.image; updates self.image, self.params_subtracted
        get_coordinates
            Create an array of ra, dec coords for the image
        make_interpolated
            Construct a version of this SCA interpolated from other, overlapping ones.
            Writes the interpolated image out to the disk, to be read/used later
    """

    def __init__(self, obsid, scaid, cfg, tempdir=tempdir, interpolated=False,
                 add_objmask=True, indata_type = 'fits'):

        if interpolated:
            file = fits.open(
                tempdir + "interpolations/" + obsid + "_" + scaid + "_interp.fits", memmap=True
            )
            image_hdu = "PRIMARY"
        else:
            if indata_type=="fits":
                file = fits.open(
                    cfg.ds_obsfile + filters[cfg.use_filter] + "_" + obsid + "_" + scaid + ".fits", memmap=True
                )
                image_hdu = "SCI"

            elif indata_type=="asdf":
                file = asdf.open(
                    cfg.ds_obsfile + filters[cfg.use_filter] + "_" + obsid + "_" + scaid + ".asdf", memmap=True
                )
                
        
        if indata_type=="fits":
            self.w = wcs.WCS(file[image_hdu].header)
            self.image = np.copy(file[image_hdu].data).astype(np.float64)
            self.header = file[image_hdu].header

        elif indata_type=="asdf":
            self.w = PyIMCOM_WCS(file["roman"]["meta"]["wcs"])
            self.image = file["roman"]["data"]
            self.header = None
        
        self.shape = np.shape(self.image)
        file.close()

        self.obsid = obsid
        self.scaid = scaid
        self.mask = np.ones(self.shape, dtype=bool)
        self.params_subtracted = False
        self.cfg = cfg

        # Calculate effecive gain
        if cfg.gaindir == False:
            if not os.path.isfile(tempdir + obsid + "_" + scaid + "_geff.dat"):
                g0 = time.time()
                g_eff = np.memmap(
                    tempdir + obsid + "_" + scaid + "_geff.dat", dtype="float64", mode="w+", shape=self.shape
                )
                ra, dec = self.get_coordinates(pad=2.0)
                ra = ra.reshape((4090, 4090))
                dec = dec.reshape((4090, 4090))
                derivs = np.array(
                    (
                        (ra[1:-1, 2:] - ra[1:-1, :-2]) / 2,
                        (ra[2:, 1:-1] - ra[:-2, 1:-1]) / 2,
                        (dec[1:-1, 2:] - dec[1:-1, :-2]) / 2,
                        (dec[2:, 1:-1] - dec[:-2, 1:-1]) / 2,
                    )
                )
                derivs_px = np.reshape(np.transpose(derivs), (4088**2, 2, 2))
                det_mat = np.reshape(np.linalg.det(derivs_px), (4088, 4088))
                g_eff[:, :] = 1 / (np.abs(det_mat) * np.cos(np.deg2rad(dec[1:4089, 1:4089])))
                g_eff.flush()
                del g_eff

            self.g_eff = np.memmap(
                tempdir + obsid + "_" + scaid + "_geff.dat", dtype="float64", mode="r", shape=self.shape
            )
        else:
            # PLACEHOLDER for reading in real flat fields as gain
            # Needs to be adapted once actual file format is known
            g_eff_file = asdf.open(cfg.gaindir + cfg.use_filter + "_geff.fits", memmap=True)
            self.g_eff = g_eff_file[int(scaid) - 1].data.astype(np.float64)
            g_eff_file.close()

        # Add a noise frame if specified in config file
        if cfg.ds_noisefile is not False:
            self.apply_noise()

        if add_objmask:
            _, object_mask = apply_object_mask(self.image)
            self.apply_asdf_mask()
            self.mask *= np.logical_not(
                object_mask
            )  # self.mask = True for good pixels, so set object_mask'ed pixels to False
            if not os.path.exists(cfg.ds_outpath + self.obsid + "_" + self.scaid + "_mask.fits") and testing:
                mask_img = self.mask.astype("uint8")
                save_fits(
                    mask_img,
                    self.obsid + "_" + self.scaid + "_mask",
                    dir=cfg.ds_outpath + "masks/",
                    overwrite=True,
                )

    def apply_noise(self):
        """
        Add detector noise to self.image
        """
        noiseframe = (
            np.copy(fits.open(self.cfg.ds_noisefile + self.obsid + "_" + self.scaid + ".fits")["PRIMARY"].data)
            * 1.458
            * 50
        )  # times gain and N_frames
        self.image += noiseframe[4:4092, 4:4092]
        filename = self.obsid + "_" + self.scaid + "_noise"

        if not os.path.exists(test_image_dir + filename + ".fits"):
            save_fits(self.image, filename, dir=test_image_dir, overwrite=True)

    def apply_permanent_mask(self):
        """
        Apply permanent pixel mask. Updates self.image and self.mask
        """
        pm = fits.open(self.cfg.permanent_mask)[0].data[int(self.scaid) - 1].astype(bool)
        self.image *= ~pm
        self.mask *= ~pm
    
    def apply_asdf_mask(self):
        """
        Apply ASDF file mask. Updates self.image and self.mask
        """
        mask = fits.open(self.cfg.ds_obsfile + filters[self.cfg.use_filter] + "_" + self.obsid 
                         + "_" + self.scaid + "_mask.fits", memmap=True)[1].data.astype(bool)
        self.image *= ~mask
        self.mask *= ~mask 
        

    def get_permanent_mask(self):
        """
        Apply permanent pixel mask.
        Updates self.image and self.mask
        """
        pm = fits.open(self.cfg.permanent_mask)[0].data[int(self.scaid) - 1]
        pm_array = np.copy(pm)
        return pm_array

    def apply_all_mask(self):
        """
        Apply permanent pixel mask.
        Updates self.image in-place
        """
        self.image *= self.mask

    def subtract_parameters(self, p, j):
        """
        Subtract a set of parameters from the SCA image.
        Updates self.image and self.params_subtracted

        Parameters
        --------
        p : Parameters object
            containing params of current iteration
        j : int
            the index of the SCA image into all_scas list

        """
        if self.params_subtracted:
            write_to_file("WARNING: PARAMS HAVE ALREADY BEEN SUBTRACTED. ABORTING NOW")
            sys.exit()

        params_image = p.forward_par(j)  # Make destriping params into an image
        self.image = self.image - params_image  # Update I_A.image to have the params image subtracted off
        self.params_subtracted = True

    def get_coordinates(self, pad=0.0):
        """
        Create an array of ra, dec coords for the image

        Parameters
        --------
        pad : Float64
            N pixels of padding to add to the array. Default 0.0

        Returns
        --------
        ra, dec; 1D np.arrays of length (height*width)
            1D arrays of ra, dec coordinates for each pixel in the image
        """
        wcs = self.w
        h = self.shape[0] + pad
        w = self.shape[1] + pad
        x_i, y_i = np.meshgrid(np.arange(h), np.arange(w), indexing="xy")
        x_i -= pad / 2.0
        y_i -= pad / 2.0
        x_flat = x_i.flatten()
        y_flat = y_i.flatten()
        ra, dec = wcs.all_pix2world(x_flat, y_flat, 0)  # 0 is for the first frame (1-indexed)
        return ra, dec


    def make_interpolated(self, ind, scalist, neighbors, tempdir=tempdir, params=None, N_eff_min=0.5):
        """
        Construct a version of this SCA interpolated from other, overlapping ones.
        Writes the interpolated image out to the disk, to be read/used later
        The N_eff_min parameter requires some minimum effective coverage, otherwise masks that pixel.

        Parameters
        --------
        ind : int
            index of this SCA in all_scas list
        scalist : List of Str
            the list of all SCAs in this mosaic
        neighbors : Dict
            dictionary where keys are SCA indices and values are lists of indices of overlapping SCAs
        params : Parameters object
             parameters to be subtracted from contributing SCAs; default Nnoe
        N_eff_min : float
             Effective coverage needed for a pixel to contribute to the interpolation

        """
        this_interp = np.zeros(self.shape)

        if not os.path.isfile(tempdir + self.obsid + "_" + self.scaid + "_Neff.dat"):
            N_eff = np.memmap(
                tempdir + self.obsid + "_" + self.scaid + "_Neff.dat",
                dtype="float32",
                mode="w+",
                shape=self.shape,
            )
            make_Neff = True
        else:
            N_eff = np.memmap(
                tempdir + self.obsid + "_" + self.scaid + "_Neff.dat",
                dtype="float32",
                mode="r",
                shape=self.shape,
            )
            make_Neff = False

        t_a_start = time.time()
        sys.stdout.flush()

        N_BinA = 0

        sca_b_list = neighbors[ind]
        
        for k in sca_b_list:
            sca_b = scalist[k]
            obsid_B, scaid_B = get_ids(sca_b)

            N_BinA += 1
            I_B = Sca_img(obsid_B, scaid_B, self.cfg)  # Initialize image B

            if params:
                I_B.subtract_parameters(params, k)

            I_B.apply_all_mask()  # now I_B is masked
            B_interp = np.zeros_like(self.image)
            interpolate_image_bilinear(I_B, self, B_interp)

            if make_Neff:
                B_mask_interp = np.zeros_like(self.image)
                interpolate_image_bilinear(
                    I_B, self, B_mask_interp, mask=I_B.mask
                )  # interpolate B pixel mask onto A grid

            if img_full_output["scaid"]!=0 and testing:
                if obsid_B == str(img_full_output["obsid"]) and scaid_B == str(img_full_output["scaid"]) and make_Neff:  # only do this once
                    save_fits(
                            B_interp, f'{img_full_output["obsid"]}_{img_full_output["scaid"]}_B_{self.obsid}_{self.scaid}_interp', dir=test_image_dir
                        )

                if self.obsid == str(img_full_output["obsid"]) and self.scaid == str(img_full_output["scaid"]) and make_Neff:
                    save_fits(B_interp, f'{img_full_output["obsid"]}_{img_full_output["scaid"]}_A_{obsid_B}_{scaid_B}_interp', dir=test_image_dir)

            this_interp += B_interp
            if make_Neff:
                N_eff += B_mask_interp

        write_to_file(
            f"Interpolation of {self.obsid}_{self.scaid} done. Number of contributing SCAs: {N_BinA}"
        )
        new_mask = N_eff > N_eff_min
        this_interp = np.where(new_mask, this_interp / np.where(new_mask, N_eff, N_eff_min), 0)

        header = self.w.to_header(relax=True)
        this_interp = np.divide(this_interp, self.g_eff)

        # KL move these to tmp?
        save_fits(
            this_interp,
            self.obsid + "_" + self.scaid + "_interp",
            dir=tempdir + "interpolations/",
            header=header,
        )

        if make_Neff:
            N_eff.flush()
        del N_eff
        return this_interp, new_mask


class Parameters:
    """
    Class holding the parameters for a given mosaic. This can be the destriping parameters, or additional
    parameters that need to be the same shape and have the same methods

    Parameters
    ----------
        cfg : Config object
            built from the configuration file
        scalist : list of Strings
            the list of SCAs in this mosaic

    Attributes
    ----------
        model : Str
            Which destriping model to use, which then specifies the number of parameters per row.
            Must be a key of the model_params dict
        n_rows : Int
            Number of rows in the image, or number of rows to fit ds model over
        params_per_row : Int
            Number of parameters per row, set by model_params[model]
        params : 2D np array
            The actual array of parameters.
        current_shape : Str
            The current shape (1D or 2D) of SCA params
        scalist : list of Strings
            the list of SCAs in this mosaic, format: filter_obsid_scaid

    Methods
    -------
        params_2_images
            Reshape params into a 2D array, with one row per SCA
        forward_par
            Reshape one row of params array (one SCA) into a 2D array by projection along rows

    """

    def __init__(self, cfg, scalist=[]):
        model_params = {"constant": 1, "linear": 2}
        if cfg.ds_model not in model_params:
            raise ValueError(f"Model {cfg.ds_model} not in model_params dictionary.")
        self.model = cfg.ds_model
        self.n_rows = cfg.ds_rows
        self.params_per_row = model_params[self.model]
        self.params = np.zeros((len(scalist), self.n_rows * self.params_per_row))
        self.current_shape = "2D"
        self.scalist = scalist

    def params_2_images(self):
        """
        Reshape flattened parameters into 2D array with 1 row per sca and n_rows (in image) * params_per_row entries
        """
        self.params = np.reshape(self.params, (len(self.scalist), self.n_rows * self.params_per_row))
        self.current_shape = "2D"

    def forward_par(self, sca_i):
        """
        Takes one SCA row (n_rows) from the params and casts it into 2D (n_rows x n_rows)

        Parameters
        --------
        sca_i : Int
            Index of which SCA to recast into 2D

        Returns
        --------
         2D np.array, the image of SCA_i's parameters
        """
        if self.current_shape != "2D":
            self.params_2_images()
        return np.array(self.params[sca_i, :])[:, np.newaxis] * np.ones((self.n_rows, self.n_rows))


def write_to_file(text, filename="destripe_out.txt"):
    """
    Function to write some text to an output file

    Parameters
    --------
    text : Str
        The text to print
    filename : Str
        Filename to write out to. Default 'destripe_out.txt'
    """

    if not os.path.exists(filename):
        with open(filename, "w+") as f:
            f.write(text + "\n")
    else:
        with open(filename, "a") as f:
            f.write(text + "\n")
    print(text)

def save_fits(image, filename, dir=None, overwrite=True, s=False, header=None, retries=3):
    """
    Save a 2D image to a FITS file with locking, retries, and atomic rename.

    Parameters
    ----------
    image : np.ndarray
        2D array to write.
    filename : str
        Output filename without extension.
    dir : str
        Directory to save into.
    overwrite : bool
        Whether to overwrite the final target file.
    s : bool
        Whether to print status messages.
    header : fits.Header or None
        Optional FITS header.
    retries : int
        Number of write retry attempts if write fails.
    """
    filepath = os.path.join(dir, filename + ".fits")
    lockpath = filepath + ".lock"
    lock = FileLock(lockpath)

    for attempt in range(retries):
        try:
            with lock.acquire(timeout=30):
                tmp_filepath = filepath + f".{uuid.uuid4().hex}.tmp"
                if header is not None:
                    hdu = fits.PrimaryHDU(image, header=header)
                else:
                    hdu = fits.PrimaryHDU(image)

                hdu.writeto(tmp_filepath, overwrite=overwrite)
                os.replace(tmp_filepath, filepath)  # Atomic move to final path

                if s:
                    write_to_file(f"Array {filename} written out to {filepath}")
                return  # Success

        except Timeout:
            write_to_file(f"Failed to write {filename}; lock acquire timeout")
            return

        except OSError as e:
            if attempt < retries - 1:
                wait_time = 1 + random.random()
                print(
                    f"Write failed for {filepath} (attempt {attempt + 1}): {e}. Retrying in {wait_time:.2f}s..."
                )
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to write {filepath} after {retries} attempts. Last error: {e}")


def apply_object_mask(image, mask=None, threshold_m=0, threshold_c=0.3, inplace=False):
    """
    Apply a bright object mask to an image.

    Parameters
    --------
    image : 2D numpy array
        the image to be masked.
    mask : 2D boolean array, optional
        the pre-existing object mask. Default: None
    threshold_m : float
        factor to multiply with the median for thresholding.
    threshold_c : float
        constant to add to the threshold.
    inplace : Bool
        Whether to modify the input image directly.

    Returns
    --------
    image_out : 2D np.array
        the masked image.
    neighbor_mask : 2D np.array
        the mask applied
    """
    if mask is not None and isinstance(mask, np.ndarray):
        neighbor_mask = mask
    else:
        median_val = np.median(image)
        high_value_mask = image >= threshold_m * median_val + threshold_c
        neighbor_mask = binary_dilation(high_value_mask, structure=np.ones((5, 5), dtype=bool))

    if inplace:
        image[neighbor_mask] = 0
        return image, neighbor_mask
    else:
        image_out = np.where(neighbor_mask, 0, image)
        return image_out, neighbor_mask


def quadratic(x):
    """Quadratic cost function f(x) = x^2"""
    return x**2


def absolute(x):
    """Absolute cost function f(x) = |x|"""
    return np.abs(x)


def huber_loss(x, d):
    """Huber loss cost function"""
    return np.where(np.abs(x) <= d, quadratic(x), d**2 + 2 * d * (np.abs(x) - d))

def quad_prime(x):
    """Derivative of quadratic cost function f'(x) = 2x"""
    return 2 * x


def abs_prime(x):
    """Derivative of absolute cost function f'(x) = sign(x)"""
    return np.sign(x)


def huber_prime(x, d):
    """Derivative of Huber loss cost function"""
    return np.where(np.abs(x) <= d, quad_prime(x), 2 * d * np.sign(x))


def get_scas(filter, obsfile, cfg, indata_type='fits'):
    """
    Function to get a list of all SCA images and their WCSs for this mosaic

    Parameters
    --------
    filter : Str
        which filter to use for this run. Options: Y106, J129, H158, F184, K213
    obsfile : Str
        prefix / path to the SCA images

    Returns
    --------
    all_scas : list of strings
        list of all the SCAs in this mosiac
    all_wcs : list of WCS objects
        the WCS object for each SCA in all_scas (same order)
    """
    n_scas = 0
    all_scas = []
    all_wcs = []
    for f in glob.glob(obsfile + filter + "_*"):
        n_scas += 1
        m = re.search(r"(\w\d+)_(\d+)_(\d+)", f)
        if m:
            this_obsfile = str(m.group(0))
            all_scas.append(this_obsfile)
            if indata_type=="fits":
                this_file = fits.open(f, memmap=True)
                this_wcs = wcs.WCS(this_file["SCI"].header)
                all_wcs.append(this_wcs)
                this_file.close()
            elif indata_type=="asdf":
                this_file = asdf.open(f, memmap=True)
                this_wcs = PyIMCOM_WCS(this_file["roman"]["meta"]["wcs"])
                all_wcs.append(this_wcs)
                this_file.close()
    write_to_file(f"N SCA images in this mosaic: {str(n_scas)}")
    write_to_file("SCA List:", cfg.ds_outpath+"SCA_list.txt")
    for i, s in enumerate(all_scas):
        write_to_file(f"SCA {i}: {s}", cfg.ds_outpath+"SCA_list.txt") 
    return all_scas, all_wcs


def interpolate_image_bilinear(image_B, image_A, interpolated_image, mask=None):
    """
    Interpolate values from a "reference" SCA image onto a "target" SCA coordinate grid
    Uses pyimcom_croutines.bilinear_interpolation(float* image, float* g_eff, int rows, int cols, float* coords,
                                                    int num_coords, float* interpolated_image)

    Parameters
    --------
    image_B : SCA object
        the image to be interpolated
    image_A : SCA object
        the image whose grid you are interpolating B onto
    interpolated_image : 2D np array
        all zeros with shape of Image A.
        Updated in place to be the interpolation of img. B onto A's grid
    mask : 2D np array, optional
        if provided, this mask is interpolated instead of image_B.image
    """

    x_target, y_target, is_in_ref = compareutils.map_sca2sca(image_A.w, image_B.w, pad=0)
    coords = np.column_stack((y_target.ravel(), x_target.ravel()))

    # Verify data just before C call
    rows = int(image_B.shape[0])
    cols = int(image_B.shape[1])
    num_coords = coords.shape[0]

    sys.stdout.flush()
    sys.stderr.flush()
    if mask is not None and isinstance(mask, np.ndarray):
        mask_geff = np.ones_like(image_A.image)
        pyimcom_croutines.bilinear_interpolation(
            mask, mask_geff, rows, cols, coords, num_coords, interpolated_image
        )
    else:
        pyimcom_croutines.bilinear_interpolation(
            image_B.image, image_B.g_eff, rows, cols, coords, num_coords, interpolated_image
        )

    sys.stdout.flush()
    sys.stderr.flush()


def transpose_interpolate(image_A, wcs_A, image_B, original_image):
    """
    Interpolate backwards from image_A to image_B space.
    Uses bilinear_transpose(float* image, int rows, int cols, float* coords, int num_coords, float* original_image)


     Parameters
     --------
     image_A : 2D np array
        the already-interpolated gradient image
     wcs_A : wcs.WCS object
        image A's WCS object
     image_B : SCA object
        the image we're interpolating the gradient back onto
     original_image : 2D np array
        the gradient image re-interpolated into image B space
        Updated in place

    """
    x_target, y_target, is_in_ref = compareutils.map_sca2sca(wcs_A, image_B.w, pad=0)
    coords = np.column_stack((y_target.ravel(), x_target.ravel()))

    rows = int(image_B.shape[0])
    cols = int(image_B.shape[1])
    num_coords = coords.shape[0]

    pyimcom_croutines.bilinear_transpose(image_A, rows, cols, coords, num_coords, original_image)


def transpose_par(I):
    """
    Sum up the values of an image across rows

    Parameters
    --------
    I : 2D np.array
        Input array

     Returns
    --------
       1D np.array, the sum across each row of I
    """
    return np.sum(I, axis=1)


def get_effective_gain(sca, tempdir=tempdir):
    """
    Retrieve the effective gain and n_eff of the image. valid only for already-interpolated images

    Parameters
    --------
    sca : Str
        format like "<prefix>_<obsid>_<scaid>" describing which SCA to get the effective gain for

    Returns
    --------
    g_eff : memmap 2D np.array
        the effective gain in each pixel
    N_eff : memmap 2D np.array
        how many image "B"s contributed to that interpolated image
    """
    m = re.search(r"_(\d+)_(\d+)", sca)
    obsid = m.group(1)
    scaid = m.group(2)
    g_eff = np.memmap(
        tempdir + obsid + "_" + scaid + "_geff.dat", dtype="float64", mode="r", shape=(4088, 4088)
    )
    N_eff = np.memmap(
        tempdir + obsid + "_" + scaid + "_Neff.dat", dtype="float32", mode="r", shape=(4088, 4088)
    )
    return g_eff, N_eff


def get_ids(sca):
    """
    Take an SCA label and parse it out to get the Obsid and SCA id strings.

    Parameters
    --------
    sca : Str
        The sca name from all_scas list, formatted like <obsid>_<scaid>

    Returns
    --------
    obsid : Str
        the observation ID
    scaid : Str
        the SCA ID (position in focal plane)
    """
    m = re.search(r"_(\d+)_(\d+)", sca)
    obsid = m.group(1)
    scaid = m.group(2)
    return obsid, scaid


def save_snapshot(
    p,
    grad,
    epsilon,
    psi,
    direction,
    grad_prev,
    direction_prev,
    cg_model,
    tol,
    thresh,
    norm_0,
    cost_model,
    i,
    restart_file,
):
    """
    Save restart state to pickle file.

    Parameters
    --------
    p : Parameters object
        current parameters
    grad : 2D np array
        current gradient
    epsilon : 2D np array
        current cost
    psi : 3D np array
        current residuals
    direction : 2D np array
        current CG direction
    grad_prev : 2D np array
        previous gradient
    direction_prev : 2D np array
        previous CG direction
    cg_model : Str
        which CG model is being used
    tol : Float
        tolerance for convergence
    thresh : Float
        threshold for Huber loss cost function
    norm_0 : Float
        initial norm of the gradient
    cost_model : Str
        which cost function is being used
    i : Int
        current iteration number
    restart_file : Str
        path to the restart pickle file
    """
    crash_state = {
        "iteration": i,
        "p": p,
        "grad": grad,
        "epsilon": epsilon,
        "direction": direction,
        "grad_prev": grad_prev,
        "psi": psi,
        "direction_prev": direction_prev,
        "cg_model": cg_model,
        "tol": tol,
        "thresh": thresh,
        "norm_0": norm_0,
        "cost_model": cost_model,
    }
    with open(restart_file, "wb") as f:
        pickle.dump(crash_state, f)
    write_to_file(f"Checkpoint saved at iteration {i+1} -> {restart_file}")


def get_neighbors(scalist, ov_mat, overlap_thresh=0.1):
    """
    Get a dictionary of overlapping SCAs for each SCA in the mosaic

    Parameters
    --------
    scalist : List of Str
        the list of all SCAs in this mosaic
    ov_mat : 2D np array
        the overlap matrix for all SCAs in this mosaic
    overlap_thresh : Float
        minimum overlap fraction to consider two SCAs as neighbors; default 0.1

    Returns
    --------
    neighbors : Dict
        dictionary where keys are SCA indices and values are lists of indices of overlapping SCAs
    """
    neighbors = {}
    for k, sca_a in enumerate(scalist):
        neighbors[k] = [j for j in range(len(scalist)) if ov_mat[k, j] >= overlap_thresh and j != k]
    return neighbors


def residual_function(
    psi, f_prime, scalist, wcslist, neighbors, thresh, workers, cfg, extrareturn=False
):
    """
    Calculate the residual image, = grad(epsilon)

    Parameters
    --------
    psi : 3D np array
        the image difference array (I_A - J_A) (N_SCA, 4088, 4088)
    f_prime : Function
        the derivative of the cost function f
        in the future this should be set by default based on what you pass for f
    scalist : List of Str
        the list of all SCAs in this mosaic
    wcslist : List of WCS objects
        the WCS object for each SCA in scalist (same order)
    neighbors : Dict
        dictionary where keys are SCA indices and values are lists of indices of overlapping SCAs
    thresh : Float
        threshold for Huber loss cost function; default None
    workers : Int
        number of parallel workers to use
    cfg : Config object 
        the configuration for this run
    extrareturn : Bool 
        if True, return residual terms 1 and 2 separately; Default False
            in addition to full residuals. returns resids, resids1, resids2

        
    Returns
    --------
    resids : 2D np array
        with one row per SCA and one col per parameter
        """
    
    resids = Parameters(cfg, scalist).params
    if extrareturn:
        resids1 = np.zeros_like(resids)
        resids2 = np.zeros_like(resids)
    write_to_file('Residual calculation started')
    sys.stdout.flush()
    t_r_0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                residual_function_single,
                k,
                sca_a,
                wcslist[k],
                psi[k,:,:],
                f_prime,
                scalist,
                neighbors,
                thresh,
                cfg
            )
            for k, sca_a in enumerate(scalist)
        ]

        for future in as_completed(futures):
            k, term_1, term_2_list = future.result()
            resids[k, :] -= term_1
            if extrareturn:
                resids1[k, :] -= term_1

            # Process term_2 contributions
            for j, term_2 in term_2_list:
                resids[j, :] += term_2
                if extrareturn:
                    resids2[j, :] += term_2

        # KL explicitly give output locations to write_to_file (these should go to the diagnostics directory)
        # could give cfg to write_to_file 
    write_to_file(f"Residuals calculation finished in {(time.time() - t_r_0) / 60} minutes.")
    write_to_file(f"Average time making resids per sca: {(time.time() - t_r_0) / len(scalist)} seconds")
    if extrareturn:
        return resids, resids1, resids2
    return resids

def residual_function_single(k, sca_a, wcs_a, psi_a, f_prime, scalist, neighbors, thresh, cfg):
    """
    Calculate the residual for a single SCA image

    Parameters
    --------
    k : Int
        index of this SCA in scalist
    sca_a : Str
        the SCA label, formatted like <obsid>_<scaid>
    wcs_a : wcs.WCS object
        the WCS object for this SCA
    psi : 3D np array
        the image difference array (I_A - J_A) (N_SCA, 4088, 4088)
    f_prime : Function
        the derivative of the cost function f
    scalist : List of Str
        the list of all SCAs in this mosaic
    neighbors : Dict
        dictionary where keys are SCA indices and values are lists of indices of overlapping SCAs
    thresh : Float
        threshold for Huber loss cost function; default None
    cfg : Config object
        the configuration for this run

    Returns
    --------
    k : Int
        index of this SCA in scalist
    term_1 : 1D np array
        the first residual term for this SCA
    term_2_list : List of tuples
        list of (j, term_2) tuples containing value for term 2
        for each SCA j that overlaps with this one
    """
    t0=time.time()
    # Go and get the WCS object for image A
    obsid_A, scaid_A = get_ids(sca_a)

    # Calculate and then transpose the gradient of I_A-J_A
    gradient_interpolated = f_prime(psi_a, thresh) if thresh is not None else f_prime(psi_a)

    term_1 = transpose_par(gradient_interpolated)

    # Retrieve the effective gain and N_eff to normalize the gradient before transposing back
    g_eff_A, n_eff_A = get_effective_gain(sca_a)

    # Avoid dividing by zero
    valid_mask = n_eff_A != 0
    # denom = g_eff_A * n_eff_A
    gradient_interpolated[valid_mask] = gradient_interpolated[valid_mask] / (
        g_eff_A[valid_mask] * n_eff_A[valid_mask]
    )
    gradient_interpolated[~valid_mask] = 0

    term_2_list = []

    for j in neighbors[k]:
        sca_b = scalist[j]
        obsid_B, scaid_B = get_ids(sca_b)

        I_B = Sca_img(obsid_B, scaid_B, cfg)
        gradient_original = np.zeros(I_B.shape)

        transpose_interpolate(gradient_interpolated, wcs_a, I_B, gradient_original)
        gradient_original *= I_B.g_eff

        term_2 = transpose_par(gradient_original)
        term_2_list.append((j, term_2))

    return k, term_1, term_2_list

def cost_function_single(j, sca_a, p, f, scalist, neighbors, thresh, cfg):
    """
    Calculate the cost function for a single SCA image

    Parameters
    --------
    j : Int
        index of this SCA in scalist
    sca_a : Str
        the SCA label, formatted like <obsid>_<scaid>
    p : Parameters object
        the current parameters for de-striping
    f : Function
        the cost function form
    scalist : List of Str
        the list of all SCAs in this mosaic
    neighbors : Dict
        dictionary where keys are SCA indices and values are lists of indices of overlapping SCAs
    thresh : Float
        threshold for Huber loss cost function; default None
    cfg : Config object
        the configuration for this run

    Returns
    --------
    j : Int
        index of this SCA in scalist
    psi : 2D np array
        the difference image I_A - J_A for this SCA
    local_epsilon : Float
        the cost function value for this SCA
    """
    m = re.search(r"_(\d+)_(\d+)", sca_a)
    obsid_A, scaid_A = m.group(1), m.group(2)

    I_A = Sca_img(obsid_A, scaid_A, cfg)
    I_A.subtract_parameters(p, j)
    I_A.apply_all_mask()

    if img_full_output["scaid"]!=0 and testing:
        if obsid_A == str(img_full_output["obsid"]) and scaid_A == str(img_full_output["scaid"]):
            hdu = fits.PrimaryHDU(I_A.image)
            hdu.writeto(test_image_dir + 
                        f'{img_full_output["obsid"]}_{img_full_output["scaid"]}_I_A_sub_masked.fits', 
                        overwrite=True)

    J_A_image, J_A_mask = I_A.make_interpolated(j, scalist, neighbors, params=p)

    J_A_mask *= I_A.mask

    psi = np.where(J_A_mask, I_A.image - J_A_image, 0).astype("float32")
    result = f(psi, thresh) if thresh is not None else f(psi)
    local_epsilon = np.sum(result)

    if img_full_output["scaid"]!=0 and testing:
        if obsid_A == str(img_full_output["obsid"]) and scaid_A == str(img_full_output["scaid"]):
            hdu = fits.PrimaryHDU(J_A_image * J_A_mask)
            hdu.writeto(test_image_dir + f'{img_full_output["obsid"]}_{img_full_output["scaid"]}_J_A_masked.fits', overwrite=True)

            hdu = fits.PrimaryHDU(psi)
            hdu.writeto(test_image_dir + f'{img_full_output["obsid"]}_{img_full_output["scaid"]}_Psi.fits', overwrite=True)

            write_to_file(f"Sample stats for SCA {img_full_output}:")
            write_to_file(f"Image A mean: {np.mean(I_A.image)}")
            write_to_file(f"Image B mean: {np.mean(J_A_image)}")
            write_to_file(f"Psi mean: {np.mean(psi)}")
            write_to_file(f"f(Psi) mean: {np.mean(result)}")
            write_to_file(f"Local epsilon for SCA {j}: {local_epsilon}")

    return j, psi, local_epsilon

def cost_function(p, f, thresh, workers, scalist, neighbors, cfg, tempdir=tempdir):
    """
    Calculate the cost function with the current de-striping parameters.

    Parameters
    --------
    p : parameters object
        the current parameters for de-striping
    f : st
        keyword for function dictionary options; should also set an f_prime
    thresh : Float
        threshold for Huber loss cost function; default None
    workers : Int
        number of parallel workers to use
    scalist : List of Str
        the list of all SCAs in this mosaic
    neighbors : Dict
        dictionary where keys are SCA indices and values are lists of indices of threshold-overlapping SCAs
    cfg : Config object 
        the configuration for this run
    
    Returns
    --------
    epsilon: int
        the total cost function summed over all images
    psi : D np array
        the difference images I_A-J_A
    """
    write_to_file('Initializing cost function')
    t0_cost = time.time()
    psi = np.memmap(tempdir+'psi_all.dat', dtype=use_output_float, mode='w+', shape=(len(scalist), 4088, 4088))
    psi.fill(0)
    epsilon = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(cost_function_single, j, sca_a, p, f, scalist, neighbors, thresh, cfg)
            for j, sca_a in enumerate(scalist)
        ]

        for future in as_completed(futures):
            j, psi_j, local_eps = future.result()
            psi[j, :, :] = psi_j
            del psi_j
            epsilon += local_eps

    write_to_file(f"Ending cost function. Time elapsed: {(time.time() - t0_cost) / 60} minutes")
    write_to_file(
        f"Average time per cost function iteration: {(time.time() - t0_cost) / len(scalist)} seconds"
    )
    return epsilon, psi


def linear_search_general(
    p,
    direction,
    f,
    f_prime,
    cost_model,
    epsilon_current,
    psi_current,
    grad_current,
    thresh,
    workers,
    scalist,
    wcslist,
    neighbors,
    cfg,
    n_iter=100,
    tol=10**-4,
):
    """
    Linear search via combination bisection and secant methods for parameters that minimize the function
        d_epsilon/d_alpha in the given direction . Note alpha = depth of step in direction

    Parameters
    --------
    p : params object
        the current de-striping parameters
    direction : 2D np array
        direction of conjugate gradient search
    f : function
        cost function form
    f_prime : function
        derivative of cost function form
    cost_model : Str
        which cost function is being used; options: 'quadratic', 'huber_loss'
    epsilon_current : float
        current cost function value
    psi_current : 3D np array
        current difference images (I_A - J_A)
    grad_current : 2D np array
        current gradient AKA current residuals
    thresh : float
        threshold for Huber loss cost function; default None
    workers : Int
        number of parallel workers to use
    scalist : List of Str
        the list of all SCAs in this mosaic
    wcslist : List of WCS objects
        the WCS object for each SCA in scalist (same order)
    neighbors : Dict
        dictionary where keys are SCA indices and values are lists of indices of overlapping SCAs
    cfg : Config object
        the configuration for this run
    n_iter : int
         number of iterations at which to stop searching
    tol : float
         absolute value of d_cost at which to converge

    Returns
    --------
    best_p : parameters object
        containing the best parameters found via search
    best_psi : 3D numpy array
        the difference images made from images with the best_p params subtracted off
    """
    best_epsilon, best_psi = epsilon_current, psi_current
    best_p = copy.deepcopy(p)

    # Simple linear search
    working_p = copy.deepcopy(p)
    max_p = copy.deepcopy(p)
    min_p = copy.deepcopy(p)

    convergence_crit = 99.0
    method = "bisection"

    eta = 0.1
    d_cost_init = np.sum(grad_current * direction)
    d_cost_tol = np.abs(d_cost_init * 1 * 10**-3)

    if cost_model == "quadratic":
        alpha_test = -eta * (np.sum(grad_current * direction)) / (np.sum(direction * direction) + 1e-12)
        if alpha_test <= 0:
            # Not a descent direction — fallback
            alpha_min = -0.9
            alpha_max = 1.0
        else:
            # Curvature-based search window
            alpha_min = alpha_test * 1e-4
            alpha_max = alpha_test * 10

    elif cost_model == "huber_loss":
        alpha_test = 1.0
        alpha_min = 1e-4
        alpha_max = 10

    # Calculate f(alpha_max) and f(alpha_min), which need to be defined for secant update
    write_to_file("### Calculating min and max epsilon and cost")
    max_params = p.params + alpha_max * direction
    max_p.params = max_params
    max_epsilon, max_psi = cost_function(max_p, f, thresh, workers, scalist, neighbors, cfg)
    max_resids = residual_function(
        max_psi, f_prime, scalist, wcslist, neighbors, thresh, workers, cfg
    )
    del max_psi
    d_cost_max = np.sum(max_resids * direction)

    min_params = p.params + alpha_min * direction
    min_p.params = min_params
    min_epsilon, min_psi = cost_function(min_p, f, thresh, workers, scalist, neighbors, cfg)
    min_resids = residual_function(
        min_psi, f_prime, scalist, wcslist, neighbors, thresh, workers, cfg
    )
    del min_psi
    d_cost_min = np.sum(min_resids * direction)

    conv_params = []

    for k in range(1, n_iter):
        t0_ls_iter = time.time()

        if k == 1:
            write_to_file("### Beginning linear search")
            write_to_file(f"LS Direction: {direction}")
            write_to_file(f"Initial params: {p.params}")
            write_to_file(f"Initial epsilon: {best_epsilon}")
            write_to_file(f"Initial d_cost: {d_cost_init}, d_cost tol: {d_cost_tol}")
            write_to_file(f"Initial alpha range (min, test, max): ({alpha_min}, {alpha_test}, {alpha_max})")

        if k == n_iter - 1:
            write_to_file(
                "WARNING: Linear search did not converge!! This is going to break because best_p is not assigned."
            )

        if k != 1:
            alpha_test = alpha_min - (
                d_cost_min * (alpha_max - alpha_min) / (d_cost_max - d_cost_min)
            )  # secant update
            write_to_file(f"Secant update: alpha_test={alpha_test}")
            method = "secant"
            if np.isnan(alpha_test):
                write_to_file("Secant update fail-- bisecting instead")
                alpha_test = 0.5 * (alpha_min + alpha_max)  # bisection update
                write_to_file(f"Bisection update: alpha_test={alpha_test}")
                method = "bisection"
        elif k == 1:
            alpha_test = 0.5 * (alpha_min + alpha_max)  # bisection update
            write_to_file(f"Bisection update: alpha_test={alpha_test}")

        working_params = p.params + alpha_test * direction
        working_p.params = working_params

        working_epsilon, working_psi = cost_function(working_p, f, thresh, workers, scalist, neighbors, cfg)
        print(f"Global elapsed t = {(time.time()-t0_global)/60:8.1f}")
        working_resids = residual_function(
            working_psi, f_prime, scalist, wcslist, neighbors, thresh, workers, cfg
        )
        print(f"Global elapsed t = {(time.time()-t0_global)/60:8.1f}")
        d_cost = np.sum(working_resids * direction)
        convergence_crit = alpha_max - alpha_min
        conv_params.append([working_epsilon, alpha_test, d_cost])

        write_to_file(f"Ending LS iteration {k}")
        write_to_file(f"Current d_cost = {d_cost}, epsilon = {working_epsilon}")
        write_to_file(f"Working resids: {working_resids}")
        write_to_file(f"Working params: {working_p.params}")
        write_to_file(f"Current alpha range (min, test, max): {alpha_min, alpha_test, alpha_max}")
        write_to_file(f"Current delta alpha: {convergence_crit}")
        write_to_file(f"Time spent in this LS iteration: {(time.time() - t0_ls_iter) / 60} minutes.")

        # Convergence and update criteria and checks
        if (working_epsilon < best_epsilon + tol * alpha_test * d_cost) and (np.abs(alpha_test) >= 1e-6):
            best_epsilon = working_epsilon
            best_p = copy.deepcopy(working_p)
            best_psi = working_psi
            best_resids = working_resids
            write_to_file(f"Linear search convergence in {k} iterations")
            if testing:
                save_fits(best_p.params, "best_p", dir=test_image_dir, overwrite=True)
                save_fits(np.array(conv_params), "conv_params", dir=test_image_dir, overwrite=True)
            return best_p, best_psi, best_resids, best_epsilon

        # Updates for next iteration, if convergence isn't yet reached
        if d_cost > tol and method == "bisection":
            alpha_max = alpha_test
            d_cost_max = d_cost
        elif d_cost < -tol and method == "bisection":
            alpha_min = alpha_test
            d_cost_min = d_cost
        elif d_cost * d_cost_min < 0 and method == "secant":
            alpha_max = alpha_test
            d_cost_max = d_cost
        elif d_cost * d_cost_max < 0 and method == "secant":
            alpha_min = alpha_test
            d_cost_min = d_cost

    return best_p, best_psi


def linear_search_quadratic(
    p, direction, f, f_prime, grad_current, thresh, workers, scalist, wcslist, neighbors, cfg
):
    """
    For the quadratic cost function, direct calculation of alpha that minimizes the function
        d_epsilon/d_alpha in the given direction . Note alpha = depth of step in direction

    Finds the best alpha, computes the new parameters and diff image, and prints the new cost and convergence criteria

    Parameters
    --------
    p : params object
        the current de-striping parameters
    direction : 2D np array
        direction of conjugate gradient search
    f : function
        cost function form
    f_prime : function
        derivative of cost function form
    grad_current : 2D np array
        current gradient AKA current residuals
    thresh : float or None
        threshold for Huber loss cost function
    workers : Int
        number of parallel workers to use
    scalist : List of Str
        the list of all SCAs in this mosaic
    wcslist : List of WCS objects
        the WCS object for each SCA in scalist (same order)
    neighbors : Dict
        dictionary where keys are SCA indices and values are lists of indices of overlapping SCAs
    cfg : Config object
        the configuration for this run


    Returns
    --------
    new_p: parameters object
        containing the new parameters found via direct calculation
    new_psi: 3D numpy array
        the difference images made from images with the new_p params subtracted off
    new_resids: 2D np array
        the new residuals calculated with new_p
    new_epsilon: float
        the new cost function value calculated with new_p

    """
    t0_ls = time.time()

    # Simple linear search
    new_p = copy.deepcopy(p)
    trial_p = copy.deepcopy(p)

    eta = 0.1
    d_cost_init = np.sum(grad_current * direction)

    alpha_test = -eta * (np.sum(grad_current * direction)) / (np.sum(direction * direction) + 1e-12)
    if alpha_test <= 0:
        # Not a descent direction — fallback
        alpha_min = -0.9
        alpha_max = 1.0
    else:
        # Curvature-based search window
        alpha_min = alpha_test * 1e-4
        alpha_max = alpha_test * 10

    # Calculate
    trial_params = p.params + alpha_max * direction
    trial_p.params = trial_params
    trial_epsilon, trial_psi = cost_function(trial_p, f, thresh, workers, scalist, neighbors, cfg)
    trial_resids = residual_function(
        trial_psi, f_prime, scalist, wcslist, neighbors, thresh, workers, cfg
    )
    del trial_psi, trial_epsilon

    alpha_new = (
        alpha_max
        * (-np.sum(direction * grad_current))
        / (np.sum(direction * (trial_resids - grad_current)) + 1e-12)
    )

    new_params = p.params + alpha_new * direction
    new_p.params = new_params
    new_epsilon, new_psi = cost_function(new_p, f, thresh, workers, scalist, neighbors, cfg)
    new_resids = grad_current + (alpha_new / alpha_max) * (trial_resids - grad_current)
    print(f"(Inside LS) Global elapsed t = {(time.time()-t0_global)/60:8.1f}")
    sys.stdout.flush()

    d_cost = np.sum(new_resids * direction)

    write_to_file("Ending LS")
    write_to_file(f"Current d_cost = {d_cost}")
    write_to_file(f"Current epsilon = {new_epsilon}")
    write_to_file(f"Working resids: {new_resids}")
    write_to_file(f"Working params: {new_p.params}")
    write_to_file(f"Current alpha: {alpha_new}")
    write_to_file(f"Time spent in this LS: {(time.time() - t0_ls) / 60} minutes.")
    sys.stdout.flush()

    # Convergence and update criteria and checks
    if testing:
        save_fits(new_p.params, "best_p", dir=test_image_dir, overwrite=True)
    return new_p, new_psi, new_resids, new_epsilon


def conjugate_gradient(
    p,
    f,
    f_prime,
    thresh,
    workers,
    scalist,
    wcslist,
    neighbors,
    restart_file=None,
    time_limit=None,
    cfg=None,
    of="destripe_out.txt"
):
    """
    Algorithm to use conjugate gradient descent to optimize the parameters for destriping.
    Direction is updated using Fletcher-Reeves method

    Parameters
    --------
    p : parameters object
        containing initial parameters guess
    f : function
        functional form to use for cost function
    f_prime : function
        the derivative of f. KL: eventually f should dictate f prime
    thresh : float or None
        threshold for Huber loss cost function; default None
    workers : Int
        number of parallel workers to use
    scalist : List of Str
        the list of all SCAs in this mosaic
    wcslist : List of WCS objects
        the WCS object for each SCA in scalist (same order)
    neighbors : Dict    
        dictionary where keys are SCA indices and values are lists of indices of overlapping SCAs
    restart_file : Str or None
        if not None, path to pickle file containing restart state
    time_limit : int or None
        if not None, how much time to elapse before stopping (minutes)
    cfg : config object
        containing all config parameters
    of : Str
        output file to write log messages to

    Returns
    --------
    p : params object
        the best fit parameters for destriping the SCA images
    """
    write_to_file("### Starting conjugate gradient optimization")
    print(f"Global elapsed t = {(time.time()-t0_global)/60:8.1f}")
    print(f"HL Threshold (None, if cost fn is not Huber Loss): {thresh}")
    print(f"Restart?: {cfg.ds_restart}\n")

    global test_image_dir
    test_image_dir = cfg.ds_outpath + "/test_images/" + str(0) + "/"
    log_file = os.path.join(cfg.ds_outpath, "cg_log.csv")

    if cfg.ds_restart is not None:
        with open(cfg.ds_restart, "rb") as f_in:
            state = pickle.load(f_in)
        write_to_file(f"Restarting CG from snapshot {cfg.ds_restart} at iteration {state['iteration']+1}")
        i = state["iteration"]
        p = state["p"]
        grad = state["grad"]
        epsilon = state["epsilon"]
        direction = state["direction"]
        grad_prev = state["grad_prev"]
        direction_prev = state["direction_prev"]
        cg_model = state["cg_model"]
        tol = state["tol"]
        thresh = state["thresh"]
        psi = state["psi"]
        norm_0 = state["norm_0"]
        cost_model = state["cost_model"]

    else:
        # Initialize variables
        cg_model = cfg.cg_model
        cost_model = cfg.cost_model
        tol = cfg.cg_tol
        i = -1
        grad_prev = None  # No previous gradient initially
        grad = None
        direction = None  # No initial direction
        write_to_file("### Starting initial cost function", of)
        epsilon, psi = cost_function(p, f, thresh, workers, scalist, neighbors, cfg)
        print(f"Global elapsed t = {(time.time()-t0_global)/60:8.1f}")

        with open(log_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Iteration",
                    "Current Norm",
                    "Convergence Rate",
                    "Step Size",
                    "Gradient Magnitude",
                    "Final d_cost",
                    "Final Epsilon",
                    "Time (min)",
                    "LS time (min)",
                    "MSE",
                    "Parameter Change",
                ]
            )

    sys.stdout.flush()

    for i in range(i + 1, cfg.cg_maxiter):
        write_to_file(f"### CG Iteration: {i + 1}", of)
        test_image_dir = cfg.ds_outpath + "/test_images/" + str(i + 1) + "/"
        os.makedirs(test_image_dir, exist_ok=True)
        t_start_CG_iter = time.time()

        # Compute the gradient
        if grad is None:
            grad = residual_function(
                psi, f_prime, scalist, wcslist, neighbors, thresh, workers, cfg
            )
            write_to_file(
                f"Minutes spent in initial residual function: {(time.time() - t_start_CG_iter) / 60}", of
            )
            print(f"Global elapsed t = {(time.time()-t0_global)/60:8.1f}")
            sys.stdout.flush()

        # Compute the norm of the gradient
        global current_norm
        current_norm = np.linalg.norm(grad)

        if i == 0 and grad_prev is None:
            write_to_file(f"Initial gradient: {grad}", of)
            norm_0 = np.linalg.norm(grad)
            write_to_file(f"Initial norm: {norm_0}", of)
            write_to_file(f"Initial epsilon: {epsilon}", of)
            tol = tol * norm_0
            direction = -grad

        elif (i + 1) % 10 == 0:
            beta = 0
            write_to_file(f"Current Beta: {beta} (using method: {cg_model})", of)
            direction = -grad + beta * direction_prev

        else:
            # Calculate beta (direction scaling) depending on cg_model
            if cg_model == "FR":
                beta = np.sum(np.square(grad)) / np.sum(np.square(grad_prev))
            elif cg_model == "PR":
                beta = max(0, np.sum(grad * (grad - grad_prev)) / (np.sum(np.square(grad_prev))))
            elif cg_model == "HS":
                beta = np.sum(grad * (grad - grad_prev)) / np.sum(-direction_prev * (grad - grad_prev))
            elif cg_model == "DY":
                beta = np.sum(np.square(grad)) / np.sum(-direction_prev * (grad - grad_prev))

            else:
                raise ValueError(f"Unknown method for CG direction update: {cg_model}")

            write_to_file(f"Current Beta: {beta} (using method: {cg_model})", of)

            direction = -grad + beta * direction_prev

        if current_norm < tol:
            write_to_file(
                f"Convergence reached at iteration: {i + 1} via norm {current_norm} < tol {tol}", of
            )
            break

        # Perform linear search
        t_start_LS = time.time()
        write_to_file(f"Initiating linear search in direction: {direction}", of)
        sys.stdout.flush()

        if cost_model=='quadratic':
            p_new, psi_new, grad_new, epsilon_new = linear_search_quadratic(
                p, 
                direction, 
                f, 
                f_prime, 
                grad, 
                thresh, 
                workers, 
                scalist, 
                wcslist, 
                neighbors, 
                cfg
            )

        else:
            p_new, psi_new, grad_new, epsilon_new = linear_search_general(
                p,
                direction,
                f,
                f_prime,
                cost_model,
                epsilon,
                psi,
                grad,
                thresh,
                workers,
                scalist,
                wcslist,
                neighbors,
                cfg
            )

        print(f"Global elapsed t = {(time.time()-t0_global)/60:8.1f}")
        ls_time = (time.time() - t_start_LS) / 60
        write_to_file(f"Total time spent in linear search: {ls_time}", of)
        write_to_file(
            f"Current norm: {current_norm}, Tol * Norm_0: {tol}, Difference (CN-TOL): {current_norm - tol}",
            of,
        )
        sys.stdout.flush()

        # Calculate additional metrics
        convergence_rate = (current_norm - np.linalg.norm(grad_new)) / current_norm
        step_size = np.linalg.norm(p_new.params - p.params)
        gradient_magnitude = np.linalg.norm(grad_new)
        mse = np.mean(psi_new**2)
        parameter_change = np.linalg.norm(p_new.params - p.params)

        with open(log_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    i + 1,
                    current_norm,
                    convergence_rate,
                    step_size,
                    gradient_magnitude,
                    np.sum(grad * direction),
                    np.sum(psi),
                    (time.time() - t_start_CG_iter) / 60,
                    ls_time,
                    mse,
                    parameter_change,
                ]
            )

        # Update to current values
        p = p_new
        psi = psi_new
        epsilon = epsilon_new
        grad_prev = grad
        grad = grad_new
        direction_prev = direction

        write_to_file(
            f"Total time spent in this CG iteration: {(time.time() - t_start_CG_iter) / 60} minutes.", of
        )
        print(f"Global elapsed t = {(time.time()-t0_global)/60:8.1f}")
        sys.stdout.flush()

        # Save checkpoint if walltime exceeded
        if time_limit is not None:
            elapsed_minutes = (time.time() - t0_global) / 60
            if elapsed_minutes >= time_limit:
                write_to_file(f"Walltime limit {time_limit} min reached. Exiting early!!!", of)
                if cfg.ds_restart is None:
                    restart_file = os.path.join(cfg.ds_outpath, "cg_restart.pkl")
                save_snapshot(
                    p,
                    grad,
                    epsilon,
                    psi,
                    direction,
                    grad_prev,
                    direction_prev,
                    cg_model,
                    tol,
                    thresh,
                    norm_0,
                    cost_model,
                    i,
                    restart_file,
                )
                return p

        if i == cfg.cg_maxiter - 1:
            write_to_file(f"CG reached MAX ITERATIONS {cfg.cg_maxiter} and DID NOT converge!!!!", of)

    write_to_file(f"Conjugate gradient complete. Finished in {i + 1} / {cfg.cg_maxiter} iterations", of)
    write_to_file(f"Final parameters: {p.params}", of)
    write_to_file(f"Final norm: {current_norm}", of)
    return p


def main():
    CG_models = {"FR", "PR", "HS", "DY"}

    cfg_file = sys.argv[1] if len(sys.argv) > 1 else None
    if cfg_file is not None:
        CFG = Config(
            cfg_file=cfg_file
        )
    else:
        raise ValueError("Please provide a config file as a command line argument.")
    
    filter_ = filters[CFG.use_filter]
    outpath = CFG.ds_outpath  

    # Prior on cost function is not yet implemented
    # if CFG.cost_prior != 0:
    #     cost_prior = CFG.cost_prior

    if CFG.cg_model not in CG_models:
        raise ValueError(f"CG model {CFG.cg_model} not in CG_models dictionary.")
    outfile = outpath + filter_ + CFG.ds_outstem  # the file that the output prints etc are written to

    CFG()

    t0 = time.time()

    workers = os.cpu_count() // int(os.environ["OMP_NUM_THREADS"]) if "OMP_NUM_THREADS" in os.environ else 12
    write_to_file(f"## Using {workers} workers for parallel processing.")

    all_scas, all_wcs = get_scas(filter_, CFG.ds_obsfile, CFG)
    write_to_file(f"{len(all_scas)} SCAs in this mosaic", filename=outfile)

    if testing:
        if os.path.isfile(outpath + "ovmat.npy"):
            ov_mat = np.load(outpath + "ovmat.npy")
        else:
            ovmat_t0 = time.time()
            write_to_file("Overlap matrix computing start", filename=outfile)
            ov_mat = compareutils.get_overlap_matrix(all_wcs, verbose=True)
            np.save(outpath + "ovmat.npy", ov_mat)
            write_to_file(
                f"Overlap matrix complete. Duration: {(time.time() - ovmat_t0) / 60} Minutes",
                filename=outfile,
            )
            write_to_file(f"Overlap matrix saved to: {outpath}ovmat.npy", filename=outfile)
    else:
        ovmat_t0 = time.time()
        write_to_file("Overlap matrix computing start", filename=outfile)
        ov_mat = compareutils.get_overlap_matrix(
            all_wcs, verbose=True
        )  # an N_wcs x N_wcs matrix containing fractional overlap
        write_to_file(
            f"Overlap matrix complete. Duration: {(time.time() - ovmat_t0) / 60} Minutes", filename=outfile
        )

    neighbors = get_neighbors(all_scas, ov_mat)
    # Initialize parameters
    p0 = Parameters(cfg=CFG, scalist=all_scas) 
    cm = Cost_models(cfg=CFG)

    # Do it
    try:
        p = conjugate_gradient(
            p0,
            cm.f,
            cm.f_prime,
            cm.thresh,
            workers,
            all_scas,
            all_wcs,
            neighbors,
            time_limit=7200,
            cfg=CFG,
            of=outfile
        )
        hdu = fits.PrimaryHDU(p.params)
        hdu.writeto(outpath + "final_params.fits", overwrite=True)
        print(outpath + "final_params.fits created \n")

    except Exception as e:
        print(f'Exception: {e}')
        logging.exception("An error occurred:")
        print("Conjugate gradient failed. Restart state saved to cg_restart.pkl\n")

    for i, sca in enumerate(all_scas):
        obsid, scaid = get_ids(sca)
        this_sca = Sca_img(obsid, scaid, CFG, add_objmask=False)
        this_param_set = p.forward_par(i)
        ds_image = this_sca.image - this_param_set

        hdu = fits.PrimaryHDU(ds_image, header=this_sca.header)
        hdu.header["TYPE"] = "DESTRIPED_IMAGE"
        hdu2 = fits.ImageHDU(this_sca.image, header=this_sca.header)
        hdu2.header["TYPE"] = "SCA_IMAGE"
        hdu3 = fits.ImageHDU(this_param_set, header=this_sca.header)
        hdu3.header["TYPE"] = "PARAMS_IMAGE"
        hdulist = fits.HDUList([hdu, hdu2, hdu3])
        hdulist.writeto(outpath + filter_ + "_DS_" + obsid + "_" + scaid + ".fits", overwrite=True)

    write_to_file(f"Destriped images saved to {outpath + filter_} _DS_*.fits", filename=outfile)
    write_to_file(f"Total hours elapsed: {(time.time() - t0) / 3600}", filename=outfile)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    mem_usage = None
    try:
        mem_usage = memory_usage(main, interval=120, retval=False)
    finally:
        profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats()
        with open("profile_results.txt", "w") as f:
            f.write(stream.getvalue())
        if mem_usage is not None:
            with open("memory_profile_results.txt", "w") as f:
                for i, mem in enumerate(mem_usage):
                    f.write(f"{i}\t{mem:.2f} MiB\n")

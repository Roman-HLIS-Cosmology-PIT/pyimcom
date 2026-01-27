"""Test functions for imdestripe.
Many of these functions / tests are adapted from test_pyimcom for internal consistency.
"""

import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.modeling import models
import tempfile
import os
import pathlib

from pyimcom import imdestripe
from pyimcom.config import Config


# Test constants
DEGREE = np.pi / 180.0

def create_test_wcs(crval, test_size=100, offset=False):
    """
    Create a simple WCS for testing (similar to make_simple_wcs in PyIMCOM tests)
    
    Parameters
    ----------
    crval : tuple of float
        Reference point (RA, Dec) in degrees
    test_size : int
        Size of image
    
    Returns
    -------
    astropy.wcs.WCS
    """
    outwcs = wcs.WCS(naxis=2)
    if offset:
        outwcs.wcs.crpix = [test_size/2+10, test_size/2+15]
    else:
        outwcs.wcs.crpix = [test_size/2, test_size/2]
    outwcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    outwcs.wcs.cdelt = np.array([-3.0556e-5, 3.0556e-5]) # ~0.11 arcsec/pixel
    outwcs.wcs.crval = list(crval)
    
    return outwcs


def create_test_config(ds_rows=100, ds_model="constant", cost_model="quadratic"):
    """
    Create a minimal config object for testing.
    
    Returns a Config-like object with necessary attributes.
    """
    class TestConfig:
        def __init__(self):
            self.ds_rows = ds_rows
            self.ds_model = ds_model
            self.cost_model = cost_model
            self.cg_model = "PR"
            self.cg_maxiter = 10
            self.cg_tol = 1e-4
            self.ds_obsfile = ""
            self.ds_outpath = ""
            self.ds_outstem = ""
            self.use_filter = "H158"
            self.permanent_mask = None
            self.ds_noisefile = False
            self.gaindir = False
            self.hub_thresh = 1.0
            self.ds_restart = None
    
    return TestConfig()


class SimpleSCA:
    """
    Simplified SCA-like object for testing without full imdestripe Sca_img object initialization.
    
    Mimics the Sca_img interface needed for interpolation tests.
    """
    def __init__(self, image, w, shape, g_eff=None):
        self.image = image.astype(np.float64)
        self.w = w
        self.shape = shape
        self.g_eff = g_eff if g_eff is not None else np.ones_like(image, dtype=np.float64)
        self.mask = np.ones(shape, dtype=bool)
        self.obsid = "00001"
        self.scaid = "01"


def make_simple_sca(type="constant", offset=False):
    """
    Create a simple SCA-like object for testing.
    Parameters
    ----------
    type : str
        Type of test image to create. Options are "constant", "gradient", "random".
    offset : bool
        Whether to offset the WCS center pixel.
    Returns
    -------
    SimpleSCA
        A simple SCA-like object with test image and WCS.
    """
    np.random.seed(13)  # for reproducibility
    test_size = 100

    if type == "constant":
        test_image = np.ones((test_size, test_size), dtype=np.float64) * 13.0
    elif type == "gradient":
        x = np.linspace(0, 100, test_size)
        y = np.linspace(0, 100, test_size)
        xv, yv = np.meshgrid(x, y)
        test_image = (xv + yv).astype(np.float64) / 2.0
    elif type == "gaussian_peak":
        y0, x0, sigma = 30, 30, 5.0
        y, x = np.indices((test_size, test_size))
        test_image = np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

    test_wcs = create_test_wcs((150.0, 2.0), test_size=test_size, offset=offset)
    test_shape = test_image.shape
    test_g_eff = np.ones(test_shape, dtype=np.float64)

    sca = SimpleSCA(test_image, test_wcs, test_shape, g_eff=test_g_eff)
    return sca

def test_get_ids():
    """Test function for splitting an obsid,sca pair."""

    s = "mybeginning_670_16.myending"  # string to parse
    obsid, scaid = imdestripe.get_ids(s)
    print(obsid, scaid)
    # check if we parsed correctly
    assert obsid == "670"
    assert scaid == "16"

def test_object_mask():
    """Test function for creating an object mask."""

    # create a test image with a bright object in the center
    test_image = np.zeros((100, 100), dtype=np.float32)
    test_image[40:60, 40:60] = 100.0  # bright square in the center

    # create an object mask
    mask = imdestripe.apply_object_mask(test_image, threshold=10.0)

    # check if the mask correctly identifies the object
    assert np.sum(mask) == 400  # 20x20 square should be masked

class TestInterpolateImageBilinear:
    """Test class for bilinear interpolation of images."""

    def test_identity_interpolation(self):
        """Test bilinear interpolation where source and target WCS are identical"""
        sca_A = make_simple_sca(type="gradient")
        sca_B = make_simple_sca(type="gradient")
        interp_image = np.zeros_like(sca_A.image)

        imdestripe.interpolate_image_bilinear(sca_A, sca_B, interp_image)

        # The interpolated image should be the same as the original image
        assert np.allclose(interp_image, sca_A.image)

    def test_interpolation_constant(self):
        """Test bilinear interpolation on a constant image with offset WCSes"""
        sca_A = make_simple_sca(type="constant")
        sca_B = make_simple_sca(type="constant", offset=True)
        interp_image = np.zeros_like(sca_A.image)

        imdestripe.interpolate_image_bilinear(sca_A, sca_B, interp_image)

        # The interpolated image should be the same as the original constant image
        assert np.allclose(interp_image, sca_A.image)

    def test_interpolation_shifted(self):
        """
        Test bilinear interpolation between two images with a known shift in WCS.
        Image A is constant, with a WCS offset.
        Image B contains a gaussian peak at (30, 30).
        Image B is interpolated onto the WCS of Image A, moving the peak to (40, 45).

        """
        sca_A = make_simple_sca(type="constant", offset=True)
        sca_B = make_simple_sca(type="gaussian_peak")
        interp_image = np.zeros_like(sca_A.image)

        imdestripe.interpolate_image_bilinear(sca_B, sca_A interp_image)

        peak_y, peak_x = np.unravel_index(np.argmax(interp_image), interp_image.shape)

        # Allow ±2 pixel tolerance due to interpolation
        assert abs(peak_x - 40) <= 2, f"Peak x-position should be ~40, got {peak_x}"
        assert abs(peak_y - 45) <= 2, f"Peak y-position should be ~45, got {peak_y}"



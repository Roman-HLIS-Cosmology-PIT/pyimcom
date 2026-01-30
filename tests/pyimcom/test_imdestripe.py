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


# Tools for tests: create WCS, create config, create simple SCA-like object

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
    outwcs.array_shape = (test_size, test_size)
    
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
    elif type == "random":
        np.random.seed(13)  # for reproducibility
        test_image = np.random.rand(test_size, test_size).astype(np.float64)

    test_wcs = create_test_wcs((150.0, 2.0), test_size=test_size, offset=offset)
    test_shape = test_image.shape
    test_g_eff = np.ones(test_shape, dtype=np.float64)

    sca = SimpleSCA(test_image, test_wcs, test_shape, g_eff=test_g_eff)
    return sca

###############################
# Big tests
###############################

class TestInterpolateImageBilinear:
    """Test class for bilinear interpolation of images."""

    def test_identity_interpolation(self):
        """Test bilinear interpolation where source and target WCS are identical"""
        sca_A = make_simple_sca(type="gradient")
        sca_B = make_simple_sca(type="gradient")
        interp_image = np.zeros_like(sca_A.image, dtype=np.float64)

        imdestripe.interpolate_image_bilinear(sca_B, sca_A, interp_image)

        # The interpolated image should be the same as the original image
        assert np.allclose(interp_image, sca_A.image)

    def test_interpolation_constant(self):
        """Test bilinear interpolation on a constant image with offset WCSes"""
        sca_A = make_simple_sca(type="constant")
        sca_B = make_simple_sca(type="constant", offset=True)
        interp_image = np.zeros_like(sca_A.image)

        imdestripe.interpolate_image_bilinear(sca_B, sca_A, interp_image)

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

        imdestripe.interpolate_image_bilinear(sca_B, sca_A, interp_image)
        peak_y, peak_x = np.unravel_index(np.argmax(interp_image), interp_image.shape)

        # Allow ±2 pixel tolerance due to interpolation
        assert abs(peak_x - 40) <= 2, f"Peak x-position should be ~40, got {peak_x}"
        assert abs(peak_y - 45) <= 2, f"Peak y-position should be ~45, got {peak_y}"

class TestTransposeInterpolate:
    """Test class for transpose bilinear interpolation of images."""

    def test_identity_transpose(self):
        """Test transpose bilinear interpolation where source and target WCS are identical"""
        sca_A = make_simple_sca(type="gradient")
        sca_B = make_simple_sca(type="gradient")
        interp_image = np.zeros_like(sca_A.image)

        imdestripe.transpose_interpolate(sca_A.image, sca_A.w, sca_B, interp_image)

        # The interpolated image should be the same as the original image
        assert np.allclose(interp_image, sca_A.image)
    
    def test_adjoint_property(self):
        """Test the adjoint property of bilinear interpolation and its transpose."""
        sca_A = make_simple_sca(type="random")
        sca_B = make_simple_sca(type="random", offset=True)

        image_A = sca_A.image
        image_B = sca_B.image

        interp_image = np.zeros_like(image_B, dtype=np.float64)
        imdestripe.interpolate_image_bilinear(sca_B, sca_A, interp_image)

        transpose_image = np.zeros_like(image_A, dtype=np.float64)
        imdestripe.transpose_interpolate(image_A, sca_A.w, sca_B, transpose_image)

        lhs = np.sum(interp_image * image_A)
        rhs = np.sum(image_B * transpose_image)

        relative_error = abs(lhs - rhs) / (abs(lhs) + 1e-10)

        print(f"\nAdjoint property test:")
        print(f"  <I(x), y>   = {lhs:.10e}")
        print(f"  <x, I^T(y)> = {rhs:.10e}")
        print(f"  Relative error = {relative_error:.10e}")
        
        # These must be equal within numerical precision
        assert relative_error < 1e-6, (
            f"CRITICAL FAILURE: Adjoint property violated!\n"
            f"  <I(x), y>   = {lhs:.10e}\n"
            f"  <x, I^T(y)> = {rhs:.10e}\n"
            f"  Relative error = {relative_error:.10e}\n"
        )

def test_residual_gradient():
    """Test function for residual gradient computation."""

    sca_A = make_simple_sca(type="random")
    sca_B = make_simple_sca(type="random", offset=True)

    cfg = create_test_config()
    scalist = ["sca_a", "sca_b"]
    wcslist = [sca_A.w, sca_B.w]
    neighbors = {0: [1], 1: [0]}
    cost_model = imdestripe.Cost_models(cfg)

    # Create two psi difference images with known values
    psi = np.zeros((2, sca_A.image.shape[0], sca_A.image.shape[1]), dtype=np.float32)
    psi[0, :, :] = 1.
    psi[1, :, :] = 2.

    # Analytical gradient
    grad = imdestripe.residual_function(
            psi, cost_model.f_prime, scalist, wcslist, neighbors, 
            thresh=None, workers=1, cfg=cfg
        )
    
    # Numerical gradient : finite difference
    delta = 1e-5
    p = imdestripe.Parameters(cfg, scalist)
    
    epsilon_0, _ = imdestripe.cost_function(
        p, cost_model.f, None, 1, scalist, neighbors, cfg
    )
    
    grad_numerical = np.zeros_like(grad)
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            p_perturbed = imdestripe.Parameters(cfg, scalist)
            p_perturbed.params = p.params.copy()
            p_perturbed.params[i, j] += delta
            
            epsilon_plus, _ = imdestripe.cost_function(
                p_perturbed, cost_model.f, None, 1, scalist, neighbors, cfg
            )
            
            grad_numerical[i, j] = (epsilon_plus - epsilon_0) / delta
    

###############################
# Little tests
###############################

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
    mask = imdestripe.apply_object_mask(test_image, threshold_c=10.0)

    # check if the mask correctly identifies the object
    assert np.sum(mask) >= 400  # At least the bright region is masked
    assert np.sum(mask) <= 700  # Plus some dilation

def test_transpose_par():
    """Test function for parameters transpose function. """
    img = np.arange(10)[:, np.newaxis] * np.ones((10, 20))
    transposed_img = imdestripe.transpose_par(img)

    expected = np.arange(10)*20
    assert np.allclose(transposed_img, expected)

def test_parameters(self):
    """Test initialization of Parameters class."""
    
    cfg = create_test_config()
    scalist = ["H158_001_01.fits"]

    p = imdestripe.Parameters(cfg, scalist)

    #Check initialization
    assert p.params.shape == (1,100)
    assert p.model == "constant"
    assert p.n_rows == 100
    assert p.params_per_row == 1

    test_vals = np.arange(10) * 0.1
    p.params[0, :] = test_vals # set some test values

    param_image = p.forward_par(0)
    assert param_image.shape == (10, 10)
    
    for i in range(10):
        expected_value = test_vals[i]
        row_values = param_image[i, :]
        np.testing.assert_array_almost_equal(
            row_values, expected_value,
            err_msg=f"Row {i} should have constant value {expected_value}"
        )

def test_cost_function():
    """ Very simple test function for cost function computation."""

    diff = 5
    diff_img = np.full((100, 100), diff, dtype=np.float32)

    expected_cost = 100**2 * diff**2
    cost = imdestripe.quadratic(diff)

    assert np.isclose(cost, expected_cost), f"Cost should be {expected_cost}, got {cost}"


def interpolation_shape_debug():
    """Debug interpolation issues."""
    sca_A = make_simple_sca(type="constant")
    sca_B = make_simple_sca(type="constant")
    
    print(f"Image A shape: {sca_A.image.shape}, dtype: {sca_A.image.dtype}")
    print(f"Image B shape: {sca_B.image.shape}, dtype: {sca_B.image.dtype}")
    print(f"Image A WCS array_shape: {sca_A.w.array_shape}")
    print(f"Image B WCS array_shape: {sca_B.w.array_shape}")
    
    interp_image = np.zeros_like(sca_A.image, dtype=np.float64)
    print(f"Interp image dtype: {interp_image.dtype}")
    print(f"Interp image before: {interp_image[0, 0]}")
    
    imdestripe.interpolate_image_bilinear(sca_B, sca_A, interp_image)
    
    print(f"Interp image after: min={np.min(interp_image)}, max={np.max(interp_image)}, mean={np.mean(interp_image)}")
    print(f"Expected: all values should be ~13.0")

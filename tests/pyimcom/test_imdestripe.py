"""Test functions for imdestripe.
Many of these functions / tests are adapted from test_pyimcom for internal consistency.
"""

import os

import numpy as np
import pytest
from astropy import wcs
from pyimcom import imdestripe

# Tools for tests: create WCS, create config, create simple SCA-like object


def create_test_wcs(ra, dec, test_size=100, offset=False):
    """
    Create a simple WCS for testing (similar to make_simple_wcs in PyIMCOM tests)

    Parameters
    ----------
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees
    test_size : int
        Size of image
    offset : bool
        Whether to offset the center pixel.

    Returns
    -------
    astropy.wcs.WCS
    """
    outwcs = wcs.WCS(naxis=2)
    outwcs.wcs.crpix = [test_size / 2, test_size / 2]
    outwcs.wcs.cdelt = np.array([-3.0556e-5, 3.0556e-5])
    outwcs.wcs.ctype = ["RA---ARC", "DEC--ARC"]
    if offset:
        # Shift by ~10 pixels worth in RA and Dec
        # pixel scale is ~3e-5 deg/pixel, so 10 pixels ≈ 3e-4 degrees
        outwcs.wcs.crval = [ra + 3e-4, dec + 3e-4]
    else:
        outwcs.wcs.crval = [ra, dec]
    outwcs.wcs.lonpole = 180.0
    outwcs.array_shape = (test_size, test_size)

    return outwcs


class MinimalConfig:
    """
    Class defining a minimal config object for testing.
    """

    def __init__(self, ds_rows=100, ds_model="constant", cost_model="quadratic"):
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


def create_test_config(ds_rows=100, ds_model="constant", cost_model="quadratic"):
    """
    Create a minimal config object for testing.

    Returns a Config-like object with necessary attributes.
    """

    config = MinimalConfig(ds_rows=ds_rows, ds_model=ds_model, cost_model=cost_model)

    return config


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
        test_image = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    elif type == "random":
        np.random.seed(13)  # for reproducibility
        test_image = np.random.rand(test_size, test_size).astype(np.float64)

    test_wcs = create_test_wcs(150.0, 2.0, test_size=test_size, offset=offset)
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

        interior_mask = np.ones((100, 100), dtype=bool)
        interior_mask[-1, :] = False  # Last row
        interior_mask[:, -1] = False  # Last column

        # Check if interior pixels match
        assert np.allclose(interp_image[interior_mask], sca_A.image[interior_mask], atol=1e-8)

        # This fails because the last row and column are skipped in interpolation so they come back as zeros.
        # assert np.allclose(interp_image, sca_A.image)

    def test_interpolation_constant(self):
        """Test bilinear interpolation on a constant image with offset WCSes"""
        sca_A = make_simple_sca(type="constant")
        sca_B = make_simple_sca(type="constant", offset=True)
        interp_image = np.zeros_like(sca_A.image, dtype=np.float64)

        imdestripe.interpolate_image_bilinear(sca_B, sca_A, interp_image)

        valid_mask = interp_image != 0.0  # the interpolated image will have zeros where no data maps
        n_valid = np.sum(valid_mask)

        assert n_valid > 5000, f"Should have substantial overlap, got {n_valid} pixels"
        assert np.allclose(interp_image[valid_mask], 13.0)

    def test_interpolation_shifted(self):
        """
        Test bilinear interpolation between two images with a known shift in WCS.
        Image A is constant, with a WCS offset.
        Image B contains a gaussian peak at (30, 30).
        Image B is interpolated onto the WCS of Image A, moving the peak to (40, 45).

        """
        sca_A = make_simple_sca(type="constant", offset=True)
        sca_B = make_simple_sca(type="gaussian_peak")
        original_peak_y, original_peak_x = 30, 30
        interp_image = np.zeros_like(sca_A.image, dtype=np.float64)

        imdestripe.interpolate_image_bilinear(sca_B, sca_A, interp_image)

        new_peak_y, new_peak_x = np.unravel_index(np.argmax(interp_image), interp_image.shape)

        # Peak should have moved by approximately 10 pixels
        x_shift = new_peak_x - original_peak_x
        y_shift = new_peak_y - original_peak_y
        total_shift = np.sqrt(x_shift**2 + y_shift**2)

        print(f"Peak moved from ({original_peak_x}, {original_peak_y}) to ({new_peak_x}, {new_peak_y})")
        print(f"Shift: ({x_shift}, {y_shift}), magnitude: {total_shift:.1f} pixels")

        # Verify peak moved by approximately the right amount (10 pixels)
        assert 8 <= total_shift <= 16, f"Peak should shift by ~10 pixels, got {total_shift:.1f}"

        # Verify peak is still strong
        assert np.max(interp_image) > 0.5, "Peak should be preserved"


class TestTransposeInterpolate:
    """Test class for transpose bilinear interpolation of images."""

    def test_identity_transpose(self):
        """Test transpose bilinear interpolation where source and target WCS are identical"""
        sca_A = make_simple_sca(type="gradient")
        sca_B = make_simple_sca(type="gradient")
        interp_image = np.zeros_like(sca_A.image, dtype=np.float64)

        imdestripe.transpose_interpolate(sca_A.image, sca_A.w, sca_B, interp_image)

        interior_mask = np.ones((100, 100), dtype=bool)
        interior_mask[-1, :] = False  # Last row
        interior_mask[:, -1] = False  # Last column

        # Check if interior pixels match
        assert np.allclose(interp_image[interior_mask], sca_A.image[interior_mask])

        # This fails because the last row and column are skipped in interpolation so they come back as zeros.
        # assert np.allclose(interp_image, sca_A.image)

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

        print("\nAdjoint property test:")
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


def f(x):
    """Quadratic cost function for testing."""
    return x**2  # Quadratic cost function


def f_prime(x):
    """Derivative of quadratic cost function for testing."""
    return 2 * x  # Derivative of quadratic cost function


@pytest.mark.skip(reason="Requires real SCA files on disk - integration test")
def test_residual_gradient():
    """Test function for residual gradient computation."""

    sca_A = make_simple_sca(type="random")
    sca_B = make_simple_sca(type="random", offset=True)
    cfg = create_test_config()

    scalist = ["sca_a", "sca_b"]
    wcslist = [sca_A.w, sca_B.w]
    neighbors = {0: [1], 1: [0]}

    # Create two psi difference images with known values
    psi = np.zeros((2, sca_A.image.shape[0], sca_A.image.shape[1]), dtype=np.float32)
    psi[0, :, :] = 1.0
    psi[1, :, :] = 2.0

    # Analytical gradient
    grad = imdestripe.residual_function(
        psi, f_prime, scalist, wcslist, neighbors, thresh=None, workers=2, cfg=cfg
    )

    # Numerical gradient : finite difference
    delta = 1e-5
    p = imdestripe.Parameters(cfg, scalist)

    epsilon_0, _ = imdestripe.cost_function(p, f, None, 1, scalist, neighbors, cfg)

    grad_numerical = np.zeros_like(grad)
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            p_perturbed = imdestripe.Parameters(cfg, scalist)
            p_perturbed.params = p.params.copy()
            p_perturbed.params[i, j] += delta

            epsilon_plus, _ = imdestripe.cost_function(p_perturbed, f, None, 1, scalist, neighbors, cfg)

            grad_numerical[i, j] = (epsilon_plus - epsilon_0) / delta

    # Compare analytical and numerical gradients
    assert np.allclose(grad, grad_numerical, atol=1e-4), "Analytical and numerical gradients do not match!"


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
    """Test function for creating an object mask.
    The function for the mask is:

    if mask is not None and isinstance(mask, np.ndarray):
        neighbor_mask = mask
    else:
        median_val = np.median(image)
        high_value_mask = image >= threshold_m * median_val + threshold_c
        neighbor_mask = binary_dilation(high_value_mask, structure=np.ones((5, 5), dtype=bool))
    image_out = np.where(neighbor_mask, 0, image)
        return image_out, neighbor_mask
        """

    # create a test image with a bright object in the center
    test_image = np.zeros((100, 100), dtype=np.float32)
    test_image[40:60, 40:60] = 100.0  # bright square in the center

    # create an object mask
    image, neighbor_mask = imdestripe.apply_object_mask(test_image, threshold_c=10.0)

    # check if the mask correctly identifies the bright object
    assert np.sum(image) == 0, "All pixels should be masked to zero"
    assert np.sum(neighbor_mask) == 576, "With dilation, 576 pixels should be masked"
    assert np.all(image[neighbor_mask] == 0), "Masked pixels should be set to zero"


    masked_image, same_mask = imdestripe.apply_object_mask(test_image, mask=neighbor_mask, inplace=True)
    assert same_mask == neighbor_mask, "Returned mask should be the same when mask is provided"
    assert np.all(masked_image[same_mask] == 0), "Masked pixels should be set to zero when inplace=True"


def test_transpose_par():
    """Test function for parameters transpose function."""
    img = np.arange(10)[:, np.newaxis] * np.ones((10, 20))
    transposed_img = imdestripe.transpose_par(img)

    expected = np.arange(10) * 20
    assert np.allclose(transposed_img, expected)


def test_parameters():
    """Test initialization of Parameters class."""

    cfg = create_test_config()
    scalist = ["H158_001_01.fits"]

    p = imdestripe.Parameters(cfg, scalist)

    # Check initialization
    assert p.params.shape == (1, 100)
    assert p.model == "constant"
    assert p.n_rows == 100
    assert p.params_per_row == 1

    test_vals = np.arange(100) * 0.1
    p.params[0, :] = test_vals  # set some test values

    param_image = p.forward_par(0)
    assert param_image.shape == (100, 100)

    p.flatten()
    p.current_shape = "1D"
    p.params_2_images()
    assert p.images.shape == (1, 100)
    assert p.current_shape == "2D"

    for i in range(10):
        expected_value = test_vals[i]
        row_values = param_image[i, :]
        np.testing.assert_array_almost_equal(
            row_values, expected_value, err_msg=f"Row {i} should have constant value {expected_value}"
        )




def test_cost_function():
    """Very simple test function for cost function computation."""

    diff = 5
    diff_img = np.full((10, 10), diff, dtype=np.float32)

    expected_cost = 10**2 * diff**2
    cost = np.sum(imdestripe.quadratic(diff_img))

    assert np.isclose(cost, expected_cost), f"Cost should be {expected_cost}, got {cost}"


# test imdestripe.write_to_file
def test_write_to_file(tmp_path):
    """Test function for writing lines of text to a file
    or to the console if no filename is provided.
    """

    # Test writing to a file
    test_file = tmp_path / "test_imdestripe_output.txt"
    text = "This is an output file"
    imdestripe.write_to_file(text, filename=str(test_file))

    with open(test_file, "r") as f:
        content = f.read()
        assert content == text, f"File content should match lines, got {content}"

    # Test writing to console (should not raise an error)
    try:
        imdestripe.write_to_file(text, filename=None)
    except Exception as e:
        pytest.fail(f"Writing to console should not raise an error, but got: {e}")

    # Delete the test file after the test
    if test_file.exists():
        os.remove(test_file)


def test_cost_models():
    """Test function for cost model assigments and computations."""

    x = 2

    quad_cfg = create_test_config(cost_model="quadratic")
    quad_cost_model = imdestripe.cost_models(quad_cfg)
    assert quad_cost_model.model == "quadratic"
    assert quad_cost_model.thresh is None
    assert quad_cost_model.f(x) == x**2
    assert quad_cost_model.f_prime(x) == 2 * x

    hub_cfg = create_test_config(cost_model="huber_loss")
    hub_cost_model = imdestripe.cost_models(hub_cfg)
    assert hub_cost_model.model == "huber_loss"
    assert hub_cost_model.thresh == 1.0
    assert hub_cost_model.f(x) == hub_cost_model.thresh**2 + 2 * hub_cost_model.thresh * (
        np.abs(x) - hub_cost_model.thresh
    )
    assert hub_cost_model.f_prime(x) == 2.0 * hub_cost_model.thresh * np.sign(x)

    abs_cfg = create_test_config(cost_model="absolute")
    abs_cost_model = imdestripe.cost_models(abs_cfg)
    assert abs_cost_model.model == "absolute"
    assert abs_cost_model.thresh is None
    assert abs_cost_model.f(x) == np.abs(x)
    assert abs_cost_model.f_prime(x) == np.sign(x)


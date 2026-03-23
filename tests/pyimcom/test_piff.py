"""Test functions for the Piff interface."""

import urllib.request
import warnings
from unittest.mock import MagicMock, patch

import galsim
import numpy as np
import pytest
from astropy.io import fits
from pyimcom.utils.piffutils import piff_to_legendre

EXAMPLE_FILE = "https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/ffov_13906_11.piff"


@pytest.fixture
def mock_piff_read():
    """Fixture to mock piff.read so we don't need real PSF files."""
    with patch("pyimcom.utils.piffutils.piff.read") as mock_read:
        # Create a fake PSF object
        mock_psf = MagicMock()
        mock_read.return_value = mock_psf

        # Create a fake image/stamp returned by psf.draw()
        mock_image = MagicMock()

        # By default, let's make the mock PSF draw return an array of 1s
        # We use a property mock to allow setting the array shape dynamically in tests
        type(mock_image).array = property(lambda self: self._array)
        mock_image._array = np.ones((128 * 6, 128 * 6), dtype=np.float32)

        mock_psf.draw.return_value = mock_image

        yield mock_read, mock_psf, mock_image


def test_output_shape(mock_piff_read):
    """Test that the output array has the correct dimensions based on inputs."""
    _, _, mock_image = mock_piff_read

    stamp_size = 32
    oversamp = 2
    legendre_order = 3

    # Adjust our mock image to match the test's expected stamp size
    mock_image._array = np.ones((stamp_size, stamp_size), dtype=np.float32)

    coeffs = piff_to_legendre(
        psf_file="dummy_path.piff",
        chipnum=4,
        stamp_size=stamp_size,
        oversamp=oversamp,
        legendre_order=legendre_order,
    )

    expected_shape = ((legendre_order + 1) ** 2, stamp_size * oversamp, stamp_size * oversamp)
    assert coeffs.shape == expected_shape
    assert coeffs.dtype == np.float32


def test_constant_psf_orthogonality(mock_piff_read):
    """
    Test the Gauss-Legendre quadrature math.
    If the PSF is constant across the focal plane (all 1s), only the 0th-order
    Legendre coefficient should be non-zero due to the orthogonality of the polynomials.
    """
    _, _, mock_image = mock_piff_read

    # Use a small stamp for faster test execution
    stamp_size = 10
    oversamp = 1
    legendre_order = 2
    mock_image._array = np.ones((stamp_size, stamp_size), dtype=np.float32)

    coeffs = piff_to_legendre("dummy.piff", 1, stamp_size, oversamp, legendre_order)

    # The 0th coefficient corresponds to L_0(x)L_0(y). Because L_0 = 1, it integrates to a positive value.
    assert np.all(coeffs[0, :, :] > 0)

    # For any order > 0, the integral of a constant function should be 0.
    # We use np.allclose to account for tiny floating-point inaccuracies.
    assert np.allclose(coeffs[1:, :, :], 0.0, atol=1e-6)


def test_psf_draw_arguments(mock_piff_read):
    """Test that piff's draw method is called with the correctly mapped arguments."""
    mock_read, mock_psf, mock_image = mock_piff_read

    stamp_size = 64
    oversamp = 2
    legendre_order = 1  # order 1 means 2 points per dimension (4 points total)
    chipnum = 14

    mock_image._array = np.ones((stamp_size, stamp_size), dtype=np.float32)

    piff_to_legendre("test.piff", chipnum, stamp_size, oversamp, legendre_order)

    # Check that piff.read was called with the right filename
    mock_read.assert_called_once_with("test.piff")

    # Check total number of draw calls (should be (legendre_order + 1)^2)
    expected_calls = (legendre_order + 1) ** 2 * oversamp**2
    assert mock_psf.draw.call_count == expected_calls

    # Inspect the first call to ensure arguments are passed correctly
    first_call_kwargs = mock_psf.draw.call_args_list[0][1]
    assert first_call_kwargs["chipnum"] == chipnum
    assert first_call_kwargs["stamp_size"] == stamp_size
    assert first_call_kwargs["sca"] == chipnum

    # Ensure x and y were scaled appropriately (between 0 and 4088)
    assert 0 <= first_call_kwargs["x"] <= 4088
    assert 0 <= first_call_kwargs["y"] <= 4088


def test_piff_decomposition(tmp_path):
    """Test decomposition of a PIFF file into Legendre polynomials."""

    # Download the test file to `floc`
    tmp_dir = str(tmp_path)
    floc = tmp_dir + "/test_F_02_11.fits"
    urllib.request.urlretrieve(EXAMPLE_FILE, floc)

    p = 3  # polynomial order
    try:
        coeffs = piff_to_legendre(floc, 11, stamp_size=64, oversamp=6, legendre_order=p, normbox=128)
    except ValueError as ve:
        # This way, we can catch the specific error if we aren't on the roman branch of Piff.
        # (This is likely to become obsolete at some point.)
        assert str(ve) == "psf type RomanOptics is not a valid Piff PSF"
        warnings.warn("Using an older version of Piff without RomanOptics support.")
        return  # abort this test

    assert np.shape(coeffs) == ((p + 1) ** 2, 384, 384)

    # center of image
    k = 12
    print(np.round(1.0e4 * coeffs[0, 192 - k : 192 + k, 192 - k : 192 + k], 0).astype(np.int16))
    assert 0.0056 < np.amax(coeffs[0]) < 0.0063

    # Gaussian fit
    moms = galsim.hsm.FindAdaptiveMom(galsim.Image(coeffs[0]))
    print(moms.moments_sigma)
    print(moms.moments_centroid.x, moms.moments_centroid.y)
    print(moms.observed_e1, moms.observed_e2)
    assert 0.70 < moms.moments_sigma / 6.0 < 0.72
    assert np.hypot(moms.moments_centroid.x - 192.5, moms.moments_centroid.y - 192.5) < 0.3
    assert -0.020 < moms.observed_e1 < -0.015
    assert -0.006 < moms.observed_e2 < -0.001

    # sums
    arr = np.sum(coeffs, axis=(1, 2))
    print(arr)
    assert 0.985 <= arr[0] <= 1.0
    assert np.all(np.abs(arr[1:]) < 2.0e-4)

    # writing is to look at the output if you do this locally
    fits.PrimaryHDU(coeffs).writeto(tmp_dir + "/coeffs.fits", overwrite=True)

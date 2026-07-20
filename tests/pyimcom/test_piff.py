"""Test functions for the Piff interface."""

import urllib.request
import warnings
from unittest.mock import MagicMock, patch

import galsim
import numpy as np
import pytest
from astropy.io import fits
from pyimcom.utils.piffutils import PiffPSFModel, piff_to_legendre, piff_to_legendre_multi

EXAMPLE_FILE = "https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/ffov_13906_17.piff"


def test_output_shape():
    """Test that the output array has the correct dimensions based on inputs."""

    stamp_size = 32
    oversamp = 2
    legendre_order = 3

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

        # Adjust our mock image to match the test's expected stamp size
        mock_image._array = np.ones((stamp_size, stamp_size), dtype=np.float32)

        coeffs = piff_to_legendre(
            psf_file="dummy_path.piff",
            sca=4,
            stamp_size=stamp_size,
            oversamp=oversamp,
            legendre_order=legendre_order,
        )

    expected_shape = ((legendre_order + 1) ** 2, stamp_size * oversamp, stamp_size * oversamp)
    assert coeffs.shape == expected_shape
    assert coeffs.dtype == np.float32


def test_constant_psf_orthogonality():
    """
    Test the Gauss-Legendre quadrature math.
    If the PSF is constant across the focal plane (all 1s), only the 0th-order
    Legendre coefficient should be non-zero due to the orthogonality of the polynomials.
    """

    with patch("pyimcom.utils.piffutils.piff.read") as mock_read:
        # Create a fake PSF object
        mock_psf = MagicMock()
        mock_read.return_value = mock_psf

        # Create a fake image/stamp returned by psf.draw()
        mock_image = MagicMock()

        # By default, let's make the mock PSF draw return an array of 1s
        # We use a property mock to allow setting the array shape dynamically in tests
        type(mock_image).array = property(lambda self: self._array)

        mock_psf.draw.return_value = mock_image

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


def test_psf_draw_arguments():
    """Test that piff's draw method is called with the correctly mapped arguments."""

    stamp_size = 64
    oversamp = 2
    legendre_order = 1  # order 1 means 2 points per dimension (4 points total)
    sca = 14

    with patch("pyimcom.utils.piffutils.piff.read") as mock_read:
        # Create a fake PSF object
        mock_psf = MagicMock()
        mock_read.return_value = mock_psf

        # Create a fake image/stamp returned by psf.draw()
        mock_image = MagicMock()

        # By default, let's make the mock PSF draw return an array of 1s
        # We use a property mock to allow setting the array shape dynamically in tests
        type(mock_image).array = property(lambda self: self._array)

        mock_psf.draw.return_value = mock_image

        mock_image._array = np.ones((stamp_size, stamp_size), dtype=np.float32)

        piff_to_legendre("test.piff", sca, stamp_size, oversamp, legendre_order)

        # Check that piff.read was called with the right filename
        mock_read.assert_called_once_with("test.piff")

        # Check total number of draw calls (should be (legendre_order + 1)^2)
        expected_calls = (legendre_order + 1) ** 2 * oversamp**2
        assert mock_psf.draw.call_count == expected_calls

        # Inspect the first call to ensure arguments are passed correctly
        first_call_kwargs = mock_psf.draw.call_args_list[0][1]
        assert first_call_kwargs["chipnum"] == sca - 1
        assert first_call_kwargs["stamp_size"] == stamp_size
        assert first_call_kwargs["sca"] == sca

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
    coeffs = piff_to_legendre(floc, 12, stamp_size=64, oversamp=6, legendre_order=p, normbox=128)
    assert np.shape(coeffs) == ((p + 1) ** 2, 384, 384)

    # center of image
    k = 12
    print(np.round(1.0e4 * coeffs[0, 192 - k : 192 + k, 192 - k : 192 + k], 0).astype(np.int16))
    assert 0.0056 < np.amax(coeffs[0]) < 0.0066

    # Gaussian fit
    moms = galsim.hsm.FindAdaptiveMom(galsim.Image(coeffs[0]))
    print(moms.moments_sigma)
    print(moms.moments_centroid.x, moms.moments_centroid.y)
    print(moms.observed_e1, moms.observed_e2)
    assert 0.68 < moms.moments_sigma / 6.0 < 0.72
    assert np.hypot(moms.moments_centroid.x - 192.5, moms.moments_centroid.y - 192.5) < 0.3
    assert -0.004 < moms.observed_e1 < 0.004
    assert 0.018 < moms.observed_e2 < 0.022

    # sums
    arr = np.sum(coeffs, axis=(1, 2))
    print(arr)
    assert 0.985 <= arr[0] <= 1.0
    assert np.all(np.abs(arr[1:]) < 2.0e-4)

    with pytest.raises(
        ValueError,
        match="If you'd like to write the coefficients to a file, please provide a valid file path.",
    ):
        piff_to_legendre(
            floc,
            11,
            stamp_size=64,
            oversamp=6,
            legendre_order=p,
            normbox=128,
            write_coeffs=True,
        )

    with pytest.raises(
        ValueError,
        match="If you'd like to write the coefficients to a file, please provide a valid file path.",
    ):
        piff_to_legendre(
            floc,
            11,
            stamp_size=64,
            oversamp=6,
            legendre_order=p,
            normbox=128,
            write_coeffs=True,
            coeffs_file=tmp_dir + "/piff_coeffs.npz",
        )

    try:
        coeffs = piff_to_legendre(
            floc,
            11,
            stamp_size=64,
            oversamp=6,
            legendre_order=p,
            normbox=128,
            write_coeffs=True,
            coeffs_file=tmp_dir + "/piff_coeffs.fits",
        )
    except ValueError as ve:
        # This way, we can catch the specific error if we aren't on the roman branch of Piff.
        # (This is likely to become obsolete at some point.)
        assert str(ve) == "psf type RomanOptics is not a valid Piff PSF"
        warnings.warn("Using an older version of Piff without RomanOptics support.")
        return  # abort this test

    assert np.shape(coeffs) == ((p + 1) ** 2, 384, 384)

    # writing is to look at the output if you do this locally
    # fits.PrimaryHDU(coeffs).writeto(tmp_dir + "/coeffs.fits", overwrite=True)

    # Test conversion to multi-HDU format
    piff_to_legendre_multi(
        floc,
        tmp_dir + "/psf_lpolyfit_13906.fits",
        "L2_2506",
        chips=[i for i in range(1, 15)],
        stamp_size=128,
        oversamp=6,
        legendre_order=2,
    )
    with fits.open(tmp_dir + "/psf_lpolyfit_13906.fits") as f:
        assert f[0].header["PORDER"] == 2
        assert f[0].header["NSCA"] == 18
        assert f[0].header["OVSAMP"] == 6

        for i in range(1, 19):
            assert np.shape(f[i].data) == (9, 768, 768)

            moms = galsim.Image(f[i].data[0, :, :]).FindAdaptiveMom()
            pars = np.array(
                [
                    moms.moments_amp,
                    moms.moments_centroid.x - 384.5,
                    moms.moments_centroid.y - 384.5,
                    moms.moments_sigma,
                    moms.observed_shape.g1,
                    moms.observed_shape.g2,
                ]
            )

            # the ones we actually filled in in this test!
            if i < 15:
                assert 0.6 < pars[0] < 0.9
                assert np.hypot(pars[1], pars[2]) < 0.6
                assert 3.5 < pars[3] < 4.5
                assert np.hypot(pars[4], pars[5]) < 0.03

    # check we don't read off the end if we ask for SCA #18
    piff_to_legendre_multi(
        floc,
        tmp_dir + "/psf_lpolyfit_13906.fits",
        "L2_2506",
        chips=[18],
        stamp_size=32,
        oversamp=4,
        legendre_order=1,
    )

    # check that a ValueError is raised if we give a bad format
    with pytest.raises(ValueError):
        piff_to_legendre_multi(
            floc,
            tmp_dir + "/psf_lpolyfit_13906.fits",
            "An alien civilization might use this format, but I don't.",
            chips=[i for i in range(1, 6)],
            stamp_size=32,
            oversamp=4,
            legendre_order=1,
        )

    # now check the PSF point model
    p = PiffPSFModel(floc, 12)
    arr1 = p.draw(0, 0, oversamp=6)
    arr2 = p.draw(0, 4087, oversamp=6)
    arr3 = p.draw(0, 4087, oversamp=6, normbox=16)

    # Gaussian fit
    moms = galsim.hsm.FindAdaptiveMom(galsim.Image(arr1))
    print(moms.moments_sigma)
    print(moms.moments_centroid.x, moms.moments_centroid.y)
    print(moms.observed_e1, moms.observed_e2)
    assert 0.68 < moms.moments_sigma / 6.0 < 0.72
    assert np.hypot(moms.moments_centroid.x - 384.5, moms.moments_centroid.y - 384.5) < 0.6
    assert 0.002 < moms.observed_e1 < 0.008
    assert 0.017 < moms.observed_e2 < 0.021

    assert np.allclose(arr2, 1.0718 * arr3, atol=1e-5, rtol=1e-3)

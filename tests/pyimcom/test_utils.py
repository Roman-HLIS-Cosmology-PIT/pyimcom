import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from pyimcom.utils.compareutils import get_overlap_matrix, getfootprint, str2dirstem


def test_footprint():
    """Simple test of ``getfootprint``."""

    # make a test WCS
    test_image = fits.PrimaryHDU(np.zeros((4088, 4088)))
    test_wcs = {
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "CRPIX1": 2044,
        "CRPIX2": 2044,
        "CD1_1": 2.88987625047648e-05,
        "CD1_2": -8.1218067073775e-06,
        "CD2_1": 7.15925943776376e-06,
        "CD2_2": 2.78134135476302e-05,
        "CRVAL1": 9.35600030831994,
        "CRVAL2": -44.2190257813423,
        "A_ORDER": 4,
        "A_0_2": 5.281722447e-10,
        "A_0_3": -4.997064734e-14,
        "A_0_4": -2.762219504e-17,
        "A_1_1": 8.217316277e-10,
        "A_1_2": -7.684245068e-14,
        "A_1_3": 1.698764265e-18,
        "A_2_0": 6.114873195e-10,
        "A_2_1": 1.705599037e-14,
        "A_2_2": -6.065426622e-17,
        "A_3_0": -1.784414693e-14,
        "A_3_1": 7.626200473e-18,
        "A_4_0": 4.537665675e-18,
        "B_ORDER": 4,
        "B_0_2": 0.000000001397217153,
        "B_0_3": -3.302067003e-14,
        "B_0_4": -6.116276268e-18,
        "B_1_1": 4.808805541e-11,
        "B_1_2": 4.3581402e-14,
        "B_1_3": 3.871934216e-17,
        "B_2_0": 5.359271745e-10,
        "B_2_1": 2.603318393e-14,
        "B_2_2": -1.678887134e-17,
        "B_3_0": -1.737411736e-14,
        "B_3_1": 8.01855e-17,
        "B_4_0": -2.410144662e-17,
    }
    for k in test_wcs:
        test_image.header[k] = test_wcs[k]

    values = getfootprint(WCS(test_image.header), 64)
    values -= np.array([7.07145476e-01, 1.16509501e-01, -6.97402905e-01, 1.21630461e-06])
    assert np.abs(values[0]) < 1e-6
    assert np.abs(values[1]) < 1e-6
    assert np.abs(values[2]) < 1e-6
    assert np.abs(values[3]) < 1e-9


def test_overlap():
    """Simple test function for overlaps."""

    wcslist = []

    # make a test WCS - OpenUniverse sim (1510, 9), with modifications
    # (sequence is 5 exposures, 0.045 degrees apart)
    for j in range(4):
        test_image = fits.PrimaryHDU(np.zeros((4088, 4088)))
        test_wcs = {
            "CTYPE1": "RA---TAN-SIP",
            "CTYPE2": "DEC--TAN-SIP",
            "CRPIX1": 2044,
            "CRPIX2": 2044,
            "CD1_1": 2.88987625047648e-05,
            "CD1_2": -8.1218067073775e-06,
            "CD2_1": 7.15925943776376e-06,
            "CD2_2": 2.78134135476302e-05,
            "CRVAL1": 9.35600030831994,
            "CRVAL2": -44.2190257813423 + 0.045 * j,
            "A_ORDER": 4,
            "A_0_2": 5.281722447e-10,
            "A_0_3": -4.997064734e-14,
            "A_0_4": -2.762219504e-17,
            "A_1_1": 8.217316277e-10,
            "A_1_2": -7.684245068e-14,
            "A_1_3": 1.698764265e-18,
            "A_2_0": 6.114873195e-10,
            "A_2_1": 1.705599037e-14,
            "A_2_2": -6.065426622e-17,
            "A_3_0": -1.784414693e-14,
            "A_3_1": 7.626200473e-18,
            "A_4_0": 4.537665675e-18,
            "B_ORDER": 4,
            "B_0_2": 0.000000001397217153,
            "B_0_3": -3.302067003e-14,
            "B_0_4": -6.116276268e-18,
            "B_1_1": 4.808805541e-11,
            "B_1_2": 4.3581402e-14,
            "B_1_3": 3.871934216e-17,
            "B_2_0": 5.359271745e-10,
            "B_2_1": 2.603318393e-14,
            "B_2_2": -1.678887134e-17,
            "B_3_0": -1.737411736e-14,
            "B_3_1": 8.01855e-17,
            "B_4_0": -2.410144662e-17,
        }
        for k in test_wcs:
            test_image.header[k] = test_wcs[k]
        wcslist.append(WCS(test_image.header))

    matrix = get_overlap_matrix(wcslist, verbose=True)
    target = np.array(
        [
            [1.0, 0.56543805, 0.20744693, 0.0],
            [0.56543805, 1.0, 0.56543805, 0.20744693],
            [0.20744693, 0.56543805, 1.0, 0.56543805],
            [0.0, 0.20744693, 0.56543805, 1.0],
        ]
    )
    assert np.amax(np.abs(matrix - target) < 0.01)

    # compare subsampled matrix
    matrix4 = get_overlap_matrix(wcslist, subsamp=4)
    assert np.amax(np.abs(matrix - matrix4) < 0.01)


def test_str2dirstem():
    """Simple test of parsing a filename."""

    a, b = str2dirstem("/scr/georgewashington/johnadams/thomasjefferson.fits")
    assert a == "/scr/georgewashington/johnadams/"
    assert b == "thomasjefferson.fits"

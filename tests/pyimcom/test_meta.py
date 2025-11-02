import urllib.request

import galsim
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from pyimcom.meta.distortimage import MetaMosaic, shearimage_to_fits
from pyimcom.meta.ginterp import MultiInterp

EXAMPLE_FILE = (
    "https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/compressiontest_F_02_11.fits"
)


def test_MultiInterp_is_successful():
    """Test for MultiInterp."""

    # This test has nf2 = 6 layers.
    # The first 4 are sine waves, then there is a sum of these, and then there is a circle of point sources.
    samp = 5.0
    rsearch = 4.5
    sigma = samp / np.sqrt(8 * np.log(2))
    n = 425
    x, y = np.meshgrid(np.linspace(0, n - 1, n), np.linspace(0, n - 1, n))
    nf = 4
    nf2 = 6
    u0 = 0.243
    v0 = 0.128
    InArr = np.zeros((nf2, n, n), dtype=np.float32)
    for j in range(nf):
        InArr[j, :, :] = 1.0 + 0.1 * np.cos(2 * np.pi * (u0 * x + v0 * y) / 2.0**j)
    InArr[-2, :, :] = np.sum(InArr[:-2, :, :], axis=0) - 3.6
    for k in range(64):
        xc = 200 + 150 * np.cos(k / 32 * np.pi)
        yc = 170 + 150 * np.sin(k / 32 * np.pi)
        InArr[-1, :, :] += np.exp(-0.5 * ((x - xc) ** 2 + (y - yc) ** 2) / sigma**2)
    InMask = np.zeros((n, n), dtype=bool)
    mat = np.array([[0.52, 0.005], [-0.015, 0.51]])
    sc = 0.5
    nout = 720
    eC = (mat @ mat.T / sc**2 - np.identity(2)) * sigma**2
    C = [eC[0, 0], eC[0, 1], eC[1, 1]]
    pos_offset = [6.0, 3.0]
    OutArr, OutMask, Umax, Smax = MultiInterp(InArr, InMask, (nout, nout), pos_offset, mat, rsearch, samp, C)

    # Now generate the expected output
    TargetArr = np.zeros((nf2, nout, nout))
    xo, yo = np.meshgrid(np.linspace(0, nout - 1, nout), np.linspace(0, nout - 1, nout))
    W = np.exp(-2 * np.pi**2 * (u0**2 * C[0] + 2 * u0 * v0 * C[1] + v0**2 * C[2]))
    for j in range(nf):
        TargetArr[j, :, :] = 1.0 + 0.1 * np.cos(
            2
            * np.pi
            * (
                (mat[0][0] * xo + mat[0][1] * yo + pos_offset[0]) * u0
                + (mat[1][0] * xo + mat[1][1] * yo + pos_offset[1]) * v0
            )
            / 2.0**j
        ) * W ** (0.25**j)
    TargetArr[-2, :, :] = np.sum(TargetArr[:-2, :, :], axis=0) - 3.6
    for k in range(64):
        xc = 200 + 150 * np.cos(k / 32 * np.pi)
        yc = 170 + 150 * np.sin(k / 32 * np.pi)
        v = np.array([xc, yc]) - pos_offset
        tt = np.linalg.solve(mat, v)
        xt = tt[0]
        yt = tt[1]
        TargetArr[-1, :, :] += np.exp(
            -0.5 * ((xo - xt) ** 2 + (yo - yt) ** 2) / (sigma / sc) ** 2
        ) / np.linalg.det(mat / sc)
    diff = np.where(OutMask, 0.0, OutArr - TargetArr).astype(np.float32)
    # print(np.amax(np.abs(diff), axis=(1,2)))
    # fits.PrimaryHDU(diff).writeto('diff.fits')
    # <-- need to import astropy.io.fits if you want to look at this

    # Check that the differences are within expected tolerances for these settings.
    assert np.amax(np.abs(diff)) < 4e-5
    assert np.amax(np.abs(diff[1, :, :])) < 1e-5
    assert np.amax(np.abs(diff[-1, :, :])) < 1e-5
    return


def test_metamosaic(tmp_path):
    """Test

    Parameters
    ----------
    tmp_path : str or str-like
        Directory for the tests.

    Returns
    -------
    None

    """

    # get the test file from the wiki
    tmp_dir = str(tmp_path)
    urllib.request.urlretrieve(EXAMPLE_FILE, tmp_dir + "/test_F_02_11.fits")
    mosaic = MetaMosaic(tmp_dir + "/test_F_02_11.fits", bbox=[2, 3, 11, 12], verbose=True)

    # mask a region
    mosaic.mask_caps([9.686], [-44.110], [0.0005])

    rot = 20 * np.pi / 180.0
    im0 = mosaic.shearimage(
        1200, jac=[[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]], psfgrow=1.02, oversamp=1.0
    )
    shearimage_to_fits(im0, tmp_dir + "/xdist.fits", overwrite=True)

    # check the star in the file
    x_ = 694
    y_ = 1014
    layer_ = 4

    with fits.open(tmp_dir + "/xdist.fits") as f:
        # check the moments of the postage stamp of a star
        im = f[0].data[layer_, y_ - 10 : y_ + 11, x_ - 10 : x_ + 11]
        my_moments = galsim.hsm.FindAdaptiveMom(galsim.Image(im))
        assert np.abs(my_moments.moments_sigma - 2.2453365) < 0.01
        assert np.abs(my_moments.moments_centroid.x - 10.74) < 0.03
        assert np.abs(my_moments.moments_centroid.y - 10.53) < 0.03
        assert np.abs(my_moments.observed_shape.g1) < 5e-4
        assert np.abs(my_moments.observed_shape.g2) < 5e-4

        # and check the WCS
        outwcs = WCS(f[0].header)
        sky = outwcs.wcs_pix2world(np.array([[x_, y_, layer_]]), 0)[0, :]
        assert np.abs(sky[0] - 9.68784117) < 1e-5
        assert np.abs(sky[1] + 44.09921653) < 1e-5
        assert np.abs(sky[2] - 5) < 1e-5

    # see if the mask is correct
    assert np.all(
        im0["mask"][287, :320:20]
        == np.array(
            [
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                False,
                False,
            ]
        )
    )

    # test function to extract original image
    L = 1000
    im2 = mosaic.origimage(L, select_layers=[layer_])
    assert im2["layers"] == ["gsstar14"]
    assert np.abs(im2["psf_fwhm"] - 0.24) < 1e-6
    with fits.open(tmp_dir + "/test_F_02_11.fits") as f_orig:
        Nx = f_orig[0].header["NAXIS1"]
        Ny = f_orig[0].header["NAXIS1"]
        wx = (Nx - L) // 2
        wy = (Ny - L) // 2

        # these should be the same to machine precision
        diff = f_orig[0].data[0, layer_, wy:-wy, wx:-wx] - im2["image"][0, :, :]
        assert np.amax(np.abs(diff)) < 1e-7

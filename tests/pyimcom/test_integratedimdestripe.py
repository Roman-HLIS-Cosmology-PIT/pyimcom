"""Integrated test for imdestripe."""

import gzip
import os
import re
import shutil
import sys
import warnings
from urllib.request import urlretrieve

import asdf
import numpy as np
from asdf.exceptions import AsdfConversionWarning, AsdfPackageVersionWarning
from astropy.io import fits
from pyimcom import imdestripe
from pyimcom.config import Config
from pyimcom.config import Settings as Stn
from pyimcom.wcsutil import LocWCS

# disable some asdf warnings
warnings.filterwarnings("ignore", category=AsdfConversionWarning)
warnings.filterwarnings("ignore", category=AsdfPackageVersionWarning)

# Example configuration file.
# Note that we will replace $TMPDIR.
# The only subfolders that are actually used in this test are:
#
# - $TMPDIR/L2/
# - $TMPDIR/fits-F/
# - $TMPDIR/ds-F/

EXAMPLE_CONFIG = """{
    "OBSFILE": "$TMPDIR/Roman_WAS_obseq_11_1_23.fits",
    "INDATA": [
        "$TMPDIR/L2/",
        "L2_2506"
    ],
    "TILESCHM": "FourSquare-Dec2025",
    "RERUN": "06",
    "MOSAIC": 1,
    "CTR": [
	9.55,
	-44.1
    ],
    "LONPOLE": 180.0,
    "OUTSIZE": [
        60,
	34,
	0.049019607843137254
    ],
    "BLOCK": 40,
    "FILTER": 1,
    "LAKERNEL": "Cholesky",
    "KAPPAC": [
         5e-4
    ],
    "INPSF": [
	"$TMPDIR/psf",
        "L2_2506",
        6
    ],
    "EXTRAINPUT": [
        "truth,0.004906087669824225",
        "gsstar14"
    ],
    "PADSIDES": "all",
    "OUTMAPS": "USTN",
    "OUT": "$TMPDIR/imtest-F1",
    "INPAD": 0.70,
    "NPIXPSF": 18,
    "FADE": 1,
    "PAD": 1,
    "NOUT": 1,
    "OUTPSF": "GAUSSIAN",
    "EXTRASMOOTH": 0.9265328730414752,
    "TEMPFILE": "$TMPDIR/pyimcomrun_2X",
    "INLAYERCACHE": "$TMPDIR/r4",
    "PSFINTERP": "G4460",
    "PSFSPLIT": [5.25, 8.75, 0.01],
    "DSMODEL": ["constant",
        4088],
    "DSOBSFILE": "$TMPDIR/fits-F/tempimage_",
    "DSOUT": ["$TMPDIR/ds-F/",
        "_ds_out.txt"
    ],
    "CGMODEL": ["PR",
        3,
        1e-3
        ],
    "DSCOST": ["quadratic",
        0,
        0
        ]
}
"""


def _collect(files, newloc):
    """Collects files for the test, sends to `newloc`."""

    for f in files:
        urlretrieve(
            "https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/wiki/test-files/imdestripe/" + f,
            newloc + "/" + f,
        )
        if f[-3:] == ".gz":
            f_final = newloc + "/" + f[:-3]
            with gzip.open(f_final + ".gz", "rb") as f1, open(f_final, "wb") as f2:
                shutil.copyfileobj(f1, f2)
            os.remove(f_final + ".gz")


def _setup_one_file(args):
    """
    Converts one file + ancillary information to FITS.
    """

    fp, outprefix, N, noiseid, wcs_order, verbose = args

    # convert the file to FITS, including 4th order TAN-SIP approximation
    outfile = outprefix + fp[1] + ".fits"
    with asdf.open(fp[0]) as input_file:
        # the WCS
        this_wcs = LocWCS(input_file["roman"]["meta"]["wcs"], N=N)  # Stn.sca_nside)
        this_wcs.wcs_approx_sip(p_order=wcs_order)

        # the main data
        if noiseid is None:
            phdu = fits.PrimaryHDU(
                input_file["roman"]["data"], header=this_wcs.approx_wcs.to_header(relax=True)
            )
        else:
            if verbose:
                print("Getting noise from", fp[0][:-5] + "_noise.asdf")
                sys.stdout.flush()
            with asdf.open(fp[0][:-5] + "_noise.asdf") as noise_file:
                phdu = fits.PrimaryHDU(
                    input_file["roman"]["data"] + noise_file["noise"][noiseid, :, :].astype(np.float32),
                    header=this_wcs.approx_wcs.to_header(relax=True),
                )

        # the mask
        mfname = fp[0][:-5] + "_mask.fits"
        with fits.open(mfname) as mf:
            if verbose:
                print(">>", outfile)
                print("    mask file:", mfname)
                print("    max WCS polynomial fit error:", this_wcs.wcs_max_err, "pix")
                sys.stdout.flush()
            fits.HDUList([phdu, mf[1]]).writeto(outfile, overwrite=True)


def _setup_all_files(fprefix, outprefix, max_files=None, wcs_order=4, noiseid=None, verbose=False):
    """
    Gets all the files starting with the specified format.

    The outputs are written as fits files starting with `outprefix`.

    Parameters
    ----------
    fprefix : str
        The prefix for the file names. Files will be accepted if they are of the
        format fprefix + (numbers and underscores) + ".asdf".
    outprefix : str
        The prefix for the output file names. Files will be written to
        outprefix + (same numbers and undercores) + ".fits".
    max_files : int or None, optional
        If provided, sets a maximum number of files (for testing).
    wcs_order : int, optional
        The polynomial fitting order for the TAN-SIP WCS.
    noiseid : int, optional
        If specified, which noise layer to add from the
        fprefix + (numbers and underscores) + "_noise.asdf" file.
    verbose : bool, optional
        If True, print lots of data to the output.

    Returns
    -------
    list of (str, str)
        The list of file names selected and substrings ("obsid_sca").

    """

    # get the input files to use. here use_files is a list of ordered pairs
    # (file, identifier)
    fdir, fileprefix = os.path.split(fprefix)
    n = len(fileprefix)
    numus = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "_"}
    use_files = []
    for f in os.listdir(fdir):
        if len(use_files) == max_files:
            break
        if f[:n] == fileprefix and f[-5:] == ".asdf" and all(c in numus for c in f[n:-5]):
            use_files.append((os.path.join(fdir, f), f[n:-5]))

    for fp in use_files:
        _setup_one_file((fp, outprefix, Stn.sca_nside, noiseid, wcs_order, verbose))

    return use_files


def test_integrated(tmp_path):
    """Integrated test for imdestripe."""

    tmp_path = str(tmp_path)  # convert to string

    # first, get the configuration file.
    with open(tmp_path + "/cfg.txt", "w") as f:
        f.write(EXAMPLE_CONFIG.replace("$TMPDIR", str(tmp_path)))

    # now download the files that imdestripe needs
    os.makedirs(tmp_path + "/L2", exist_ok=True)
    cpath = [
        "sim_L2_F184_1433_11.asdf",
        "sim_L2_F184_14844_6.asdf",
        "sim_L2_F184_1433_12.asdf",
        "sim_L2_F184_1433_11_noise.asdf",
        "sim_L2_F184_14844_6_noise.asdf",
        "sim_L2_F184_1433_12_noise.asdf",
        "sim_L2_F184_1433_11_mask.fits.gz",
        "sim_L2_F184_14844_6_mask.fits.gz",
        "sim_L2_F184_1433_12_mask.fits.gz",
    ]
    _collect(
        cpath,
        tmp_path + "/L2",
    )

    # make directories for imdestripe
    os.makedirs(tmp_path + "/ds-F", exist_ok=True)
    os.makedirs(tmp_path + "/fits-F", exist_ok=True)

    # move files to the new directory
    # this one adds the noise from realization #1 (i.e., the last one since we start at 0)
    _setup_all_files(tmp_path + "/L2/sim_L2_F184_", tmp_path + "/fits-F/tempimage_F184_", noiseid=1)

    # run imdestripe command
    imdestripe.main(cfg_file=tmp_path + "/cfg.txt", of=tmp_path + "/output.log")

    # check overlap matrix
    mtarget = np.array([[1.0, 0.0, 0.05942073], [0.0, 1.0, 0.33928332], [0.05942073, 0.33928332, 1.0]])
    mat = np.load(tmp_path + "/ds-F/ovmat.npy")
    assert np.all(np.abs(mat - mtarget) < 0.02)

    # check log
    with open(tmp_path + "/output.log", "r") as f:
        lines = f.readlines()
    # did we get a final residual?
    resfinal = -1
    for line in lines:
        m = re.match(r"Final norm\: ([\d\.eE-]+)", line)
        if m:
            resfinal = float(m.group(1))
    print("Final -->", resfinal)
    assert 1.0 < resfinal < 600.0
    # after first iteration
    res1 = -1
    for line in lines:
        m = re.match(r"Current norm\: ([\d\.eE-]+)", line)
        if m:
            res1 = float(m.group(1))
            break
    print("After iter1 -->", res1)
    assert 3000.0 < res1 < 6000.0

    # checks of parameters
    with fits.open(tmp_path + "/ds-F/final_params.fits") as f:
        pars = f[0].data
    assert np.std(pars[0, :]) < 0.002
    assert 0.006 < np.std(pars[1, :]) < 0.01
    assert 0.006 < np.std(pars[2, :]) < 0.01

    # now get the striping parameters
    iqr_new = []
    iqr_old = []
    obsid = [1433, 1433, 14844]
    sca = [11, 12, 6]
    for j in range(3):
        with fits.open(tmp_path + f"/fits-F/tempimage_F184_{obsid[j]:d}_{sca[j]:d}.fits") as f:
            rows = np.median(f[0].data, axis=1)
        iqr_old.append(np.percentile(rows, 75) - np.percentile(rows, 25))
        with fits.open(tmp_path + f"/ds-F/F184_DS_{obsid[j]:d}_{sca[j]:d}.fits") as f:
            rows = np.median(f[0].data, axis=1)
        iqr_new.append(np.percentile(rows, 75) - np.percentile(rows, 25))
    ratio = np.array(iqr_new) / np.array(iqr_old)
    print(ratio)
    assert np.all(ratio > 0.1)
    assert ratio[0] < 1.02
    assert ratio[1] < 0.6
    assert ratio[2] < 0.6

    # subtraction test
    sca = imdestripe.Sca_img("1433", "11", Config(tmp_path + "/cfg.txt"))
    x = np.median(sca.image, axis=1)[7:10]

    # make container for an empty function that acts like a parameter
    class _EmptyClass:
        pass

    p = _EmptyClass()

    def _f(q):
        im = np.zeros((4088, 4088), dtype=np.float32)
        im[8, :] = 0.03
        return im

    p.forward_par = _f
    sca.subtract_parameters(p, 0)
    x2 = x - np.median(sca.image, axis=1)[7:10]
    print(x2)
    assert -0.001 < x2[0] < 0.001
    assert 0.029 < x2[1] < 0.031
    assert -0.001 < x2[2] < 0.001
    assert sca.params_subtracted

    # now clear old files (this part also asserts that they exist!)
    delfiles = [
        "ds-F/F184_DS_14844_6.fits",
        "ds-F/F184_DS_1433_12.fits",
        "ds-F/F184_DS_1433_11.fits",
        "ds-F/final_params.fits",
        "ds-F/cg_log.csv",
        "ds-F/ovmat.npy",
        "fits-F/tempimage_F184_1433_12.fits",
        "fits-F/tempimage_F184_14844_6.fits",
        "fits-F/tempimage_F184_1433_11.fits",
    ]
    for df in delfiles:
        os.remove(tmp_path + "/" + df)
    for df in cpath:
        df2 = df[:-3] if df[-3:] == ".gz" else df
        os.remove(tmp_path + "/L2/" + df2)

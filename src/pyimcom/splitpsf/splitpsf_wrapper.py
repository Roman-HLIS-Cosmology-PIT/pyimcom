import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from astropy.io import fits

from ..coadd import InImage
from ..config import Settings
from ..layer import _get_sca_imagefile
from .splitpsf import split_psf_to_fits


def split_psf_single(cfg_dict, iobs, filter, targetdir, psfsplit_pars, TEST_FILES=None):
    """
    Run Split PSF on one observation

    Parameters
    ----------
    cfg_dict : dict
        The configuration dictionary containing all necessary parameters for the split PSF process.
    iobs : int
        The index of the observation to process.
    filter : str
        The filter name for the observation.
    targetdir : str
        The directory where the output PSF files will be saved.
    psfsplit_pars : dict
        The PSF splitting parameters.
    TEST_FILES : list of strs, optional
        List of [ "path/to//PSF/file", "wcs/format", "path/to/output/file" ] FOR TESTING ONLY
    """

    if TEST_FILES is not None:
        import urllib.request

        psf_file = TEST_FILES[0]
        sci_filename = TEST_FILES[1]
        if psf_file.startswith("http"):
            psf_file2 = TEST_FILES[2][:-5] + "_temp_psf.fits"
            urllib.request.urlretrieve(psf_file, psf_file2)
            psf_file = psf_file2
    else:
        psf_file = cfg_dict["INPSF"][0] + "/" + InImage.psf_filename(cfg_dict["INPSF"][1], iobs)
        sci_filename = _get_sca_imagefile(cfg_dict["INDATA"][0], (iobs, -1), filter, cfg_dict["INPSF"][1])

    if os.path.exists(psf_file):
        outfile = TEST_FILES[2] if TEST_FILES is not None else targetdir + f"/psf_{iobs:d}.fits"
        print("File is at " + psf_file, "-->", outfile)
        print("   sci in =", sci_filename)
        split_psf_to_fits(
            psf_file,
            sci_filename,
            psfsplit_pars,
            outfile,
        )


def split_psf_all(cfg, workers, max_observations=np.inf):
    """
    Run Split PSF on all observations

    Parameters
    ----------
    cfg : pyimcom.config.Config
        The configuration object containing all necessary parameters for the split PSF process.
    workers : int
        The number of worker processes to use for parallel processing.
    max_observations : int, optional
        The maximum number of observations to process (for testing purposes).
        Default is infinity, meaning all observations will be processed
    """
    # Convert config to dictionary
    cfg_dict = cfg.to_dict()

    if "INLAYERCACHE" not in cfg_dict:
        raise KeyError("Couldn't find INLAYERCACHE.")

    # get target PSF properties
    if cfg_dict["OUTPSF"] != "GAUSSIAN":
        raise ValueError("SplitPSF currently only works for Gaussian PSF.")

    sigma = float(cfg_dict["EXTRASMOOTH"])
    print("PSF sigma (input pixels) -->", sigma)

    # get number of rows
    with fits.open(cfg_dict["OBSFILE"]) as f:
        Nobs = f[1].header["NAXIS2"]
        filters_obs = f[1].data["filter"]  # filters_obs[iobs] is the filter for observation iobs
    print(Nobs, "observations to search")
    print(filters_obs)

    # extract oversampling factor
    ovsamp = int(cfg_dict["INPSF"][2])
    print(f"Input PSFs are {ovsamp:f}x oversampled")

    # extract PSF splitting parameters
    r1 = float(cfg_dict["PSFSPLIT"][0])
    r2 = float(cfg_dict["PSFSPLIT"][1])
    epsilon = float(cfg_dict["PSFSPLIT"][2])
    print(r1, r2, epsilon)

    # decide on stamp size; multiple of 8, must include r2 radius
    smallstampsize = int(np.ceil(r2 * ovsamp * 2 + 4))
    smallstampsize += 8 - smallstampsize % 8
    print("chosen stamp size = ", smallstampsize)

    # where to put the files
    targetdir = cfg_dict["INLAYERCACHE"] + ".psf"
    try:
        os.mkdir(targetdir)
        print("made directory -->", targetdir)
    except OSError as error:
        print("Couldn't make directory", targetdir, ":", error)

    use_filter = Settings.RomanFilters[int(cfg_dict["FILTER"])]
    print("Selecting observations from filter", use_filter)
    print("")

    psfsplit_pars = {
        "smallstamp_size": smallstampsize,
        "sigmaGamma": sigma,
        "r_in": r1,
        "r_out": r2,
        "eps": epsilon,
        "SAVEZETA": False,
        "oversamp": ovsamp,
    }
    # Note: 'SAVEZETA': True is for diagnostics/figures only. The zeta HDUs are not actually needed for
    # the calculation.

    count = 0
    #  Set up ProcessPoolExecutor for safety with python 3.12, 3.13, 3.14
    start_method = "forkserver" if os.name.lower() == "posix" else "spawn"
    ctx = mp.get_context(start_method)
    nfail = 0

    # Process Nobs observations in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = []
        for iobs in range(Nobs):
            if filters_obs[iobs] == use_filter:
                futures.append(
                    executor.submit(
                        split_psf_single, cfg_dict, iobs, filters_obs[iobs], targetdir, psfsplit_pars
                    )
                )

        # Wait for all futures to complete and handle any exceptions
        for future in as_completed(futures):
            count += 1
            if count == max_observations:
                break
            try:
                future.result()  # This will raise an exception if the function raised one
            except Exception as e:
                nfail += 1
                print(f"Worker failed with exception: {e}", flush=True)

    if nfail > 0:
        raise Exception(f"{nfail:d} instances of split_psf_single failed.")

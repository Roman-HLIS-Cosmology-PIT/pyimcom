import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# import from furry_parakeet
from ..config import Config

# local imports
from .imsubtract import run_imsubtract_single


def run_imsubtract_all(cfg_file, workers=4, max_imgs=None, display=None, local_output=False, mmap=None):
    """
    Main routine to run imsubtract on all images in the cache.

    Parameters
    ----------
    cfg_file: str
        Path to the config file.
    workers: int, optional
        Number of workers to use for parallel processing. Default is 4.
    max_imgs: int, optional
        If provided, does computations for a maximum number of SCAs. Most users will
        want the default of None; this is provided mainly for testing.
    display: str or None, optional
        Display location for intermediate steps. Default is None.
    local_output: bool, optional
        Whether to direct the file to local output instead of the cache directory.
    mmap : str or str-like, optional
        Directory to put temporary mmap files.
    """

    # Additional imports
    import multiprocessing as mp
    import traceback

    # load the file using Config and get information
    cfgdata = Config(cfg_file)

    cacheinfo = cfgdata.inlayercache

    # separate the path from the inlayercache info
    m = re.search(r"^(.*)\/(.*)", cacheinfo)
    if m:
        path = m.group(1)
        stem = m.group(2)

    # create empty list of exposures
    exps = []

    # find all the fits files and add them to the list
    for _, _, files in os.walk(path):
        for file in files:
            if file.startswith(stem) and file.endswith(".fits") and file[-6].isdigit():
                exps.append(file)

    # print("List of exposures:", exps)

    # Run imsubtract on each exposure in parallel using ProcessPoolExecutor
    count = 0
    start_method = "forkserver" if os.name.lower() == "posix" else "spawn"
    ctx = mp.get_context(start_method)
    nfail = 0

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = []
        for exp in exps:
            if max_imgs is not None and count > max_imgs:
                break
            m2 = re.search(r"(\w*)_0*(\d*)_(\d*).fits", exp)
            if m2:
                obsid = int(m2.group(2))
                scaid = int(m2.group(3))
                futures.append(
                    executor.submit(
                        run_imsubtract_single,
                        cfgdata,
                        scaid,
                        obsid,
                        path,
                        exp,
                        display=display,
                        fft_workers=None,
                        local_output=local_output,
                        max_layers=max_imgs,
                        mmap=mmap,
                    )
                )
                count += 1

        # Wait for all futures to complete
        for future in as_completed(futures):
            try:
                future.result()
                print(f"Completed {count}/{len(futures)}", flush=True)

            except Exception as e:
                nfail += 1
                print(f"Worker failed with exception {e}", flush=True)
                traceback.print_exc()

    if nfail > 0:
        print(f"{nfail}/{len(futures)} instances of run_imsubtract_single failed.", flush=True)

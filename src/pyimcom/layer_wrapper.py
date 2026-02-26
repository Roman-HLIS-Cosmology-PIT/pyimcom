"""
Wrapper functions to build layers.

Functions
---------
build_one_layer
    Build layers for one InImage
build_all_layers
    Build layers for all the exposures in the directory path

"""

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import coadd
from .layer import get_all_data


def build_one_layer(cfg, idsca):
    """
    Build layers for one InImage
    Instantiates the inimage with a "dummy" block 0

    Parameters
    ----------
    cfg : pyimcom.config.Config object
        the configuration file
    idsca : tuple, (int, int)
        the obsid, scaid pair of the image to build

    Returns
    -------
    None

    """

    block_zero = coadd.Block(cfg, this_sub=0, run_coadd=False)
    print("block ->", block_zero)
    block_zero.parse_config()  # This loads the observation table, which is needed next.
    print("-->", block_zero.cfg, block_zero.obsdata)
    inimage = coadd.InImage(block_zero, idsca)
    print(inimage.idsca, inimage.inwcs, inimage.blk, inimage.blk.cfg, inimage.blk.obsdata)
    assert idsca[0] == "-1"

    get_all_data(inimage)

    return None


def build_all_layers(cfg, workers=2):
    """
    Build layers for all the exposures in the directory path

    Parameters
    ----------
    cfg : pyimcom.config.Config object
        the configuration file
    workers : int
        number of workers to use in parallel processing

    """

    m = re.search(r"^(.*)\/(.*)", cfg.inpath)
    if m:
        path = m.group(1)
        exp = m.group(2)

    idsca_list = []

    for _, _, files in os.walk(path):
        for file in files:
            if file.startswith(exp):
                m = re.search(r"_(\d+)_(\d+)\.(fits|asdf)$", file[len(exp) :])
                if m:
                    idsca_list.append((int(m.group(1)), int(m.group(2))))

    print(idsca_list)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for idsca in idsca_list:
            futures.append(executor.submit(build_one_layer, cfg, idsca))

        # Wait for all futures to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Worker failed with exception {e}", flush=True)

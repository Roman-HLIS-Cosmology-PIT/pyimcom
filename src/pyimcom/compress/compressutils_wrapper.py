import multiprocessing as mp
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from .compressutils import CompressedOutput


def compress_one_block(cfg, layer_pars_dic, ibx, iby):
    """Helper function to compress one block of the IMCOM output.

    Parameters
    ----------
    cfg : pyimcom.config.Config
        The IMCOM configuration.
    layer_pars_dic : dict
        A dictionary of layer parameters, where the keys are the layer names and the values are
        dictionaries of parameters for that layer.
    ibx : int
        Block x index.
    iby : int
        Block y index.

    Returns
    -------
    fout : str
        The file name of the compressed output file for this block.

    """
    fname = cfg.outstem + f"_{ibx:02d}_{iby:02d}.fits"
    fout = cfg.outstem + f"_{ibx:02d}_{iby:02d}.cpr.fits.gz"
    print(fname, "-->", fout)
    sys.stdout.flush()

    with CompressedOutput(fname) as f:
        # Get types of each layer (e.g., 'gsstar', 'cstar', etc.)
        # These match the keys in config.yaml and will set the pars
        layers_types = [re.sub(r"\d+$", "", item.split(",")[0]) for item in f.cfg.extrainput]

        # Loop over layers
        for j in range(1, len(f.cfg.extrainput)):
            f.compress_layer(j, scheme="I24B", pars=layer_pars_dic[layers_types[j]])

        f.to_file(fout, overwrite=True)

    return fout


def compress_all_blocks(cfg, layer_pars_dic, workers):
    """
    Helper function to compress all blocks of the IMCOM output in parallel.

    Parameters
    ----------
    cfg : pyimcom.config.Config
        The IMCOM configuration.
    layer_pars_dic : dict
        A dictionary of layer parameters, where the keys are the layer names and the values are
        dictionaries of parameters for that layer.
    workers : int
        The number of parallel workers to use for compression.
    Returns
    -------
    None

    """
    nblock = cfg.nblock
    nblock2 = cfg.nblock**2

    start_method = "forkserver" if os.name.lower() == "posix" else "spawn"
    ctx = mp.get_context(start_method)
    nfail = 0

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = []
        for i in range(nblock2):
            ibx = i % nblock
            iby = i // nblock
            futures.append(executor.submit(compress_one_block, cfg, layer_pars_dic, ibx, iby))

        for future in as_completed(futures):
            # Check for existence of output file to confirm completion
            try:
                fout = future.result()
                if not os.path.exists(fout):
                    print(f"Error: {fout} was not created.")
            except Exception as e:
                nfail += 1
                print(f"Worker failed with exception {e}", flush=True)

    if nfail > 0:
        raise Exception(f"{nfail:d} instances of compress_one_block failed.")

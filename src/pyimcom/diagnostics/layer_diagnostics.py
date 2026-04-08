"""
Report section for layer diagnostics.

Classes
-------
LayerReport
    Layer report section.

"""

import concurrent.futures
import contextlib
import gc
import os
import sys
from os.path import exists

import numpy as np

from ..compress.compressutils import ReadFile
from .report import ReportSection


def _percentiles_and_delete(arr, pctiles, target, delete_arr):
    """
    Sorts/constructs percentiles from a memmapped array, then (optionally) deletes it.

    Parameters
    ----------
    arr : np.memmap
        The memory-mapped array.
    pctiles : array-like
        The percentiles to generate.
    target : np.ndarray
        The array to save the percentiles.
    delete_arr : bool
        Whether to delete the array when done.

    Returns
    -------
    None

    """

    # Begin by sorting the array in place
    arr.sort(kind="mergesort")

    # Read percentiles from sorted array
    npc = len(pctiles)
    nsize = arr.size
    for k in range(npc):
        pos = (nsize - 1) * pctiles[k] / 100.0
        p1 = int(np.floor(pos))
        if p1 < 0:
            p1 = 0
        if p1 >= nsize - 1:
            p1 = nsize - 2
        frac = np.clip(pos - p1, 0.0, 1.0)
        target[k] = (1 - frac) * arr[p1] + frac * arr[p1 + 1]

    if delete_arr:
        fn = arr.filename
        del arr
        with contextlib.suppress(FileNotFoundError):
            os.remove(fn)


class LayerReport(ReportSection):
    """
    The layer section of the report.

    Inherits from pyimcom.diagnostics.report.ReportSection. Overrides build.

    """

    def build(self, nblockmax=100):
        """
        Generates the LaTeX for the statistical report on layers.

        Parameters
        ----------
        nblockmax : int, optional
            The maximum number of blocks on a side to allow.

        Returns
        -------
        None

        """

        self.tex += "\\section{Layer statistics}\n\n"

        # figure out which layers we have
        layers = self.cfg.extrainput + []
        layers[0] = "SCI"
        nlayers = len(layers)

        # dimensions
        ns = self.cfg.Nside  # the side length of the unique area
        d = self.cfg.postage_pad * self.cfg.n2  # the gap between the unique area and block edge
        nblock = min(self.cfg.nblock, nblockmax)  # block dimension

        # percentiles to use
        pctiles = [0, 0.01, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 99.99, 100]
        npc = len(pctiles)
        pcarray = np.zeros((nlayers, npc), dtype=np.float32)

        nsize = (ns * nblock) ** 2
        data = [None] * nlayers
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
            jobs = [None] * nlayers
            for ilayer in range(nlayers):
                # get data
                if self.tmp_dir is None:
                    data[ilayer] = np.zeros((nsize,), dtype=np.float32)
                else:
                    data[ilayer] = np.memmap(
                        str(self.tmp_dir) + f"/layer_{ilayer:d}.npy",
                        dtype="float32",
                        mode="w+",
                        shape=((nsize,)),
                    )
                    data[ilayer][:] = 0.0

                for iby in range(nblock):

                    def _load_row(arg):
                        # not binding iby and ilayer is fine since the function is used and removed in the
                        # loop
                        (ibx, _data) = arg
                        print(f"building layer {ilayer:2d}, block ({ibx:2d},{iby:2d}) ")  # noqa: B023
                        sys.stdout.flush()
                        infile = self.infile(ibx, iby)  # noqa: B023
                        if not exists(infile):
                            return None
                        chunk = iby * nblock + ibx  # noqa: B023
                        with ReadFile(infile, layers=[ilayer]) as f:  # noqa: B023
                            x_ = f[0].data[0, ilayer, :, :]  # noqa: B023
                            if d > 0:
                                x_ = x_[d:-d, d:-d]
                            _data[chunk * ns * ns : (chunk + 1) * ns * ns] = x_.ravel()

                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:
                        e.map(_load_row, [(ix, data[ilayer]) for ix in range(nblock)])

                    del _load_row

                # wait for previous layer
                if ilayer > 0:
                    jobs[ilayer - 1].result()
                    data[ilayer - 1] = None
                    gc.collect()
                    print("percentiles", ilayer - 1, pcarray[ilayer - 1, :])  # remove from final version
                    sys.stdout.flush()

                # get percentiles --- now uses sorting method since this works out of core
                jobs[ilayer] = exe.submit(
                    _percentiles_and_delete, data[ilayer], pctiles, pcarray[ilayer, :], True
                )
                # data.sort(kind="mergesort")
                # for k in range(npc):
                #     pos = (nsize - 1) * pctiles[k] / 100.0
                #     p1 = int(np.floor(pos))
                #     if p1 < 0:
                #         p1 = 0
                #     if p1 >= nsize - 1:
                #         p1 = nsize - 2
                #     frac = np.clip(pos - p1, 0.0, 1.0)
                #     pcarray[ilayer, k] = (1 - frac) * data[p1] + frac * data[p1 + 1]
                # del data

            # wait for the last one
            jobs[-1].result()
            del data
            del jobs
        gc.collect()

        # now build table, in segments of size up to ncolmax
        ncolmax = 6
        self.tex += "The percentiles of the various layers are included in the table below."
        self.tex += " Note that a maximum of " + str(ncolmax) + "layers are shown in each table"
        self.tex += " to preserve horizontal space.\n\n"
        self.tex += "The layers are:\n\\begin{itemize}\n"
        for il in range(nlayers):
            self.tex += "\\item"
            self.tex += f" [{il:d}] "
            self.tex += "{\\tt " + str(layers[il]) + "}"
            if il != nlayers - 1:
                self.tex += ";"
                if il == nlayers - 2:
                    self.tex += " and"
                self.tex += "\n"
        self.tex += ".\n\\end{itemize}\n"
        start = 0
        while start < nlayers:
            cols = list(range(start, min(start + ncolmax, nlayers)))
            self.data += "# PCTILE |"
            for il in cols:
                self.data += f"  LAYER {il:02d} "
            self.data += "\n"
            for k in range(npc):
                self.data += f"{pctiles[k]:7.3f}   "
                for il in cols:
                    self.data += f" {pcarray[il, k]:10.3E}"
                self.data += "\n"
            self.data += "\n"
            start += ncolmax

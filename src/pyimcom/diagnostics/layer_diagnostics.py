"""
Report section for layer diagnostics.

Classes
-------
LayerReport
    Layer report section.

"""

import concurrent.futures
import sys
from os.path import exists

import numpy as np

from ..compress.compressutils import ReadFile
from .report import ReportSection


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

        for ilayer in range(nlayers):
            # get data
            data = np.zeros((ns * nblock, ns * nblock), dtype=np.float32)
            for iby in range(nblock):

                def _load_row(arg):
                    # not binding iby and ilayer is fine since the function is used and removed in the loop
                    (ibx, _data) = arg
                    print(f"building layer {ilayer:2d}, block ({ibx:2d},{iby:2d}) ")  # noqa: B023
                    sys.stdout.flush()
                    infile = self.infile(ibx, iby)  # noqa: B023
                    if not exists(infile):
                        return None
                    with ReadFile(infile) as f:
                        _data[ns * iby : ns * (iby + 1), ns * ibx : ns * (ibx + 1)] = f[  # noqa: B023
                            0
                        ].data[
                            0, ilayer, d:-d, d:-d  # noqa: B023
                        ]

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:
                    e.map(_load_row, [(ix, data) for ix in range(nblock)])

                del _load_row

            # get percentiles
            for k in range(npc):
                pcarray[ilayer, k] = np.percentile(data, pctiles[k])
            del data

        # now build table, in segments of size up to ncolmax
        ncolmax = 6
        self.tex += "The percentiles of the various layers are included in the table below."
        self.tex += " Note that a maximum of " + str(ncolmax) + "layers are shown in each table"
        self.tex += " to preserve horizontal space.\n\n"
        self.tex += "The layers are:\n\\begin{itemize}\n"
        for il in range(nlayers):
            if il == nlayers - 1:
                self.tex += " and "
            self.tex += f" [{il:d}] "
            self.tex += "\\item {\\tt " + str(layers[il]) + "}"
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

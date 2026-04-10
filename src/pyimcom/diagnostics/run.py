"""
This is an example script to generate a report.

Command-line arguments are:
#. Input FITS file
#. Output stem

"""

import sys

from .layer_diagnostics import LayerReport
from .mosaicimage import MosaicImage
from .noise_diagnostics import NoiseReport
from .report import ValidationReport
from .stars import SimulatedStar


def run_report(input_fits, output_stem="_report", inpath=None):
    """
    Run the report generation.

    Parameters
    ----------
    input_fits : str
        The name of the input FITS file.
    output_stem : str, optional
        The ending stem for the output files. Default is "_report".
    inpath : str, optional
        The path to the input FITS file.
        If None, input_fits should include the relative path. Default is None.

    Returns
    -------
    None
    """
    if inpath is not None:
        input_fits = inpath + "/" + input_fits

    rpt = ValidationReport(input_fits, output_stem, clear_all=True)
    sectionlist = [MosaicImage, LayerReport, SimulatedStar, NoiseReport]
    for cls in sectionlist:
        s = cls(rpt)
        s.build()  # specify nblockmax to do just the lower corner
        rpt.addsections([s])
        del s
    rpt.compile()

    print("--> pdflatex log -->")
    print(str(rpt.compileproc.stdout))


if __name__ == "__main__":
    tmp_dir = sys.argv[3] if len(sys.argv) > 3 else None
    rpt = ValidationReport(sys.argv[1], sys.argv[2], clear_all=True, tmp_dir=tmp_dir)
    sectionlist = [MosaicImage, LayerReport, SimulatedStar, NoiseReport]
    sectionlist = [SimulatedStar]  # <-- TAKE THIS OUT!!!!!
    for cls in sectionlist:
        s = cls(rpt)
        s.build()  # specify nblockmax to do just the lower corner
        rpt.addsections([s])
        del s
    rpt.compile()

    print("--> pdflatex log -->")
    print(str(rpt.compileproc.stdout))

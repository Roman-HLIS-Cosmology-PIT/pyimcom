PyIMCOM Report module
############################

This is the report module for PyIMCOM outputs. It generates a report covering the performance of a PyIMCOM run, including information about layers, noise, simulations, mosiac, and validation. 

Overview
**********

There are several components to generating the report. All the relevant parts are included inside the ``pyimcom.diagnostics`` folder. 
  
Base classes are defined in ``pyimcom.diagnostics.report``, along with the ``ValidationReport`` section. Then more specific report categories, which are built on the base classes, are defined inside ``layer_diagnostics.py``, ``mosaicimage.py``, ``noise_diagnostics.py``, and ''stars.py``.

The module is run once per mosiac using ``pyimcom.diagnostics.run``, which creates the report file (with sections MosaicImage, LayerReport, SimulatedStar, NoiseReport) and compiles it into a PDF.
The format to run the diagnostic report is:
  ``python3 -m pyimcom.diagnostics.run $outstem\_00\_00.cpr.fits.gz $tag\_report > $tag-S$njob.txt" ``
where ``outstem`` indicates the location of the pyimcom output files (one output file is needed for information in the FITS header and the Config file and $tag indicates some output tag.
The default is for the report to be generated with the output stem ``_report.pdf``.

Available Reports
*********************

There are several report sections that are included in the final report generated via ``pyimcom.diagnostics.run``. 

MosaicImage
===========

Information.

LayerReport
===========                                                                 

Information.
                                                                      
SimulatedStar
======

Information.

NoiseReport
======

Information
                                                                      

Tools
*****

A useful utility for exploring compressed files is ``CompressedOutput.get_compression_dict``. You can call it as::

   with CompressedOutput(fname) as f:
        cprs_dict = CompressedOutput.get_compression_dict(f, ilayer)

and this will give the compression scheme in dictionary form used for layer ``ilayer`` (an uncompressed layer returns the empty dictionary ``{}``). Note that the values in this form are strings because the function does not know the expected type; the calling routine must typecast them if needed.

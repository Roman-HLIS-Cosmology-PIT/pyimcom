PyIMCOM Report module
############################

This is the report module for PyIMCOM outputs. It generates a report covering the performance of a PyIMCOM run, including information about layers, noise, simulations, mosiac, and validation. 

Overview
**********

There are several components to generating the report. All the relevant parts are included inside the ``pyimcom.diagnostics`` folder. 
  
Base classes are defined in ``pyimcom.diagnostics.report``, along with the ``ValidationReport`` section. Then more specific report categories, which are built on the base classes, are defined inside ``layer_diagnostics.py``, ``mosaicimage.py``, ``noise_diagnostics.py``, and ''stars.py``.

The module is run once per mosiac using ``pyimcom.diagnostics.run``, which creates the report file (with sections MosaicImage, LayerReport, SimulatedStar, NoiseReport) and compiles it into a PDF.
The bash command to run the diagnostic report is:

.. code-block:: bash

    python3 -m pyimcom.diagnostics.run $outstem\_00_00.cpr.fits.gz $tag\_report > $tag-S$njob.txt

where ``$outstem`` indicates the location of the pyimcom output files (one output file is needed for information in the FITS header and the Config file); ``$tag`` indicates some output tag (this is to tell the script where to save files); and ``$njob`` is a job number (just for where to send STDOUT, it doesn't affect the script itself).
The default is for the report to be generated with the output stem ``_report.pdf``.

Available Reports
*********************

There are several report sections that are included in the final report generated via ``pyimcom.diagnostics.run``. 

MosaicImage
===========
The ``mosaicimage`` module creates a report section called MosaicImage, which inherits from the base class ``ReportSection``.
To build the report section via MosaicImage.build, the class does not require any additional arguments. 
However the user may optionally specify the following arguments:
- ``nblockmax``: an optional int giving the maximum size of mosaic to build. If the given value is larger than nblock in the configuration, the whole mosaic will be built. The default is 100.
- ``srange`` : an optional tuple of ints giving the stretch scale to use for the mosaic image. The default is (-0.1, 100.0)

The build method calls ``pyimcom.pictures.genpic.make_picture_1band`` to create the mosaic image, which is written out to ``self.datastem_mosaic.png``.
The image is then included in the report, and the section is written out.

LayerReport
===========                                                                 

The LayerReport section is built using ``pyimcom.diagnostics.layer_diagnostics``. The module defines the class LayerReport, which inherits from the base class ReportSection. 
To build the report section via LayerReport.build, the class does not require any additional arguments, though it accepts the following optional arguments:
- ``nblockmax``: an optional int giving the maximum number of blocks per side to allow. If the given value is larger than nblock in the configuration, the whole mosaic will be built. The default is 100.

For each layer, the diagnostic calculates the value of the image percentiles: [0, 0.01, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 99.99, 100] and includes those values in a table in the report.

                                                                      
SimulatedStar
=============

The SimulatedStar report section is constructed using the ``pyimcom.diagnostics.stars`` module, which defines the class SimulatedStar, inheriting from the base class ReportSection.
It generates a set of diagnostics for the simulated stars in the PyIMCOM run.

The build method calls the functions:

- ``pyimcom.diagnostics.dynrange.get_dynrange_data``: Takes files from the inpath and writes out hisograms of the estimated dynamic range. The output files are:

  - Histograms:

    - ``outstem +'_sqrtS_hist.dat'``: histogram of noise amplification factor sqrtS
    - ``outstem +'_neff_hist.dat'``: histogram of effective exposure number
    Both of these have a header that indicates the fraction of data that is off scale high.

  - dynamic range file:

    - ``outstem +'_dynrange.dat'``: table of percentiles of noisy star images
    (Columns are radius and [1,5,25,50,75,95,99] percentiles)

- ``pyimcom.diagnostics.starcube_nonoise.gen_starcube_nonoise``: Generates a catalog of injected stars and their properties in the coadd image, and extracts the star cube and the star moments. 

The output files are:
    Star Catalog:
        ``outstem + "_StarCat_galsim.fits"``: images of injected stars in the coadd image, generated using GalSim
        ``outstem + "_StarCat.txt"``: catalog of injected sources and their properties (eg. positions, shapes) 
    Fidelity Histogram:
        ``outstem + "_fidHist.txt"``: histogram of the fidelity of the star images, which is a measure of how well the star images are reconstructed in the coadd image. 
        The histogram reports instances of fidelity integer values in dB.

- ``pyimcom.diagnostics.context_figure.ReportFigContext`` : This is a context manager for report figures in PyIMCOM.


The SimulatedStar build method calls ``gen_dynrange_data`` and ``gen_starcube_nonoise`` to generate the relevant data files.
Then it creates the summary figure for the report by calling the internal function ``_starplot_diagnostic``, which writes out the file ``datastem + _SimulatedStar_all.pdf``, a multi-panel figure which is included in the report.
The section also creates and writes the figure ``datastem + _SimulatedStar_etmap.pdf`` showing the ellipticity and size distribution of the star images.


NoiseReport
===========

The NoiseReport section is constructed using the ``pyimcom.diagnostics.noise_diagnostics`` module, which defines the class NoiseReport, inheriting from the base class ReportSection.
This section shows a report on noise properties of the PyIMCOM run.

The class has several methods:

- ``build_noisespec``: Computes 1 and 2-dimensional noise power spectrum for each block and writes out the file ``<datastem>_<blockID>_ps.fits``,
  which contains 3 HDUs: the 2D power spectrum, the block config, and the 1D power spectrum.
- ``measure_pwoer_spectrum``: Measures the 2D power spectrum for each block. May be windowed and/or binned.
- ``_get_wavenumbers``: Internal method to get the wavenumber arrays for the power spectrum.
- ``azimuthal_average``: Internal method to compute the azimuthal average of the 2D power spectrum to get the 1D power spectrum.
- ``get_powerspectra``: Calculates the azimuthally-averaged 1D power spectrum of the image.
- ``average_spectra``: Averages together all the per-block power spectra in one band.
- ``gen_overview_fig``: Create and write LaTeX code to include an overview figure of the noise power spectra showing the 2D power spectrum in each of the 4 main noise layers.
- ``build``: The main method to build the NoiseReport section, which calls the above methods to compute the power spectra and generate the overview figure, and then writes out the section for inclusion in the report.

  The build method of NoiseReport requires some optional arguments, whose defaults are given:

  -  nblockmax : int, optional. Maximum size of mosaic to build. Default: 100
  -  m_ab : float, optional. Scaling magnitude (not currently used). Default: 23.9
  -  bin_flag : int, optional. Whether to bin? (1 = bin 8x8, 0 = do not bin). Default: 1
  -  alpha : float, optional. Tukey window width for noise power spectrum. Default: 0.9
  -  tarfiles : bool, optional. Generate a tarball of the data files? Default: True
                                                                      

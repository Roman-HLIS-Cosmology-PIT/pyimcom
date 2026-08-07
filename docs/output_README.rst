Reading PyIMCOM output files
############################

There are several options for reading PyIMCOM output files. From "lowest" to "highest" level interaction:

* FITS readers: The PyIMCOM output files conform to the FITS standard (including WCS headers), and can be
  read with any standard FITS reader.

* Compressed files have their non-science layers scrambled. They can be read using the
  ``pyimcom.compress.compressutils.ReadFile`` utility, which can be used just like
  ``astropy.io.fits.open`` (including the context management):

  .. code-block:: python

    from pyimcom.compress.compressutils import ReadFile
    # here fout is the file name that we want to read
    with ReadFile(fout) as f:
       x = f[0].data
       ...

  ``ReadFile`` also supports regular expression formatting for file names, with the block indices and suffix
  separated from the format string by the ``^`` character. For example, you may use:

  .. code-block:: python

    # This reads the file "coadds/H158/Row14/prod_H_18_14_map.fits"
    ReadFile("coadds/H158/Row{1:02d}/prod_H_{0:02d}_{1:02d}^_18_14_map.fits")

  It is even possible to read from the Internet via https:

  .. code-block:: python

    ReadFile("https://irsa.ipac.caltech.edu/data/theory/openuniverse2024/roman/preview/RomanWAS/images/"
      "coadds/H158/Row14/prod_H_18_14_map.fits")

  The other tools listed below are built on top of ``ReadFile``.

* The ``pyimcom.analysis.OutImage`` method builds one output block from the FITS file:

  .. code-block:: python

    from pyimcom.analysis import OutImage
    # here fout is the file name that we want to read
    my_block = OutImage(fout)
    # extracts a layer: 'SCI' is the science image, or you can use other layer names
    sci_image = my_block.get_coadded_layer('SCI')
    # extracts a metadata map
    fidelity_map = my_block.get_output_map('FIDELITY') # options are: 'FIDELITY', 'SIGMA', 'KAPPA', 'INTWTSUM', 'EFFCOVER'
    # computes an inverse variance map
    weight_map = my_block.get_weight_map("noise,Rz4PbrS2C2") #Generate inverse variance weight map according to Appendix A of 2607.09849.

* The ``pyimcom.meta.distortimage.MetaMosaic`` class is the highest-level interface and constructs a sub-mosaic
  from the 3x3 set of blocks centered on the specified file. It can be subarrayed, sheared, masked, etc.:

  .. code-block:: python

    from pyimcom.meta import distortimage
    # here fout is the file name that we want to read
    mosaic = distortimage.MetaMosaic(fout)
    im = mosaic.origimage(4000) # 4000x4000 subarray

  Detailed instructions for this class are on the `Meta Readme <meta_README.rst>`_.

Examples
========

Some examples incorporating the above ideas:

* `Read a block output from the IPAC OpenUniverse 2024 page, and return both the PyIMCOM output and a
  sheared version <../examples/read_and_shear_output_from_web.py>`_.

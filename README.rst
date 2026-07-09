.. image:: https://codecov.io/gh/Roman-HLIS-Cosmology-PIT/pyimcom/graph/badge.svg?token=GLM6LWD3F7

PyIMCOM: Image combination package
##################################

This is the Python implementation of Image Combination (PyIMCOM) package being developed for the Roman Space Telescope "Cosmology with the High Latitude Survey" Project Infrastructure Team.

Documentation on this site
**************************

See also the `readthedocs page <https://pyimcom.readthedocs.io/en/latest/autoapi/index.html>`_.

Requirements
------------

PyIMCOM runs with Python 3.12 and higher.

Required dependencies should install when you run ``pip``. Pip will automatically install the C kernel `furry-parakeet <https://github.com/Roman-HLIS-Cosmology-PIT/furry-parakeet>`_. There are optional dependencies for some features:

- If you are running coaddition with PyIMCOM with Piff input files for the PSF (instead of FITS images), you need to install `piff <https://github.com/rmjarvis/Piff>`_. Roman (including interfacing with PyIMCOM) is supported from the main branch (starting June 2026), or starting with v1.7 (when available).

  (Note that Piff is **not** required to *read* PyIMCOM output files, even if they were made using Piff.)

Installing PyIMCOM
------------------

Conda
^^^^^

.. code-block:: bash

    conda install -c conda-forge --file requirements.txt
    pip install .

Pip
^^^

.. code-block:: bash

    pip install .

Overview of PyIMCOM concepts
----------------------------

- `Hierarchy of input and output information <docs/hierarchy.rst>`_

Running PyIMCOM
---------------

- `How to build a mosaic in PyIMCOM <docs/run_README.rst>`_

- `How to write a PyIMCOM configuration file <docs/config_README.rst>`_

- `Input file formats <docs/input_README.rst>`_

- `Image destriping <docs/destripe_README.rst>`_

- `PSF splitting <docs/splitpsf_README.rst>`_

- `How to compress PyIMCOM output files <docs/compress_README.rst>`_

- `How to generate a report after running PyIMCOM <docs/report_README.rst>`_

Post-processing of PyIMCOM outputs
----------------------------------

- `Options for interacting with the PyIMCOM output files <docs/output_README.rst>`_

- Instructions for using the `meta post-processing module <docs/meta_README.rst>`_

Information relevant to various runs we have done in the past
-------------------------------------------------------------

- `Preparing the PSFs for the OpenUniverse simulation <historical/OpenUniverse2024/README.rst>`_

References
**********

- Optimal linear image combination. `Rowe et al. ApJ 741:46 (2011) <https://ui.adsabs.harvard.edu/abs/2011ApJ...741...46R/abstract>`_

- Simulating image coaddition with the Nancy Grace Roman Space Telescope - I. Simulation methodology and general results. `Hirata et al. MNRAS 528:2533 (2024) <https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.2533H/abstract>`_

- Simulating image coaddition with the Nancy Grace Roman Space Telescope - II. Analysis of the simulated images and implications for weak lensing. `Yamamoto et al. MNRAS 528:6680 (2024) <https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.6680Y/abstract>`_

- Analysis of Biasing from Noise from the Nancy Grace Roman Space Telescope: Implications for Weak Lensing. `Laliotis et al. PASP 136:124506 (2024) <https://ui.adsabs.harvard.edu/abs/2024PASP..136l4506L/abstract>`_

- Simulating image coaddition with the Nancy Grace Roman Space Telescope: III. Software improvements and new linear algebra strategies. `Cao et al., ApJS 277:55 (2025) <https://ui.adsabs.harvard.edu/abs/2025ApJS..277...55C/abstract>`_

- OpenUniverse2024: A shared, simulated view of the sky for the next generation of cosmological surveys. `OpenUniverse et al. MNRAS 544:3799 (2025) <https://ui.adsabs.harvard.edu/abs/2025MNRAS.544.3799O/abstract>`_

- Simulating image coaddition with the Nancy Grace Roman Space Telescope. IV. Hyperparameter Optimization and Experimental Features. `Cao et al. ApJ, 998:304 (2026) <https://ui.adsabs.harvard.edu/abs/2026ApJ...998..304C/abstract>`_

- Removing correlated noise stripes from the Nancy Grace Roman Space Telescope survey images. `Laliotis et al. PASJ, 78:810 (2026) <https://ui.adsabs.harvard.edu/abs/2026PASJ...78..810L/abstract>`_

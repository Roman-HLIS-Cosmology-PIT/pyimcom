Image Destriping
###################

imDestripe is an iterative algorithm for removing stripes of noise from images prior to combining them with PyIMCOM. 

The key idea is that we take a single image I_A and, using other overlapping images I_B, I_C, ..., 
and construct a stripe-less version of it by interpolating those overlapping images into an image J_A that
has the same WCS as I_A.
We then subtract J_A from I_A to get a residual image.
The optimal stripes are solved for by minimizing the difference between I_A - J_A and the stripe model.

This happens iteratively using the method of conjugate gradient descent-- an initial guess is made for the stripe model, the residuals are computed, and the stripe model is updated.

The full process is outlined in the flowchart below:

.. image:: destripe_flowchart.png
  :width: 800
  :alt: Workflow for image destriping

For more details on the algorithm, see the `imDestripe paper <https://arxiv.org/pdf/2512.05949>`_.

**Key Features:**

* Supports multiple cost functions (quadratic, absolute value, Huber loss; only quadratic is implemented currently)
* Parallel processing for efficient computation
* Checkpoint/restart capability for long runs
* Handles both FITS and ASDF input formats

Quick start: running the workflow
==================================

The destriping workflow consists of two main steps:

1. **Setup**: Convert ASDF files to FITS with WCS approximations
2. **Destripe**: Run the conjugate gradient optimization

Basic usage
~~~~~~~~~~~

.. code-block:: python

    from pyimcom.destripe import destripe_all_layers

    # Destripe science data and all noise layers
    n_noise_layers = destripe_all_layers('config.json', verbose=True)

This will:

* Read input ASDF files specified in the configuration
* Convert them to temporary FITS files
* Run the destriping algorithm on the science layer
* Run destriping on each noise layer sequentially
* Update the original ASDF files in place with destriped data

To destripe only the science layer:

.. code-block:: python

    from pyimcom.destripe import destripe_one_layer

    # Destripe only the science data
    n_noise_layers = destripe_one_layer('config.json', verbose=True)

To destripe a specific noise layer:

.. code-block:: python

    # Destripe noise layer 0
    destripe_one_layer('config.json', noiseid=0, verbose=True)


Setup
=======

Configuration
----------------

The configuration file (JSON format) must specify:

.. code-block:: json

    {
        "FILTER": "F184",
        "DSOBSFILE": "/path/to/input/data/",
        "DSOUT": ["/path/to/output/", "_output_stem.txt"],
        "DSMODEL": ["constant", 4088],
        "DSCOST": ["quadratic", 0, 0],
        "CGMODEL": ["PR", 12, 1e-3],
    }

The required fields are described as follows:
* ``FILTER``: Filter name (Y106, J129, H158, F184, K213)
* ``DSOBSFILE``: Path to input SCA images to destripe
* ``DSOUT``: Output directory and filename stem
* ``DSMODEL``: Stripe model type and number of rows per image
* ``DSCOST``: Cost function type (one of "quadratic", "absolute", "huber_loss"), 
    cost function prior (not implemented-- must be 0 for now), 
    and threshold for Huber Loss function (leave as 0 if not using Huber Loss)
* ``CGMODEL``: Conjugate gradient model type (one of "PR" (Polak-Ribiere), "FR" (Fletcher-Reeves)), 
    maximum number of iterations, and convergence tolerance


Major Classes
=============

Sca_img
-------

Represents a single SCA image with metadata, masking, and coordinate transforms.

**Key attributes:**

* ``image``: 2D array of pixel values (4088 x 4088)
* ``w``: WCS object for coordinate transformations
* ``mask``: Boolean array marking valid pixels
* ``g_eff``: Effective gain per pixel

**Key methods:**

* ``subtract_parameters(p, j)``: Remove destriping parameters from image
* ``make_interpolated(...)``: Create interpolated version from overlapping SCAs
* ``apply_all_mask()``: Apply pixel masks

Parameters
----------

Holds destriping parameters for all SCAs in the mosaic.

**Attributes:**

* ``params``: 2D array (N_SCAs x N_parameters)
* ``model``: Destriping model ('constant' or 'linear')
* ``n_rows``: Number of detector rows

**Key methods:**

* ``forward_par(sca_i)``: Convert 1D parameter vector to 2D image for a single SCA

Cost_models
-----------

Manages cost function and derivative selection.

**Supported models:**

* ``quadratic``: f(x) = x^2
* ``absolute``: f(x) = |x|
* ``huber_loss``: Smooth combination of quadratic and absolute


Details: Iteration Step
==========================

Each conjugate gradient iteration performs the following:

1. **Cost Calculation**: Compute :math:`\varepsilon = \sum f(I_A - J_A)` where I_A is the original SCA,
    J_A is interpolated from overlapping SCAs, and f is the cost function model

2. **Gradient Computation**: Calculate :math:`\nabla \varepsilon` via ``residual_function()`` using forward and transpose interpolations

3. **Direction Update**: Compute search direction using selected CG variant (Fletcher-Reeves, or Polak-Ribiere (default))

4. **Line Search**: Find optimal step size :math:`\alpha` along direction

   * Quadratic cost: Direct calculation via ``linear_search_quadratic()``
   * Other cost models: Bisection+secant method via ``linear_search_general()``

5. **Parameter Update**: p_new = p + :math:`\alpha \times` direction

6. **Convergence Check**: Stop if :math:`\|\|\nabla \varepsilon\|\| <` tolerance

Interfaces: C Routines
==========================

This module relies on ``pyimcom_croutines`` (C routines) for performance-critical operations.

bilinear_interpolation
----------------------

.. code-block:: python

    pyimcom_croutines.bilinear_interpolation(
        image,           # Input 2D array
        g_eff,          # Effective gain
        rows, cols,     # Image dimensions
        coords,         # Target coordinates
        num_coords,     # Number of points
        output_image    # Pre-allocated output (modified in-place)
    )

Interpolates image values onto new coordinate grid.

bilinear_transpose
------------------

.. code-block:: python

    pyimcom_croutines.bilinear_transpose(
        image,          # Input interpolated image
        rows, cols,     # Original dimensions
        coords,         # Original coordinates
        num_coords,     # Number of points
        output_image    # Pre-allocated output (modified in-place)
    )

Reverse operation: distributes interpolated values back to original grid (adjoint of interpolation).

The transpose operation is essential for computing gradients in the optimization: gradients computed on the 
interpolated grid must be mapped back to the original detector coordinates.


Interfaces: PyIMCOM
====================

The destriping module can be called through the roman-hlis-l2-driver as part of the Pyimcom overall routine.
This allows destriping to be integrated into the full image combination pipeline for RST data.


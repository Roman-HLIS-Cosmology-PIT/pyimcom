June 2026 end-to-end test information
#####################################

The PyIMCOM runs for the June 2026 end-to-end test were run on the Ohio Supercomputer Center.

Job submission was via the `writejobs.pl <writejobs.pl>`_ script, which constructs a set of slurm jobs with the proper dependencies. This has command line arguments:

- ``$useAccount`` : Which account to charge the job to.
- ``$cfg`` : The configuration file to use.
- ``$tag`` : The prefix for where to put the output logs (I usually put them in their own subdirectory).
- ``$job`` : The prefix for where to put the job scripts that are being sumitted.

The configuration files provided were:

- the **main run** of H-band, 1 mosaic at 1.24 deg^2: `config-Jun26-H1.json <config-Jun26-H1.json>`_

Installation
============

Requirements
------------

``hydra_tod`` requires Python 3.8 or later. Core dependencies (numpy, scipy,
matplotlib, astropy, mpmath, joblib) are installed automatically.

Basic Installation
------------------

.. code-block:: bash

   pip install hydra-tod

Optional Dependencies
---------------------

The package has several optional dependency groups for different features:

.. code-block:: bash

   # MPI parallelization (recommended for multi-TOD analysis)
   pip install hydra-tod[mpi]

   # HEALPix sky maps
   pip install hydra-tod[healpix]

   # JAX + NumPyro (for NUTS sampling and JIT compilation)
   pip install hydra-tod[jax]

   # PyTorch (for GPU-accelerated linear solvers)
   pip install hydra-tod[torch]

   # emcee (for MCMC noise parameter sampling)
   pip install hydra-tod[emcee]

   # All optional dependencies
   pip install hydra-tod[all]

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/zzhang0123/flicker.git
   cd flicker
   pip install -e ".[dev]"

Conda Environment
-----------------

A complete conda environment specification is provided:

.. code-block:: bash

   conda env create -f scripts/environment.yml
   conda activate TOD

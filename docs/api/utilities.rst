Utilities
=========

General Utilities
-----------------

:mod:`hydra_tod.utils` collects low-level numerical routines organised
into four groups:

**Matrix operations**
   :func:`~hydra_tod.utils.cho_compute_mat_inv` — Cholesky-based inversion
   of symmetric positive-definite matrices (with automatic regularisation
   fallback).  :func:`~hydra_tod.utils.cho_compute_mat_inv_sqrt` — returns
   the inverse Cholesky factor :math:`L^{-T}` needed for drawing Gaussian
   samples.

**Polynomial projections**
   :class:`~hydra_tod.utils.polyn_proj` — callable class that stores a
   time list and returns a Legendre polynomial projection matrix on demand.
   Used to build gain and temperature design matrices.

**Log-likelihood evaluation**
   :func:`~hydra_tod.utils.log_likeli` — fast :math:`\mathcal{O}(N^2)`
   Gaussian log-likelihood for Toeplitz covariances via the Levinson
   algorithm.  :func:`~hydra_tod.utils.log_likeli_general` —
   :math:`\mathcal{O}(N^3)` Cholesky-based log-likelihood for arbitrary
   symmetric positive-definite covariances (returns ``-inf`` on failure).
   :func:`~hydra_tod.utils.log_det_symmetric_toeplitz` — standalone
   log-determinant for Toeplitz matrices.

**DFT and miscellaneous**
   :func:`~hydra_tod.utils.DFT_matrix`,
   :func:`~hydra_tod.utils.cov_conjugate` — transform covariances between
   time and frequency domains.
   :func:`~hydra_tod.utils.lag_list`,
   :func:`~hydra_tod.utils.overall_operator`,
   :func:`~hydra_tod.utils.linear_model`.

.. automodule:: hydra_tod.utils
   :members:
   :undoc-members:
   :show-inheritance:

MPI Utilities
-------------

Thin wrappers around ``mpi4py`` providing package-wide MPI communicator
objects (``world``, ``rank``, ``rank0``) and helpers for parallel
function execution and collective operations used across the pipeline.

.. automodule:: hydra_tod.mpiutil
   :members:
   :undoc-members:
   :show-inheritance:

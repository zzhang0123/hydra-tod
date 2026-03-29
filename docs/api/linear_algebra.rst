Linear Algebra
==============

This page documents the two modules that form the numerical core of
every Gibbs sampling step: the **linear solvers** consumed by the normal
equations, and the **linear sampler** that assembles those normal
equations and draws posterior samples from the resulting Gaussian.

Most users will not need to call these functions directly — they are
invoked through the high-level gain and temperature samplers.  The
information here is most useful when:

* swapping in a different linear solver (e.g. PyTorch MPS backend on
  Apple Silicon);
* implementing a new Gibbs step that reuses the iterative GLS machinery;
* debugging convergence of the inner linear solves.

Linear Solvers
--------------

All Gibbs steps reduce to solving :math:`\mathbf{A}\mathbf{x} =
\mathbf{b}` with a symmetric positive-definite matrix.
:func:`~hydra_tod.linear_solver.cg` is the default serial CG solver
throughout the package; it accepts either an explicit matrix or an
implicit ``linear_op`` callable.  For block-distributed systems across
MPI ranks, use :func:`~hydra_tod.linear_solver.cg_mpi` together with
:func:`~hydra_tod.linear_solver.setup_mpi_blocks`.  On Apple Silicon,
:func:`~hydra_tod.linear_solver.pytorch_lin_solver` may be faster for
small-to-medium systems.

To use a different solver, pass it as the ``solver`` keyword argument to
any sampler function::

    from hydra_tod.linear_solver import pytorch_lin_solver
    from hydra_tod.gain_sampler import gain_sampler

    gain_coeffs, gains = gain_sampler(..., solver=pytorch_lin_solver)

.. automodule:: hydra_tod.linear_solver
   :members:
   :undoc-members:
   :show-inheritance:

Linear Sampler
--------------

The iterative GLS and Gaussian constrained realisation (GCR) machinery.
The data model is multiplicative — :math:`\mathbf{d} =
(\mathbf{U}\mathbf{p} + \boldsymbol{\mu})\circ(\mathbf{1} +
\mathbf{n})` — so the noise covariance depends on the current parameter
estimate.  :func:`~hydra_tod.linear_sampler.iterative_gls` alternates
between updating the effective additive noise covariance
:math:`\boldsymbol{\Sigma}(\mathbf{p})` and solving the resulting
weighted normal equations until convergence.

:func:`~hydra_tod.linear_sampler.sample_p_v2` is the **preferred
sampling interface**: it takes pre-computed normal equation matrices
``(A, b, A_sqrt_wn)`` — already output by the iterative GLS step — and
adds the white-noise perturbation needed for a proper GCR draw without
recomputing the covariance.

For MPI-parallel use over a list of TODs, use
:func:`~hydra_tod.linear_sampler.iterative_gls_mpi_list` and
:func:`~hydra_tod.linear_sampler.params_space_oper_and_data_list`.

.. automodule:: hydra_tod.linear_sampler
   :members:
   :undoc-members:
   :show-inheritance:

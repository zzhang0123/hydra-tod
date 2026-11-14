Parameter Samplers
==================

Each module on this page implements one *conditional* Gibbs sampling step.
They are normally called by
:func:`~hydra_tod.full_Gibbs_sampler.TOD_Gibbs_sampler`, but can also be
used standalone when only one block of parameters needs to be updated.

.. list-table:: Gibbs step summary
   :widths: 25 30 45
   :header-rows: 1

   * - Module
     - Parameters sampled
     - Entry point
   * - :mod:`~hydra_tod.gain_sampler`
     - Smooth gain coefficients :math:`\mathbf{p}_g`
     - :func:`~hydra_tod.gain_sampler.gain_sampler`
   * - :mod:`~hydra_tod.tsys_sampler`
     - System temperature :math:`\mathbf{p}_{\rm loc}`, :math:`\mathbf{p}_{\rm sky}`
     - :func:`~hydra_tod.tsys_sampler.Tsys_sampler_multi_TODs`
   * - :mod:`~hydra_tod.map_sampler`
     - :math:`\mathbf{p}_{\rm loc}`, :math:`\mathbf{p}_{\rm sky}` with gains and noise fixed (no Gibbs chain)
     - :func:`~hydra_tod.map_sampler.sample_map_posterior`
   * - :mod:`~hydra_tod.noise_sampler_fixed_fc`
     - Noise params :math:`(\log f_0, \alpha)` — **preferred**
     - :func:`~hydra_tod.noise_sampler_fixed_fc.flicker_sampler`
   * - :mod:`~hydra_tod.noise_sampler_old`
     - Noise params :math:`(\log f_0, \alpha)` — legacy
     - :func:`~hydra_tod.noise_sampler_old.flicker_noise_sampler`

Gain Sampler
------------

Samples the Legendre-polynomial gain coefficients conditioned on the
current system temperature and noise parameters.  Supports three gain
models: ``"linear"``, ``"log"``, and ``"factorized"`` (DC gain +
fluctuations).  The main entry point is
:func:`~hydra_tod.gain_sampler.gain_sampler`, which dispatches to the
model-specific function via its ``model`` string argument.

.. automodule:: hydra_tod.gain_sampler
   :members:
   :undoc-members:
   :show-inheritance:

System Temperature Sampler
--------------------------

Samples sky and local temperature coefficients conditioned on gains and
noise.  Use :func:`~hydra_tod.tsys_sampler.Tsys_coeff_sampler` for a
single TOD (no MPI required) or
:func:`~hydra_tod.tsys_sampler.Tsys_sampler_multi_TODs` to jointly sample
shared sky parameters across multiple TODs via MPI.  Set
``Est_mode=True`` to return the MAP estimate instead of a posterior sample
(useful for burn-in).

.. automodule:: hydra_tod.tsys_sampler
   :members:
   :undoc-members:
   :show-inheritance:

Map Sampler (Gains and Noise Fixed)
-----------------------------------

Posterior draws of the sky and local temperature coefficients with the
gains and the noise covariance fixed, for callers outside a Gibbs chain
such as a map-maker. :func:`~hydra_tod.map_sampler.sample_map_posterior`
takes gain-calibrated TODs, their operators, inverse noise covariances
(dense, possibly semi-definite, or the diagonal for uncorrelated noise), an
estimate ``p_hat`` and an optional Gaussian prior precision. It returns
``n_samples`` draws from
:math:`\mathcal{N}(\hat{\mathbf p}, (\mathbf A + \mathbf S^{-1})^{-1})`
and the precision :math:`\mathbf A + \mathbf S^{-1}`. The per-TOD terms
come from :func:`~hydra_tod.linear_sampler.params_space_oper_and_data` and
each draw from :func:`~hydra_tod.linear_sampler.sample_p_v2`. With a
diagonal noise covariance, memory grows linearly with the number of time
samples. An integer or ``Generator`` ``rng`` leaves NumPy's global state
untouched. A singular posterior precision raises ``LinAlgError``. The
function runs on one process, with no MPI reduction.

.. automodule:: hydra_tod.map_sampler
   :members: sample_map_posterior

Noise Sampler (Fixed Cutoff)
----------------------------

The **preferred** noise-parameter sampler.  Samples
:math:`(\log f_0, \alpha)` with the cutoff frequency :math:`f_c` fixed.
Supports both ``emcee`` (default) and NUTS backends.  Set
``consecutive=False`` together with ``time_list`` to handle flagged or
non-contiguous time samples.

.. automodule:: hydra_tod.noise_sampler_fixed_fc
   :members:
   :undoc-members:
   :show-inheritance:

Noise Sampler (Legacy)
----------------------

Legacy emcee-only noise sampler.  Still called internally by
:func:`~hydra_tod.full_Gibbs_sampler.TOD_Gibbs_sampler` when
``sampler="emcee_old"`` is requested.  For new code, prefer
:mod:`~hydra_tod.noise_sampler_fixed_fc`.

.. automodule:: hydra_tod.noise_sampler_old
   :members:
   :undoc-members:
   :show-inheritance:

MCMC Sampler
------------

Low-level ``emcee`` ensemble sampler wrapper used by the noise samplers
above.  Not intended to be called directly in most workflows.

.. automodule:: hydra_tod.mcmc_sampler
   :members:
   :undoc-members:
   :show-inheritance:

NUTS Sampler
------------

Low-level NumPyro/JAX No-U-Turn Sampler wrapper used by
:mod:`~hydra_tod.noise_sampler_fixed_fc` when ``sampler="NUTS"``.  Requires
JAX and NumPyro to be installed.

.. automodule:: hydra_tod.nuts_sampler
   :members:
   :undoc-members:
   :show-inheritance:

Local Parameter Sampler
-----------------------

Specialised sampler for local (per-TOD) temperature parameters used when
the sky and local components are sampled jointly.

.. automodule:: hydra_tod.local_sampler
   :members:
   :undoc-members:
   :show-inheritance:

Bayesian Utilities
------------------

Helper functions for prior wrapping, parameter transformations
(constrained ↔ unconstrained), and Jeffreys prior construction.  Used
internally by the noise samplers.

.. automodule:: hydra_tod.bayes_util
   :members:
   :undoc-members:
   :show-inheritance:

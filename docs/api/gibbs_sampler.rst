Gibbs Sampler
=============

:mod:`hydra_tod.full_Gibbs_sampler` is the top-level entry point for
Bayesian joint calibration and map-making.  It orchestrates four
conditionally conjugate sampling steps that alternate until the Markov
chain converges:

1. **Gain** — draws smooth polynomial gain coefficients for each TOD
   independently (see :mod:`~hydra_tod.gain_sampler`).
2. **Local temperatures** — draws receiver-noise and noise-diode
   coefficients for each TOD independently
   (see :mod:`~hydra_tod.tsys_sampler`).
3. **Noise parameters** — draws :math:`(\log f_0, \alpha)` for each TOD
   independently via MCMC or NUTS
   (see :mod:`~hydra_tod.noise_sampler_fixed_fc`).
4. **Sky temperature** — draws shared celestial pixel temperatures
   jointly across all TODs, synchronised over MPI ranks
   (see :mod:`~hydra_tod.tsys_sampler`).

Steps 1–3 are independent per TOD and are parallelised across MPI ranks
(one or more TODs per rank).  Step 4 uses an ``MPI_Allreduce`` over
accumulated normal equations, so all ranks must call it collectively.

:func:`~hydra_tod.full_Gibbs_sampler.TOD_Gibbs_sampler` accepts its
many arguments in four conceptual groups:

* **Data** — ``local_TOD_list``, ``local_t_lists``
* **Operators** — ``local_gain_operator_list``,
  ``local_Tsky_operator_list``, ``local_Tloc_operator_list``
* **Priors** — ``prior_cov_inv_*``, ``prior_mean_*``
* **Config** — ``n_samples``, ``gain_model``, ``sampler``,
  ``Est_mode``, ``jeffreys``, ``bounds``, solver tolerances

For a working end-to-end example, see the :doc:`/quickstart` page and
the tutorial notebooks in the ``examples/`` directory.

.. seealso::

   :doc:`samplers` — individual Gibbs step modules.
   :doc:`simulation` — generating synthetic TOD for input.
   :doc:`/quickstart` — annotated end-to-end example.

API Reference
-------------

.. automodule:: hydra_tod.full_Gibbs_sampler
   :members:
   :undoc-members:
   :show-inheritance:

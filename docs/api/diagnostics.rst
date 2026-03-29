MCMC Diagnostics
================

This module provides standard MCMC convergence diagnostics for assessing
the quality of posterior samples returned by the Gibbs sampler.

The typical workflow after collecting samples is:

.. code-block:: python

    from hydra_tod.mcmc_diagnostics import diagnostics

    # samples shape: (n_chains, n_iterations, n_params)
    summary = diagnostics(samples, param_names=["logf0", "alpha"])
    # summary["logf0"] -> {"ESS_min": ..., "ESS_median": ..., "Rhat_split": ...}

The three key quantities are:

:func:`~hydra_tod.mcmc_diagnostics.ess_1d`
    **Effective Sample Size (ESS)** — estimates the number of
    effectively independent samples accounting for autocorrelation.
    Values well below ``n_iterations`` indicate poor mixing.

:func:`~hydra_tod.mcmc_diagnostics.rhat_split`
    **Split-:math:`\hat{R}`** — inter-chain convergence diagnostic.
    Values close to 1.0 (< 1.01 is a common threshold) indicate that
    all chains have converged to the same distribution.

:func:`~hydra_tod.mcmc_diagnostics.acf_1d`
    **Autocorrelation function (ACF)** — FFT-based computation up to a
    specified maximum lag.  Useful for choosing the thinning interval.

:func:`~hydra_tod.mcmc_diagnostics.diagnostics`
    Convenience wrapper that computes ESS and split-:math:`\hat{R}` for
    all parameters and optionally produces trace and ACF plots.

API Reference
-------------

.. automodule:: hydra_tod.mcmc_diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

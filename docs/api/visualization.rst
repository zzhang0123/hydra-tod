Visualization
=============

This module collects plotting utilities for inspecting posterior samples,
sky maps, and diagnostic summaries produced by the Gibbs sampler.
Functions are designed to accept the output arrays of
:func:`~hydra_tod.full_Gibbs_sampler.TOD_Gibbs_sampler` directly.

Typical categories of plots include:

* **Sky maps** — posterior mean, residual, uncertainty (std), and
  Z-score maps on HEALPix projections.
* **Parameter traces** — per-iteration chains for noise parameters,
  gain coefficients, and temperature components.
* **Correlation matrices** — full parameter correlation matrices and
  sky/nuisance cross-correlation blocks.
* **TOD reconstruction** — posterior predictive data overlaid on the
  true simulated TOD with credible intervals.
* **PSD comparison** — posterior noise power spectral density compared
  with the truth or prior model.

API Reference
-------------

.. automodule:: hydra_tod.visualisation
   :members:
   :undoc-members:
   :show-inheritance:

Data Model
==========

``hydra_tod`` implements a hierarchical Bayesian model for radio telescope
observations. This page describes the mathematical framework.

Observation Model
-----------------

The time-ordered data (TOD) from a radio telescope is modeled as:

.. math::

   d(t) = T_{\mathrm{sys}}(t) \cdot \bigl[1 + n(t)\bigr] \cdot g(t)

where:

- :math:`d(t)` is the observed data at time :math:`t`
- :math:`T_{\mathrm{sys}}(t)` is the system temperature
- :math:`n(t)` is multiplicative correlated noise
- :math:`g(t)` is the time-varying instrument gain

System Temperature
------------------

The system temperature decomposes into sky and local components:

.. math::

   T_{\mathrm{sys}}(t) = T_{\mathrm{sky}}(t) + T_{\mathrm{loc}}(t)

The sky temperature is obtained by projecting a pixelized sky map through
the telescope beam:

.. math::

   T_{\mathrm{sky}}(t) = \sum_p B(t, p) \, T_p

where :math:`B(t,p)` is the beam response at pixel :math:`p` and time
:math:`t`, and :math:`T_p` is the sky temperature at pixel :math:`p`.

In matrix form: :math:`\mathbf{T}_{\mathrm{sky}} = \mathbf{U}_{\mathrm{sky}} \, \boldsymbol{\theta}_{\mathrm{sky}}`.

The local temperature includes receiver noise and noise diode contributions:

.. math::

   T_{\mathrm{loc}}(t) = T_{\mathrm{rec}}(t) + T_{\mathrm{nd}}(t)

parameterized as :math:`\mathbf{T}_{\mathrm{loc}} = \mathbf{U}_{\mathrm{loc}} \, \boldsymbol{\theta}_{\mathrm{loc}}`.

Gain Model
----------

The instrument gain is parameterized using Legendre polynomial basis functions:

- **Linear model**: :math:`g(t) = \mathbf{G} \, \mathbf{p}_g`
- **Log-linear model**: :math:`g(t) = \exp(\mathbf{G} \, \mathbf{p}_g)`
- **Factorized model**: :math:`g(t) = g_0 \cdot (\mathbf{G} \, \mathbf{p}_g + 1)`

where :math:`\mathbf{G}` is the polynomial projection matrix and :math:`\mathbf{p}_g`
are the gain coefficients.

Gibbs Sampling
--------------

The joint posterior is sampled using a Gibbs sampler that alternates between
conditionally conjugate updates:

1. **Sample gains** :math:`\mathbf{p}_g \mid T_{\mathrm{sys}}, n, d`
2. **Sample local temperatures** :math:`\boldsymbol{\theta}_{\mathrm{loc}} \mid T_{\mathrm{sky}}, g, n, d`
3. **Sample sky temperatures** :math:`\boldsymbol{\theta}_{\mathrm{sky}} \mid T_{\mathrm{loc}}, g, n, d`
4. **Sample noise parameters** :math:`(\log f_0, \alpha) \mid T_{\mathrm{sys}}, g, d`

Steps 1--3 are conjugate Gaussian updates solved via iterative linear solvers.
Step 4 uses MCMC (emcee) or NUTS (NumPyro) to sample from the non-Gaussian
noise parameter posterior.

For full details, see Zhang et al. (2026), RASTI, rzag024.

Quick Start
===========

This guide walks through the core ``hydra_tod`` workflow: simulating
time-ordered data, running the full Gibbs sampler, assessing convergence,
and using individual sub-samplers in isolation.

For complete annotated examples, see the Jupyter notebooks in the
``examples/`` directory of the repository.


1. Simulating a Single TOD
--------------------------

:class:`~hydra_tod.simulation.TODSimulation` generates a synthetic
MeerKAT-like observation — scan pattern, beam projection, gain variations,
noise-diode injections, and correlated :math:`1/f` noise — in one call.

.. code-block:: python

   from hydra_tod.simulation import TODSimulation

   sim = TODSimulation(
       nside=64,              # HEALPix resolution
       elevation=41.5,        # Telescope elevation (degrees)
       freq=750,              # Observing frequency (MHz)
       alpha=2.0,             # Flicker noise spectral index
       logf0=-4.87,           # Log10 of knee frequency (rad/s)
       ptsrc_path="gleam_nside512_K_allsky_408MHz.npy",
   )

   tod    = sim.TOD_setting       # Observed data vector
   sky    = sim.sky_params        # True sky pixel temperatures
   gains  = sim.gains_setting     # True gain time series
   noise  = sim.noise_setting     # True 1/f noise realisation

The simulation also creates the linear operators needed by the sampler:

.. code-block:: python

   gain_proj    = sim.gain_proj       # Legendre gain projection matrix
   Tsky_op      = sim.Tsky_operator   # Beam projection (sky → TOD)
   Tloc_op      = sim.Tloc_operator   # Local-temp projection (rec + nd)
   t_list       = sim.time_list       # Observation times (seconds)
   logfc        = sim.logfc           # Log cutoff frequency


2. Running the Full Gibbs Sampler
----------------------------------

:func:`~hydra_tod.full_Gibbs_sampler.TOD_Gibbs_sampler` takes the
operators from the simulation and returns posterior samples for all
parameter blocks.

.. code-block:: python

   import numpy as np
   from hydra_tod.full_Gibbs_sampler import TOD_Gibbs_sampler

   # --- priors ---
   n_sky   = sim.sky_params.shape[0]
   n_loc   = sim.Tloc_operator.shape[1]
   n_gain  = sim.gain_proj.shape[1]

   # 20 % Gaussian prior on sky pixels
   prior_sky_cov_inv  = (1.0 / (0.2 * sim.sky_params)) ** 2
   prior_sky_mean     = sim.sky_params          # prior centred on truth

   # 10 % prior on local temperature and gain coefficients
   prior_loc_cov_inv  = np.ones(n_loc) * (10.0) ** 2
   prior_loc_mean     = np.zeros(n_loc)
   prior_gain_cov_inv = np.ones(n_gain) * (10.0) ** 2
   prior_gain_mean    = np.zeros(n_gain)

   # --- initial conditions ---
   init_sky  = sim.sky_params * 1.05           # slight offset from truth
   init_loc  = np.zeros(n_loc)
   init_noise = [sim.logf0, sim.alpha]

   # --- run sampler ---
   Tsky_samples, gain_samples, noise_samples, Tloc_samples = TOD_Gibbs_sampler(
       local_TOD_list=[tod],
       local_t_lists=[t_list],
       local_gain_operator_list=[gain_proj],
       local_Tsky_operator_list=[Tsky_op],
       local_Tloc_operator_list=[Tloc_op],
       init_Tsky_params=init_sky,
       init_Tloc_params_list=[init_loc],
       init_noise_params_list=[init_noise],
       local_logfc_list=[logfc],
       prior_cov_inv_Tsky=prior_sky_cov_inv,
       prior_mean_Tsky=prior_sky_mean,
       prior_cov_inv_Tloc_list=[prior_loc_cov_inv],
       prior_mean_Tloc_list=[prior_loc_mean],
       prior_cov_inv_gain_list=[prior_gain_cov_inv],
       prior_mean_gain_list=[prior_gain_mean],
       n_samples=2000,
       gain_model="linear",
       sampler="emcee",
       jeffreys=True,
   )

   # Tsky_samples  shape: (2000, n_sky)
   # gain_samples  shape: (2000, n_gain)
   # noise_samples shape: (2000, 2)   — (logf0, alpha) per iteration
   # Tloc_samples  shape: (2000, n_loc)

.. note::

   For multi-TOD analysis with MPI, launch the script with
   ``mpiexec -n N python script.py``.  Each rank receives a slice of
   ``local_TOD_list`` and the sampler handles synchronisation
   automatically.


3. Inspecting Posterior Samples
---------------------------------

Use the built-in diagnostics to check convergence before interpreting
results.

.. code-block:: python

   from hydra_tod.mcmc_diagnostics import diagnostics
   import matplotlib
   matplotlib.use("Agg")

   # Wrap single chain as shape (1, n_iter, n_params)
   noise_chains = noise_samples[np.newaxis, :, :]   # (1, 2000, 2)
   summary = diagnostics(
       noise_chains,
       param_names=["logf0", "alpha"],
       max_plots=2,      # produce trace + ACF plots
   )

   print(summary["logf0"])
   # {"ESS_min": 342.1, "ESS_median": 342.1, "Rhat_split": 1.002}

A well-converged chain should have ESS > ~100 and split-:math:`\hat{R}`
close to 1.0.

Posterior summaries for sky maps:

.. code-block:: python

   sky_mean = Tsky_samples.mean(axis=0)          # posterior mean
   sky_std  = Tsky_samples.std(axis=0)           # posterior std
   residual = sky_mean - sim.sky_params           # bias map
   z_score  = residual / sky_std                 # should be ~ N(0,1)


4. Using Sub-samplers Directly
--------------------------------

Each Gibbs step can be called in isolation, which is useful for debugging
or for building custom inference pipelines.

**Sample gains only**

.. code-block:: python

   from hydra_tod.gain_sampler import gain_sampler
   from hydra_tod.flicker_model import flicker_cov
   from scipy.linalg import toeplitz
   import numpy as np

   # Build noise covariance inverse from current noise params
   from hydra_tod.utils import lag_list, cho_compute_mat_inv
   from hydra_tod.flicker_model import flicker_cov_vec

   lags     = lag_list(t_list)
   cov_row  = flicker_cov_vec(lags, 10**sim.logf0, 10**sim.logfc, sim.alpha)
   Ncov_inv = cho_compute_mat_inv(toeplitz(cov_row))

   gain_coeffs, gains = gain_sampler(
       data=tod,
       t_list=t_list,
       gain_proj=gain_proj,
       Tsys=sky[..., np.newaxis] * np.ones_like(tod),  # current Tsys estimate
       noise_params=(sim.logf0, sim.alpha),
       logfc=sim.logfc,
       model="linear",
       prior_cov_inv=prior_gain_cov_inv,
       prior_mean=prior_gain_mean,
   )

**Sample noise parameters only**

.. code-block:: python

   from hydra_tod.noise_sampler_fixed_fc import flicker_sampler

   noise_sample = flicker_sampler(
       TOD=tod,
       gains=gains,          # current gain estimate
       Tsys=tsys_estimate,   # current Tsys estimate
       init_params=[sim.logf0, sim.alpha],
       n_samples=1,
       jeffreys=True,
       sampler="emcee",
       consecutive=True,     # set False for flagged/non-contiguous data
   )
   logf0_new, alpha_new = noise_sample

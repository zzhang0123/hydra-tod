Quick Start
===========

This guide demonstrates the basic workflow for simulating and analyzing
time-ordered data with ``hydra_tod``.

Simulating a Single TOD
------------------------

.. code-block:: python

   from hydra_tod.simulation import TODSimulation

   # Create a simulated MeerKAT observation
   sim = TODSimulation(
       nside=64,              # HEALPix resolution
       elevation=41.5,        # Telescope elevation (degrees)
       freq=750,              # Observing frequency (MHz)
       alpha=2.0,             # Flicker noise spectral index
       logf0=-4.87,           # Log10 knee frequency
       ptsrc_path="gleam_nside512_K_allsky_408MHz.npy",
   )

   # Access simulation components
   tod = sim.TOD_setting          # Observed time-ordered data
   sky = sim.sky_params           # True sky temperatures
   gains = sim.gains_setting      # True gain time series
   noise = sim.noise_setting      # True noise realization

Running the Gibbs Sampler
--------------------------

.. code-block:: python

   from hydra_tod.full_Gibbs_sampler import TOD_Gibbs_sampler

   # Run the Gibbs sampler (requires MPI for multi-TOD)
   Tsky_samples, gain_samples, noise_samples, Tloc_samples = TOD_Gibbs_sampler(
       local_TOD_list=[tod],
       local_t_lists=[sim.time_list],
       local_gain_operator_list=[sim.gain_proj],
       local_Tsky_operator_list=[sim.Tsky_operator],
       local_Tloc_operator_list=[sim.Tloc_operator],
       init_Tsky_params=sky * 1.1,       # Initial guess
       init_Tloc_params_list=[...],
       init_noise_params_list=[[sim.logf0, sim.alpha]],
       local_logfc_list=[sim.logfc],
       n_samples=2000,
   )

For complete working examples, see the Jupyter notebooks in the ``examples/``
directory of the repository.

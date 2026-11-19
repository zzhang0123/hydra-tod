# Changelog

## Unreleased

### Added

- `hydra_tod.map_sampler.sample_map_posterior` draws the linear Tsys
  parameters (sky and local) from their posterior with the gains and the
  noise fixed, for callers outside a Gibbs chain such as a map-maker. It
  takes gain-calibrated TODs, operators, inverse noise covariances, an
  estimate `p_hat` and an optional Gaussian prior precision `S^-1`. It
  returns the draws, from `N(p_hat, (A + S^-1)^-1)`, and the precision
  `A + S^-1`, where `A` is the data precision. Each draw is `p_hat` plus a
  zero-mean realisation from `sample_p_v2`, so the draws are centred on
  `p_hat` before the GLS has converged and keep their spread when `A` is
  ill-conditioned. The per-TOD terms come from `params_space_oper_and_data`.
  The noise may be dense, semi-definite (flagged samples zeroed out) or
  diagonal, and the model multiplicative or additive. A singular
  `A + S^-1` raises `LinAlgError`; asymmetric matrices and non-finite or
  negative inverse variances raise `ValueError`. It runs on one process.
  limTOD's `GLS_mapmaking.posterior_samples` calls it.
- `params_space_oper_and_data` accepts the `(M,)` diagonal of `N_inv`, and
  `sqrt(N_inv)` as `Ninv_sqrt`, for uncorrelated noise. It then forms no
  `(M, M)` array.
- `params_space_oper_and_data(..., multiplicative=False)` builds the normal
  equations of the additive model `d = U p + mu + n`.
- `sample_p_v2` takes `rng`, a `numpy.random.Generator` for the prior
  perturbation. The default `None` draws from, and advances, NumPy's
  global random state, as before.
- `hydra_tod.gls` holds `params_space_oper_and_data`. It can be imported
  without mpi4py, JAX, PyTorch or matplotlib.
  `hydra_tod.linear_sampler.params_space_oper_and_data` is the same
  function.
- `params_space_oper_and_data` documents `p=None` for
  `multiplicative=False`, where `p` is unused. With `multiplicative=True`
  and `p=None` it raises `ValueError`; it raised `TypeError` before.
- `iterative_gls_mpi_list(..., draw=False)` skips the white-noise term and
  returns `None` in its place. The default is `draw=True`, as before.
- `params_space_oper_and_data` takes a `scipy.sparse` array as `U` when
  `N_inv` is a diagonal, and then returns sparse matrices. This worked
  before; it is now documented and tested.

### Changed

- `import hydra_tod` loads `hydra_tod.utils` only. `flicker_model` and
  `mcmc_diagnostics` are imported on first access, like the other modules.
  Importing the package no longer imports JAX or matplotlib and no longer
  initialises MPI.
- `sample_p_v2` raises `ValueError` when `A_sqrt_wn` is `None` in sampling
  mode. In estimation mode the term is unused and may be `None`.
- `params_space_oper_and_data(..., multiplicative=False)` with a dense
  `N_inv` uses `N_inv` and `Ninv_sqrt` as given. It allocated one `(M, M)`
  array, or two with `Ninv_sqrt`, and now allocates none.

### Fixed

- `sample_p_v2` no longer modifies its inputs. It added the prior precision
  to the caller's `A` in place (and the prior term to `b` in estimation
  mode), so a second call with the same arrays added the prior twice.
- `get_Tsys_operator` used the previous TOD's dimension as the column
  offset of each TOD's local-temperature block, where the sum over all
  previous TODs on the rank is needed. The two agree for the first two
  TODs of a rank. From the third TOD on, with `joint_Tsys_sampling=True`,
  TODs shared columns and the last columns were zero.
- `Tsys_sampler_multi_TODs(Est_mode=True)` no longer draws the white-noise
  term, which it discarded. The step leaves NumPy's global random state
  unchanged, so a seeded run with `Est_mode=True` follows a different
  random stream than before.
- `hydra_tod.map_sampler` is listed in `__all__` and is an attribute of
  the package after `import hydra_tod`. It raised `AttributeError` unless
  the module had been imported explicitly.

## v0.1.0 (2026)

Initial release accompanying Zhang et al. (2026), "Joint Bayesian calibration
and map-making for intensity mapping experiments", RASTI, rzag024.

### Features

- Flicker (1/f) noise covariance modeling with polynomial emulators
- Gibbs sampler for joint calibration and map-making from TOD
- Multiple gain models: linear, log-linear, factorized
- MPI-distributed iterative linear solvers (CG, MINRES)
- MCMC (emcee) and NUTS (NumPyro) noise parameter sampling
- MeerKAT telescope simulation with realistic sky models
- Comprehensive visualization and diagnostic tools

# Changelog

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

# hydra_tod

**Bayesian calibration and map-making for radio intensity mapping experiments**

[![Documentation](https://readthedocs.org/projects/hydra-tod/badge/?version=latest)](https://hydra-tod.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

`hydra_tod` implements joint Bayesian inference of sky temperature maps,
instrument gains, and correlated noise parameters from radio telescope
time-ordered data (TOD). It uses Gibbs sampling with MPI parallelization
to efficiently handle multi-receiver, multi-scan datasets.

## Features

- **Flicker noise modeling**: 1/f^alpha noise covariance via incomplete gamma functions, with polynomial emulators for fast evaluation
- **Gibbs sampler**: Joint calibration and map-making alternating between sky, gain, and noise parameter updates
- **Multiple gain models**: Linear, log-linear, and factorized parameterizations
- **MPI parallelization**: Distributed iterative linear solvers (conjugate gradient, MINRES) for large-scale problems
- **Flexible noise sampling**: MCMC (emcee) and NUTS (NumPyro) samplers for non-Gaussian noise posteriors
- **Telescope simulation**: Realistic MeerKAT scan simulations with sky models (GSM + point sources)
- **Diagnostics**: MCMC convergence diagnostics (ESS, R-hat), posterior visualization, sky map reconstruction

## Data Model

The package models radio telescope observations as:

```
TOD = Tsys * (1 + n) * g
```

where:
- **Tsys** = Tsky + Tloc is the system temperature (sky + local components)
- **n** ~ 1/f^alpha is correlated flicker noise
- **g** is the time-varying instrument gain

## Installation

```bash
# Basic installation
pip install hydra-tod

# With MPI support
pip install hydra-tod[mpi]

# With all optional dependencies
pip install hydra-tod[all]

# Development installation
git clone https://github.com/zzhang0123/flicker.git
cd flicker
pip install -e ".[dev]"
```

## Quick Start

```python
from hydra_tod.simulation import TODSimulation

# Create a simulated observation
sim = TODSimulation(
    nside=64,
    elevation=41.5,
    freq=750,
    alpha=2.0,
    ptsrc_path="gleam_nside512_K_allsky_408MHz.npy",
)

# Access simulated data
print(f"TOD shape: {sim.TOD_setting.shape}")
print(f"Sky pixels: {len(sim.pixel_indices)}")
```

## Citation

If you use `hydra_tod` in your research, please cite:

```bibtex
@article{zhang2026joint,
  title={Joint Bayesian calibration and map-making for intensity mapping experiments},
  author={Zhang, Zheng and Bull, Philip and Santos, Mario G and Nasirudin, Ainulnabilah},
  journal={RAS Techniques and Instruments},
  pages={rzag024},
  year={2026},
  publisher={Oxford University Press}
}
```

## Documentation

Full documentation is available at [hydra-tod.readthedocs.io](https://hydra-tod.readthedocs.io).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

"""
hydra_tod: Bayesian calibration and map-making for radio intensity mapping experiments.

This package provides tools for analyzing time-ordered data (TOD) from radio
telescopes, including flicker noise modeling, Gibbs sampling for joint
calibration and map-making, and sky map reconstruction.

Reference
---------
If you use this package in your research, please cite:

    Zhang, Z., Bull, P., Santos, M. G., & Nasirudin, A. (2026).
    Joint Bayesian calibration and map-making for intensity mapping experiments.
    RAS Techniques and Instruments, rzag024.
    https://doi.org/10.1093/rasti/rzag024

.. code-block:: bibtex

    @article{zhang2026joint,
      title={Joint Bayesian calibration and map-making for intensity mapping experiments},
      author={Zhang, Zheng and Bull, Philip and Santos, Mario G and Nasirudin, Ainulnabilah},
      journal={RAS Techniques and Instruments},
      pages={rzag024},
      year={2026},
      publisher={Oxford University Press}
    }
"""

__version__ = "0.1.0"
__author__ = "Zheng Zhang"

# Core modules (lightweight, always available)
from . import utils
from . import flicker_model
from . import mcmc_diagnostics


def __getattr__(name):
    """Lazy import for modules with heavy dependencies (MPI, JAX, PyTorch)."""
    import importlib

    _lazy_modules = {
        "mpiutil",
        "linear_solver",
        "linear_sampler",
        "gain_sampler",
        "tsys_sampler",
        "noise_sampler_old",
        "noise_sampler_fixed_fc",
        "full_Gibbs_sampler",
        "simulation",
        "visualisation",
        "bayes_util",
        "mcmc_sampler",
        "nuts_sampler",
        "local_sampler",
    }
    if name in _lazy_modules:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "flicker_model",
    "utils",
    "mcmc_diagnostics",
    "simulation",
    "full_Gibbs_sampler",
    "gain_sampler",
    "tsys_sampler",
    "noise_sampler_old",
    "noise_sampler_fixed_fc",
    "linear_solver",
    "linear_sampler",
    "mpiutil",
    "visualisation",
    "bayes_util",
    "mcmc_sampler",
    "nuts_sampler",
    "local_sampler",
]
"""Public modules available via ``import hydra_tod``."""

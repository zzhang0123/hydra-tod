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
    RAS Techniques and Instruments, 5, rzag024.
    https://doi.org/10.1093/rasti/rzag024

.. code-block:: bibtex

    @ARTICLE{2026RASTI...5ag024Z,
           author = {{Zhang}, Zheng and {Bull}, Philip and {Santos}, Mario G. and {Nasirudin}, Ainulnabilah},
            title = "{Joint Bayesian calibration and map-making for intensity mapping experiments}",
          journal = {RAS Techniques and Instruments},
         keywords = {Data Methods, methods: data analysis, techniques: spectroscopic, radio lines: general, Instrumentation and Methods for Astrophysics},
             year = 2026,
            month = jan,
           volume = {5},
              eid = {rzag024},
            pages = {rzag024},
              doi = {10.1093/rasti/rzag024},
    archivePrefix = {arXiv},
           eprint = {2509.10992},
     primaryClass = {astro-ph.IM},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2026RASTI...5ag024Z},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
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

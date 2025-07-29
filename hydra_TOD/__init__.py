"""
Hydra TOD: Time-Ordered Data analysis package for radio astronomy

This package provides tools for analyzing time-ordered data (TOD) from radio telescopes,
including flicker noise modeling, Gibbs sampling, and sky map reconstruction.

Modules:
--------
- flicker_model: Core flicker noise modeling
- mcmc_sampler: MCMC sampling algorithms
- gain_sampler: Gain calibration sampling
- noise_sampler: Noise parameter sampling
- linear_sampler: Linear parameter sampling
- TOD_simulator: Time-ordered data simulation
- utils: Utility functions
- visualisation: Plotting and visualization tools
"""

__version__ = "0.1.0"
__author__ = "Zheng Zhang"

# Core modules
from . import flicker_model
from . import mcmc_sampler
from . import gain_sampler
from . import noise_sampler
from . import linear_sampler
from . import TOD_simulator
from . import utils
from . import visualisation

# Optional imports
try:
    from . import noise_sampler_fixed_fc
except ImportError:
    pass

try:
    from . import linear_solver
    from . import Tsys_sampler
    from . import full_Gibbs_sampler
    from . import simulation
    from . import mpiutil
except ImportError:
    pass

__all__ = [
    'flicker_model',
    'mcmc_sampler', 
    'gain_sampler',
    'noise_sampler',
    'linear_sampler',
    'TOD_simulator',
    'utils',
    'visualisation'
]

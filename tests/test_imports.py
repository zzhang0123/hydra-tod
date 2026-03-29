"""Test that all modules import cleanly."""
import pytest


def test_top_level_import():
    import hydra_tod
    assert hasattr(hydra_tod, "__version__")
    assert hydra_tod.__version__ == "0.1.0"


def test_eager_imports():
    from hydra_tod import utils
    from hydra_tod import flicker_model
    from hydra_tod import mcmc_diagnostics


def test_lazy_imports():
    from hydra_tod import simulation
    from hydra_tod import full_Gibbs_sampler
    from hydra_tod import gain_sampler
    from hydra_tod import tsys_sampler
    from hydra_tod import linear_solver
    from hydra_tod import linear_sampler
    from hydra_tod import noise_sampler_old
    from hydra_tod import noise_sampler_fixed_fc
    from hydra_tod import mcmc_sampler
    from hydra_tod import bayes_util
    from hydra_tod import visualisation


def test_key_classes_importable():
    from hydra_tod.simulation import TODSimulation, MultiTODSimulation
    from hydra_tod.full_Gibbs_sampler import TOD_Gibbs_sampler
    from hydra_tod.flicker_model import FlickerCorrEmulator, LogDetEmulator
    from hydra_tod.mcmc_diagnostics import diagnostics


def test_renamed_modules_not_importable_by_old_name():
    """Old module names should not be importable."""
    with pytest.raises(ImportError):
        from hydra_tod import MCMC_diagnostics  # noqa: F401
    with pytest.raises(ImportError):
        from hydra_tod import Tsys_sampler  # noqa: F401
    with pytest.raises(ImportError):
        from hydra_tod import TOD_simulator  # noqa: F401

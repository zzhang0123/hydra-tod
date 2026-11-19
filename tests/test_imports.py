"""Test that all modules import cleanly."""
import subprocess
import sys
from textwrap import dedent

import pytest


def _run_in_fresh_interpreter(script):
    """Run ``script`` in a new interpreter. Test modules import submodules
    at collection, which binds them on the package, so only a new process
    shows what a bare ``import hydra_tod`` provides."""
    subprocess.run([sys.executable, "-c", dedent(script)], check=True)


def test_top_level_import():
    import hydra_tod
    assert hasattr(hydra_tod, "__version__")
    assert hydra_tod.__version__ == "0.1.0"


def test_eager_imports():
    from hydra_tod import utils


def test_package_import_loads_only_utils():
    _run_in_fresh_interpreter(
        """
        import sys

        before = set(sys.modules)
        import hydra_tod

        new = set(sys.modules) - before
        submodules = {name for name in new if name.startswith("hydra_tod.")}
        assert submodules == {"hydra_tod.utils"}, submodules
        backends = {"mpi4py", "jax", "torch", "matplotlib"}
        loaded = backends & {name.split(".")[0] for name in new}
        assert not loaded, loaded
        """
    )


def test_every_module_is_registered():
    """Each public module on disk is in ``__all__`` and is an attribute of
    the package after a bare ``import hydra_tod``."""
    _run_in_fresh_interpreter(
        """
        import importlib
        import pkgutil

        import hydra_tod

        on_disk = {
            module.name
            for module in pkgutil.iter_modules(hydra_tod.__path__)
            if not module.name.startswith("_")
        }
        listed = hydra_tod.__all__
        assert sorted(listed) == sorted(on_disk), sorted(set(listed) ^ on_disk)

        # Record the lazy imports without running them: this checks that a
        # module is registered, not that its dependencies are installed.
        importlib.import_module = lambda name, package=None: name
        unreachable = sorted(m for m in on_disk if not hasattr(hydra_tod, m))
        assert not unreachable, unreachable
        """
    )


def test_lazy_imports_without_mcmc_packages():
    """``flicker_model``, ``mcmc_diagnostics``, ``gls`` and ``map_sampler``
    import with numpyro, emcee and pygdsm unavailable."""
    _run_in_fresh_interpreter(
        """
        import importlib.abc
        import sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".")[0] in {"numpyro", "emcee", "pygdsm"}:
                    raise ImportError(f"{fullname} is blocked in this test")

        sys.meta_path.insert(0, Block())
        from hydra_tod import flicker_model, gls, map_sampler, mcmc_diagnostics
        """
    )


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

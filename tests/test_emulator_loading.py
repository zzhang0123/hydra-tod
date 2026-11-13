"""The bundled flicker-noise emulators load from the installed package.

The ``.pkl`` files were written when ``flicker_model`` was a top-level module,
so they name their classes ``flicker_model.FlickerCorrEmulator`` and
``flicker_model.LogDetEmulator``. :mod:`hydra_tod.emulator_io` maps that
module name onto ``hydra_tod.flicker_model``.
"""

import importlib
import io

import numpy as np
import pytest

from hydra_tod.flicker_model import (
    FlickerCorrEmulator,
    LogDetEmulator,
    flicker_cov_vec,
)

# Corners and centre of the training box: logf0 in [-6, -3], alpha in [1.1, 4].
TRAINING_BOX_CELLS = [
    (-6.0, 1.1),
    (-6.0, 4.0),
    (-3.0, 1.1),
    (-3.0, 4.0),
    (-4.5, 2.0),
]


def _require_emulator_deps():
    pytest.importorskip("MomentEmu")
    pytest.importorskip("sklearn")


@pytest.fixture(scope="module")
def emulators():
    _require_emulator_deps()
    from hydra_tod.emulator_io import (
        CORR_EMULATOR_FILE,
        LOGDET_EMULATOR_FILE,
        load_emulator,
    )

    return load_emulator(CORR_EMULATOR_FILE), load_emulator(LOGDET_EMULATOR_FILE)


def test_legacy_module_name_resolves_to_package():
    from hydra_tod.emulator_io import _PackageUnpickler

    # Protocol-0 GLOBAL opcode naming the pre-package module path.
    stream = io.BytesIO(b"cflicker_model\nLogDetEmulator\n.")
    assert _PackageUnpickler(stream).load() is LogDetEmulator


def test_other_module_names_are_unchanged():
    from hydra_tod.emulator_io import _PackageUnpickler

    stream = io.BytesIO(b"cnumpy\nndarray\n.")
    assert _PackageUnpickler(stream).load() is np.ndarray


@pytest.mark.parametrize(
    "filename", ["/tmp/flicker_corr_emulator.pkl", "../flicker_corr_emulator.pkl"]
)
def test_load_emulator_rejects_paths(filename):
    from hydra_tod.emulator_io import load_emulator

    with pytest.raises(ValueError):
        load_emulator(filename)


def test_bundled_emulators_load(emulators):
    corr, logdet = emulators
    assert isinstance(corr, FlickerCorrEmulator)
    assert isinstance(logdet, LogDetEmulator)
    # Training configuration of the bundled correlation emulator.
    assert corr.tau_list.shape == (2860,)
    assert corr.wnoise_var == pytest.approx(2.5e-6)


@pytest.mark.parametrize("logf0, alpha", TRAINING_BOX_CELLS)
def test_corr_emulator_matches_exact_covariance(emulators, logf0, alpha):
    """Emulator and exact row agree to 2e-3 of the zero-lag variance.

    The largest observed deviation is 7e-4, at alpha = 1.1.
    """
    jax = pytest.importorskip("jax")
    corr, _ = emulators
    exact = flicker_cov_vec(
        corr.tau_list,
        10.0**logf0,
        10.0**corr.logfc,
        alpha,
        white_n_variance=corr.wnoise_var,
    )
    emulated = corr(logf0, alpha)
    emulated_jax = np.asarray(jax.jit(corr.create_jax())(logf0, alpha))[0]

    assert np.max(np.abs(emulated - exact)) / exact[0] < 2e-3
    assert np.max(np.abs(emulated_jax - exact)) / exact[0] < 2e-3


@pytest.mark.parametrize("logf0, alpha", TRAINING_BOX_CELLS)
def test_logdet_emulator_matches_exact_logdet(emulators, logf0, alpha):
    """Emulated ln det C is within 15 nats of the exact value.

    The largest observed deviation is about 12 nats, at logf0 = -3,
    alpha = 4, out of |ln det C| ~ 3.7e4.
    """
    from scipy.linalg import toeplitz

    corr, logdet = emulators
    row = flicker_cov_vec(
        corr.tau_list,
        10.0**logf0,
        10.0**corr.logfc,
        alpha,
        white_n_variance=corr.wnoise_var,
    )
    sign, exact = np.linalg.slogdet(toeplitz(row))
    assert sign > 0
    assert abs(float(np.ravel(logdet(logf0, alpha))[0]) - exact) < 15.0


@pytest.mark.parametrize("module_name", ["gain_sampler", "noise_sampler_old"])
def test_sampler_uses_emulator(module_name):
    pytest.importorskip("jax")
    _require_emulator_deps()
    module = importlib.import_module(f"hydra_tod.{module_name}")
    assert module.use_emulator is True
    assert isinstance(module.flicker_cov, FlickerCorrEmulator)


@pytest.mark.parametrize("module_name", ["noise_sampler_fixed_fc", "local_sampler"])
def test_numpyro_sampler_uses_emulator(module_name):
    pytest.importorskip("numpyro", exc_type=ImportError)
    _require_emulator_deps()
    module = importlib.import_module(f"hydra_tod.{module_name}")
    assert isinstance(module.flicker_cov, FlickerCorrEmulator)
    assert module.flicker_cov_jax is not None

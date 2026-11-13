"""
Load the pre-trained flicker-noise emulators bundled with the package.

The two ``.pkl`` files were written when ``flicker_model`` was a top-level
module, so they name their classes ``flicker_model.FlickerCorrEmulator`` and
``flicker_model.LogDetEmulator``. :class:`_PackageUnpickler` resolves that
module name to :mod:`hydra_tod.flicker_model`, so the files load from an
installed package without being regenerated.

Unpickling also imports ``MomentEmu`` and ``scikit-learn``, which the
emulator objects reference; without them :func:`load_emulator` raises
:class:`ImportError`.

Public API
----------
- :func:`load_emulator` -- unpickle an emulator file.
- :data:`CORR_EMULATOR_FILE` -- correlation-function emulator file name.
- :data:`LOGDET_EMULATOR_FILE` -- log-determinant emulator file name.
"""

from __future__ import annotations

import os
import pickle
from typing import Any

CORR_EMULATOR_FILE = "flicker_corr_emulator.pkl"
LOGDET_EMULATOR_FILE = "flicker_logdet_emulator.pkl"

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Module names used when the emulators were pickled -> current module names.
_LEGACY_MODULES = {"flicker_model": f"{__package__}.flicker_model"}


class _PackageUnpickler(pickle.Unpickler):
    """Unpickler that resolves pre-package module names inside ``hydra_tod``."""

    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(_LEGACY_MODULES.get(module, module), name)


def load_emulator(filename: str) -> Any:
    """Unpickle an emulator file shipped inside the package directory.

    Unpickling runs code named in the file, so this function only reads
    files that ship with ``hydra_tod``; it rejects paths.

    Parameters
    ----------
    filename : str
        Bare file name inside the package directory, e.g.
        :data:`CORR_EMULATOR_FILE`.

    Returns
    -------
    FlickerCorrEmulator or LogDetEmulator
        The unpickled emulator.

    Raises
    ------
    ValueError
        If *filename* contains a directory component.
    FileNotFoundError
        If the file does not exist.
    ImportError
        If a module the pickle references (``MomentEmu``, ``sklearn``)
        is not installed.
    """
    if os.path.basename(filename) != filename:
        raise ValueError(
            f"load_emulator takes a file name inside the package, got {filename!r}"
        )
    with open(os.path.join(_PACKAGE_DIR, filename), "rb") as f:
        return _PackageUnpickler(f).load()

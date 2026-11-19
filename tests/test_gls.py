"""Tests for the standalone serial GLS API."""

import subprocess
import sys
from textwrap import dedent


def test_gls_without_optional_backends():
    script = dedent(
        """
        import importlib.abc
        import sys

        import numpy as np

        class BlockOptionalBackends(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".")[0] in {"mpi4py", "jax", "torch", "matplotlib"}:
                    raise AssertionError(f"Serial GLS attempted to load {fullname}")

        sys.meta_path.insert(0, BlockOptionalBackends())
        from hydra_tod.gls import params_space_oper_and_data

        design = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        data = np.array([2.0, 3.0, 5.0])
        offset = np.array([1.0, 0.0, 2.0])
        weights = np.array([1.0, 2.0, 4.0])
        correlated = np.array([[2.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 4.0]])

        for precision, expected_precision in (
            (weights, np.diag(weights)),
            (correlated, correlated),
        ):
            lhs, rhs = params_space_oper_and_data(
                data, design, None, precision, mu=offset, multiplicative=False
            )
            np.testing.assert_allclose(lhs, design.T @ expected_precision @ design)
            np.testing.assert_allclose(rhs, design.T @ expected_precision @ (data - offset))

        with np.testing.assert_raises_regex(ValueError, "p is required"):
            params_space_oper_and_data(data, design, None, weights, mu=offset)
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)

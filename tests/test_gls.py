"""Tests for the standalone serial GLS API."""

import subprocess
import sys
import tracemalloc
from textwrap import dedent

import numpy as np
import pytest

from hydra_tod.gls import params_space_oper_and_data


def _weights(kind, n, rng):
    if kind == "unit":
        return np.ones(n)
    if kind == "wide":
        return 10.0 ** rng.uniform(-12.0, 12.0, size=n)
    return np.where(rng.uniform(size=n) < 0.4, 0.0, rng.uniform(1.0, 2.0, size=n))


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("weights", ["unit", "wide", "flagged"])
@pytest.mark.parametrize("shape", [(1, 1), (2, 1), (7, 3), (60, 5)])
def test_additive_equals_multiplicative_at_unit_model(shape, weights, dense):
    """Boundary check between the two noise models. With ``p = 0`` and
    ``mu = 1`` the multiplicative weights ``1 / (U p + mu)`` equal one, so
    both branches must return the same three arrays."""
    n, k = shape
    rng = np.random.default_rng(n + k)
    U = rng.uniform(0.5, 2.0, size=(n, k))
    d = 1.0 + rng.uniform(0.5, 2.0, size=n)  # d - mu > 0: sums of same-sign terms
    w = _weights(weights, n, rng)
    N_inv, sqrt = (np.diag(w), np.diag(np.sqrt(w))) if dense else (w, np.sqrt(w))

    additive = params_space_oper_and_data(
        d, U, None, N_inv, mu=1.0, Ninv_sqrt=sqrt, multiplicative=False
    )
    multiplicative = params_space_oper_and_data(
        d, U, np.zeros(k), N_inv, mu=1.0, Ninv_sqrt=sqrt, multiplicative=True
    )
    for x, y in zip(additive, multiplicative):
        assert np.all(np.isfinite(x))
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("with_sqrt", [False, True])
def test_additive_dense_noise_allocates_no_square_array(with_sqrt):
    """``Sigma_inv = N_inv`` under the additive model, so the terms need
    ``U^T N_inv`` and no further ``(M, M)`` array."""
    n, k = 1000, 4
    rng = np.random.default_rng(0)
    U = rng.random((n, k))
    d = rng.random(n)
    N_inv = np.eye(n) + 0.01
    sqrt = np.linalg.cholesky(N_inv) if with_sqrt else None
    tracemalloc.start()
    try:
        terms = params_space_oper_and_data(
            d, U, None, N_inv, Ninv_sqrt=sqrt, multiplicative=False
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < n * n * 8 / 4, f"peak {peak / 1e6:.1f} MB"
    np.testing.assert_allclose(terms[0], U.T @ N_inv @ U, rtol=1e-12)
    np.testing.assert_allclose(terms[1], U.T @ N_inv @ d, rtol=1e-12)
    if with_sqrt:
        np.testing.assert_allclose(terms[2] @ terms[2].T, terms[0], rtol=1e-10)


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

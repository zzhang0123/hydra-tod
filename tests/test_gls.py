"""Tests for the standalone serial GLS API."""

import subprocess
import sys
import tracemalloc
from textwrap import dedent

import numpy as np
import pytest
from scipy import sparse

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


@pytest.mark.parametrize("multiplicative", [False, True])
@pytest.mark.parametrize("sparse_format", ["csr", "csc", "coo"])
@pytest.mark.parametrize("shape", [(1, 1), (40, 7), (500, 60)])
def test_sparse_operator_matches_dense(shape, sparse_format, multiplicative):
    """With a diagonal ``N_inv``, ``U`` may be a ``scipy.sparse`` array. The
    matrices returned are then sparse and equal the dense ones."""
    n, k = shape
    rng = np.random.default_rng(n + k)
    dense = rng.uniform(0.5, 2.0, size=(n, k)) * (rng.uniform(size=(n, k)) < 0.3)
    dense[np.arange(n), rng.integers(0, k, size=n)] = 1.0  # no empty row
    U = sparse.coo_array(dense).asformat(sparse_format)
    d = rng.uniform(1.0, 2.0, size=n)
    w = rng.uniform(0.5, 2.0, size=n)
    w[::4] = 0.0  # flagged samples
    p = rng.uniform(1.0, 2.0, size=k) if multiplicative else None
    kwargs = dict(mu=0.5, Ninv_sqrt=np.sqrt(w), multiplicative=multiplicative)

    expected = params_space_oper_and_data(d, dense, p, w, **kwargs)
    actual = params_space_oper_and_data(d, U, p, w, **kwargs)

    assert sparse.issparse(actual[0]) and sparse.issparse(actual[2])
    assert isinstance(actual[1], np.ndarray) and actual[1].shape == (k,)
    for sparse_term, dense_term in zip(
        (actual[0], actual[2]), (expected[0], expected[2])
    ):
        np.testing.assert_allclose(
            sparse_term.toarray(), dense_term, rtol=1e-12, atol=1e-12
        )
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)


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

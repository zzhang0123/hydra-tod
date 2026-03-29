"""Tests for linear_solver and linear_sampler."""
import numpy as np
import pytest


def test_cg_solves_simple_system():
    """CG should solve A x = b for a small SPD system."""
    from hydra_tod.linear_solver import cg

    np.random.seed(42)
    n = 10
    A = np.random.randn(n, n)
    A = A @ A.T + n * np.eye(n)  # SPD
    b = np.random.randn(n)

    x = cg(None, b, abs_tol=1e-14, linear_op=lambda v: A @ v)
    np.testing.assert_allclose(A @ x, b, atol=1e-8)


def test_cg_converges():
    """CG should converge for a well-conditioned system."""
    from hydra_tod.linear_solver import cg

    n = 20
    A = np.eye(n) * 5 + np.random.randn(n, n) * 0.01
    A = (A + A.T) / 2
    b = np.random.randn(n)

    x = cg(None, b, abs_tol=1e-14, linear_op=lambda v: A @ v)
    np.testing.assert_allclose(A @ x, b, atol=1e-8)


def test_sample_p_shape():
    """sample_p should return correct shape."""
    from hydra_tod.linear_sampler import sample_p

    np.random.seed(0)
    n, p = 50, 4
    H = np.random.randn(n, p)
    sigma_inv = np.eye(n) * 10
    d = np.random.randn(n)

    sample = sample_p(d, H, sigma_inv, num_samples=0, prior_cov_inv=np.ones(p), prior_mean=np.zeros(p))
    assert sample.shape == (p,)

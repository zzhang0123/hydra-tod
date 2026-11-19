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


def _global_rng_state():
    name, key, pos, has_gauss, cached_gaussian = np.random.get_state()
    return name, key.tobytes(), pos, has_gauss, cached_gaussian


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("n_tod", [1, 2, 3, 5])
def test_normal_equations_do_not_depend_on_draw(n_tod, scale):
    """``draw`` only adds the white-noise term. The estimate and the normal
    equations are the same arrays with and without it, and ``draw=False``
    leaves NumPy's global random state unchanged."""
    from hydra_tod.linear_sampler import iterative_gls_mpi_list

    rng = np.random.default_rng(n_tod)
    n_params = 4
    truth = scale * rng.uniform(5.0, 15.0, size=n_params)
    d_list, U_list, sqrt_list, mu_list = [], [], [], []
    for i in range(n_tod):
        n = 30 + 7 * i
        U = rng.uniform(0.1, 1.0, size=(n, n_params))
        mu = scale * (4.0 + i)
        lag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        cov = 1e-4 * (np.exp(-lag / 5.0) + 0.3 * np.eye(n))
        noise = np.linalg.cholesky(cov) @ rng.standard_normal(n)
        d_list.append((U @ truth + mu) * (1.0 + noise))
        U_list.append(U)
        sqrt_list.append(np.linalg.cholesky(np.linalg.inv(cov)))
        mu_list.append(mu)
    args = (d_list, U_list, sqrt_list, mu_list)

    np.random.seed(5)
    seeded = _global_rng_state()
    p_drawn, A_drawn, b_drawn, wn = iterative_gls_mpi_list(*args, tol=1e-12, draw=True)
    after_draw = _global_rng_state()
    p, A, b, no_wn = iterative_gls_mpi_list(*args, tol=1e-12, draw=False)

    assert after_draw != seeded
    assert _global_rng_state() == after_draw
    assert no_wn is None and wn.shape == (n_params,)
    np.testing.assert_array_equal(p, p_drawn)
    np.testing.assert_array_equal(A, A_drawn)
    np.testing.assert_array_equal(b, b_drawn)
    np.testing.assert_allclose(p, truth, rtol=0.2)  # also rules out NaN == NaN

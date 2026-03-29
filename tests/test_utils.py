"""Tests for utils module."""
import numpy as np
import pytest


def test_lag_list():
    from hydra_tod.utils import lag_list

    t = np.array([5.0, 6.0, 7.0, 8.0])
    lags = lag_list(t)
    np.testing.assert_array_equal(lags, [0.0, 1.0, 2.0, 3.0])


def test_lag_list_single():
    from hydra_tod.utils import lag_list

    lags = lag_list([10.0])
    np.testing.assert_array_equal(lags, [0.0])


def test_polyn_proj():
    from hydra_tod.utils import polyn_proj

    t = np.arange(10, dtype=float)
    proj = polyn_proj(t)
    mat = proj(3)
    assert mat.shape == (10, 3)


def test_cho_compute_mat_inv():
    from hydra_tod.utils import cho_compute_mat_inv

    A = np.array([[4.0, 2.0], [2.0, 3.0]])
    A_inv = cho_compute_mat_inv(A)
    product = A @ A_inv
    np.testing.assert_allclose(product, np.eye(2), atol=1e-12)


def test_cho_compute_mat_inv_sqrt():
    from hydra_tod.utils import cho_compute_mat_inv_sqrt

    A = np.array([[4.0, 2.0], [2.0, 3.0]])
    L_inv = cho_compute_mat_inv_sqrt(A)
    # L_inv @ L_inv.T should equal A_inv
    A_inv = np.linalg.inv(A)
    np.testing.assert_allclose(L_inv @ L_inv.T, A_inv, atol=1e-12)


def test_log_det_symmetric_toeplitz():
    from hydra_tod.utils import log_det_symmetric_toeplitz
    from scipy.linalg import toeplitz

    r = np.array([4.0, 1.0, 0.5, 0.2])
    T = toeplitz(r)
    expected = np.log(np.linalg.det(T))
    result = log_det_symmetric_toeplitz(r)
    assert np.isclose(result, expected, rtol=1e-6)


def test_log_likeli_vs_general():
    """Toeplitz and general log-likelihoods should match for consecutive data."""
    from hydra_tod.utils import log_likeli, log_likeli_general
    from scipy.linalg import toeplitz

    np.random.seed(42)
    r = np.array([4.0, 1.0, 0.5, 0.2, 0.1])
    data = np.random.randn(5)

    ll_toeplitz = log_likeli(r, data)
    C = toeplitz(r)
    ll_general = log_likeli_general(C, data)
    assert np.isclose(ll_toeplitz, ll_general, rtol=1e-8)


def test_log_likeli_general_singular():
    """Should return -inf for a singular matrix."""
    from hydra_tod.utils import log_likeli_general

    C = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1, singular
    data = np.array([1.0, 2.0])
    assert log_likeli_general(C, data) == -np.inf


def test_log_likeli_general_nan_input():
    """Should return -inf for NaN inputs."""
    from hydra_tod.utils import log_likeli_general

    C = np.eye(3)
    data = np.array([1.0, np.nan, 3.0])
    assert log_likeli_general(C, data) == -np.inf


def test_DFT_matrix():
    from hydra_tod.utils import DFT_matrix

    D = DFT_matrix(4)
    assert D.shape == (4, 4)
    # DFT matrix should be unitary (up to scaling)
    product = D @ D.conj().T
    np.testing.assert_allclose(product, 4 * np.eye(4), atol=1e-12)

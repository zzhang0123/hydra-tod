"""Tests for flicker_model: correlation functions and covariance matrices."""
import numpy as np
import pytest
from scipy.linalg import toeplitz


def test_flicker_corr_zero_lag():
    """C(0) should equal the analytic expression."""
    from hydra_tod.flicker_model import flicker_corr

    f0, fc, alpha, var_w = 1e-4, 1e-5, 2.0, 5e-6
    c0 = flicker_corr(0, f0, fc, alpha, var_w=var_w)
    expected = fc / np.pi * (f0 / fc) ** alpha / (alpha - 1) + var_w
    assert np.isclose(c0, expected, rtol=1e-10)


def test_flicker_corr_symmetric():
    """C(tau) == C(-tau)."""
    from hydra_tod.flicker_model import flicker_corr

    f0, fc, alpha = 1e-4, 1e-5, 2.0
    c_pos = flicker_corr(3.0, f0, fc, alpha)
    c_neg = flicker_corr(-3.0, f0, fc, alpha)
    assert np.isclose(c_pos, c_neg, rtol=1e-12)


def test_flicker_cov_vec_first_element():
    """First element of flicker_cov_vec should be C(0)."""
    from hydra_tod.flicker_model import flicker_cov_vec, flicker_corr

    f0, fc, alpha, wn = 1e-4, 1e-5, 2.0, 2.5e-6
    taus = np.array([0.0, 1.0, 2.0, 3.0])
    cov_vec = flicker_cov_vec(taus, f0, fc, alpha, white_n_variance=wn)
    c0 = flicker_corr(0, f0, fc, alpha, var_w=wn)
    assert np.isclose(cov_vec[0], c0, rtol=1e-10)


def test_flicker_cov_is_symmetric_positive_definite():
    """The Toeplitz covariance matrix should be SPD."""
    from hydra_tod.flicker_model import flicker_cov

    t = np.arange(20, dtype=float)
    C = flicker_cov(t, 1e-4, 1e-5, 2.0, white_n_variance=5e-6, only_row_0=False)
    assert np.allclose(C, C.T)
    eigvals = np.linalg.eigvalsh(C)
    assert np.all(eigvals > 0), f"Not positive definite: min eigval = {eigvals.min()}"


def test_flicker_cov_general_matches_toeplitz_for_consecutive():
    """For evenly-spaced times, general should equal Toeplitz."""
    from hydra_tod.flicker_model import flicker_cov_general, flicker_cov_vec
    from hydra_tod.utils import lag_list

    t = np.arange(15, dtype=float)
    f0, fc, alpha, wn = 1e-4, 1e-5, 2.0, 5e-6

    cov_gen = flicker_cov_general(t, f0, fc, alpha, white_n_variance=wn)
    lags = lag_list(t)
    corr = flicker_cov_vec(lags, f0, fc, alpha, white_n_variance=wn)
    cov_toe = toeplitz(corr)

    assert np.allclose(cov_gen, cov_toe, rtol=1e-10)


def test_flicker_cov_general_submatrix():
    """Non-consecutive should be the correct submatrix of the full Toeplitz."""
    from hydra_tod.flicker_model import flicker_cov_general

    f0, fc, alpha, wn = 1e-4, 1e-5, 2.0, 5e-6

    t_full = np.arange(8, dtype=float)
    t_gap = np.array([0.0, 1.0, 2.0, 4.0, 7.0])

    cov_full = flicker_cov_general(t_full, f0, fc, alpha, white_n_variance=wn)
    cov_gap = flicker_cov_general(t_gap, f0, fc, alpha, white_n_variance=wn)

    idx = [0, 1, 2, 4, 7]
    cov_sub = cov_full[np.ix_(idx, idx)]
    assert np.allclose(cov_gap, cov_sub, rtol=1e-10)


def test_flicker_cov_general_is_spd():
    """General covariance for non-consecutive times should still be SPD."""
    from hydra_tod.flicker_model import flicker_cov_general

    t = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0])
    C = flicker_cov_general(t, 1e-4, 1e-5, 2.0, white_n_variance=5e-6)
    assert np.allclose(C, C.T)
    eigvals = np.linalg.eigvalsh(C)
    assert np.all(eigvals > 0), f"Not positive definite: min eigval = {eigvals.min()}"


def test_sim_noise_shape():
    """sim_noise should return the correct shape."""
    from hydra_tod.flicker_model import sim_noise

    t = np.arange(20, dtype=float)
    n1 = sim_noise(1e-4, 1e-5, 2.0, t, n_samples=1)
    assert n1.shape == (20,)

    n3 = sim_noise(1e-4, 1e-5, 2.0, t, n_samples=3)
    assert n3.shape == (3, 20)

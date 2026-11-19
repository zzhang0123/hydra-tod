"""Regression coverage for random draws in the multi-TOD temperature step."""

import numpy as np
import pytest

from hydra_tod import mpiutil, tsys_sampler


@pytest.mark.parametrize("estimate_only", [True, False])
def test_multi_tod_estimation_draws_only_for_sampling(monkeypatch, estimate_only):
    design = [
        np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]),
        np.array([[1.0, -1.0], [1.0, 0.0]]),
    ]
    truth = np.array([10.0, 2.0])
    known_offsets = [2.0, 3.0]
    gain = [np.full(3, 2.0), np.full(2, 3.0)]
    model = [U @ truth + mu for U, mu in zip(design, known_offsets)]
    data = [d * g for d, g in zip(model, gain)]
    monkeypatch.setattr(
        tsys_sampler, "flicker_cov", lambda times, *a, **k: np.eye(len(times))
    )
    draws = []

    def standard_normal(loc=0.0, scale=1.0, size=None):
        if estimate_only:
            pytest.fail("estimation mode generated a random draw")
        noise = np.ones(size)
        draws.append(noise.shape)
        return noise

    monkeypatch.setattr(np.random, "normal", standard_normal)
    prior_precision = np.array([0.04, 0.16])
    prior_mean = np.array([8.0, 1.0])
    result = tsys_sampler.Tsys_sampler_multi_TODs(
        data,
        [np.arange(len(d)) for d in data],
        gain,
        design,
        [(-4.0, 2.0)] * 2,
        [-3.0] * 2,
        local_mu_list=known_offsets,
        prior_cov_inv=prior_precision,
        prior_mean=prior_mean,
        solver=None,
        init_coeffs=truth,
        Est_mode=estimate_only,
    )

    whitened = np.vstack([U / d[:, None] for U, d in zip(design, model)])
    precision = mpiutil.size * (whitened.T @ whitened)
    rhs = precision @ truth + prior_precision * prior_mean
    if not estimate_only:
        rhs += mpiutil.size * whitened.sum(axis=0) + np.sqrt(prior_precision)
    expected = np.linalg.solve(precision + np.diag(prior_precision), rhs)
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
    expected_draws = []
    if not estimate_only:
        expected_draws = [(3,), (2,)]
        if mpiutil.rank0:
            expected_draws.append((2,))
    assert draws == expected_draws

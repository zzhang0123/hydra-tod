"""Tests for MCMC diagnostics."""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI


def test_acf_1d_at_lag_zero():
    """ACF at lag 0 should be 1."""
    from hydra_tod.mcmc_diagnostics import acf_1d

    x = np.random.randn(500)
    acf = acf_1d(x, max_lag=10)
    assert np.isclose(acf[0], 1.0, atol=1e-10)


def test_ess_1d_iid():
    """ESS of iid samples should be close to n."""
    from hydra_tod.mcmc_diagnostics import ess_1d

    np.random.seed(0)
    x = np.random.randn(1000)
    ess = ess_1d(x)
    # For iid, ESS ~ n (allow 50% tolerance)
    assert ess > 500, f"ESS too low for iid: {ess}"


def test_ess_1d_constant():
    """ESS of a constant chain should return n."""
    from hydra_tod.mcmc_diagnostics import ess_1d

    x = np.ones(100)
    ess = ess_1d(x)
    assert ess == 100


def test_rhat_split_converged():
    """Rhat of identical chains should be ~1."""
    from hydra_tod.mcmc_diagnostics import rhat_split

    np.random.seed(0)
    chains = np.random.randn(4, 1000)
    rhat = rhat_split(chains)
    assert 0.99 < rhat < 1.05, f"Rhat not close to 1: {rhat}"


def test_diagnostics_output_shape():
    """diagnostics() should return summary for each parameter."""
    from hydra_tod.mcmc_diagnostics import diagnostics
    import matplotlib.pyplot as plt

    np.random.seed(0)
    samples = np.random.randn(2, 500, 3)
    summary = diagnostics(samples, param_names=["a", "b", "c"], max_plots=0)
    plt.close("all")

    assert set(summary.keys()) == {"a", "b", "c"}
    for v in summary.values():
        assert "ESS_min" in v
        assert "ESS_median" in v
        assert "Rhat_split" in v

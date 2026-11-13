from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

# ========== Generic MCMC Diagnostics Toolkit ==========
# Input: samples as numpy array of shape (n_chains, n_draws, n_params)


def acf_1d(x: NDArray[np.floating], max_lag: int = 200) -> NDArray[np.floating]:
    """
    Compute the normalised autocorrelation function (ACF) of a 1-D series.

    Uses the FFT-based method for efficient computation of the ACF up to
    ``max_lag``.

    Parameters
    ----------
    x : NDArray[np.floating]
        Input time series of shape ``(n,)``.
    max_lag : int, optional
        Maximum lag to compute. Default is 200.

    Returns
    -------
    acf : NDArray[np.floating]
        Normalised ACF values from lag 0 to ``max_lag`` (inclusive), shape
        ``(max_lag + 1,)``.

    Notes
    -----
    The ACF is normalised by the variance and the number of overlapping
    samples at each lag.
    """
    x = np.asarray(x)
    x = x - x.mean()
    n = len(x)
    var = np.var(x, ddof=0)
    fft = np.fft.rfft(x, n=2 * n)
    acf_full = np.fft.irfft(fft * np.conjugate(fft))[:n]
    acf_full = acf_full / (var * np.arange(n, 0, -1))
    return acf_full[: max_lag + 1]


def ess_1d(x: NDArray[np.floating]) -> float:
    """
    Estimate the effective sample size (ESS) of a 1-D MCMC chain.

    Computes the ESS using the initial-positive-sequence estimator: the ACF
    is summed until the first non-positive value is encountered.

    Parameters
    ----------
    x : NDArray[np.floating]
        MCMC chain of shape ``(n,)``.

    Returns
    -------
    ess : float
        Estimated effective sample size.

    Notes
    -----
    If the variance of *x* is zero (constant chain), returns ``n`` directly.
    """
    x = np.asarray(x)
    x = x - x.mean()
    n = len(x)
    var = np.var(x, ddof=0)
    if var == 0:
        return n
    fft = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(fft * np.conjugate(fft))[:n]
    acf = acf / (var * np.arange(n, 0, -1))
    positive = acf[1:]
    pos_idx = np.argmax(positive <= 0.0)
    k = int(pos_idx) if (positive <= 0.0).any() else len(positive)
    s = 1 + 2 * np.sum(acf[1 : 1 + k])
    return n / s if s > 0 else n


def rhat_split(chains_dim: NDArray[np.floating]) -> float:
    r"""
    Compute the split-\ :math:`\hat{R}` convergence diagnostic.

    Each chain is split in half, doubling the number of chains and halving
    their length, before computing the standard :math:`\hat{R}` statistic.

    Parameters
    ----------
    chains_dim : NDArray[np.floating]
        Array of shape ``(n_chains, n_draws)`` for a single parameter
        dimension.

    Returns
    -------
    Rhat : float
        Split-\ :math:`\hat{R}` value.  Values close to 1.0 indicate
        convergence; values above ~1.01 suggest the chains have not mixed.

    References
    ----------
    Gelman, A. et al. (2013), *Bayesian Data Analysis*, 3rd edition.
    """
    m, n = chains_dim.shape
    if n % 2 == 1:
        chains_dim = chains_dim[:, :-1]
        n -= 1
    halves = np.reshape(chains_dim, (m * 2, n // 2))
    m2, n2 = halves.shape
    chain_means = halves.mean(axis=1)
    chain_vars = halves.var(axis=1, ddof=1)
    B = n2 * np.var(chain_means, ddof=1)
    W = np.mean(chain_vars)
    var_hat = (n2 - 1) / n2 * W + B / n2
    Rhat = np.sqrt(var_hat / W)
    return float(Rhat)


def diagnostics(
    samples: NDArray[np.floating],
    param_names: list[str] | None = None,
    max_plots: int = 3,
) -> dict[str, dict[str, float]]:
    r"""
    Compute and display MCMC convergence diagnostics.

    For each parameter, computes the minimum and median effective sample
    size (ESS) across chains and the split-\ :math:`\hat{R}`.
    Optionally plots trace plots for the first few parameters.

    Parameters
    ----------
    samples : NDArray[np.floating]
        MCMC samples of shape ``(n_chains, n_draws, n_params)``.
    param_names : list of str or None, optional
        Names for each parameter dimension.  If ``None``, defaults to
        ``['param0', 'param1', ...]``.
    max_plots : int, optional
        Maximum number of parameters to produce trace plots for.
        Default is 3.

    Returns
    -------
    summary : dict
        Dictionary keyed by parameter name, each containing:

        - ``'ESS_min'`` : minimum ESS across chains.
        - ``'ESS_median'`` : median ESS across chains.
        - ``'Rhat_split'`` : split-Rhat value.
    """
    n_chains, n_draws, n_params = samples.shape
    if param_names is None:
        param_names = [f"param{i}" for i in range(n_params)]

    summary: dict[str, dict[str, float]] = {}
    for d, name in enumerate(param_names):
        ess_list = [ess_1d(samples[c, :, d]) for c in range(n_chains)]
        summary[name] = {
            "ESS_min": float(np.min(ess_list)),
            "ESS_median": float(np.median(ess_list)),
            "Rhat_split": rhat_split(samples[:, :, d]),
        }

    # --- Plot some quick diagnostics for first few parameters ---
    for d in range(min(n_params, max_plots)):
        name = param_names[d]

        # Trace
        plt.figure()
        for c in range(n_chains):
            plt.plot(samples[c, :, d], alpha=0.6, lw=0.8)
        plt.title(f"Trace plot: {name}")
        plt.xlabel("Iteration")
        plt.ylabel(name)
        plt.show()

    return summary

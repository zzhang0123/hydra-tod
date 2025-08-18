import numpy as np
import matplotlib.pyplot as plt

# ========== Generic MCMC Diagnostics Toolkit ==========
# Input: samples as numpy array of shape (n_chains, n_draws, n_params)

def acf_1d(x, max_lag=200):
    x = np.asarray(x)
    x = x - x.mean()
    n = len(x)
    var = np.var(x, ddof=0)
    fft = np.fft.rfft(x, n=2*n)
    acf_full = np.fft.irfft(fft * np.conjugate(fft))[:n]
    acf_full = acf_full / (var * np.arange(n, 0, -1))
    return acf_full[:max_lag+1]

def ess_1d(x):
    x = np.asarray(x)
    x = x - x.mean()
    n = len(x)
    var = np.var(x, ddof=0)
    if var == 0:
        return n
    fft = np.fft.rfft(x, n=2*n)
    acf = np.fft.irfft(fft * np.conjugate(fft))[:n]
    acf = acf / (var * np.arange(n, 0, -1))
    positive = acf[1:]
    pos_idx = np.argmax(positive <= 0.0)
    k = int(pos_idx) if (positive <= 0.0).any() else len(positive)
    s = 1 + 2 * np.sum(acf[1:1+k])
    return n / s if s > 0 else n

def rhat_split(chains_dim):
    m, n = chains_dim.shape
    if n % 2 == 1:
        chains_dim = chains_dim[:, :-1]
        n -= 1
    halves = np.reshape(chains_dim, (m*2, n//2))
    m2, n2 = halves.shape
    chain_means = halves.mean(axis=1)
    chain_vars = halves.var(axis=1, ddof=1)
    B = n2 * np.var(chain_means, ddof=1)
    W = np.mean(chain_vars)
    var_hat = (n2 - 1)/n2 * W + B/n2
    Rhat = np.sqrt(var_hat / W)
    return float(Rhat)

def diagnostics(samples, param_names=None, max_plots=3):
    n_chains, n_draws, n_params = samples.shape
    if param_names is None:
        param_names = [f"param{i}" for i in range(n_params)]
    
    summary = {}
    for d, name in enumerate(param_names):
        ess_list = [ess_1d(samples[c, :, d]) for c in range(n_chains)]
        summary[name] = {
            "ESS_min": float(np.min(ess_list)),
            "ESS_median": float(np.median(ess_list)),
            "Rhat_split": rhat_split(samples[:, :, d])
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
        
        # ACF (first chain)
        acf_vals = acf_1d(samples[0, :, d])
        plt.figure()
        plt.plot(acf_vals)
        plt.title(f"ACF: {name} (chain 1)")
        plt.xlabel("Lag")
        plt.ylabel("ACF")
        plt.show()
        
        # Cumulative mean (first chain)
        cum_mean = np.cumsum(samples[0, :, d]) / np.arange(1, n_draws+1)
        plt.figure()
        plt.plot(cum_mean)
        plt.title(f"Cumulative mean: {name} (chain 1)")
        plt.xlabel("Iteration")
        plt.ylabel("Cumulative mean")
        plt.show()
        
        # Histogram
        flat = samples[:, :, d].reshape(-1)
        plt.figure()
        plt.hist(flat, bins=50, density=True, alpha=0.6)
        plt.title(f"Posterior marginal: {name}")
        plt.xlabel(name)
        plt.ylabel("Density")
        plt.show()
    
    return summary


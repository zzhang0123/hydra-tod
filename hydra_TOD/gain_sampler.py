import numpy as np
from linear_solver import cg
from utils import cho_compute_mat_inv
from flicker_model import flicker_cov
from linear_sampler import sample_p_old, iterative_gls
from scipy.linalg import toeplitz

try:
    import pickle
    import os
    from jax import jit
    
    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    corr_emulator_path = os.path.join(module_dir, 'flicker_corr_emulator.pkl')
    logdet_emulator_path = os.path.join(module_dir, 'flicker_logdet_emulator.pkl')

    # Import the class definition before unpickling
    from flicker_model import FlickerCorrEmulator
    # Load the emulator
    with open(corr_emulator_path, 'rb') as f:
        flicker_cov = pickle.load(f)

    from flicker_model import LogDetEmulator
    with open(logdet_emulator_path, 'rb') as f:
        flicker_logdet = pickle.load(f)
        
    use_emulator = True
    print("Using the emulator for flicker noise correlation function.")

    # JIT the flicker covariance function
    flicker_cov_jax = jit(flicker_cov.create_jax())
    flicker_logdet_jax = jit(flicker_logdet.create_jax())

except (FileNotFoundError, ImportError, AttributeError) as e:
    print(f"Emulator for flicker noise correlation function not found or failed to load: {e}")
    print("Using flicker_cov_vec instead.")
    from flicker_model import flicker_cov_vec as flicker_cov
    use_emulator = False

def linear_gain_sampler(data, 
                        t_list,
                        gain_proj, 
                        Tsys, 
                        noise_params,
                        logfc,
                        wnoise_var=2.5e-6, 
                        mu=0.0,
                        n_samples=1,
                        tol=1e-13,
                        prior_cov_inv=None, 
                        prior_mean=None, 
                        solver=cg):
    d_vec = data/Tsys
    logf0, alpha = noise_params
    if use_emulator:
        Ncov_inv = cho_compute_mat_inv( toeplitz(flicker_cov_jax(logf0, alpha)[0]) )
    else:
        Ncov_inv = cho_compute_mat_inv( flicker_cov(t_list, 10.**logf0, 10.**logfc, alpha,  white_n_variance=wnoise_var, only_row_0=False) )
    p_GLS, sigma_inv = iterative_gls(d_vec, gain_proj, Ncov_inv, mu=mu, tol=tol)
    return sample_p_old(d_vec, gain_proj, sigma_inv, num_samples=n_samples, mu=mu, prior_cov_inv=prior_cov_inv, prior_mean=prior_mean, solver=solver)

def log_gain_sampler(data, 
                     t_list,
                     gain_proj, 
                     Tsys, 
                     noise_params,
                     logfc,
                     wnoise_var=2.5e-6, 
                     mu=0.0,
                     n_samples=1,
                     prior_cov_inv=None, 
                     prior_mean=None, 
                     solver=cg):
    d_vec = np.log(data/Tsys)
    logf0, alpha = noise_params
    if use_emulator:
        Ncov_inv = cho_compute_mat_inv( toeplitz(flicker_cov_jax(logf0, alpha)[0]) )
    else:
        Ncov_inv = cho_compute_mat_inv( flicker_cov(t_list, 10.**logf0, 10.**logfc, alpha,  white_n_variance=wnoise_var, only_row_0=False) )
    return sample_p_old(d_vec, gain_proj, Ncov_inv, num_samples=n_samples, mu=mu, prior_cov_inv=prior_cov_inv, prior_mean=prior_mean, solver=solver)

def factorized_gain_sampler(data, 
                          t_list,
                          gain_proj, 
                          Tsys, 
                          noise_params,
                          logfc,
                          wnoise_var=2.5e-6, 
                          mu=1.0,
                          n_samples=1,
                          tol=1e-13,
                          prior_cov_inv=None, 
                          prior_mean=None, 
                          solver=cg):
    DC_gain, logf0, alpha = noise_params
    d_vec = data/Tsys/DC_gain
    if use_emulator:
        Ncov_inv = cho_compute_mat_inv( toeplitz(flicker_cov_jax(logf0, alpha)[0]) )
    else:
        Ncov_inv = cho_compute_mat_inv( flicker_cov(t_list, 10.**logf0, 10.**logfc, alpha,  white_n_variance=wnoise_var, only_row_0=False) )
    p_GLS, sigma_inv = iterative_gls(d_vec, gain_proj, Ncov_inv, mu=mu, tol=tol)
    return sample_p_old(d_vec, gain_proj, sigma_inv, num_samples=n_samples, mu=mu, prior_cov_inv=prior_cov_inv, prior_mean=prior_mean, solver=solver)

def gain_sampler(data,
                 t_list,
                 gain_proj,
                 Tsys,
                 noise_params,
                 logfc,
                 model="linear",
                 wnoise_var=2.5e-6,
                 n_samples=1,
                 tol=1e-13,
                 prior_cov_inv=None,
                 prior_mean=None,
                 solver=cg):
    if model == "linear":
        sample = linear_gain_sampler(data, t_list, gain_proj, Tsys, noise_params, logfc, wnoise_var=wnoise_var, mu=0.0, n_samples=n_samples, tol=tol, prior_cov_inv=prior_cov_inv, prior_mean=prior_mean, solver=solver)
        gains = gain_proj@sample
    elif model == "log":
        sample = log_gain_sampler(data, t_list, gain_proj, Tsys, noise_params, logfc, wnoise_var=wnoise_var, mu=0.0, n_samples=n_samples, prior_cov_inv=prior_cov_inv, prior_mean=prior_mean, solver=solver)
        gains = np.exp(gain_proj@sample)
    elif model == "factorized":
        sample = factorized_gain_sampler(data, t_list, gain_proj, Tsys, noise_params, logfc, wnoise_var=wnoise_var, mu=1.0, n_samples=n_samples, tol=tol, prior_cov_inv=prior_cov_inv, prior_mean=prior_mean, solver=solver)
        gains = gain_proj@sample + 1.0
    else:
        raise ValueError(f"Unknown smooth_gain_model: {model}")
    
    return sample, gains

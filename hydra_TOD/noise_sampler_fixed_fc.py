import numpy as np
# from comat import logdet_quad
from scipy.linalg import solve_toeplitz
from utils import lag_list
from mcmc_sampler import mcmc_sampler
import jax.numpy as jnp
from jax.scipy.linalg import solve
import jax
from nuts_sampler import NUTS_sampler
from bayes_util import wrap_with_priors, make_transforms, constrained_to_unconstrained, unconstrained_to_constrained
from scipy.linalg._solve_toeplitz import levinson

# if the emulator of the correlation function exists, load it
# otherwise, use flicker_cov_vec
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
    from flicker_model import flicker_cov_vec
    use_emulator = False

def log_det_symmetric_toeplitz(r):
    r = np.asarray(r)
    n = len(r)  
    a0 = r[0] 
    
    b = np.zeros(n, dtype=r.dtype)
    b[:n-1] = -r[1:] 
    
    a = np.concatenate((r[::-1], r[1:]))
    x, reflection_coeff = levinson(a, b)
    
    k = reflection_coeff[1:n]  
    k = np.clip(k, -0.999999, 0.999999)  # Clip values close to 1 to avoid numerical issues
    
    factors = np.arange(n-1, 0, -1)
    terms = np.log(1 - k**2)
    result=n * np.log(a0) + np.dot(factors, terms) 
    if np.isnan(result):
        return -np.inf
    else:
        return result

def log_likeli(corr_list, data):
    # Add parameter validation
    if np.any(np.isnan(corr_list)) or np.any(np.isinf(corr_list)):
        return -np.inf
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return -np.inf
        
    try:
        result = np.dot(data, solve_toeplitz(corr_list, data)) + log_det_symmetric_toeplitz(corr_list)
        return -0.5 * result
    except:
        return -np.inf
    
def log_likeli_emu(logf0, alpha, data, DC_scaler=1.0):

    corr_list = flicker_cov_jax(logf0, alpha)[0]
    # # Add parameter validation
    # if np.any(np.isnan(corr_list)) or np.any(np.isinf(corr_list)):
    #     return -np.inf
    # if np.any(np.isnan(data)) or np.any(np.isinf(data)):
    #     return -np.inf
    data /= DC_scaler
    try:
        result = np.dot(data, solve_toeplitz(corr_list, data)) + flicker_logdet_jax(logf0, alpha)[0]
        return -0.5 * result
    except:
        return -np.inf


@jit
def log_likeli_emu_jax(logf0, alpha, data, DC_scaler=1.0):
    """
    JAX version of log_likeli_emu function with validation removed.
    """
    # Get correlation list and logdet using emulators (need JAX versions)
    corr_list = flicker_cov_jax(logf0, alpha)[0]
    logdet = flicker_logdet_jax(logf0, alpha)[0]

    # Convert to JAX arrays
    corr_list = jnp.asarray(corr_list)
    data = jnp.asarray(data/DC_scaler)

    n = len(data)
    # Build Toeplitz matrix
    indices = jnp.arange(n)
    toeplitz_matrix = corr_list[jnp.abs(indices[:, None] - indices[None, :])]
    
    # Solve the linear system
    solved = solve(toeplitz_matrix, data, assume_a='pos')
    quad_form = jnp.dot(data, solved)
    
    result = quad_form + logdet
    return -0.5 * result

   
# Define the likelihood function for the flicker noise model.
def flicker_likeli_func(data, gain, Tsys, logfc, time_list=None, wnoise_var=2.5e-6):
    '''
    Note that f0 and fc are in unit of angular frequency, differently from that of FFT frequencies by a factor of 2pi.
    '''
    # corr_list = flicker_cov(tau_list, f0, fc, alpha,  white_n_variance=wnoise_var, only_row_0=True)

    dvec = data / gain / Tsys - 1.
    dvec = np.asarray(dvec, dtype=np.float64)

    # Let dvec be mean-centered
    dvec -= np.mean(dvec)

    if use_emulator:
        def log_like(params):
            logf0, alpha = params
            return log_likeli_emu(logf0, alpha, dvec)
    else:
        tau_list = lag_list(time_list)
        def log_like(params):
            logf0, alpha = params
            corr_list = flicker_cov_vec(tau_list, 10.**logf0, 10.**logfc, alpha,  white_n_variance=wnoise_var)
            return log_likeli(corr_list, dvec)
    return log_like

def flicker_log_post_JAX(data, gain, Tsys, 
                         include_DC_Gain=False, 
                         prior_func=None, 
                         jeffreys=False, 
                         transform=False,
                         bounds=None):
    """
    JAX-compatible likelihood function for NUTS sampling.
    """
    # Convert inputs to JAX arrays
    data = jnp.asarray(data)
    gain = jnp.asarray(gain)
    Tsys = jnp.asarray(Tsys)
    
    # Preprocess data using JAX operations
    dvec = data / gain / Tsys - 1.0
    dvec = dvec - jnp.mean(dvec)  # Mean-center using JAX
    
    assert use_emulator

    if jeffreys is False:
        log_ll = log_likeli_emu
        jaxjit = False
    else:
        log_ll = log_likeli_emu_jax
        jaxjit = True

    if include_DC_Gain:
        @wrap_with_priors(add_jeffreys=jeffreys, prior_func=prior_func, bounds=bounds, dim=3, jaxjit=jaxjit, transform=transform)
        def log_like(params):
            DC_gain, logf0, alpha = params
            likelihood = log_ll(logf0, alpha, dvec, DC_scaler=DC_gain)
            return likelihood
    else:
        @wrap_with_priors(add_jeffreys=jeffreys, prior_func=prior_func, bounds=bounds, dim=2, jaxjit=jaxjit, transform=transform)
        def log_like(params):
            logf0, alpha = params
            likelihood = log_ll(logf0, alpha, dvec)
            return likelihood

    return log_like

def flicker_sampler(TOD,
                    gains,
                    Tsys,
                    init_params=None,
                    n_samples=1,
                    include_DC_Gain=False, 
                    prior_func=None,
                    jeffreys=True,
                    bounds=None,
                    transform=False,
                    sampler='emcee',
                    rng_key=None):
    
    # log_likeli_func = flicker_likeli_func(t_list, TOD, gains, Tsys, logfc, wnoise_var=wnoise_var, bounds=bounds)

    dim = 3 if include_DC_Gain else 2

    if sampler == 'emcee':
        log_post = flicker_log_post_JAX(TOD, gains, Tsys, 
                                        include_DC_Gain=include_DC_Gain, 
                                        prior_func=prior_func, 
                                        jeffreys=jeffreys, 
                                        bounds=bounds,
                                        transform=transform)
        if transform:
            tfms = make_transforms(bounds, dim)
            if init_params is not None:
                theta0 = jnp.atleast_1d(jnp.array(init_params))
                # Invert to z0
                z0 = constrained_to_unconstrained(theta0, tfms)
                print(f"Initial params in unconstrained space: {z0}")
            else:
                z0 = jnp.zeros(dim)
        else:
            z0 = init_params
        zsample = mcmc_sampler(
            log_post, 
            z0, 
            p_std=0.2,  # Large step size in unconstrained space
            nsteps=80,  # steps for each chain
            n_samples=n_samples,
            return_sampler=False
        )
        if transform:
            sample, _ = unconstrained_to_constrained(zsample, tfms)
        else:
            sample = zsample

    elif sampler == 'nuts':
        log_ll = flicker_log_post_JAX(
            TOD, gains, Tsys, 
            include_DC_Gain=include_DC_Gain, 
            prior_func=prior_func, 
            jeffreys=jeffreys, 
            bounds=None,
            transform=False
        )
        sample = NUTS_sampler(
            log_ll, 
            init_params=None,
            event_shape=(dim,), 
            initial_warmup=1500, 
            max_warmup=5000, 
            N_samples=n_samples,
            target_r_hat=1.01, 
            single_return=True,
            N_chains=4,
            rng_key=rng_key,
            prior_type=None,
            bounds=bounds
        )
    else:
        raise ValueError(f"Unknown sampler: {sampler}")
        
    return sample


import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import jit
from nuts_sampler import NUTS_sampler
from bayes_util import wrap_with_priors

try:
    import pickle
    import os
    
    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    corr_emulator_path = os.path.join(module_dir, 'flicker_corr_emulator.pkl')
    logdet_emulator_path = os.path.join(module_dir, 'flicker_logdet_emulator.pkl')

    # Import the class definition before unpickling
    from flicker_model import FlickerCorrEmulator
    # Load the emulator
    with open(corr_emulator_path, 'rb') as f:
        flicker_cov = pickle.load(f)
except (FileNotFoundError, ImportError, AttributeError) as e:
    print(f"Error loading flicker covariance emulator: {e}")

flicker_cov_jax = flicker_cov.create_jax()
# JIT the flicker covariance function
flicker_cov_jax = jit(flicker_cov_jax)


def log_likelihood(params, data, Tsys_sky, gain_proj, Tnd_vec, Tloc_proj, logf0, alpha):
    """
    Log-likelihood function optimized for JIT compilation.
    Separated logf0, alpha from noise_params tuple for better JIT performance.
    """
    p_gain = params[:4]
    gains = jnp.exp(gain_proj @ p_gain)

    Tnd = jnp.exp(params[4])
    Tsys_nd = Tnd * Tnd_vec
    
    p_loc = params[5:]
    Tsys_loc = jnp.exp(Tloc_proj @ p_loc)

    d_vec = data / gains / (Tsys_loc + Tsys_nd + Tsys_sky) - 1.0

    corr_list = flicker_cov_jax(logf0, alpha)[0] 

    n = len(d_vec)
    # Build Toeplitz matrix
    indices = jnp.arange(n)
    toeplitz_matrix = corr_list[jnp.abs(indices[:, None] - indices[None, :])]

    # Solve the linear system
    solved = solve(toeplitz_matrix, d_vec, assume_a='pos')
    quad_form = jnp.dot(d_vec, solved)

    return -0.5 * quad_form

# JIT-compiled version for repeated calls with same static arguments
@jit
def log_likelihood_jit(params, data, Tsys_sky, gain_proj, Tnd_vec, Tloc_proj, logf0, alpha):
    """JIT-compiled version of log_likelihood for better performance."""
    return log_likelihood(params, data, Tsys_sky, gain_proj, Tnd_vec, Tloc_proj, logf0, alpha)


def local_params_sampler(data, Tsys_sky, gain_proj, Tnd_vec, Tloc_proj, noise_params,
                         rng_key=None,
                         add_jeffreys=True,
                         prior_func=None,
                         bounds=None,
                         jaxjit=True
                        ): 
    
    # Extract noise parameters for JIT compilation
    logf0, alpha = noise_params

    # Turn numpy objects to jnp objects
    data = jnp.asarray(data)
    Tsys_sky = jnp.asarray(Tsys_sky)
    gain_proj = jnp.asarray(gain_proj)
    Tnd_vec = jnp.asarray(Tnd_vec)
    Tloc_proj = jnp.asarray(Tloc_proj)

    # Create a wrapper that uses the JIT-compiled function
    @wrap_with_priors(add_jeffreys=add_jeffreys, prior_func=prior_func, bounds=bounds, dim=9, jaxjit=jaxjit)
    def likelihood_wrapper(params):
        return log_likelihood_jit(params, data, Tsys_sky, gain_proj, Tnd_vec, Tloc_proj, logf0, alpha)
    
    sample = NUTS_sampler(likelihood_wrapper, 
                    init_params=None,
                    log_likeli_args=(), 
                    event_shape=(9,), 
                    initial_warmup=1500, 
                    max_warmup=5000, 
                    N_samples=1000,
                    target_r_hat=1.01, 
                    single_return=True,
                    N_chains=4,
                    rng_key=rng_key,
                    prior_type=None
                   )
    
    return sample


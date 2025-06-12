import numpy as np
# from comat import logdet_quad
from scipy.linalg import solve_toeplitz
from utils import lag_list
from flicker_model import flicker_cov_vec
from mcmc_sampler import mcmc_sampler
import emcee

from scipy.linalg._solve_toeplitz import levinson

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

   
# Define the likelihood function for the flicker noise model.
def flicker_likeli_func(time_list, data, gain, Tsys, logfc, wnoise_var=2.5e-6, boundaries=None):
    '''
    Note that f0 and fc are in unit of angular frequency, differently from that of FFT frequencies by a factor of 2pi.
    '''
    # corr_list = flicker_cov(tau_list, f0, fc, alpha,  white_n_variance=wnoise_var, only_row_0=True)
    tau_list = lag_list(time_list)

    dvec = data / gain / Tsys - 1.
    dvec = np.asarray(dvec, dtype=np.float64)

    # Let dvec be mean-centered
    dvec -= np.mean(dvec)

    if boundaries is not None:
        def log_like(params):
            logf0, alpha = params
            if logf0 < boundaries[0][0] or logf0 > boundaries[0][1] or alpha < boundaries[1][0] or alpha > boundaries[1][1]:
                return -np.inf  # Log of zero for invalid regions
            corr_list = flicker_cov_vec(tau_list, 10.**logf0, 10.**logfc, alpha,  white_n_variance=wnoise_var)
            return log_likeli(corr_list, dvec)
    else:
        def log_like(params):
            logf0, alpha = params
            corr_list = flicker_cov_vec(tau_list, 10.**logf0, 10.**logfc, alpha,  white_n_variance=wnoise_var)
            return log_likeli(corr_list, dvec)

    return log_like


def flicker_noise_sampler(TOD,
                          t_list,
                          gains,
                          Tsys,
                          init_params,
                          logfc,
                          n_samples=1,
                          wnoise_var=2.5e-6,
                          prior_func=None,
                          num_Jeffrey=False,
                          boundaries=None,):
    if boundaries is None:
        boundaries = [[-6, -3. ], [1.3, 4.]]  # Default boundaries
    
    log_likeli = flicker_likeli_func(t_list, TOD, gains, Tsys, logfc, wnoise_var=wnoise_var, boundaries=boundaries)

    return mcmc_sampler(log_likeli, 
                        init_params, 
                        p_std=0.2, 
                        nsteps=50,  # steps for each chain
                        n_samples=n_samples,
                        prior_func=prior_func,
                        num_Jeffrey=num_Jeffrey,
                        return_sampler=False)

def generate_log_prob_func(t_list, data, gain, Tsys, fc, wnoise_var=2.5e-6, log_scale=True):

    dvec = data / gain / Tsys - 1
    dvec = np.asarray(dvec, dtype=np.float64)
    tau_list = lag_list(t_list)
    var_white = wnoise_var


    def log_prob(params, prior_func=None, log_scale=log_scale):
        if log_scale:
            f0, alpha = 10**params[0], params[2]
        else:
            f0, alpha = params

        if alpha < 1.001 or alpha > 5 or f0 < 1e-10 or f0 > 1e3:
            return -np.inf  # Log of zero for invalid regions
        else: 
            corr_list = flicker_cov_vec(tau_list, 
                                    f0, fc, alpha,  
                                    white_n_variance=var_white)

            if prior_func is None:
                return log_likeli(corr_list, dvec)
            else:
                return log_likeli(corr_list, dvec) + prior_func(params)

    return log_prob

# Define an MCMC sampler
def noise_params_sampler(t_list, data, gain, Tsys, wnoise_var, fc,
                        initial_guess=np.array([-3 , 2]),
                        nwalkers=4, 
                        nsteps=200, 
                        n_samples=1,
                        log_scale=True, 
                        prior_func=None,
                        return_sampler=False):
    '''
    This function samples the noise parameters using MCMC.

    Output:
        a single sample of [f0, alpha]
    '''
    log_prob = generate_log_prob_func(t_list, data, gain, Tsys, fc,
                                      wnoise_var=wnoise_var, 
                                      log_scale=log_scale)
    ndim = 2

    # Initialize the walkers
    p0 = np.random.randn(nwalkers, ndim)*0.05 + initial_guess

    # Run the MCMC sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(prior_func, log_scale))
    sampler.run_mcmc(p0, nsteps, progress=True)

    if return_sampler:
        return sampler
    
    # Estimate the autocorrelation time
    try:
        tau = sampler.get_autocorr_time()
        burnin = int(3 * np.max(tau))  # Burn-in time (usually 2-3 times the autocorrelation time)
        thin = int(0.5 * np.min(tau))  # Thinning factor (usually 1/2 of the autocorrelation time)
    except:
        burnin = nsteps // 3
        thin = 1
    
    # Get the chain after burn-in
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    
    if n_samples == 1:
        # Randomly select one sample
        idx = np.random.randint(len(flat_samples))
        return flat_samples[idx]
    else:
        # Pick the last n_samples from the chain
        return flat_samples[-n_samples:]

# Define Jefferay's prior function


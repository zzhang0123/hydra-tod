import numpy as np
from numpy.linalg import slogdet
#from scipy.differentiate import hessian
import emcee
import logging, warnings


def hessian(func, params, epsilon=1e-5):
    """
    Compute the Hessian matrix of a function using numerical differentiation.
    
    Parameters:
    func : callable
        The log-probability function that takes a parameter vector as input.
    params : np.ndarray
        The parameter values at which to evaluate the Hessian.
    epsilon : float, optional
        A small value for numerical differentiation (default is 1e-5).
    
    Returns:
    np.ndarray
        The Hessian matrix.
    """
    params = np.asarray(params)
    n = params.size
    hessian_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):  # Compute only for upper triangle
            params_ij = params.copy()
            params_i = params.copy()
            params_j = params.copy()
            
            params_ij[i] += epsilon
            params_ij[j] += epsilon
            
            params_i[i] += epsilon
            params_j[j] += epsilon
            
            f_ij = func(params_ij)
            f_i = func(params_i)
            f_j = func(params_j)
            f_0 = func(params)
            
            hessian_matrix[i, j] = (f_ij - f_i - f_j + f_0) / (epsilon ** 2)
            if i != j:
                hessian_matrix[j, i] = hessian_matrix[i, j]  # Use symmetry
    
    return hessian_matrix


def generate_jeffreys_prior_func(log_like_func):
    """
    Compute the Jeffreys prior using the Hessian of the log-probability function.

    Parameters:
    - log_like_func: The log-probability function.
    - params: The parameters at which to evaluate the prior.
    - epsilon: A small value for numerical differentiation.

    Returns:
    - The Jeffreys prior value.
    """
    def log_prior_func(params):
        hess = - hessian(log_like_func, params)
        sign, val = slogdet(hess)
        if sign <= 0 or np.isnan(val):
            return -np.inf
        return 0.5 * val

    return log_prior_func

# Define an MCMC sampler
def mcmc_sampler(log_like, p_guess, p_std=0.3, 
                nsteps=100, 
                n_samples=1,
                prior_func=None,
                num_Jeffrey=False,
                return_sampler=False):
    '''
    This function samples the noise parameters using MCMC.

    Output:
        a single sample of [f0, fc, alpha]
    '''
    ndim = len(p_guess)

    if prior_func is None:
        if num_Jeffrey: # Numerical Jeffrey prior
            prior_function = generate_jeffreys_prior_func(log_like)
        else:
            prior_function = lambda x: 0
    else:
        prior_function = prior_func

    def log_prob(params):
        return log_like(params) + prior_function(params)

    nwalkers=2*ndim

    # Initialize the sampler
    p0 = np.random.randn(nwalkers, ndim)*p_std + p_guess
    # Run the MCMC sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)


    n_rounds = 5
    for i in range(n_rounds):
        logging.info(f'Running MCMC sampler for the {i+1}th time...')
        sampler.run_mcmc(p0, nsteps, progress=False)

        try: # Estimate the autocorrelation time
            # Catch warnings related to autocorrelation time estimation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=emcee.autocorr.AutocorrError)
                tau = sampler.get_autocorr_time(tol=3, quiet=True)
                # tau = emcee.autocorr.integrated_time(sampler.chain, quiet=True)

            # When using MCMC as a step in a Gibbs sampler, especially if you're only interested in drawing a single sample, 
            # the requirement for the chain length to be significantly longer than the autocorrelation time can be relaxed. 
            # The primary goal is to ensure that the sample you draw is representative of the target distribution.
            # For this purpose, the burn-in period is typically chosen to be 2-3 times the autocorrelation time.

            burnin = int(2.5 * np.max(tau))  # Burn-in time (usually 2-3 times the autocorrelation time)
            thin = int(0.5 * np.min(tau))  # Thinning factor (usually 1/2 of the autocorrelation time)
            thin = max(thin, 1)  # Ensure thinning is at least 1
            logging.info(f'Estimated burn-in: {burnin}')
            logging.info(f'Estimated thinning: {thin}')
            # if burnin > nsteps - n_samples, then continue with the last sample to run more steps
            if burnin < nsteps - n_samples:
                logging.info('Enough steps for burnin..')
                if return_sampler:
                    return sampler
                break
            else:
                logging.warning('Burn-in is greater than nsteps - n_samples, continuing with last sample.')
                p0 = sampler.chain[:, -1, :]
        except Exception as e:
            logging.error(f'Error estimating autocorrelation time: {e}')
            p0 = sampler.chain[:, -1, :]
            if i == n_rounds - 1:
                logging.info('Reached maximum number of iterations, will return last sample.')
                if return_sampler:
                    return sampler
                burnin = nsteps // 3
                thin = 1
                logging.info('Using default burn-in and thinning values.')

    
    # Get the chain after burn-in
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    
    if n_samples == 0:
        # # Return mean of the samples
        # return np.mean(flat_samples, axis=0)
        log_probs = np.array([log_prob(p) for p in flat_samples])
        return flat_samples[np.argmax(log_probs)]
    elif n_samples == 1:
        # Randomly select one sample
        # idx = np.random.randint(len(flat_samples))
        # return flat_samples[idx]
        
        # Select the last sample
        return flat_samples[-1] 
    else:
        # Pick the last n_samples from the chain
        assert len(flat_samples) >= n_samples, "Not enough samples in the chain."
        # Randomly select n_samples
        idx = np.random.randint(len(flat_samples), size=n_samples)
        return flat_samples[idx]


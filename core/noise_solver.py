# This file contains the functions to solve the ``Per Receiver Per Scan'' parameters 
# for the correlated noise using MCMC sampling.

import numpy as np
import numpy.linalg as LA
import emcee
import core.corr_noise_model as corr_noise_model
import util


def log_likelihood(time_width, timestep, time_chunks, freq_chunks, f0, alpha, beta, zeta, data_vector):
    """"
    This function calculates the per-chunk minus log likelihood of the data given the noise model parameters.

    Note that the total minus log likelihood is the sum over all time and frequency chunks, all receivers and all scans.

    """
    n_time_chunks = len(time_chunks)
    n_freq_chunks = len(freq_chunks)

    cov_Time_chunks = []
    cov_Freq_chunks = []

    eigenvalues_Time_chunks = []
    eigenvalues_Freq_chunks = []

    eigenvectors_Time_chunks = []
    eigenvectors_Freq_chunks = []

    for i in range(n_time_chunks):
        cov_Time = corr_noise_model.time_covariance(time_width, timestep, time_chunks[i], f0, alpha)
        eigenvalues_Time, eigenvectors_Time = LA.eigh(cov_Time)
        cov_Time_chunks.append(cov_Time)
        eigenvalues_Time_chunks.append(eigenvalues_Time)
        eigenvectors_Time_chunks.append(eigenvectors_Time)
        
    for i in range(n_freq_chunks):
        cov_Freq = corr_noise_model.freq_covariance(beta, zeta, freq_chunks[i])
        eigenvalues_Freq, eigenvectors_Freq = LA.eigh(cov_Freq)
        cov_Freq_chunks.append(cov_Freq)
        eigenvalues_Freq_chunks.append(eigenvalues_Freq)
        eigenvectors_Freq_chunks.append(eigenvectors_Freq)

    result = 0

    for i in range(n_time_chunks):
        for j in range(n_freq_chunks):
            eigvalues = np.outer(eigenvalues_Time_chunks[i], eigenvalues_Freq_chunks[j]) + 1
            data_proj_squared = util.project_onto_tensor_basis(data_vector, eigenvectors_Time_chunks[i], eigenvectors_Freq_chunks[j]) ** 2
            result += np.sum(np.log(eigvalues) + (1 / eigvalues) * data_proj_squared)

    return -result

def generate_log_prob(time_width, timestep, time_chunks, freq_chunks, residual_data_vector):
    def log_prob(params):
        f0, alpha, beta, zeta = params
        if f0 > 0 and alpha > 0 and zeta > 0:
            return log_likelihood(time_width, timestep, time_chunks, freq_chunks, 
                                  f0, alpha, beta, zeta, residual_data_vector)
        else:
            return -np.inf
    return log_prob

def sampling_coloured_noise_parameter(size, timestep, nu_pivot, freqs, residual_data_vector):
    log_prob_function = generate_log_prob(size, timestep, nu_pivot, freqs, residual_data_vector)
    ndim = 5
    nwalkers = 32
    nsteps = 1000
    initial = np.array([1e-3, 1.0, 1e-12, 1.0, 1.0])

    # Create initial positions for walkers with small random perturbations
    initial_positions = np.tile(initial, (nwalkers, 1)) + 1e-4 * np.random.randn(nwalkers, ndim)
    
    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_function)
    
    # Run the MCMC sampler for the initial burn-in period
    state = sampler.run_mcmc(initial_positions, 100)
    
    # Reset the sampler to discard the initial samples
    sampler.reset()
    
    # Run the MCMC sampler for the main sampling
    sampler.run_mcmc(state, nsteps)

    # Extract the samples
    #samples = sampler.get_chain(flat=True)
    
    return sampler
    
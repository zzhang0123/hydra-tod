import numpy as np
from numpy import linalg as LA
from scipy.fft import rfft, irfft, rfftfreq, fft, fftfreq
import matplotlib.pyplot as plt

def generate_colored_noise_time_covariance(size, timestep, f0, alpha, eps=1e-12):
    epsilon = eps / timestep**2
    freq = fftfreq(size)
    f0_alpha_over_2 = (f0 ** 2 / epsilon) ** (alpha / 2)
    
    result = np.full((size, size), f0_alpha_over_2)
    
    xv, yv = np.meshgrid(np.arange(size), np.arange(size))
    diff_matrix = yv - xv

    for i in range(1, size):
        cos_term = np.cos(diff_matrix * 2 * np.pi * i / size)
        freq_term = (f0 ** 2 / np.sqrt((freq[i] ** 2 + epsilon) * (freq[size - i] ** 2 + epsilon))) ** (alpha / 2)
        result += cos_term * freq_term
    
    return result / size

def calculate_function_values(var1_values, var2_values, your_function):
    """
    Calculate function values for all combinations of two input variable arrays using NumPy.

    Parameters:
    var1_values: NumPy array of the first input variable values
    var2_values: NumPy array of the second input variable values
    your_function: The function to calculate, which takes two NumPy arrays as input.

    Returns:
    A NumPy array containing the function values for all combinations.
    """
    # Create a grid of all combinations of var1 and var2
    var1_grid, var2_grid = np.meshgrid(var1_values, var2_values, indexing='ij')

    # Use your_function to calculate the function values for all combinations
    result = your_function(var1_grid, var2_grid)

    return result

def generate_colored_noise_freq_covariance(nu_pivot, beta, zeta, freqs):
    def frequency_covariance(nu1, nu2):
        return (nu1*nu2/nu_pivot**2)** beta * np.exp( -0.5 * (np.log(nu1/nu2) / zeta)**2)
    calculate_function_values(freqs, freqs, frequency_covariance)

def project_onto_tensor_basis(data_vector, eigvecs_t, eigvecs_f):
    """
    Project a data vector onto the tensor product basis more economically.

    Parameters:
    data_vector (numpy array): The data vector of length Nt * Nf.
    eigvecs_t (numpy array): Nt x Nt orthogonal matrix of eigenvectors.
    eigvecs_f (numpy array): Nf x Nf orthogonal matrix of eigenvectors.

    Returns:
    numpy array: The projection of the data vector onto the tensor basis.
    """
    Nt, _ = eigvecs_t.shape
    Nf, _ = eigvecs_f.shape
    
    # Reshape the data vector to a matrix
    data_matrix = data_vector.reshape(Nt, Nf)
    
    # Project onto the tensor product basis using the orthogonal properties
    proj_t = eigvecs_t.T @ data_matrix
    proj_tf = proj_t @ eigvecs_f
    
    return proj_tf**2

def minus_log_likelihood(size, timestep, f0, alpha, epsilon, nu_pivot, beta, zeta, freqs, data_vector):
    cov_Time = generate_colored_noise_time_covariance(size, timestep, f0, alpha, epsilon)
    cov_Freq = generate_colored_noise_freq_covariance(nu_pivot, beta, zeta, freqs)

    eigenvalues_Time, eigenvectors_Time = LA.eigh(cov_Time)
    eigenvalues_Freq, eigenvectors_Freq = LA.eigh(cov_Freq)

    eigvalues = np.outer(eigenvalues_Time, eigenvalues_Freq) + 1
    data_proj_squared = project_onto_tensor_basis(data_vector, eigenvectors_Time, eigenvectors_Freq)
    result = np.sum(np.log(eigvalues) + (1 / eigvalues) * data_proj_squared) 
    return result
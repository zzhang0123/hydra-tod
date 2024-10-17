import numpy as np
from numpy.fft import rfft, irfft, rfftfreq, fftfreq
from util import calculate_function_values


def generate_colored_noise_1D(mean, psd, size):
    """
    Generate 1D colored noise with the mean value and a given power spectral density (zero DC mode).

    Parameters:
    mean (float): The mean value of the noise.
    psd (ndarray): The power spectral density of the noise.
    size (int): The size of the noise array.

    Returns:
    ndarray: The colored noise array.
    """

    # Generate white noise with normal distribution
    white_noise = np.random.normal(0, 1, size)

    # Compute the FFT of the white noise
    fft_white = rfft(white_noise)

    
    # Apply the filter to the FFT of the white noise
    fft_colored = fft_white * np.sqrt(psd)

    # Compute the inverse FFT to obtain the colored noise
    colored_noise = irfft(fft_colored)

    return colored_noise + mean

def flicker_noise_psd_1D(size, timestep, f0, alpha):
    """
    This function generates the power spectral density of flicker noise.

    Parameters:
    size (int): The size of the noise array.
    timestep (float): The time step of the noise array.
    f0 (float): The characteristic frequency of the flicker noise.
    alpha (float): The power law index of the flicker noise.

    Returns:
    ndarray: The power spectral density of the flicker noise.
    """

    freq = rfftfreq(size, timestep)
    #epsilon = 1e-12 / timestep**2
    result = np.zeros(len(freq))
    result[1:] = (f0 / freq[1:]) ** alpha
    return result

def generate_flicker_noise_1D(mean, size, timestep = 2, f0 = 0.1, alpha = 2.5):
    psd = flicker_noise_psd_1D(size, timestep, f0, alpha)
    return generate_colored_noise_1D(mean, psd, size)

def calculate_power_spectrum_1D(data, timestep):
    size = len(data)
    freq = rfftfreq(size, timestep)
    fft_data = rfft(data)
    return freq, np.abs(fft_data) ** 2

def time_covariance(time_width, timestep, times, f0, alpha):
    """"
    Generate the time covariance matrix for flicker noise.

    Parameters:
    time_width (int): The width of a time chunk.
    timestep (float): The time step of the data.
    times (ndarray): The time coordinates for calculating the time-time covariance.
    f0 (float): The characteristic frequency of the flicker noise.
    alpha (float): The power law index of the flicker noise.
    """
    size = int(time_width / timestep)

    freq = np.abs(fftfreq(size, timestep))
    
    xv, yv = np.meshgrid(times, times)
    diff_matrix = yv - xv

    result = 0
    
    for n in range(1, size):
        cos_term = np.cos(diff_matrix * 2 * np.pi * n / size)
        freq_term = (f0  / freq[n] ) ** alpha 
        result += cos_term * freq_term
    
    return result / size

def freq_covariance(beta, zeta, freqs):
    """
    Generate the frequency covariance matrix for the colored noise.
    """
    nu_pivot = np.mean(freqs)

    def frequency_covariance(nu1, nu2):
        return (nu1*nu2/nu_pivot**2)** beta * np.exp( -0.5 * (np.log(nu1/nu2) / zeta)**2)
    
    return calculate_function_values(freqs, freqs, frequency_covariance)



import numpy as np
from mpmath import gammainc,mp
from scipy.linalg import toeplitz, solve_toeplitz
from numpy.linalg import slogdet
from utils import lag_list
from joblib import Parallel, delayed
import psutil

# Get the current process
process = psutil.Process()

# Check if cpu_affinity() is available (only supported on Linux)
if hasattr(process, 'cpu_affinity'):
    cpu_affinity = process.cpu_affinity()
else:
    # Not supported on this OS
    cpu_affinity = -1

def aux_int_old(mu, u):
    # Exp[- ( Pi / 2 ) I Mu]  Gamma[Mu, I  u]
    aux = gammainc(mu, 1j * u)
    ang = np.pi/2 * mu
    return float(aux.real)*np.cos(ang)+float(aux.imag)*np.sin(ang)

def aux_int(mu, u):
    try:
        aux = gammainc(mu, 1j * u)
        ang = np.pi/2 * mu
        return float(aux.real)*np.cos(ang) + float(aux.imag)*np.sin(ang)
    except ValueError as e:
        print(f"Error in aux_int with mu={mu}, u={u}: {e}")
        return np.inf  # or some other default value

def aux_calculation(u, mu):
    mp.dps = 15
    try:
        aux = gammainc(mu, 1j * u)
        ang = np.pi/2 * mu
        return float(aux.real)*np.cos(ang) + float(aux.imag)*np.sin(ang)
    except ValueError as e:
        print(f"Error in aux_int with mu={mu}, u={u}: {e}")
        return np.inf  # or some other default value


def flicker_corr(tau, f0, fc, alpha, var_w=0.0):
    '''
    Note that f0 and fc are in unit of angular frequency, differently from that of FFT frequency convention by a factor of 2pi.
    '''
    if tau == 0:
        return fc/np.pi * (f0/fc)**alpha  / (alpha-1) + var_w
    tau = np.abs(tau)
    theta_c = fc * tau
    theta_0 = f0 * tau
    norm = 1/(np.pi * tau)
    mu = 1-alpha
    result = theta_0**alpha * aux_int(mu, theta_c) 
    return result*norm


# def flicker_corr_vec(taus, f0, fc, alpha, var_w=0.0):
#     '''
#     Note that f0 and fc are in unit of angular frequency, differently from that of FFT frequency convention by a factor of 2pi.
#     '''
#     theta_c = fc * taus
#     norm = (f0 * taus)**alpha /(np.pi * taus)
#     mu = 1-alpha
#     return np.array([norm[i] * aux_int(mu, theta_c[i]) for i in range(len(taus))])

def flicker_corr_vec(taus, f0, fc, alpha, var_w=0.0, njobs=cpu_affinity):
    '''
    Note that f0 and fc are in unit of angular frequency, differently from that of FFT frequency convention by a factor of 2pi.
    '''
    theta_c = fc * taus
    norm = (f0 * taus)**alpha /(np.pi * taus)
    mu = 1-alpha
    result = np.array(Parallel(n_jobs=njobs)(delayed(aux_calculation)(theta_c[i], mu) for i in range(len(taus))))
    return result * norm

# Define the covariance matrix function
def flicker_cov(time_list, f0, fc, alpha, white_n_variance=5e-6, only_row_0=True):
    lags = lag_list(time_list)
    corr_list = [flicker_corr(t, f0, fc, alpha, var_w=white_n_variance) for t in lags]
    corr_list = np.array(corr_list, dtype=np.float64)
    if only_row_0:
        return corr_list
    return toeplitz(corr_list)

def flicker_cov_vec(tau_list, f0, fc, alpha, white_n_variance=5e-6):
    if tau_list[0] == 0:
        result = np.zeros_like(tau_list)
        result[0] = fc/np.pi * (f0/fc)**alpha  / (alpha-1) + white_n_variance
        result[1:] = flicker_corr_vec(tau_list[1:], f0, fc, alpha, var_w=white_n_variance)
        return result
    else:
        return flicker_corr_vec(tau_list, f0, fc, alpha, var_w=white_n_variance)
    

# This is another flicker noise PSD model with non-vanishing DC mode and its adjacent modes.
def flicker_corr_full(tau, f0, fc, alpha, var_w=0.0):
    if tau == 0:
        return fc/np.pi * (f0/fc)**alpha * alpha / (alpha-1) + var_w
    tau = np.abs(tau)
    theta_c = fc * tau
    theta_0 = f0 * tau
    norm = 1/(np.pi * tau)
    mu = 1 - alpha
    result = theta_0**alpha * aux_int(mu, theta_c) + (f0/fc)**alpha * np.sin(theta_c)
    return result*norm

def sim_noise(f0, fc, alpha, time_list, n_samples=1, white_n_variance=5e-6):
    lags = lag_list(time_list)
    corr_list = [flicker_corr(t, f0, fc, alpha, var_w=white_n_variance) for t in lags]
    covmat = toeplitz(corr_list)
    return np.random.multivariate_normal(np.zeros_like(time_list), covmat, n_samples)
    
class FNoise_traditional:
    def __init__(self, dtime, alpha, fknee=1.0):
        """
        Initialize the FNoise generator.

        Parameters:
        - dtime: Time step for the time series.
        - alpha: Scaling exponent in the frequency power law.
        """
        self.dtime = dtime
        self.alpha = alpha
        self.fknee = fknee

    def generate(self, ntime):
        """
        Generate a realization of 1/f noise in the time domain.

        Parameters:
        - ntime: Number of time steps.

        Returns:
        - Time sequence of 1/f noise.
        """
        # Frequency axis for the rFFT
        freqs = np.fft.fftfreq(ntime, d=self.dtime)
        freqs[0] = np.inf  # Avoid division by zero at f=0
        
        # Define the power spectrum scaling as 1/f^alpha plus white noise
        psd = (self.fknee/np.abs(freqs))**self.alpha #+ 1  # Adding white noise component

        # Generate random white noise in the frequency domain
        white_noise = np.random.normal(size=len(freqs)) + 1j * np.random.normal(size=len(freqs))

        # Weight the white noise by the power spectrum
        weighted_noise = white_noise * np.sqrt(psd) / np.sqrt(2)

        # Transform back to the time domain using irfft
        time_series = np.fft.irfft(weighted_noise, n=ntime)

        # Normalize the time series
        return time_series


'''
aux_int_vectorized = np.frompyfunc(aux_int, 2, 1)

def flicker_corr_vectorized(tau, f0, fc, alpha, var_w=0.0):
    if np.isscalar(tau):
        tau = np.array([tau])
    tau = np.abs(tau)
    theta_c = fc * tau
    theta_0 = f0 * tau
    norm = 1/(np.pi * tau)
    mu = 1-alpha
    result = theta_0**alpha * aux_int_vectorized(mu, theta_c) * norm
    result = np.where(tau == 0, fc/np.pi * (f0/fc)**alpha / (alpha-1) + var_w, result)
    return result
'''
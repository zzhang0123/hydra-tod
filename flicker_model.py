import mpiutil
import numpy as np
from mpmath import gammainc,mp
from scipy.linalg import toeplitz
from utils import lag_list
from joblib import delayed, Parallel
import cmath
from scipy.integrate import quad, IntegrationWarning
import warnings

def my_gamma_inc(z, R_vals, epsabs=1e-6, epsrel=1e-6, vectorize=True): 
    """Calculate the vectorized line integral of the incomplete gamma function.
    Parameters:
        z (complex): Complex (or just real) variable.  
        R (float or array_like): Positive real number or array.
        epsabs (float): Absolute error tolerance.
        epsrel (float): Relative error tolerance.
    Returns:
        complex or ndarray: Complex result (same shape as R).
    """
    R_vals = np.atleast_1d(R_vals)
    
    # function for single input R
    def _integral_single(R_val):
        if R_val <= 0:
            raise ValueError("R must be positive")
        
        def integrand(t):
            s = 1j * R_val + t
            val = cmath.exp(-s) * (s ** (z - 1))
            return val
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=IntegrationWarning)
            result, _ = quad(integrand, 0, np.inf, epsabs=epsabs, epsrel=epsrel, complex_func=True)
        return result
    
    # Vectorize the function
    vfunc = np.vectorize(_integral_single, otypes=[np.complex128])
    return vfunc(R_vals)

    # # Or Parallelize the function:
    # results = np.array(mpiutil.local_parallel_func(_integral_single, R_vals))
    # # Return scalar if input was scalar
    # return results[0] if results.size == 1 else results

def aux_int_v1(mu, u):
    try:
        aux = gammainc(mu, 1j * u)
        ang = np.pi/2 * mu
        return float(aux.real)*np.cos(ang) + float(aux.imag)*np.sin(ang)
    except ValueError as e:
        print(f"Error in aux_int with mu={mu}, u={u}: {e}")
        return np.inf  # or some other default value

def aux_int_v2(mu, u_list):
    # Exp[- ( Pi / 2 ) I Mu]  Gamma[Mu, I  u]
    aux = my_gamma_inc(mu, u_list) # This is actually the same as gammainc(mu, 1j * u)
    cos_ang, sin_ang = np.cos(np.pi/2 * mu), np.sin(np.pi/2 * mu)
    result = aux.real*cos_ang + aux.imag*sin_ang
    return result.astype(np.float64)

def aux_calculation(u, mu):
    try:
        aux = gammainc(mu, 1j * u)
        ang = np.pi/2 * mu
        return float(aux.real)*np.cos(ang) + float(aux.imag)*np.sin(ang)
    except ValueError as e:
        print(f"Error in aux calculation with mu={mu}, u={u}: {e}")
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
    result = theta_0**alpha * aux_int_v1(mu, theta_c) 
    return result*norm

def flicker_corr_vec(taus, f0, fc, alpha):
    '''
    Note that f0 and fc are in unit of angular frequency, differently from that of FFT frequency convention by a factor of 2pi.
    Tau non-zero.
    '''
    theta_c = fc * taus
    norm = (f0 * taus)**alpha /(np.pi * taus)
    mu = 1-alpha
    # result = np.array([aux_calculation(theta_c[i], mu) 
    #                                               for i in range(len(taus))])
    result = aux_int_v2(mu, theta_c)
    return result * norm

def flicker_cov_vec(tau_list, f0, fc, alpha, white_n_variance=2.5e-6):
    assert tau_list[0] == 0, "tau_list[0] must be 0"
    result = np.zeros_like(tau_list)
    result[0] = fc/np.pi * (f0/fc)**alpha  / (alpha-1) + white_n_variance
    result[1:] = flicker_corr_vec(tau_list[1:], f0, fc, alpha)
    return result.astype(np.float64)

    
# Define the covariance matrix function
def flicker_cov(time_list, f0, fc, alpha, white_n_variance=5e-6, only_row_0=False):
    lags = lag_list(time_list)
    corr_list = [flicker_corr(t, f0, fc, alpha, var_w=white_n_variance) for t in lags]
    corr_list = np.array(corr_list, dtype=np.float64)
    # corr_list = flicker_cov_vec(lags, f0, fc, alpha, white_n_variance=white_n_variance)
    if only_row_0:
        return corr_list
    return toeplitz(corr_list)

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



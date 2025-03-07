import numpy as np
from numpy.polynomial.legendre import Legendre
import matplotlib.pyplot as plt


def Leg_poly_proj(ndeg, xs):
    # Generate the projection matrix U such that the columns are the Legendre polynomials evaluated at the rescaled xs.

    x_min = np.min(xs)
    x_max = np.max(xs)

    xs_rescaled = 2*(xs - x_min)/(x_max - x_min) - 1
    proj = np.zeros((len(xs), ndeg))

    for i in range(ndeg):
        proj[:,i] = Legendre.basis(i)(xs_rescaled) 
    
    return proj

class polyn_proj:

    def __init__(self, t_list):
        self.t_list = np.array(t_list)

    def __call__(self, n_deg, func=None):
        if func is None:
            xs = self.t_list
        else:
            xs = func(self.t_list)
        
        return Leg_poly_proj(n_deg, xs)

def DFT_matrix(n):
    # using the default norm
    return np.fft.fft(np.eye(n))

def cov_conjugate(cov, time_to_freq=True):
    n = cov.shape[0]
    DFT_mat = DFT_matrix(n)
    if time_to_freq:
        return DFT_mat @ cov @ DFT_mat.conj().T 
    else: # freq covariance to time covariance
        return DFT_mat.conj().T @ cov @ DFT_mat / n**2

def lag_list(time_list):
    time_list = np.array(time_list)
    return time_list - time_list[0]


def view_samples(p_samples, true_values):
    n_params = p_samples.shape[1]
    mean = np.mean(p_samples, axis=0)
    std = np.std(p_samples, axis=0)

    # Create subplots for four parameters
    # Set figure size according to number of parameters
    fig, axes = plt.subplots(n_params, 1, figsize=(8, 4*n_params))
    axes = axes.ravel()

    for i in range(n_params):
        # Plot histogram of samples for each parameter
        axes[i].hist(p_samples[:, i], bins=50, density=True, alpha=0.6, label='Samples')
        
        # Plot true value line
        axes[i].axvline(x=true_values[i], color='r', linestyle='-', label='True Value', linewidth=2, alpha=0.7)
        
        # Plot mean value line
        axes[i].axvline(x=mean[i], color='g', linestyle='--', label='Mean')
        
        # Add labels and title
        axes[i].set_xlabel('Coefficient')
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'Parameter {i+1}')
        axes[i].legend()
        
        # Print numerical comparison for each parameter
        print(f"\n Parameter {i+1}:")
        print(f"True value: {true_values[i]:.6f}")
        print(f"Mean sampled: {mean[i]:.6f}")
        print(f"Standard deviation: {std[i]:.6f}")
        print(f"Relative error: {abs(mean[i] - true_values[i])/true_values[i]*100:.2f}%")

    plt.tight_layout()
    plt.show()


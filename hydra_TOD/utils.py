import numpy as np
from numpy.polynomial.legendre import Legendre
import matplotlib.pyplot as plt


import scipy.linalg


"""
Example usage of the cholesky decomposition based functions.
A = np.random.rand(100,100)
A = A@A.mT

L_inv_sqrt = cho_compute_mat_inv_sqrt(A)
N_inv = cho_compute_mat_inv(A)

np.allclose(L_inv_sqrt@L_inv_sqrt.T, N_inv)
"""

def cho_compute_mat_inv(Ncov):
    try:
        L = np.linalg.cholesky(Ncov)
        Ncov_inv = scipy.linalg.cho_solve((L, True), np.eye(Ncov.shape[0]))
        return Ncov_inv
    except np.linalg.LinAlgError:
        Ncov_reg = Ncov + 1e-5 * np.eye(Ncov.shape[0])
        L = np.linalg.cholesky(Ncov_reg)
        return scipy.linalg.cho_solve((L, True), np.eye(Ncov.shape[0]))

def cho_compute_mat_inv_sqrt(Ncov):
    try:
        L = np.linalg.cholesky(Ncov)
        # Compute the inverse of L.T
        L_inv_T = scipy.linalg.solve_triangular(L, np.eye(Ncov.shape[0]), trans='T', lower=True)
        return L_inv_T 
    except np.linalg.LinAlgError:
        Ncov_reg = Ncov + 1e-5 * np.eye(Ncov.shape[0])
        L = np.linalg.cholesky(Ncov_reg)
        return scipy.linalg.solve_triangular(L, np.eye(Ncov.shape[0]), trans='T', lower=True)

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

def pixel_angular_size(nside):
    """Compute the angular size (in degrees and arcminutes) of a HEALPix pixel."""
    npix = hp.nside2npix(nside)  # Total number of pixels
    omega_pix = 4 * np.pi / npix  # Pixel area in steradians
    theta_pix_deg = np.sqrt(omega_pix) * (180 / np.pi)  # Approximate pixel width in degrees
    theta_pix_arcmin = theta_pix_deg * 60  # Convert to arcminutes
    return theta_pix_deg, theta_pix_arcmin

def overall_operator(operator_list):
    '''
    This function calculates the overall operator.
    Parameters:
    ----------
    operator_list : list of ndarray
        List of projection matrices. For example, [beam_proj, rec_proj, ndiode_proj].

    Returns:
    -------
    overall_operator : ndarray
        Overall operator.
    '''
    aux_list = []
    for proj in operator_list:
        assert proj.shape[0] == operator_list[0].shape[0], "All projection matrices must have the same length.."
        if len(proj.shape) == 1:
            proj = proj.reshape(-1, 1)
        aux_list.append(proj)
    return np.hstack(aux_list)


def linear_model(operator_list, params_vec_list):
    '''
    This function calculates the system temperature.

    Parameters:
    ----------
    operator_list : list of ndarray
        List of projection matrices. For example, [beam_proj, rec_proj, ndiode_proj].
    params_vec_list : list of ndarray
        List of parameter vectors. For example, [true_Tsky, rec_params, T_ndiode].

    Returns:
    -------
    Tsys : ndarray
        System temperature.
    '''
    assert len(operator_list) == len(params_vec_list), "Operator list and params list must have the same length.."
    n_components = len(operator_list)
    n_data = operator_list[0].shape[0]
    Tsys = np.zeros(n_data)

    for i in range(n_components):
        # if params_vec_list[i] is a scalar:
        if len(operator_list[i].shape) == 1:
            Tsys += operator_list[i] * params_vec_list[i]
        else:
            Tsys += operator_list[i] @ params_vec_list[i]

    return Tsys
from linear_solver import cg
from utils import cho_compute_mat_inv
from flicker_model import flicker_cov
from linear_sampler import sample_p_old, iterative_gls

def gain_coeff_sampler(data, 
                       t_list,
                       gain_proj, 
                       Tsys, 
                       noise_params,
                       wnoise_var=2.5e-6, 
                       mu=0.0,
                       n_samples=1,
                       tol=1e-13,
                       prior_cov_inv=None, 
                       prior_mean=None, 
                       solver=cg):
    d_vec = data/Tsys
    logf0, logfc, alpha = noise_params
    Ncov_inv = cho_compute_mat_inv( flicker_cov(t_list, 10.**logf0, 10.**logfc, alpha,  white_n_variance=wnoise_var, only_row_0=False) )
    p_GLS, sigma_inv = iterative_gls(d_vec, gain_proj, Ncov_inv, mu=mu, tol=tol)
    return sample_p_old(d_vec, gain_proj, sigma_inv, num_samples=n_samples, mu=mu, prior_cov_inv=prior_cov_inv, prior_mean=prior_mean, solver=solver)



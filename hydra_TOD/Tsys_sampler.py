import numpy as np
import mpiutil
from flicker_model import flicker_cov
from linear_solver import cg, pytorch_lin_solver
from linear_sampler import iterative_gls, iterative_gls_mpi_list, sample_p, sample_p_old, sample_p_v2
from utils import cho_compute_mat_inv, cho_compute_mat_inv_sqrt

comm = mpiutil.world
rank0 = mpiutil.rank0

def Tsys_coeff_sampler(data, 
                       t_list,
                       gain, 
                       Tsys_proj, 
                       noise_params, 
                       logfc=None,
                       wnoise_var=2.5e-6,
                       n_samples=1,
                       mu=0.0,
                       tol=1e-13,
                       prior_cov_inv=None, 
                       prior_mean=None, 
                       solver=cg):

    d_vec = data/gain
    if logfc is None:
        logf0, logfc, alpha = noise_params
    else:
        logf0, alpha = noise_params
    Ncov_inv = cho_compute_mat_inv( flicker_cov(t_list, 10.**logf0, 10.**logfc, alpha,  white_n_variance=wnoise_var, only_row_0=False) )

    p_GLS, sigma_inv = iterative_gls(d_vec, Tsys_proj, Ncov_inv, mu=mu, tol=tol)
    return sample_p_old(d_vec, Tsys_proj, sigma_inv, num_samples=n_samples,  mu=mu, prior_cov_inv=prior_cov_inv, prior_mean=prior_mean, solver=solver)

def Tsys_sampler_multi_TODs(local_data_list,
                            local_t_list,
                            local_gain_list,
                            local_Tsys_proj_list,
                            local_Noise_params_list,
                            local_logfc_list,
                            local_mu_list=None,
                            wnoise_var=2.5e-6,
                            tol=1e-13,
                            prior_cov_inv=None,
                            prior_mean=None,
                            solver=cg,
                            init_coeffs=None,
                            Est_mode=False):
    dim = len(local_data_list)
    d_vec_list = [local_data_list[i]/local_gain_list[i] for i in range(dim)]

    local_Ninv_sqrt_list = []
    for di in range(dim):
        logfc = local_logfc_list[di]
        logf0, alpha = local_Noise_params_list[di]
        t_list = local_t_list[di]
        Ncov_inv_sqrt = cho_compute_mat_inv_sqrt( flicker_cov(t_list, 10.**logf0, 10.**logfc, alpha,  white_n_variance=wnoise_var, only_row_0=False) )
        local_Ninv_sqrt_list.append(Ncov_inv_sqrt)

    p_GLS, A, b, Asqrt_wn =  iterative_gls_mpi_list(d_vec_list, local_Tsys_proj_list, local_Ninv_sqrt_list, local_mu_list=local_mu_list, tol=tol, p_init=init_coeffs)

    # Compute on rank 0 only to avoid redundant computation
    if mpiutil.rank0:
        p_sample = sample_p_v2(A, b, Asqrt_wn, prior_cov_inv=prior_cov_inv, prior_mean=prior_mean, solver=solver, Est_mode=Est_mode)
    else:
        p_sample = None

    # broadcast result to all ranks
    p_sample = comm.bcast(p_sample, root=0)
    return p_sample






        
    

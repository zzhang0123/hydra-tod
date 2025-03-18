import numpy as np
from linear_solver import cg, cg_mpi
from scipy.linalg import solve
from linear_sampler import sample_p, iterative_gls

def gain_coeff_sampler(data, 
                       gain_proj, 
                       Tsys, 
                       Ncov, 
                       mu=0.0,
                       n_samples=1,
                       tol=1e-13,
                       prior_cov_inv=None, 
                       prior_mean=None, 
                       solver=None):

    d_vec = data/Tsys
    Ncov_inv = np.linalg.inv(Ncov)
    p_GLS, sigma_inv = iterative_gls(d_vec, gain_proj, Ncov_inv, mu=mu, tol=tol)
    return sample_p(d_vec, gain_proj, sigma_inv, num_samples=n_samples, mu=mu, prior_cov_inv=prior_cov_inv, prior_mean=prior_mean, solver=solver, p_GLS=p_GLS)



# def gain_coeff_sampler(data, inv_gain_proj, 
#                        Ninv, Tsys, 
#                        prior_cov_inv=None, prior_mean=None, solver=None):

#     proj = np.einsum('i,ij->ij', data/Tsys, inv_gain_proj) # Compute the projection matrix - real matrix.

#     Nfreqs, Nmodes = proj.shape

#     # Draw unit Gaussian random numbers
#     omega_n = np.random.randn(Nfreqs)

#     lhs_op =  Ninv @ proj
#     lhs_op = proj.T @ lhs_op 

#     #rhs = proj.T @ ( Ninv @ data + sqrtm(Ninv) @ omega_n ) 
#     rhs = sqrtm(Ninv) @ (1 + omega_n)
#     rhs = proj.T @ rhs  

#     if prior_cov_inv is not None:
#         assert prior_mean is not None, "Prior mean must be provided if prior covariance is provided.."
#         lhs_op += prior_cov_inv
#         prior_cov_half_inv = sqrtm(prior_cov_inv)
#         omega_g = np.random.randn(Nmodes)
#         rhs += prior_cov_inv @ prior_mean + prior_cov_half_inv @ omega_g
    
#     if solver is not None:
#         return solver(lhs_op, rhs) # Solve the linear system
#     else:
#         return solve(lhs_op, rhs, assume_a='sym') # Solve the linear system
        

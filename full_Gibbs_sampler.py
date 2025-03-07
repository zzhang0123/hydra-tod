# This file contains the full Gibbs sampler for all the parameters in data model, 
# including the system temperature parameters, noise parameters, gain parameters.

# Full Gibbs sampler

from gain_sampler import gain_coeff_sampler
from noise_sampler import flicker_noise_sampler
from flicker_model import flicker_cov
from Tsys_sampler import Tsys_coeff_sampler

def full_Gibbs_sampler_singledish(TOD, 
                                  t_list,
                                  TOD_diode,
                                  Tsys_operator,
                                  gain_operator,
                                  init_Tsys_params, 
                                  init_noise_params, 
                                  gain_mu0=0.0,
                                  wnoise_var=2.5e-6,
                                  Tsys_prior_cov_inv=None,
                                  Tsys_prior_mean=None,
                                  gain_prior_cov_inv=None,
                                  gain_prior_mean=None,
                                  noise_prior_func=None,
                                  n_samples=100,
                                  tol=1e-15,
                                  linear_solver=None,):   

    p_gain_samples = []
    p_sys_samples = [] 
    p_noise_samples = []

    # Initialize parameters
    Tsys = Tsys_operator@init_Tsys_params + TOD_diode
    noise_params = init_noise_params
    logf0, logfc, alpha = noise_params
    Ncov = flicker_cov(t_list, 10.**logf0, 10.**logfc, alpha, white_n_variance=wnoise_var, only_row_0=False)
    
    for i in range(n_samples):
        # Given Tsys and noise parameters, sample gain parameters
        gain_sample = gain_coeff_sampler(TOD, gain_operator, Tsys, Ncov, mu=gain_mu0, n_samples=1, tol=tol, prior_cov_inv=gain_prior_cov_inv, prior_mean=gain_prior_mean, solver=linear_solver)[0]
        p_gain_samples.append(gain_sample)
        gains = gain_operator@gain_sample + gain_mu0

        # Given Tsys and gain parameters, sample noise parameters
        noise_params = flicker_noise_sampler(TOD,
                                             t_list,
                                             gains,
                                             Tsys,
                                             #noise_params, # using the previous noise_params as initial point
                                             init_noise_params, # using the input init_noise_params as initial point
                                             n_samples=1,
                                             wnoise_var=wnoise_var,
                                             prior_func=noise_prior_func,
                                             num_Jeffrey=False,
                                             boundaries=None,)

        p_noise_samples.append(noise_params)
        logf0, logfc, alpha = noise_params
        Ncov = flicker_cov(t_list, 10.**logf0, 10.**logfc, alpha, white_n_variance=wnoise_var, only_row_0=False)

        # Given gain and noise parameters, sample Tsys parameters
        Tsys_params = Tsys_coeff_sampler(TOD, gains, Tsys_operator, Ncov, n_samples=1, mu=TOD_diode, tol=tol, prior_cov_inv=Tsys_prior_cov_inv, prior_mean=Tsys_prior_mean, solver=linear_solver)[0]
        p_sys_samples.append(Tsys_params)
        Tsys = Tsys_operator@Tsys_params + TOD_diode

    return p_gain_samples, p_sys_samples, p_noise_samples


        


    

    
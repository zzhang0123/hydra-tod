# This file contains the full Gibbs sampler for all the parameters in data model, 
# including the system temperature parameters, noise parameters, gain parameters.

# Full Gibbs sampler

from gain_sampler import gain_coeff_sampler
from noise_sampler import flicker_noise_sampler
from flicker_model import flicker_cov
from Tsys_sampler import Tsys_coeff_sampler
import mpiutil

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
rank0 = rank == 0
        

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



def full_Gibbs_sampler_multi_TODS(local_TOD_list, 
                                  local_t_lists,
                                  local_TOD_diode_list,
                                  local_gain_operator_list,
                                  local_Tsky_operator_list,
                                  local_Trec_operator_list,
                                  init_Tsky_params,
                                  init_Trec_params, 
                                  init_noise_params, 
                                  wnoise_var=2.5e-6,
                                  Tsky_prior_cov_inv=None,
                                  Tsky_prior_mean=None,
                                  Trec_prior_cov_inv=None,
                                  Trec_prior_mean=None,
                                  gain_prior_cov_inv=None,
                                  gain_prior_mean=None,
                                  noise_prior_func=None,
                                  n_samples=100,
                                  tol=1e-15,
                                  linear_solver=None,
                                  root=None,):   
    """
    This function is used to sample the data model parameters using a list of TOD chunks, say multi-receiver and/or multi-scan data.
    We use mpi to parallelize the computation.

    Note that gain parameters and noise parameters are defined per data vector, 
    while system temperature parameters are shared for the whole TOD list.
    Thus, we sample the gain parameters and the noise parameters for each data vector in the list, 
    while we sample the system temperature parameters for the whole TOD list.
    """

    # Synchronize the processes
    mpiutl.barrier()

    # Check the length of the input lists
    num_TODs = len(local_TOD_list)

    if num_TODs != len(local_t_list) \
        or num_TODs != len(local_Tsky_operator_list) \
            or num_TODs != len(local_Trec_operator_list) \
                or num_TODs != len(local_gain_operator_list) \
                    or num_TODs != len(local_TOD_diode_list) :
        raise ValueError("The length of the input lists must be the same.")
    
    n_g_modes = local_gain_operator_list[0].shape[1]
    local_gain_samples = np.zeros((num_TODs, n_samples, n_g_modes))

    n_rec_modes = local_Trec_operator_list[0].shape[1]
    local_Trec_samples = np.zeros((num_TODs, n_samples, n_rec_modes))

    n_N_modes = len(init_noise_params_list[0])
    local_noise_samples = np.zeros((num_TODs, n_samples, n_N_modes))

    Tsky_samples = [] 

    # Initialize noise, Tsky, and Trec parameters
    local_Ncov_list = []
    logf0, logfc, alpha = init_noise_params

    for di in range(num_TODs):
        t_list = local_t_lists[di]
        Ncov = flicker_cov(t_list, 10.**logf0, 10.**logfc, alpha, white_n_variance=wnoise_var, only_row_0=False)
        local_Ncov_list.append(Ncov)

    Tsky_params=init_Tsky_params
    Trec_params=init_Trec_params

    # Sample the parameters
    
    for si in range(n_samples):
        local_gain_list = []
        local_Tsky_mu_list = []
        for di in range(num_TODs):
            TOD = local_TOD_list[di]
            t_list = local_t_lists[di]
            gain_operator = local_gain_operator_list[di]
            Tsky_operator = local_Tsky_operator_list[di]
            TOD_sky = Tsys_operator@Tsky_params
            Trec_operator = local_Trec_operator_list[di]
            TOD_rec = Trec_operator@Trec_params
            TOD_diode = local_TOD_diode_list[di]

            Tsys = TOD_sky + TOD_rec + TOD_diode
            
            # Sample gain parameters
            Ncov =  local_Ncov_list[di]
            gain_sample = gain_coeff_sampler(TOD, gain_operator, Tsys, Ncov, 
                                            n_samples=1, tol=tol, prior_cov_inv=gain_prior_cov_inv, 
                                            prior_mean=gain_prior_mean, solver=linear_solver)[0]
            local_gain_samples[di, si, :] = gain_sample
            gains = gain_operator@gain_sample
            local_gain_list.append(gains)

            # Sample receiver temperature parameters
            Trec_sample = Tsys_coeff_sampler(TOD, gains, Trec_operator, Ncov, 
                                            n_samples=1, mu=TOD_sky + TOD_diode,
                                            tol=1e-16, prior_cov_inv=Trec_prior_cov_inv, 
                                            prior_mean=Trec_prior_mean, solver=linear_solver)[0]
            local_Trec_samples[di, si, :] = Trec_sample


            # Sample noise parameters
            noise_params = flicker_noise_sampler(TOD,
                                                t_list,
                                                gains,
                                                Tsys,
                                                init_noise_params, # using the input init_noise_params as fixed initial point for MCMC sampling
                                                n_samples=1,
                                                wnoise_var=wnoise_var,
                                                prior_func=noise_prior_func)
            local_noise_samples[di, si, :] = noise_params

            # Update Ncov
            logf0, logfc, alpha = noise_params
            local_Ncov_list[di] = flicker_cov(t_list, 10.**logf0, 10.**logfc, alpha, white_n_variance=wnoise_var, only_row_0=False)

            # Collect mu vectors for Tsky sampler
            local_Tsky_mu_list.append(TOD_rec + TOD_diode)

        # Given gain and noise parameters, sample Tsys parameters

        Tsky_params = Tsys_coeff_sampler_multi_TODs(local_TOD_list,
                                                    local_gain_list,
                                                    local_Tsys_operator_list,
                                                    local_Ncov_list,
                                                    n_samples=1,
                                                    local_mu_list=local_Tsky_mu_list,
                                                    tol=tol,
                                                    prior_cov_inv=Tsys_prior_cov_inv,
                                                    prior_mean=Tsys_prior_mean,
                                                    solver=linear_solver)
        
        Tsky_samples.append(Tsky_params)


    # Gather local gain and noise samples
    mpiutil.barrier()

    all_gain_samples = None
    all_noise_samples = None
    all_rec_samples = None
    if root is None:
        # Gather all results onto all ranks
        all_gain_samples = comm.allgather(local_gain_samples)
        all_noise_samples = comm.allgather(local_noise_samples)
        all_rec_samples = comm.allgather(local_rec_samples)
    else:
        # Gather all results onto the specified rank
        all_gain_samples = comm.gather(local_gain_samples, root=root)
        all_noise_samples = comm.gather(local_noise_samples, root=root)
        all_rec_samples = comm.gather(local_rec_samples, root=root)

    return Tsky_samples, all_gain_samples, all_noise_samples, all_rec_samples


      

    

    
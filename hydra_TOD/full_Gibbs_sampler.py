# This file contains the full Gibbs sampler for all the parameters in data model, 
# including the system temperature parameters, noise parameters, gain parameters.

# Full Gibbs sampler
import numpy as np
import mpiutil
from gain_sampler import gain_coeff_sampler
from noise_sampler import flicker_noise_sampler
from flicker_model import flicker_cov
from Tsys_sampler import Tsys_coeff_sampler, Tsky_coeff_sampler_multi_TODs, Tsys_sampler_multi_TODs
from linear_solver import cg
from mpi4py import MPI
from scipy.linalg import block_diag


comm = mpiutil.world
rank = mpiutil.rank
rank0 = rank == 0
        

def full_Gibbs_sampler_single_TOD(TOD, 
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
                                  linear_solver=cg,
                                  Est_mode=False):   

    p_gain_samples = []
    p_sys_samples = [] 
    p_noise_samples = []

    # Initialize parameters
    Tsys = Tsys_operator@init_Tsys_params + TOD_diode
    noise_params = init_noise_params
    
    if Est_mode:
        num_sample = 0 
    else:
        num_sample = 1

    for i in range(n_samples):
        # Given Tsys and noise parameters, sample gain parameters
        
        gain_sample = gain_coeff_sampler(TOD, t_list, gain_operator, Tsys, noise_params, 
                                            wnoise_var=wnoise_var,
                                            mu=gain_mu0, n_samples=num_sample, tol=tol, 
                                            prior_cov_inv=gain_prior_cov_inv, 
                                            prior_mean=gain_prior_mean, 
                                            solver=linear_solver)
        p_gain_samples.append(gain_sample)
        gains = gain_operator@gain_sample + gain_mu0

        mpiutil.barrier()
        # Given Tsys and gain parameters, sample noise parameters
        noise_params = flicker_noise_sampler(TOD,
                                             t_list,
                                             gains,
                                             Tsys,
                                             #noise_params, # using the previous noise_params as initial point
                                             init_noise_params, # using the input init_noise_params as initial point
                                             n_samples=num_sample,
                                             wnoise_var=wnoise_var,
                                             prior_func=noise_prior_func,
                                             num_Jeffrey=False,
                                             boundaries=None,)

        p_noise_samples.append(noise_params)
        # Given gain and noise parameters, sample Tsys parameters
        Tsys_params = Tsys_coeff_sampler(TOD, t_list, gains, Tsys_operator, noise_params, wnoise_var=wnoise_var, 
                                        n_samples=num_sample, mu=TOD_diode, tol=tol, 
                                        prior_cov_inv=Tsys_prior_cov_inv, prior_mean=Tsys_prior_mean, solver=linear_solver)
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
                                  tol=1e-12,
                                  linear_solver=cg,
                                  Est_mode=False,
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
    mpiutil.barrier()

    # Check the length of the input lists
    num_TODs = len(local_TOD_list)

    if num_TODs != len(local_t_lists) \
        or num_TODs != len(local_Tsky_operator_list) \
            or num_TODs != len(local_Trec_operator_list) \
                or num_TODs != len(local_gain_operator_list) \
                    or num_TODs != len(local_TOD_diode_list) :
        raise ValueError("The length of the input lists must be the same.")
    
    n_g_modes = local_gain_operator_list[0].shape[1]
    local_gain_samples = np.zeros((num_TODs, n_samples, n_g_modes))

    n_rec_modes = local_Trec_operator_list[0].shape[1]
    local_Trec_samples = np.zeros((num_TODs, n_samples, n_rec_modes))

    n_sky_modes = len(init_Tsky_params)
    Tsky_samples = np.zeros((n_samples, n_sky_modes))

    n_N_modes = len(init_noise_params)
    local_noise_samples = np.zeros((num_TODs, n_samples, n_N_modes))

    # Initialize noise, Tsky, and Trec parameters
    #logf0, logfc, alpha = init_noise_params

    Tsky_params=init_Tsky_params
    for di in range(num_TODs):
        t_list = local_t_lists[di]
        local_noise_samples[di, -1, :] = init_noise_params
        local_Trec_samples[di, -1, :] = init_Trec_params
        # Ncov = flicker_cov(t_list, 10.**logf0, 10.**logfc, alpha, white_n_variance=wnoise_var, only_row_0=False)
        # local_Ncov_list.append(Ncov)

    
    local_gain_list = [np.zeros_like(local_TOD_list[di]) for di in range(num_TODs)]
    local_Tsky_mu_list = [np.zeros_like(local_TOD_list[di]) for di in range(num_TODs)]

    if Est_mode:
        num_sample = 0 
    else:
        num_sample = 1

    # Sample the parameters
    for si in range(n_samples):
        for di in range(num_TODs):
            TOD = local_TOD_list[di]
            t_list = local_t_lists[di]
            gain_operator = local_gain_operator_list[di]
            Tsky_operator = local_Tsky_operator_list[di]
            Trec_operator = local_Trec_operator_list[di]
            TOD_diode = local_TOD_diode_list[di]

            noise_params = local_noise_samples[di, si-1, :] 
            Trec_params = local_Trec_samples[di, si-1, :]

            TOD_sky = Tsky_operator@Tsky_params
            TOD_rec = Trec_operator@Trec_params
            Tsys = TOD_sky + TOD_rec + TOD_diode
            
            
            # Sample gain parameters

            gain_sample = gain_coeff_sampler(TOD, t_list, gain_operator, Tsys, noise_params, 
                                            wnoise_var=wnoise_var,
                                            n_samples=num_sample, 
                                            tol=tol, 
                                            prior_cov_inv=gain_prior_cov_inv, 
                                            prior_mean=gain_prior_mean, 
                                            solver=linear_solver)
            local_gain_samples[di, si, :] = gain_sample
            gains = gain_operator@gain_sample
            local_gain_list[di]=gains
            print("Rank: {}, local id: {}, gain_sample {}: {}".format(rank, di, si, gain_sample))

            # Sample receiver temperature parameters
            Trec_params = Tsys_coeff_sampler(TOD, t_list, gains, Trec_operator, noise_params,
                                            wnoise_var=wnoise_var, 
                                            n_samples=num_sample, 
                                            mu=TOD_sky + TOD_diode,
                                            tol=tol, 
                                            prior_cov_inv=Trec_prior_cov_inv, 
                                            prior_mean=Trec_prior_mean, 
                                            solver=linear_solver)
            local_Trec_samples[di, si, :] = Trec_params
            TOD_rec = Trec_operator@Trec_params
            Tsys = TOD_sky + TOD_rec + TOD_diode
            # Collect mu vectors for Tsky sampler
            local_Tsky_mu_list[di]=TOD_rec + TOD_diode
            print("Rank: {}, local id: {}, Trec_sample {}: {}".format(rank, di, si, Trec_params))

            # Sample noise parameters
            noise_sample = flicker_noise_sampler(TOD,
                                                t_list,
                                                gains,
                                                Tsys,
                                                init_noise_params, # using the input init_noise_params as fixed initial point for MCMC sampling
                                                n_samples=num_sample,
                                                wnoise_var=wnoise_var,
                                                prior_func=noise_prior_func)
            local_noise_samples[di, si, :] = noise_sample
            print("Rank: {}, local id: {}, noise_sample {}: {}".format(rank, di, si, noise_sample))


        # Given gain, noise and other Tsys components, sample Tsky parameters
        Tsky_params = Tsky_coeff_sampler_multi_TODs(local_TOD_list,
                                                    local_t_lists,
                                                    local_gain_list,
                                                    local_Tsky_operator_list,
                                                    local_noise_samples[:, si, :],
                                                    local_Tsky_mu_list,
                                                    wnoise_var=wnoise_var,
                                                    tol=tol,
                                                    prior_cov_inv=Tsky_prior_cov_inv,
                                                    prior_mean=Tsky_prior_mean,
                                                    solver=linear_solver,
                                                    Est_mode=Est_mode)
        
        Tsky_samples[si, :] = Tsky_params



    # Gather local gain and noise samples
    mpiutil.barrier()

    all_gain_samples = None
    all_noise_samples = None
    all_Trec_samples = None
    if root is None:
        # Gather all results onto all ranks
        all_gain_samples = comm.allgather(local_gain_samples)
        all_noise_samples = comm.allgather(local_noise_samples)
        all_Trec_samples = comm.allgather(local_Trec_samples)
    else:
        # Gather all results onto the specified rank
        all_gain_samples = comm.gather(local_gain_samples, root=root)
        all_noise_samples = comm.gather(local_noise_samples, root=root)
        all_Trec_samples = comm.gather(local_Trec_samples, root=root)

    return Tsky_samples, all_gain_samples, all_noise_samples, all_Trec_samples



def get_Tsys_operator(local_Tsky_operator_list, local_Trec_operator_list):
    # Generate Tsys operator from Tsky and Trec operators
    # For example, we illustrate this in a block matrix form:
    # say we have two ranks,
    # rank0 has local Tsky operators U1, U2, and local Trec operators R1, R2, 
    # rank1 has local Tsky operators U3, and local Trec operators R3,
    # then the overall Tsys operator U is:
    #        ( U1 R1  0  0 )
    # U =    ( U2  0 R2  0 )
    #        ( U3  0  0  R3)
    # It will be again saved as local lists: rank0 has [( U1 R1  0  0 ), (U2  0 R2  0 )],
    # and rank1 has [( U3  0  0  R3)].

    # The linear system reads
    # ( U1 R1  0  0 )       (Tsky)       (n1)       (d1)
    # ( U2  0 R2  0 )   @   (Trec1)   +  (n2)   =   (d2)
    # ( U3  0  0  R3)       (Trec2)      (n3)       (d3)
    #                       (Trec3)
    """
    Construct combined Tsys operator matrix from Tsky and Trec operators.
    
    Args:
        local_Tsky_operator_list: List of Tsky operators for each TOD
        local_Trec_operator_list: List of Trec operators for each TOD
        
    Returns:
        List of combined Tsys operators for each TOD
    """
    num_TODs = len(local_Tsky_operator_list)

    local_rec_dim = [op.shape[1] for op in local_Trec_operator_list]
    local_total_rec_dim = sum(local_rec_dim)
    glist_total_rec_dims = comm.allgather(local_total_rec_dim)
    global_total_rec_dim = sum(glist_total_rec_dims)
    # cumulative sum of glist_total_rec_dims
    rank_offset_list = [0] + np.cumsum(glist_total_rec_dims).tolist()
    local_rank_offset = rank_offset_list[rank]
    local_rec_dim = [0] + local_rec_dim

    local_Tsys_operator_list = []
    for di in range(num_TODs):
        dim_data = local_Trec_operator_list[di].shape[0]
        dim_params = local_Trec_operator_list[di].shape[1]
        Trec_operator = np.zeros((dim_data, global_total_rec_dim))
        start_ind = local_rank_offset + local_rec_dim[di]
        end_ind = start_ind + dim_params
        Trec_operator[:, start_ind:end_ind] = local_Trec_operator_list[di]
        Tsys_operator = np.hstack((local_Tsky_operator_list[di], Trec_operator))
        local_Tsys_operator_list.append(Tsys_operator)
    
    return local_Tsys_operator_list
      


def full_Gibbs_sampler_multi_TODS_v2(local_TOD_list, 
                                    local_t_lists,
                                    # local_TOD_diode_list,
                                    local_gain_operator_list,
                                    local_Tsky_operator_list,
                                    local_Trec_operator_list,
                                    init_Tsys_params,
                                    init_noise_params, 
                                    wnoise_var=2.5e-6,
                                    Tsky_prior_cov_inv=None,
                                    Tsky_prior_mean=None,
                                    local_Trec_prior_cov_inv_list=None,
                                    local_Trec_prior_mean_list=None,
                                    local_gain_prior_cov_inv_list=None,
                                    local_gain_prior_mean_list=None,
                                    local_noise_prior_func_list=None,
                                    n_samples=100,
                                    tol=1e-12,
                                    linear_solver=cg,
                                    Est_mode=False,
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
    mpiutil.barrier()

    Trec_prior_cov_inv_list = [item for sublist in comm.allgather(local_Trec_prior_cov_inv_list) 
                                for item in sublist]
    Trec_prior_mean_list = [item for sublist in comm.allgather(local_Trec_prior_mean_list)
                            for item in sublist]

    if Tsky_prior_cov_inv.ndim==1: 
        # The prior covariance matrices are diagonal, the inputs are the diagonal elements
        Tsys_prior_cov_inv = np.hstack([Tsky_prior_cov_inv]+Trec_prior_cov_inv_list)
    else:
        Tsys_prior_cov_inv = block_diag(Tsky_prior_cov_inv, *Trec_prior_cov_inv_list)
    Tsys_prior_mean = np.hstack([Tsky_prior_mean]+Trec_prior_mean_list)

    # Check the length of the input lists
    num_TODs = len(local_TOD_list)

    if num_TODs != len(local_t_lists) \
        or num_TODs != len(local_Tsky_operator_list) \
            or num_TODs != len(local_Trec_operator_list) \
                or num_TODs != len(local_gain_operator_list):
        raise ValueError("The length of the input lists must be the same.")
    
    n_g_modes = local_gain_operator_list[0].shape[1]
    local_gain_samples = np.zeros((num_TODs, n_samples, n_g_modes))

    local_Tsys_operator_list = get_Tsys_operator(local_Tsky_operator_list, local_Trec_operator_list)

    n_sys_modes = local_Tsys_operator_list[0].shape[1]
    Tsys_samples = np.zeros((n_samples, n_sys_modes))

    n_N_modes = len(init_noise_params)
    local_noise_samples = np.zeros((num_TODs, n_samples, n_N_modes))

    # Initialize noise, Tsky, and Trec parameters
    #logf0, logfc, alpha = init_noise_params

    Tsys_samples[-1, :len(init_Tsys_params)] = init_Tsys_params

    for di in range(num_TODs):
        t_list = local_t_lists[di]
        local_noise_samples[di, -1, :] = init_noise_params
        # Ncov = flicker_cov(t_list, 10.**logf0, 10.**logfc, alpha, white_n_variance=wnoise_var, only_row_0=False)
        # local_Ncov_list.append(Ncov)

    
    local_gain_list = [np.zeros_like(local_TOD_list[di]) for di in range(num_TODs)]

    if Est_mode:
        num_sample = 0 
    else:
        num_sample = 1

    # Sample the parameters
    for si in range(n_samples):
        for di in range(num_TODs):
            TOD = local_TOD_list[di]
            t_list = local_t_lists[di]
            gain_operator = local_gain_operator_list[di]
            Tsys_operator = local_Tsys_operator_list[di]
            # TOD_diode = local_TOD_diode_list[di]
            
            noise_params = local_noise_samples[di, si-1, :] 

            # Tsys = Tsys_operator@Tsys_samples[si-1, :] + TOD_diode
            Tsys = Tsys_operator@Tsys_samples[si-1, :] 
            
            
            # Sample gain parameters
            gain_sample = gain_coeff_sampler(TOD, t_list, gain_operator, Tsys, noise_params, 
                                            wnoise_var=wnoise_var,
                                            n_samples=num_sample, 
                                            tol=tol, 
                                            prior_cov_inv=local_gain_prior_cov_inv_list[di], 
                                            prior_mean=local_gain_prior_mean_list[di], 
                                            solver=linear_solver)
            local_gain_samples[di, si, :] = gain_sample
            gains = gain_operator@gain_sample
            local_gain_list[di]=gains
            print("Rank: {}, local id: {}, gain_sample {}: {}".format(rank, di, si, gain_sample))

            # Sample noise parameters
            noise_sample = flicker_noise_sampler(TOD,
                                                t_list,
                                                gains,
                                                Tsys,
                                                init_noise_params, # using the input init_noise_params as fixed initial point for MCMC sampling
                                                n_samples=num_sample,
                                                wnoise_var=wnoise_var,
                                                prior_func=local_noise_prior_func_list[di])
            local_noise_samples[di, si, :] = noise_sample
            print("Rank: {}, local id: {}, noise_sample {}: {}".format(rank, di, si, noise_sample))


        # Given gain, noise and other Tsys components, sample Tsky parameters
        Tsys_samples[si, :] = Tsys_sampler_multi_TODs(local_TOD_list,
                                                    local_t_lists,
                                                    local_gain_list,
                                                    local_Tsys_operator_list,
                                                    local_noise_samples[:, si, :],
                                                    # local_TOD_diode_list,
                                                    wnoise_var=wnoise_var,
                                                    tol=tol,
                                                    prior_cov_inv=Tsys_prior_cov_inv,
                                                    prior_mean=Tsys_prior_mean,
                                                    solver=linear_solver,
                                                    Est_mode=Est_mode)



    # Gather local gain and noise samples
    mpiutil.barrier()

    all_gain_samples = None
    all_noise_samples = None
    if root is None:
        # Gather all results onto all ranks
        all_gain_samples = comm.allgather(local_gain_samples)
        all_noise_samples = comm.allgather(local_noise_samples)
    else:
        # Gather all results onto the specified rank
        all_gain_samples = comm.gather(local_gain_samples, root=root)
        all_noise_samples = comm.gather(local_noise_samples, root=root)

    return Tsys_samples, all_gain_samples, all_noise_samples





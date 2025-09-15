# This file contains the full Gibbs sampler for all the parameters in data model, 
# including the system temperature parameters, noise parameters, gain parameters.

# Full Gibbs sampler
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import mpiutil
from gain_sampler import gain_sampler
from flicker_model import FlickerCorrEmulator
from noise_sampler_fixed_fc import flicker_sampler
from noise_sampler_old import flicker_noise_sampler
from Tsys_sampler import Tsys_coeff_sampler, Tsys_sampler_multi_TODs
from linear_solver import cg
from scipy.linalg import block_diag
from tqdm import tqdm


comm = mpiutil.world
rank = mpiutil.rank
rank0 = rank == 0

 # Default bounds format: [[lower1, upper1], [lower2, upper2], ...]
flicker_bounds = [(-6.0, -3.0), (1.1, 4.0)] 
Gflicker_bounds = [[4.0, 10.0], [-6.0, -3.0], [1.1, 4.0]]


def get_Tsys_operator(local_Tsky_operator_list, local_Tloc_operator_list):
    # Generate Tsys operator from Tsky and Tloc operators
    # For example, we illustrate this in a block matrix form:
    # say we have two ranks,
    # rank0 has local Tsky operators U1, U2, and local Tloc operators R1, R2, 
    # rank1 has local Tsky operators U3, and local Tloc operators R3,
    # then the overall Tsys operator U is:
    #        ( U1 R1  0  0 )
    # U =    ( U2  0 R2  0 )
    #        ( U3  0  0  R3)
    # It will be again saved as local lists: rank0 has [( U1 R1  0  0 ), (U2  0 R2  0 )],
    # and rank1 has [( U3  0  0  R3)].

    # The linear system reads
    # ( U1 R1  0  0 )       (Tsky)       (n1)       (d1)
    # ( U2  0 R2  0 )   @   (Tloc1)   +  (n2)   =   (d2)
    # ( U3  0  0  R3)       (Tloc2)      (n3)       (d3)
    #                       (Tloc3)
    """
    Construct combined Tsys operator matrix from Tsky and Tloc operators.
    
    Args:
        local_Tsky_operator_list: List of Tsky operators for each TOD
        local_Tloc_operator_list: List of Tloc operators for each TOD
        
    Returns:
        List of combined Tsys operators for each TOD
    """
    num_TODs = len(local_Tsky_operator_list)

    local_loc_dim = [op.shape[1] for op in local_Tloc_operator_list]
    local_total_loc_dim = sum(local_loc_dim)
    glist_total_loc_dims = comm.allgather(local_total_loc_dim)
    global_total_loc_dim = sum(glist_total_loc_dims)
    # cumulative sum of glist_total_loc_dims
    rank_offset_list = [0] + np.cumsum(glist_total_loc_dims).tolist()
    local_rank_offset = rank_offset_list[rank]
    local_loc_dim = [0] + local_loc_dim

    local_Tsys_operator_list = []
    for di in range(num_TODs):
        dim_data = local_Tloc_operator_list[di].shape[0]
        dim_params = local_Tloc_operator_list[di].shape[1]
        Tloc_operator = np.zeros((dim_data, global_total_loc_dim))
        start_ind = local_rank_offset + local_loc_dim[di]
        end_ind = start_ind + dim_params
        Tloc_operator[:, start_ind:end_ind] = local_Tloc_operator_list[di]
        Tsys_operator = np.hstack((local_Tsky_operator_list[di], Tloc_operator))
        local_Tsys_operator_list.append(Tsys_operator)
    
    return local_Tsys_operator_list


def TOD_Gibbs_sampler(
        local_TOD_list, 
        local_t_lists,
        local_gain_operator_list,
        local_Tsky_operator_list,
        local_Tloc_operator_list,
        init_Tsky_params,
        init_Tloc_params_list,
        init_noise_params_list,
        local_logfc_list,
        local_Tsys_injection_list=None,
        wnoise_var=2.5e-6,
        Tsky_prior_cov_inv=None,
        Tsky_prior_mean=None,
        local_Tloc_prior_cov_inv_list=None,
        local_Tloc_prior_mean_list=None,
        local_gain_prior_cov_inv_list=None,
        local_gain_prior_mean_list=None,
        local_noise_prior_func_list=None,
        joint_Tsys_sampling=False,
        smooth_gain_model="linear", # "linear", "log", or "factorized"
        noise_sampler_type="emcee", # "emcee" or "nuts"
        noise_Jeffreys_prior=False, # "uniform", "jeffreys", "gaussian", or None
        noise_params_bounds=flicker_bounds,
        n_samples=100,
        tol=1e-12,
        linear_solver=cg,
        root=None,
        Est_mode=False,
        debug=False
    ):   
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
            or num_TODs != len(local_Tloc_operator_list) \
                or num_TODs != len(local_gain_operator_list) :
        raise ValueError("The length of the input lists must be the same.")

    if smooth_gain_model not in ["linear", "log", "factorized"]:
        raise ValueError("Unknown smooth_gain_model: {}. Supported models are 'linear', 'log', and 'factorized'.".format(smooth_gain_model))
    if smooth_gain_model == "factorized":
        include_DC_Gain = True
    else:
        include_DC_Gain = False

    if noise_params_bounds is None:
        if include_DC_Gain:
            noise_params_bounds = Gflicker_bounds
        else:
            noise_params_bounds = flicker_bounds
    
    if joint_Tsys_sampling:
        Tloc_prior_cov_inv_list = [item for sublist in comm.allgather(local_Tloc_prior_cov_inv_list) 
                                    for item in sublist]
        Tloc_prior_mean_list = [item for sublist in comm.allgather(local_Tloc_prior_mean_list)
                                for item in sublist]
        init_Tloc_params_list = [item for sublist in comm.allgather(init_Tloc_params_list)
                                 for item in sublist]

        if Tsky_prior_cov_inv.ndim==1: 
            # assert all
            assert all(Tloc_prior_cov_inv.ndim==1 for Tloc_prior_cov_inv in Tloc_prior_cov_inv_list), "All prior_cov_inv must have the same number of dimensions [1D (i.e., diagonals) or 2D]."
            # The prior covariance matrices are diagonal, the inputs are the diagonal elements
            Tsys_prior_cov_inv = np.hstack([Tsky_prior_cov_inv]+Tloc_prior_cov_inv_list)
        elif Tsky_prior_cov_inv.ndim==2:
            assert all(Tloc_prior_cov_inv.ndim==2 for Tloc_prior_cov_inv in Tloc_prior_cov_inv_list), "All prior_cov_inv must have the same number of dimensions [1D (i.e., diagonals) or 2D]."
            Tsys_prior_cov_inv = block_diag(Tsky_prior_cov_inv, *Tloc_prior_cov_inv_list)
        else:
            raise ValueError("Invalid number of dimensions for Tsky_prior_cov_inv.")
        Tsys_prior_mean = np.hstack([Tsky_prior_mean]+Tloc_prior_mean_list)
        init_Tsys_params = np.hstack([init_Tsky_params]+init_Tloc_params_list)

        local_Tsys_operator_list = get_Tsys_operator(local_Tsky_operator_list, local_Tloc_operator_list)
        n_sys_modes = local_Tsys_operator_list[0].shape[1]
        Tsys_samples = np.zeros((n_samples, n_sys_modes))
        Tsys_samples[-1, :] = init_Tsys_params
    else:
        n_loc_modes = local_Tloc_operator_list[0].shape[1]
        local_Tloc_samples = np.zeros((num_TODs, n_samples, n_loc_modes))

        n_sky_modes = len(init_Tsky_params)
        Tsky_samples = np.zeros((n_samples, n_sky_modes))
        Tsky_params = init_Tsky_params

        local_Tsky_mu_list = [np.zeros_like(local_TOD_list[di]) for di in range(num_TODs)]

        for di in range(num_TODs):
            local_Tloc_samples[di, -1, :] = init_Tloc_params_list[di]
    
    n_g_modes = local_gain_operator_list[0].shape[1]
    local_gain_samples = np.zeros((num_TODs, n_samples, n_g_modes))

    n_N_modes = 3 if include_DC_Gain else 2
    local_noise_samples = np.zeros((num_TODs, n_samples, n_N_modes))

    for di in range(num_TODs):
        t_list = local_t_lists[di]
        local_noise_samples[di, -1, :] = np.array(init_noise_params_list[di])
    
    local_gain_list = [np.zeros_like(local_TOD_list[di]) for di in range(num_TODs)]

    if Est_mode:
        num_sample = 0 
    else:
        num_sample = 1

    if local_Tsys_injection_list is None:
        local_Tsys_injection_list = [0.] * num_TODs
    if local_noise_prior_func_list is None:
        local_noise_prior_func_list = [None] * num_TODs
    if local_gain_prior_mean_list is None:
        local_gain_prior_mean_list = [None] * num_TODs
    if local_gain_prior_cov_inv_list is None:
        local_gain_prior_cov_inv_list = [None] * num_TODs
    if local_Tloc_prior_mean_list is None:
        local_Tloc_prior_mean_list = [None] * num_TODs
    if local_Tloc_prior_cov_inv_list is None:
        local_Tloc_prior_cov_inv_list = [None] * num_TODs


    master_rng_key = jr.PRNGKey(42)
    noise_key = None

    from tqdm import tqdm
    # Sample the parameters
    pbar = tqdm(range(n_samples), desc="Gibbs Sampling", disable=(not rank0))
    for si in pbar:
        for di in range(num_TODs):
            TOD = local_TOD_list[di]
            t_list = local_t_lists[di]
            gain_operator = local_gain_operator_list[di]

            if joint_Tsys_sampling:
                Tsys_operator = local_Tsys_operator_list[di]
                Tsys = Tsys_operator@Tsys_samples[si-1, :] + local_Tsys_injection_list[di]
            else:
                Tsky_operator = local_Tsky_operator_list[di]
                Tloc_operator = local_Tloc_operator_list[di]
                Tloc_params = local_Tloc_samples[di, si-1, :]
                TOD_sky = Tsky_operator@Tsky_params
                TOD_loc = Tloc_operator@Tloc_params
                Tsys = TOD_sky + TOD_loc + local_Tsys_injection_list[di] 
                mu_loc = TOD_sky + local_Tsys_injection_list[di]

            noise_params = local_noise_samples[di, si-1, :] 

            logfc = local_logfc_list[di]

            # Sample gain parameters
            gain_sample, gains = gain_sampler(
                TOD, t_list, gain_operator, Tsys, noise_params,
                logfc,
                model=smooth_gain_model,
                wnoise_var=wnoise_var,
                n_samples=num_sample,
                tol=tol,
                prior_cov_inv=local_gain_prior_cov_inv_list[di],
                prior_mean=local_gain_prior_mean_list[di],
                solver=linear_solver
            )

            if debug:
                print("Rank: {}, local id: {}, gain_sample {}: {}".format(rank, di, si, gain_sample))

            local_gain_samples[di, si, :] = gain_sample
              
            if include_DC_Gain:
                DC_gain = noise_params[0]
                aux_gains = gains * DC_gain
                noise_params = noise_params[1:]
            else:
                aux_gains = gains

            if not joint_Tsys_sampling:
                # Sample local Tsys temperature parameters
                Tloc_params = Tsys_coeff_sampler(
                    TOD,
                    t_list,
                    aux_gains,
                    Tloc_operator,
                    noise_params,
                    logfc=local_logfc_list[di],
                    wnoise_var=wnoise_var,
                    n_samples=num_sample,
                    mu=mu_loc,
                    tol=tol,
                    prior_cov_inv=local_Tloc_prior_cov_inv_list[di],
                    prior_mean=local_Tloc_prior_mean_list[di],
                    solver=linear_solver
                )
                
                local_Tloc_samples[di, si, :] = Tloc_params
                TOD_loc = Tloc_operator@Tloc_params

                Tsys = TOD_sky + TOD_loc + local_Tsys_injection_list[di]
                # Collect mu vectors for Tsky sampler
                local_Tsky_mu_list[di]= TOD_loc + local_Tsys_injection_list[di]


            # Sample noise parameters
            
            # if noise_sampler_type == "nuts":
            #     master_rng_key, noise_key = jr.split(master_rng_key)
            
            # noise_sample = flicker_sampler(
            #     TOD,
            #     gains,
            #     Tsys,
            #     init_params=jnp.array(init_noise_params_list[di]), # using the input init_noise_params as fixed initial point for MCMC sampling
            #     n_samples=num_sample,
            #     include_DC_Gain=include_DC_Gain, 
            #     prior_func=local_noise_prior_func_list[di],
            #     jeffreys=noise_Jeffreys_prior,
            #     bounds=noise_params_bounds,
            #     sampler=noise_sampler_type,
            #     rng_key=noise_key
            # )

            noise_sample = flicker_noise_sampler(TOD,
                                                t_list,
                                                gains,
                                                Tsys,
                                                init_noise_params_list[di], # using the input init_noise_params as fixed initial point for MCMC sampling
                                                logfc,
                                                n_samples=num_sample,
                                                wnoise_var=wnoise_var,
                                                prior_func=local_noise_prior_func_list[di])

            if include_DC_Gain:
                local_gain_list[di] = gains * noise_sample[0]
            else:
                local_gain_list[di] = gains


            local_noise_samples[di, si, :] = noise_sample

            if debug:
                # print("Rank: {}, local id: {}, gain_sample {}: {}".format(rank, di, si, gain_sample))
                print("Rank: {}, local id: {}, noise_sample {}: {}".format(rank, di, si, noise_sample))

        if joint_Tsys_sampling:
            # Given gain and noise, sample Tsys parameters
            linear_op_aux = local_Tsys_operator_list
            mu_aux = local_Tsys_injection_list
            prior_icov_aux = Tsys_prior_cov_inv
            prior_mean_aux = Tsys_prior_mean
        else:
            linear_op_aux = local_Tsky_operator_list
            mu_aux = local_Tsky_mu_list
            prior_icov_aux = Tsky_prior_cov_inv
            prior_mean_aux = Tsky_prior_mean

        if include_DC_Gain:
            flicker_parameters = local_noise_samples[:, si, 1:]
        else:
            flicker_parameters = local_noise_samples[:, si, :]

        sample = Tsys_sampler_multi_TODs(
            local_TOD_list,
            local_t_lists,
            local_gain_list,
            linear_op_aux,
            flicker_parameters,
            local_logfc_list,
            local_mu_list=mu_aux,
            wnoise_var=wnoise_var,
            tol=tol,
            prior_cov_inv=prior_icov_aux,
            prior_mean=prior_mean_aux,
            solver=linear_solver,
            Est_mode=Est_mode
        )
        if joint_Tsys_sampling:
            Tsys_samples[si, :] = sample
        else:
            Tsky_params = sample
            Tsky_samples[si, :] = Tsky_params
        
        # Update after completing all TODs for this sample
        if rank0:
            pbar.set_postfix({
                'Sample': si+1,
                'Status': 'Complete'
            })
    
    pbar.close()

    # Gather local gain and noise samples
    mpiutil.barrier()

    all_gain_samples = None
    all_noise_samples = None
    all_Tloc_samples = None
    if root is None:
        # Gather all results onto all ranks
        all_gain_samples = comm.allgather(local_gain_samples)
        all_noise_samples = comm.allgather(local_noise_samples)
        if not joint_Tsys_sampling:
            all_Tloc_samples = comm.allgather(local_Tloc_samples)
    else:
        # Gather all results onto the specified rank
        all_gain_samples = comm.gather(local_gain_samples, root=root)
        all_noise_samples = comm.gather(local_noise_samples, root=root)
        if not joint_Tsys_sampling:
            all_Tloc_samples = comm.gather(local_Tloc_samples, root=root)

    if not joint_Tsys_sampling:
        return Tsky_samples, all_gain_samples, all_noise_samples, all_Tloc_samples
    return Tsys_samples, all_gain_samples, all_noise_samples


def TOD_Gibbs_sampler_joint_loc(local_TOD_list, 
                                  local_t_lists,
                                #   local_TOD_diode_list,
                                  local_gain_operator_list,
                                  local_Tsky_operator_list,
                                  local_Tloc_operator_list,
                                  init_Tsky_params,
                                  init_Tloc_params_list,
                                  init_noise_params_list,
                                  local_logfc_list,
                                  wnoise_var=2.5e-6,
                                  Tsky_prior_cov_inv=None,
                                  Tsky_prior_mean=None,
                                  local_Tloc_prior_cov_inv_list=None,
                                  local_Tloc_prior_mean_list=None,
                                  local_gain_prior_cov_inv_list=None,
                                  local_gain_prior_mean_list=None,
                                  local_noise_prior_func_list=None,
                                  noise_sampler_type="emcee", # "emcee" or "nuts"
                                  ploc_Jeffreys_prior=True, 
                                  noise_Jeffreys_prior=True, 
                                  noise_params_bounds=flicker_bounds,
                                  n_samples=100,
                                  tol=1e-12,
                                  linear_solver=cg,
                                  Est_mode=False,
                                  root=None,
                                  debug=False):   
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
            or num_TODs != len(local_Tloc_operator_list) \
                or num_TODs != len(local_gain_operator_list) :
        raise ValueError("The length of the input lists must be the same.")
    
    n_g_modes = local_gain_operator_list[0].shape[1]
    local_gain_samples = np.zeros((num_TODs, n_samples, n_g_modes))

    n_loc_modes = local_Tloc_operator_list[0].shape[1]
    local_Tloc_samples = np.zeros((num_TODs, n_samples, n_loc_modes))

    n_sky_modes = len(init_Tsky_params)
    Tsky_samples = np.zeros((n_samples, n_sky_modes))

    n_N_modes = len(init_noise_params_list[0])
    local_noise_samples = np.zeros((num_TODs, n_samples, n_N_modes))

    Tsky_params=init_Tsky_params

    for di in range(num_TODs):
        t_list = local_t_lists[di]
        local_noise_samples[di, -1, :] = init_noise_params_list[di]
        local_Tloc_samples[di, -1, :] = init_Tloc_params_list[di]
        # Ncov = flicker_cov(t_list, 10.**logf0, 10.**logfc, alpha, white_n_variance=wnoise_var, only_row_0=False)
        # local_Ncov_list.append(Ncov)

    if local_noise_prior_func_list is None:
        local_noise_prior_func_list = [None] * num_TODs

    local_gain_list = [np.zeros_like(local_TOD_list[di]) for di in range(num_TODs)]
    local_Tsky_mu_list = [np.zeros_like(local_TOD_list[di]) for di in range(num_TODs)]

    if Est_mode:
        num_sample = 0 
    else:
        num_sample = 1

    from tqdm import tqdm
    from local_sampler import local_params_sampler

    master_rng_key = jr.PRNGKey(42)
    
    # Sample the parameters
    for si in tqdm(range(n_samples)):
        for di in range(num_TODs):
            TOD = local_TOD_list[di]
            gain_operator = local_gain_operator_list[di]
            Tsky_operator = local_Tsky_operator_list[di]
            Tloc_operator = local_Tloc_operator_list[di]
            # TOD_diode = local_TOD_diode_list[di]

            noise_params = local_noise_samples[di, si-1, :] 

            TOD_sky = Tsky_operator@Tsky_params
            Tnd_vec = Tloc_operator[:,0]
            Tres_proj = Tloc_operator[:,1:]

            # local_gain_prior_cov_inv_list[di]
            local_params_prior_cov_inv = np.concatenate([local_gain_prior_cov_inv_list[di], local_Tloc_prior_cov_inv_list[di]])
            local_params_prior_mean = np.concatenate([local_gain_prior_mean_list[di], local_Tloc_prior_mean_list[di]])

            def p_loc_prior(x):
                return jnp.sum(-0.5 * (x - local_params_prior_mean)**2 * local_params_prior_cov_inv)

            master_rng_key, noise_key = jr.split(master_rng_key)


            local_params = local_params_sampler(TOD, TOD_sky, gain_operator, Tnd_vec, Tres_proj, noise_params, 
                                                rng_key=noise_key,
                                                add_jeffreys=ploc_Jeffreys_prior,
                                                prior_func=p_loc_prior,
                                                jaxjit=False
                                                ) 
            print("local_params {}: \n {} \n {}".format(si, local_params[:4], local_params[4:]))

            gain_sample = local_params[:4]
            local_gain_samples[di, si, :] = gain_sample
            gains = np.exp(gain_operator@gain_sample)
            local_gain_list[di]=gains

            Tloc_params = local_params[4:]
            local_Tloc_samples[di, si, :] = Tloc_params
            Tloc_TOD = Tnd_vec * np.exp(Tloc_params[0]) + np.exp(Tres_proj@Tloc_params[1:])
            Tsys = TOD_sky + Tloc_TOD
            local_Tsky_mu_list[di]=Tloc_TOD

            # Sample noise parameters
            noise_sample = flicker_sampler(
                TOD,
                gains,
                Tsys,
                init_params=init_noise_params_list[di], # using the input init_noise_params as fixed initial point for MCMC sampling
                n_samples=num_sample,
                include_DC_Gain=False, 
                prior_func=local_noise_prior_func_list[di],
                jeffreys=noise_Jeffreys_prior,
                bounds=noise_params_bounds,
                sampler=noise_sampler_type,
                rng_key=noise_key
            )

            local_noise_samples[di, si, :] = noise_sample
            if debug:
                print("Rank: {}, local id: {}, noise_sample {}: {}".format(rank, di, si, noise_sample))


        # Given gain, noise and other Tsys components, sample Tsky parameters
        Tsky_params = Tsys_sampler_multi_TODs(local_TOD_list,
                                              local_t_lists,
                                              local_gain_list,
                                              local_Tsky_operator_list,
                                              local_noise_samples[:, si, :],
                                              local_logfc_list,
                                              local_mu_list=local_Tsky_mu_list,
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
    all_Tloc_samples = None
    if root is None:
        # Gather all results onto all ranks
        all_gain_samples = comm.allgather(local_gain_samples)
        all_noise_samples = comm.allgather(local_noise_samples)
        all_Tloc_samples = comm.allgather(local_Tloc_samples)
    else:
        # Gather all results onto the specified rank
        all_gain_samples = comm.gather(local_gain_samples, root=root)
        all_noise_samples = comm.gather(local_noise_samples, root=root)
        all_Tloc_samples = comm.gather(local_Tloc_samples, root=root)

    return Tsky_samples, all_gain_samples, all_noise_samples, all_Tloc_samples


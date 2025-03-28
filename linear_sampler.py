import numpy as np
import mpiutil
from scipy.linalg import solve
from numpy.linalg import cholesky
from scipy.linalg import block_diag
from linear_solver import cg, pytorch_lin_solver
from mpi4py import MPI

comm = mpiutil.world
rank = mpiutil.rank
rank0 = rank == 0


def params_space_oper_and_data(d, U, p, N_inv, mu=0.0, Ninv_sqrt=None):
    D_p_inv = 1./(U @ p + mu)
    sigma_inv = N_inv * np.outer(D_p_inv, D_p_inv)
    aux = U.T @ sigma_inv
    if Ninv_sqrt is None:
        return aux@U, aux@(d-mu)
    else:
        sigma_inv_sqrt = D_p_inv[:, np.newaxis] * Ninv_sqrt 
        return aux @ U, aux @ (d-mu), U.T @ sigma_inv_sqrt

# def params_space_oper_and_data_list(d_list, U_list, p, N_inv_list,  Ninv_sqrt_list, mu_list):
#     dim = len(d_list)
#     UT_sigma_U, UT_sigma_d, UT_sigma_inv_sqrt = 0., 0., 0.
#     for i in range(dim):
#         U_i = U_list[i]
#         d_i = d_list[i]
#         Ninv_i = N_inv_list[i]
#         Ninv_sqrt_i = Ninv_sqrt_list[i]
#         mu_i = mu_list[i]

#         D_p_inv = 1.0 / (U_i @ p + mu_i)
#         # Sigma_inv_i = Ninv_i * np.outer(D_p_inv, D_p_inv)
#         # UT_sigma_U += U_i.T @ Sigma_inv_i @ U_i
#         D_p_inv_U = D_p_inv[:, np.newaxis] * U_i
#         UT_sigma_U += D_p_inv_U.T @ Ninv_i @ D_p_inv_U
#         UT_sigma_d += U_i.T @ (d_i - mu_i)
#         UT_sigma_inv_sqrt += U_i.T @ (D_p_inv[:, np.newaxis] * Ninv_sqrt_i)

#     return UT_sigma_U, UT_sigma_d, UT_sigma_inv_sqrt

def params_space_oper_and_data_list_v1(d_list, U_list, p, Ninv_sqrt_list, mu_list, draw=True, root=None):
    dim = len(d_list)
    n_params = U_list[0].shape[1]

    # Preallocate arrays
    UT_sigma_U = np.zeros((n_params, n_params))
    UT_sigma_d = np.zeros(n_params)

    if draw:
        UT_sigma_sqrt_wn = np.zeros(n_params)

    for i in range(dim):
        U_i = U_list[i]
        d_i = d_list[i]
        Ninv_sqrt_i = Ninv_sqrt_list[i]
        mu_i = mu_list[i]

        D_p_inv = 1.0 / (U_i @ p + mu_i)
        aux = U_i.T @ (D_p_inv[:, np.newaxis] * Ninv_sqrt_i)
        UT_sigma_U += aux @ aux.T
        UT_sigma_d += U_i.T @ (d_i - mu_i)
        if draw:
            # white noise vector
            wn = np.random.normal(0, 1, size=U_i.shape[0])
            UT_sigma_sqrt_wn += aux @ wn

    # Reduce the results; if root is None, all reduce; otherwise, reduce from root
    if root is None:
        UT_sigma_U = comm.allreduce(UT_sigma_U, op=MPI.SUM)
        UT_sigma_d = comm.allreduce(UT_sigma_d, op=MPI.SUM)
        if draw:
            UT_sigma_sqrt_wn = comm.allreduce(UT_sigma_sqrt_wn, op=MPI.SUM)
            return UT_sigma_U, UT_sigma_d, UT_sigma_sqrt_wn
        else:
            return UT_sigma_U, UT_sigma_d
    else:
        UT_sigma_U = comm.reduce(UT_sigma_U, op=MPI.SUM, root=root)
        UT_sigma_d = comm.reduce(UT_sigma_d, op=MPI.SUM, root=root)
        if draw:
            UT_sigma_sqrt_wn = comm.reduce(UT_sigma_sqrt_wn, op=MPI.SUM, root=root)
            if rank==root:
                return UT_sigma_U, UT_sigma_d, UT_sigma_sqrt_wn
            else:
                return None, None, None
        else:
            if rank==root:
                return UT_sigma_U, UT_sigma_d
            else:
                return None, None

def params_space_oper_and_data_list(d_list, U_list, p, Ninv_sqrt_list, mu_list, draw=True, root=None):

    dim = len(d_list)
    n_params = U_list[0].shape[1]

    # Preallocate combined array - a stack of UT_sigma_U, UT_sigma_d, and UT_sigma_sqrt_wn
    # The first n_params rows are UT_sigma_U, the next row is UT_sigma_d, and the last row is UT_sigma_sqrt_wn
    if draw:
        combined_matrix = np.zeros((n_params + 2, n_params), dtype=np.float64)
    else:
        combined_matrix = np.zeros((n_params + 1, n_params), dtype=np.float64)

    for i in range(dim):
        U_i = U_list[i]
        d_i = d_list[i]
        Ninv_sqrt_i = Ninv_sqrt_list[i]
        mu_i = mu_list[i]

        D_p_inv = 1.0 / (U_i @ p + mu_i)
        aux = U_i.T @ (D_p_inv[:, np.newaxis] * Ninv_sqrt_i)
        combined_matrix[:n_params, :] += aux @ aux.T
        combined_matrix[n_params, :] += U_i.T @ (d_i - mu_i)
        if draw:
            # white noise vector
            wn = np.random.normal(0, 1, size=U_i.shape[0])
            combined_matrix[n_params+1, :] += aux @ wn

    # Reduce the combined matrix
    
    if root is None:
        global_combined_matrix = np.zeros_like(combined_matrix)
        comm.Allreduce(combined_matrix, global_combined_matrix, op=MPI.SUM)
    elif rank == root:
        global_combined_matrix = np.zeros_like(combined_matrix)
        comm.Reduce(combined_matrix, global_combined_matrix, op=MPI.SUM, root=root)
    else:  # non-root processes
        if draw:
            return None, None, None
        else:
            return None, None

    # For root processes (or no specified root), extract UT_sigma_U and UT_sigma_d from the combined matrix
    UT_sigma_U = global_combined_matrix[:n_params, :]
    UT_sigma_d = global_combined_matrix[n_params, :]
    if draw:
        UT_sigma_sqrt_wn = global_combined_matrix[n_params+1, :]
        return UT_sigma_U, UT_sigma_d, UT_sigma_sqrt_wn
    else:
        return UT_sigma_U, UT_sigma_d



def iterative_gls(d, 
                  U, 
                  N_inv, 
                  mu=0.0, 
                  tol=1e-10, 
                  min_iter=5, 
                  max_iter=100, 
                  solver=pytorch_lin_solver,
                  ):
    """
    Iteratively solves for p using GLS with heteroskedastic noise.

    Data model: d = U p (1 + n), where n ~ Gaussian(0, N).
    
    Parameters:
    U : ndarray (M, N) or a list of ndarrays [..., (M_i, N), ...]
        Projection matrix
    d : ndarray (M,) or a list of ndarrays [..., (M_i,), ...]
        Observed data
    N : ndarray (M, M) or a list of ndarrays [..., (M_i, M_i), ...]
        Covariance matrix of noise (assumed positive definite)
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum number of iterations

    Returns:
    p_gls : ndarray (N,)
        Estimated parameter vector p
    Sigma_inv : ndarray (M, M)
        Covariance matrix for sampling
    """

    # Initialize p using ordinary least squares (OLS) as a starting point
    p = np.linalg.lstsq(U, d-mu, rcond=None)[0]  

    for iteration in range(1, max_iter+1):
        # Compute noise covariance Σ_ε = diag(U p) N diag(U p)
        # D_p = np.diag(U @ p)
        # Sigma_epsilon = D_p @ N @ D_p
        # Sigma_inv = np.linalg.inv(Sigma_epsilon)

        # Solve GLS: (U^T Σ_ε⁻¹ U) p = U^T Σ_ε⁻¹ d
        UTSigma_invU, UTSigma_invD = params_space_oper_and_data(d, U, p, N_inv, mu=mu)

        if solver is None:
            p_new = solve(UTSigma_invU, UTSigma_invD, assume_a='sym')
        else:
            p_new = solver(UTSigma_invU, UTSigma_invD)

        # Check for convergence
        if np.linalg.norm(p_new - p) < tol*np.linalg.norm(p) and iteration >= min_iter:
            print(f"Converged in {iteration} iterations.")
            break
        
        p = p_new

    # Compute final covariance for posterior sampling
    # covariance_p = np.linalg.inv(U.T @ Sigma_inv @ U)
    D_p_inv = 1./(U @ p + mu)
    Sigma_inv = N_inv * np.outer(D_p_inv, D_p_inv)
    return p_new, Sigma_inv



def iterative_gls_mpi_list(local_d_list, 
                           local_U_list, 
                           local_Ninv_sqrt_list, 
                           local_mu_list, 
                           tol=1e-10, 
                           min_iter=5, 
                           max_iter=200, 
                           solver=cg
                           ):
    """
    Iteratively solves for p using GLS with heteroskedastic noise.

    Data model: d = U p (1 + n), where n ~ Gaussian(0, N).
    
    Parameters:
    U : ndarray (M, N) or a list of ndarrays [..., (M_i, N), ...]
        Projection matrix
    d : ndarray (M,) or a list of ndarrays [..., (M_i,), ...]
        Observed data
    N : ndarray (M, M) or a list of ndarrays [..., (M_i, M_i), ...]
        Covariance matrix of noise (assumed positive definite)
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum number of iterations

    Returns:
    p_gls : ndarray (N,)
        Estimated parameter vector p
    Sigma_inv : ndarray (M, M)
        Covariance matrix for sampling
    """

    dim = len(local_U_list)
    assert dim==len(local_Ninv_sqrt_list) and dim==len(local_mu_list) and dim==len(local_d_list), \
        "U, d and N must have the same length if they are lists."

    

    n_params = local_U_list[0].shape[-1]

    local_UT_U = np.zeros((n_params, n_params), dtype=np.float64)
    local_UT_d = np.zeros(n_params, dtype=np.float64)
    #local_Ninv_sqrt_list = []
    for i in range(dim):
        Ui = local_U_list[i]
        di = local_d_list[i]
        local_UT_U += Ui.T @ Ui
        local_UT_d += Ui.T @ di
        #local_Ninv_sqrt_list.append(cholesky(local_N_inv_list[i], upper=False))
            
    # initialize receive buffer on the root process
    if rank0:
        UT_U = np.zeros((n_params, n_params), dtype=np.float64)
        UT_d = np.zeros(n_params, dtype=np.float64)
    else:
        UT_U = None
        UT_d = None

    comm.Reduce(local_UT_U, UT_U, op=MPI.SUM, root=0)
    comm.Reduce(local_UT_d, UT_d, op=MPI.SUM, root=0)

    if mpiutil.rank0:
        p = cg(UT_U, UT_d)
        p = p.astype(np.float64)
    else:
        p = np.empty(n_params, dtype=np.float64)  # Initialize buffer on all processes
    # broadcast p to all processes
    comm.Bcast(p, root=0)
    
    mpiutil.barrier()
    
    for iteration in range(1, max_iter+1):

        UTSigma_invU, UTSigma_invD = \
            params_space_oper_and_data_list(local_d_list, local_U_list, p, 
                                            local_Ninv_sqrt_list, local_mu_list, draw=False)
    

        if mpiutil.rank0:
            if solver is None:
                p_new = solve(UTSigma_invU, UTSigma_invD, assume_a='sym')
                # set the dtype of p_new to float64
                p_new = p_new.astype(np.float64)
            else:
                p_new = solver(UTSigma_invU, UTSigma_invD)
                p_new = p_new.astype(np.float64)
        else:
            p_new = np.empty(n_params, dtype=np.float64)  # Initialize buffer on all processes

        comm.Bcast(p_new, root=0)

        frac_norm_error = np.linalg.norm(p_new - p) / np.linalg.norm(p)
        # Broadcast the norm error of rank 0 to all processes - make sure to use the same error value
        frac_norm_error = comm.bcast(frac_norm_error, root=0)
        # Check for convergence
        if frac_norm_error < tol and iteration >= min_iter:
            if mpiutil.rank0:
                print(f"Iterative GLS converged in {iteration} iterations.")  
            break
        elif iteration >= max_iter:
            print(f"Reached max iterations with fractional norm error {frac_norm_error}.")
            break
        
        p = p_new

    # Compute final covariance (and other products) for posterior sampling
    UT_Sigma_U, UT_Sigma_D, draw_vec = \
            params_space_oper_and_data_list(local_d_list, local_U_list, p_new, 
                                            local_Ninv_sqrt_list, local_mu_list, draw=True)

    return p_new, UT_Sigma_U, UT_Sigma_D, draw_vec 



def iterative_gls_list(U, d, N, tol=1e-10, min_iter=5, max_iter=100, solver=cg):
    """
    Iteratively solves for p using GLS with heteroskedastic noise.

    Data model: d = U p (1 + n), where n ~ Gaussian(0, N).
    
    Parameters:
    U : a list of ndarrays [..., (M_i, N), ...]
        Projection matrix
    d : a list of ndarrays [..., (M_i,), ...]
        Observed data
    N : a list of ndarrays [..., (M_i, M_i), ...]
        Covariance matrix of noise (assumed positive definite)
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum number of iterations

    Returns:
    p_gls : ndarray (N,)
        Estimated parameter vector p
    Sigma_inv : ndarray (M, M)
        Covariance matrix for sampling
    """

    # If U, d and N are lists, then stack them into matrices
    
    assert isinstance(U, list) and isinstance(d, list) and isinstance(N, list), "If U, d and N are lists, they must all be lists."
    assert len(U) == len(d) == len(N), "U, d and N must have the same length if they are lists."
    U_stack = np.vstack(U)
    d_stack = np.hstack(d)
    N = block_diag(*N)

    # Initialize p using ordinary least squares (OLS) as a starting point
    p = np.linalg.lstsq(U_stack, d_stack, rcond=None)[0]  

    
    for iteration in range(1, max_iter+1):
        # Compute noise covariance Σ_ε = diag(U p) N diag(U p)
        # D_p = np.diag(U @ p)
        # Sigma_epsilon = D_p @ N @ D_p
        D_p = U @ p
        Sigma_epsilon = N * np.outer(D_p, D_p)
        
        # Solve GLS: (U^T Σ_ε⁻¹ U) p = U^T Σ_ε⁻¹ d
        Sigma_inv = np.linalg.inv(Sigma_epsilon)

        D_p_inv = 1/D_p
        UT_Sigma_inv = U.T @ (N_inv * np.outer(D_p_inv, D_p_inv))

        UTSigma_invU = UT_Sigma_inv @ U
        UTSigma_invD = UT_Sigma_inv @ d
        if solver is None:
            p_new = solve(UTSigma_invU, UTSigma_invD, assume_a='sym')
        else:
            p_new = solver(UTSigma_invU, UTSigma_invD)

        # Check for convergence
        if np.linalg.norm(p_new - p) < tol*np.linalg.norm(p) and iteration >= min_iter:
            print(f"Converged in {iteration} iterations.")
            break
        p = p_new

    # Compute final covariance for posterior sampling
    # covariance_p = np.linalg.inv(U.T @ Sigma_inv @ U)

    return p, Sigma_inv



def sample_p(d, 
            U, 
            Sigma_inv, 
            mu=0.0,
            num_samples=1, 
            prior_cov_inv=None, 
            prior_mean=None,
            solver=cg,
            p_GLS=None,
            ):
    """
    Draw samples from the likelihood distribution of p.

    Parameters:
    d : ndarray (M,)
        Observed data
    U : ndarray (M, N)
        Projection matrix
    Sigma_inv : ndarray (M, M)
        Inverse of noise covariance matrix Σ_ε
    num_samples : int
        Number of samples to draw

    Returns:
    samples : ndarray (num_samples, N)
        Samples of p from likelihood distribution
    """
    A = U.T @ Sigma_inv @ U 
    b = U.T @ Sigma_inv @ (d-mu)

    if prior_cov_inv is not None:
        assert prior_mean is not None, "Prior mean must be provided if prior covariance is provided.."
        A += prior_cov_inv
        b += prior_cov_inv @ prior_mean
    
        if solver is None:
            p_est = solve(A, b, assume_a='sym')
        else:
            p_est = solver(A, b) 

    elif p_GLS is not None: # No prior, and p_GLS estimation is provided
        p_est = p_GLS 
    else: # No prior, and p_GLS estimation is not provided - solve for it
        p_est = solve(A, b, assume_a='sym') 
        
    if num_samples == 0:
        return p_est

    p_cov = np.linalg.inv(A) 
    
    # return samples array with "mean: p_est" and "covariance: p_cov"
    # return np.random.multivariate_normal(p_est, p_cov, num_samples) # shape: (num_samples, n_modes)

    try:
        # First try with standard multivariate_normal
        return np.random.multivariate_normal(p_est, p_cov, num_samples)
    except np.linalg.LinAlgError:
        # If that fails, try with regularization
        reg = 1e-6
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                p_cov_reg = p_cov + reg * np.eye(p_cov.shape[0])
                return np.random.multivariate_normal(p_est, p_cov_reg, num_samples)
            except np.linalg.LinAlgError:
                reg *= 10
        # If all attempts fail, use Cholesky decomposition with regularization
        L = np.linalg.cholesky(p_cov + 1e-4 * np.eye(p_cov.shape[0]))
        return p_est + L @ np.random.normal(size=(num_samples, p_cov.shape[1])).T




def sample_p_v2(A, 
                b, 
                A_sqrt_wn,
                prior_cov_inv=None, 
                prior_mean=None,
                solver=cg
                ):
    """
    Draw samples from the likelihood or posterior distribution of p.

    Parameters:
    A : ndarray (N, N)
        Left-hand side of the linear system
    b : ndarray (N,)
        Right-hand side of the linear system
    prior_cov_inv : ndarray (N, N), optional
        Inverse of prior covariance matrix
    prior_mean : ndarray (N,), optional
        Prior mean vector
    solver : function, optional
        Linear solver function (e.g., scipy.linalg.solve)

    Returns:
    samples : ndarray (N,)
        A sample of p 
    """
    left_op = A
    right_vec = b + A_sqrt_wn

    
    if prior_cov_inv is not None:
        assert prior_mean is not None, "Prior mean must be provided if prior covariance is provided.."
        left_op += prior_cov_inv
        right_vec += prior_cov_inv @ prior_mean + cholesky(prior_cov_inv) @ np.random.normal(size=prior_mean.shape)
    
    if solver is None:
        p_sample = solve(left_op, right_vec, assume_a='sym')
    else:
        p_sample = solver(left_op, right_vec) 

    return p_sample

def sample_p_old(d, 
            U, 
            Sigma_inv, 
            num_samples=1, 
            mu=0.0,
            prior_cov_inv=None, 
            prior_mean=None,
            solver=cg
            ):
    A = U.T @ Sigma_inv @ U
    aux = U.T @ Sigma_inv @ (d-mu)
    Sigma_sqrt_inv = U.T @ cholesky(Sigma_inv, upper=False) # Careful with the convention of cholesky decomposition!
    num_d = len(d)
    _, n_modes = U.shape
    p_samples = np.zeros((num_samples, n_modes))

    if prior_cov_inv is not None:
        assert prior_mean is not None, "Prior mean must be provided if prior covariance is provided.."
        A += prior_cov_inv
        prior_cov_sqrt_inv = cholesky(prior_cov_inv, upper=False) # Careful with the convention of cholesky decomposition!
        aux += prior_cov_inv @ prior_mean 

    if solver is None:
        for i in range(num_samples):
            b = aux + Sigma_sqrt_inv @ np.random.randn(num_d) 
            if prior_cov_inv is not None:
                b += prior_cov_sqrt_inv @ np.random.normal(0, 1, n_modes)
            p_samples[i,:] = solve(A, b, assume_a='sym')   
    else:
        for i in range(num_samples):
            b = aux + Sigma_sqrt_inv @ np.random.randn(num_d) 
            if prior_cov_inv is not None:
                b += prior_cov_sqrt_inv @ np.random.normal(0, 1, n_modes)
            p_samples[i,:] = solver(A, b)

    return p_samples



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

def params_space_oper_and_data_list(d_list, U_list, p, Ninv_sqrt_list, mu_list=None, draw=True, root=None):

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
        if mu_list is None:
            mu_i = 0.0
        else:
            mu_i = mu_list[i]

        D_p_inv = 1.0 / (U_i @ p + mu_i)
        Sigma_inv_sqrt_i = D_p_inv[:, np.newaxis] * Ninv_sqrt_list[i]
        aux = U_i.T @ Sigma_inv_sqrt_i
        combined_matrix[:n_params, :] += aux @ aux.T
        combined_matrix[n_params, :] += aux @ Sigma_inv_sqrt_i.T @ (d_i - mu_i)
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


def iterative_gls_mpi_list(local_d_list, 
                           local_U_list, 
                           local_Ninv_sqrt_list, 
                           local_mu_list=None, 
                           tol=1e-10, 
                           min_iter=5, 
                           max_iter=100, 
                           solver=cg,
                           p_init=None
                           ):
    """
    Iteratively solves for p using GLS with heteroskedastic noise.
    """

    dim = len(local_U_list)
    assert dim==len(local_Ninv_sqrt_list) and dim==len(local_d_list), \
        "U, d and N must have the same length if they are lists."
    

    n_params = local_U_list[0].shape[-1]

    if p_init is None:
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
    else:
        p = p_init
    
    mpiutil.barrier()
    
    for iteration in range(1, max_iter+1):

        UTSigma_invU, UTSigma_invD = \
            params_space_oper_and_data_list(local_d_list, local_U_list, p, 
                                            local_Ninv_sqrt_list, local_mu_list, draw=False)
                                            
        if rank==1:
            # Calculate the rank of the matrix
            print(f"Rank of U^T Σ_ε⁻¹ U is {np.linalg.matrix_rank(UTSigma_invU)}.")
            # Evaluate the condition number
            print(f"Condition number of U^T Σ_ε⁻¹ U is {np.linalg.cond(UTSigma_invU)}.")

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
            if mpiutil.rank0:
                print(f"Reached max iterations with fractional norm error {frac_norm_error}.")
            break
        
        p = p_new

    # Compute final covariance (and other products) for posterior sampling
    UT_Sigma_U, UT_Sigma_D, draw_vec = \
            params_space_oper_and_data_list(local_d_list, local_U_list, p_new, 
                                            local_Ninv_sqrt_list, local_mu_list, draw=True)

    return p_new, UT_Sigma_U, UT_Sigma_D, draw_vec 



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
                solver=cg,
                Est_mode=True
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
    if Est_mode:
        right_vec = b
    else:
        right_vec = b + A_sqrt_wn

    
    if prior_cov_inv is not None:
        assert prior_mean is not None, "Prior mean must be provided if prior covariance is provided.."
        # if prior_cov_inv is 1D, then diagonalize it
        if prior_cov_inv.ndim == 1:
            left_op += np.diag(prior_cov_inv)
            if Est_mode:
                right_vec += prior_cov_inv * prior_mean
            else:
                # right_vec += prior_cov_inv * prior_mean + np.diag(np.sqrt(prior_cov_inv)) @ np.random.normal(size=prior_mean.shape)
                right_vec += prior_cov_inv * prior_mean + np.sqrt(prior_cov_inv) * np.random.normal(size=prior_mean.shape)
        else:
            left_op += prior_cov_inv
            if Est_mode:
                right_vec += prior_cov_inv @ prior_mean
            else:
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
    """
    If num_samples==0, no sampling, return the GLS estimate instead.
    """
    A = U.T @ Sigma_inv @ U
    aux = U.T @ Sigma_inv @ (d-mu)
    Sigma_sqrt_inv = U.T @ cholesky(Sigma_inv, upper=False) # Careful with the convention of cholesky decomposition!
    num_d = len(d)
    _, n_modes = U.shape
    if num_samples==0:
        p_samples = np.zeros(n_modes)
    else:
        p_samples = np.zeros((num_samples, n_modes))

    if prior_cov_inv is not None:
        assert prior_mean is not None, "Prior mean must be provided if prior covariance is provided.."
        if prior_cov_inv.ndim == 1:
            A += np.diag(prior_cov_inv)
            prior_cov_sqrt_inv = np.diag(np.sqrt(prior_cov_inv))
        else:
            A += prior_cov_inv
            prior_cov_sqrt_inv = cholesky(prior_cov_inv, upper=False) # Careful with the convention of cholesky decomposition!
        aux += prior_cov_inv @ prior_mean 

    if solver is None:
        if num_samples==0:
            p_samples = solve(A, aux, assume_a='sym')
        else:
            for i in range(num_samples):
                b = aux + Sigma_sqrt_inv @ np.random.randn(num_d) 
                if prior_cov_inv is not None:
                    b += prior_cov_sqrt_inv @ np.random.normal(0, 1, n_modes)
                p_samples[i,:] = solve(A, b, assume_a='sym')   
    else:
        if num_samples==0:
            p_samples = solver(A, aux)
        else:
            for i in range(num_samples):
                b = aux + Sigma_sqrt_inv @ np.random.randn(num_d) 
                if prior_cov_inv is not None:
                    b += prior_cov_sqrt_inv @ np.random.normal(0, 1, n_modes)
                p_samples[i,:] = solver(A, b)

    if num_samples==1:
        return p_samples[0]
    else:
        return p_samples



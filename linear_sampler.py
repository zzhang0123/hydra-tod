import numpy as np
from scipy.linalg import solve
from numpy.linalg import cholesky
from scipy.linalg import block_diag


def params_space_oper_and_data(d, U, p, N_inv, mu=0.0):
    D_p_inv = 1./(U @ p + mu)
    sigma_inv = N_inv * np.outer(D_p_inv, D_p_inv)
    aux = U.T @ sigma_inv
    return aux@U, aux@(d-mu)


def iterative_gls(d, U, N_inv, mu=0.0, tol=1e-10, min_iter=5, max_iter=100, solver=None):
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
    return p, Sigma_inv


def iterative_gls_list(U, d, N, tol=1e-10, min_iter=5, max_iter=100, solver=None):
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
            num_samples=1, 
            mu=0.0,
            prior_cov_inv=None, 
            prior_mean=None,
            solver=None
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
    aux = Sigma_inv @ (d-mu)
    Sigma_sqrt_inv = cholesky(Sigma_inv, upper=True) # Careful with the convention of cholesky decomposition!
    num_d = len(d)
    _, n_modes = U.shape
    p_samples = np.zeros((num_samples, n_modes))

    if prior_cov_inv is not None:
        assert prior_mean is not None, "Prior mean must be provided if prior covariance is provided.."
        A += prior_cov_inv
        prior_cov_sqrt_inv = cholesky(prior_cov_inv, upper=True) # Careful with the convention of cholesky decomposition!
        omega_g = np.random.randn(n_modes)
        aux += prior_cov_inv @ prior_mean + prior_cov_sqrt_inv @ omega_g

    if solver is None:
        for i in range(num_samples):
            b = aux + Sigma_sqrt_inv @ np.random.randn(num_d)
            b = U.T @ b
            p_samples[i,:] = solve(A, b, assume_a='sym')   
    else:
        for i in range(num_samples):
            b = aux + Sigma_sqrt_inv @ np.random.randn(num_d)
            b = U.T @ b
            p_samples[i,:] = solver(A, b)

    return p_samples
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray
from . import mpiutil
from scipy.linalg import solve
from numpy.linalg import cholesky
from .linear_solver import cg
from mpi4py import MPI

if TYPE_CHECKING:
    pass

comm = mpiutil.world
rank = mpiutil.rank
rank0 = rank == 0


def params_space_oper_and_data(
    d: NDArray[np.floating],
    U: NDArray[np.floating],
    p: NDArray[np.floating],
    N_inv: NDArray[np.floating],
    mu: float | NDArray[np.floating] = 0.0,
    Ninv_sqrt: NDArray[np.floating] | None = None,
) -> (
    tuple[NDArray[np.floating], NDArray[np.floating]]
    | tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
):
    """
    Project the data model into parameter space for a heteroskedastic GLS.

    Given the data model ``d = (U p + mu)(1 + n)`` where ``n`` has covariance
    ``N``, this function constructs the parameter-space normal equations:

    * ``A = U^T Sigma_inv U``
    * ``b = U^T Sigma_inv (d - mu)``

    where ``Sigma_inv = diag(1/(Up+mu)) N_inv diag(1/(Up+mu))``.

    Parameters
    ----------
    d : NDArray[np.floating]
        Observed data vector of shape ``(M,)``.
    U : NDArray[np.floating]
        Projection (design) matrix of shape ``(M, N)``.
    p : NDArray[np.floating]
        Current parameter estimate of shape ``(N,)``.
    N_inv : NDArray[np.floating]
        Inverse noise covariance, shape ``(M, M)``.
    mu : float or NDArray[np.floating], optional
        Offset / mean term. Default is 0.0.
    Ninv_sqrt : NDArray[np.floating] or None, optional
        Square root of the inverse noise covariance (lower Cholesky factor),
        shape ``(M, M)``.  If provided, the function also returns
        ``U^T Sigma_inv_sqrt`` for sampling.

    Returns
    -------
    UTSigmaU : NDArray[np.floating]
        Parameter-space operator ``U^T Sigma_inv U``, shape ``(N, N)``.
    UTSigmaD : NDArray[np.floating]
        Parameter-space data vector ``U^T Sigma_inv (d - mu)``, shape ``(N,)``.
    UTSigma_sqrt : NDArray[np.floating], optional
        Returned only when ``Ninv_sqrt`` is not None.  Product
        ``U^T Sigma_inv_sqrt``, shape ``(N, M)``.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    D_p_inv = 1.0 / (U @ p + mu)
    sigma_inv = N_inv * np.outer(D_p_inv, D_p_inv)
    aux = U.T @ sigma_inv
    if Ninv_sqrt is None:
        return aux @ U, aux @ (d - mu)
    else:
        sigma_inv_sqrt = D_p_inv[:, np.newaxis] * Ninv_sqrt
        return aux @ U, aux @ (d - mu), U.T @ sigma_inv_sqrt


def iterative_gls(
    d: NDArray[np.floating],
    U: NDArray[np.floating],
    N_inv: NDArray[np.floating],
    mu: float | NDArray[np.floating] = 0.0,
    tol: float = 1e-10,
    min_iter: int = 5,
    max_iter: int = 100,
    solver: Callable[..., NDArray[np.floating]] | None = cg,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Iteratively solve for parameters using GLS with heteroskedastic noise.

    Implements the iteratively-reweighted least squares (IRLS) algorithm for
    the multiplicative noise model ``d = (U p + mu)(1 + n)``, where ``n``
    has covariance ``N``.

    Parameters
    ----------
    d : NDArray[np.floating]
        Observed data vector of shape ``(M,)``.
    U : NDArray[np.floating]
        Projection matrix of shape ``(M, N)``.
    N_inv : NDArray[np.floating]
        Inverse noise covariance of shape ``(M, M)``.
    mu : float or NDArray[np.floating], optional
        Offset term in the data model. Default is 0.0.
    tol : float, optional
        Relative convergence tolerance on the parameter norm. Default is 1e-10.
    min_iter : int, optional
        Minimum number of iterations before convergence is checked.
        Default is 5.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.
    solver : callable or None, optional
        Linear solver function with signature ``solver(A, b) -> x``.
        If ``None``, uses ``scipy.linalg.solve``. Default is :func:`cg`.

    Returns
    -------
    p_gls : NDArray[np.floating]
        Estimated parameter vector of shape ``(N,)``.
    Sigma_inv : NDArray[np.floating]
        Final inverse covariance matrix of shape ``(M, M)``, evaluated at
        the converged parameter estimate.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """

    # Initialize p using ordinary least squares (OLS) as a starting point
    p = np.linalg.lstsq(U, d - mu, rcond=None)[0]

    for iteration in range(1, max_iter + 1):
        # Compute noise covariance Sigma_epsilon = diag(U p) N diag(U p)
        # Sigma_inv = np.linalg.inv(Sigma_epsilon)

        # Solve GLS: (U^T Sigma_inv U) p = U^T Sigma_inv d
        UTSigma_invU, UTSigma_invD = params_space_oper_and_data(d, U, p, N_inv, mu=mu)

        if solver is None:
            p_new = solve(UTSigma_invU, UTSigma_invD, assume_a="sym")
        else:
            p_new = solver(UTSigma_invU, UTSigma_invD)

        # Check for convergence
        if (
            np.linalg.norm(p_new - p) < tol * np.linalg.norm(p)
            and iteration >= min_iter
        ):
            print(f"Converged in {iteration} iterations.")
            break

        p = p_new

    # Compute final covariance for posterior sampling
    # covariance_p = np.linalg.inv(U.T @ Sigma_inv @ U)
    D_p_inv = 1.0 / (U @ p + mu)
    Sigma_inv = N_inv * np.outer(D_p_inv, D_p_inv)
    return p_new, Sigma_inv


def params_space_oper_and_data_list(
    d_list: list[NDArray[np.floating]],
    U_list: list[NDArray[np.floating]],
    p: NDArray[np.floating],
    Ninv_sqrt_list: list[NDArray[np.floating]],
    mu_list: list[float | NDArray[np.floating]] | None = None,
    draw: bool = True,
    root: int | None = None,
) -> (
    tuple[NDArray[np.floating] | None, NDArray[np.floating] | None]
    | tuple[
        NDArray[np.floating] | None,
        NDArray[np.floating] | None,
        NDArray[np.floating] | None,
    ]
):
    """
    Build parameter-space operator and data vector from multiple TODs via MPI.

    Accumulates the contributions from each local TOD into the parameter-space
    normal equations and reduces across MPI ranks.  Optionally generates a
    white-noise draw for Gibbs sampling.

    Parameters
    ----------
    d_list : list of NDArray[np.floating]
        Local data vectors, one per TOD on this rank.
    U_list : list of NDArray[np.floating]
        Local projection matrices, one per TOD.
    p : NDArray[np.floating]
        Current shared parameter estimate of shape ``(N,)``.
    Ninv_sqrt_list : list of NDArray[np.floating]
        Square root of inverse noise covariance for each local TOD.
    mu_list : list of float or NDArray or None, optional
        Offset terms for each local TOD. Default is ``None`` (zero offset).
    draw : bool, optional
        If ``True``, also compute a white-noise contribution for Gibbs
        sampling. Default is ``True``.
    root : int or None, optional
        If specified, only that rank receives the reduced result (via
        ``Reduce``).  If ``None``, all ranks receive the result (via
        ``Allreduce``).

    Returns
    -------
    UT_sigma_U : NDArray[np.floating] or None
        Reduced parameter-space operator, shape ``(N, N)``.
    UT_sigma_d : NDArray[np.floating] or None
        Reduced parameter-space data vector, shape ``(N,)``.
    UT_sigma_sqrt_wn : NDArray[np.floating] or None
        White-noise draw projected into parameter space, shape ``(N,)``.
        Only returned when ``draw=True``.

    Notes
    -----
    Non-root ranks return ``None`` values when ``root`` is specified.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """

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
            combined_matrix[n_params + 1, :] += aux @ wn

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
        UT_sigma_sqrt_wn = global_combined_matrix[n_params + 1, :]
        return UT_sigma_U, UT_sigma_d, UT_sigma_sqrt_wn
    else:
        return UT_sigma_U, UT_sigma_d


def iterative_gls_mpi_list(
    local_d_list: list[NDArray[np.floating]],
    local_U_list: list[NDArray[np.floating]],
    local_Ninv_sqrt_list: list[NDArray[np.floating]],
    local_mu_list: list[float | NDArray[np.floating]] | None = None,
    tol: float = 1e-10,
    min_iter: int = 5,
    max_iter: int = 100,
    solver: Callable[..., NDArray[np.floating]] | None = cg,
    p_init: NDArray[np.floating] | None = None,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating] | None,
    NDArray[np.floating] | None,
    NDArray[np.floating] | None,
]:
    """
    MPI-parallel iteratively-reweighted GLS for multiple TODs.

    Each MPI rank holds a local subset of TODs.  The algorithm iteratively
    solves for the shared parameter vector ``p`` by reducing contributions
    from all ranks, and upon convergence also produces a white-noise draw
    for subsequent Gibbs sampling.

    Parameters
    ----------
    local_d_list : list of NDArray[np.floating]
        Data vectors local to this MPI rank.
    local_U_list : list of NDArray[np.floating]
        Projection matrices local to this MPI rank.
    local_Ninv_sqrt_list : list of NDArray[np.floating]
        Square root of inverse noise covariance for each local TOD.
    local_mu_list : list of float or NDArray or None, optional
        Offset terms for each local TOD. Default is ``None``.
    tol : float, optional
        Relative convergence tolerance. Default is 1e-10.
    min_iter : int, optional
        Minimum number of iterations. Default is 5.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.
    solver : callable or None, optional
        Linear solver function.  If ``None``, uses ``scipy.linalg.solve``.
        Default is :func:`cg`.
    p_init : NDArray[np.floating] or None, optional
        Initial parameter estimate.  If ``None``, an OLS initialisation is
        performed via MPI reduction.

    Returns
    -------
    p_new : NDArray[np.floating]
        Converged parameter estimate, shape ``(N,)``.
    UT_Sigma_U : NDArray[np.floating] or None
        Final parameter-space operator (root rank only when
        ``root`` is specified inside sub-calls).
    UT_Sigma_D : NDArray[np.floating] or None
        Final parameter-space data vector.
    draw_vec : NDArray[np.floating] or None
        White-noise draw for Gibbs sampling.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """

    dim = len(local_U_list)
    assert dim == len(local_Ninv_sqrt_list) and dim == len(
        local_d_list
    ), "U, d and N must have the same length if they are lists."

    n_params = local_U_list[0].shape[-1]

    if p_init is None:
        local_UT_U = np.zeros((n_params, n_params), dtype=np.float64)
        local_UT_d = np.zeros(n_params, dtype=np.float64)
        # local_Ninv_sqrt_list = []
        for i in range(dim):
            Ui = local_U_list[i]
            di = local_d_list[i]
            local_UT_U += Ui.T @ Ui
            local_UT_d += Ui.T @ di
            # local_Ninv_sqrt_list.append(cholesky(local_N_inv_list[i], upper=False))

        # Ensure NumPy float64, contiguous buffers for MPI
        local_UT_U = np.ascontiguousarray(np.asarray(local_UT_U, dtype=np.float64))
        local_UT_d = np.ascontiguousarray(np.asarray(local_UT_d, dtype=np.float64))

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
            p = np.empty(
                n_params, dtype=np.float64
            )  # Initialize buffer on all processes
        # broadcast p to all processes
        comm.Bcast(p, root=0)
    else:
        p = p_init

    mpiutil.barrier()

    for iteration in range(1, max_iter + 1):

        UTSigma_invU, UTSigma_invD = params_space_oper_and_data_list(
            local_d_list,
            local_U_list,
            p,
            local_Ninv_sqrt_list,
            local_mu_list,
            draw=False,
        )

        # if rank==1:
        #     # Calculate the rank of the matrix
        #     print(f"Rank of U^T Sigma_inv U is {np.linalg.matrix_rank(UTSigma_invU)}.")
        #     # Evaluate the condition number
        #     print(f"Condition number of U^T Sigma_inv U is {np.linalg.cond(UTSigma_invU)}.")

        if mpiutil.rank0:
            if solver is None:
                p_new = solve(UTSigma_invU, UTSigma_invD, assume_a="sym")
                # set the dtype of p_new to float64
                p_new = p_new.astype(np.float64)
            else:
                p_new = solver(UTSigma_invU, UTSigma_invD)
                p_new = p_new.astype(np.float64)
        else:
            p_new = np.empty(
                n_params, dtype=np.float64
            )  # Initialize buffer on all processes

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
                print(
                    f"Reached max iterations with fractional norm error {frac_norm_error}."
                )
            break

        p = p_new

    # Compute final covariance (and other products) for posterior sampling
    UT_Sigma_U, UT_Sigma_D, draw_vec = params_space_oper_and_data_list(
        local_d_list,
        local_U_list,
        p_new,
        local_Ninv_sqrt_list,
        local_mu_list,
        draw=True,
    )

    return p_new, UT_Sigma_U, UT_Sigma_D, draw_vec


def sample_p(
    d: NDArray[np.floating],
    U: NDArray[np.floating],
    Sigma_inv: NDArray[np.floating],
    mu: float | NDArray[np.floating] = 0.0,
    num_samples: int = 1,
    prior_cov_inv: NDArray[np.floating] | None = None,
    prior_mean: NDArray[np.floating] | None = None,
    solver: Callable[..., NDArray[np.floating]] | None = cg,
    p_GLS: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """
    Draw samples from the posterior (or likelihood) distribution of parameters.

    Constructs the normal equations ``A p = b`` from the data likelihood and
    optional Gaussian prior, then draws from the resulting multivariate
    Gaussian posterior.

    Parameters
    ----------
    d : NDArray[np.floating]
        Observed data vector of shape ``(M,)``.
    U : NDArray[np.floating]
        Projection matrix of shape ``(M, N)``.
    Sigma_inv : NDArray[np.floating]
        Inverse noise covariance of shape ``(M, M)``.
    mu : float or NDArray[np.floating], optional
        Offset term. Default is 0.0.
    num_samples : int, optional
        Number of samples to draw.  If 0, returns the MAP / GLS estimate
        without sampling. Default is 1.
    prior_cov_inv : NDArray[np.floating] or None, optional
        Inverse of the prior covariance matrix, shape ``(N, N)``.
    prior_mean : NDArray[np.floating] or None, optional
        Prior mean vector, shape ``(N,)``.  Required if ``prior_cov_inv``
        is provided.
    solver : callable or None, optional
        Linear solver function. Default is :func:`cg`.
    p_GLS : NDArray[np.floating] or None, optional
        Pre-computed GLS estimate to avoid redundant solves.

    Returns
    -------
    samples : NDArray[np.floating]
        If ``num_samples > 1``: array of shape ``(num_samples, N)``.
        If ``num_samples == 1``: array of shape ``(N,)``.
        If ``num_samples == 0``: MAP estimate of shape ``(N,)``.

    Notes
    -----
    If the covariance matrix is not positive definite (e.g. due to numerical
    issues), the function attempts regularisation with progressively larger
    diagonal perturbations.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    A = U.T @ Sigma_inv @ U
    b = U.T @ Sigma_inv @ (d - mu)

    if prior_cov_inv is not None:
        assert (
            prior_mean is not None
        ), "Prior mean must be provided if prior covariance is provided.."
        A += prior_cov_inv
        b += prior_cov_inv @ prior_mean

        if solver is None:
            p_est = solve(A, b, assume_a="sym")
        else:
            p_est = solver(A, b)

    elif p_GLS is not None:  # No prior, and p_GLS estimation is provided
        p_est = p_GLS
    else:  # No prior, and p_GLS estimation is not provided - solve for it
        p_est = solve(A, b, assume_a="sym")

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


def sample_p_v2(
    A: NDArray[np.floating],
    b: NDArray[np.floating],
    A_sqrt_wn: NDArray[np.floating],
    prior_cov_inv: NDArray[np.floating] | None = None,
    prior_mean: NDArray[np.floating] | None = None,
    solver: Callable[..., NDArray[np.floating]] | None = cg,
    Est_mode: bool = True,
) -> NDArray[np.floating]:
    """
    Draw a sample from the posterior using pre-computed normal-equation terms.

    This is the preferred sampling interface for the Gibbs sampler, as it
    avoids redundant matrix constructions.  In estimation mode, it returns
    the MAP estimate; in sampling mode, it adds a white-noise realisation
    to the right-hand side before solving.

    Parameters
    ----------
    A : NDArray[np.floating]
        Left-hand side of the normal equations, shape ``(N, N)``.
    b : NDArray[np.floating]
        Right-hand side data vector, shape ``(N,)``.
    A_sqrt_wn : NDArray[np.floating]
        White-noise term ``U^T N^{-1/2} w`` where ``w ~ N(0, I)``,
        shape ``(N,)``.
    prior_cov_inv : NDArray[np.floating] or None, optional
        Inverse prior covariance.  Can be 1-D (diagonal) or 2-D (full
        matrix).
    prior_mean : NDArray[np.floating] or None, optional
        Prior mean vector.  Required if ``prior_cov_inv`` is provided.
    solver : callable or None, optional
        Linear solver function. Default is :func:`cg`.
    Est_mode : bool, optional
        If ``True``, return the MAP estimate (no noise added).  If
        ``False``, draw one posterior sample. Default is ``True``.

    Returns
    -------
    p_sample : NDArray[np.floating]
        A single parameter sample (or MAP estimate), shape ``(N,)``.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    left_op = A
    if Est_mode:
        right_vec = b
    else:
        right_vec = b + A_sqrt_wn

    if prior_cov_inv is not None:
        assert (
            prior_mean is not None
        ), "Prior mean must be provided if prior covariance is provided.."
        # if prior_cov_inv is 1D, then diagonalize it
        if prior_cov_inv.ndim == 1:
            left_op += np.diag(prior_cov_inv)
            if Est_mode:
                right_vec += prior_cov_inv * prior_mean
            else:
                # right_vec += prior_cov_inv * prior_mean + np.diag(np.sqrt(prior_cov_inv)) @ np.random.normal(size=prior_mean.shape)
                right_vec += prior_cov_inv * prior_mean + np.sqrt(
                    prior_cov_inv
                ) * np.random.normal(size=prior_mean.shape)
        else:
            left_op += prior_cov_inv
            if Est_mode:
                right_vec += prior_cov_inv @ prior_mean
            else:
                right_vec += prior_cov_inv @ prior_mean + cholesky(
                    prior_cov_inv
                ) @ np.random.normal(size=prior_mean.shape)

    if solver is None:
        p_sample = solve(left_op, right_vec, assume_a="sym")
    else:
        p_sample = solver(left_op, right_vec)

    return p_sample


def sample_p_old(
    d: NDArray[np.floating],
    U: NDArray[np.floating],
    Sigma_inv: NDArray[np.floating],
    num_samples: int = 1,
    mu: float | NDArray[np.floating] = 0.0,
    prior_cov_inv: NDArray[np.floating] | None = None,
    prior_mean: NDArray[np.floating] | None = None,
    solver: Callable[..., NDArray[np.floating]] | None = cg,
) -> NDArray[np.floating]:
    """
    Draw posterior samples using the explicit Cholesky-based sampling method.

    This is an older implementation kept for reference and validation.  For
    each sample, a white-noise realisation is projected through the square
    root of the inverse covariance and added to the right-hand side before
    solving.

    Parameters
    ----------
    d : NDArray[np.floating]
        Observed data vector of shape ``(M,)``.
    U : NDArray[np.floating]
        Projection matrix of shape ``(M, N)``.
    Sigma_inv : NDArray[np.floating]
        Inverse noise covariance of shape ``(M, M)``.
    num_samples : int, optional
        Number of samples to draw.  If 0, returns the GLS/MAP estimate.
        Default is 1.
    mu : float or NDArray[np.floating], optional
        Offset term. Default is 0.0.
    prior_cov_inv : NDArray[np.floating] or None, optional
        Inverse prior covariance.  Can be 1-D (diagonal) or 2-D.
    prior_mean : NDArray[np.floating] or None, optional
        Prior mean vector.
    solver : callable or None, optional
        Linear solver function. Default is :func:`cg`.

    Returns
    -------
    p_samples : NDArray[np.floating]
        If ``num_samples > 1``: array of shape ``(num_samples, N)``.
        If ``num_samples == 1``: array of shape ``(N,)``.
        If ``num_samples == 0``: MAP estimate of shape ``(N,)``.

    Notes
    -----
    Superseded by :func:`sample_p_v2` which is more efficient for the Gibbs
    sampler.  Kept for backward compatibility and validation.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    A = U.T @ Sigma_inv @ U
    aux = U.T @ Sigma_inv @ (d - mu)
    Sigma_sqrt_inv = U.T @ cholesky(
        Sigma_inv, upper=False
    )  # Careful with the convention of cholesky decomposition!
    num_d = len(d)
    _, n_modes = U.shape
    if num_samples == 0:
        p_samples = np.zeros(n_modes)
    else:
        p_samples = np.zeros((num_samples, n_modes))

    if prior_cov_inv is not None:
        assert (
            prior_mean is not None
        ), "Prior mean must be provided if prior covariance is provided.."
        if prior_cov_inv.ndim == 1:
            A += np.diag(prior_cov_inv)
            prior_cov_sqrt_inv = np.diag(np.sqrt(prior_cov_inv))
        else:
            A += prior_cov_inv
            prior_cov_sqrt_inv = cholesky(
                prior_cov_inv, upper=False
            )  # Careful with the convention of cholesky decomposition!
        aux += prior_cov_inv @ prior_mean

    if solver is None:
        if num_samples == 0:
            p_samples = solve(A, aux, assume_a="sym")
        else:
            for i in range(num_samples):
                b = aux + Sigma_sqrt_inv @ np.random.randn(num_d)
                if prior_cov_inv is not None:
                    b += prior_cov_sqrt_inv @ np.random.normal(0, 1, n_modes)
                p_samples[i, :] = solve(A, b, assume_a="sym")
    else:
        if num_samples == 0:
            p_samples = solver(A, aux)
        else:
            for i in range(num_samples):
                b = aux + Sigma_sqrt_inv @ np.random.randn(num_d)
                if prior_cov_inv is not None:
                    b += prior_cov_sqrt_inv @ np.random.normal(0, 1, n_modes)
                p_samples[i, :] = solver(A, b)

    if num_samples == 1:
        return p_samples[0]
    else:
        return p_samples

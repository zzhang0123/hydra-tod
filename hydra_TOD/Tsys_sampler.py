"""Gibbs-sampling step for system-temperature (sky + local) parameters.

This module draws posterior samples of the system-temperature coefficient
vector conditioned on the current gain and noise parameters.  It handles
both single-TOD and MPI-distributed multi-TOD inference.

The two main entry points are:

:func:`Tsys_coeff_sampler`
    Single-TOD sampler.  Takes one TOD, its gain vector, and the combined
    sky + local projection matrix, and returns a posterior sample of the
    coefficient vector :math:`\\mathbf{p}` in
    :math:`\\mathbf{T}_{\\rm sys} = \\mathbf{A}\\,\\mathbf{p}`.

:func:`Tsys_sampler_multi_TODs`
    Multi-TOD MPI sampler.  Each MPI rank holds a list of TODs; the
    function accumulates normal-equation contributions via ``MPI_Allreduce``
    and jointly samples the shared sky parameters while keeping local
    temperature parameters rank-local.

Both functions use the iterative GLS scheme from
:mod:`~hydra_tod.linear_sampler` to handle the heteroskedastic
multiplicative noise model (noise variance scales with
:math:`T_{\\rm sys}^2`).

Key parameters
--------------
``noise_params``
    Tuple ``(logf0, alpha)`` or ``(logf0, logfc, alpha)``; the function
    extracts the required elements automatically.
``Est_mode``
    If ``True``, return the MAP estimate instead of drawing a sample
    (faster; useful for burn-in or debugging).
``prior_cov_inv``
    ``1-D`` array for diagonal prior, or ``2-D`` for full covariance.
    Pass ``None`` for an uninformative prior.

Typical usage
-------------
.. code-block:: python

    from hydra_tod.tsys_sampler import Tsys_sampler_multi_TODs

    Tsys_params = Tsys_sampler_multi_TODs(
        local_data_list=[tod1, tod2],
        local_t_list=[t1, t2],
        local_gain_list=[gains1, gains2],
        local_Tsys_proj_list=[proj1, proj2],
        local_Noise_params_list=[(logf0, alpha), (logf0, alpha)],
        local_logfc_list=[logfc, logfc],
        prior_cov_inv=prior_cov_inv_tsys,
        prior_mean=prior_mean_tsys,
    )

See Also
--------
hydra_tod.gain_sampler : Gain Gibbs step.
hydra_tod.linear_sampler : Underlying GLS and Gaussian sampling machinery.
hydra_tod.full_Gibbs_sampler : Orchestrates all Gibbs steps.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable
from . import mpiutil
from .flicker_model import flicker_cov
from .linear_solver import cg
from .linear_sampler import (
    iterative_gls,
    iterative_gls_mpi_list,
    sample_p_old,
    sample_p_v2,
)
from .utils import cho_compute_mat_inv, cho_compute_mat_inv_sqrt

comm = mpiutil.world
rank0 = mpiutil.rank0


def Tsys_coeff_sampler(
    data: NDArray[np.floating],
    t_list: NDArray[np.floating],
    gain: NDArray[np.floating],
    Tsys_proj: NDArray[np.floating],
    noise_params: tuple[float, ...],
    logfc: float | None = None,
    wnoise_var: float = 2.5e-6,
    n_samples: int = 1,
    mu: float = 0.0,
    tol: float = 1e-13,
    prior_cov_inv: NDArray[np.floating] | None = None,
    prior_mean: NDArray[np.floating] | None = None,
    solver: Callable = cg,
) -> NDArray[np.float64]:
    r"""Sample system-temperature basis coefficients for a single TOD.

    Draws a posterior sample of the coefficient vector :math:`\mathbf{p}`
    in the linear model:

    .. math::

        \frac{\mathrm{TOD}}{g(t)} = \mathbf{A}\,\mathbf{p} + \mathbf{n}

    where :math:`\mathbf{A}` is the system-temperature projection matrix
    (combining sky and local components), :math:`g(t)` is the gain, and
    :math:`\mathbf{n}` is correlated flicker noise.

    The posterior is Gaussian-conjugate:

    .. math::

        \mathbf{p} \mid \mathrm{data} \sim \mathcal{N}\!\bigl(
            \boldsymbol{\mu}_{\mathrm{post}},\,
            \boldsymbol{\Sigma}_{\mathrm{post}}\bigr)

    Parameters
    ----------
    data : NDArray[np.floating]
        Time-ordered data vector of length :math:`N_t`.
    t_list : NDArray[np.floating]
        Observation time stamps (seconds).
    gain : NDArray[np.floating]
        Gain vector :math:`g(t)` of length :math:`N_t`.
    Tsys_proj : NDArray[np.floating]
        System-temperature projection (design) matrix :math:`\mathbf{A}`,
        shape ``(N_t, N_p)``. Columns correspond to sky pixels and local
        temperature components.
    noise_params : tuple[float, ...]
        Noise parameters. If *logfc* is ``None``: ``(logf0, logfc, alpha)``.
        Otherwise: ``(logf0, alpha)``.
    logfc : float or None, optional
        Log10 of the cutoff angular frequency. If ``None``, it is extracted
        from *noise_params*. Default is ``None``.
    wnoise_var : float, optional
        White-noise variance :math:`\sigma_w^2`. Default is ``2.5e-6``.
    n_samples : int, optional
        Number of posterior samples. Default is ``1``.
    mu : float, optional
        Prior mean offset. Default is ``0.0``.
    tol : float, optional
        Tolerance for the iterative GLS solver. Default is ``1e-13``.
    prior_cov_inv : NDArray[np.floating] or None, optional
        Inverse prior covariance matrix (or 1-D diagonal). If ``None``,
        an uninformative prior is used.
    prior_mean : NDArray[np.floating] or None, optional
        Prior mean vector. If ``None``, zero mean is assumed.
    solver : Callable, optional
        Linear solver function. Default is :func:`hydra_tod.linear_solver.cg`.

    Returns
    -------
    NDArray[np.float64]
        Posterior sample(s) of system-temperature coefficients
        :math:`\mathbf{p}`.

    Notes
    -----
    The noise covariance is constructed from the flicker-noise model via
    :func:`hydra_tod.flicker_model.flicker_cov` and inverted using
    Cholesky decomposition.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """

    d_vec = data / gain
    if logfc is None:
        logf0, logfc, alpha = noise_params
    else:
        logf0, alpha = noise_params
    Ncov_inv = cho_compute_mat_inv(
        flicker_cov(
            t_list,
            10.0**logf0,
            10.0**logfc,
            alpha,
            white_n_variance=wnoise_var,
            only_row_0=False,
        )
    )

    p_GLS, sigma_inv = iterative_gls(d_vec, Tsys_proj, Ncov_inv, mu=mu, tol=tol)
    return sample_p_old(
        d_vec,
        Tsys_proj,
        sigma_inv,
        num_samples=n_samples,
        mu=mu,
        prior_cov_inv=prior_cov_inv,
        prior_mean=prior_mean,
        solver=solver,
    )


def Tsys_sampler_multi_TODs(
    local_data_list: list[NDArray[np.floating]],
    local_t_list: list[NDArray[np.floating]],
    local_gain_list: list[NDArray[np.floating]],
    local_Tsys_proj_list: list[NDArray[np.floating]],
    local_Noise_params_list: list[tuple[float, float]],
    local_logfc_list: list[float],
    local_mu_list: list[float] | None = None,
    wnoise_var: float = 2.5e-6,
    tol: float = 1e-13,
    prior_cov_inv: NDArray[np.floating] | None = None,
    prior_mean: NDArray[np.floating] | None = None,
    solver: Callable = cg,
    init_coeffs: NDArray[np.floating] | None = None,
    Est_mode: bool = False,
) -> NDArray[np.float64]:
    r"""Jointly sample system-temperature coefficients across multiple TODs.

    Combines data from all TODs assigned to the local MPI rank to form a
    joint linear system and draws a posterior sample of the shared
    coefficient vector :math:`\mathbf{p}`. The normal equations from each
    TOD are accumulated via MPI collective operations so that the sky
    temperature parameters (:math:`T_{\mathrm{sky}}`) are shared across
    all ranks while local temperature components (:math:`T_{\mathrm{loc}}`)
    remain per-TOD.

    The joint posterior is:

    .. math::

        \mathbf{p} \mid \{\mathrm{data}_i\} \sim \mathcal{N}\!\bigl(
            \boldsymbol{\mu}_{\mathrm{post}},\,
            \boldsymbol{\Sigma}_{\mathrm{post}}\bigr)

    where the posterior precision and mean are aggregated over all TODs.

    Parameters
    ----------
    local_data_list : list[NDArray[np.floating]]
        List of TOD vectors assigned to this MPI rank.
    local_t_list : list[NDArray[np.floating]]
        List of time-stamp arrays, one per local TOD.
    local_gain_list : list[NDArray[np.floating]]
        List of gain vectors, one per local TOD.
    local_Tsys_proj_list : list[NDArray[np.floating]]
        List of system-temperature projection matrices, one per local TOD.
        Each matrix has shape ``(N_t_i, N_p)`` where :math:`N_p` is the
        total number of shared + local parameters.
    local_Noise_params_list : list[tuple[float, float]]
        List of noise parameter tuples ``(logf0, alpha)`` for each local
        TOD.
    local_logfc_list : list[float]
        List of :math:`\log_{10}\omega_c` values for each local TOD.
    local_mu_list : list[float] or None, optional
        List of prior mean offsets for each local TOD. If ``None``, zeros
        are used.
    wnoise_var : float, optional
        White-noise variance :math:`\sigma_w^2`. Default is ``2.5e-6``.
    tol : float, optional
        Tolerance for the iterative GLS solver. Default is ``1e-13``.
    prior_cov_inv : NDArray[np.floating] or None, optional
        Inverse prior covariance matrix (or 1-D diagonal) for the joint
        parameter vector.
    prior_mean : NDArray[np.floating] or None, optional
        Prior mean vector for the joint parameter vector.
    solver : Callable, optional
        Linear solver function. Default is :func:`hydra_tod.linear_solver.cg`.
    init_coeffs : NDArray[np.floating] or None, optional
        Initial guess for the iterative solver. If ``None``, a default
        starting point is used.
    Est_mode : bool, optional
        If ``True``, return the MAP estimate without adding a random
        fluctuation (faster, deterministic). Default is ``False``.

    Returns
    -------
    NDArray[np.float64]
        Posterior sample (or MAP estimate) of the joint
        system-temperature coefficient vector :math:`\mathbf{p}`,
        broadcast to all MPI ranks.

    Notes
    -----
    The computation is distributed across MPI ranks:

    1. Each rank constructs and inverts the noise covariance for its local
       TODs.
    2. The GLS normal equations are formed locally and reduced across
       ranks via :func:`hydra_tod.linear_sampler.iterative_gls_mpi_list`.
    3. The posterior sample is drawn on rank 0 and broadcast.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    dim = len(local_data_list)
    d_vec_list = [local_data_list[i] / local_gain_list[i] for i in range(dim)]

    local_Ninv_sqrt_list = []
    for di in range(dim):
        logfc = local_logfc_list[di]
        logf0, alpha = local_Noise_params_list[di]
        t_list = local_t_list[di]
        Ncov_inv_sqrt = cho_compute_mat_inv_sqrt(
            flicker_cov(
                t_list,
                10.0**logf0,
                10.0**logfc,
                alpha,
                white_n_variance=wnoise_var,
                only_row_0=False,
            )
        )
        local_Ninv_sqrt_list.append(Ncov_inv_sqrt)

    p_GLS, A, b, Asqrt_wn = iterative_gls_mpi_list(
        d_vec_list,
        local_Tsys_proj_list,
        local_Ninv_sqrt_list,
        local_mu_list=local_mu_list,
        tol=tol,
        p_init=init_coeffs,
    )

    # Compute on rank 0 only to avoid redundant computation
    if mpiutil.rank0:
        p_sample = sample_p_v2(
            A,
            b,
            Asqrt_wn,
            prior_cov_inv=prior_cov_inv,
            prior_mean=prior_mean,
            solver=solver,
            Est_mode=Est_mode,
        )
    else:
        p_sample = None

    # broadcast result to all ranks
    p_sample = comm.bcast(p_sample, root=0)
    return p_sample

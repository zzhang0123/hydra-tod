from __future__ import annotations

# This file contains the full Gibbs sampler for all the parameters in data model,
# including the system temperature parameters, noise parameters, gain parameters.

# Full Gibbs sampler
from typing import Callable, Literal, Optional

import numpy as np
import jax.numpy as jnp
import jax.random as jr
from . import mpiutil
from .gain_sampler import gain_sampler
from .noise_sampler_fixed_fc import flicker_sampler
from .noise_sampler_old import flicker_noise_sampler
from .tsys_sampler import Tsys_coeff_sampler, Tsys_sampler_multi_TODs
from .linear_solver import cg
from scipy.linalg import block_diag
from tqdm import tqdm

comm = mpiutil.world
rank = mpiutil.rank
rank0 = rank == 0

# Default bounds format: [[lower1, upper1], [lower2, upper2], ...]
flicker_bounds = [(-6.0, -3.0), (1.1, 4.0)]
Gflicker_bounds = [[4.0, 10.0], [-6.0, -3.0], [1.1, 4.0]]


def get_Tsys_operator(
    local_Tsky_operator_list: list[np.ndarray],
    local_Tloc_operator_list: list[np.ndarray],
) -> list[np.ndarray]:
    """Construct combined Tsys operator matrix from Tsky and Tloc operators.

    Assembles a block-structured system temperature operator that maps the
    joint parameter vector ``[Tsky_params, Tloc1_params, Tloc2_params, ...]``
    to time-ordered data for each TOD on this MPI rank. The Tsky columns are
    shared across all TODs (and ranks), while the Tloc columns are block-diagonal
    (each TOD has its own local temperature component).

    For example, with two ranks where rank 0 owns TODs 1--2 and rank 1 owns
    TOD 3, the global operator has the structure::

              ( U1  R1   0   0  )
        U  =  ( U2   0  R2   0  )
              ( U3   0   0  R3  )

    where ``U_i`` is the sky operator and ``R_i`` is the local operator for
    TOD *i*. Each rank stores only its own rows.

    The resulting linear system is:

    .. math::
        \\mathbf{U} \\begin{pmatrix} T_{\\rm sky} \\\\ T_{\\rm loc,1} \\\\
        T_{\\rm loc,2} \\\\ T_{\\rm loc,3} \\end{pmatrix}
        + \\mathbf{n} = \\mathbf{d}

    Parameters
    ----------
    local_Tsky_operator_list : list[np.ndarray]
        List of sky temperature projection operators for each TOD on this rank.
        Each element has shape ``(n_time_i, n_sky_modes)``.
    local_Tloc_operator_list : list[np.ndarray]
        List of local temperature projection operators for each TOD on this
        rank. Each element has shape ``(n_time_i, n_loc_modes_i)``.

    Returns
    -------
    list[np.ndarray]
        List of combined Tsys operators for each TOD on this rank. Each element
        has shape ``(n_time_i, n_sky_modes + total_loc_modes)`` where
        ``total_loc_modes`` is the sum of all local mode dimensions across all
        ranks.

    Notes
    -----
    This function uses MPI collective operations (``allgather``) to determine
    the global block structure. All ranks must call this function together.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.

    Examples
    --------
    >>> Tsys_ops = get_Tsys_operator(Tsky_op_list, Tloc_op_list)
    >>> Tsys = Tsys_ops[0] @ np.concatenate([Tsky_params, Tloc_params_all])
    """
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
    local_TOD_list: list[np.ndarray],
    local_t_lists: list[np.ndarray],
    local_gain_operator_list: list[np.ndarray],
    local_Tsky_operator_list: list[np.ndarray],
    local_Tloc_operator_list: list[np.ndarray],
    init_Tsky_params: np.ndarray,
    init_Tloc_params_list: list[np.ndarray],
    init_noise_params_list: list[np.ndarray],
    local_logfc_list: list[float],
    local_Tsys_injection_list: Optional[list[float | np.ndarray]] = None,
    wnoise_var: float = 2.5e-6,
    Tsky_prior_cov_inv: Optional[np.ndarray] = None,
    Tsky_prior_mean: Optional[np.ndarray] = None,
    local_Tloc_prior_cov_inv_list: Optional[list[np.ndarray]] = None,
    local_Tloc_prior_mean_list: Optional[list[np.ndarray]] = None,
    local_gain_prior_cov_inv_list: Optional[list[np.ndarray]] = None,
    local_gain_prior_mean_list: Optional[list[np.ndarray]] = None,
    local_noise_prior_func_list: Optional[list[Callable | None]] = None,
    joint_Tsys_sampling: bool = False,
    smooth_gain_model: Literal["linear", "log", "factorized"] = "linear",
    noise_sampler_type: Literal["emcee", "nuts"] = "emcee",
    noise_Jeffreys_prior: bool = False,
    noise_params_bounds: Optional[list[tuple[float, float]]] = flicker_bounds,
    n_samples: int = 100,
    tol: float = 1e-12,
    linear_solver: Callable = cg,
    root: Optional[int] = None,
    Est_mode: bool = False,
    debug: bool = False,
) -> (
    tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]
    | tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]
):
    """Run the full Gibbs sampler for multi-TOD Bayesian inference.

    This is the main inference engine of the ``hydra_tod`` package. It
    implements a Gibbs sampler that alternates between conditionally sampling
    the gain parameters, system temperature parameters, and noise parameters
    from a hierarchical Bayesian model for radio telescope time-ordered data.

    The data model is:

    .. math::
        d(t) = T_{\\rm sys}(t) \\, [1 + n(t)] \\, g(t)

    where :math:`T_{\\rm sys} = T_{\\rm sky} + T_{\\rm loc}`, :math:`g(t)` is
    the time-varying instrument gain, and :math:`n(t)` is flicker noise with a
    :math:`1/f^\\alpha` power spectrum.

    The Gibbs sampling scheme iterates through the following conditional
    updates at each step *s*:

    1. **Gain sampling**: :math:`g^{(s)} \\sim p(g \\mid T_{\\rm sys}^{(s-1)},
       n^{(s-1)}, d)`
    2. **Local temperature sampling** (if not joint):
       :math:`T_{\\rm loc}^{(s)} \\sim p(T_{\\rm loc} \\mid T_{\\rm sky}^{(s-1)},
       g^{(s)}, n^{(s-1)}, d)`
    3. **Noise parameter sampling**: :math:`(\\log f_0, \\alpha)^{(s)} \\sim
       p(\\theta_n \\mid T_{\\rm sys}^{(s-1)}, g^{(s)}, d)`
    4. **Sky temperature sampling**: :math:`T_{\\rm sky}^{(s)} \\sim
       p(T_{\\rm sky} \\mid T_{\\rm loc}^{(s)}, g^{(s)}, n^{(s)}, d)`

    Computation is parallelized across MPI ranks, where each rank processes a
    subset of TODs. Gain and noise parameters are local to each TOD, while
    sky temperature parameters are shared and synchronized across all ranks.

    Parameters
    ----------
    local_TOD_list : list[np.ndarray]
        List of time-ordered data vectors for this MPI rank. Each element
        has shape ``(n_time_i,)``.
    local_t_lists : list[np.ndarray]
        List of time-lag arrays for each TOD on this rank. Each element has
        shape ``(n_time_i,)`` and is used for constructing the flicker noise
        covariance matrix.
    local_gain_operator_list : list[np.ndarray]
        List of gain projection operators (design matrices) for each TOD.
        Each element has shape ``(n_time_i, n_gain_modes)``, mapping gain
        polynomial coefficients to time-domain gains.
    local_Tsky_operator_list : list[np.ndarray]
        List of sky temperature projection operators for each TOD. Each
        element has shape ``(n_time_i, n_sky_modes)``, typically a beam-
        weighted HEALPix pointing matrix.
    local_Tloc_operator_list : list[np.ndarray]
        List of local temperature projection operators for each TOD. Each
        element has shape ``(n_time_i, n_loc_modes)``.
    init_Tsky_params : np.ndarray
        Initial sky temperature parameter vector, shape ``(n_sky_modes,)``.
        Shared across all ranks.
    init_Tloc_params_list : list[np.ndarray]
        Initial local temperature parameter vectors for each TOD on this
        rank. Each element has shape ``(n_loc_modes,)``.
    init_noise_params_list : list[np.ndarray]
        Initial noise parameter vectors for each TOD on this rank. Each
        element contains ``[logf0, alpha]`` (2 elements) or
        ``[DC_gain, logf0, alpha]`` (3 elements) if using the factorized
        gain model.
    local_logfc_list : list[float]
        List of log10 cutoff frequencies for the flicker noise model for
        each TOD. Typically ``log10(1 / (N * dt) * 2 * pi)``.
    local_Tsys_injection_list : list[float | np.ndarray], optional
        List of additional system temperature injections (e.g., noise diode
        contribution) for each TOD. Defaults to zeros.
    wnoise_var : float, optional
        White noise variance, by default ``2.5e-6``. This corresponds to
        :math:`\\sigma_w^2 = 1 / (B \\tau)` where *B* is bandwidth and
        :math:`\\tau` is integration time.
    Tsky_prior_cov_inv : np.ndarray, optional
        Inverse prior covariance matrix for sky temperature parameters.
        Shape ``(n_sky_modes,)`` for diagonal or
        ``(n_sky_modes, n_sky_modes)`` for full matrix. If ``None``, an
        uninformative prior is used.
    Tsky_prior_mean : np.ndarray, optional
        Prior mean vector for sky temperature parameters, shape
        ``(n_sky_modes,)``. If ``None``, zero mean is assumed.
    local_Tloc_prior_cov_inv_list : list[np.ndarray], optional
        List of inverse prior covariance matrices for local temperature
        parameters for each TOD. If ``None``, uninformative priors are used.
    local_Tloc_prior_mean_list : list[np.ndarray], optional
        List of prior mean vectors for local temperature parameters for
        each TOD. If ``None``, zero means are assumed.
    local_gain_prior_cov_inv_list : list[np.ndarray], optional
        List of inverse prior covariance matrices for gain parameters for
        each TOD. If ``None``, uninformative priors are used.
    local_gain_prior_mean_list : list[np.ndarray], optional
        List of prior mean vectors for gain parameters for each TOD.
        If ``None``, zero means are assumed.
    local_noise_prior_func_list : list[Callable | None], optional
        List of custom log-prior functions for noise parameters for each
        TOD. Each callable takes a noise parameter vector and returns the
        log-prior probability. If ``None``, default priors are used.
    joint_Tsys_sampling : bool, optional
        If ``True``, sample the full system temperature parameter vector
        :math:`[T_{\\rm sky}, T_{\\rm loc,1}, T_{\\rm loc,2}, \\ldots]`
        jointly using a single linear system. If ``False`` (default),
        sample :math:`T_{\\rm sky}` and :math:`T_{\\rm loc}` separately.
    smooth_gain_model : ``{'linear', 'log', 'factorized'}``, optional
        Gain parameterization model:

        - ``'linear'``: :math:`g(t) = \\exp(\\mathbf{G} \\mathbf{p})`
        - ``'log'``: :math:`\\log g(t) = \\mathbf{G} \\mathbf{p}`
        - ``'factorized'``: :math:`g(t) = g_0 \\exp(\\mathbf{G} \\mathbf{p})`
          where :math:`g_0` is a DC gain sampled with the noise parameters.

        Default is ``'linear'``.
    noise_sampler_type : ``{'emcee', 'nuts'}``, optional
        MCMC sampler used for noise parameter inference. Default is
        ``'emcee'``.
    noise_Jeffreys_prior : bool, optional
        If ``True``, use a Jeffreys prior for noise scale parameters,
        providing scale invariance. Default is ``False``.
    noise_params_bounds : list[tuple[float, float]], optional
        Parameter bounds for noise MCMC sampling. Each tuple specifies
        ``(lower, upper)`` for one parameter. Default bounds are
        ``[(-6.0, -3.0), (1.1, 4.0)]`` for ``[logf0, alpha]``.
    n_samples : int, optional
        Number of Gibbs samples to draw. Default is ``100``.
    tol : float, optional
        Tolerance for iterative linear solvers (CG, MINRES). Default is
        ``1e-12``.
    linear_solver : Callable, optional
        Linear solver function used for the conjugate sampling steps.
        Default is :func:`hydra_tod.linear_solver.cg`.
    root : int, optional
        MPI rank to gather results onto. If ``None`` (default), results
        are gathered onto all ranks via ``allgather``.
    Est_mode : bool, optional
        If ``True``, return MAP estimates instead of posterior samples
        (no stochastic sampling step). Default is ``False``.
    debug : bool, optional
        If ``True``, print diagnostic information during sampling.
        Default is ``False``.

    Returns
    -------
    Tsky_samples : np.ndarray
        Sky (or joint Tsys) temperature samples, shape
        ``(n_samples, n_sky_modes)`` or ``(n_samples, n_sys_modes)`` if
        ``joint_Tsys_sampling=True``.
    all_gain_samples : list[np.ndarray]
        Gain parameter samples gathered from all ranks. Each element has
        shape ``(n_local_TODs, n_samples, n_gain_modes)``.
    all_noise_samples : list[np.ndarray]
        Noise parameter samples gathered from all ranks. Each element has
        shape ``(n_local_TODs, n_samples, n_noise_modes)``.
    all_Tloc_samples : list[np.ndarray], optional
        Local temperature parameter samples, only returned when
        ``joint_Tsys_sampling=False``. Each element has shape
        ``(n_local_TODs, n_samples, n_loc_modes)``.

    Notes
    -----
    - All ranks must call this function together due to MPI collective
      operations (barriers, allgather/gather).
    - The function uses index ``si - 1`` (i.e., the previous sample) for
      conditional parameters, which wraps to the last index at ``si = 0``,
      using the initialization values stored there.
    - The ``factorized`` gain model includes a DC gain :math:`g_0` that is
      sampled jointly with the noise parameters, yielding 3 noise parameter
      components instead of 2.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.

    See Also
    --------
    TOD_Gibbs_sampler_joint_loc : Variant with joint local parameter sampling.
    gain_sampler : Conditional gain sampling step.
    Tsys_sampler_multi_TODs : Conditional Tsys sampling across multiple TODs.
    flicker_noise_sampler : Noise parameter MCMC sampling step.
    """

    # Synchronize the processes
    mpiutil.barrier()

    # Check the length of the input lists
    num_TODs = len(local_TOD_list)

    if (
        num_TODs != len(local_t_lists)
        or num_TODs != len(local_Tsky_operator_list)
        or num_TODs != len(local_Tloc_operator_list)
        or num_TODs != len(local_gain_operator_list)
    ):
        raise ValueError("The length of the input lists must be the same.")

    if smooth_gain_model not in ["linear", "log", "factorized"]:
        raise ValueError(
            "Unknown smooth_gain_model: {}. Supported models are 'linear', 'log', and 'factorized'.".format(
                smooth_gain_model
            )
        )
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
        Tloc_prior_cov_inv_list = [
            item
            for sublist in comm.allgather(local_Tloc_prior_cov_inv_list)
            for item in sublist
        ]
        Tloc_prior_mean_list = [
            item
            for sublist in comm.allgather(local_Tloc_prior_mean_list)
            for item in sublist
        ]
        init_Tloc_params_list = [
            item
            for sublist in comm.allgather(init_Tloc_params_list)
            for item in sublist
        ]

        if Tsky_prior_cov_inv.ndim == 1:
            # assert all
            assert all(
                Tloc_prior_cov_inv.ndim == 1
                for Tloc_prior_cov_inv in Tloc_prior_cov_inv_list
            ), "All prior_cov_inv must have the same number of dimensions [1D (i.e., diagonals) or 2D]."
            # The prior covariance matrices are diagonal, the inputs are the diagonal elements
            Tsys_prior_cov_inv = np.hstack(
                [Tsky_prior_cov_inv] + Tloc_prior_cov_inv_list
            )
        elif Tsky_prior_cov_inv.ndim == 2:
            assert all(
                Tloc_prior_cov_inv.ndim == 2
                for Tloc_prior_cov_inv in Tloc_prior_cov_inv_list
            ), "All prior_cov_inv must have the same number of dimensions [1D (i.e., diagonals) or 2D]."
            Tsys_prior_cov_inv = block_diag(
                Tsky_prior_cov_inv, *Tloc_prior_cov_inv_list
            )
        else:
            raise ValueError("Invalid number of dimensions for Tsky_prior_cov_inv.")
        Tsys_prior_mean = np.hstack([Tsky_prior_mean] + Tloc_prior_mean_list)
        init_Tsys_params = np.hstack([init_Tsky_params] + init_Tloc_params_list)

        local_Tsys_operator_list = get_Tsys_operator(
            local_Tsky_operator_list, local_Tloc_operator_list
        )
        n_sys_modes = local_Tsys_operator_list[0].shape[1]
        Tsys_samples = np.zeros((n_samples, n_sys_modes))
        Tsys_samples[-1, :] = init_Tsys_params
    else:
        n_loc_modes = local_Tloc_operator_list[0].shape[1]
        local_Tloc_samples = np.zeros((num_TODs, n_samples, n_loc_modes))

        n_sky_modes = len(init_Tsky_params)
        Tsky_samples = np.zeros((n_samples, n_sky_modes))
        Tsky_params = init_Tsky_params

        local_Tsky_mu_list = [
            np.zeros_like(local_TOD_list[di]) for di in range(num_TODs)
        ]

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
        local_Tsys_injection_list = [0.0] * num_TODs
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
                Tsys = (
                    Tsys_operator @ Tsys_samples[si - 1, :]
                    + local_Tsys_injection_list[di]
                )
            else:
                Tsky_operator = local_Tsky_operator_list[di]
                Tloc_operator = local_Tloc_operator_list[di]
                Tloc_params = local_Tloc_samples[di, si - 1, :]
                TOD_sky = Tsky_operator @ Tsky_params
                TOD_loc = Tloc_operator @ Tloc_params
                Tsys = TOD_sky + TOD_loc + local_Tsys_injection_list[di]
                mu_loc = TOD_sky + local_Tsys_injection_list[di]

            noise_params = local_noise_samples[di, si - 1, :]

            logfc = local_logfc_list[di]

            # Sample gain parameters
            gain_sample, gains = gain_sampler(
                TOD,
                t_list,
                gain_operator,
                Tsys,
                noise_params,
                logfc,
                model=smooth_gain_model,
                wnoise_var=wnoise_var,
                n_samples=num_sample,
                tol=tol,
                prior_cov_inv=local_gain_prior_cov_inv_list[di],
                prior_mean=local_gain_prior_mean_list[di],
                solver=linear_solver,
            )

            if debug:
                print(
                    "Rank: {}, local id: {}, gain_sample {}: {}".format(
                        rank, di, si, gain_sample
                    )
                )

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
                    solver=linear_solver,
                )

                local_Tloc_samples[di, si, :] = Tloc_params
                TOD_loc = Tloc_operator @ Tloc_params

                Tsys = TOD_sky + TOD_loc + local_Tsys_injection_list[di]
                # Collect mu vectors for Tsky sampler
                local_Tsky_mu_list[di] = TOD_loc + local_Tsys_injection_list[di]

            # Sample noise parameters
            noise_sample = flicker_noise_sampler(
                TOD,
                t_list,
                gains,
                Tsys,
                init_noise_params_list[
                    di
                ],  # using the input init_noise_params as fixed initial point for MCMC sampling
                logfc,
                n_samples=num_sample,
                wnoise_var=wnoise_var,
                prior_func=local_noise_prior_func_list[di],
            )

            if include_DC_Gain:
                local_gain_list[di] = gains * noise_sample[0]
            else:
                local_gain_list[di] = gains

            local_noise_samples[di, si, :] = noise_sample

            if debug:
                # print("Rank: {}, local id: {}, gain_sample {}: {}".format(rank, di, si, gain_sample))
                print(
                    "Rank: {}, local id: {}, noise_sample {}: {}".format(
                        rank, di, si, noise_sample
                    )
                )

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
            Est_mode=Est_mode,
        )
        if joint_Tsys_sampling:
            Tsys_samples[si, :] = sample
        else:
            Tsky_params = sample
            Tsky_samples[si, :] = Tsky_params

        # Update after completing all TODs for this sample
        if rank0:
            pbar.set_postfix({"Sample": si + 1, "Status": "Complete"})

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


def TOD_Gibbs_sampler_joint_loc(
    local_TOD_list: list[np.ndarray],
    local_t_lists: list[np.ndarray],
    #   local_TOD_diode_list,
    local_gain_operator_list: list[np.ndarray],
    local_Tsky_operator_list: list[np.ndarray],
    local_Tloc_operator_list: list[np.ndarray],
    init_Tsky_params: np.ndarray,
    init_Tloc_params_list: list[np.ndarray],
    init_noise_params_list: list[np.ndarray],
    local_logfc_list: list[float],
    wnoise_var: float = 2.5e-6,
    Tsky_prior_cov_inv: Optional[np.ndarray] = None,
    Tsky_prior_mean: Optional[np.ndarray] = None,
    local_Tloc_prior_cov_inv_list: Optional[list[np.ndarray]] = None,
    local_Tloc_prior_mean_list: Optional[list[np.ndarray]] = None,
    local_gain_prior_cov_inv_list: Optional[list[np.ndarray]] = None,
    local_gain_prior_mean_list: Optional[list[np.ndarray]] = None,
    local_noise_prior_func_list: Optional[list[Callable | None]] = None,
    noise_sampler_type: Literal["emcee", "nuts"] = "emcee",
    ploc_Jeffreys_prior: bool = True,
    noise_Jeffreys_prior: bool = True,
    noise_params_bounds: list[tuple[float, float]] = flicker_bounds,
    n_samples: int = 100,
    tol: float = 1e-12,
    linear_solver: Callable = cg,
    Est_mode: bool = False,
    root: Optional[int] = None,
    debug: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Run the Gibbs sampler with joint gain and local temperature sampling.

    This is a variant of :func:`TOD_Gibbs_sampler` that samples the gain
    parameters and local temperature parameters jointly using a combined
    local parameter sampler, rather than sampling them in separate conditional
    steps.

    The data model is identical:

    .. math::
        d(t) = T_{\\rm sys}(t) \\, [1 + n(t)] \\, g(t)

    where :math:`T_{\\rm sys} = T_{\\rm sky} + T_{\\rm loc}`. However, the
    Gibbs update scheme differs:

    1. **Joint local parameter sampling**: Sample gain and local temperature
       parameters jointly:
       :math:`(g, T_{\\rm loc})^{(s)} \\sim p(g, T_{\\rm loc} \\mid
       T_{\\rm sky}^{(s-1)}, n^{(s-1)}, d)`
    2. **Noise parameter sampling**: :math:`(\\log f_0, \\alpha)^{(s)} \\sim
       p(\\theta_n \\mid T_{\\rm sys}^{(s)}, g^{(s)}, d)`
    3. **Sky temperature sampling**: :math:`T_{\\rm sky}^{(s)} \\sim
       p(T_{\\rm sky} \\mid T_{\\rm loc}^{(s)}, g^{(s)}, n^{(s)}, d)`

    The joint sampling of gain and local temperature can improve mixing when
    these parameters are correlated.

    Parameters
    ----------
    local_TOD_list : list[np.ndarray]
        List of time-ordered data vectors for this MPI rank. Each element
        has shape ``(n_time_i,)``.
    local_t_lists : list[np.ndarray]
        List of time-lag arrays for each TOD on this rank. Each element has
        shape ``(n_time_i,)``.
    local_gain_operator_list : list[np.ndarray]
        List of gain projection operators for each TOD. Each element has
        shape ``(n_time_i, n_gain_modes)``.
    local_Tsky_operator_list : list[np.ndarray]
        List of sky temperature projection operators for each TOD. Each
        element has shape ``(n_time_i, n_sky_modes)``.
    local_Tloc_operator_list : list[np.ndarray]
        List of local temperature projection operators for each TOD. Each
        element has shape ``(n_time_i, n_loc_modes)``. The first column is
        expected to be the noise diode template, and the remaining columns
        are the receiver temperature basis functions.
    init_Tsky_params : np.ndarray
        Initial sky temperature parameter vector, shape ``(n_sky_modes,)``.
    init_Tloc_params_list : list[np.ndarray]
        Initial local temperature parameter vectors for each TOD on this
        rank. Each element has shape ``(n_loc_modes,)``.
    init_noise_params_list : list[np.ndarray]
        Initial noise parameter vectors for each TOD. Each element contains
        ``[logf0, alpha]``.
    local_logfc_list : list[float]
        List of log10 cutoff frequencies for the flicker noise model for
        each TOD.
    wnoise_var : float, optional
        White noise variance. Default is ``2.5e-6``.
    Tsky_prior_cov_inv : np.ndarray, optional
        Inverse prior covariance for sky temperature parameters. Shape
        ``(n_sky_modes,)`` for diagonal or ``(n_sky_modes, n_sky_modes)``
        for full matrix.
    Tsky_prior_mean : np.ndarray, optional
        Prior mean for sky temperature parameters, shape ``(n_sky_modes,)``.
    local_Tloc_prior_cov_inv_list : list[np.ndarray], optional
        List of inverse prior covariance (diagonal) for local temperature
        parameters. Required for the joint sampler.
    local_Tloc_prior_mean_list : list[np.ndarray], optional
        List of prior mean vectors for local temperature parameters.
        Required for the joint sampler.
    local_gain_prior_cov_inv_list : list[np.ndarray], optional
        List of inverse prior covariance (diagonal) for gain parameters.
        Required for the joint sampler.
    local_gain_prior_mean_list : list[np.ndarray], optional
        List of prior mean vectors for gain parameters. Required for the
        joint sampler.
    local_noise_prior_func_list : list[Callable | None], optional
        List of custom log-prior functions for noise parameters for each
        TOD. If ``None``, default priors are used.
    noise_sampler_type : ``{'emcee', 'nuts'}``, optional
        MCMC sampler for noise parameter inference. Default is ``'emcee'``.
    ploc_Jeffreys_prior : bool, optional
        If ``True``, use a Jeffreys prior in the joint local parameter
        sampler. Default is ``True``.
    noise_Jeffreys_prior : bool, optional
        If ``True``, use a Jeffreys prior for noise scale parameters.
        Default is ``True``.
    noise_params_bounds : list[tuple[float, float]], optional
        Parameter bounds for noise MCMC sampling. Default is
        ``[(-6.0, -3.0), (1.1, 4.0)]``.
    n_samples : int, optional
        Number of Gibbs samples to draw. Default is ``100``.
    tol : float, optional
        Tolerance for iterative linear solvers. Default is ``1e-12``.
    linear_solver : Callable, optional
        Linear solver function. Default is
        :func:`hydra_tod.linear_solver.cg`.
    Est_mode : bool, optional
        If ``True``, return MAP estimates instead of posterior samples.
        Default is ``False``.
    root : int, optional
        MPI rank to gather results onto. If ``None`` (default), results
        are gathered onto all ranks via ``allgather``.
    debug : bool, optional
        If ``True``, print diagnostic information. Default is ``False``.

    Returns
    -------
    Tsky_samples : np.ndarray
        Sky temperature parameter samples, shape
        ``(n_samples, n_sky_modes)``.
    all_gain_samples : list[np.ndarray]
        Gain parameter samples gathered from all ranks. Each element has
        shape ``(n_local_TODs, n_samples, n_gain_modes)``.
    all_noise_samples : list[np.ndarray]
        Noise parameter samples gathered from all ranks. Each element has
        shape ``(n_local_TODs, n_samples, n_noise_modes)``.
    all_Tloc_samples : list[np.ndarray]
        Local temperature parameter samples gathered from all ranks. Each
        element has shape ``(n_local_TODs, n_samples, n_loc_modes)``.

    Notes
    -----
    - This variant uses :func:`local_params_sampler` from the
      ``local_sampler`` module for joint gain and local temperature
      inference.
    - The local temperature operator is decomposed as
      :math:`T_{\\rm loc}(t) = T_{\\rm nd}(t) \\, e^{p_0}
      + e^{\\mathbf{R}_{\\rm res} \\mathbf{p}_{1:}}` where :math:`p_0`
      controls the noise diode amplitude and :math:`\\mathbf{p}_{1:}` are
      the receiver temperature coefficients.
    - The gain model uses a log parameterization:
      :math:`g(t) = \\exp(\\mathbf{G} \\mathbf{p}_g)`.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.

    See Also
    --------
    TOD_Gibbs_sampler : Standard Gibbs sampler with separate conditional steps.
    local_params_sampler : Joint local parameter sampler.
    """

    # Synchronize the processes
    mpiutil.barrier()

    # Check the length of the input lists
    num_TODs = len(local_TOD_list)

    if (
        num_TODs != len(local_t_lists)
        or num_TODs != len(local_Tsky_operator_list)
        or num_TODs != len(local_Tloc_operator_list)
        or num_TODs != len(local_gain_operator_list)
    ):
        raise ValueError("The length of the input lists must be the same.")

    n_g_modes = local_gain_operator_list[0].shape[1]
    local_gain_samples = np.zeros((num_TODs, n_samples, n_g_modes))

    n_loc_modes = local_Tloc_operator_list[0].shape[1]
    local_Tloc_samples = np.zeros((num_TODs, n_samples, n_loc_modes))

    n_sky_modes = len(init_Tsky_params)
    Tsky_samples = np.zeros((n_samples, n_sky_modes))

    n_N_modes = len(init_noise_params_list[0])
    local_noise_samples = np.zeros((num_TODs, n_samples, n_N_modes))

    Tsky_params = init_Tsky_params

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
    from .local_sampler import local_params_sampler

    master_rng_key = jr.PRNGKey(42)

    # Sample the parameters
    for si in tqdm(range(n_samples)):
        for di in range(num_TODs):
            TOD = local_TOD_list[di]
            gain_operator = local_gain_operator_list[di]
            Tsky_operator = local_Tsky_operator_list[di]
            Tloc_operator = local_Tloc_operator_list[di]
            # TOD_diode = local_TOD_diode_list[di]

            noise_params = local_noise_samples[di, si - 1, :]

            TOD_sky = Tsky_operator @ Tsky_params
            Tnd_vec = Tloc_operator[:, 0]
            Tres_proj = Tloc_operator[:, 1:]

            # local_gain_prior_cov_inv_list[di]
            local_params_prior_cov_inv = np.concatenate(
                [local_gain_prior_cov_inv_list[di], local_Tloc_prior_cov_inv_list[di]]
            )
            local_params_prior_mean = np.concatenate(
                [local_gain_prior_mean_list[di], local_Tloc_prior_mean_list[di]]
            )

            def p_loc_prior(x):
                return jnp.sum(
                    -0.5
                    * (x - local_params_prior_mean) ** 2
                    * local_params_prior_cov_inv
                )

            master_rng_key, noise_key = jr.split(master_rng_key)

            local_params = local_params_sampler(
                TOD,
                TOD_sky,
                gain_operator,
                Tnd_vec,
                Tres_proj,
                noise_params,
                rng_key=noise_key,
                add_jeffreys=ploc_Jeffreys_prior,
                prior_func=p_loc_prior,
                jaxjit=False,
            )
            print(
                "local_params {}: \n {} \n {}".format(
                    si, local_params[:4], local_params[4:]
                )
            )

            gain_sample = local_params[:4]
            local_gain_samples[di, si, :] = gain_sample
            gains = np.exp(gain_operator @ gain_sample)
            local_gain_list[di] = gains

            Tloc_params = local_params[4:]
            local_Tloc_samples[di, si, :] = Tloc_params
            Tloc_TOD = Tnd_vec * np.exp(Tloc_params[0]) + np.exp(
                Tres_proj @ Tloc_params[1:]
            )
            Tsys = TOD_sky + Tloc_TOD
            local_Tsky_mu_list[di] = Tloc_TOD

            # Sample noise parameters
            noise_sample = flicker_sampler(
                TOD,
                gains,
                Tsys,
                init_params=init_noise_params_list[
                    di
                ],  # using the input init_noise_params as fixed initial point for MCMC sampling
                n_samples=num_sample,
                include_DC_Gain=False,
                prior_func=local_noise_prior_func_list[di],
                jeffreys=noise_Jeffreys_prior,
                bounds=noise_params_bounds,
                sampler=noise_sampler_type,
                rng_key=noise_key,
            )

            local_noise_samples[di, si, :] = noise_sample
            if debug:
                print(
                    "Rank: {}, local id: {}, noise_sample {}: {}".format(
                        rank, di, si, noise_sample
                    )
                )

        # Given gain, noise and other Tsys components, sample Tsky parameters
        Tsky_params = Tsys_sampler_multi_TODs(
            local_TOD_list,
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
            Est_mode=Est_mode,
        )

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

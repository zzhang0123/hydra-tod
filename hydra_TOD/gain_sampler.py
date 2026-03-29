"""Gibbs-sampling step for smooth gain coefficients.

This module draws posterior samples of the time-varying instrument gain
conditioned on the current system temperature and noise parameters.  It
implements the iterative GLS formulation from Zhang et al. (2026, §3.2).

Three gain parameterisations are supported:

``"linear"``
    :math:`g(t) = \\mathbf{G}\\,\\mathbf{p}_g` — direct polynomial model.
    Noise covariance is heteroskedastic (scales with :math:`T_{\\rm sys}`),
    so an iterative GLS step is required.

``"log"``
    :math:`g(t) = \\exp(\\mathbf{G}\\,\\mathbf{p}_g)` — log-gain model.
    Data are log-transformed, converting the multiplicative noise to
    additive.  No iterative GLS needed.

``"factorized"``
    :math:`g(t) = g_0 \\cdot (\\mathbf{G}\\,\\mathbf{p}_g + 1)` — smooth
    fluctuation around a DC gain :math:`g_0`.  The DC gain is passed as
    the first element of ``noise_params``.

Public API
----------
gain_sampler
    Dispatcher: select model via a string and return ``(coeffs, gains)``.
linear_gain_sampler
    Posterior sampler for the linear model.
log_gain_sampler
    Posterior sampler for the log model.
factorized_gain_sampler
    Posterior sampler for the factorized model.

Typical usage
-------------
.. code-block:: python

    from hydra_tod.gain_sampler import gain_sampler

    gain_coeffs, gains = gain_sampler(
        data=tod,
        t_list=t_list,
        gain_proj=gain_proj,   # Legendre projection matrix
        Tsys=tsys_vector,
        noise_params=(logf0, alpha),
        logfc=logfc,
        model="linear",
        prior_cov_inv=prior_cov_inv_g,
        prior_mean=prior_mean_g,
    )

These functions are called automatically by
:func:`~hydra_tod.full_Gibbs_sampler.TOD_Gibbs_sampler`.  Call them
directly only when sampling gains in isolation.

See Also
--------
hydra_tod.tsys_sampler : System-temperature Gibbs step.
hydra_tod.linear_sampler : Underlying GLS and Gaussian sampling machinery.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable
from .linear_solver import cg
from .utils import cho_compute_mat_inv
from .flicker_model import flicker_cov
from .linear_sampler import sample_p_old, iterative_gls
from scipy.linalg import toeplitz

try:
    import pickle
    import os
    from jax import jit

    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    corr_emulator_path = os.path.join(module_dir, "flicker_corr_emulator.pkl")
    logdet_emulator_path = os.path.join(module_dir, "flicker_logdet_emulator.pkl")

    # Import the class definition before unpickling
    from .flicker_model import FlickerCorrEmulator

    # Load the emulator
    with open(corr_emulator_path, "rb") as f:
        flicker_cov = pickle.load(f)

    from .flicker_model import LogDetEmulator

    with open(logdet_emulator_path, "rb") as f:
        flicker_logdet = pickle.load(f)

    use_emulator = True
    print("Using the emulator for flicker noise correlation function.")

    # JIT the flicker covariance function
    flicker_cov_jax = jit(flicker_cov.create_jax())
    flicker_logdet_jax = jit(flicker_logdet.create_jax())

except (FileNotFoundError, ImportError, AttributeError) as e:
    print(
        f"Emulator for flicker noise correlation function not found or failed to load: {e}"
    )
    print("Using flicker_cov_vec instead.")
    from .flicker_model import flicker_cov_vec as flicker_cov

    use_emulator = False


def linear_gain_sampler(
    data: NDArray[np.floating],
    t_list: NDArray[np.floating],
    gain_proj: NDArray[np.floating],
    Tsys: NDArray[np.floating],
    noise_params: tuple[float, float],
    logfc: float,
    wnoise_var: float = 2.5e-6,
    mu: float = 0.0,
    n_samples: int = 1,
    tol: float = 1e-13,
    prior_cov_inv: NDArray[np.floating] | None = None,
    prior_mean: NDArray[np.floating] | None = None,
    solver: Callable = cg,
) -> NDArray[np.float64]:
    r"""Sample gain coefficients under the linear gain model.

    In the linear model the gain is a linear function of basis coefficients:

    .. math::

        g(t) = \mathbf{G}\,\mathbf{p}

    where :math:`\mathbf{G}` is the gain projection matrix and
    :math:`\mathbf{p}` is the parameter vector. The data vector is
    formed as :math:`\mathbf{d} = \mathrm{TOD} / T_{\mathrm{sys}}`.

    Parameters
    ----------
    data : NDArray[np.floating]
        Time-ordered data vector of length :math:`N_t`.
    t_list : NDArray[np.floating]
        Observation time stamps (seconds).
    gain_proj : NDArray[np.floating]
        Gain projection (design) matrix :math:`\mathbf{G}` of shape
        ``(N_t, N_p)``.
    Tsys : NDArray[np.floating]
        System temperature vector of length :math:`N_t`.
    noise_params : tuple[float, float]
        Noise parameters ``(logf0, alpha)`` where ``logf0`` is
        :math:`\log_{10}\omega_0` and ``alpha`` is the spectral index.
    logfc : float
        Log10 of the cutoff angular frequency :math:`\log_{10}\omega_c`.
    wnoise_var : float, optional
        White-noise variance :math:`\sigma_w^2`. Default is ``2.5e-6``.
    mu : float, optional
        Prior mean offset for the gain model. Default is ``0.0``.
    n_samples : int, optional
        Number of posterior samples to draw. Default is ``1``.
    tol : float, optional
        Tolerance for the iterative GLS solver. Default is ``1e-13``.
    prior_cov_inv : NDArray[np.floating] or None, optional
        Inverse prior covariance matrix (or 1-D diagonal). If ``None``,
        an uninformative prior is used.
    prior_mean : NDArray[np.floating] or None, optional
        Prior mean vector. If ``None``, zero mean is assumed.
    solver : Callable, optional
        Linear solver function (e.g., ``cg``, ``minres``). Default is
        :func:`hydra_tod.linear_solver.cg`.

    Returns
    -------
    NDArray[np.float64]
        Posterior sample(s) of gain coefficients :math:`\mathbf{p}`.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    d_vec = data / Tsys
    logf0, alpha = noise_params
    if use_emulator:
        Ncov_inv = cho_compute_mat_inv(toeplitz(flicker_cov_jax(logf0, alpha)[0]))
    else:
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
    p_GLS, sigma_inv = iterative_gls(d_vec, gain_proj, Ncov_inv, mu=mu, tol=tol)
    return sample_p_old(
        d_vec,
        gain_proj,
        sigma_inv,
        num_samples=n_samples,
        mu=mu,
        prior_cov_inv=prior_cov_inv,
        prior_mean=prior_mean,
        solver=solver,
    )


def log_gain_sampler(
    data: NDArray[np.floating],
    t_list: NDArray[np.floating],
    gain_proj: NDArray[np.floating],
    Tsys: NDArray[np.floating],
    noise_params: tuple[float, float],
    logfc: float,
    wnoise_var: float = 2.5e-6,
    mu: float = 0.0,
    n_samples: int = 1,
    prior_cov_inv: NDArray[np.floating] | None = None,
    prior_mean: NDArray[np.floating] | None = None,
    solver: Callable = cg,
) -> NDArray[np.float64]:
    r"""Sample gain coefficients under the log-linear gain model.

    In the log model the logarithm of the gain is a linear function of
    basis coefficients:

    .. math::

        g(t) = \exp(\mathbf{G}\,\mathbf{p})

    The data vector is formed as
    :math:`\mathbf{d} = \ln(\mathrm{TOD}/T_{\mathrm{sys}})`, linearising
    the model so that standard conjugate Gaussian sampling applies.

    Parameters
    ----------
    data : NDArray[np.floating]
        Time-ordered data vector of length :math:`N_t`.
    t_list : NDArray[np.floating]
        Observation time stamps (seconds).
    gain_proj : NDArray[np.floating]
        Gain projection matrix :math:`\mathbf{G}`, shape ``(N_t, N_p)``.
    Tsys : NDArray[np.floating]
        System temperature vector of length :math:`N_t`.
    noise_params : tuple[float, float]
        Noise parameters ``(logf0, alpha)``.
    logfc : float
        Log10 of the cutoff angular frequency.
    wnoise_var : float, optional
        White-noise variance. Default is ``2.5e-6``.
    mu : float, optional
        Prior mean offset. Default is ``0.0``.
    n_samples : int, optional
        Number of posterior samples. Default is ``1``.
    prior_cov_inv : NDArray[np.floating] or None, optional
        Inverse prior covariance (or 1-D diagonal).
    prior_mean : NDArray[np.floating] or None, optional
        Prior mean vector.
    solver : Callable, optional
        Linear solver function. Default is :func:`hydra_tod.linear_solver.cg`.

    Returns
    -------
    NDArray[np.float64]
        Posterior sample(s) of log-gain coefficients :math:`\mathbf{p}`.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    d_vec = np.log(data / Tsys)
    logf0, alpha = noise_params
    if use_emulator:
        Ncov_inv = cho_compute_mat_inv(toeplitz(flicker_cov_jax(logf0, alpha)[0]))
    else:
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
    return sample_p_old(
        d_vec,
        gain_proj,
        Ncov_inv,
        num_samples=n_samples,
        mu=mu,
        prior_cov_inv=prior_cov_inv,
        prior_mean=prior_mean,
        solver=solver,
    )


def factorized_gain_sampler(
    data: NDArray[np.floating],
    t_list: NDArray[np.floating],
    gain_proj: NDArray[np.floating],
    Tsys: NDArray[np.floating],
    noise_params: tuple[float, float, float],
    logfc: float,
    wnoise_var: float = 2.5e-6,
    mu: float = 1.0,
    n_samples: int = 1,
    tol: float = 1e-13,
    prior_cov_inv: NDArray[np.floating] | None = None,
    prior_mean: NDArray[np.floating] | None = None,
    solver: Callable = cg,
) -> NDArray[np.float64]:
    r"""Sample gain coefficients under the factorized gain model.

    In the factorized model the gain is decomposed into a DC component
    and a time-varying perturbation:

    .. math::

        g(t) = g_0 \bigl(\mathbf{G}\,\mathbf{p} + 1\bigr)

    where :math:`g_0` is the DC gain (sampled jointly with the noise
    parameters). The data vector is
    :math:`\mathbf{d} = \mathrm{TOD}/(T_{\mathrm{sys}}\,g_0)`.

    Parameters
    ----------
    data : NDArray[np.floating]
        Time-ordered data vector of length :math:`N_t`.
    t_list : NDArray[np.floating]
        Observation time stamps (seconds).
    gain_proj : NDArray[np.floating]
        Gain projection matrix :math:`\mathbf{G}`, shape ``(N_t, N_p)``.
    Tsys : NDArray[np.floating]
        System temperature vector of length :math:`N_t`.
    noise_params : tuple[float, float, float]
        Noise parameters ``(DC_gain, logf0, alpha)`` where ``DC_gain`` is
        the overall gain factor :math:`g_0`.
    logfc : float
        Log10 of the cutoff angular frequency.
    wnoise_var : float, optional
        White-noise variance. Default is ``2.5e-6``.
    mu : float, optional
        Prior mean offset. Default is ``1.0`` (since perturbations are
        centred around unity).
    n_samples : int, optional
        Number of posterior samples. Default is ``1``.
    tol : float, optional
        Tolerance for the iterative GLS solver. Default is ``1e-13``.
    prior_cov_inv : NDArray[np.floating] or None, optional
        Inverse prior covariance (or 1-D diagonal).
    prior_mean : NDArray[np.floating] or None, optional
        Prior mean vector.
    solver : Callable, optional
        Linear solver function. Default is :func:`hydra_tod.linear_solver.cg`.

    Returns
    -------
    NDArray[np.float64]
        Posterior sample(s) of gain perturbation coefficients
        :math:`\mathbf{p}`.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    DC_gain, logf0, alpha = noise_params
    d_vec = data / Tsys / DC_gain
    if use_emulator:
        Ncov_inv = cho_compute_mat_inv(toeplitz(flicker_cov_jax(logf0, alpha)[0]))
    else:
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
    p_GLS, sigma_inv = iterative_gls(d_vec, gain_proj, Ncov_inv, mu=mu, tol=tol)
    return sample_p_old(
        d_vec,
        gain_proj,
        sigma_inv,
        num_samples=n_samples,
        mu=mu,
        prior_cov_inv=prior_cov_inv,
        prior_mean=prior_mean,
        solver=solver,
    )


def gain_sampler(
    data: NDArray[np.floating],
    t_list: NDArray[np.floating],
    gain_proj: NDArray[np.floating],
    Tsys: NDArray[np.floating],
    noise_params: tuple[float, ...],
    logfc: float,
    model: str = "linear",
    wnoise_var: float = 2.5e-6,
    n_samples: int = 1,
    tol: float = 1e-13,
    prior_cov_inv: NDArray[np.floating] | None = None,
    prior_mean: NDArray[np.floating] | None = None,
    solver: Callable = cg,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Sample gain coefficients and reconstruct the gain time series.

    Dispatcher that selects the appropriate gain sampling function based
    on the chosen model and returns both the basis coefficients and the
    reconstructed gain vector.

    The three supported models are:

    * **"linear"**: :math:`g(t) = \mathbf{G}\,\mathbf{p}`
    * **"log"**: :math:`g(t) = \exp(\mathbf{G}\,\mathbf{p})`
    * **"factorized"**: :math:`g(t) = g_0\,(\mathbf{G}\,\mathbf{p} + 1)`

    Parameters
    ----------
    data : NDArray[np.floating]
        Time-ordered data vector of length :math:`N_t`.
    t_list : NDArray[np.floating]
        Observation time stamps (seconds).
    gain_proj : NDArray[np.floating]
        Gain projection matrix :math:`\mathbf{G}`, shape ``(N_t, N_p)``.
    Tsys : NDArray[np.floating]
        System temperature vector of length :math:`N_t`.
    noise_params : tuple[float, ...]
        Noise parameters. For ``"linear"`` and ``"log"`` models:
        ``(logf0, alpha)``. For ``"factorized"`` model:
        ``(DC_gain, logf0, alpha)``.
    logfc : float
        Log10 of the cutoff angular frequency.
    model : str, optional
        Gain model name: ``"linear"``, ``"log"``, or ``"factorized"``.
        Default is ``"linear"``.
    wnoise_var : float, optional
        White-noise variance. Default is ``2.5e-6``.
    n_samples : int, optional
        Number of posterior samples. Default is ``1``.
    tol : float, optional
        Tolerance for the iterative GLS solver. Default is ``1e-13``.
    prior_cov_inv : NDArray[np.floating] or None, optional
        Inverse prior covariance (or 1-D diagonal).
    prior_mean : NDArray[np.floating] or None, optional
        Prior mean vector.
    solver : Callable, optional
        Linear solver function. Default is :func:`hydra_tod.linear_solver.cg`.

    Returns
    -------
    sample : NDArray[np.float64]
        Posterior sample(s) of the gain basis coefficients.
    gains : NDArray[np.float64]
        Reconstructed gain time series :math:`g(t)`.

    Raises
    ------
    ValueError
        If *model* is not one of ``"linear"``, ``"log"``, ``"factorized"``.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    if model == "linear":
        sample = linear_gain_sampler(
            data,
            t_list,
            gain_proj,
            Tsys,
            noise_params,
            logfc,
            wnoise_var=wnoise_var,
            mu=0.0,
            n_samples=n_samples,
            tol=tol,
            prior_cov_inv=prior_cov_inv,
            prior_mean=prior_mean,
            solver=solver,
        )
        gains = gain_proj @ sample
    elif model == "log":
        sample = log_gain_sampler(
            data,
            t_list,
            gain_proj,
            Tsys,
            noise_params,
            logfc,
            wnoise_var=wnoise_var,
            mu=0.0,
            n_samples=n_samples,
            prior_cov_inv=prior_cov_inv,
            prior_mean=prior_mean,
            solver=solver,
        )
        gains = np.exp(gain_proj @ sample)
    elif model == "factorized":
        sample = factorized_gain_sampler(
            data,
            t_list,
            gain_proj,
            Tsys,
            noise_params,
            logfc,
            wnoise_var=wnoise_var,
            mu=1.0,
            n_samples=n_samples,
            tol=tol,
            prior_cov_inv=prior_cov_inv,
            prior_mean=prior_mean,
            solver=solver,
        )
        gains = gain_proj @ sample + 1.0
    else:
        raise ValueError(f"Unknown smooth_gain_model: {model}")

    return sample, gains

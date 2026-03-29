"""Legacy flicker-noise parameter sampler (emcee backend).

.. note::
   For new code, prefer :mod:`hydra_tod.noise_sampler_fixed_fc`, which
   supports both emcee and NUTS backends, JAX-differentiable likelihoods,
   and the ``consecutive=False`` path for flagged time samples.
   This module is retained because it is still called internally by
   :func:`~hydra_tod.full_Gibbs_sampler.TOD_Gibbs_sampler` when
   ``sampler="emcee_old"`` is requested.

This module samples the flicker-noise parameters
:math:`(\\log f_0,\\, \\alpha)` using the ``emcee`` ensemble sampler.  The
cutoff frequency :math:`f_c` is fixed (passed as ``logfc``).

The emulator for the correlation function is loaded at import time if
available; otherwise :func:`~hydra_tod.flicker_model.flicker_cov_vec` is
used directly.

Public API
----------
flicker_noise_sampler
    High-level convenience wrapper — **use this function directly**.
flicker_likeli_func
    Builds a log-likelihood closure from pre-processed data.

Typical usage
-------------
.. code-block:: python

    from hydra_tod.noise_sampler_old import flicker_noise_sampler

    noise_sample = flicker_noise_sampler(
        TOD=tod,
        t_list=t_list,
        gains=gains,
        Tsys=tsys,
        init_params=[logf0_init, alpha_init],
        logfc=logfc,
        n_samples=1,
    )

See Also
--------
hydra_tod.noise_sampler_fixed_fc : Preferred noise sampler (emcee + NUTS).
hydra_tod.flicker_model : Analytic correlation and covariance functions.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Optional

from .utils import lag_list, log_det_symmetric_toeplitz, log_likeli, log_likeli_general
from .mcmc_sampler import mcmc_sampler
import emcee

# if the emulator of the correlation function exists, load it
# otherwise, use flicker_cov_vec
try:
    import pickle
    import os

    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    emulator_path = os.path.join(module_dir, "flicker_corr_emulator.pkl")

    # Import the class definition before unpickling
    from .flicker_model import FlickerCorrEmulator

    # Load the emulator
    with open(emulator_path, "rb") as f:
        flicker_cov = pickle.load(f)
    use_emulator = True
    print("Using the emulator for flicker noise correlation function.")
except (FileNotFoundError, ImportError, AttributeError) as e:
    print(
        f"Emulator for flicker noise correlation function not found or failed to load: {e}"
    )
    print("Using flicker_cov_vec instead.")
    from .flicker_model import flicker_cov_vec

    use_emulator = False

from .flicker_model import flicker_cov_general


# Define the likelihood function for the flicker noise model.
def flicker_likeli_func(
    time_list: NDArray[np.floating],
    data: NDArray[np.floating],
    gain: NDArray[np.floating],
    Tsys: NDArray[np.floating],
    logfc: float,
    wnoise_var: float = 2.5e-6,
    boundaries: Optional[list[list[float]]] = None,
    consecutive: bool = True,
) -> Callable[[NDArray[np.floating]], float]:
    """
    Construct a log-likelihood closure for the flicker noise model.

    Preprocesses the data by dividing out the gain and system temperature
    and mean-centering the residual, then returns a callable that evaluates
    the Gaussian log-likelihood for a given pair of noise parameters
    ``(logf0, alpha)`` under a Toeplitz covariance structure.

    Parameters
    ----------
    time_list : NDArray[np.floating]
        1-D array of observation time stamps (seconds).
    data : NDArray[np.floating]
        1-D array of time-ordered data values.
    gain : NDArray[np.floating]
        1-D array of instrument gain values, same length as *data*.
    Tsys : NDArray[np.floating]
        1-D array of system temperature values, same length as *data*.
    logfc : float
        Log10 of the low-frequency cutoff angular frequency
        (omega units, i.e. ``2 * pi * f``).
    wnoise_var : float, optional
        White noise variance added to the diagonal of the covariance
        matrix. Default is ``2.5e-6``.
    boundaries : list of list of float or None, optional
        Parameter boundaries ``[[logf0_min, logf0_max], [alpha_min, alpha_max]]``.
        If provided, the likelihood returns ``-inf`` outside these bounds.
    consecutive : bool, optional
        If ``True`` (default), assume evenly-spaced time stamps and use
        efficient Toeplitz solvers.  If ``False``, build the full
        covariance matrix for non-consecutive time stamps using
        :func:`~hydra_tod.flicker_model.flicker_cov_general`.

    Returns
    -------
    log_like : callable
        A function ``log_like(params) -> float`` where
        ``params = [logf0, alpha]``.

    Notes
    -----
    The frequencies ``f0`` and ``fc`` are in angular frequency units
    (rad/s), differing from FFT frequencies by a factor of ``2 * pi``.

    When the pre-trained emulator is available it is used in place of
    the analytic ``flicker_cov_vec`` for faster evaluation.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    tau_list = lag_list(time_list)
    _time_list = np.asarray(time_list, dtype=np.float64)

    dvec = data / gain / Tsys - 1.0
    dvec = np.asarray(dvec, dtype=np.float64)
    dvec -= np.mean(dvec)

    def _check_bounds(logf0, alpha):
        if boundaries is not None:
            if (
                logf0 < boundaries[0][0]
                or logf0 > boundaries[0][1]
                or alpha < boundaries[1][0]
                or alpha > boundaries[1][1]
            ):
                return False
        return True

    if not consecutive:
        # Non-consecutive path: full covariance at each evaluation

        def log_like(params: NDArray[np.floating]) -> float:
            logf0, alpha = params
            if not _check_bounds(logf0, alpha):
                return -np.inf
            cov = flicker_cov_general(
                _time_list,
                10.0**logf0,
                10.0**logfc,
                alpha,
                white_n_variance=wnoise_var,
            )
            return log_likeli_general(cov, dvec)

        return log_like

    # Consecutive (Toeplitz) path — original behaviour
    if boundaries is not None:
        if use_emulator:

            def log_like(params: NDArray[np.floating]) -> float:
                logf0, alpha = params
                if not _check_bounds(logf0, alpha):
                    return -np.inf
                corr_list = flicker_cov(logf0, alpha)
                return log_likeli(corr_list, dvec)

        else:

            def log_like(params: NDArray[np.floating]) -> float:
                logf0, alpha = params
                if not _check_bounds(logf0, alpha):
                    return -np.inf
                corr_list = flicker_cov_vec(
                    tau_list,
                    10.0**logf0,
                    10.0**logfc,
                    alpha,
                    white_n_variance=wnoise_var,
                )
                return log_likeli(corr_list, dvec)

    else:
        if use_emulator:

            def log_like(params: NDArray[np.floating]) -> float:
                logf0, alpha = params
                corr_list = flicker_cov(logf0, alpha)
                return log_likeli(corr_list, dvec)

        else:

            def log_like(params: NDArray[np.floating]) -> float:
                logf0, alpha = params
                corr_list = flicker_cov_vec(
                    tau_list,
                    10.0**logf0,
                    10.0**logfc,
                    alpha,
                    white_n_variance=wnoise_var,
                )
                return log_likeli(corr_list, dvec)

    return log_like


def flicker_noise_sampler(
    TOD: NDArray[np.floating],
    t_list: NDArray[np.floating],
    gains: NDArray[np.floating],
    Tsys: NDArray[np.floating],
    init_params: NDArray[np.floating],
    logfc: float,
    n_samples: int = 1,
    wnoise_var: float = 2.5e-6,
    prior_func: Optional[Callable[[NDArray[np.floating]], float]] = None,
    boundary: Optional[list[list[float]]] = None,
    consecutive: bool = True,
) -> NDArray[np.floating]:
    """
    Sample flicker noise parameters ``(logf0, alpha)`` using the emcee
    ensemble MCMC sampler.

    This is a convenience wrapper that constructs the likelihood via
    :func:`flicker_likeli_func` and delegates to :func:`mcmc_sampler`.

    Parameters
    ----------
    TOD : NDArray[np.floating]
        1-D time-ordered data array.
    t_list : NDArray[np.floating]
        1-D array of observation time stamps (seconds).
    gains : NDArray[np.floating]
        1-D array of instrument gain values.
    Tsys : NDArray[np.floating]
        1-D array of system temperature values.
    init_params : NDArray[np.floating]
        Initial guess for ``[logf0, alpha]``.
    logfc : float
        Log10 of the low-frequency cutoff angular frequency.
    n_samples : int, optional
        Number of posterior samples to return. ``1`` returns a single
        sample (default), ``0`` returns the MAP estimate.
    wnoise_var : float, optional
        White noise variance. Default is ``2.5e-6``.
    prior_func : callable or None, optional
        A function ``prior_func(params) -> float`` returning the
        log-prior probability. If ``None`` a flat prior is used.
    boundary : list of list of float or None, optional
        Parameter boundaries ``[[logf0_min, logf0_max], [alpha_min, alpha_max]]``.
        Defaults to ``[[-6.0, -3.0], [1.1, 4.0]]`` when ``None``.
    consecutive : bool, optional
        If ``True`` (default), assume evenly-spaced time stamps.  If
        ``False``, build the full covariance matrix for non-consecutive
        time stamps.

    Returns
    -------
    samples : NDArray[np.floating]
        Posterior sample(s) of shape ``(ndim,)`` if ``n_samples <= 1``,
        or ``(n_samples, ndim)`` otherwise.

    Notes
    -----
    Internally runs ``emcee`` with 50 steps per chain and a proposal
    standard deviation of 0.2, relying on the adaptive burn-in logic
    of :func:`mcmc_sampler`.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    if boundary is None:
        boundary = [[-6.0, -3.0], [1.1, 4.0]]  # Default boundaries

    log_likeli_func = flicker_likeli_func(
        t_list,
        TOD,
        gains,
        Tsys,
        logfc,
        wnoise_var=wnoise_var,
        boundaries=boundary,
        consecutive=consecutive,
    )

    return mcmc_sampler(
        log_likeli_func,
        init_params,
        p_std=0.2,
        nsteps=50,  # steps for each chain
        n_samples=n_samples,
        prior_func=prior_func,
        return_sampler=False,
    )

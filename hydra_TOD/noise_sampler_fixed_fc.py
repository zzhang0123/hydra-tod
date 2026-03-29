"""Gibbs-sampling step for flicker-noise parameters with fixed cutoff frequency.

This is the **preferred** noise-parameter sampler.  It draws posterior
samples of :math:`(\\log f_0,\\, \\alpha)` — the knee frequency and spectral
index of the :math:`1/f` noise model — conditioned on the current gain and
system temperature.

The cutoff frequency :math:`f_c` is treated as fixed (typically set to the
inverse of the TOD duration) rather than sampled, which avoids the
gain/noise-scale degeneracy described in Zhang et al. (2026, §2.4).

Architecture
------------
The log-likelihood is evaluated via one of two paths, selected at import
time:

**Emulator path** (preferred)
    If ``flicker_corr_emulator.pkl`` and ``flicker_logdet_emulator.pkl``
    are found in the package directory, polynomial emulators
    (:class:`~hydra_tod.flicker_model.FlickerCorrEmulator` /
    :class:`~hydra_tod.flicker_model.LogDetEmulator`) are used.  These
    speed up covariance evaluation by ~1700× compared to the ``mpmath``
    reference implementation.

**Direct path**
    Falls back to :func:`~hydra_tod.flicker_model.flicker_cov_vec` if
    the emulators are unavailable.

MCMC backends
-------------
``sampler="emcee"`` (default)
    Ensemble sampler via the :mod:`hydra_tod.mcmc_sampler` wrapper.
    Differentiable gradients are *not* required.

``sampler="NUTS"``
    No-U-Turn Sampler (NumPyro/JAX).  Uses JAX-compatible
    log-likelihood functions for automatic differentiation.

Non-consecutive time lists
--------------------------
Pass ``consecutive=False`` together with the actual ``time_list`` when
time samples have been flagged or removed (e.g., RFI excision).  The
Toeplitz fast-path is replaced by the full
:func:`~hydra_tod.flicker_model.flicker_cov_general` covariance matrix
and a Cholesky-based log-likelihood.

Public API
----------
flicker_sampler
    High-level dispatcher — **use this function directly**.
flicker_log_post_JAX
    Builds a JAX-compatible log-posterior closure (used by NUTS).
flicker_likeli_func
    Builds a log-likelihood closure (used by emcee path).
log_likeli_emu
    Evaluates the log-likelihood directly (NumPy, non-differentiable).
log_likeli_emu_jax
    Evaluates the log-likelihood with JAX (differentiable).

Typical usage
-------------
.. code-block:: python

    from hydra_tod.noise_sampler_fixed_fc import flicker_sampler

    noise_sample = flicker_sampler(
        TOD=tod,
        gains=gains,
        Tsys=tsys,
        init_params=[logf0_init, alpha_init],
        n_samples=1,
        jeffreys=True,
        sampler="emcee",
        consecutive=True,
    )

See Also
--------
hydra_tod.noise_sampler_old : Legacy emcee-only noise sampler.
hydra_tod.flicker_model : Analytic correlation and covariance functions.
hydra_tod.full_Gibbs_sampler : Orchestrates all Gibbs steps.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Optional

from scipy.linalg import solve_toeplitz
from .utils import lag_list, log_likeli, log_likeli_general
from .mcmc_sampler import mcmc_sampler
import jax.numpy as jnp
from jax.scipy.linalg import solve
import jax
from .nuts_sampler import NUTS_sampler
from .bayes_util import (
    wrap_with_priors,
    make_transforms,
    constrained_to_unconstrained,
    unconstrained_to_constrained,
)

# if the emulator of the correlation function exists, load it
# otherwise, use flicker_cov_vec
try:
    import pickle
    import os
    from jax import jit

    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    corr_emulator_path = os.path.join(module_dir, "flicker_corr_emulator.pkl")
    logdet_emulator_path = os.path.join(module_dir, "flicker_logdet_emulator.pkl")

    # Import the class definition before unpickling
    from .flicker_model import FlickerCorrEmulator  # noqa: F401

    # Load the emulator
    with open(corr_emulator_path, "rb") as f:
        flicker_cov = pickle.load(f)

    from .flicker_model import LogDetEmulator  # noqa: F401

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
    from .flicker_model import flicker_cov_vec

    use_emulator = False

from .flicker_model import flicker_cov_general, flicker_corr


def log_likeli_emu(
    logf0: float,
    alpha: float,
    data: NDArray[np.floating],
    DC_scaler: float = 1.0,
    consecutive: bool = True,
    time_list: NDArray[np.floating] | None = None,
) -> float:
    """
    Evaluate the Gaussian log-likelihood using the pre-trained emulator.

    Uses the emulated flicker covariance and log-determinant to compute
    the log-likelihood via Levinson-style Toeplitz solve. Falls back to
    ``-inf`` when the solve fails (e.g. non-positive-definite matrix).

    Parameters
    ----------
    logf0 : float
        Log10 of the knee angular frequency.
    alpha : float
        Spectral index of the 1/f noise power spectrum.
    data : NDArray[np.floating]
        Mean-centred residual data vector ``(d / g / Tsys - 1)``.
    DC_scaler : float, optional
        Multiplicative DC gain factor applied to the data before
        evaluating the likelihood. Default is ``1.0``.
    consecutive : bool, optional
        If ``True`` (default), use the efficient Toeplitz solve.  If
        ``False``, build the full covariance matrix from *time_list* and
        use Cholesky decomposition.
    time_list : NDArray[np.floating] or None, optional
        Observation time stamps (seconds).  Required when
        ``consecutive=False``; ignored otherwise.

    Returns
    -------
    float
        Scalar log-likelihood value, or ``-inf`` on failure.

    Notes
    -----
    This function uses NumPy for the Toeplitz solve and is therefore
    *not* JAX-differentiable. For a fully JAX-compatible version see
    :func:`log_likeli_emu_jax`.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """

    corr_list = flicker_cov_jax(logf0, alpha)[0]
    data = np.asarray(data) / DC_scaler

    if not consecutive:
        # Build full covariance from emulated correlation at actual time lags
        time_list = np.asarray(time_list)
        tau_matrix = np.abs(time_list[:, None] - time_list[None, :])
        n = len(data)
        cov = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                tau_idx = int(round(tau_matrix[i, j]))
                if tau_idx < len(corr_list):
                    cov[i, j] = corr_list[tau_idx]
                else:
                    # Lag exceeds emulator range — use the analytic function
                    cov[i, j] = flicker_corr(
                        tau_matrix[i, j], 10.0**logf0, 10.0**flicker_cov.logfc, alpha
                    )
                cov[j, i] = cov[i, j]
        return log_likeli_general(cov, data)

    try:
        result = (
            np.dot(data, solve_toeplitz(corr_list, data))
            + flicker_logdet_jax(logf0, alpha)[0]
        )
        return -0.5 * result
    except Exception:
        return -np.inf


@jit
def _log_likeli_emu_jax_toeplitz(
    logf0: float,
    alpha: float,
    data: jnp.ndarray,
    DC_scaler: float = 1.0,
) -> jnp.ndarray:
    """JAX log-likelihood assuming consecutive (Toeplitz) covariance."""
    corr_list = flicker_cov_jax(logf0, alpha)[0]
    logdet = flicker_logdet_jax(logf0, alpha)[0]

    corr_list = jnp.asarray(corr_list)
    data = jnp.asarray(data / DC_scaler)

    n = len(data)
    indices = jnp.arange(n)
    toeplitz_matrix = corr_list[jnp.abs(indices[:, None] - indices[None, :])]

    solved = solve(toeplitz_matrix, data, assume_a="pos")
    quad_form = jnp.dot(data, solved)

    result = quad_form + logdet
    return -0.5 * result


def _make_log_likeli_emu_jax_general(
    time_list: NDArray[np.floating],
) -> Callable:
    """Build a JIT-compiled JAX log-likelihood for non-consecutive times.

    The time-difference index matrix is precomputed and baked into the
    closure so that the returned function can be JIT-compiled.
    """
    time_list = np.asarray(time_list, dtype=np.float64)
    tau_matrix = np.abs(time_list[:, None] - time_list[None, :])
    # Precompute integer lag indices (for indexing into emulator output)
    lag_indices = jnp.array(np.round(tau_matrix).astype(int))

    @jit
    def _log_likeli_general(
        logf0: float,
        alpha: float,
        data: jnp.ndarray,
        DC_scaler: float = 1.0,
    ) -> jnp.ndarray:
        corr_list = flicker_cov_jax(logf0, alpha)[0]
        corr_list = jnp.asarray(corr_list)
        data = jnp.asarray(data / DC_scaler)

        # Build general covariance from correlation vector at actual lags
        cov_matrix = corr_list[lag_indices]

        solved = solve(cov_matrix, data, assume_a="pos")
        quad_form = jnp.dot(data, solved)
        # Log-determinant via Cholesky
        L = jnp.linalg.cholesky(cov_matrix)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

        result = quad_form + logdet
        return -0.5 * result

    return _log_likeli_general


def log_likeli_emu_jax(
    logf0: float,
    alpha: float,
    data: jnp.ndarray,
    DC_scaler: float = 1.0,
) -> jnp.ndarray:
    """
    JAX-differentiable Gaussian log-likelihood using the emulator.

    Constructs the full Toeplitz covariance matrix in JAX and solves
    the linear system with ``jax.scipy.linalg.solve``, enabling
    automatic differentiation for gradient-based samplers (e.g. NUTS).

    Parameters
    ----------
    logf0 : float
        Log10 of the knee angular frequency.
    alpha : float
        Spectral index of the 1/f noise power spectrum.
    data : jnp.ndarray
        Mean-centred residual data vector.
    DC_scaler : float, optional
        Multiplicative DC gain factor. Default is ``1.0``.

    Returns
    -------
    jnp.ndarray
        Scalar (0-D) log-likelihood value.

    Notes
    -----
    Unlike :func:`log_likeli_emu` this builds the full dense Toeplitz
    matrix (O(N^2) memory) and is JIT-compiled. Suitable for moderate
    time-stream lengths where auto-differentiation is required.

    For non-consecutive time lists, use
    :func:`_make_log_likeli_emu_jax_general` to build a specialised
    JIT-compiled closure.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    return _log_likeli_emu_jax_toeplitz(logf0, alpha, data, DC_scaler)


# Define the likelihood function for the flicker noise model.
def flicker_likeli_func(
    data: NDArray[np.floating],
    gain: NDArray[np.floating],
    Tsys: NDArray[np.floating],
    logfc: float,
    time_list: Optional[NDArray[np.floating]] = None,
    wnoise_var: float = 2.5e-6,
    consecutive: bool = True,
) -> Callable[[NDArray[np.floating]], float]:
    """
    Construct a log-likelihood closure for the flicker noise model.

    Preprocesses the data by removing the gain and system temperature
    contributions, mean-centres the residual, and returns a callable
    that evaluates the Gaussian log-likelihood for ``(logf0, alpha)``.

    Parameters
    ----------
    data : NDArray[np.floating]
        1-D time-ordered data array.
    gain : NDArray[np.floating]
        1-D array of instrument gain values.
    Tsys : NDArray[np.floating]
        1-D array of system temperature values.
    logfc : float
        Log10 of the low-frequency cutoff angular frequency.
    time_list : NDArray[np.floating] or None, optional
        1-D array of observation timestamps (seconds). Required when the
        emulator is not available, or when ``consecutive=False``.
    wnoise_var : float, optional
        White noise variance. Default is ``2.5e-6``.
    consecutive : bool, optional
        If ``True`` (default), assume evenly-spaced time stamps and use
        efficient Toeplitz solvers.  If ``False``, build the full
        covariance matrix for non-consecutive time stamps using
        :func:`~hydra_tod.flicker_model.flicker_cov_general`.

    Returns
    -------
    log_like : callable
        A function ``log_like(params) -> float`` with
        ``params = [logf0, alpha]``.

    Notes
    -----
    Frequencies ``f0`` and ``fc`` are in angular frequency units
    (rad/s), differing from FFT frequencies by a factor of ``2 * pi``.

    When the pre-trained emulator is available it is used for faster
    evaluation; otherwise the analytic ``flicker_cov_vec`` is called.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    dvec = data / gain / Tsys - 1.0
    dvec = np.asarray(dvec, dtype=np.float64)
    dvec -= np.mean(dvec)

    if not consecutive:
        # Non-consecutive path: build full covariance at each evaluation
        _time_list = np.asarray(time_list, dtype=np.float64)

        if use_emulator:

            def log_like(params: NDArray[np.floating]) -> float:
                logf0, alpha = params
                return log_likeli_emu(
                    logf0, alpha, dvec, consecutive=False, time_list=_time_list
                )

        else:

            def log_like(params: NDArray[np.floating]) -> float:
                logf0, alpha = params
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
    if use_emulator:

        def log_like(params: NDArray[np.floating]) -> float:
            logf0, alpha = params
            return log_likeli_emu(logf0, alpha, dvec)

    else:
        tau_list = lag_list(time_list)

        def log_like(params: NDArray[np.floating]) -> float:
            logf0, alpha = params
            corr_list = flicker_cov_vec(
                tau_list, 10.0**logf0, 10.0**logfc, alpha, white_n_variance=wnoise_var
            )
            return log_likeli(corr_list, dvec)

    return log_like


def flicker_log_post_JAX(
    data: NDArray[np.floating],
    gain: NDArray[np.floating],
    Tsys: NDArray[np.floating],
    include_DC_Gain: bool = False,
    prior_func: Optional[Callable[[jnp.ndarray], float]] = None,
    jeffreys: bool = False,
    transform: bool = False,
    bounds: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
    consecutive: bool = True,
    time_list: Optional[NDArray[np.floating]] = None,
) -> Callable[[jnp.ndarray], float]:
    """
    Build a JAX-compatible log-posterior closure for flicker noise parameters.

    Constructs the preprocessed residual vector and returns a log-posterior
    function that combines the Gaussian likelihood (via emulator) with
    optional Jeffreys prior, user-supplied prior, and/or parameter-space
    transforms.

    Parameters
    ----------
    data : NDArray[np.floating]
        1-D time-ordered data array.
    gain : NDArray[np.floating]
        1-D instrument gain array.
    Tsys : NDArray[np.floating]
        1-D system temperature array.
    include_DC_Gain : bool, optional
        If ``True`` the model includes a DC gain parameter, making the
        parameter vector ``(DC_gain, logf0, alpha)`` (3-D). Default is
        ``False`` (2-D: ``(logf0, alpha)``).
    prior_func : callable or None, optional
        Additional log-prior function ``prior_func(params) -> float``.
    jeffreys : bool, optional
        If ``True``, add the Jeffreys prior computed from the Fisher
        information matrix. When enabled the JAX-differentiable
        likelihood (:func:`log_likeli_emu_jax`) is used. Default is
        ``False``.
    transform : bool, optional
        If ``True``, the returned function expects parameters in
        unconstrained space and internally maps them to constrained
        space via sigmoid-type bijectors. Default is ``False``.
    bounds : list of tuple or None, optional
        Per-parameter ``(low, high)`` bounds. ``None`` entries indicate
        no bound in that direction.
    consecutive : bool, optional
        If ``True`` (default), assume evenly-spaced time stamps.  If
        ``False``, build the full covariance matrix for non-consecutive
        time stamps.
    time_list : NDArray[np.floating] or None, optional
        Observation time stamps.  Required when ``consecutive=False``.

    Returns
    -------
    log_like : callable
        A function ``log_like(params) -> float`` returning the
        (possibly transformed) log-posterior value.

    Notes
    -----
    The closure is decorated by :func:`~hydra_tod.bayes_util.wrap_with_priors`
    which handles bounds checking, Jeffreys prior injection, and the
    constrained-to-unconstrained reparameterisation.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    # Convert inputs to JAX arrays
    data = jnp.asarray(data)
    gain = jnp.asarray(gain)
    Tsys = jnp.asarray(Tsys)

    # Preprocess data using JAX operations
    dvec = data / gain / Tsys - 1.0
    dvec = dvec - jnp.mean(dvec)  # Mean-center using JAX

    assert use_emulator

    if jeffreys is False:
        if not consecutive:
            _time_list = np.asarray(time_list, dtype=np.float64)

            def _log_ll(logf0, alpha, d, DC_scaler=1.0):
                return log_likeli_emu(
                    logf0,
                    alpha,
                    d,
                    DC_scaler=DC_scaler,
                    consecutive=False,
                    time_list=_time_list,
                )

            log_ll = _log_ll
        else:
            log_ll = log_likeli_emu
        jaxjit = False
    else:
        if not consecutive:
            log_ll = _make_log_likeli_emu_jax_general(time_list)
        else:
            log_ll = log_likeli_emu_jax
        jaxjit = True

    if include_DC_Gain:

        @wrap_with_priors(
            add_jeffreys=jeffreys,
            prior_func=prior_func,
            bounds=bounds,
            dim=3,
            jaxjit=jaxjit,
            transform=transform,
        )
        def log_like(params: jnp.ndarray) -> float:
            DC_gain, logf0, alpha = params
            likelihood = log_ll(logf0, alpha, dvec, DC_scaler=DC_gain)
            return likelihood

    else:

        @wrap_with_priors(
            add_jeffreys=jeffreys,
            prior_func=prior_func,
            bounds=bounds,
            dim=2,
            jaxjit=jaxjit,
            transform=transform,
        )
        def log_like(params: jnp.ndarray) -> float:
            logf0, alpha = params
            likelihood = log_ll(logf0, alpha, dvec)
            return likelihood

    return log_like


def flicker_sampler(
    TOD: NDArray[np.floating],
    gains: NDArray[np.floating],
    Tsys: NDArray[np.floating],
    init_params: Optional[NDArray[np.floating]] = None,
    n_samples: int = 1,
    include_DC_Gain: bool = False,
    prior_func: Optional[Callable[[jnp.ndarray], float]] = None,
    jeffreys: bool = True,
    bounds: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
    transform: bool = False,
    sampler: str = "emcee",
    rng_key: Optional[jax.Array] = None,
    consecutive: bool = True,
    time_list: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """
    Sample flicker noise parameters using either emcee or NUTS.

    High-level interface that constructs the log-posterior via
    :func:`flicker_log_post_JAX` and dispatches to the requested
    sampler backend.

    Parameters
    ----------
    TOD : NDArray[np.floating]
        1-D time-ordered data array.
    gains : NDArray[np.floating]
        1-D instrument gain array.
    Tsys : NDArray[np.floating]
        1-D system temperature array.
    init_params : NDArray[np.floating] or None, optional
        Initial parameter guess. Shape ``(2,)`` when ``include_DC_Gain``
        is ``False``, ``(3,)`` otherwise. If ``None``, defaults are
        chosen internally.
    n_samples : int, optional
        Number of posterior samples to draw. Default is ``1``.
    include_DC_Gain : bool, optional
        Whether to include a DC gain parameter. Default is ``False``.
    prior_func : callable or None, optional
        Additional log-prior function.
    jeffreys : bool, optional
        Whether to include the Jeffreys prior. Default is ``True``.
    bounds : list of tuple or None, optional
        Per-parameter ``(low, high)`` bounds.
    transform : bool, optional
        If ``True``, sample in unconstrained space using sigmoid
        bijectors. Only used with the ``"emcee"`` backend. Default is
        ``False``.
    sampler : str, optional
        Backend sampler: ``"emcee"`` (ensemble MCMC) or ``"nuts"``
        (No-U-Turn Sampler via NumPyro). Default is ``"emcee"``.
    rng_key : jax.Array or None, optional
        JAX PRNG key. Required for deterministic results with NUTS;
        a default key is used if ``None``.
    consecutive : bool, optional
        If ``True`` (default), assume evenly-spaced time stamps.  If
        ``False``, build the full covariance matrix for non-consecutive
        time stamps.
    time_list : NDArray[np.floating] or None, optional
        Observation time stamps.  Required when ``consecutive=False``.

    Returns
    -------
    sample : NDArray[np.floating]
        Posterior sample(s). Shape ``(ndim,)`` when ``n_samples == 1``,
        ``(n_samples, ndim)`` otherwise.

    Raises
    ------
    ValueError
        If *sampler* is not ``"emcee"`` or ``"nuts"``.

    Notes
    -----
    * When ``sampler="emcee"`` with ``transform=True`` the chain runs
      in unconstrained space and samples are mapped back before return.
    * When ``sampler="nuts"`` the ``transform`` and ``bounds`` options
      are handled inside :func:`~hydra_tod.nuts_sampler.NUTS_sampler`
      via NumPyro's built-in reparameterisation.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """

    # log_likeli_func = flicker_likeli_func(t_list, TOD, gains, Tsys, logfc, wnoise_var=wnoise_var, bounds=bounds)

    dim = 3 if include_DC_Gain else 2

    if sampler == "emcee":
        log_post = flicker_log_post_JAX(
            TOD,
            gains,
            Tsys,
            include_DC_Gain=include_DC_Gain,
            prior_func=prior_func,
            jeffreys=jeffreys,
            bounds=bounds,
            transform=transform,
            consecutive=consecutive,
            time_list=time_list,
        )
        if transform:
            tfms = make_transforms(bounds, dim)
            if init_params is not None:
                theta0 = jnp.atleast_1d(jnp.array(init_params))
                # Invert to z0
                z0 = constrained_to_unconstrained(theta0, tfms)
                print(f"Initial params in unconstrained space: {z0}")
            else:
                z0 = jnp.zeros(dim)
        else:
            z0 = init_params
        zsample = mcmc_sampler(
            log_post,
            z0,
            p_std=0.2,  # Large step size in unconstrained space
            nsteps=80,  # steps for each chain
            n_samples=n_samples,
            return_sampler=False,
        )
        if transform:
            sample, _ = unconstrained_to_constrained(zsample, tfms)
        else:
            sample = zsample

    elif sampler == "nuts":
        log_ll = flicker_log_post_JAX(
            TOD,
            gains,
            Tsys,
            include_DC_Gain=include_DC_Gain,
            prior_func=prior_func,
            jeffreys=jeffreys,
            bounds=None,
            transform=False,
            consecutive=consecutive,
            time_list=time_list,
        )
        sample = NUTS_sampler(
            log_ll,
            init_params=None,
            event_shape=(dim,),
            initial_warmup=1500,
            max_warmup=5000,
            N_samples=n_samples,
            target_r_hat=1.01,
            single_return=True,
            N_chains=4,
            rng_key=rng_key,
            prior_type=None,
            bounds=bounds,
        )
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    return sample

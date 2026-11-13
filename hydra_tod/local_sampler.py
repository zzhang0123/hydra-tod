from __future__ import annotations

"""
Local parameter sampler using JAX and NUTS.

Jointly samples gain, noise-diode, and local temperature parameters for a
single TOD using the No-U-Turn Sampler (NUTS) with JAX-accelerated
likelihood evaluation.

References
----------
Zhang et al. (2026), RASTI, rzag024.
"""

import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import jit
from numpy.typing import NDArray
import numpy as np

from .nuts_sampler import NUTS_sampler
from .bayes_util import wrap_with_priors

try:
    import pickle
    import os

    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    corr_emulator_path = os.path.join(module_dir, "flicker_corr_emulator.pkl")
    logdet_emulator_path = os.path.join(module_dir, "flicker_logdet_emulator.pkl")

    # Import the class definition before unpickling
    from .flicker_model import FlickerCorrEmulator  # noqa: F401

    # Load the emulator
    with open(corr_emulator_path, "rb") as f:
        flicker_cov = pickle.load(f)
except (FileNotFoundError, ImportError, AttributeError) as e:
    print(f"Error loading flicker covariance emulator: {e}")
    flicker_cov = None

flicker_cov_jax = None
if flicker_cov is not None:
    flicker_cov_jax = jit(flicker_cov.create_jax())


def log_likelihood(
    params: jnp.ndarray,
    data: jnp.ndarray,
    Tsys_sky: jnp.ndarray,
    gain_proj: jnp.ndarray,
    Tnd_vec: jnp.ndarray,
    Tloc_proj: jnp.ndarray,
    logf0: float,
    alpha: float,
) -> float:
    r"""
    Compute the log-likelihood for local parameters given fixed noise params.

    Evaluates :math:`-\tfrac{1}{2}\,\mathbf{d}^\top C_n^{-1}\,\mathbf{d}`
    where :math:`\mathbf{d}` is the normalised residual after subtracting
    the deterministic signal model and :math:`C_n` is the flicker-noise
    Toeplitz covariance matrix.

    Parameters
    ----------
    params : jnp.ndarray
        Parameter vector of length 9:
        ``params[:4]`` — log-gain polynomial coefficients,
        ``params[4]`` — log noise-diode amplitude,
        ``params[5:]`` — local temperature polynomial coefficients.
    data : jnp.ndarray
        Observed TOD of shape ``(ntime,)``.
    Tsys_sky : jnp.ndarray
        Sky contribution to system temperature, shape ``(ntime,)``.
    gain_proj : jnp.ndarray
        Gain polynomial basis matrix, shape ``(ntime, 4)``.
    Tnd_vec : jnp.ndarray
        Noise-diode switching vector, shape ``(ntime,)``.
    Tloc_proj : jnp.ndarray
        Local temperature polynomial basis, shape ``(ntime, n_loc)``.
    logf0 : float
        Log-10 knee frequency of the flicker noise.
    alpha : float
        Spectral index of the flicker noise power spectrum.

    Returns
    -------
    ll : float
        Log-likelihood value (un-normalised, omits the log-determinant).
    """
    p_gain = params[:4]
    gains = jnp.exp(gain_proj @ p_gain)

    Tnd = jnp.exp(params[4])
    Tsys_nd = Tnd * Tnd_vec

    p_loc = params[5:]
    Tsys_loc = jnp.exp(Tloc_proj @ p_loc)

    d_vec = data / gains / (Tsys_loc + Tsys_nd + Tsys_sky) - 1.0

    if flicker_cov_jax is None:
        raise RuntimeError(
            "Flicker covariance emulator not loaded. "
            "local_sampler requires the .pkl emulator files."
        )
    corr_list = flicker_cov_jax(logf0, alpha)[0]

    n = len(d_vec)
    # Build Toeplitz matrix
    indices = jnp.arange(n)
    toeplitz_matrix = corr_list[jnp.abs(indices[:, None] - indices[None, :])]

    # Solve the linear system
    solved = solve(toeplitz_matrix, d_vec, assume_a="pos")
    quad_form = jnp.dot(d_vec, solved)

    return -0.5 * quad_form


# JIT-compiled version for repeated calls with same static arguments
@jit
def log_likelihood_jit(
    params: jnp.ndarray,
    data: jnp.ndarray,
    Tsys_sky: jnp.ndarray,
    gain_proj: jnp.ndarray,
    Tnd_vec: jnp.ndarray,
    Tloc_proj: jnp.ndarray,
    logf0: float,
    alpha: float,
) -> float:
    """
    JIT-compiled version of :func:`log_likelihood`.

    Parameters and return value are identical to :func:`log_likelihood`.
    This wrapper enables JAX tracing and XLA compilation for repeated
    evaluations with the same array shapes.
    """
    return log_likelihood(
        params, data, Tsys_sky, gain_proj, Tnd_vec, Tloc_proj, logf0, alpha
    )


def local_params_sampler(
    data: NDArray[np.float64],
    Tsys_sky: NDArray[np.float64],
    gain_proj: NDArray[np.float64],
    Tnd_vec: NDArray[np.float64],
    Tloc_proj: NDArray[np.float64],
    noise_params: tuple[float, float],
    rng_key: jnp.ndarray | None = None,
    add_jeffreys: bool = True,
    prior_func: callable | None = None,
    bounds: NDArray[np.float64] | None = None,
    jaxjit: bool = True,
) -> NDArray[np.float64]:
    r"""
    Sample local parameters (gains, noise diode, local temps) via NUTS.

    Constructs a JAX-accelerated log-posterior from the TOD likelihood and
    optional priors, then draws samples using the No-U-Turn Sampler.

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed TOD, shape ``(ntime,)``.
    Tsys_sky : NDArray[np.float64]
        Sky contribution to system temperature, shape ``(ntime,)``.
    gain_proj : NDArray[np.float64]
        Gain polynomial basis matrix, shape ``(ntime, 4)``.
    Tnd_vec : NDArray[np.float64]
        Noise-diode switching vector, shape ``(ntime,)``.
    Tloc_proj : NDArray[np.float64]
        Local temperature polynomial basis, shape ``(ntime, n_loc)``.
    noise_params : tuple of float
        ``(logf0, alpha)`` — fixed noise parameters.
    rng_key : jnp.ndarray or None, optional
        JAX PRNG key.  If ``None``, a default key is used.
    add_jeffreys : bool, optional
        Whether to add a Jeffreys prior for scale parameters.
    prior_func : callable or None, optional
        Additional log-prior function ``f(params) -> float``.
    bounds : NDArray or None, optional
        Parameter bounds array of shape ``(9, 2)``.
    jaxjit : bool, optional
        Whether to JIT-compile the likelihood. Default is ``True``.

    Returns
    -------
    sample : NDArray[np.float64]
        Posterior sample of shape ``(9,)``.
    """

    # Extract noise parameters for JIT compilation
    logf0, alpha = noise_params

    # Turn numpy objects to jnp objects
    data = jnp.asarray(data)
    Tsys_sky = jnp.asarray(Tsys_sky)
    gain_proj = jnp.asarray(gain_proj)
    Tnd_vec = jnp.asarray(Tnd_vec)
    Tloc_proj = jnp.asarray(Tloc_proj)

    # Create a wrapper that uses the JIT-compiled function
    @wrap_with_priors(
        add_jeffreys=add_jeffreys,
        prior_func=prior_func,
        bounds=bounds,
        dim=9,
        jaxjit=jaxjit,
    )
    def likelihood_wrapper(params):
        return log_likelihood_jit(
            params, data, Tsys_sky, gain_proj, Tnd_vec, Tloc_proj, logf0, alpha
        )

    sample = NUTS_sampler(
        likelihood_wrapper,
        init_params=None,
        log_likeli_args=(),
        event_shape=(9,),
        initial_warmup=1500,
        max_warmup=5000,
        N_samples=1000,
        target_r_hat=1.01,
        single_return=True,
        N_chains=4,
        rng_key=rng_key,
        prior_type=None,
    )

    return sample

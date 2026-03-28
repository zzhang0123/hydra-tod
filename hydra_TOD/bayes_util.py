from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from functools import wraps, partial
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to, IdentityTransform, Transform


def make_transforms(
    bounds: Optional[list[tuple[Optional[float], Optional[float]]]],
    dim: int,
) -> list[Transform]:
    """
    Build a list of per-parameter bijective transforms from bounds.

    Each transform maps an unconstrained real value to the corresponding
    constrained interval, enabling samplers (e.g. NUTS, emcee) to
    operate in :math:`\\mathbb{R}^D` while the likelihood is evaluated
    in the original bounded parameter space.

    Parameters
    ----------
    bounds : list of tuple or None
        Per-parameter ``(low, high)`` bounds. ``None`` entries within a
        tuple indicate no bound in that direction. If *bounds* itself
        is ``None``, identity transforms are returned for all
        dimensions. A single ``(low, high)`` tuple (not nested) is
        broadcast to all *dim* parameters.
    dim : int
        Number of parameters.

    Returns
    -------
    list of Transform
        A list of length *dim* containing NumPyro ``Transform`` objects.

    Raises
    ------
    ValueError
        If the length of *bounds* does not match *dim*.

    Notes
    -----
    The underlying bijectors are obtained from
    ``numpyro.distributions.transforms.biject_to`` applied to the
    appropriate constraint (``greater_than``, ``less_than``, or
    ``interval``).

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    if bounds is None:
        return [IdentityTransform()] * dim

    # Normalize bounds to length dim
    if (
        isinstance(bounds, tuple)
        and len(bounds) == 2
        and not isinstance(bounds[0], (list, tuple))
    ):
        low, high = bounds
        bounds = [(low, high)] * dim

    if len(bounds) != dim:
        raise ValueError(f"bounds length {len(bounds)} != dim {dim}")

    tfms: list[Transform] = []
    for low, high in bounds:
        if low is None and high is None:
            tfms.append(IdentityTransform())
        elif low is None:
            tfms.append(biject_to(constraints.less_than(high)))
        elif high is None:
            tfms.append(biject_to(constraints.greater_than(low)))
        else:
            tfms.append(biject_to(constraints.interval(low, high)))
    return tfms


def unconstrained_to_constrained(
    z: jnp.ndarray,
    tfms: list[Transform],
) -> Tuple[jnp.ndarray, float]:
    """
    Map unconstrained parameters to constrained space.

    Applies the per-parameter bijectors in *tfms* element-wise and
    accumulates the log absolute Jacobian determinant needed for
    correct density evaluation under the change of variables.

    Parameters
    ----------
    z : jnp.ndarray
        Parameter vector in unconstrained :math:`\\mathbb{R}^D` space.
    tfms : list of Transform
        Per-parameter bijective transforms (from :func:`make_transforms`).

    Returns
    -------
    theta : jnp.ndarray
        Parameter vector in constrained space.
    logabsdet : float
        Log absolute determinant of the Jacobian
        :math:`\\sum_i \\log |\\partial \\theta_i / \\partial z_i|`.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    z_vec = jnp.atleast_1d(z)
    thetas = []
    logabsdet = 0.0
    for zi, t in zip(z_vec, tfms):
        xi = t(zi)
        thetas.append(xi)
        logabsdet = logabsdet + t.log_abs_det_jacobian(zi, xi)
    theta = jnp.stack(thetas) if len(thetas) > 1 else thetas[0]
    return theta, logabsdet


def constrained_to_unconstrained(
    theta: jnp.ndarray,
    tfms: list[Transform],
) -> jnp.ndarray:
    """
    Map constrained parameters to unconstrained space.

    Applies the inverse of each per-parameter bijector in *tfms*.

    Parameters
    ----------
    theta : jnp.ndarray
        Parameter vector in constrained space.
    tfms : list of Transform
        Per-parameter bijective transforms (from :func:`make_transforms`).

    Returns
    -------
    z : jnp.ndarray
        Parameter vector in unconstrained :math:`\\mathbb{R}^D` space.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    th_vec = jnp.atleast_1d(theta)
    zs = []
    for vi, t in zip(th_vec, tfms):
        zs.append(t.inv(vi))
    z = jnp.stack(zs) if len(zs) > 1 else zs[0]
    return z


def prepare_bounds(
    bounds: Union[
        tuple[float, float],
        list[tuple[float, float]],
    ],
    n_params: Optional[int] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Normalize parameter bounds to JAX arrays.

    Accepts either a single ``(lower, upper)`` tuple (broadcast to all
    parameters) or a list of per-parameter tuples, and returns two
    arrays of shape ``(n_params,)``.

    Parameters
    ----------
    bounds : tuple or list of tuples
        ``(lower, upper)`` scalar bounds or a list
        ``[(l1, u1), (l2, u2), ...]`` of per-parameter bounds.
    n_params : int or None, optional
        Number of parameters. Used to broadcast scalar bounds to vectors
        and to validate the length of per-parameter bounds.

    Returns
    -------
    lower : jnp.ndarray
        Lower bounds array.
    upper : jnp.ndarray
        Upper bounds array.

    Raises
    ------
    ValueError
        If *bounds* format is unrecognised or if the length does not
        match *n_params*.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    if isinstance(bounds, tuple) and len(bounds) == 2:
        lower, upper = bounds
        lower = jnp.asarray(lower)
        upper = jnp.asarray(upper)

        # Expand scalar bounds to vector bounds if n_params provided
        if n_params is not None and lower.ndim == 0 and upper.ndim == 0:
            lower = jnp.full((n_params,), lower)
            upper = jnp.full((n_params,), upper)

    elif isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
        # Already per-parameter bounds
        lower = jnp.array([b[0] for b in bounds])
        upper = jnp.array([b[1] for b in bounds])

        if n_params is not None and len(lower) != n_params:
            raise ValueError(
                f"Length of bounds ({len(lower)}) != n_params ({n_params})"
            )
    else:
        raise ValueError(
            "Bounds must be (lower, upper) or [(low1, high1), (low2, high2), ...]"
        )

    return lower, upper


@jax.jit
def check_bounds_jit(
    value: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
) -> jnp.ndarray:
    """
    JIT-compiled bounds check with pre-shaped lower and upper arrays.

    Parameters
    ----------
    value : jnp.ndarray
        Parameter value(s) to check.
    lower : jnp.ndarray
        Lower bounds (same shape as *value*).
    upper : jnp.ndarray
        Upper bounds (same shape as *value*).

    Returns
    -------
    jnp.ndarray
        Scalar boolean, ``True`` if all elements satisfy
        ``lower <= value <= upper``.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    value = jnp.asarray(value)
    return jnp.all((value >= lower) & (value <= upper))


def check_bounds(
    value: jnp.ndarray,
    bounds: Optional[Union[tuple[float, float], list[tuple[float, float]]]],
) -> bool:
    """
    Check whether *value* lies within *bounds*.

    Parameters
    ----------
    value : jnp.ndarray
        Parameter value(s) to check.
    bounds : tuple, list of tuples, or None
        Parameter bounds in any format accepted by
        :func:`prepare_bounds`. ``None`` means no bounds (always
        returns ``True``).

    Returns
    -------
    bool
        ``True`` if all elements of *value* are within bounds.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    if bounds is None:
        return True
    lower, upper = prepare_bounds(bounds, n_params=value.shape[-1])
    return jnp.all((value >= lower) & (value <= upper))


@jax.jit
def log_gaussian_prior(
    value: jnp.ndarray,
    mean: Union[float, jnp.ndarray] = 0.0,
    std: Union[float, jnp.ndarray] = 1.0,
) -> float:
    """
    Independent Gaussian prior log-probability (JIT-compiled).

    Parameters
    ----------
    value : jnp.ndarray
        Parameter value(s).
    mean : float or jnp.ndarray, optional
        Prior mean(s). Default is ``0.0``.
    std : float or jnp.ndarray, optional
        Prior standard deviation(s). Default is ``1.0``.

    Returns
    -------
    float
        Sum of per-dimension log-normal densities.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    value = jnp.atleast_1d(value)
    mean = jnp.atleast_1d(mean)
    return stats.norm.logpdf(value, loc=mean, scale=std).sum()


@jax.jit
def log_gaussian_prior_full(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    cov_inv: jnp.ndarray,
    log_det_cov: float = 0.0,
) -> float:
    """
    Multivariate Gaussian prior log-probability (JIT-compiled).

    Parameters
    ----------
    value : jnp.ndarray
        Parameter value(s) of shape ``(D,)``.
    mean : jnp.ndarray
        Prior mean of shape ``(D,)``.
    cov_inv : jnp.ndarray
        Inverse covariance matrix of shape ``(D, D)``.
    log_det_cov : float, optional
        Log determinant of the covariance matrix. Default is ``0.0``.

    Returns
    -------
    float
        Log multivariate normal density.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    value = jnp.atleast_1d(value)
    mean = jnp.atleast_1d(mean)

    d = value.size
    diff = value - mean
    quad_form = diff @ cov_inv @ diff
    return -0.5 * (d * jnp.log(2 * jnp.pi) + log_det_cov + quad_form)


@partial(jax.jit, static_argnums=(1,))
def log_jeffreys_prior_jax(
    value: jnp.ndarray,
    likelihood_func: Callable[[jnp.ndarray], float],
) -> float:
    """
    Compute the Jeffreys prior log-probability using JAX auto-diff.

    Uses ``jax.hessian`` to compute the Fisher information matrix as the
    negative Hessian of the log-likelihood, then returns
    :math:`\\tfrac{1}{2} \\log \\det I(\\theta)`.

    Parameters
    ----------
    value : jnp.ndarray
        Parameter value(s) in constrained space.
    likelihood_func : callable
        Log-likelihood function (must be JAX-differentiable).

    Returns
    -------
    float
        Log-Jeffreys-prior value, or ``-inf`` if the Fisher information
        matrix is not positive definite.

    Notes
    -----
    A small ridge regularisation (``1e-10 * I``) is added to the Fisher
    information matrix to improve numerical stability.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """

    # Compute Fisher Information Matrix as negative Hessian of log-likelihood
    hessian_fn = jax.hessian(likelihood_func)
    hessian = hessian_fn(value)
    fisher_info = -hessian
    fisher_info = jnp.atleast_2d(fisher_info)

    n = fisher_info.shape[0]
    regularization = 1e-10 * jnp.eye(n)
    fisher_info_reg = fisher_info + regularization
    sign, log_det = jnp.linalg.slogdet(fisher_info_reg)
    return jnp.where(sign > 0, 0.5 * log_det, -jnp.inf)


def wrap_with_priors(
    add_jeffreys: bool = False,
    prior_func: Optional[Callable[[jnp.ndarray], float]] = None,
    bounds: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
    dim: Optional[int] = None,
    jaxjit: bool = False,
    transform: bool = True,
) -> Callable[[Callable[..., float]], Callable[[jnp.ndarray], float]]:
    """
    Decorator factory that augments a log-likelihood with priors and transforms.

    The returned decorator wraps a bare log-likelihood function so that
    bounds checking (or sigmoid reparameterisation), an optional
    Jeffreys prior, and/or a user-supplied prior are applied
    transparently.

    Parameters
    ----------
    add_jeffreys : bool, optional
        If ``True``, add the Jeffreys prior computed via
        :func:`log_jeffreys_prior_jax`. Default is ``False``.
    prior_func : callable or None, optional
        Additional log-prior ``prior_func(params) -> float``.
    bounds : list of tuple or None, optional
        Per-parameter ``(low, high)`` bounds. Accepted formats:

        * ``None`` -- no bounds.
        * ``[(l1, u1), (l2, u2), ...]`` -- per-parameter bounds.
        * ``(scalar_low, scalar_high)`` -- broadcast to all dims.
    dim : int or None, optional
        Number of parameters. Required when *bounds* are scalar or
        ``None``.
    jaxjit : bool, optional
        If ``True``, JIT-compile the wrapped function. Default is
        ``False``.
    transform : bool, optional
        If ``True`` (and *bounds* is not ``None``), reparameterise
        from unconstrained to constrained space via sigmoid bijectors
        and add the log Jacobian determinant. When ``False``, hard
        bounds checking is used instead. Default is ``True``.

    Returns
    -------
    decorator : callable
        A decorator that transforms a ``log_likelihood(theta)`` function
        into a ``log_posterior(z)`` function.

    Examples
    --------
    >>> @wrap_with_priors(add_jeffreys=True, bounds=[(-6, -3), (1, 4)], dim=2)
    ... def log_likelihood(theta):
    ...     return some_computation(theta)

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    if bounds is None:
        transform = False
    if transform:
        tfms = make_transforms(bounds, dim)

    def decorator(func: Callable[..., float]) -> Callable[[jnp.ndarray], float]:
        @wraps(func)
        def wrapper(value: jnp.ndarray) -> float:
            # Reparameterize
            if transform:
                theta, logabsdet = unconstrained_to_constrained(value, tfms)
            else:
                if not check_bounds(value, bounds):
                    return -jnp.inf
                theta = value
                logabsdet = 0.0
            log_likelihood = func(theta) + logabsdet
            if prior_func is not None:
                log_likelihood += prior_func(value)
            if add_jeffreys:
                log_likelihood += log_jeffreys_prior_jax(value, func)
            return log_likelihood

        # JIT the wrapper function
        if jaxjit:
            return jax.jit(wrapper)
        return wrapper

    return decorator

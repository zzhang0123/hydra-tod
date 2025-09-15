import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from functools import wraps, partial
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to, IdentityTransform


def make_transforms(bounds, dim):
    """
    Build a per-parameter bijector list from bounds.
    bounds: list/tuple of (low, high) where low/high can be None for unbounded.
    dim: number of parameters.
    """
    if bounds is None:
        return [IdentityTransform()] * dim

    # Normalize bounds to length dim
    if isinstance(bounds, tuple) and len(bounds) == 2 and not isinstance(bounds[0], (list, tuple)):
        low, high = bounds
        bounds = [(low, high)] * dim

    if len(bounds) != dim:
        raise ValueError(f"bounds length {len(bounds)} != dim {dim}")

    tfms = []
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

def unconstrained_to_constrained(z, tfms):
    """
    Map unconstrained z (R^D) -> constrained theta using tfms, accumulate log|det J|.
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

def constrained_to_unconstrained(theta, tfms):
    """
    Map constrained theta -> unconstrained z using inverse transforms.
    """
    th_vec = jnp.atleast_1d(theta)
    zs = []
    for vi, t in zip(th_vec, tfms):
        zs.append(t.inv(vi))
    z = jnp.stack(zs) if len(zs) > 1 else zs[0]
    return z

def prepare_bounds(bounds, n_params=None):
    """
    Normalize bounds to arrays, optionally expanding scalars to length `n_params`.

    Args:
        bounds: tuple (lower, upper) or list of tuples [(l1, u1), (l2, u2), ...]
        n_params: optional int, number of parameters for broadcasting scalar bounds

    Returns:
        (lower, upper): jnp.arrays with shape (n_params,) if n_params is provided
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
            raise ValueError(f"Length of bounds ({len(lower)}) != n_params ({n_params})")
    else:
        raise ValueError(
            "Bounds must be (lower, upper) or [(low1, high1), (low2, high2), ...]"
        )

    return lower, upper

@jax.jit
def check_bounds_jit(value, lower, upper):
    """Check bounds with correctly shaped lower and upper."""
    value = jnp.asarray(value)
    return jnp.all((value >= lower) & (value <= upper))

def check_bounds(value, bounds):
    """Check if value is within bounds."""
    if bounds is None:
        return True
    lower, upper = prepare_bounds(bounds, n_params=value.shape[-1])
    return jnp.all((value >= lower) & (value <= upper))

@jax.jit
def log_gaussian_prior(value, mean=0.0, std=1.0):
    """
    JIT-compatible Gaussian prior log-probability for independent dimensions.
    """
    value = jnp.atleast_1d(value)
    mean = jnp.atleast_1d(mean)
    return stats.norm.logpdf(value, loc=mean, scale=std).sum()

@jax.jit
def log_gaussian_prior_full(value, mean, cov_inv, log_det_cov=0.0):
    """
    JIT-compatible multivariate Gaussian log-probability.
    Args:
        cov_inv: inverse covariance matrix
        log_det_cov: log determinant of covariance matrix
    """
    value = jnp.atleast_1d(value)
    mean = jnp.atleast_1d(mean)
    
    d = value.size
    diff = value - mean
    quad_form = diff @ cov_inv @ diff
    return -0.5 * (d * jnp.log(2 * jnp.pi) + log_det_cov + quad_form)

@partial(jax.jit, static_argnums=(1,))
def log_jeffreys_prior_jax(value, likelihood_func):
    """
    Compute the Jeffreys prior log-probability at `value`
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


def wrap_with_priors(add_jeffreys=False, prior_func=None, bounds=None, dim=None, jaxjit=False, transform=True):
    """
    Decorator that optionally adds Jeffreys prior and JITs the function.
    
    Args:
        add_jeffreys (bool): Whether to add Jeffreys prior to the function output
        prior_func (callable, optional): An additional prior function to include
        bounds (tuple, optional): Bounds for the parameter values. 
            Bounds should be specified as one of:
                1. a list/tuple of (lower, upper) pairs, e.g. [(0, 1), (None, 2), (0, 3), (None, None)].
                2. None.
                3. (scalar_lower, scalar_upper)
        dim: Number of parameters, required if bounds are scalars or None.

    Usage:
        @jit_with_jeffreys(add_jeffreys=True, prior_func=None, bounds=None)
        def log_likelihood_wrapper(value):
            return some_likelihood_computation(value)
    """
    if bounds is None:
        transform = False
    if transform:
        tfms = make_transforms(bounds, dim)

    def decorator(func):
        @wraps(func)
        def wrapper(value):
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


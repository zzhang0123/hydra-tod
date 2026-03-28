from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import gelman_rubin
from numpyro.distributions import constraints
from numpyro.distributions.transforms import Transform
from .bayes_util import (
    log_jeffreys_prior_jax,
    make_transforms,
    unconstrained_to_constrained,
    constrained_to_unconstrained,
)


class FlexiblePosterior(dist.Distribution):
    """
    NumPyro distribution wrapping an arbitrary log-likelihood with optional priors.

    This class enables the NUTS sampler to target a user-defined
    log-posterior built from:

    1. A log-likelihood function (required).
    2. An optional Jeffreys, Gaussian, or flat prior.
    3. Per-parameter bijective transforms derived from bounds, so that
       NUTS always operates in unconstrained :math:`\\mathbb{R}^D` space
       while the likelihood is evaluated in constrained space.

    Parameters
    ----------
    log_likeli_fn : callable
        Function ``log_likeli_fn(params, *args, **kwargs) -> float``
        returning the scalar log-likelihood.
    log_likeli_args : tuple, optional
        Extra positional arguments forwarded to *log_likeli_fn*.
    log_likeli_kwargs : dict or None, optional
        Extra keyword arguments forwarded to *log_likeli_fn*.
    event_shape : tuple of int, optional
        Shape of the parameter vector. ``()`` for a scalar,
        ``(D,)`` for a D-dimensional vector.
    prior_type : str or None, optional
        Type of prior to apply: ``"jeffreys"``, ``"gaussian"``, or
        ``None`` (flat / improper uniform).
    validate_args : bool, optional
        Whether to validate distribution arguments. Default is ``False``.
    bounds : list of tuple or None, optional
        Per-parameter ``(low, high)`` bounds used to build sigmoid
        bijectors. ``None`` entries indicate no bound.
    **prior_kwargs
        Additional keyword arguments forwarded to the prior method
        (e.g. ``mean``, ``std``, ``cov`` for the Gaussian prior).

    Notes
    -----
    The ``support`` is always set to ``constraints.real`` (or
    ``constraints.real_vector``) so that NumPyro/NUTS does **not** add
    its own transform layer. Constraint enforcement is handled
    internally via the bijectors built from *bounds*.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """

    reparametrized_params: list[str] = []

    def __init__(
        self,
        log_likeli_fn: Callable[..., float],
        log_likeli_args: tuple[Any, ...] = (),
        log_likeli_kwargs: Optional[Dict[str, Any]] = None,
        event_shape: tuple[int, ...] = (),
        prior_type: Optional[str] = None,
        validate_args: bool = False,
        bounds: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
        **prior_kwargs: Any,
    ) -> None:
        self.log_likeli_fn = log_likeli_fn
        self.log_likeli_args = log_likeli_args or ()
        self.log_likeli_kwargs = log_likeli_kwargs or {}

        self.prior_type = prior_type
        self.prior_kwargs = prior_kwargs

        # Dimension
        if event_shape == ():
            dim = 1
        else:
            if not (isinstance(event_shape, tuple) and len(event_shape) == 1):
                raise ValueError("event_shape must be () or (D,)")
            dim = event_shape[0]

        # Per-parameter transforms from bounds, sampling stays in R^D
        self.bounds = bounds
        self.tfms: list[Transform] = make_transforms(bounds, dim)

        # Keep support as real so NumPyro/NUTS does NOT add its own transform
        self.support = constraints.real if dim == 1 else constraints.real_vector

        super().__init__(event_shape=event_shape, validate_args=validate_args)

    def _log_likelihood_wrapper(self, theta: jnp.ndarray) -> float:
        """Evaluate the log-likelihood forwarding stored args/kwargs.

        Parameters
        ----------
        theta : jnp.ndarray
            Parameter vector in **constrained** space.

        Returns
        -------
        float
            Scalar log-likelihood value.
        """
        return self.log_likeli_fn(
            theta, *self.log_likeli_args, **self.log_likeli_kwargs
        )

    def log_jeffreys_prior(self, theta: jnp.ndarray) -> float:
        """Compute the log-Jeffreys-prior at *theta*.

        Parameters
        ----------
        theta : jnp.ndarray
            Parameter vector in constrained space.

        Returns
        -------
        float
            Log-Jeffreys-prior value.
        """
        return log_jeffreys_prior_jax(theta, self._log_likelihood_wrapper)

    def log_gaussian_prior(
        self,
        theta: jnp.ndarray,
        mean: Union[float, jnp.ndarray] = 0.0,
        std: Union[float, jnp.ndarray] = 1.0,
        cov: Optional[jnp.ndarray] = None,
    ) -> float:
        """
        Compute the Gaussian prior log-probability at *theta*.

        Parameters
        ----------
        theta : jnp.ndarray
            Parameter value(s).
        mean : float or jnp.ndarray, optional
            Prior mean. Default is ``0.0``.
        std : float or jnp.ndarray, optional
            Prior standard deviation for an independent (diagonal)
            Gaussian. Ignored when *cov* is provided. Default is ``1.0``.
        cov : jnp.ndarray or None, optional
            Full covariance matrix for a multivariate Gaussian prior.

        Returns
        -------
        float
            Log-prior probability.
        """

        theta = jnp.atleast_1d(theta)
        mean = jnp.atleast_1d(mean)

        if cov is None:
            # Independent Gaussian (diagonal covariance)
            return dist.Normal(mean, std).log_prob(theta).sum()
        else:
            # Full multivariate Gaussian
            mvn = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
            return mvn.log_prob(theta)

    def log_prob(self, value: jnp.ndarray) -> float:
        """
        Evaluate the log-posterior at *value* (unconstrained space).

        Applies the bijective transform to map from unconstrained to
        constrained space, evaluates the log-likelihood and the chosen
        prior, and adds the log absolute Jacobian determinant.

        Parameters
        ----------
        value : jnp.ndarray
            Parameter value(s) in unconstrained space.

        Returns
        -------
        float
            Log-posterior value
            ``log(prior) + log(likelihood) + log|det J|``.

        Raises
        ------
        ValueError
            If ``prior_type`` is not recognised.
        """
        # Reparameterize
        theta, logabsdet = unconstrained_to_constrained(value, self.tfms)

        # Log-likelihood in constrained space
        log_likelihood = self._log_likelihood_wrapper(theta)

        pt = self.prior_type
        pk = self.prior_kwargs

        # Compute the prior component based on type
        if pt is None:
            log_prior = 0.0
        elif pt == "jeffreys":
            log_prior = self.log_jeffreys_prior(theta)
        elif pt == "gaussian":
            log_prior = self.log_gaussian_prior(theta, **pk)
        else:
            raise ValueError(f"Unknown prior_type: {pt}")

        # Return log(prior) + log(likelihood)
        return log_prior + log_likelihood + logabsdet


def NUTS_sampler(
    log_likeli_fn: Callable[..., float],
    init_params: Optional[jnp.ndarray] = None,
    log_likeli_args: tuple[Any, ...] = (),
    log_likeli_kwargs: Optional[Dict[str, Any]] = None,
    event_shape: tuple[int, ...] = (),
    initial_warmup: int = 1500,
    max_warmup: int = 5000,
    N_samples: int = 1000,
    target_r_hat: float = 1.01,
    single_return: bool = False,
    N_chains: int = 4,
    rng_key: Optional[jax.Array] = None,
    prior_type: Optional[str] = None,
    bounds: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
    **prior_kwargs: Any,
) -> Union[jnp.ndarray, Tuple[MCMC, jnp.ndarray]]:
    """
    Run the No-U-Turn Sampler (NUTS) with adaptive warmup via NumPyro.

    Wraps the target log-posterior inside a :class:`FlexiblePosterior`
    NumPyro distribution and runs NUTS with progressively increasing
    warmup until the Gelman-Rubin convergence diagnostic drops below
    *target_r_hat* or *max_warmup* steps are exhausted.

    Parameters
    ----------
    log_likeli_fn : callable
        Log-likelihood function ``log_likeli_fn(params, *args, **kwargs) -> float``.
    init_params : jnp.ndarray or None, optional
        Initial parameter values in constrained space. If ``None``,
        NUTS chooses its own initialisation.
    log_likeli_args : tuple, optional
        Extra positional arguments forwarded to *log_likeli_fn*.
    log_likeli_kwargs : dict or None, optional
        Extra keyword arguments forwarded to *log_likeli_fn*.
    event_shape : tuple of int, optional
        Shape of the parameter vector (e.g. ``(2,)``).
    initial_warmup : int, optional
        Number of warmup steps in the first round. Default is ``1500``.
    max_warmup : int, optional
        Maximum total warmup steps across all rounds. Default is ``5000``.
    N_samples : int, optional
        Number of post-warmup samples per chain. Default is ``1000``.
    target_r_hat : float, optional
        Gelman-Rubin :math:`\\hat R` convergence threshold. Default is
        ``1.01``.
    single_return : bool, optional
        If ``True``, return only the last sample from the first chain
        (mapped to constrained space). Useful inside a Gibbs step.
        Default is ``False``.
    N_chains : int, optional
        Number of parallel chains. Default is ``4``.
    rng_key : jax.Array or None, optional
        JAX PRNG key. If ``None`` a default key is used and a warning
        is printed.
    prior_type : str or None, optional
        Prior type forwarded to :class:`FlexiblePosterior`.
    bounds : list of tuple or None, optional
        Per-parameter ``(low, high)`` bounds.
    **prior_kwargs
        Additional keyword arguments forwarded to
        :class:`FlexiblePosterior` (e.g. ``mean``, ``std`` for
        Gaussian prior).

    Returns
    -------
    jnp.ndarray or tuple of (MCMC, jnp.ndarray)
        * If ``single_return=True``: a 1-D array of shape ``(dim,)``
          containing the last sample mapped to constrained space.
        * Otherwise: a tuple ``(mcmc, theta_samples)`` where *mcmc* is
          the NumPyro ``MCMC`` object and *theta_samples* has shape
          ``(N_chains * N_samples, dim)`` in constrained space.

    Notes
    -----
    The adaptive warmup doubles the warmup budget (factor 1.5x each
    round) and continues from the last sampler state rather than
    restarting, making it significantly more efficient than a
    from-scratch restart.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """

    # Dimension
    if event_shape == ():
        dim = 1
    else:
        dim = event_shape[0]

    def model() -> None:
        parameters = numpyro.sample(
            "parameters",
            FlexiblePosterior(
                log_likeli_fn,
                log_likeli_args=log_likeli_args,
                log_likeli_kwargs=log_likeli_kwargs,
                event_shape=event_shape,
                prior_type=prior_type,
                bounds=bounds,
                **prior_kwargs,
            ),
        )

    nuts_kernel = NUTS(model)

    if rng_key is None:
        # Provide a default but warn the user
        print("Warning: No rng_key provided to NUTS_sampler, using default key")
        rng_key = jax.random.PRNGKey(0)

    key = rng_key  # Use the provided key directly

    tfms: list[Transform] = make_transforms(bounds, dim)

    def map_batch(z_batch: jnp.ndarray) -> jnp.ndarray:
        """Map a batch of unconstrained samples to constrained space.

        Parameters
        ----------
        z_batch : jnp.ndarray
            Array of shape ``(M, dim)`` in unconstrained space.

        Returns
        -------
        jnp.ndarray
            Array of shape ``(M, dim)`` in constrained space.
        """
        thetas = []
        for zi in z_batch:
            theta_i, _ = unconstrained_to_constrained(zi, tfms)
            thetas.append(theta_i)
        return jnp.stack(thetas)

    # Prepare initial state in unconstrained space (z), broadcasting to chains
    init_dict: Optional[Dict[str, jnp.ndarray]] = None
    if init_params is not None:
        theta0 = jnp.atleast_1d(jnp.array(init_params))
        # Invert to z0
        z0 = constrained_to_unconstrained(theta0, tfms)
        # Broadcast across chains with small noise in z-space
        if dim == 1:
            z0 = jnp.atleast_1d(z0)
            noise = 1 * jax.random.normal(key, shape=(N_chains,))
            z0_chains = z0[0] + noise
            init_dict = {"parameters": z0_chains}
        else:
            noise = 1 * jax.random.normal(key, shape=(N_chains, dim))
            z0_chains = z0[None, :] + noise
            init_dict = {"parameters": z0_chains}

    # Track total warmup done and accumulated samples
    total_warmup: int = 0
    current_warmup: int = initial_warmup
    mcmc: Optional[MCMC] = None

    while current_warmup <= max_warmup:
        print(
            f"Running warmup round: {current_warmup} additional steps (total warmup: {total_warmup + current_warmup})"
        )

        if mcmc is None:
            # First round: start from scratch
            mcmc = MCMC(
                nuts_kernel,
                num_warmup=current_warmup,
                num_samples=N_samples,
                num_chains=N_chains,
                progress_bar=False,
            )
            mcmc.run(key, init_params=init_dict)
        else:
            # Continue from the last state!
            # Get the final state from previous run
            last_state = mcmc.last_state
            # Create a new MCMC with additional warmup
            mcmc_continue = MCMC(
                nuts_kernel,
                num_warmup=current_warmup,
                num_samples=N_samples,
                num_chains=N_chains,
                progress_bar=False,
            )
            # Continue from the last state instead of starting fresh
            mcmc_continue.run(
                key,
                init_params=last_state.z,
                extra_fields=("potential_energy", "adapt_state"),
            )
            mcmc = mcmc_continue  # Update our reference

        total_warmup += current_warmup

        # Check convergence
        samples = mcmc.get_samples(group_by_chain=True)
        r_hat = {k: gelman_rubin(v) for k, v in samples.items()}
        max_r_hat = max([float(jnp.max(v)) for v in r_hat.values()])
        print(f"Max R-hat: {max_r_hat:.4f} (after {total_warmup} total warmup steps)")

        if max_r_hat < target_r_hat:
            print(f"Convergence achieved with {total_warmup} total warmup steps!")
            if single_return:
                last_sample = samples["parameters"][0, -1]
                return unconstrained_to_constrained(last_sample, tfms)[0]
            z = samples[
                "parameters"
            ]  # shape: (chains, samples, D) in unconstrained space
            z_flat = z.reshape(-1, z.shape[-1])
            theta_samples = map_batch(z_flat)  # shape: (chains*samples, D)
            return mcmc, theta_samples

        # Increase warmup for next round
        current_warmup = int(current_warmup * 1.5)

    print(
        f"Did not achieve convergence within {max_warmup} maximum additional warmup steps"
    )
    print(f"   Total warmup used: {total_warmup} steps")

    if single_return:
        last_sample = samples["parameters"][0, -1]
        return unconstrained_to_constrained(last_sample, tfms)[0]
    z = samples["parameters"]  # shape: (chains, samples, D) in unconstrained space
    z_flat = z.reshape(-1, z.shape[-1])
    theta_samples = map_batch(z_flat)  # shape: (chains*samples, D)
    return mcmc, theta_samples

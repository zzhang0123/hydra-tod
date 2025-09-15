import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import gelman_rubin
from numpyro.distributions import constraints
from bayes_util import log_jeffreys_prior_jax, make_transforms, unconstrained_to_constrained, constrained_to_unconstrained


class FlexiblePosterior(dist.Distribution):
    reparametrized_params = []

    def __init__(self, log_likeli_fn, 
                 log_likeli_args=(), log_likeli_kwargs=None, 
                 event_shape=(), prior_type=None, validate_args=False, 
                 bounds=None, **prior_kwargs):
        """
        Flexible Jeffreys prior implementation.
        
        Args:
            log_likeli_fn: function(parameters, *args, **kwargs) -> scalar log-likelihood
            support: constraint for the parameter space (default: constraints.real)
            log_likeli_args: tuple of additional positional arguments for log_likeli_fn
            log_likeli_kwargs: dict of additional keyword arguments for log_likeli_fn
            event_shape: shape of the parameter
            validate_args: whether to validate arguments
        """
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
        self.tfms = make_transforms(bounds, dim)

        # Keep support as real so NumPyro/NUTS does NOT add its own transform
        self.support = constraints.real if dim == 1 else constraints.real_vector
            
        super().__init__(event_shape=event_shape, validate_args=validate_args)

    def _log_likelihood_wrapper(self, theta):
        """Wrapper to handle additional arguments for log_likeli_fn"""
        return self.log_likeli_fn(theta, *self.log_likeli_args, **self.log_likeli_kwargs)

    def log_jeffreys_prior(self, theta):
        return log_jeffreys_prior_jax(theta, self._log_likelihood_wrapper)

    def log_gaussian_prior(self, theta, mean=0.0, std=1.0, cov=None):
        """
        Compute Gaussian prior log-probability at `theta`

        Args:
            theta: parameter value(s), array-like
            mean: prior mean (scalar or array)
            std: prior std for diagonal Gaussian (ignored if cov is provided)
            cov: full covariance matrix for multivariate Gaussian
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

    def log_prob(self, value):
        """
        Returns the log-probability at `value` = log(prior) + log(likelihood)
        
        Args:
            value: parameter value(s)
            prior_type: "jeffreys", "uniform", or "gaussian"
            **prior_kwargs: additional arguments for the specific prior
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

def NUTS_sampler(log_likeli_fn, 
                 init_params=None,
                 log_likeli_args=(), 
                 log_likeli_kwargs=None, 
                 event_shape=(), 
                 initial_warmup=1500, 
                 max_warmup=5000, 
                 N_samples=1000,
                 target_r_hat=1.01, 
                 single_return=False,
                 N_chains=4,
                 rng_key=None,  
                 prior_type=None, 
                 bounds=None,
                 **prior_kwargs):
    """
    Run MCMC with adaptive warmup that CONTINUES from previous warmup state.
    This is much more efficient than restarting warmup from scratch each time.
    """

    # Dimension
    if event_shape == ():
        dim = 1
    else:
        dim = event_shape[0]
    
    def model():
        parameters = numpyro.sample(
            "parameters", 
            FlexiblePosterior(
                log_likeli_fn, 
                log_likeli_args=log_likeli_args, 
                log_likeli_kwargs=log_likeli_kwargs, 
                event_shape=event_shape, 
                prior_type=prior_type,
                bounds=bounds,
                **prior_kwargs
            )
        )

    nuts_kernel = NUTS(model)

    if rng_key is None:
        # Provide a default but warn the user
        print("Warning: No rng_key provided to NUTS_sampler, using default key")
        rng_key = jax.random.PRNGKey(0)
    
    key = rng_key  # Use the provided key directly

    tfms = make_transforms(bounds, dim)

    def map_batch(z_batch):
        thetas = []
        for zi in z_batch:
            theta_i, _ = unconstrained_to_constrained(zi, tfms)
            thetas.append(theta_i)
        return jnp.stack(thetas)

    # Prepare initial state in unconstrained space (z), broadcasting to chains
    init_dict = None
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
    total_warmup = 0
    current_warmup = initial_warmup
    mcmc = None
    
    while current_warmup <= max_warmup:
        print(f"Running warmup round: {current_warmup} additional steps (total warmup: {total_warmup + current_warmup})")
        
        if mcmc is None:
            # First round: start from scratch
            mcmc = MCMC(nuts_kernel, num_warmup=current_warmup, num_samples=N_samples, num_chains=N_chains, progress_bar=False)
            mcmc.run(key, init_params=init_dict)
        else:
            # Continue from the last state!
            # Get the final state from previous run
            last_state = mcmc.last_state
            # Create a new MCMC with additional warmup
            mcmc_continue = MCMC(nuts_kernel, num_warmup=current_warmup, num_samples=N_samples, num_chains=N_chains, progress_bar=False)
            # Continue from the last state instead of starting fresh
            mcmc_continue.run(key, init_params=last_state.z, extra_fields=('potential_energy', 'adapt_state'))
            mcmc = mcmc_continue  # Update our reference
        
        total_warmup += current_warmup
        
        # Check convergence
        samples = mcmc.get_samples(group_by_chain=True)
        r_hat = {k: gelman_rubin(v) for k, v in samples.items()}
        max_r_hat = max([float(jnp.max(v)) for v in r_hat.values()])
        print(f"Max R-hat: {max_r_hat:.4f} (after {total_warmup} total warmup steps)")
        
        if max_r_hat < target_r_hat:
            print(f"✅ Convergence achieved with {total_warmup} total warmup steps!")
            if single_return: 
                last_sample = samples['parameters'][0, -1]
                return unconstrained_to_constrained(last_sample, tfms)[0]
            z = samples["parameters"]  # shape: (chains, samples, D) in unconstrained space
            z_flat = z.reshape(-1, z.shape[-1])
            theta_samples = map_batch(z_flat)  # shape: (chains*samples, D)
            return mcmc, theta_samples
        
        # Increase warmup for next round
        current_warmup = int(current_warmup * 1.5)
    
    print(f"⚠️  Did not achieve convergence within {max_warmup} maximum additional warmup steps")
    print(f"   Total warmup used: {total_warmup} steps")

    if single_return:
        last_sample = samples['parameters'][0, -1]
        return unconstrained_to_constrained(last_sample, tfms)[0]
    z = samples["parameters"]  # shape: (chains, samples, D) in unconstrained space
    z_flat = z.reshape(-1, z.shape[-1])
    theta_samples = map_batch(z_flat)  # shape: (chains*samples, D)
    return mcmc, theta_samples

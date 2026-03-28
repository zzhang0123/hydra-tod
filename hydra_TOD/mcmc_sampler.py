from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from numpy.linalg import slogdet
from typing import Callable, Optional, Union

# from scipy.differentiate import hessian
import emcee
import logging, warnings


def hessian(
    func: Callable[[NDArray[np.floating]], float],
    params: NDArray[np.floating],
    epsilon: float = 1e-5,
) -> NDArray[np.floating]:
    """
    Compute the Hessian matrix of a scalar function via finite differences.

    Uses second-order forward differences to approximate the matrix of
    second partial derivatives.

    Parameters
    ----------
    func : callable
        Scalar-valued function ``func(params) -> float``.
    params : NDArray[np.floating]
        1-D parameter vector at which the Hessian is evaluated.
    epsilon : float, optional
        Step size for the finite-difference stencil. Default is ``1e-5``.

    Returns
    -------
    hessian_matrix : NDArray[np.floating]
        Symmetric Hessian matrix of shape ``(n, n)`` where
        ``n = len(params)``.

    Notes
    -----
    The stencil is

    .. math::

        H_{ij} = \\frac{f(x + e_i h + e_j h) - f(x + e_i h)
                  - f(x + e_j h) + f(x)}{h^2}

    which requires ``O(n^2)`` function evaluations.  Only the upper
    triangle is computed; the lower triangle is filled by symmetry.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    params = np.asarray(params)
    n = params.size
    hessian_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):  # Compute only for upper triangle
            params_ij = params.copy()
            params_i = params.copy()
            params_j = params.copy()

            params_ij[i] += epsilon
            params_ij[j] += epsilon

            params_i[i] += epsilon
            params_j[j] += epsilon

            f_ij = func(params_ij)
            f_i = func(params_i)
            f_j = func(params_j)
            f_0 = func(params)

            hessian_matrix[i, j] = (f_ij - f_i - f_j + f_0) / (epsilon**2)
            if i != j:
                hessian_matrix[j, i] = hessian_matrix[i, j]  # Use symmetry

    return hessian_matrix


def generate_jeffreys_prior_func(
    log_like_func: Callable[[NDArray[np.floating]], float],
    Hess: Callable[
        [Callable[[NDArray[np.floating]], float], NDArray[np.floating]],
        NDArray[np.floating],
    ] = hessian,
) -> Callable[[NDArray[np.floating]], float]:
    """
    Create a Jeffreys prior log-probability function from a likelihood.

    The Jeffreys prior is the square root of the determinant of the
    Fisher information matrix, approximated here as the negative
    Hessian of the log-likelihood.

    Parameters
    ----------
    log_like_func : callable
        Log-likelihood function ``log_like_func(params) -> float``.
    Hess : callable, optional
        A function ``Hess(func, params) -> NDArray`` that returns the
        Hessian matrix. Defaults to :func:`hessian`.

    Returns
    -------
    log_prior_func : callable
        Function ``log_prior_func(params) -> float`` returning the
        log-Jeffreys-prior value. Returns ``-inf`` when the Fisher
        information matrix is not positive definite.

    Notes
    -----
    The Jeffreys prior is

    .. math::

        \\pi(\\theta) \\propto \\sqrt{\\det I(\\theta)}

    where :math:`I(\\theta) = -\\nabla^2 \\log L(\\theta)` is the
    observed Fisher information. In log-space this becomes
    :math:`\\log \\pi = \\tfrac{1}{2} \\log \\det I`.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """

    def log_prior_func(params: NDArray[np.floating]) -> float:
        hess = -Hess(log_like_func, params)
        sign, val = slogdet(hess)
        if sign <= 0 or np.isnan(val):
            return -np.inf
        return 0.5 * val

    return log_prior_func


# Define an MCMC sampler
def mcmc_sampler(
    log_like: Callable[[NDArray[np.floating]], float],
    p_guess: NDArray[np.floating],
    p_std: float = 0.3,
    nsteps: int = 1000,
    prior_func: Optional[Callable[[NDArray[np.floating]], float]] = None,
    n_samples: int = 1,
    return_sampler: bool = False,
) -> Union[NDArray[np.floating], emcee.EnsembleSampler]:
    """
    Sample from a posterior distribution using the ``emcee`` ensemble MCMC.

    Runs an affine-invariant ensemble sampler with adaptive burn-in: the
    chain is extended in rounds until the estimated autocorrelation time
    is short enough relative to the chain length, or until the maximum
    number of rounds is reached.

    Parameters
    ----------
    log_like : callable
        Log-likelihood function ``log_like(params) -> float``.
    p_guess : NDArray[np.floating]
        1-D initial parameter guess of shape ``(ndim,)``.
    p_std : float, optional
        Standard deviation of the Gaussian ball used to initialise
        walkers around *p_guess*. Default is ``0.3``.
    nsteps : int, optional
        Number of MCMC steps per round. Default is ``1000``.
    prior_func : callable or None, optional
        Log-prior function ``prior_func(params) -> float``. If ``None``
        a flat (improper) prior is used.
    n_samples : int, optional
        Number of posterior samples to return. Behaviour depends on the
        value:

        * ``n_samples == 0`` -- return the MAP estimate (mode of the
          thinned chain).
        * ``n_samples == 1`` -- return the last sample from the thinned
          chain (default).
        * ``n_samples > 1`` -- return a random subset of *n_samples*
          from the thinned chain.
    return_sampler : bool, optional
        If ``True``, return the ``emcee.EnsembleSampler`` object
        directly (useful for diagnostics). Default is ``False``.

    Returns
    -------
    samples : NDArray[np.floating] or emcee.EnsembleSampler
        Posterior sample(s) of shape ``(ndim,)`` when
        ``n_samples <= 1``, ``(n_samples, ndim)`` when
        ``n_samples > 1``, or the raw sampler when *return_sampler*
        is ``True``.

    Notes
    -----
    The sampler uses ``2 * ndim`` walkers and runs up to 10 rounds of
    *nsteps* each, checking convergence via ``emcee``'s integrated
    autocorrelation time estimator after each round. The burn-in is set
    to ``2.5 * max(tau)`` and thinning to ``0.5 * min(tau)``.

    When used as a within-Gibbs step the convergence requirement is
    relaxed: a single representative sample from the conditional
    posterior is sufficient.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    ndim = len(p_guess)

    if prior_func is None:
        prior_function = lambda x: 0
    else:
        prior_function = prior_func

    def log_prob(params: NDArray[np.floating]) -> float:
        return log_like(params) + prior_function(params)

    nwalkers = 2 * ndim

    # Initialize the sampler
    p0 = np.random.randn(nwalkers, ndim) * p_std + p_guess
    # Run the MCMC sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

    n_rounds = 10
    for i in range(n_rounds):
        logging.info(f"Running MCMC sampler for the {i+1}th time...")
        sampler.run_mcmc(p0, nsteps, progress=False)

        try:  # Estimate the autocorrelation time
            # Catch warnings related to autocorrelation time estimation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=emcee.autocorr.AutocorrError)
                tau = sampler.get_autocorr_time(tol=3, quiet=True)
                # tau = emcee.autocorr.integrated_time(sampler.chain, quiet=True)

            # When using MCMC as a step in a Gibbs sampler, especially if you're only interested in drawing a single sample,
            # the requirement for the chain length to be significantly longer than the autocorrelation time can be relaxed.
            # The primary goal is to ensure that the sample you draw is representative of the target distribution.
            # For this purpose, the burn-in period is typically chosen to be 2-3 times the autocorrelation time.

            burnin = int(
                2.5 * np.max(tau)
            )  # Burn-in time (usually 2-3 times the autocorrelation time)
            thin = int(
                0.5 * np.min(tau)
            )  # Thinning factor (usually 1/2 of the autocorrelation time)
            thin = max(thin, 1)  # Ensure thinning is at least 1
            logging.info(f"Estimated burn-in: {burnin}")
            logging.info(f"Estimated thinning: {thin}")
            # if burnin > nsteps - n_samples, then continue with the last sample to run more steps
            if burnin < nsteps - n_samples:
                logging.info("Enough steps for burnin..")
                if return_sampler:
                    return sampler
                break
            else:
                logging.warning(
                    "Burn-in is greater than nsteps - n_samples, continuing with last sample."
                )
                p0 = sampler.chain[:, -1, :]
        except Exception as e:
            logging.error(f"Error estimating autocorrelation time: {e}")
            p0 = sampler.chain[:, -1, :]
            if i == n_rounds - 1:
                logging.info(
                    "Reached maximum number of iterations, will return last sample."
                )
                if return_sampler:
                    return sampler
                burnin = nsteps // 3
                thin = 1
                logging.info("Using default burn-in and thinning values.")

    # Get the chain after burn-in
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

    if n_samples == 0:
        # # Return mean of the samples
        # return np.mean(flat_samples, axis=0)
        log_probs = np.array([log_prob(p) for p in flat_samples])
        return flat_samples[np.argmax(log_probs)]
    elif n_samples == 1:
        # Randomly select one sample
        # idx = np.random.randint(len(flat_samples))
        # return flat_samples[idx]

        # Select the last sample
        return flat_samples[-1]
    else:
        # Pick the last n_samples from the chain
        assert len(flat_samples) >= n_samples, "Not enough samples in the chain."
        # Randomly select n_samples
        idx = np.random.randint(len(flat_samples), size=n_samples)
        return flat_samples[idx]

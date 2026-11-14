"""
Posterior draws of the linear system-temperature parameters with gains and
noise held fixed.

:func:`sample_map_posterior` performs the draw of the Tsys Gibbs step
(:mod:`hydra_tod.tsys_sampler`) without the gain and noise steps. It serves
callers that hold the gains and the noise covariance fixed, such as a
map-maker. The inputs are gain-calibrated TODs ``d_i``, their projection
operators ``U_i`` (sky pixels and any other Tsys components), inverse noise
covariances ``N_i^-1``, an estimate ``p_hat`` and an optional Gaussian prior
precision ``S^-1``. The draws follow

.. math::

    \\mathbf{p} \\sim \\mathcal{N}\\big(\\hat{\\mathbf{p}},
        (\\mathbf{A} + \\mathbf{S}^{-1})^{-1}\\big), \\qquad
    \\mathbf{A} = \\sum_i \\mathbf{U}_i^\\top \\boldsymbol{\\Sigma}_i^{-1}
        \\mathbf{U}_i,

with :math:`\\boldsymbol{\\Sigma}_i^{-1} = \\mathbf{D}_i^{-1}
\\mathbf{N}_i^{-1} \\mathbf{D}_i^{-1}` and :math:`\\mathbf{D}_i =
\\mathrm{diag}(\\mathbf{U}_i \\hat{\\mathbf{p}} + \\boldsymbol{\\mu}_i)` for
the multiplicative model ``d = (U p + mu)(1 + n)``, or
:math:`\\boldsymbol{\\Sigma}_i^{-1} = \\mathbf{N}_i^{-1}` for the additive
model ``d = U p + mu + n``. The per-TOD terms come from
:func:`~hydra_tod.linear_sampler.params_space_oper_and_data`.

Each draw is ``p_hat + delta``, where ``delta`` is one Gaussian constrained
realisation from :func:`~hydra_tod.linear_sampler.sample_p_v2` with a zero
mean,

.. math::

    (\\mathbf{A} + \\mathbf{S}^{-1})\\,\\boldsymbol{\\delta} =
    \\sum_i \\mathbf{U}_i^\\top \\boldsymbol{\\Sigma}_i^{-1/2}
      \\boldsymbol{\\omega}_i
    + \\mathbf{S}^{-1/2} \\boldsymbol{\\omega}'.

The draws are therefore centred on ``p_hat`` itself. The GLS right-hand side
``sum_i U_i^T Sigma_i^-1 (d_i - mu_i) + S^-1 m`` would centre them on the
result of one more GLS iteration started at ``p_hat``; that point equals
``p_hat`` once the GLS has converged. The prior mean enters only through
``p_hat``, which the caller computes with it, so it is not an argument.

Under the multiplicative model ``Sigma`` depends on ``p``; it is held at
``p_hat``, as the Gibbs step holds it at its GLS estimate. Each ``N_i^-1``
may be dense (correlated noise) or its diagonal (white noise). With a
diagonal, each time sample is weighted by its own inverse variance, and
memory grows linearly with the number of time samples. The function runs
on one process and uses every TOD it is given; it does no MPI reduction.

References
----------
Zhang et al. (2026), RASTI, rzag024.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from .linear_sampler import params_space_oper_and_data, sample_p_v2

# Eigenvalues of a dense N^-1 down to -_PSD_RTOL times the largest are
# round-off of a semi-definite matrix and are clipped to zero.
_PSD_RTOL = 1e-10
# A dense N^-1 or 2D prior may differ from its transpose by this fraction
# of its largest entry (the round-off of np.linalg.inv is ~1e-12).
_SYM_RTOL = 1e-8


class _NoiseProjection(NamedTuple):
    """One TOD's ``U^T Sigma^{-1/2}``. With ``scale`` None, ``matrix`` is
    that dense ``(n_params, n_time)`` product. With ``scale`` set (diagonal
    noise), ``matrix`` is the ``(n_time, n_params)`` operator ``U`` and
    ``scale`` the diagonal of ``Sigma^{-1/2}``; ``apply`` then returns
    ``U^T (scale * omega)`` without forming the product."""

    matrix: NDArray[np.floating]
    scale: NDArray[np.floating] | None

    @property
    def n_time(self) -> int:
        return self.matrix.shape[1] if self.scale is None else self.matrix.shape[0]

    def apply(self, omega: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.scale is None:
            return self.matrix @ omega
        return self.matrix.T @ (self.scale * omega)


def _check_symmetric(M: NDArray[np.floating], name: str) -> None:
    if np.abs(M - M.T).max() > _SYM_RTOL * np.abs(M).max():
        raise ValueError(f"{name} must be symmetric")


def _checked_inv_cov(N_inv, n_time: int, index: int) -> NDArray[np.floating]:
    """Validate one ``Ninv_list`` entry and return it as a float array: the
    ``(n_time,)`` diagonal of ``N^-1`` or the dense symmetric matrix."""
    N_inv = np.asarray(N_inv, dtype=float)
    if N_inv.shape not in ((n_time,), (n_time, n_time)):
        raise ValueError(
            f"Ninv_list[{index}] has shape {N_inv.shape}, expected ({n_time},) "
            f"for a diagonal or ({n_time}, {n_time})"
        )
    if not np.all(np.isfinite(N_inv)):
        raise ValueError(f"Ninv_list[{index}] must be finite")
    if N_inv.ndim == 1:
        if np.any(N_inv < 0):
            raise ValueError(
                f"Ninv_list[{index}]: a diagonal N^-1 holds inverse variances, "
                "which must be non-negative"
            )
    else:
        _check_symmetric(N_inv, f"Ninv_list[{index}]")
    return N_inv


def _inv_cov_sqrt(N_inv: NDArray[np.floating], index: int) -> NDArray[np.floating]:
    """A factor ``L`` with ``L L^T = N^-1``: ``sqrt`` of a diagonal, the
    Cholesky factor of a positive-definite matrix, or ``V Lambda^{1/2}`` of
    a semi-definite one (e.g. flagged samples zeroed out)."""
    if N_inv.ndim == 1:
        return np.sqrt(N_inv)
    try:
        return np.linalg.cholesky(N_inv)
    except np.linalg.LinAlgError:
        pass
    lam, V = np.linalg.eigh(N_inv)
    if lam[0] < -_PSD_RTOL * max(lam[-1], 0.0):
        raise ValueError(
            f"Ninv_list[{index}] is not positive semi-definite (eigenvalues "
            f"{lam[0]:.3g} to {lam[-1]:.3g})"
        )
    return V * np.sqrt(np.clip(lam, 0.0, None))


def _checked_prior(prior_cov_inv, n_params: int):
    """The prior precision in a form ``sample_p_v2`` takes: none, a finite
    non-negative diagonal (1D), or a symmetric positive-definite matrix (2D)."""
    if prior_cov_inv is None:
        return None
    prior_cov_inv = np.asarray(prior_cov_inv, dtype=float)
    if prior_cov_inv.shape == (n_params,):
        if not np.all(np.isfinite(prior_cov_inv)) or np.any(prior_cov_inv < 0):
            raise ValueError("a diagonal prior_cov_inv must be finite and non-negative")
        return prior_cov_inv
    if prior_cov_inv.shape != (n_params, n_params):
        raise ValueError(
            f"prior_cov_inv has shape {prior_cov_inv.shape}, expected "
            f"({n_params},) or ({n_params}, {n_params})"
        )
    if not np.all(np.isfinite(prior_cov_inv)):
        raise ValueError("prior_cov_inv must be finite")
    _check_symmetric(prior_cov_inv, "prior_cov_inv")
    try:
        np.linalg.cholesky(prior_cov_inv)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "a 2D prior_cov_inv must be positive definite over every "
            "parameter, because sample_p_v2 draws the prior term with its "
            "Cholesky factor. Pass a diagonal prior as 1D (zeros allowed), "
            "or give every parameter a prior."
        ) from exc
    return prior_cov_inv


def _check_proper(precision: NDArray[np.floating]) -> None:
    """Raise unless ``A + S^-1`` is positive definite to working precision.
    No ridge is added: a singular precision means the posterior is improper,
    and any draw along the unconstrained direction would be arbitrary."""
    lam = np.linalg.eigvalsh(precision)
    if not lam[0] > precision.shape[0] * np.finfo(float).eps * lam[-1]:
        raise np.linalg.LinAlgError(
            "the posterior precision A + S^-1 is singular to working "
            f"precision (eigenvalues {lam[0]:.3g} to {lam[-1]:.3g}): some "
            "parameter combination is constrained by neither the data nor "
            "the prior. Give it a prior, or remove it from the parameters."
        )


def _tod_terms(d, U, N_inv, mu, p_hat, multiplicative: bool, index: int):
    """One TOD's ``U^T Sigma^-1 U`` and noise projection at ``p_hat``."""
    sqrt_inv = _inv_cov_sqrt(N_inv, index)
    model = U @ p_hat + mu
    if multiplicative and np.any(model == 0.0):
        raise FloatingPointError(
            f"TOD {index}: the model U p_hat + mu is zero at some sample, so "
            "the multiplicative noise weights diverge"
        )
    # params_space_oper_and_data returns (A, b) without Ninv_sqrt and
    # (A, b, U^T Sigma^{-1/2}) with it; the casts state which.
    if N_inv.ndim == 1:
        A_i, _ = cast(
            "tuple[NDArray[np.floating], NDArray[np.floating]]",
            params_space_oper_and_data(
                d, U, p_hat, N_inv, mu=mu, multiplicative=multiplicative
            ),
        )
        scale = sqrt_inv / model if multiplicative else sqrt_inv
        return A_i, _NoiseProjection(U, scale)
    A_i, _, T = cast(
        "tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]",
        params_space_oper_and_data(
            d, U, p_hat, N_inv, mu=mu, Ninv_sqrt=sqrt_inv, multiplicative=multiplicative
        ),
    )
    return A_i, _NoiseProjection(T, None)


def _check_lengths(d_list, U_list, Ninv_list, mu_list, p_hat, n_samples) -> None:
    if not (len(d_list) == len(U_list) == len(Ninv_list) == len(mu_list)) or not d_list:
        raise ValueError(
            "d_list, U_list, Ninv_list and mu_list must be non-empty and have "
            "the same length"
        )
    if (
        isinstance(n_samples, bool)
        or not isinstance(n_samples, (int, np.integer))
        or n_samples < 1
    ):
        raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}")
    for i, (d, U) in enumerate(zip(d_list, U_list)):
        if U.ndim != 2 or U.shape[1] != p_hat.size or d.shape != (U.shape[0],):
            raise ValueError(
                f"TOD {i}: U has shape {U.shape} and d {d.shape}; expected "
                f"U (n_time, {p_hat.size}) and d (n_time,)"
            )


def sample_map_posterior(
    d_list: Sequence[NDArray[np.floating]],
    U_list: Sequence[NDArray[np.floating]],
    Ninv_list: Sequence[NDArray[np.floating]],
    p_hat: NDArray[np.floating],
    *,
    mu_list: Sequence[float | NDArray[np.floating]] | None = None,
    prior_cov_inv: NDArray[np.floating] | None = None,
    multiplicative: bool = True,
    n_samples: int = 1,
    rng: int | np.random.Generator | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Draw the linear Tsys parameters from their posterior, gains and noise fixed.

    Parameters
    ----------
    d_list : sequence of NDArray
        Gain-calibrated TODs, ``d_i`` of shape ``(n_time_i,)``.
    U_list : sequence of NDArray
        Projection operators, ``U_i`` of shape ``(n_time_i, n_params)``.
    Ninv_list : sequence of NDArray
        Inverse noise covariances: the dense symmetric
        ``(n_time_i, n_time_i)`` matrix, or its ``(n_time_i,)`` diagonal of
        per-sample inverse variances (zero for a flagged sample), which may
        vary from sample to sample. A dense matrix may be positive
        semi-definite. Under the multiplicative model ``N`` is the
        covariance of the fractional noise ``n``, so each entry is the
        inverse covariance of ``n``.
    p_hat : NDArray
        Centre of the draws, shape ``(n_params,)``: the GLS (or MAP)
        estimate, computed with the prior mean if there is a prior. Under
        the multiplicative model ``Sigma`` is evaluated here.
    mu_list : sequence of float or NDArray, optional
        Known additive temperatures per TOD (e.g. a noise diode). Default 0.
    prior_cov_inv : NDArray, optional
        Gaussian prior precision ``S^-1``: a non-negative diagonal
        ``(n_params,)`` or a symmetric positive-definite
        ``(n_params, n_params)``.
    multiplicative : bool, optional
        ``True`` (default) for ``d = (U p + mu)(1 + n)``; ``False`` for
        ``d = U p + mu + n``.
    n_samples : int, optional
        Number of draws. Default 1.
    rng : int, numpy.random.Generator or None, optional
        An integer seed or a ``Generator`` supplies every random number and
        leaves NumPy's global state untouched. ``None`` (default) draws
        from, and advances, the global state, as the rest of hydra-tod does.

    Returns
    -------
    samples : NDArray
        Draws, shape ``(n_samples, n_params)``.
    precision : NDArray
        Posterior precision ``A + S^-1`` at ``p_hat``, shape
        ``(n_params, n_params)``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``A + S^-1`` is singular to working precision, i.e. some
        parameter combination is constrained by neither the data nor the
        prior.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    d_list = [np.asarray(d, dtype=float) for d in d_list]
    U_list = [np.asarray(U, dtype=float) for U in U_list]
    mu_list = [0.0] * len(d_list) if mu_list is None else list(mu_list)
    _check_lengths(d_list, U_list, Ninv_list, mu_list, p_hat, n_samples)
    prior_cov_inv = _checked_prior(prior_cov_inv, p_hat.size)

    A = np.zeros((p_hat.size, p_hat.size))
    projections = []
    for i, (d, U, N_inv, mu) in enumerate(zip(d_list, U_list, Ninv_list, mu_list)):
        N_inv = _checked_inv_cov(N_inv, U.shape[0], i)
        A_i, projection = _tod_terms(d, U, N_inv, mu, p_hat, multiplicative, i)
        A += A_i
        projections.append(projection)

    if prior_cov_inv is None:
        precision = A.copy()
    elif prior_cov_inv.ndim == 1:
        precision = A + np.diag(prior_cov_inv)
    else:
        precision = A + prior_cov_inv
    _check_proper(precision)

    gen = None if rng is None else np.random.default_rng(rng)
    source = np.random if gen is None else gen
    zeros = np.zeros(p_hat.size)
    samples = np.empty((n_samples, p_hat.size))
    for k in range(n_samples):
        wn = sum(
            proj.apply(source.normal(0.0, 1.0, size=proj.n_time))
            for proj in projections
        )
        # zero-mean fluctuation, so no large (A + S^-1) p_hat is rounded
        delta = sample_p_v2(
            A,
            zeros,
            wn,
            prior_cov_inv=prior_cov_inv,
            prior_mean=None if prior_cov_inv is None else zeros,
            solver=None,
            Est_mode=False,
            rng=gen,
        )
        samples[k] = p_hat + delta
    return samples, precision

"""Serial GLS normal-equation assembly for one time-ordered data stream."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def params_space_oper_and_data(
    d: NDArray[np.floating],
    U: NDArray[np.floating],
    p: NDArray[np.floating] | None,
    N_inv: NDArray[np.floating],
    mu: float | NDArray[np.floating] = 0.0,
    Ninv_sqrt: NDArray[np.floating] | None = None,
    multiplicative: bool = True,
) -> (
    tuple[NDArray[np.floating], NDArray[np.floating]]
    | tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
):
    """
    Project the data model into parameter space for a heteroskedastic GLS.

    Given the data model ``d = (U p + mu)(1 + n)`` where ``n`` has covariance
    ``N``, this function constructs the parameter-space normal equations:

    * ``A = U^T Sigma_inv U``
    * ``b = U^T Sigma_inv (d - mu)``

    where ``Sigma_inv = diag(1/(Up+mu)) N_inv diag(1/(Up+mu))``. With
    ``multiplicative=False`` the model is additive, ``d = U p + mu + n``,
    and ``Sigma_inv = N_inv``.

    ``N_inv`` (and ``Ninv_sqrt``) may be the dense ``(M, M)`` matrix or, for
    uncorrelated noise, its ``(M,)`` diagonal. With a diagonal, each sample
    is weighted by its own inverse variance, and no ``(M, M)`` array is
    formed.

    Parameters
    ----------
    d : NDArray[np.floating]
        Observed data vector of shape ``(M,)``.
    U : NDArray[np.floating]
        Projection (design) matrix of shape ``(M, N)``.
    p : NDArray[np.floating] or None
        Current parameter estimate of shape ``(N,)``. Required for
        multiplicative noise. Pass ``None`` for fixed additive noise.
    N_inv : NDArray[np.floating]
        Inverse noise covariance, shape ``(M, M)``, or its diagonal,
        shape ``(M,)``.
    mu : float or NDArray[np.floating], optional
        Offset / mean term. Default is 0.0.
    Ninv_sqrt : NDArray[np.floating] or None, optional
        A square root ``L`` of the inverse noise covariance with
        ``L L^T = N_inv`` (e.g. the lower Cholesky factor), shape
        ``(M, M)``, or ``sqrt(N_inv)`` when ``N_inv`` is a diagonal.  If
        provided, the function also returns ``U^T Sigma_inv_sqrt`` for
        sampling.
    multiplicative : bool, optional
        ``True`` (default) for ``d = (U p + mu)(1 + n)``; ``False`` for the
        additive model ``d = U p + mu + n``. ``Sigma_inv`` then does not
        depend on ``p``, and the argument ``p`` is ignored.

    Returns
    -------
    UTSigmaU : NDArray[np.floating]
        Parameter-space operator ``U^T Sigma_inv U``, shape ``(N, N)``.
    UTSigmaD : NDArray[np.floating]
        Parameter-space data vector ``U^T Sigma_inv (d - mu)``, shape ``(N,)``.
    UTSigma_sqrt : NDArray[np.floating], optional
        Returned only when ``Ninv_sqrt`` is not None.  Product
        ``U^T Sigma_inv_sqrt``, shape ``(N, M)``.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    if multiplicative:
        if p is None:
            raise ValueError("p is required for multiplicative noise")
        D_p_inv = 1.0 / (U @ p + mu)
    else:
        D_p_inv = np.ones(U.shape[0])
    if np.ndim(N_inv) == 1:  # uncorrelated noise: Sigma_inv is diagonal
        weighted = U * (N_inv * D_p_inv**2)[:, np.newaxis]
        if Ninv_sqrt is None:
            return weighted.T @ U, weighted.T @ (d - mu)
        T = (U * (D_p_inv * Ninv_sqrt)[:, np.newaxis]).T
        return weighted.T @ U, weighted.T @ (d - mu), T
    sigma_inv = N_inv * np.outer(D_p_inv, D_p_inv)
    aux = U.T @ sigma_inv
    if Ninv_sqrt is None:
        return aux @ U, aux @ (d - mu)
    else:
        sigma_inv_sqrt = D_p_inv[:, np.newaxis] * Ninv_sqrt
        return aux @ U, aux @ (d - mu), U.T @ sigma_inv_sqrt

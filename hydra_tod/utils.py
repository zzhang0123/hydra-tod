"""General-purpose utilities for matrix operations and likelihood evaluation.

This module collects the low-level numerical building blocks used across
the rest of the package.  It is organised into four functional groups:

**Matrix operations**
    :func:`cho_compute_mat_inv`
        Invert a symmetric positive-definite matrix via Cholesky
        factorisation with automatic regularisation fallback.
    :func:`cho_compute_mat_inv_sqrt`
        Compute the inverse Cholesky factor :math:`L^{-T}` such that
        :math:`L^{-T} L^{-1} = C^{-1}`.

**Polynomial projections**
    :class:`polyn_proj`
        Callable class storing a time list; returns the Legendre
        projection matrix when called with the number of polynomial
        modes.  Used to build gain and temperature design matrices.
    :func:`Leg_poly_proj`
        Lower-level Legendre matrix builder (rescales to :math:`[-1,1]`).

**Log-likelihood evaluation**
    :func:`log_likeli`
        Gaussian log-likelihood for a symmetric Toeplitz covariance —
        fast :math:`\\mathcal{O}(N^2)` via the Levinson algorithm.
    :func:`log_likeli_general`
        Gaussian log-likelihood for a **general** (non-Toeplitz)
        symmetric positive-definite covariance — :math:`\\mathcal{O}(N^3)`
        via Cholesky.  Returns ``-inf`` for singular or NaN inputs.
    :func:`log_det_symmetric_toeplitz`
        Log-determinant of a symmetric Toeplitz matrix via Levinson
        reflection coefficients.

**DFT utilities**
    :func:`DFT_matrix`
        :math:`N \\times N` discrete Fourier transform matrix (unitary
        up to a factor of :math:`N`).
    :func:`cov_conjugate`
        Transform a covariance matrix between time and frequency domains.

**Miscellaneous**
    :func:`lag_list`
        Time lags relative to the first sample.
    :func:`overall_operator`
        Stack projection operators into a single combined matrix.
    :func:`linear_model`
        Evaluate :math:`T_{\\rm sys} = \\sum_i \\mathbf{U}_i \\mathbf{p}_i`.
    :func:`pixel_angular_size`
        HEALPix pixel angular size in degrees and arcminutes.

See Also
--------
hydra_tod.flicker_model : Produces the covariance first-rows consumed by
    :func:`log_likeli` and :func:`log_det_symmetric_toeplitz`.
hydra_tod.linear_sampler : Uses :func:`cho_compute_mat_inv` and
    :func:`cho_compute_mat_inv_sqrt`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.legendre import Legendre
import scipy.linalg
from scipy.linalg import solve_toeplitz
from scipy.linalg._solve_toeplitz import levinson

"""
Example usage of the cholesky decomposition based functions.
A = np.random.rand(100,100)
A = A@A.mT

L_inv_sqrt = cho_compute_mat_inv_sqrt(A)
N_inv = cho_compute_mat_inv(A)

np.allclose(L_inv_sqrt@L_inv_sqrt.T, N_inv)
"""


def cho_compute_mat_inv(Ncov: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the inverse of a symmetric positive-definite matrix via Cholesky.

    Falls back to a regularised Cholesky decomposition (adding ``1e-5 * I``)
    if the matrix is numerically singular.

    Parameters
    ----------
    Ncov : NDArray[np.floating]
        Symmetric positive-definite matrix of shape ``(N, N)``.

    Returns
    -------
    Ncov_inv : NDArray[np.floating]
        Inverse of ``Ncov``, shape ``(N, N)``.

    Notes
    -----
    Used throughout the Gibbs sampler to invert noise covariance matrices.
    """
    try:
        L = np.linalg.cholesky(Ncov)
        Ncov_inv = scipy.linalg.cho_solve((L, True), np.eye(Ncov.shape[0]))
        return Ncov_inv
    except np.linalg.LinAlgError:
        Ncov_reg = Ncov + 1e-5 * np.eye(Ncov.shape[0])
        L = np.linalg.cholesky(Ncov_reg)
        return scipy.linalg.cho_solve((L, True), np.eye(Ncov.shape[0]))


def cho_compute_mat_inv_sqrt(Ncov: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute ``L^{-T}`` where ``Ncov = L L^T`` (Cholesky decomposition).

    The result satisfies ``L^{-T} (L^{-T})^T = Ncov^{-1}``, so it acts as
    a square root of the inverse covariance.

    Falls back to a regularised decomposition if ``Ncov`` is numerically
    singular.

    Parameters
    ----------
    Ncov : NDArray[np.floating]
        Symmetric positive-definite matrix of shape ``(N, N)``.

    Returns
    -------
    L_inv_T : NDArray[np.floating]
        Inverse of the transposed Cholesky factor, shape ``(N, N)``.

    Notes
    -----
    Used to construct ``N^{-1/2}`` for the Gibbs sampling whitening step.
    """
    try:
        L = np.linalg.cholesky(Ncov)
        # Compute the inverse of L.T
        L_inv_T = scipy.linalg.solve_triangular(
            L, np.eye(Ncov.shape[0]), trans="T", lower=True
        )
        return L_inv_T
    except np.linalg.LinAlgError:
        Ncov_reg = Ncov + 1e-5 * np.eye(Ncov.shape[0])
        L = np.linalg.cholesky(Ncov_reg)
        return scipy.linalg.solve_triangular(
            L, np.eye(Ncov.shape[0]), trans="T", lower=True
        )


def Leg_poly_proj(ndeg: int, xs: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Build a Legendre polynomial projection matrix.

    Constructs a matrix ``U`` whose columns are Legendre polynomials of
    degree 0 through ``ndeg-1``, evaluated at the linearly rescaled ``xs``
    values (mapped to ``[-1, 1]``).

    Parameters
    ----------
    ndeg : int
        Number of Legendre polynomial degrees (columns in the output).
    xs : NDArray[np.floating]
        Sample points of shape ``(M,)``.

    Returns
    -------
    proj : NDArray[np.floating]
        Projection matrix of shape ``(M, ndeg)``.

    Notes
    -----
    The rescaling maps ``[min(xs), max(xs)]`` to ``[-1, 1]`` to ensure
    numerical stability.
    """
    # Generate the projection matrix U such that the columns are the Legendre polynomials evaluated at the rescaled xs.

    x_min = np.min(xs)
    x_max = np.max(xs)

    xs_rescaled = 2 * (xs - x_min) / (x_max - x_min) - 1
    proj = np.zeros((len(xs), ndeg))

    for i in range(ndeg):
        proj[:, i] = Legendre.basis(i)(xs_rescaled)

    return proj


class polyn_proj:
    """
    Callable Legendre polynomial projection builder.

    Stores a time list and returns a Legendre projection matrix when called
    with a specified polynomial degree.

    Parameters
    ----------
    t_list : array_like
        Time (or coordinate) values used as sample points.

    Examples
    --------
    >>> proj_builder = polyn_proj([0, 1, 2, 3, 4])
    >>> U = proj_builder(3)          # 5x3 Legendre projection
    >>> U_log = proj_builder(3, np.log)  # evaluated at log(t)
    """

    def __init__(self, t_list: NDArray[np.floating] | list[float]) -> None:
        self.t_list = np.array(t_list)

    def __call__(
        self,
        n_deg: int,
        func: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
    ) -> NDArray[np.floating]:
        """
        Build the Legendre projection matrix.

        Parameters
        ----------
        n_deg : int
            Number of polynomial degrees.
        func : callable or None, optional
            If provided, applied to the stored time list before building
            the projection (e.g. ``np.log``).

        Returns
        -------
        proj : NDArray[np.floating]
            Legendre projection matrix of shape ``(len(t_list), n_deg)``.
        """
        if func is None:
            xs = self.t_list
        else:
            xs = func(self.t_list)

        return Leg_poly_proj(n_deg, xs)


def DFT_matrix(n: int) -> NDArray[np.complexfloating]:
    """
    Construct the DFT matrix of size ``n``.

    Parameters
    ----------
    n : int
        Size of the square DFT matrix.

    Returns
    -------
    F : NDArray[np.complexfloating]
        DFT matrix of shape ``(n, n)``, using the default NumPy FFT
        normalisation.
    """
    # using the default norm
    return np.fft.fft(np.eye(n))


def cov_conjugate(
    cov: NDArray[np.floating], time_to_freq: bool = True
) -> NDArray[np.complexfloating]:
    """
    Transform a covariance matrix between time and frequency domains.

    Parameters
    ----------
    cov : NDArray[np.floating]
        Covariance matrix of shape ``(n, n)``.
    time_to_freq : bool, optional
        If ``True`` (default), transform from time to frequency domain.
        If ``False``, transform from frequency to time domain.

    Returns
    -------
    cov_transformed : NDArray[np.complexfloating]
        Transformed covariance matrix of shape ``(n, n)``.
    """
    n = cov.shape[0]
    DFT_mat = DFT_matrix(n)
    if time_to_freq:
        return DFT_mat @ cov @ DFT_mat.conj().T
    else:  # freq covariance to time covariance
        return DFT_mat.conj().T @ cov @ DFT_mat / n**2


def lag_list(time_list: NDArray[np.floating] | list[float]) -> NDArray[np.floating]:
    """
    Compute time lags relative to the first element.

    Parameters
    ----------
    time_list : array_like
        Array of time stamps.

    Returns
    -------
    lags : NDArray[np.floating]
        Array of lags ``time_list - time_list[0]``.
    """
    time_list = np.array(time_list)
    return time_list - time_list[0]


def pixel_angular_size(nside: int) -> tuple[float, float]:
    """
    Compute the approximate angular size of a HEALPix pixel.

    Parameters
    ----------
    nside : int
        HEALPix ``NSIDE`` parameter.

    Returns
    -------
    theta_deg : float
        Approximate pixel width in degrees.
    theta_arcmin : float
        Approximate pixel width in arcminutes.
    """
    import healpy as hp

    npix = hp.nside2npix(nside)  # Total number of pixels
    omega_pix = 4 * np.pi / npix  # Pixel area in steradians
    theta_pix_deg = np.sqrt(omega_pix) * (
        180 / np.pi
    )  # Approximate pixel width in degrees
    theta_pix_arcmin = theta_pix_deg * 60  # Convert to arcminutes
    return theta_pix_deg, theta_pix_arcmin


def overall_operator(
    operator_list: list[NDArray[np.floating]],
) -> NDArray[np.floating]:
    """
    Horizontally stack a list of projection operators into one matrix.

    Each operator in the list must have the same number of rows (time
    samples).  1-D arrays are promoted to column vectors.

    Parameters
    ----------
    operator_list : list of NDArray[np.floating]
        List of projection matrices, e.g.
        ``[beam_proj, rec_proj, ndiode_proj]``.

    Returns
    -------
    U : NDArray[np.floating]
        Combined operator obtained by horizontal concatenation.

    Raises
    ------
    AssertionError
        If the operators do not share the same number of rows.
    """
    aux_list = []
    for proj in operator_list:
        assert (
            proj.shape[0] == operator_list[0].shape[0]
        ), "All projection matrices must have the same length.."
        if len(proj.shape) == 1:
            proj = proj.reshape(-1, 1)
        aux_list.append(proj)
    return np.hstack(aux_list)


def linear_model(
    operator_list: list[NDArray[np.floating]],
    params_vec_list: list[NDArray[np.floating] | float],
) -> NDArray[np.floating]:
    """
    Evaluate a linear model ``T_sys = sum_i U_i p_i``.

    Parameters
    ----------
    operator_list : list of NDArray[np.floating]
        List of projection matrices, e.g.
        ``[beam_proj, rec_proj, ndiode_proj]``.
    params_vec_list : list of NDArray[np.floating] or float
        List of parameter vectors (or scalars), e.g.
        ``[true_Tsky, rec_params, T_ndiode]``.

    Returns
    -------
    Tsys : NDArray[np.floating]
        System temperature vector of shape ``(M,)``.

    Raises
    ------
    AssertionError
        If ``operator_list`` and ``params_vec_list`` have different lengths.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    assert len(operator_list) == len(
        params_vec_list
    ), "Operator list and params list must have the same length.."
    n_components = len(operator_list)
    n_data = operator_list[0].shape[0]
    Tsys = np.zeros(n_data)

    for i in range(n_components):
        # if params_vec_list[i] is a scalar:
        if len(operator_list[i].shape) == 1:
            Tsys += operator_list[i] * params_vec_list[i]
        else:
            Tsys += operator_list[i] @ params_vec_list[i]

    return Tsys


def log_det_symmetric_toeplitz(r: NDArray[np.floating]) -> float:
    """
    Compute the log-determinant of a symmetric Toeplitz matrix.

    Uses the Levinson-Durbin recursion to obtain the reflection
    coefficients, from which the log-determinant is computed as:

    .. math::

        \\log\\det T = n \\log r_0 + \\sum_{k=1}^{n-1} (n-k) \\log(1-k_k^2)

    Parameters
    ----------
    r : NDArray[np.floating]
        First row of the symmetric Toeplitz matrix, shape ``(n,)``.

    Returns
    -------
    logdet : float
        Log-determinant value, or ``-inf`` if computation fails.

    Notes
    -----
    Reflection coefficients are clipped to ``(-0.999999, 0.999999)`` for
    numerical stability.
    """
    r = np.asarray(r)
    n = len(r)
    a0 = r[0]

    b = np.zeros(n, dtype=r.dtype)
    b[: n - 1] = -r[1:]

    a = np.concatenate((r[::-1], r[1:]))
    x, reflection_coeff = levinson(a, b)

    k = reflection_coeff[1:n]
    k = np.clip(k, -0.999999, 0.999999)

    factors = np.arange(n - 1, 0, -1)
    terms = np.log(1 - k**2)
    result = n * np.log(a0) + np.dot(factors, terms)
    if np.isnan(result):
        return -np.inf
    else:
        return result


def log_likeli(corr_list: NDArray[np.floating], data: NDArray[np.floating]) -> float:
    """
    Compute the Gaussian log-likelihood for a Toeplitz covariance model.

    Evaluates:

    .. math::

        \\log p(d | C) = -\\tfrac{1}{2} \\bigl(
            d^T C^{-1} d + \\log\\det C
        \\bigr)

    where ``C`` is the symmetric Toeplitz matrix defined by ``corr_list``.

    Parameters
    ----------
    corr_list : NDArray[np.floating]
        First row of the Toeplitz covariance matrix, shape ``(n,)``.
    data : NDArray[np.floating]
        Data vector of shape ``(n,)``.

    Returns
    -------
    logL : float
        Log-likelihood value, or ``-inf`` if computation fails (e.g. due
        to NaN/Inf in inputs or a singular Toeplitz matrix).

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    if np.any(np.isnan(corr_list)) or np.any(np.isinf(corr_list)):
        return -np.inf
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return -np.inf

    try:
        result = np.dot(
            data, solve_toeplitz(corr_list, data)
        ) + log_det_symmetric_toeplitz(corr_list)
        return -0.5 * result
    except Exception:
        return -np.inf


def log_likeli_general(
    cov_matrix: NDArray[np.floating], data: NDArray[np.floating]
) -> float:
    r"""
    Compute the Gaussian log-likelihood for a general covariance matrix.

    Evaluates:

    .. math::

        \log p(d \mid C) = -\tfrac{1}{2} \bigl(
            d^\top C^{-1} d + \log\det C
        \bigr)

    using Cholesky decomposition for numerical stability.  This is the
    non-Toeplitz counterpart of :func:`log_likeli`, intended for
    covariance matrices arising from non-consecutive time stamps.

    Parameters
    ----------
    cov_matrix : NDArray[np.floating]
        Symmetric positive-definite covariance matrix, shape ``(n, n)``.
    data : NDArray[np.floating]
        Data vector of shape ``(n,)``.

    Returns
    -------
    logL : float
        Log-likelihood value, or ``-inf`` if the Cholesky decomposition
        fails (e.g. non-positive-definite matrix).

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
        return -np.inf
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return -np.inf

    try:
        L, lower = scipy.linalg.cho_factor(cov_matrix)
        alpha = scipy.linalg.cho_solve((L, lower), data)
        quad_form = np.dot(data, alpha)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        return -0.5 * (quad_form + log_det)
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
        return -np.inf

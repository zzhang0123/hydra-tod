"""Analytic 1/f flicker-noise covariance model.

This module implements the time-domain flicker-noise correlation function
derived via the Wiener–Khinchin theorem from the analytic power-spectral
density (PSD)

.. math::

    P(f) = \\begin{cases} 0, & |f| < f_c, \\\\
                          (f_0 / |f|)^\\alpha, & |f| \\geq f_c,
           \\end{cases}

where :math:`f_0` is the knee frequency, :math:`f_c` is the low-frequency
cutoff (typically the inverse TOD duration), and :math:`\\alpha` is the
spectral index.  Unlike the conventional DFT-diagonal model, this
formulation avoids spurious periodic correlations in the time domain.

The correlation function :math:`\\xi(\\tau)` is evaluated analytically
using the upper incomplete gamma function.  For large data sizes this is
accelerated by pre-trained polynomial emulators
(:class:`FlickerCorrEmulator`, :class:`LogDetEmulator`).

Key functions and classes
--------------------------
:func:`flicker_corr`
    Analytic correlation at a single lag :math:`\\tau`.
:func:`flicker_corr_vec`
    Vectorised correlation for an array of non-zero lags.
:func:`flicker_cov_vec`
    First row of the symmetric Toeplitz covariance matrix (includes
    zero-lag white-noise term).
:func:`flicker_cov`
    Full or partial Toeplitz covariance matrix for evenly-spaced times.
:func:`flicker_cov_general`
    Full :math:`N \\times N` covariance matrix for **arbitrary**
    (non-consecutive) time stamps — use this when samples have been
    flagged or removed.
:func:`sim_noise`
    Draw correlated :math:`1/f` noise realisations via Cholesky sampling.
:class:`FlickerCorrEmulator`
    Polynomial emulator for the correlation function (fast evaluation).
:class:`LogDetEmulator`
    Polynomial emulator for the log-determinant of the Toeplitz matrix.

Covariance path selection
--------------------------
For **consecutive** (evenly-spaced) times, prefer :func:`flicker_cov`
which exploits the Toeplitz structure.  For **non-consecutive** times
(flagged data), use :func:`flicker_cov_general` which builds the full
matrix — at :math:`\\mathcal{O}(N^2)` extra memory but correct for any
time ordering.

See Also
--------
hydra_tod.utils : :func:`~hydra_tod.utils.log_likeli` (Toeplitz
    log-likelihood) and :func:`~hydra_tod.utils.log_likeli_general`
    (general covariance log-likelihood).
hydra_tod.noise_sampler_fixed_fc : Noise-parameter Gibbs step that
    consumes these covariances.
"""

from __future__ import annotations

from . import mpiutil
import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp
from mpmath import gammainc
from scipy.linalg import toeplitz
from .utils import lag_list
import cmath
from scipy.integrate import quad, IntegrationWarning
import warnings
from typing import Callable

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kw):
        return x


def my_gamma_inc(
    z: complex | float,
    R_vals: float | NDArray[np.floating],
    epsabs: float = 1e-6,
    epsrel: float = 1e-6,
    para_run: bool = True,
) -> complex | NDArray[np.complexfloating]:
    r"""Compute the upper incomplete gamma function via line-integral representation.

    Evaluates :math:`\Gamma(z, iR)` by numerically integrating along the
    contour :math:`s = iR + t` for :math:`t \in [0, \infty)`:

    .. math::

        \Gamma(z, iR) = \int_0^\infty e^{-(iR + t)} (iR + t)^{z-1} \, dt

    Parameters
    ----------
    z : complex or float
        Complex (or real) order of the incomplete gamma function.
    R_vals : float or array_like
        Positive real shift(s) defining the imaginary starting point of the
        contour. Must be strictly positive.
    epsabs : float, optional
        Absolute error tolerance for ``scipy.integrate.quad``.
        Default is ``1e-6``.
    epsrel : float, optional
        Relative error tolerance for ``scipy.integrate.quad``.
        Default is ``1e-6``.
    para_run : bool, optional
        If ``True`` (default), evaluate entries in parallel using
        ``mpiutil.local_parallel_func``. Otherwise, use ``np.vectorize``.

    Returns
    -------
    complex or NDArray[np.complexfloating]
        Result with the same shape as *R_vals*. Scalar input yields a scalar
        output.

    Notes
    -----
    This is a numerically robust alternative to ``mpmath.gammainc`` for
    moderate-precision requirements where vectorised evaluation is needed.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    R_vals = np.atleast_1d(R_vals)

    # function for single input R
    def _integral_single(R_val):
        if R_val <= 0:
            raise ValueError("R must be positive")

        def integrand(t):
            s = 1j * R_val + t
            val = cmath.exp(-s) * (s ** (z - 1))
            return val

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=IntegrationWarning)
            result, _ = quad(
                integrand, 0, np.inf, epsabs=epsabs, epsrel=epsrel, complex_func=True
            )
        return result

    # Or Parallelize the function:
    if para_run:
        results = np.array(mpiutil.local_parallel_func(_integral_single, R_vals))
        # Return scalar if input was scalar
        return results[0] if results.size == 1 else results

    # Vectorize the function
    vfunc = np.vectorize(_integral_single, otypes=[np.complex128])
    return vfunc(R_vals)


def aux_int_v1(mu: float, u: float) -> float:
    r"""Evaluate the auxiliary integral for the flicker correlation at a single lag.

    Computes:

    .. math::

        \operatorname{Re}\!\bigl[e^{-i\pi\mu/2}\,\Gamma(\mu, iu)\bigr]
        = \operatorname{Re}(\Gamma(\mu, iu))\cos\!\bigl(\tfrac{\pi}{2}\mu\bigr)
          + \operatorname{Im}(\Gamma(\mu, iu))\sin\!\bigl(\tfrac{\pi}{2}\mu\bigr)

    Uses ``mpmath.gammainc`` for arbitrary-precision evaluation.

    Parameters
    ----------
    mu : float
        Order parameter, typically :math:`\mu = 1 - \alpha`.
    u : float
        Dimensionless lag argument :math:`u = \omega_c \tau`.

    Returns
    -------
    float
        Real-valued result of the auxiliary integral.
    """
    try:
        aux = gammainc(mu, 1j * u)
        ang = np.pi / 2 * mu
        return float(aux.real) * np.cos(ang) + float(aux.imag) * np.sin(ang)
    except ValueError as e:
        print(f"Error in aux_int with mu={mu}, u={u}: {e}")
        return np.inf  # or some other default value


def aux_int_v2(mu: float, u_list: NDArray[np.floating]) -> NDArray[np.float64]:
    r"""Vectorised auxiliary integral for the flicker correlation function.

    Computes the same quantity as :func:`aux_int_v1` but for an array of lag
    arguments, using :func:`my_gamma_inc` for parallel evaluation.

    Parameters
    ----------
    mu : float
        Order parameter, typically :math:`\mu = 1 - \alpha`.
    u_list : NDArray[np.floating]
        Array of dimensionless lag arguments :math:`u = \omega_c \tau`.

    Returns
    -------
    NDArray[np.float64]
        Real-valued results with the same shape as *u_list*.
    """
    # Exp[- ( Pi / 2 ) I Mu]  Gamma[Mu, I  u]
    aux = my_gamma_inc(mu, u_list)  # This is actually the same as gammainc(mu, 1j * u)
    cos_ang, sin_ang = np.cos(np.pi / 2 * mu), np.sin(np.pi / 2 * mu)
    result = aux.real * cos_ang + aux.imag * sin_ang
    return result.astype(np.float64)


def aux_calculation(u: float, mu: float) -> float:
    r"""Compute the auxiliary integral (single-lag, argument-order swapped).

    Identical to :func:`aux_int_v1` but with swapped argument order for
    compatibility with legacy call sites.

    Parameters
    ----------
    u : float
        Dimensionless lag argument :math:`u = \omega_c \tau`.
    mu : float
        Order parameter, typically :math:`\mu = 1 - \alpha`.

    Returns
    -------
    float
        Real-valued result of the auxiliary integral.
    """
    try:
        aux = gammainc(mu, 1j * u)
        ang = np.pi / 2 * mu
        return float(aux.real) * np.cos(ang) + float(aux.imag) * np.sin(ang)
    except ValueError as e:
        print(f"Error in aux calculation with mu={mu}, u={u}: {e}")
        return np.inf  # or some other default value


def flicker_corr(
    tau: float,
    f0: float,
    fc: float,
    alpha: float,
    var_w: float = 0.0,
) -> float:
    r"""Compute the flicker-noise correlation at a single time lag.

    Evaluates the analytic covariance function of :math:`1/f^\alpha` noise with
    a low-frequency cutoff:

    .. math::

        C(\tau) = \frac{(\omega_0 \tau)^\alpha}{\pi \tau}
                  \operatorname{Re}\!\bigl[e^{-i\pi\mu/2}\,
                  \Gamma(\mu, i\omega_c\tau)\bigr],
        \quad \mu = 1 - \alpha

    At zero lag (:math:`\tau = 0`):

    .. math::

        C(0) = \frac{\omega_c}{\pi}
               \left(\frac{\omega_0}{\omega_c}\right)^\alpha
               \frac{1}{\alpha - 1} + \sigma_w^2

    Parameters
    ----------
    tau : float
        Time lag in seconds.
    f0 : float
        Knee angular frequency :math:`\omega_0 = 2\pi f_{\mathrm{knee}}`
        (rad/s).
    fc : float
        Cutoff angular frequency :math:`\omega_c = 2\pi / (N \Delta t)`
        (rad/s).
    alpha : float
        Spectral index of the :math:`1/f^\alpha` noise (:math:`\alpha > 1`).
    var_w : float, optional
        White-noise variance added at zero lag. Default is ``0.0``.

    Returns
    -------
    float
        Correlation value :math:`C(\tau)`.

    Notes
    -----
    The angular frequency convention (:math:`\omega = 2\pi f`) is used
    throughout, which differs from the FFT frequency convention by a factor
    of :math:`2\pi`.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    if tau == 0:
        return fc / np.pi * (f0 / fc) ** alpha / (alpha - 1) + var_w
    tau = np.abs(tau)
    theta_c = fc * tau
    theta_0 = f0 * tau
    norm = 1 / (np.pi * tau)
    mu = 1 - alpha
    result = theta_0**alpha * aux_int_v1(mu, theta_c)
    return result * norm


def flicker_corr_vec(
    taus: NDArray[np.floating],
    f0: float,
    fc: float,
    alpha: float,
) -> NDArray[np.float64]:
    r"""Vectorised flicker-noise correlation for an array of non-zero lags.

    Computes the same quantity as :func:`flicker_corr` but for multiple lags
    simultaneously, using the vectorised auxiliary integral :func:`aux_int_v2`.

    Parameters
    ----------
    taus : NDArray[np.floating]
        Array of **strictly positive** time lags (seconds).
    f0 : float
        Knee angular frequency :math:`\omega_0` (rad/s).
    fc : float
        Cutoff angular frequency :math:`\omega_c` (rad/s).
    alpha : float
        Spectral index (:math:`\alpha > 1`).

    Returns
    -------
    NDArray[np.float64]
        Correlation values for each lag in *taus*.

    Notes
    -----
    The angular frequency convention (:math:`\omega = 2\pi f`) is used.
    All elements of *taus* must be non-zero; use :func:`flicker_cov_vec` to
    include the zero-lag term.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    theta_c = fc * taus
    norm = (f0 * taus) ** alpha / (np.pi * taus)
    mu = 1 - alpha
    # result = np.array([aux_calculation(theta_c[i], mu)
    #                                               for i in range(len(taus))])
    result = aux_int_v2(mu, theta_c)
    return result * norm


def flicker_cov_vec(
    tau_list: NDArray[np.floating],
    f0: float,
    fc: float,
    alpha: float,
    white_n_variance: float = 2.5e-6,
) -> NDArray[np.float64]:
    r"""Compute the first row of the flicker covariance matrix (vectorised).

    Evaluates the correlation function at all lags in *tau_list*, handling
    the zero-lag term separately (analytic expression plus white-noise
    variance).

    Parameters
    ----------
    tau_list : NDArray[np.floating]
        Array of time lags starting with ``0``. Typically produced by
        :func:`hydra_tod.utils.lag_list`.
    f0 : float
        Knee angular frequency :math:`\omega_0` (rad/s).
    fc : float
        Cutoff angular frequency :math:`\omega_c` (rad/s).
    alpha : float
        Spectral index (:math:`\alpha > 1`).
    white_n_variance : float, optional
        White-noise variance :math:`\sigma_w^2` added at zero lag.
        Default is ``2.5e-6``.

    Returns
    -------
    NDArray[np.float64]
        Correlation vector (first row of the Toeplitz covariance matrix).

    Raises
    ------
    AssertionError
        If ``tau_list[0] != 0``.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    assert tau_list[0] == 0, "tau_list[0] must be 0"
    result = np.zeros_like(tau_list)
    result[0] = fc / np.pi * (f0 / fc) ** alpha / (alpha - 1) + white_n_variance
    result[1:] = flicker_corr_vec(tau_list[1:], f0, fc, alpha)
    return result.astype(np.float64)


# Define the covariance matrix function
def flicker_cov(
    time_list: NDArray[np.floating],
    f0: float,
    fc: float,
    alpha: float,
    white_n_variance: float = 5e-6,
    only_row_0: bool = False,
) -> NDArray[np.float64]:
    r"""Construct the Toeplitz covariance matrix of flicker noise.

    Computes the correlation at each unique lag from *time_list* and
    assembles the full symmetric Toeplitz matrix, or optionally returns
    only the first row.

    Parameters
    ----------
    time_list : NDArray[np.floating]
        Observation time stamps (seconds). The lag list is derived via
        :func:`hydra_tod.utils.lag_list`.
    f0 : float
        Knee angular frequency :math:`\omega_0` (rad/s).
    fc : float
        Cutoff angular frequency :math:`\omega_c` (rad/s).
    alpha : float
        Spectral index (:math:`\alpha > 1`).
    white_n_variance : float, optional
        White-noise variance :math:`\sigma_w^2`. Default is ``5e-6``.
    only_row_0 : bool, optional
        If ``True``, return only the first row of the Toeplitz matrix.
        Default is ``False``.

    Returns
    -------
    NDArray[np.float64]
        If *only_row_0* is ``False``, the full :math:`N \times N` Toeplitz
        covariance matrix. Otherwise, a 1-D array of length :math:`N`.

    Notes
    -----
    The matrix is constructed via ``scipy.linalg.toeplitz`` from the
    correlation vector. For large :math:`N`, consider using
    :class:`FlickerCorrEmulator` for faster evaluation.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    lags = lag_list(time_list)
    corr_list = [flicker_corr(t, f0, fc, alpha, var_w=white_n_variance) for t in lags]
    corr_list = np.array(corr_list, dtype=np.float64)
    # corr_list = flicker_cov_vec(lags, f0, fc, alpha, white_n_variance=white_n_variance)
    if only_row_0:
        return corr_list
    return toeplitz(corr_list)


# This is another flicker noise PSD model with non-vanishing DC mode and its adjacent modes.
def flicker_corr_full(
    tau: float,
    f0: float,
    fc: float,
    alpha: float,
    var_w: float = 0.0,
) -> float:
    r"""Compute the flicker-noise correlation including the DC mode contribution.

    Extends :func:`flicker_corr` with an additional sinusoidal term that
    accounts for non-vanishing power at the DC mode and its adjacent
    frequencies:

    .. math::

        C_{\mathrm{full}}(\tau) = \frac{1}{\pi\tau}\Bigl[
            (\omega_0\tau)^\alpha\,\operatorname{Re}[e^{-i\pi\mu/2}\Gamma(\mu,i\omega_c\tau)]
            + \bigl(\omega_0/\omega_c\bigr)^\alpha\sin(\omega_c\tau)
        \Bigr]

    Parameters
    ----------
    tau : float
        Time lag in seconds.
    f0 : float
        Knee angular frequency :math:`\omega_0` (rad/s).
    fc : float
        Cutoff angular frequency :math:`\omega_c` (rad/s).
    alpha : float
        Spectral index (:math:`\alpha > 1`).
    var_w : float, optional
        White-noise variance added at zero lag. Default is ``0.0``.

    Returns
    -------
    float
        Correlation value including the DC contribution.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    if tau == 0:
        return fc / np.pi * (f0 / fc) ** alpha * alpha / (alpha - 1) + var_w
    tau = np.abs(tau)
    theta_c = fc * tau
    theta_0 = f0 * tau
    norm = 1 / (np.pi * tau)
    mu = 1 - alpha
    result = theta_0**alpha * aux_int_v1(mu, theta_c) + (f0 / fc) ** alpha * np.sin(
        theta_c
    )
    return result * norm


def sim_noise(
    f0: float,
    fc: float,
    alpha: float,
    time_list: NDArray[np.floating],
    n_samples: int = 1,
    white_n_variance: float = 5e-6,
) -> NDArray[np.float64]:
    r"""Generate realisations of correlated flicker noise.

    Constructs the full Toeplitz covariance matrix from the analytic
    correlation function and draws samples from the corresponding
    multivariate normal distribution.

    Parameters
    ----------
    f0 : float
        Knee angular frequency :math:`\omega_0` (rad/s).
    fc : float
        Cutoff angular frequency :math:`\omega_c` (rad/s).
    alpha : float
        Spectral index (:math:`\alpha > 1`).
    time_list : NDArray[np.floating]
        Observation time stamps (seconds).
    n_samples : int, optional
        Number of independent noise realisations. Default is ``1``.
    white_n_variance : float, optional
        White-noise variance :math:`\sigma_w^2`. Default is ``5e-6``.

    Returns
    -------
    NDArray[np.float64]
        If ``n_samples == 1``, a 1-D array of length ``len(time_list)``.
        Otherwise, a 2-D array of shape ``(n_samples, len(time_list))``.

    Notes
    -----
    This function builds the dense covariance matrix and is therefore
    :math:`O(N^2)` in memory and :math:`O(N^3)` in compute due to the
    Cholesky decomposition inside ``np.random.multivariate_normal``.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    lags = lag_list(time_list)
    corr_list = [flicker_corr(t, f0, fc, alpha, var_w=white_n_variance) for t in lags]
    covmat = toeplitz(corr_list)
    if n_samples == 1:
        return np.random.multivariate_normal(np.zeros_like(time_list), covmat)
    else:
        return np.random.multivariate_normal(
            np.zeros_like(time_list), covmat, n_samples
        )


def flicker_cov_general(
    time_list: NDArray[np.floating],
    f0: float,
    fc: float,
    alpha: float,
    white_n_variance: float = 5e-6,
) -> NDArray[np.float64]:
    r"""Construct the full covariance matrix for arbitrary (non-consecutive) time stamps.

    Unlike :func:`flicker_cov`, which assumes evenly-spaced times and builds
    a Toeplitz matrix from the first-row correlation vector, this function
    computes all pairwise time differences and evaluates the correlation
    function at each unique lag.  The result is a general symmetric positive-
    definite matrix suitable for flagged / non-contiguous time samples.

    Parameters
    ----------
    time_list : NDArray[np.floating]
        Observation time stamps (seconds), not necessarily evenly spaced.
    f0 : float
        Knee angular frequency :math:`\omega_0` (rad/s).
    fc : float
        Cutoff angular frequency :math:`\omega_c` (rad/s).
    alpha : float
        Spectral index (:math:`\alpha > 1`).
    white_n_variance : float, optional
        White-noise variance :math:`\sigma_w^2` added to the diagonal.
        Default is ``5e-6``.

    Returns
    -------
    NDArray[np.float64]
        Full :math:`N \times N` covariance matrix.

    Notes
    -----
    For consecutive (evenly-spaced) time stamps, prefer :func:`flicker_cov`
    which exploits the Toeplitz structure for faster evaluation.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    time_list = np.asarray(time_list, dtype=np.float64)
    n = len(time_list)

    # All pairwise absolute time differences
    tau_matrix = np.abs(time_list[:, None] - time_list[None, :])

    # Find unique non-zero lags and evaluate correlation once
    unique_taus = np.unique(tau_matrix)
    unique_taus = unique_taus[unique_taus > 0]

    if len(unique_taus) > 0:
        corr_values = flicker_corr_vec(unique_taus, f0, fc, alpha)
        # Build a lookup: tau -> correlation value
        tau_to_corr = dict(zip(unique_taus, corr_values))
    else:
        tau_to_corr = {}

    # Zero-lag value
    c0 = fc / np.pi * (f0 / fc) ** alpha / (alpha - 1) + white_n_variance

    # Assemble the matrix
    cov = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        cov[i, i] = c0
        for j in range(i + 1, n):
            val = tau_to_corr[tau_matrix[i, j]]
            cov[i, j] = val
            cov[j, i] = val

    return cov


class FNoise_traditional:
    r"""Traditional :math:`1/f^\alpha` noise generator using FFT filtering.

    Generates time-domain noise realisations by shaping white noise in the
    Fourier domain with a :math:`1/f^\alpha` power spectral density (PSD).
    The white-noise variance is set by the radiometer equation:

    .. math::

        \sigma^2 = \frac{1}{\Delta t \, \Delta\nu}

    where :math:`\Delta t` is the integration time and :math:`\Delta\nu`
    is the frequency channel width.

    Parameters
    ----------
    alpha : float
        Spectral index of the :math:`1/f^\alpha` noise.
    dtime : float, optional
        Sampling / integration time per data point (seconds). Default is ``2``.
    dnu : float, optional
        Frequency channel width (Hz). Default is ``2e5``.
    fknee : float, optional
        Knee angular frequency :math:`\omega_{\mathrm{knee}} = 2\pi f_{\mathrm{knee}}`
        (rad/s). Default is ``0.001``.

    Attributes
    ----------
    sigma_2 : float
        White-noise variance :math:`1/(\Delta t \, \Delta\nu)`.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """

    def __init__(
        self,
        alpha: float,
        dtime: float = 2,
        dnu: float = 2e5,
        fknee: float = 0.001,
    ) -> None:
        self.dtime = dtime
        self.alpha = alpha
        self.fknee = fknee
        self.sigma_2 = 1.0 / (dtime * dnu)  # dtime is also the integration time.

    def generate(self, ntime: int) -> NDArray[np.float64]:
        r"""Generate a single realisation of :math:`1/f^\alpha` noise.

        Parameters
        ----------
        ntime : int
            Number of time samples.

        Returns
        -------
        NDArray[np.float64]
            Time-domain noise array of length *ntime*.

        Notes
        -----
        The procedure is:

        1. Compute the FFT frequency axis and convert to angular frequency.
        2. Build the PSD filter :math:`\sqrt{(\omega_{\mathrm{knee}}/|\omega|)^\alpha}`.
        3. Generate white noise, FFT, multiply by the filter, and IFFT.
        """
        # Frequency axis for the rFFT
        freqs = (
            np.fft.fftfreq(ntime, d=self.dtime) * 2 * np.pi
        )  # Note that fftfreq is not in unit of angular frequency.
        freqs[0] = np.inf  # Avoid division by zero at f=0

        # Define the power spectrum scaling as 1/f^alpha plus white noise
        psd_sqrt = (self.fknee / np.abs(freqs)) ** (
            self.alpha / 2
        )  # + 1  # Adding white noise component

        # Generate random white noise
        white_noise = np.random.normal(size=len(freqs)) * np.sqrt(self.sigma_2)
        white_noise_fft = np.fft.fft(white_noise)

        # Weight the white noise by the power spectrum
        weighted_noise_fft = white_noise_fft * psd_sqrt

        # Transform back to the time domain using irfft
        time_series = np.fft.ifft(weighted_noise_fft)

        # Normalize the time series
        return time_series.real


class FlickerCorrEmulator:
    r"""Polynomial emulator for the flicker-noise correlation function.

    Trains a :class:`MomentEmu.PolyEmu` polynomial emulator to approximate
    :func:`flicker_cov_vec` as a function of the spectral index
    :math:`\alpha`, enabling fast evaluation without repeated numerical
    integration.

    The emulator is split into two parts:

    * **Auto-correlation** (small lags, indices ``0:5``): emulates
      :math:`\ln C(\tau)` to preserve positivity.
    * **Cross-correlation** (remaining lags): emulates :math:`C(\tau)`
      directly.

    Scaling with :math:`\omega_0` is handled analytically:

    .. math::

        C(\tau;\,\omega_0,\alpha)
        = 10^{(\log_{10}\omega_0 - \log_{10}\omega_0^{\mathrm{ref}})\,\alpha}
          \; C(\tau;\,\omega_0^{\mathrm{ref}},\alpha)

    Parameters
    ----------
    logfc : float
        Log10 of the cutoff angular frequency :math:`\log_{10}\omega_c`.
    tau_list : NDArray[np.floating]
        Array of time lags (seconds), starting with zero.
    wnoise_var : float, optional
        White-noise variance added at zero lag. Default is ``2.5e-6``.
    alpha_training : NDArray[np.floating] or None, optional
        Training values of :math:`\alpha`. Default generates 3000 points
        linearly spaced in ``[1.1, 4]``.
    corr_training : NDArray[np.floating] or None, optional
        Pre-computed training correlation vectors. If ``None``, computed
        from *alpha_training* via :func:`flicker_cov_vec`.
    alpha_test : NDArray[np.floating] or None, optional
        Test values of :math:`\alpha`. Default generates 500 uniform
        random points in ``[1.1, 4]``.
    corr_test : NDArray[np.floating] or None, optional
        Pre-computed test correlation vectors.

    Attributes
    ----------
    ref_logf0 : float
        Reference :math:`\log_{10}\omega_0` used for training (``-4.0``).
    lag_emulator : PolyEmu
        Emulator for lags with index >= 5.
    auto_emulator : PolyEmu
        Emulator for lags with index < 5 (log-space).

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """

    def __init__(
        self,
        logfc: float,
        tau_list: NDArray[np.floating],
        wnoise_var: float = 2.5e-6,
        alpha_training: NDArray[np.floating] | None = None,
        corr_training: NDArray[np.floating] | None = None,
        alpha_test: NDArray[np.floating] | None = None,
        corr_test: NDArray[np.floating] | None = None,
    ) -> None:
        self.ref_logf0 = -4.0  # Reference logf0 for the emulator
        self.wnoise_var = wnoise_var
        self.logfc = logfc
        self.tau_list = tau_list

        # Generate training data
        if alpha_training is None:
            alpha_training = np.linspace(1.1, 4, 3000)
        if corr_training is None:
            corr_training = np.array(
                [
                    flicker_cov_vec(
                        tau_list,
                        10.0**self.ref_logf0,
                        10.0**logfc,
                        alpha,
                        white_n_variance=0.0,
                    )
                    for alpha in tqdm(alpha_training)
                ]
            )
        if alpha_test is None:
            alpha_test = np.random.uniform(1.1, 4, 500)
        if corr_test is None:
            corr_test = np.array(
                [
                    flicker_cov_vec(
                        tau_list,
                        10.0**self.ref_logf0,
                        10.0**logfc,
                        alpha,
                        white_n_variance=0.0,
                    )
                    for alpha in tqdm(alpha_test)
                ]
            )

        # Train the emulator
        from MomentEmu import PolyEmu

        ind = 5  # Index separating lag=0 and lag>0
        print("Training emulator for lag>0 ...")
        self.lag_emulator = PolyEmu(
            alpha_training.reshape(-1, 1),
            corr_training[:, ind:],
            X_test=alpha_test.reshape(-1, 1),
            Y_test=corr_test[:, ind:],
            RMSE_upper=1e-1,
            RMSE_lower=1e-7,
            fRMSE_tol=0.1,
            forward=True,
            backward=False,
            max_degree_forward=20,
            return_max_frac_err=True,
        )

        print("Training emulator for lag=0 ...")
        self.auto_emulator = PolyEmu(
            alpha_training.reshape(-1, 1),
            np.log(corr_training[:, :ind]).reshape(-1, ind),
            X_test=alpha_test.reshape(-1, 1),
            Y_test=np.log(corr_test[:, :ind]).reshape(-1, ind),
            RMSE_upper=1e-1,
            RMSE_lower=1e-5,
            fRMSE_tol=1e-1,
            forward=True,
            backward=False,
            max_degree_forward=20,
            return_max_frac_err=True,
        )

    def __call__(
        self,
        logf0: float,
        alpha: float,
        indices: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        r"""Emulate the flicker correlation vector for given noise parameters.

        Parameters
        ----------
        logf0 : float
            Log10 of the knee angular frequency :math:`\log_{10}\omega_0`.
        alpha : float
            Spectral index.
        indices : NDArray[np.intp] or None, optional
            If provided, return only the correlation values at these lag
            indices. Default is ``None`` (return the full vector).

        Returns
        -------
        NDArray[np.float64]
            Emulated correlation vector (or subset if *indices* is given).
        """
        correction_factor = 10.0 ** ((logf0 - self.ref_logf0) * alpha)
        corr = np.concatenate(
            [
                np.exp(self.auto_emulator.forward_emulator(np.array([alpha]))),
                self.lag_emulator.forward_emulator(np.array([alpha])),
            ]
        )
        result = corr * correction_factor
        result[0] += self.wnoise_var  # Add white noise variance to the first element
        if indices is None:
            return result
        return result[indices]

    def create_jax(self) -> Callable:
        """Create a JAX-autodifferentiable version of the emulator.

        Returns
        -------
        Callable
            A JAX-compatible function with signature
            ``(logf0, alpha, indices=None) -> jnp.ndarray``.

        Notes
        -----
        Requires JAX and ``MomentEmu.jax_momentemu``. The returned function
        is suitable for use with ``jax.jit`` and ``jax.grad``.
        """

        print("Get the JAX version of the emulators")
        from MomentEmu.jax_momentemu import create_jax_emulator

        lag_emulator_jax = create_jax_emulator(self.lag_emulator)
        auto_emulator_jax = create_jax_emulator(self.auto_emulator)
        wn_var = self.wnoise_var

        def jax_emu(logf0, alpha, indices=None):
            correction_factor = 10.0 ** ((logf0 - self.ref_logf0) * alpha)

            # Get emulator outputs (these need to be JAX-compatible)
            auto_output = jnp.exp(auto_emulator_jax(alpha))
            lag_output = lag_emulator_jax(alpha)

            corr = jnp.concatenate([auto_output, lag_output])
            result = corr * correction_factor

            # Add white noise variance to the first element
            result = result.at[0].add(wn_var)

            # Handle indices selection
            return jnp.where(indices is None, result, result[indices])

        return jax_emu


class LogDetEmulator:
    r"""Polynomial emulator for the log-determinant of the flicker covariance.

    Trains a :class:`MomentEmu.PolyEmu` emulator to approximate
    :math:`\ln\det\mathbf{C}(\log_{10}\omega_0, \alpha)`, enabling fast
    evaluation of the Gaussian log-likelihood normalisation term during
    MCMC sampling.

    Parameters
    ----------
    params_list : NDArray[np.floating]
        Training parameter array of shape ``(n_train, 2)`` with columns
        ``[logf0, alpha]``.
    log_det_list : NDArray[np.floating]
        Corresponding log-determinant values, shape ``(n_train, 1)``.
    X_test : NDArray[np.floating]
        Test parameter array, shape ``(n_test, 2)``.
    Y_test : NDArray[np.floating]
        Test log-determinant values, shape ``(n_test, 1)``.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """

    def __init__(
        self,
        params_list: NDArray[np.floating],
        log_det_list: NDArray[np.floating],
        X_test: NDArray[np.floating],
        Y_test: NDArray[np.floating],
    ) -> None:

        from MomentEmu import PolyEmu

        self.log_det_emulator = PolyEmu(
            params_list,
            log_det_list,
            X_test=X_test,
            Y_test=Y_test,
            forward=True,
            backward=False,
            max_degree_forward=20,
            RMSE_upper=1e-1,
            RMSE_lower=1e-5,
            fRMSE_tol=1e-1,
            return_max_frac_err=True,
        )

    def __call__(
        self,
        logf0: float,
        alpha: float,
    ) -> NDArray[np.float64]:
        """Evaluate the emulated log-determinant.

        Parameters
        ----------
        logf0 : float
            Log10 of the knee angular frequency.
        alpha : float
            Spectral index.

        Returns
        -------
        NDArray[np.float64]
            Emulated :math:`\\ln\\det\\mathbf{C}` value(s).
        """
        return self.log_det_emulator.forward_emulator(np.array([[logf0, alpha]]))

    def create_jax(self) -> Callable:
        """Create a JAX-autodifferentiable version of the log-det emulator.

        Returns
        -------
        Callable
            A JAX-compatible function with signature
            ``(logf0, alpha) -> jnp.ndarray``.
        """
        print("Get the JAX version of the log-det emulator")
        from MomentEmu.jax_momentemu import create_jax_emulator

        log_det_emulator_jax = create_jax_emulator(self.log_det_emulator)

        def jax_emu(logf0, alpha):
            return log_det_emulator_jax(jnp.array([logf0, alpha]))

        return jax_emu

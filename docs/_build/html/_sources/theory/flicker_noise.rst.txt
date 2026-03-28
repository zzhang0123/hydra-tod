Flicker Noise Model
===================

The correlated noise in radio telescope data is modeled as flicker (1/f) noise
with a power spectral density that follows a power law.

Power Spectral Density
----------------------

The noise power spectral density is:

.. math::

   P(\omega) = \begin{cases}
   \left(\frac{f_0}{\omega}\right)^\alpha & \omega > \omega_c \\
   0 & \omega \leq \omega_c
   \end{cases}

where:

- :math:`f_0` is the knee frequency (angular frequency convention)
- :math:`\alpha` is the spectral index (typically 1.5--3)
- :math:`\omega_c = 2\pi / (N \cdot \Delta t)` is the low-frequency cutoff

Correlation Function
--------------------

The time-domain correlation function is obtained via Fourier transform:

.. math::

   C(\tau) = \frac{1}{\pi \tau} \left(f_0 \tau\right)^\alpha \,
   \mathrm{Re}\!\left[e^{-i\pi\mu/2} \, \Gamma(\mu, i\omega_c\tau)\right]

where :math:`\mu = 1 - \alpha` and :math:`\Gamma(\mu, z)` is the upper
incomplete gamma function.

At zero lag:

.. math::

   C(0) = \frac{\omega_c}{\pi} \frac{(f_0/\omega_c)^\alpha}{\alpha - 1} + \sigma_w^2

where :math:`\sigma_w^2` is the white noise variance.

Covariance Matrix
-----------------

The noise covariance matrix is Toeplitz, constructed from the correlation
function evaluated at the time lags:

.. math::

   \mathbf{N}_{ij} = C(|t_i - t_j|)

The Toeplitz structure enables efficient :math:`O(n \log n)` matrix-vector
products and :math:`O(n^2)` Levinson-based solves.

Emulator
--------

For repeated evaluation during sampling, the correlation function can be
approximated using a polynomial emulator (``FlickerCorrEmulator``) trained
over a grid of spectral index values. The emulator provides both NumPy and
JAX-compatible evaluation for automatic differentiation in NUTS sampling.

For full details, see Zhang et al. (2026), RASTI, rzag024.

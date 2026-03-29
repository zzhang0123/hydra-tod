Flicker Noise Model
===================

This module implements the analytic :math:`1/f` noise covariance model
used throughout the pipeline (Zhang et al. 2026, §2.2).  Unlike the
conventional DFT-diagonal approach, the covariance is defined directly in
the time domain via the Wiener–Khinchin theorem, avoiding spurious
periodic correlations.

The correlation function

.. math::

   \xi(\tau) = \frac{1}{\pi\tau}\,\Theta_0^\alpha\,
   \mathrm{Re}\!\left[\Gamma(1{-}\alpha,\,i\Theta_c)\,
   e^{-i\frac{\pi}{2}(1{-}\alpha)}\right],
   \quad \Theta_0 = \tau f_0,\; \Theta_c = \tau f_c,

is evaluated analytically using the upper incomplete gamma function and
accelerated by a pre-trained polynomial emulator.

Covariance path selection
--------------------------

For **evenly-spaced** (consecutive) time samples, use
:func:`~hydra_tod.flicker_model.flicker_cov`, which returns a symmetric
Toeplitz matrix (or its first row) and enables the fast Levinson
:math:`\mathcal{O}(N^2)` log-likelihood in
:func:`~hydra_tod.utils.log_likeli`.

For **non-consecutive** time samples (e.g., after RFI flagging), use
:func:`~hydra_tod.flicker_model.flicker_cov_general`, which builds the
full :math:`N\times N` covariance matrix from pairwise time differences.
Pair it with :func:`~hydra_tod.utils.log_likeli_general` for the
Cholesky-based log-likelihood, and pass ``consecutive=False`` to the
noise samplers.

Emulator classes
----------------

:class:`~hydra_tod.flicker_model.FlickerCorrEmulator` and
:class:`~hydra_tod.flicker_model.LogDetEmulator` wrap pre-trained
polynomial emulators (from the ``MomentEmu`` package) for fast
evaluation of the correlation function and Toeplitz log-determinant
respectively.  They are loaded automatically at import time from pickle
files in the package directory and expose a JAX-compatible interface via
``.create_jax()``.

.. automodule:: hydra_tod.flicker_model
   :members:
   :undoc-members:
   :show-inheritance:

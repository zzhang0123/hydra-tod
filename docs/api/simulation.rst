Simulation
==========

This module generates synthetic MeerKAT-like time-ordered data for
testing and validation.  It replicates the pipeline from Zhang et al.
(2026, §4): azimuth raster scan → equatorial coordinate conversion →
Gaussian beam projection → flicker noise generation.

The primary class is :class:`~hydra_tod.simulation.TODSimulation`, which
runs the full simulation on construction and stores all components as
attributes.

.. list-table:: Key attributes of ``TODSimulation``
   :widths: 30 70
   :header-rows: 1

   * - Attribute
     - Description
   * - ``TOD_setting``
     - Observed time-ordered data (gain × Tsys × (1 + noise))
   * - ``sky_params``
     - True sky pixel temperatures (HEALPix)
   * - ``gains_setting``
     - True gain time series
   * - ``noise_setting``
     - True :math:`1/f` noise realisation
   * - ``time_list``
     - Observation times (seconds from scan start)
   * - ``gain_proj``
     - Legendre projection matrix for the gain model
   * - ``Tsky_operator``
     - Beam projection matrix mapping sky pixels → TOD samples
   * - ``Tloc_operator``
     - Local-temperature projection matrix (noise diode + receiver)
   * - ``logfc``, ``logf0``, ``alpha``
     - True noise parameters
   * - ``wnoise_var``
     - White-noise variance :math:`\sigma_w^2`

:class:`~hydra_tod.simulation.MultiTODSimulation` combines a
``TODSimulation`` in setting mode and one in rising mode, providing the
two overlapping scans used in the ``2×TOD`` analysis of Zhang et al.
(2026).

.. automodule:: hydra_tod.simulation
   :members:
   :undoc-members:
   :show-inheritance:

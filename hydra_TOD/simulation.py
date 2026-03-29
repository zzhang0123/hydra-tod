"""Simulated MeerKAT-like time-ordered data for testing and validation.

This module generates synthetic TOD that replicate MeerKLASS single-dish
observations (Zhang et al. 2026).  It is used throughout the test suite
and the tutorial notebooks to produce ground-truth data for the Gibbs
sampler.

The simulation pipeline is:

1. **Scan pattern** — raster scan in azimuth at fixed elevation, converted
   to equatorial (RA, Dec) via :func:`sim_MeerKAT_scan`.
2. **Beam + sky** — Gaussian beam convolved with a HEALPix sky map
   (GSM diffuse + GLEAM point sources) via :func:`generate_Tsky_proj`.
3. **Gain** — smooth Legendre polynomial gain with optional DC component.
4. **Noise** — periodic noise-diode injections plus correlated
   :math:`1/f` flicker noise from :func:`~hydra_tod.flicker_model.sim_noise`.

Main classes
------------
:class:`TODSimulation`
    Single-scan simulation.  After construction all components are
    accessible as attributes:

    * ``TOD_setting`` — observed data vector
    * ``sky_params`` — true sky pixel temperatures
    * ``gains_setting`` — true gain time series
    * ``noise_setting`` — true noise realisation
    * ``time_list`` — observation times
    * ``gain_proj`` — gain Legendre projection matrix
    * ``Tsky_operator`` — sky beam projection matrix
    * ``Tloc_operator`` — local-temperature projection matrix
    * ``logfc``, ``logf0``, ``alpha`` — true noise parameters

:class:`MultiTODSimulation`
    Two-scan (setting + rising) simulation combining two
    :class:`TODSimulation` instances.

Utility functions
-----------------
:func:`sim_MeerKAT_scan`
    Generate an Az/El raster-scan time list and pointing coordinates.
:func:`stacked_beam_map`
    Build the HEALPix boolean coverage map for a scan.
:func:`generate_Tsky_proj`
    Construct the beam projection matrix :math:`\\mathbf{U}_{\\rm sky}`.

Typical usage
-------------
.. code-block:: python

    from hydra_tod.simulation import TODSimulation

    sim = TODSimulation(
        nside=64,
        elevation=41.5,
        freq=750,
        alpha=2.0,
        logf0=-4.87,
        ptsrc_path="gleam_nside512_K_allsky_408MHz.npy",
    )

    tod   = sim.TOD_setting
    sky   = sim.sky_params
    gains = sim.gains_setting

See Also
--------
hydra_tod.flicker_model : :func:`~hydra_tod.flicker_model.sim_noise`
    used internally for noise generation.
hydra_tod.full_Gibbs_sampler : Consumes the operators produced here.
"""
from __future__ import annotations

import numpy as np
import healpy as hp
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time, TimeDelta
import astropy.units as u
from numpy.typing import NDArray
from typing import Any

from .utils import Leg_poly_proj
from .flicker_model import sim_noise, flicker_cov
from . import mpiutil
from mpi4py import MPI

_comm = MPI.COMM_WORLD


def sim_MeerKAT_scan(
    telescope_lat: float = -30.7130,
    telescope_lon: float = 21.4430,
    telescope_height: float = 1054,
    elevation: float = 41.5,
    az_s: float = -60.3,
    az_e: float = -42.3,
    start_time_utc: str = "2019-04-23 20:41:56.397",
    dt: float = 2.0,
    return_eq_coords: bool = False,
) -> tuple[NDArray[np.floating], ...] | tuple[NDArray[np.floating], SkyCoord]:
    """Generate a simulated MeerKAT telescope scan pattern.

    Simulates a raster scan in azimuth at fixed elevation for the MeerKAT
    radio telescope located in the Karoo, South Africa. The scan pattern
    consists of 13 repeated back-and-forth sweeps across the specified
    azimuth range (111 samples per sweep), producing a total of
    ``13 * (2 * 111 - 2) = 2860`` time samples.

    The horizontal (Az/El) coordinates are converted to equatorial (RA/Dec)
    coordinates using the observation time and telescope location via
    astropy coordinate transforms.

    See Zhang et al. (2026), RASTI, rzag024 for details on the MeerKAT
    observation model.

    Parameters
    ----------
    telescope_lat : float, optional
        Telescope geodetic latitude in degrees. Default is -30.7130 (MeerKAT).
    telescope_lon : float, optional
        Telescope geodetic longitude in degrees. Default is 21.4430 (MeerKAT).
    telescope_height : float, optional
        Telescope height above sea level in metres. Default is 1054 (MeerKAT).
    elevation : float, optional
        Fixed telescope elevation angle in degrees. Default is 41.5.
    az_s : float, optional
        Start azimuth angle of the scan in degrees. Default is -60.3.
    az_e : float, optional
        End azimuth angle of the scan in degrees. Default is -42.3.
    start_time_utc : str, optional
        UTC start time of the observation in ISO format.
        Default is ``"2019-04-23 20:41:56.397"``.
    dt : float, optional
        Time sampling interval in seconds. Default is 2.0.
    return_eq_coords : bool, optional
        If True, return an astropy ``SkyCoord`` object instead of
        (theta, phi) arrays. Default is False.

    Returns
    -------
    t_list : NDArray[np.floating]
        Array of time stamps in seconds, shape ``(ntime,)``.
    theta_c : NDArray[np.floating]
        HEALPix colatitude of beam centres in radians, shape ``(ntime,)``.
        Only returned when ``return_eq_coords=False``.
    phi_c : NDArray[np.floating]
        HEALPix longitude (RA) of beam centres in radians, shape ``(ntime,)``.
        Only returned when ``return_eq_coords=False``.
    eq_coords : SkyCoord
        Equatorial coordinates of beam centres. Only returned when
        ``return_eq_coords=True``.
    """

    location = EarthLocation(
        lat=telescope_lat * u.deg,
        lon=telescope_lon * u.deg,
        height=telescope_height * u.m,
    )

    aux = np.linspace(az_s, az_e, 111)
    azimuths = np.concatenate((aux[1:-1][::-1], aux))
    azimuths = np.tile(azimuths, 13)

    # Length of TOD
    ntime = len(azimuths)
    t_list = np.arange(ntime) * dt

    # ---- Define start and end times in UTC ----
    start_time = Time(start_time_utc)
    UTC_list = start_time + TimeDelta(t_list, format="sec")  # Time list in UTC

    # ---- Create AltAz coordinate frame ----
    altaz_frame = AltAz(obstime=UTC_list, location=location)

    # ---- Convert Az/El to Equatorial (RA, Dec) ----
    eq_coords = SkyCoord(
        az=azimuths * u.deg, alt=elevation * u.deg, frame=altaz_frame
    ).transform_to("icrs")

    if return_eq_coords:
        return t_list, eq_coords
    else:
        theta_c = np.pi / 2 - eq_coords.dec.radian  # Convert Dec to theta
        phi_c = eq_coords.ra.radian  # RA is already phi

        return t_list, theta_c, phi_c


def sim_MeerKAT_scan_v2(
    az_s: float = -60.3,
    az_e: float = -42.3,
    dt: float = 2.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Generate a simplified MeerKAT scan pattern in azimuth only.

    Produces the same raster scan pattern as :func:`sim_MeerKAT_scan` but
    returns azimuth values directly without converting to equatorial
    coordinates. Useful for quick tests that do not require sky projection.

    Parameters
    ----------
    az_s : float, optional
        Start azimuth angle in degrees. Default is -60.3.
    az_e : float, optional
        End azimuth angle in degrees. Default is -42.3.
    dt : float, optional
        Time sampling interval in seconds. Default is 2.0.

    Returns
    -------
    t_list : NDArray[np.floating]
        Array of time stamps in seconds, shape ``(ntime,)``.
    azimuths : NDArray[np.floating]
        Azimuth angles in degrees at each time sample, shape ``(ntime,)``.
    """

    aux = np.linspace(az_s, az_e, 111)
    azimuths = np.concatenate((aux[1:-1][::-1], aux))
    azimuths = np.tile(azimuths, 13)

    # Length of TOD
    ntime = len(azimuths)
    t_list = np.arange(ntime) * dt

    return t_list, azimuths


def stacked_beam_map(
    theta_c: NDArray[np.floating],
    phi_c: NDArray[np.floating],
    FWHM: float = 1.1,
    NSIDE: int = 64,
    threshold: float = 0.011,
) -> tuple[NDArray[np.bool_], NDArray[np.floating]]:
    """Generate a stacked beam boolean map from a sequence of beam centres.

    For each time sample, a Gaussian beam is evaluated on the HEALPix grid
    and accumulated into a sum map. Pixels whose beam response exceeds
    ``threshold`` in *any* time sample are flagged True in the boolean map.

    The Gaussian beam profile is

    .. math::

        B(\\psi) = \\exp\\!\\left(-\\frac{\\psi^2}{2\\sigma^2}\\right),

    where :math:`\\psi` is the angular separation from the beam centre and
    :math:`\\sigma = \\mathrm{FWHM} / (2\\sqrt{2\\ln 2})`.

    Parameters
    ----------
    theta_c : NDArray[np.floating]
        HEALPix colatitudes of beam centres in radians, shape ``(ntime,)``.
    phi_c : NDArray[np.floating]
        HEALPix longitudes of beam centres in radians, shape ``(ntime,)``.
    FWHM : float, optional
        Full Width at Half Maximum of the Gaussian beam in degrees.
        Default is 1.1.
    NSIDE : int, optional
        HEALPix ``NSIDE`` resolution parameter. Default is 64.
    threshold : float, optional
        Minimum beam response for a pixel to be included in the boolean
        map. Default is 0.011 (approximately the 3-sigma level).

    Returns
    -------
    bool_map : NDArray[np.bool_]
        Boolean HEALPix map of shape ``(NPIX,)`` indicating which pixels
        fall within the beam footprint.
    sum_map : NDArray[np.floating]
        Sum of all beam responses at each pixel, shape ``(NPIX,)``.
        Useful for assessing cumulative beam coverage.
    """

    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma (degrees)
    sigma_rad = np.radians(sigma)  # Convert to radians

    NPIX = hp.nside2npix(NSIDE)

    # Get HEALPix pixel coordinates (theta, phi)
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))

    # Generate a initial boolean map with all pixels zero
    bool_map = np.zeros(NPIX, dtype=bool)
    sum_map = np.zeros(NPIX, dtype=float)

    ntime = len(theta_c)

    for ti in range(ntime):
        # Compute angular separation between each pixel and the beam center
        cos_sep = np.cos(theta) * np.cos(theta_c[ti]) + np.sin(theta) * np.sin(
            theta_c[ti]
        ) * np.cos(phi - phi_c[ti])
        cos_sep = np.clip(cos_sep, -1, 1)  # Ensure within valid range
        angular_sep = np.arccos(cos_sep)  # Separation in radians
        # Compute Gaussian beam response centered at (RA_center, Dec_center)
        beam_map = np.exp(-0.5 * (angular_sep / sigma_rad) ** 2)
        # Normalize the beam (optional, ensures peak = 1)
        beam_map /= np.max(beam_map)
        sum_map += beam_map
        # Get the "or" map of the bool_map and beam_map
        bool_map = np.logical_or(bool_map, beam_map > threshold)

    return bool_map, sum_map


def reduce_bool_maps_LOR(
    bool_maps: list[NDArray[np.bool_]],
) -> tuple[NDArray[np.bool_], NDArray[np.intp]]:
    """Reduce a list of boolean HEALPix maps using logical OR.

    Combines multiple per-scan boolean beam coverage maps into a single
    union map, identifying all pixels observed by *any* scan.

    Parameters
    ----------
    bool_maps : list of NDArray[np.bool\_]
        List of boolean HEALPix maps, each of shape ``(NPIX,)``.

    Returns
    -------
    reduced_map : NDArray[np.bool_]
        Combined boolean map, shape ``(NPIX,)``.
    pixel_indices : NDArray[np.intp]
        Indices of pixels that are True in the combined map.
    """
    reduced_map = np.logical_or.reduce(bool_maps)
    # Get the pixel indices of the "1" pixels:
    pixel_indices = np.where(reduced_map)[0]
    return reduced_map, pixel_indices


def reduce_bool_maps_LAND(
    bool_maps: list[NDArray[np.bool_]],
) -> tuple[NDArray[np.bool_], NDArray[np.intp]]:
    """Reduce a list of boolean HEALPix maps using logical AND.

    Combines multiple per-scan boolean beam coverage maps into a single
    intersection map, identifying only pixels observed by *all* scans.

    Parameters
    ----------
    bool_maps : list of NDArray[np.bool\_]
        List of boolean HEALPix maps, each of shape ``(NPIX,)``.

    Returns
    -------
    reduced_map : NDArray[np.bool_]
        Combined boolean map, shape ``(NPIX,)``.
    pixel_indices : NDArray[np.intp]
        Indices of pixels that are True in the combined map.
    """
    reduced_map = np.logical_and.reduce(bool_maps)
    # Get the pixel indices of the "1" pixels:
    pixel_indices = np.where(reduced_map)[0]
    return reduced_map, pixel_indices


def gaussian(
    x: float | NDArray[np.floating],
    mu: float = 0,
    sigma: float = 1,
) -> float | NDArray[np.floating]:
    """Evaluate a peak-normalised 1-D Gaussian function.

    Computes

    .. math::

        G(x) = \\exp\\!\\left(-\\frac{(x - \\mu)^2}{2\\sigma^2}\\right),

    which has a peak value of 1 at :math:`x = \\mu`. Note that this is
    *not* the probability-normalised Gaussian; the :math:`1/(\\sigma\\sqrt{2\\pi})`
    prefactor is omitted so that the beam peak equals unity.

    Parameters
    ----------
    x : float or NDArray[np.floating]
        Input value(s) at which to evaluate the Gaussian.
    mu : float, optional
        Mean (centre) of the Gaussian. Default is 0.
    sigma : float, optional
        Standard deviation (width) of the Gaussian. Default is 1.

    Returns
    -------
    float or NDArray[np.floating]
        Gaussian values at ``x``, same shape as the input.
    """
    # coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2  # Peak normalisation: peak = 1
    return np.exp(exponent)


def generate_Tsky_proj(
    full_bool_map: NDArray[np.bool_],
    theta_c: NDArray[np.floating],
    phi_c: NDArray[np.floating],
    FWHM: float = 1.1,
) -> NDArray[np.floating]:
    """Generate the sky temperature projection (beam) operator.

    Constructs a matrix :math:`\\mathbf{B}` of shape ``(ntime, npix)`` that
    projects a sky temperature vector :math:`\\mathbf{T}_\\mathrm{sky}` into
    the time domain via

    .. math::

        T_\\mathrm{sky}(t) = \\sum_p B_{tp}\\, T_\\mathrm{sky}(p),

    where each row of :math:`\\mathbf{B}` is a Gaussian beam response
    evaluated at the observed pixels for the beam centre at time *t*.

    The computation is parallelised across time samples using MPI via
    :func:`mpiutil.local_parallel_func`.

    See Zhang et al. (2026), RASTI, rzag024, Section 2 for the beam
    convolution model.

    Parameters
    ----------
    full_bool_map : NDArray[np.bool_]
        Boolean HEALPix map of shape ``(NPIX,)`` indicating the observed
        pixel footprint. Only True pixels are included in the projection.
    theta_c : NDArray[np.floating]
        HEALPix colatitudes of beam centres in radians, shape ``(ntime,)``.
    phi_c : NDArray[np.floating]
        HEALPix longitudes of beam centres in radians, shape ``(ntime,)``.
    FWHM : float, optional
        Full Width at Half Maximum of the Gaussian beam in degrees.
        Default is 1.1.

    Returns
    -------
    Tsky_proj : NDArray[np.floating]
        Beam projection operator of shape ``(ntime, npix)`` where
        ``npix = sum(full_bool_map)``.
    """
    NPIX = len(full_bool_map)
    NSIDE = hp.npix2nside(NPIX)
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    sigma_rad = np.radians(sigma)  # Convert to radians

    ntime = len(theta_c)

    # Get HEALPix pixel coordinates (theta, phi)
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))

    # Count the number of "1" pixels in bool_map
    num_pixels = np.sum(full_bool_map)
    # Get the pixel indices of the "1" pixels:
    pixel_indices = np.where(full_bool_map)[0]

    Tsky_proj = np.zeros((ntime, num_pixels))

    def func(ti: int) -> NDArray[np.floating]:
        theta_p = theta[pixel_indices]
        phi_p = phi[pixel_indices]
        cos_sep = np.cos(theta_p) * np.cos(theta_c[ti]) + np.sin(theta_p) * np.sin(
            theta_c[ti]
        ) * np.cos(phi_p - phi_c[ti])
        cos_sep = np.clip(cos_sep, -1, 1)  # Ensure within valid range
        angular_sep = np.arccos(cos_sep)  # Separation in radians
        # Compute Gaussian beam response centered at (RA_center, Dec_center)
        beam_map = gaussian(angular_sep, sigma=sigma_rad)
        return beam_map

    Tsky_proj = np.array(mpiutil.local_parallel_func(func, np.arange(ntime)))
    return Tsky_proj


class TODSimulation:
    """Single-scan Time-Ordered Data (TOD) simulation for MeerKAT.

    Generates a realistic simulated radio telescope observation following the
    data model described in Zhang et al. (2026), RASTI, rzag024:

    .. math::

        d(t) = T_\\mathrm{sys}(t)\\,\\bigl[1 + n(t)\\bigr]\\,g(t),

    where

    - :math:`T_\\mathrm{sys}(t) = T_\\mathrm{sky}(t) + T_\\mathrm{loc}(t)` is
      the system temperature,
    - :math:`T_\\mathrm{sky}(t) = \\mathbf{B}(t)\\,\\mathbf{T}_\\mathrm{sky}` is the
      beam-convolved sky temperature,
    - :math:`T_\\mathrm{loc}(t) = T_\\mathrm{ndiode}(t) + T_\\mathrm{rec}(t)` contains
      the noise diode and receiver temperature,
    - :math:`g(t) = \\exp(\\mathbf{G}\\,\\mathbf{p}_g)` is the time-varying
      instrument gain modelled by Legendre polynomials,
    - :math:`n(t)` is correlated flicker noise with a :math:`1/f^\\alpha`
      power spectrum.

    The simulation proceeds in the following steps:

    1. Generate a MeerKAT raster scan pattern (Az/El to RA/Dec).
    2. Build a Gaussian beam coverage map on the HEALPix grid.
    3. Construct the beam projection operator :math:`\\mathbf{B}`.
    4. Extract sky temperatures from the Global Sky Model (GSM) plus
       optional point source catalogue.
    5. Generate gain, noise diode, and receiver temperature time streams.
    6. Draw a flicker noise realisation.
    7. Combine all components into the final TOD.

    Attributes
    ----------
    nside : int
        HEALPix ``NSIDE`` resolution parameter.
    ntime : int
        Number of time samples in the scan.
    t_list : NDArray[np.floating]
        Time stamps in seconds, shape ``(ntime,)``.
    TOD_setting : NDArray[np.floating]
        Simulated TOD data, shape ``(ntime,)``.
    Tsys_setting : NDArray[np.floating]
        System temperature time stream, shape ``(ntime,)``.
    sky_params : NDArray[np.floating]
        Sky temperature values at observed pixels, shape ``(npix,)``.
    gains_setting : NDArray[np.floating]
        Gain time stream, shape ``(ntime,)``.
    noise_setting : NDArray[np.floating]
        Flicker noise realisation, shape ``(ntime,)``.
    Tsky_operator_setting : NDArray[np.floating]
        Beam projection operator, shape ``(ntime, npix)``.
    nd_rec_operator : NDArray[np.floating]
        Local temperature operator (noise diode + receiver), shape ``(ntime, 5)``.
    full_bool_map : NDArray[np.bool_]
        Boolean HEALPix map of observed pixels.
    pixel_indices : NDArray[np.intp]
        HEALPix pixel indices of observed pixels.
    gain_proj : NDArray[np.floating]
        Legendre polynomial basis for gain model, shape ``(ntime, 4)``.
    gain_params_setting : NDArray[np.floating]
        True gain parameters, shape ``(4,)``.
    nd_rec_params : NDArray[np.floating]
        Combined noise diode and receiver parameters, shape ``(5,)``.
    f0 : float
        Noise knee frequency in Hz (angular).
    fc : float
        Low-frequency cutoff :math:`f_c = 2\\pi / (N \\Delta t)` in rad/s.
    logf0 : float
        :math:`\\log_{10}(f_0)`.
    logfc : float
        :math:`\\log_{10}(f_c)`.
    alpha : float
        Flicker noise spectral index.
    sigma_2 : float
        White noise variance.

    Examples
    --------
    >>> sim = TODSimulation(nside=64, freq=750)
    >>> print(sim.TOD_setting.shape)
    (2860,)
    >>> summary = sim.get_simulation_summary()
    """

    def __init__(
        self,
        nside: int = 64,
        elevation: float = 41.5,
        az_s: float = -60.3,
        az_e: float = -42.3,
        start_time_utc: str = "2019-04-23 20:41:56.397",
        FWHM: float = 1.1,
        threshold: float = 0.0111,
        freq: float = 750,
        T_ndiode: float = 15.0,
        rec_params: NDArray[np.floating] | list[float] | None = None,
        gain_params: NDArray[np.floating] | list[float] | None = None,
        dtime: float = 2,
        logf0: float = -4.874571109426952,
        alpha: float = 2.0,
        sigma_2: float | None = None,
        ptsrc_path: str | None = None,
    ) -> None:
        """Initialise and run a single-scan TOD simulation.

        All simulation products are computed immediately upon construction
        and stored as instance attributes.

        Parameters
        ----------
        nside : int, optional
            HEALPix ``NSIDE`` resolution parameter. Default is 64.
        elevation : float, optional
            Telescope elevation angle in degrees. Default is 41.5.
        az_s : float, optional
            Start azimuth angle in degrees. Default is -60.3.
        az_e : float, optional
            End azimuth angle in degrees. Default is -42.3.
        start_time_utc : str, optional
            UTC start time in ISO format.
            Default is ``"2019-04-23 20:41:56.397"``.
        FWHM : float, optional
            Full Width at Half Maximum of the Gaussian beam in degrees.
            Default is 1.1.
        threshold : float, optional
            Beam response threshold for pixel inclusion (approximately
            3-sigma level). Default is 0.0111.
        freq : float, optional
            Observation frequency in MHz. Default is 750.
        T_ndiode : float, optional
            Noise diode temperature in Kelvin. Default is 15.0.
        rec_params : array-like or None, optional
            Legendre polynomial coefficients for the receiver temperature
            model, shape ``(4,)``. Default is ``[12.6, 0.5, 0.5, 0.5]``.
        gain_params : array-like or None, optional
            Legendre polynomial coefficients for the gain model,
            shape ``(4,)``. Default is
            ``[6.312, 0.420, 0.264, 0.056]``.
        dtime : float, optional
            Time sampling interval in seconds. Default is 2.
        logf0 : float, optional
            :math:`\\log_{10}` of the flicker noise knee frequency (Hz).
            Default is -4.875.
        alpha : float, optional
            Flicker noise spectral index :math:`\\alpha` in the power
            spectrum :math:`P(f) \\propto 1/f^\\alpha`. Default is 2.0.
        sigma_2 : float or None, optional
            White noise variance. Default is ``1 / 4e5``.
        ptsrc_path : str or None, optional
            Path to a ``.npy`` file containing a point source catalogue
            at 408 MHz (GLEAM format). If provided, sources are scaled
            to the observation frequency assuming a spectral index of
            -2.3. Default is None.
        """
        # Store input parameters
        self.nside = nside
        self.elevation = elevation
        self.az_s = az_s
        self.az_e = az_e
        self.start_time_utc = start_time_utc
        self.FWHM = FWHM
        self.threshold = threshold
        self.freq = freq
        self.T_ndiode = T_ndiode
        self.dtime = dtime
        self.logf0 = logf0
        self.alpha = alpha
        self.ptsrc_path = ptsrc_path

        # Set default parameters if not provided
        if rec_params is None:
            self.rec_params = np.array([12.6, 0.5, 0.5, 0.5])
        else:
            self.rec_params = np.array(rec_params)

        if gain_params is None:
            self.gain_params_setting = np.array(
                [6.31194264, 0.42038942, 0.264222, 0.05578821]
            )
        else:
            self.gain_params_setting = np.array(gain_params)

        if sigma_2 is None:
            self.sigma_2 = 1 / (4e5)
        else:
            self.sigma_2 = sigma_2

        # Run the simulation
        self._simulate()

    def _simulate(self) -> None:
        """Run the complete simulation pipeline and populate all attributes.

        This is called automatically by ``__init__`` and should not normally
        be called directly. It performs the following steps:

        1. Generate MeerKAT scan coordinates.
        2. Build the beam coverage boolean map.
        3. Construct the beam projection operator.
        4. Extract sky temperatures from the Global Sky Model.
        5. Build noise diode, receiver, and gain time streams.
        6. Generate a flicker noise realisation.
        7. Assemble the final TOD.
        8. Identify calibration pixel indices.
        """

        # Get the timestream of beam centers
        self.t_list, self.theta_c_setting, self.phi_c_setting = sim_MeerKAT_scan(
            elevation=self.elevation,
            az_s=self.az_s,
            az_e=self.az_e,
            start_time_utc=self.start_time_utc,
        )

        # Generate beam map
        self.bool_map_setting, self.integrated_beam_setting = stacked_beam_map(
            self.theta_c_setting,
            self.phi_c_setting,
            FWHM=self.FWHM,
            NSIDE=self.nside,
            threshold=self.threshold,
        )

        # Set up maps and indices
        self.full_bool_map = self.bool_map_setting
        self.pixel_indices = np.where(self.full_bool_map)[0]
        self.integrated_beam = self.integrated_beam_setting

        # Generate sky projection operator
        self.Tsky_operator_setting = generate_Tsky_proj(
            self.full_bool_map, self.theta_c_setting, self.phi_c_setting, FWHM=self.FWHM
        )

        # Generate sky parameters
        self.sky_params = self._sky_vector(self.pixel_indices, self.freq, self.nside)
        print(f"Number of pixels: {len(self.pixel_indices)}")

        # Set up time-dependent parameters
        self.ntime = len(self.t_list)
        self.ndiode_proj = self._generate_vector(self.ntime)

        # Set up noise diode and receiver operator
        self.nd_rec_operator = np.zeros((self.ntime, 5))
        self.nd_rec_operator[:, 0] = self.ndiode_proj
        self.nd_rec_operator[:, 1:] = Leg_poly_proj(4, self.t_list)

        # Combine noise diode and receiver parameters
        self.nd_rec_params = np.zeros(5)
        self.nd_rec_params[0] = self.T_ndiode
        self.nd_rec_params[1:] = self.rec_params

        # Set up gain projection
        self.gain_proj = Leg_poly_proj(4, self.t_list)
        self.gains_setting = self.gain_proj @ self.gain_params_setting
        print(f"Gain parameters: {self.gain_params_setting}")

        # Calculate flicker noise parameters
        self.fc = (1 / self.ntime / self.dtime) * 2 * np.pi
        self.logfc = np.log10(self.fc)
        self.f0 = 10**self.logf0

        # Generate noise
        self.noise_setting = sim_noise(
            self.f0, self.fc, self.alpha, self.t_list, white_n_variance=self.sigma_2
        )

        # Calculate system temperature and final TOD
        self.Tsys_setting = (
            self.Tsky_operator_setting @ self.sky_params
            + self.nd_rec_operator @ self.nd_rec_params
        )
        self.TOD_setting = (
            self.Tsys_setting * (1 + self.noise_setting) * self.gains_setting
        )

        # Set up calibration indices
        top_200_beam_indices = np.argpartition(
            self.integrated_beam_setting[self.pixel_indices], -200
        )[-200:]
        self.calibration_5_indices = [
            top_200_beam_indices[int(i * 200 / 5)] for i in range(5)
        ]

        # For a 1D array 'integrated_beam_setting'
        n_cal_pixs = 1
        top_20_beam_indices = np.argpartition(
            self.integrated_beam_setting[self.pixel_indices], -20
        )[-20:]
        top_n_sky_indices = np.argpartition(
            self.sky_params[top_20_beam_indices], -n_cal_pixs
        )[-n_cal_pixs:]
        self.calibration_1_indice = top_20_beam_indices[top_n_sky_indices]

    def _sky_vector(
        self,
        pixel_indices: NDArray[np.intp],
        freq: float,
        Nside: int = 64,
        sky_model: Any | None = None,
    ) -> NDArray[np.floating]:
        """Generate sky temperature vector for observed pixels.

        Produces the sky model using the Global Sky Model (GSM) at the
        requested frequency, optionally adding point sources from the
        GLEAM catalogue scaled with a spectral index of -2.3.

        Parameters
        ----------
        pixel_indices : NDArray[np.intp]
            HEALPix pixel indices to extract, shape ``(npix,)``.
        freq : float
            Observation frequency in MHz.
        Nside : int, optional
            HEALPix ``NSIDE`` for the output map. Default is 64.
        sky_model : callable or None, optional
            Custom sky model function that takes frequency (MHz) and returns
            a full-sky HEALPix map. If None, uses ``pygdsm.GlobalSkyModel``.

        Returns
        -------
        NDArray[np.floating]
            Sky temperatures at the requested pixels in Kelvin,
            shape ``(npix,)``.
        """
        if sky_model is None:
            from pygdsm import GlobalSkyModel

            gsm = GlobalSkyModel()
            skymap = gsm.generate(freq)
        else:
            skymap = sky_model(freq)

        skymap = hp.ud_grade(skymap, nside_out=Nside)
        if self.ptsrc_path is not None:
            ptsrc = np.load(self.ptsrc_path) * (freq / 408) ** (-2.3)
            ptsrc_map = hp.ud_grade(ptsrc, nside_out=Nside)
            skymap = skymap + ptsrc_map
        return skymap[pixel_indices]

    def _generate_vector(self, ntime: int) -> NDArray[np.floating]:
        """Generate a periodic noise diode injection pattern.

        Creates a binary vector where the noise diode is fired every
        10th time sample (duty cycle of 10%).

        Parameters
        ----------
        ntime : int
            Total number of time samples.

        Returns
        -------
        NDArray[np.floating]
            Binary vector of shape ``(ntime,)`` with 1 at every 10th
            element and 0 elsewhere.
        """
        vector = np.zeros(ntime)
        for i in range(0, ntime, 10):
            vector[i] = 1
        return vector

    def get_simulation_summary(self) -> dict[str, Any]:
        """Return a summary of simulation parameters and key results.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            - ``nside``: HEALPix resolution
            - ``n_pixels``: number of observed pixels
            - ``n_time_samples``: number of time samples
            - ``frequency_MHz``: observation frequency
            - ``beam_FWHM_deg``: beam FWHM
            - ``elevation_deg``: telescope elevation
            - ``azimuth_range_deg``: (az_start, az_end) tuple
            - ``gain_params``: gain polynomial coefficients
            - ``noise_knee_freq_Hz``: knee frequency
            - ``noise_spectral_index``: flicker noise alpha
            - ``T_ndiode_K``: noise diode temperature
            - ``receiver_params_K``: receiver temperature coefficients
        """
        summary = {
            "nside": self.nside,
            "n_pixels": len(self.pixel_indices),
            "n_time_samples": self.ntime,
            "frequency_MHz": self.freq,
            "beam_FWHM_deg": self.FWHM,
            "elevation_deg": self.elevation,
            "azimuth_range_deg": (self.az_s, self.az_e),
            "gain_params": self.gain_params_setting,
            "noise_knee_freq_Hz": self.f0,
            "noise_spectral_index": self.alpha,
            "T_ndiode_K": self.T_ndiode,
            "receiver_params_K": self.rec_params,
        }
        return summary


class MultiTODSimulation:
    """Multiple-scan Time-Ordered Data (TOD) simulation for MeerKAT.

    Generates simulated TODs for two independent scans -- a "setting" scan
    and a "rising" scan -- that observe overlapping regions of sky. This
    enables testing of the joint Gibbs sampler that recovers a shared sky
    map :math:`\\mathbf{T}_\\mathrm{sky}` from multiple observations, each
    with independent gains and noise realisations.

    The data model for each scan *i* is

    .. math::

        d_i(t) = T_\\mathrm{sys,i}(t)\\,\\bigl[1 + n_i(t)\\bigr]\\,g_i(t),

    where the sky component :math:`T_\\mathrm{sky}` is shared and
    all other components are per-scan. See Zhang et al. (2026), RASTI,
    rzag024 for the full hierarchical model.

    The simulation proceeds by:

    1. Generating scan patterns for both setting and rising scans.
    2. Building per-scan beam coverage maps and combining them with
       logical OR to define the full observed pixel footprint.
    3. Constructing per-scan beam projection operators on the shared
       pixel set.
    4. Extracting sky temperatures from the GSM.
    5. Generating per-scan gains, noise, and local temperature streams.
    6. Assembling per-scan TODs.
    7. Identifying calibration pixel indices.

    Attributes
    ----------
    nside : int
        HEALPix ``NSIDE`` resolution parameter.
    ntime : int
        Number of time samples per scan.
    t_list : NDArray[np.floating]
        Time stamps in seconds, shape ``(ntime,)``.
    TOD_setting : NDArray[np.floating]
        Simulated TOD for the setting scan, shape ``(ntime,)``.
    TOD_rising : NDArray[np.floating]
        Simulated TOD for the rising scan, shape ``(ntime,)``.
    Tsys_setting : NDArray[np.floating]
        System temperature for the setting scan, shape ``(ntime,)``.
    Tsys_rising : NDArray[np.floating]
        System temperature for the rising scan, shape ``(ntime,)``.
    sky_params : NDArray[np.floating]
        Sky temperature values at observed pixels, shape ``(npix,)``.
    gains_setting : NDArray[np.floating]
        Gain time stream for the setting scan, shape ``(ntime,)``.
    gains_rising : NDArray[np.floating]
        Gain time stream for the rising scan, shape ``(ntime,)``.
    noise_setting : NDArray[np.floating]
        Flicker noise realisation for the setting scan, shape ``(ntime,)``.
    noise_rising : NDArray[np.floating]
        Flicker noise realisation for the rising scan, shape ``(ntime,)``.
    Tsky_operator_setting : NDArray[np.floating]
        Beam projection operator for setting scan, shape ``(ntime, npix)``.
    Tsky_operator_rising : NDArray[np.floating]
        Beam projection operator for rising scan, shape ``(ntime, npix)``.
    nd_rec_operator : NDArray[np.floating]
        Local temperature operator, shape ``(ntime, 5)``.
    full_bool_map : NDArray[np.bool_]
        Union boolean HEALPix map of observed pixels.
    pixel_indices : NDArray[np.intp]
        HEALPix pixel indices of the union footprint.
    gain_proj : NDArray[np.floating]
        Legendre polynomial basis for gain model, shape ``(ntime, 4)``.
    gain_params_setting : NDArray[np.floating]
        True gain parameters for the setting scan, shape ``(4,)``.
    gain_params_rising : NDArray[np.floating]
        True gain parameters for the rising scan, shape ``(4,)``.
    calibration_1_index : NDArray[np.intp]
        Index of the single brightest calibration pixel.
    calibration_5_indices : list[int]
        Indices of five calibration pixels spread across the beam.

    Examples
    --------
    >>> sim = MultiTODSimulation(nside=64, freq=750)
    >>> print(sim.TOD_setting.shape, sim.TOD_rising.shape)
    (2860,) (2860,)
    >>> data = sim.get_tod_data()
    """

    def __init__(
        self,
        nside: int = 64,
        FWHM: float = 1.1,
        threshold: float = 0.0111,
        freq: float = 750,
        T_ndiode: float = 15.0,
        rec_params: NDArray[np.floating] | list[float] | None = None,
        dtime: float = 2,
        alpha: float = 2.0,
        logf0: float = -4.874571109426952,
        sigma_2: float | None = None,
        # Setting scan parameters
        setting_elevation: float = 41.5,
        setting_az_s: float = -60.3,
        setting_az_e: float = -42.3,
        setting_start_time: str = "2019-04-23 20:41:56.397",
        setting_gain_params: NDArray[np.floating] | list[float] | None = None,
        # Rising scan parameters
        rising_elevation: float = 40.5,
        rising_az_s: float = 43.7,
        rising_az_e: float = 61.7,
        rising_start_time: str = "2019-03-30 17:19:02.397",
        rising_gain_params: NDArray[np.floating] | list[float] | None = None,
        ptsrc_path: str | None = None,
    ) -> None:
        """Initialise and run a multi-scan TOD simulation.

        All simulation products for both setting and rising scans are
        computed immediately upon construction and stored as instance
        attributes.

        Parameters
        ----------
        nside : int, optional
            HEALPix ``NSIDE`` resolution parameter. Default is 64.
        FWHM : float, optional
            Full Width at Half Maximum of the Gaussian beam in degrees.
            Default is 1.1.
        threshold : float, optional
            Beam response threshold for pixel inclusion (approximately
            3-sigma level). Default is 0.0111.
        freq : float, optional
            Observation frequency in MHz. Default is 750.
        T_ndiode : float, optional
            Noise diode temperature in Kelvin. Default is 15.0.
        rec_params : array-like or None, optional
            Legendre polynomial coefficients for the receiver temperature
            model, shape ``(4,)``. Default is ``[12.6, 0.5, 0.5, 0.5]``.
        dtime : float, optional
            Time sampling interval in seconds. Default is 2.
        alpha : float, optional
            Flicker noise spectral index :math:`\\alpha`. Default is 2.0.
        logf0 : float, optional
            :math:`\\log_{10}` of the flicker noise knee frequency (Hz).
            Default is -4.875.
        sigma_2 : float or None, optional
            White noise variance. Default is ``1 / 4e5``.
        setting_elevation : float, optional
            Elevation angle for the setting scan in degrees. Default is 41.5.
        setting_az_s : float, optional
            Start azimuth for the setting scan in degrees. Default is -60.3.
        setting_az_e : float, optional
            End azimuth for the setting scan in degrees. Default is -42.3.
        setting_start_time : str, optional
            UTC start time for the setting scan. Default is
            ``"2019-04-23 20:41:56.397"``.
        setting_gain_params : array-like or None, optional
            Gain polynomial coefficients for the setting scan, shape ``(4,)``.
            Default is ``[6.312, 0.420, 0.264, 0.056]``.
        rising_elevation : float, optional
            Elevation angle for the rising scan in degrees. Default is 40.5.
        rising_az_s : float, optional
            Start azimuth for the rising scan in degrees. Default is 43.7.
        rising_az_e : float, optional
            End azimuth for the rising scan in degrees. Default is 61.7.
        rising_start_time : str, optional
            UTC start time for the rising scan. Default is
            ``"2019-03-30 17:19:02.397"``.
        rising_gain_params : array-like or None, optional
            Gain polynomial coefficients for the rising scan, shape ``(4,)``.
            Default is ``[6.845, 0.142, 0.744, 0.779]``.
        ptsrc_path : str or None, optional
            Path to a ``.npy`` file containing a point source catalogue
            at 408 MHz. If provided, sources are scaled to the observation
            frequency with spectral index -2.3. Default is None.
        """
        # Store input parameters
        self.nside = nside
        self.FWHM = FWHM
        self.threshold = threshold
        self.freq = freq
        self.T_ndiode = T_ndiode
        self.dtime = dtime
        self.alpha = alpha
        self.logf0 = logf0
        self.ptsrc_path = ptsrc_path

        # Setting scan parameters
        self.setting_elevation = setting_elevation
        self.setting_az_s = setting_az_s
        self.setting_az_e = setting_az_e
        self.setting_start_time = setting_start_time

        # Rising scan parameters
        self.rising_elevation = rising_elevation
        self.rising_az_s = rising_az_s
        self.rising_az_e = rising_az_e
        self.rising_start_time = rising_start_time

        # Set default parameters if not provided
        if rec_params is None:
            self.rec_params = np.array([12.6, 0.5, 0.5, 0.5])
        else:
            self.rec_params = np.array(rec_params)

        if setting_gain_params is None:
            self.gain_params_setting = np.array(
                [6.31194264, 0.42038942, 0.264222, 0.05578821]
            )
        else:
            self.gain_params_setting = np.array(setting_gain_params)

        if rising_gain_params is None:
            self.gain_params_rising = np.array(
                [6.84507868, 0.14156859, 0.7441104, 0.77863955]
            )
        else:
            self.gain_params_rising = np.array(rising_gain_params)

        if sigma_2 is None:
            self.sigma_2 = 1 / (4e5)
        else:
            self.sigma_2 = sigma_2

        # Run the simulation
        self._simulate()

    def _simulate(self) -> None:
        """Run the complete multi-scan simulation pipeline.

        This is called automatically by ``__init__``. It generates scan
        patterns, beam maps, sky projections, gains, noise, and TODs for
        both the setting and rising scans, then sets up calibration indices.
        """

        # Generate setting scan
        _, self.theta_c_setting, self.phi_c_setting = sim_MeerKAT_scan(
            elevation=self.setting_elevation,
            az_s=self.setting_az_s,
            az_e=self.setting_az_e,
            start_time_utc=self.setting_start_time,
        )
        self.bool_map_setting, self.integrated_beam_setting = stacked_beam_map(
            self.theta_c_setting,
            self.phi_c_setting,
            FWHM=self.FWHM,
            NSIDE=self.nside,
            threshold=self.threshold,
        )

        # Generate rising scan
        self.t_list, self.theta_c_rising, self.phi_c_rising = sim_MeerKAT_scan(
            elevation=self.rising_elevation,
            az_s=self.rising_az_s,
            az_e=self.rising_az_e,
            start_time_utc=self.rising_start_time,
        )
        self.bool_map_rising, self.integrated_beam_rising = stacked_beam_map(
            self.theta_c_rising,
            self.phi_c_rising,
            FWHM=self.FWHM,
            NSIDE=self.nside,
            threshold=self.threshold,
        )

        # Combine maps using logical OR
        self.full_bool_map, self.pixel_indices = reduce_bool_maps_LOR(
            [self.bool_map_setting, self.bool_map_rising]
        )
        self.integrated_beam = (
            self.integrated_beam_setting + self.integrated_beam_rising
        )

        # Generate sky projection operators
        self.Tsky_operator_setting = generate_Tsky_proj(
            self.full_bool_map, self.theta_c_setting, self.phi_c_setting, FWHM=self.FWHM
        )
        self.Tsky_operator_rising = generate_Tsky_proj(
            self.full_bool_map, self.theta_c_rising, self.phi_c_rising, FWHM=self.FWHM
        )

        # Generate sky parameters
        self.sky_params = self._sky_vector(self.pixel_indices, self.freq, self.nside)
        print(f"Number of pixels: {len(self.pixel_indices)}")

        # Set up time-dependent parameters
        self.ntime = len(self.t_list)
        self.ndiode_proj = self._generate_vector(self.ntime)

        # Set up noise diode and receiver operator
        self.nd_rec_operator = np.zeros((self.ntime, 5))
        self.nd_rec_operator[:, 0] = self.ndiode_proj
        self.nd_rec_operator[:, 1:] = Leg_poly_proj(4, self.t_list)

        # Combine noise diode and receiver parameters
        self.nd_rec_params = np.zeros(5)
        self.nd_rec_params[0] = self.T_ndiode
        self.nd_rec_params[1:] = self.rec_params

        # Set up gain projections and gains
        self.gain_proj = Leg_poly_proj(4, self.t_list)
        self.gains_setting = self.gain_proj @ self.gain_params_setting
        self.gains_rising = self.gain_proj @ self.gain_params_rising

        # Calculate flicker noise parameters
        self.fc = (1 / self.ntime / self.dtime) * 2 * np.pi
        self.logfc = np.log10(self.fc)
        self.f0 = 10**self.logf0

        # Generate noise for both scans
        self.noise_setting = sim_noise(
            self.f0, self.fc, self.alpha, self.t_list, white_n_variance=self.sigma_2
        )
        self.noise_rising = sim_noise(
            self.f0, self.fc, self.alpha, self.t_list, white_n_variance=self.sigma_2
        )

        # Calculate system temperatures and final TODs
        self.Tsys_setting = (
            self.Tsky_operator_setting @ self.sky_params
            + self.nd_rec_operator @ self.nd_rec_params
        )
        self.TOD_setting = (
            self.Tsys_setting * (1 + self.noise_setting) * self.gains_setting
        )

        self.Tsys_rising = (
            self.Tsky_operator_rising @ self.sky_params
            + self.nd_rec_operator @ self.nd_rec_params
        )
        self.TOD_rising = self.Tsys_rising * (1 + self.noise_rising) * self.gains_rising

        # Generate additional maps for analysis
        self._generate_analysis_maps()

        # Set up calibration indices
        self._setup_calibration_indices()

    def _sky_vector(
        self,
        pixel_indices: NDArray[np.intp],
        freq: float,
        Nside: int = 64,
        sky_model: Any | None = None,
    ) -> NDArray[np.floating]:
        """Generate sky temperature vector for observed pixels.

        Produces the sky model using the Global Sky Model (GSM) at the
        requested frequency, optionally adding point sources from the
        GLEAM catalogue scaled with a spectral index of -2.3.

        Parameters
        ----------
        pixel_indices : NDArray[np.intp]
            HEALPix pixel indices to extract, shape ``(npix,)``.
        freq : float
            Observation frequency in MHz.
        Nside : int, optional
            HEALPix ``NSIDE`` for the output map. Default is 64.
        sky_model : callable or None, optional
            Custom sky model function that takes frequency (MHz) and returns
            a full-sky HEALPix map. If None, uses ``pygdsm.GlobalSkyModel``.

        Returns
        -------
        NDArray[np.floating]
            Sky temperatures at the requested pixels in Kelvin,
            shape ``(npix,)``.
        """
        if sky_model is None:
            from pygdsm import GlobalSkyModel

            gsm = GlobalSkyModel()
            skymap = gsm.generate(freq)
        else:
            skymap = sky_model(freq)

        skymap = hp.ud_grade(skymap, nside_out=Nside)
        if self.ptsrc_path is not None:
            ptsrc = np.load(self.ptsrc_path) * (freq / 408) ** (-2.3)
            ptsrc_map = hp.ud_grade(ptsrc, nside_out=Nside)
            skymap = skymap + ptsrc_map
        return skymap[pixel_indices]

    def _generate_vector(self, ntime: int) -> NDArray[np.floating]:
        """Generate a periodic noise diode injection pattern.

        Creates a binary vector where the noise diode is fired every
        10th time sample (duty cycle of 10%).

        Parameters
        ----------
        ntime : int
            Total number of time samples.

        Returns
        -------
        NDArray[np.floating]
            Binary vector of shape ``(ntime,)`` with 1 at every 10th
            element and 0 elsewhere.
        """
        vector = np.zeros(ntime)
        for i in range(0, ntime, 10):
            vector[i] = 1
        return vector

    def _generate_analysis_maps(self) -> None:
        """Generate additional HEALPix maps for diagnostic analysis.

        Converts the beam centre coordinates of the setting scan to
        HEALPix pixel indices and builds a boolean map of beam centre
        locations (as opposed to the full beam footprint).
        """
        # Convert theta/phi coordinates to HEALPix pixels for setting scan
        self.pixels_c_setting = [
            hp.ang2pix(nside=self.nside, theta=theta, phi=phi)
            for theta, phi in zip(self.theta_c_setting, self.phi_c_setting)
        ]
        self.bool_map_c_setting = np.zeros(hp.nside2npix(self.nside))
        self.bool_map_c_setting[self.pixels_c_setting] = 1

    def _setup_calibration_indices(self) -> None:
        """Identify calibration pixel indices for gain calibration.

        Selects calibration pixels by finding the brightest sky pixels
        within the most-observed beam region. Two sets are provided:

        - A single pixel (``calibration_1_index``): the brightest sky
          pixel among the top-20 most-observed pixels.
        - Five pixels (``calibration_5_indices``): evenly spaced among
          the top-200 most-observed pixels.
        """
        # Single calibration pixel
        n_cal_pixs = 1
        top_20_beam_indices = np.argpartition(
            self.integrated_beam[self.pixel_indices], -20
        )[-20:]
        top_n_sky_indices = np.argpartition(
            self.sky_params[top_20_beam_indices], -n_cal_pixs
        )[-n_cal_pixs:]
        self.calibration_1_index = top_20_beam_indices[top_n_sky_indices]

        # Five calibration pixels
        n_cal_pixs = 5
        top_200_beam_indices = np.argpartition(
            self.integrated_beam[self.pixel_indices], -200
        )[-200:]
        self.calibration_5_indices = [
            top_200_beam_indices[int(i * 200 / n_cal_pixs)] for i in range(n_cal_pixs)
        ]

    def get_simulation_summary(self) -> dict[str, Any]:
        """Return a summary of simulation parameters and key results.

        Returns
        -------
        dict[str, Any]
            Nested dictionary containing:

            - ``nside``: HEALPix resolution
            - ``n_pixels``: number of observed pixels in the union footprint
            - ``n_time_samples``: number of time samples per scan
            - ``frequency_MHz``: observation frequency
            - ``beam_FWHM_deg``: beam FWHM
            - ``setting_scan``: dict with elevation, azimuth range, and
              gain parameters for the setting scan
            - ``rising_scan``: dict with elevation, azimuth range, and
              gain parameters for the rising scan
            - ``noise_knee_freq_Hz``: knee frequency
            - ``noise_spectral_index``: flicker noise alpha
            - ``T_ndiode_K``: noise diode temperature
            - ``receiver_params_K``: receiver temperature coefficients
            - ``calibration_pixels``: dict with calibration pixel indices
        """
        summary = {
            "nside": self.nside,
            "n_pixels": len(self.pixel_indices),
            "n_time_samples": self.ntime,
            "frequency_MHz": self.freq,
            "beam_FWHM_deg": self.FWHM,
            "setting_scan": {
                "elevation_deg": self.setting_elevation,
                "azimuth_range_deg": (self.setting_az_s, self.setting_az_e),
                "gain_params": self.gain_params_setting,
            },
            "rising_scan": {
                "elevation_deg": self.rising_elevation,
                "azimuth_range_deg": (self.rising_az_s, self.rising_az_e),
                "gain_params": self.gain_params_rising,
            },
            "noise_knee_freq_Hz": self.f0,
            "noise_spectral_index": self.alpha,
            "T_ndiode_K": self.T_ndiode,
            "receiver_params_K": self.rec_params,
            "calibration_pixels": {
                "single_cal_index": self.calibration_1_index,
                "five_cal_indices": self.calibration_5_indices,
            },
        }
        return summary

    def get_tod_data(self) -> dict[str, Any]:
        """Return TOD data for both scans in a format suitable for the Gibbs sampler.

        Packages the simulated data and operators needed by
        :func:`hydra_tod.full_Gibbs_sampler.TOD_Gibbs_sampler` into a
        single dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            - ``TOD_setting``: setting scan TOD, shape ``(ntime,)``
            - ``TOD_rising``: rising scan TOD, shape ``(ntime,)``
            - ``t_list``: time stamps, shape ``(ntime,)``
            - ``gain_proj``: Legendre polynomial basis, shape ``(ntime, 4)``
            - ``Tsky_operators``: list of beam operators for
              [setting, rising] scans
            - ``nd_rec_operator``: local temperature operator,
              shape ``(ntime, 5)``
            - ``logfc``: :math:`\\log_{10}` of the low-frequency cutoff
        """
        return {
            "TOD_setting": self.TOD_setting,
            "TOD_rising": self.TOD_rising,
            "t_list": self.t_list,
            "gain_proj": self.gain_proj,
            "Tsky_operators": [self.Tsky_operator_setting, self.Tsky_operator_rising],
            "nd_rec_operator": self.nd_rec_operator,
            "logfc": self.logfc,
        }


def eq_coordinates() -> tuple[SkyCoord, SkyCoord]:
    """Compute equatorial coordinates for default setting and rising scans.

    Convenience function that calls :func:`sim_MeerKAT_scan` with
    hard-coded parameters for the standard MeerKAT setting and rising
    scan configurations, returning astropy ``SkyCoord`` objects for
    each.

    Returns
    -------
    eq_coords_setting : SkyCoord
        Equatorial coordinates for the setting scan beam centres.
    eq_coords_rising : SkyCoord
        Equatorial coordinates for the rising scan beam centres.
    """
    # Get the timestream of beam centers (theta_c, phi_c) for each scan
    _, eq_coords_setting = sim_MeerKAT_scan(
        elevation=41.5,
        az_s=-60.3,
        az_e=-42.3,
        start_time_utc="2019-04-23 20:41:56.397",
        return_eq_coords=True,
    )

    _, eq_coords_rising = sim_MeerKAT_scan(
        elevation=40.5,
        az_s=43.7,
        az_e=61.7,
        start_time_utc="2019-03-30 17:19:02.397",
        return_eq_coords=True,
    )
    return eq_coords_setting, eq_coords_rising

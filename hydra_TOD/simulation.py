import numpy as np
import healpy as hp
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time, TimeDelta
import astropy.units as u
from utils import Leg_poly_proj, view_samples
from flicker_model import sim_noise, flicker_cov
import mpiutil


def sim_MeerKAT_scan(telescope_lat=-30.7130, 
                     telescope_lon=21.4430, 
                     telescope_height=1054, 
                     elevation=41.5, 
                     az_s=-60.3, 
                     az_e=-42.3, 
                     start_time_utc = "2019-04-23 20:41:56.397", 
                     dt=2.0,
                     return_eq_coords=False):

    location = EarthLocation(lat=telescope_lat * u.deg, lon=telescope_lon * u.deg, height=telescope_height * u.m)

    aux = np.linspace(az_s, az_e, 111)
    azimuths = np.concatenate((aux[1:-1][::-1], aux))
    azimuths = np.tile(azimuths, 13)

    # Length of TOD
    ntime = len(azimuths)
    t_list = np.arange(ntime) * dt

    # ---- Define start and end times in UTC ----
    start_time = Time(start_time_utc) 
    UTC_list = start_time + TimeDelta(t_list, format='sec') # Time list in UTC

    # ---- Create AltAz coordinate frame ----
    altaz_frame = AltAz(obstime=UTC_list, location=location)

    # ---- Convert Az/El to Equatorial (RA, Dec) ----
    eq_coords = SkyCoord(az=azimuths*u.deg, alt=elevation*u.deg, frame=altaz_frame).transform_to("icrs")

    if return_eq_coords:
        return t_list, eq_coords
    else:
        theta_c = np.pi/2 - eq_coords.dec.radian  # Convert Dec to theta
        phi_c = eq_coords.ra.radian               # RA is already phi
        
        return t_list, theta_c, phi_c

def stacked_beam_map(theta_c, phi_c, 
                     FWHM=1.1, 
                     NSIDE=64, 
                     threshold=0.011):
    """
    Generate a stacked beam map from a list of beam centers.
    The beam map is a boolean map, where True indicates that the pixel is within the beam.
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
        cos_sep = np.cos(theta) * np.cos(theta_c[ti]) + np.sin(theta) * np.sin(theta_c[ti]) * np.cos(phi - phi_c[ti])
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


def reduce_bool_maps_LOR(bool_maps):
    """
    Reduce a list of boolean maps using the "logical or" operation.
    """
    reduced_map = np.logical_or.reduce(bool_maps)
    # Get the pixel indices of the "1" pixels:
    pixel_indices = np.where(reduced_map)[0]
    return reduced_map, pixel_indices

def reduce_bool_maps_LAND(bool_maps):
    """
    Reduce a list of boolean maps using the "logical and" operation.
    """
    reduced_map = np.logical_and.reduce(bool_maps)
    # Get the pixel indices of the "1" pixels:
    pixel_indices = np.where(reduced_map)[0]
    return reduced_map, pixel_indices

def gaussian(x, mu=0, sigma=1):
    """
    Calculate normalized 1D Gaussian function values
    
    Parameters:
    x (float or array): Input value(s)
    mu (float): Mean (default: 0)
    sigma (float): Standard deviation (default: 1)
    
    Returns:
    float or array: Gaussian values at x
    """
    #coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2  # Peak normalisation: peak = 1
    return np.exp(exponent)

def generate_Tsky_proj(full_bool_map, theta_c, phi_c, FWHM=1.1):
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

    def func(ti):
        theta_p = theta[pixel_indices]
        phi_p = phi[pixel_indices]
        cos_sep = np.cos(theta_p) * np.cos(theta_c[ti]) + np.sin(theta_p) * np.sin(theta_c[ti]) * np.cos(phi_p - phi_c[ti])
        cos_sep = np.clip(cos_sep, -1, 1)  # Ensure within valid range
        angular_sep = np.arccos(cos_sep)  # Separation in radians
        # Compute Gaussian beam response centered at (RA_center, Dec_center)
        beam_map = gaussian(angular_sep, sigma=sigma_rad)
        return beam_map
    Tsky_proj = np.array( mpiutil.local_parallel_func(func, np.arange(ntime)) )
    return Tsky_proj






class TODSimulation:
    """
    A class to encapsulate Time-Ordered Data (TOD) simulation parameters and results.
    
    This class handles the simulation of radio telescope observations including:
    - Sky model and beam patterns
    - Gain variations
    - Noise characteristics
    - System temperature components
    """
    
    def __init__(self, nside=64, elevation=41.5, az_s=-60.3, az_e=-42.3, 
                 start_time_utc="2019-04-23 20:41:56.397", FWHM=1.1, 
                 threshold=0.0111, freq=750, T_ndiode=15.0, 
                 rec_params=None, gain_params=None, dtime=2, 
                 logf0=-4.874571109426952, 
                 alpha=2.0, sigma_2=None
                 ):
        """
        Initialize the TOD simulation with given parameters.
        
        Parameters:
        -----------
        nside : int
            HEALPix nside parameter
        elevation : float
            Telescope elevation in degrees
        az_s, az_e : float
            Start and end azimuth angles in degrees
        start_time_utc : str
            UTC start time for the observation
        FWHM : float
            Full Width Half Maximum of the beam in degrees
        threshold : float
            Threshold for beam cutoff (3-sigma region)
        freq : float
            Observation frequency in MHz
        T_ndiode : float
            Noise diode temperature in K
        rec_params : array-like
            Receiver temperature parameters
        gain_params : array-like
            Gain parameters
        dtime : float
            Time sampling interval
        logf0 : float
            Log10 of knee frequency
        alpha : float
            Flicker noise spectral index
        sigma_2 : float
            White noise variance
        n_cal_pixs : int
            Number of calibration pixels
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
        
        # Set default parameters if not provided
        if rec_params is None:
            self.rec_params = np.array([12.6, 0.5, 0.5, 0.5])
        else:
            self.rec_params = np.array(rec_params)
            
        if gain_params is None:
            self.gain_params_setting = np.array([6.31194264, 0.42038942, 0.264222, 0.05578821])
        else:
            self.gain_params_setting = np.array(gain_params)
            
        if sigma_2 is None:
            self.sigma_2 = 1/(4e5)
        else:
            self.sigma_2 = sigma_2
            
        # Run the simulation
        self._simulate()
        
    def _simulate(self):
        """Run the complete simulation and populate all attributes."""
        
        # Get the timestream of beam centers
        self.t_list, self.theta_c_setting, self.phi_c_setting = sim_MeerKAT_scan(
            elevation=self.elevation, az_s=self.az_s, az_e=self.az_e, 
            start_time_utc=self.start_time_utc
        )
        
        # Generate beam map
        self.bool_map_setting, self.integrated_beam_setting = stacked_beam_map(
            self.theta_c_setting, self.phi_c_setting, 
            FWHM=self.FWHM, NSIDE=self.nside, threshold=self.threshold
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
        self.fc = (1/self.ntime/self.dtime)*2*np.pi
        self.logfc = np.log10(self.fc)
        self.f0 = 10**self.logf0
        
        # Generate noise
        self.noise_setting = sim_noise(
            self.f0, self.fc, self.alpha, self.t_list, white_n_variance=self.sigma_2
        )
        
        # Calculate system temperature and final TOD
        self.Tsys_setting = (self.Tsky_operator_setting @ self.sky_params + 
                            self.nd_rec_operator @ self.nd_rec_params)
        self.TOD_setting = self.Tsys_setting * (1 + self.noise_setting) * self.gains_setting
        
        # Set up calibration indices
        top_200_beam_indices = np.argpartition(
            self.integrated_beam_setting[self.pixel_indices], -200
        )[-200:]
        self.calibration_5_indices = [
            top_200_beam_indices[int(i * 200 / 5)] 
            for i in range(5)
        ]

        # For a 1D array 'integrated_beam_setting'
        n_cal_pixs=1
        top_20_beam_indices = np.argpartition(self.integrated_beam_setting[self.pixel_indices], -20)[-20:]
        top_n_sky_indices = np.argpartition(self.sky_params[top_20_beam_indices], -n_cal_pixs)[-n_cal_pixs:]
        self.calibration_1_indice = top_20_beam_indices[top_n_sky_indices]
        
    def _sky_vector(self, pixel_indices, freq, Nside=64, sky_model=None):
        """Generate sky temperature vector for given pixels and frequency."""
        if sky_model is None:
            from pygdsm import GlobalSkyModel
            gsm = GlobalSkyModel()
            skymap = gsm.generate(freq)
        else:
            skymap = sky_model(freq)
            
        skymap = hp.ud_grade(skymap, nside_out=Nside)
        ptsrc = np.load("gleam_nside512_K_allsky_408MHz.npy") * (freq/408) ** (-2.3)
        ptsrc_map = hp.ud_grade(ptsrc, nside_out=Nside)
        skymap = skymap + ptsrc_map
        return skymap[pixel_indices]
    
    def _generate_vector(self, ntime):
        """Generate a vector with 1s every 10 elements, 0s elsewhere."""
        vector = np.zeros(ntime)
        for i in range(0, ntime, 10):
            vector[i] = 1
        return vector
    
    def get_simulation_summary(self):
        """Return a summary of simulation parameters and results."""
        summary = {
            'nside': self.nside,
            'n_pixels': len(self.pixel_indices),
            'n_time_samples': self.ntime,
            'frequency_MHz': self.freq,
            'beam_FWHM_deg': self.FWHM,
            'elevation_deg': self.elevation,
            'azimuth_range_deg': (self.az_s, self.az_e),
            'gain_params': self.gain_params_setting,
            'noise_knee_freq_Hz': self.f0,
            'noise_spectral_index': self.alpha,
            'T_ndiode_K': self.T_ndiode,
            'receiver_params_K': self.rec_params
        }
        return summary


class MultiTODSimulation:
    """
    A class to encapsulate multiple Time-Ordered Data (TOD) simulations.
    
    This class handles the simulation of multiple radio telescope observations including:
    - Multiple scans (setting and rising)
    - Sky model and beam patterns
    - Gain variations for each scan
    - Noise characteristics
    - System temperature components
    - Calibration pixel selection
    """
    
    def __init__(self, nside=64, FWHM=1.1, threshold=0.0111, freq=750, 
                 T_ndiode=15.0, rec_params=None, dtime=2, alpha=2.0, 
                 logf0=-4.874571109426952, sigma_2=None,
                 # Setting scan parameters
                 setting_elevation=41.5, setting_az_s=-60.3, setting_az_e=-42.3,
                 setting_start_time="2019-04-23 20:41:56.397",
                 setting_gain_params=None,
                 # Rising scan parameters  
                 rising_elevation=40.5, rising_az_s=43.7, rising_az_e=61.7,
                 rising_start_time="2019-03-30 17:19:02.397",
                 rising_gain_params=None):
        """
        Initialize the multi-TOD simulation with given parameters.
        
        Parameters:
        -----------
        nside : int
            HEALPix nside parameter
        FWHM : float
            Full Width Half Maximum of the beam in degrees
        threshold : float
            Threshold for beam cutoff (3-sigma region)
        freq : float
            Observation frequency in MHz
        T_ndiode : float
            Noise diode temperature in K
        rec_params : array-like
            Receiver temperature parameters
        dtime : float
            Time sampling interval
        alpha : float
            Flicker noise spectral index
        logf0 : float
            Log10 of knee frequency
        sigma_2 : float
            White noise variance
        setting_* : various
            Parameters for the setting scan
        rising_* : various
            Parameters for the rising scan
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
            self.gain_params_setting = np.array([6.31194264, 0.42038942, 0.264222, 0.05578821])
        else:
            self.gain_params_setting = np.array(setting_gain_params)
            
        if rising_gain_params is None:
            self.gain_params_rising = np.array([6.84507868, 0.14156859, 0.7441104, 0.77863955])
        else:
            self.gain_params_rising = np.array(rising_gain_params)
            
        if sigma_2 is None:
            self.sigma_2 = 1/(4e5)
        else:
            self.sigma_2 = sigma_2
            
        # Run the simulation
        self._simulate()
        
    def _simulate(self):
        """Run the complete simulation and populate all attributes."""
        
        # Generate setting scan
        _, self.theta_c_setting, self.phi_c_setting = sim_MeerKAT_scan(
            elevation=self.setting_elevation, az_s=self.setting_az_s, 
            az_e=self.setting_az_e, start_time_utc=self.setting_start_time
        )
        self.bool_map_setting, self.integrated_beam_setting = stacked_beam_map(
            self.theta_c_setting, self.phi_c_setting, 
            FWHM=self.FWHM, NSIDE=self.nside, threshold=self.threshold
        )
        
        # Generate rising scan
        self.t_list, self.theta_c_rising, self.phi_c_rising = sim_MeerKAT_scan(
            elevation=self.rising_elevation, az_s=self.rising_az_s, 
            az_e=self.rising_az_e, start_time_utc=self.rising_start_time
        )
        self.bool_map_rising, self.integrated_beam_rising = stacked_beam_map(
            self.theta_c_rising, self.phi_c_rising, 
            FWHM=self.FWHM, NSIDE=self.nside, threshold=self.threshold
        )
        
        # Combine maps using logical OR
        self.full_bool_map, self.pixel_indices = reduce_bool_maps_LOR([
            self.bool_map_setting, self.bool_map_rising
        ])
        self.integrated_beam = self.integrated_beam_setting + self.integrated_beam_rising
        
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
        self.fc = (1/self.ntime/self.dtime)*2*np.pi
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
        self.Tsys_setting = (self.Tsky_operator_setting @ self.sky_params + 
                            self.nd_rec_operator @ self.nd_rec_params)
        self.TOD_setting = self.Tsys_setting * (1 + self.noise_setting) * self.gains_setting
        
        self.Tsys_rising = (self.Tsky_operator_rising @ self.sky_params + 
                           self.nd_rec_operator @ self.nd_rec_params)
        self.TOD_rising = self.Tsys_rising * (1 + self.noise_rising) * self.gains_rising
        
        # Generate additional maps for analysis
        self._generate_analysis_maps()
        
        # Set up calibration indices
        self._setup_calibration_indices()
        
    def _sky_vector(self, pixel_indices, freq, Nside=64, sky_model=None):
        """Generate sky temperature vector for given pixels and frequency."""
        if sky_model is None:
            from pygdsm import GlobalSkyModel
            gsm = GlobalSkyModel()
            skymap = gsm.generate(freq)
        else:
            skymap = sky_model(freq)
            
        skymap = hp.ud_grade(skymap, nside_out=Nside)
        ptsrc = np.load("gleam_nside512_K_allsky_408MHz.npy") * (freq/408) ** (-2.3)
        ptsrc_map = hp.ud_grade(ptsrc, nside_out=Nside)
        skymap = skymap + ptsrc_map
        return skymap[pixel_indices]
    
    def _generate_vector(self, ntime):
        """Generate a vector with 1s every 10 elements, 0s elsewhere."""
        vector = np.zeros(ntime)
        for i in range(0, ntime, 10):
            vector[i] = 1
        return vector
    
    def _generate_analysis_maps(self):
        """Generate additional maps for analysis."""
        # Convert theta/phi coordinates to HEALPix pixels for setting scan
        self.pixels_c_setting = [hp.ang2pix(nside=self.nside, theta=theta, phi=phi)
                                for theta, phi in zip(self.theta_c_setting, self.phi_c_setting)]
        self.bool_map_c_setting = np.zeros(hp.nside2npix(self.nside))
        self.bool_map_c_setting[self.pixels_c_setting] = 1
        
    def _setup_calibration_indices(self):
        """Set up calibration pixel indices."""
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
            top_200_beam_indices[int(i * 200 / n_cal_pixs)] 
            for i in range(n_cal_pixs)
        ]
    
    def get_simulation_summary(self):
        """Return a summary of simulation parameters and results."""
        summary = {
            'nside': self.nside,
            'n_pixels': len(self.pixel_indices),
            'n_time_samples': self.ntime,
            'frequency_MHz': self.freq,
            'beam_FWHM_deg': self.FWHM,
            'setting_scan': {
                'elevation_deg': self.setting_elevation,
                'azimuth_range_deg': (self.setting_az_s, self.setting_az_e),
                'gain_params': self.gain_params_setting
            },
            'rising_scan': {
                'elevation_deg': self.rising_elevation,
                'azimuth_range_deg': (self.rising_az_s, self.rising_az_e),
                'gain_params': self.gain_params_rising
            },
            'noise_knee_freq_Hz': self.f0,
            'noise_spectral_index': self.alpha,
            'T_ndiode_K': self.T_ndiode,
            'receiver_params_K': self.rec_params,
            'calibration_pixels': {
                'single_cal_index': self.calibration_1_index,
                'five_cal_indices': self.calibration_5_indices
            }
        }
        return summary
    
    def get_tod_data(self):
        """Return the TOD data for both scans in a format suitable for analysis."""
        return {
            'TOD_setting': self.TOD_setting,
            'TOD_rising': self.TOD_rising,
            't_list': self.t_list,
            'gain_proj': self.gain_proj,
            'Tsky_operators': [self.Tsky_operator_setting, self.Tsky_operator_rising],
            'nd_rec_operator': self.nd_rec_operator,
            'logfc': self.logfc
        }



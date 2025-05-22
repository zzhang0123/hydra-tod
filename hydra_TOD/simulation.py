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
                     dt=2.0):

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


def sky_vector(pixel_indices, freq, Nside=64, sky_model=None):
    if sky_model is None:
        from pygdsm import GlobalSkyModel
        gsm = GlobalSkyModel()
        skymap = gsm.generate(500)
    else:
        skymap = sky_model(freq)
    skymap = hp.ud_grade(skymap, nside_out=Nside)
    return skymap[pixel_indices]






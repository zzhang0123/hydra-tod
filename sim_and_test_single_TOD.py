import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pygdsm import GlobalSkyModel


from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time, TimeDelta
import astropy.units as u
from utils import Leg_poly_proj
from flicker_model import sim_noise

from Tsys_sampler import Tsys_model, overall_operator
from full_Gibbs_sampler import full_Gibbs_sampler_singledish 


# Antenna position: Latitude: -30.7130° S; Longitude: 21.4430° E.

# ---- Define telescope location ----
telescope_lat = -30.7130  # MeerKAT latitude in degrees
telescope_lon = 21.4430   # MeerKAT longitude in degrees
telescope_height = 1054    # MeerKAT altitude in meters

location = EarthLocation(lat=telescope_lat * u.deg, lon=telescope_lon * u.deg, height=telescope_height * u.m)


# ---- Define observation parameters ----

# Antenna pointings: Azimuth list and Elevation, in degrees
azimuths = np.linspace(-60, -40, 111)
azimuths = np.concatenate((azimuths, azimuths[1:-1][::-1]))

# Generate 13 repeats of the azimuths
azimuths = np.tile(azimuths, 15)

elevation = 41.7    # Elevation in degrees

# Total length of TOD
ntime = len(azimuths)
print("Total length of TOD: ", ntime)
dtime = 2.0

t_list = np.arange(ntime) * dtime 

# ---- Define start and end times in SAST ----
start_time_sast = "2024-02-23 19:54:07.397"
#end_time_sast = "2024-02-23 22:12:04.632"

# ---- Convert to UTC (SAST = UTC+2) ----
start_time = Time(start_time_sast) - TimeDelta(2 * u.hour)
#end_time = Time(end_time_sast) - TimeDelta(2 * u.hour)

# ---- Generate time list using numpy.arange ----
dt = 2  # Time step in seconds
time_list = start_time + TimeDelta(t_list, format='sec')

# ---- Create AltAz coordinate frame ----
altaz_frame = AltAz(obstime=time_list, location=location)

# ---- Convert Az/El to Equatorial (RA, Dec) ----
horizon_coords = SkyCoord(az=azimuths*u.deg, alt=elevation*u.deg, frame=altaz_frame)
equatorial_coords = horizon_coords.transform_to("icrs")

# Convert the equatorial coordinates to pixel indices
# Note: healpy expects (theta, phi) in spherical coordinates
theta_c = np.pi/2 - equatorial_coords.dec.radian  # Convert Dec to theta
phi_c = equatorial_coords.ra.radian               # RA is already phi

# Define beam parameters
FWHM = 1.1  # Full Width at Half Maximum in degrees
sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma (degrees)
sigma_rad = np.radians(sigma)  # Convert to radians

def pixel_angular_size(nside):
    """Compute the angular size (in degrees and arcminutes) of a HEALPix pixel."""
    npix = hp.nside2npix(nside)  # Total number of pixels
    omega_pix = 4 * np.pi / npix  # Pixel area in steradians
    theta_pix_deg = np.sqrt(omega_pix) * (180 / np.pi)  # Approximate pixel width in degrees
    theta_pix_arcmin = theta_pix_deg * 60  # Convert to arcminutes
    return theta_pix_deg, theta_pix_arcmin

# Example usage
nside = 64 # Change NSIDE as needed
theta_deg, theta_arcmin = pixel_angular_size(nside)

# Define HEALPix resolution
NSIDE = 64
NPIX = hp.nside2npix(NSIDE)  

# Get HEALPix pixel coordinates (theta, phi)
theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))

# Generate a initial boolean map with all pixels zero
bool_map = np.zeros(NPIX, dtype=bool)
sum_map = np.zeros(NPIX, dtype=float)
# ---- Set the threshold ----
threshold = 1e-1  # Example: Get all pixels where the value is > 0.5

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

# Count the number of "1" pixels in bool_map
num_pixels = np.sum(bool_map)
print(f"Number of covered pixels: {num_pixels}")

# Get the pixel indices of the "1" pixels:
pixel_indices = np.where(bool_map)[0]

# Save HEALPix map to file
hp.write_map("gaussian_beam_pointing.fits", beam_map, overwrite=True)

# Get pixels of skymap where corresponding mask value (bool_map) is true 

beam_proj = np.zeros((ntime, num_pixels))

for ti in range(ntime):
    # Compute angular separation between each pixel and the beam center
    cos_sep = np.cos(theta) * np.cos(theta_c[ti]) + np.sin(theta) * np.sin(theta_c[ti]) * np.cos(phi - phi_c[ti])
    cos_sep = np.clip(cos_sep, -1, 1)  # Ensure within valid range
    angular_sep = np.arccos(cos_sep)  # Separation in radians
    # Compute Gaussian beam response centered at (RA_center, Dec_center)
    beam_map = np.exp(-0.5 * (angular_sep / sigma_rad) ** 2)
    # Normalize the beam (optional, ensures peak = 1)
    beam_map /= np.max(beam_map)
    beam_proj[ti] = beam_map[pixel_indices]

norm=np.sum(beam_proj, axis=1)
beam_proj/=norm[:,None]

gsm = GlobalSkyModel()
gsm.nside =NSIDE
skymap = gsm.generate(500)
true_Tsky = skymap[pixel_indices]

# generate a vector of length ntime, every 10 elements there is a 1, the rest is 0
def generate_vector(ntime):
    vector = np.zeros(ntime)
    for i in range(0, ntime, 10):
        vector[i] = 1
    return vector

ndiode_proj = generate_vector(ntime)

T_ndiode = 10.0

rec_proj = Leg_poly_proj(4, t_list)[:,1:]
rec_params=np.array([1, 0.5, 1])

gain_proj = Leg_poly_proj(4, t_list)
gain_params=np.array([2, 0.5, 1.5, 0.5])*2
mu0 = np.sin(2*np.pi*0.1*t_list)
gains = gain_proj @ gain_params + mu0

f0, fc, alpha = 1e-4, 2e-5, 2.0
sigma_2 = 1/(4e5)

noise = sim_noise(f0, fc, alpha, t_list, n_samples=1, white_n_variance=sigma_2)[0]

TOD_ndiode = T_ndiode*ndiode_proj
TOD_rec = rec_proj @ rec_params
Tsys_sim = Tsys_model([beam_proj, rec_proj, ndiode_proj], [true_Tsky, rec_params, T_ndiode])
TOD_sim = Tsys_sim * (1+noise) * gains 
Tsys_proj = overall_operator([beam_proj, rec_proj])

logn_params = [np.log10(f0), np.log10(fc), alpha]

p_gain_samples, p_sys_samples, p_noise_samples = full_Gibbs_sampler_singledish(TOD_sim, 
                                  t_list,
                                  TOD_ndiode,
                                  Tsys_proj,
                                  gain_proj,
                                  np.hstack((true_Tsky, rec_params)), 
                                  logn_params, 
                                  gain_mu0=mu0,
                                  wnoise_var=2.5e-6,
                                  Tsys_prior_cov_inv=None,
                                  Tsys_prior_mean=None,
                                  gain_prior_cov_inv=None,
                                  gain_prior_mean=None,
                                  noise_prior_func=None,
                                  n_samples=50,
                                  tol=1e-15,
                                  linear_solver=None,)


# Save 500 samples of gain, Tsys, and noise

np.save('gain_samples.npy', np.array(p_gain_samples))
np.save('Tsys_samples.npy', np.array(p_sys_samples))
np.save('noise_samples.npy', np.array(p_noise_samples))
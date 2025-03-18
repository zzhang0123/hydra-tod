import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time, TimeDelta
import astropy.units as u
from utils import Leg_poly_proj, view_samples
from flicker_model import sim_noise, flicker_cov

import mpiutil

from mpi4py import MPI
from joblib import Parallel, delayed

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
rank0 = rank == 0

def example_scan(n_ele, location, delta_elevation=1.0, start_time_sast = "2024-02-23 19:54:07.397", dt=2.0, njobs=mpiutil.cpu_affinity):
    elevation_list = np.arange(n_ele)*delta_elevation + 40.0
    elevation_list = np.repeat(elevation_list, 2) # Repeat each elevation twice
    local_ele_list = mpiutil.partition_list_mpi(elevation_list, method="con")
    n_local_TOD = len(local_ele_list)

    aux = np.linspace(-65, -35, 167)
    azimuths_a = np.concatenate((aux[1:-1][::-1], aux))
    azimuths_b = np.concatenate((aux, aux[1:-1][::-1]))
    # Generate a number of repeats of the azimuths
    azimuths_a = np.tile(azimuths_a, 25)
    azimuths_b = np.tile(azimuths_b, 25)
    azimuths_list = [azimuths_a, azimuths_b]*n_ele
    local_az_list = mpiutil.partition_list_mpi(azimuths_list, method="con")
    assert len(local_az_list) == n_local_TOD

    # Length of TOD
    ntime = len(azimuths_a)
    t_list = np.arange(ntime) * dt

    # ---- Convert to UTC (SAST = UTC+2) ----
    start_time = Time(start_time_sast) - TimeDelta(2 * u.hour)
    # ---- Generate time list using numpy.arange ----
    time_list = start_time + TimeDelta(t_list, format='sec') # Time list in UTC
    # ---- Create AltAz coordinate frame ----
    altaz_frame = AltAz(obstime=time_list, location=location)
    # ---- Convert Az/El to Equatorial (RA, Dec) ----
    def func_az_el_to_eq(i):
        return SkyCoord(az=local_az_list[i]*u.deg, alt=local_ele_list[i]*u.deg, frame=altaz_frame).transform_to("icrs")
    num_jobs = np.min([njobs, len(local_az_list)])
    local_eq_coords_list = Parallel(n_jobs=num_jobs)(delayed(func_az_el_to_eq)(i) \
                                                  for i in range(len(local_az_list)) )
    # Convert the equatorial coordinates to pixel indices
    # Note: healpy expects (theta, phi) in spherical coordinates
    local_theta_c_list = [np.pi/2 - equatorial_coords.dec.radian for equatorial_coords in local_eq_coords_list]
    local_phi_c_list = [equatorial_coords.ra.radian for equatorial_coords in local_eq_coords_list] # RA is already phi

    return [t_list.copy() for _ in range(n_local_TOD)], local_theta_c_list, local_phi_c_list


def example_beam(local_theta_c_list, local_phi_c_list, 
                FWHM=1.1, NSIDE=64, threshold = 5e-2, root=None, njobs=mpiutil.cpu_affinity):
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma (degrees)
    sigma_rad = np.radians(sigma)  # Convert to radians

    NPIX = hp.nside2npix(NSIDE)  

    # Get HEALPix pixel coordinates (theta, phi)
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))

    # Generate a initial boolean map with all pixels zero
    bool_map = np.zeros(NPIX, dtype=bool)
    sum_map = np.zeros(NPIX, dtype=float)

    n_chunks = len(local_theta_c_list)
    ntime = len(local_theta_c_list[0])

    for ci in range(n_chunks):
        theta_c = local_theta_c_list[ci]
        phi_c = local_phi_c_list[ci]
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

    full_bool_map = np.zeros(NPIX, dtype=bool)  # Initialize a full map
    # Gather the boolean maps from all processes
    if root is None:
        comm.Allreduce(bool_map, full_bool_map, op=MPI.LOR)
    else:
        comm.Reduce(bool_map, full_bool_map, op=MPI.LOR, root=root)

    # Count the number of "1" pixels in bool_map
    num_pixels = np.sum(bool_map)
    # Get the pixel indices of the "1" pixels:
    pixel_indices = np.where(bool_map)[0]

    local_Tsky_proj_list = []
    for ci in range(n_chunks):
        theta_c = local_theta_c_list[ci]
        phi_c = local_phi_c_list[ci]
        Tsky_proj = np.zeros((ntime, num_pixels))
        def func(ti):
            cos_sep = np.cos(theta) * np.cos(theta_c[ti]) + np.sin(theta) * np.sin(theta_c[ti]) * np.cos(phi - phi_c[ti])
            cos_sep = np.clip(cos_sep, -1, 1)  # Ensure within valid range
            angular_sep = np.arccos(cos_sep)  # Separation in radians
            # Compute Gaussian beam response centered at (RA_center, Dec_center)
            beam_map = np.exp(-0.5 * (angular_sep / sigma_rad) ** 2)
            # Normalize the beam (optional, ensures peak = 1)
            beam_map /= np.max(beam_map)
            return beam_map[pixel_indices]
        Tsys_proj = np.array(Parallel(n_jobs=njobs)(delayed(func)(i) for i in range(ntime)))
        # norm=np.sum(beam_proj, axis=1)
        # beam_proj/=norm[:,None]
        local_Tsky_proj_list.append(Tsky_proj)  # Append the beam projection to the list
    return local_Tsky_proj_list, pixel_indices, full_bool_map, sum_map


class TOD_sim():
    def __init__(self, T_ndiode=10, ant_coords=[-30.7130, 21.4430, 1054.0]):
        self.ant_lat = ant_coords[0] # Telescope latitude (deg)
        self.ant_lon = ant_coords[1] # Telescope longitude (deg)
        self.ant_hei = ant_coords[2] # Telescope height (m)
        self.T_ndiode = T_ndiode 
        self.location = EarthLocation(lat=self.ant_lon*u.deg, lon=self.ant_lat*u.deg, height=self.ant_hei*u.m)

    def generate_T_ndiode(self, ntime, T_nd, T_rec_mean=3.):
        TOD_ndiode = np.zeros(ntime)
        for i in range(0, ntime, 10):
            TOD_ndiode[i] = T_nd
        return TOD_ndiode + T_rec_mean

    def generate(self, n_elevation, rec_params_list, gain_params_list, noise_params_list, Tmap, beam_cutoff=5.e-2, sigma_2=1./(4e5)):
        self.local_t_list, local_theta_c_list, local_phi_c_list = example_scan(n_elevation, self.location)
        self.n_chunks = len(local_theta_c_list)

        # Get the NSIDE of the map
        self.nside = hp.get_nside(Tmap)
        self.local_Tsky_proj_list, self.pixel_indices, self.full_bool_map, self.full_sum_map = example_beam(local_theta_c_list, local_phi_c_list,
        NSIDE=self.nside, threshold = beam_cutoff)

        print("Rank: {}, local sky projector list has been generated.".format(mpiutil.rank))
        
        self.Tsky = Tmap[self.pixel_indices]
        local_TOD_list=[]
        local_rec_proj_list=[]
        local_gain_proj_list=[]
        local_TOD_ndiode_list=[]
        for ci in range(self.n_chunks):
            t_list = self.local_t_list[ci]
            ntime = len(t_list)
            TOD_ndiode = self.generate_T_ndiode(ntime, self.T_ndiode)
            local_TOD_ndiode_list.append(TOD_ndiode)
            rec_proj = Leg_poly_proj(4, t_list)[:, 1:]
            local_rec_proj_list.append(rec_proj)
            gain_proj = Leg_poly_proj(4, t_list)
            local_gain_proj_list.append(gain_proj)

            TOD_Tsys = self.local_Tsky_proj_list[ci] @ self.Tsky + rec_proj@rec_params_list[ci] + TOD_ndiode
            TOD_gain = gain_proj @ gain_params_list[ci]
            logf0, logfc, alpha = noise_params_list[ci]
            noise = sim_noise(10.**logf0, 10.**logfc, alpha, self.t_list, n_samples=1, white_n_variance=sigma_2)[0]
            TOD = TOD_Tsys * TOD_gain * (1 + noise)
            local_TOD_list.append(TOD)
        self.local_TOD_list = local_TOD_list
        self.local_rec_proj_list = local_rec_proj_list
        self.local_gain_proj_list = local_gain_proj_list
        self.local_TOD_ndiode_list = local_TOD_ndiode_list
        print("Rank: {}, local TOD list has been generated.".format(mpiutil.rank))
        return None


def Tsky_proj(ntime, 
            dt, 
            start_time_UTC,
            azimuths, 
            elevation,
            ant_coords=[-30.7130, 21.4430, 1054], 
            beam_FWHM=1.5,
            Nside=128,
           ):
    """"
    Simulate the TOD for a given set of parameters

    ant_coords: [latitude (deg), longitude (deg), height (m)]

    """

    t_list = np.arange(ntime) * dt
    time_list = start_time + TimeDelta(t_list, format='sec')

    telescope_lat, telescope_lon, telescope_height = ant_coords
    location = EarthLocation(lat=telescope_lat * u.deg, lon=telescope_lon * u.deg, height=telescope_height * u.m)

    # ---- Create AltAz coordinate frame ----
    altaz_frame = AltAz(obstime=time_list, location=location)

    # ---- Convert Az/El to Equatorial (RA, Dec) ----
    horizon_coords = SkyCoord(az=azimuths*u.deg, alt=elevation*u.deg, frame=altaz_frame)
    equatorial_coords = horizon_coords.transform_to("icrs")

    # Convert the equatorial coordinates to pixel coordinates
    # Note: healpy expects (theta, phi) in spherical coordinates
    theta_c = np.pi/2 - equatorial_coords.dec.radian  # Convert Dec to theta
    phi_c = equatorial_coords.ra.radian               # RA is already phi

    # Define beam parameters
    sigma = beam_FWHM / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma (degrees)
    sigma_rad = np.radians(sigma)  # Convert to radians

    NPIX = hp.nside2npix(NSIDE)  

    # Get HEALPix pixel coordinates (theta, phi)
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))

    # Generate a initial boolean map with all pixels zero
    bool_map = np.zeros(NPIX, dtype=bool)
    sum_map = np.zeros(NPIX, dtype=float)

    # Set the threshold be 3sigma
    threshold = np.exp(-0.5 * 3 ** 2)

    for ti in range(ntime):
        # Compute angular separation between each pixel and the beam center
        cos_sep = np.cos(theta) * np.cos(theta_c[ti]) + np.sin(theta) * np.sin(theta_c[ti]) * np.cos(phi - phi_c[ti])
        cos_sep = np.clip(cos_sep, -1, 1)  # Ensure within valid range
        angular_sep = np.arccos(cos_sep)  # Separation in radians
        # Compute Gaussian beam response centered at (RA_center, Dec_center)
        beam_map = np.exp(-0.5 * (angular_sep / sigma_rad) ** 2)
        sum_map += beam_map
        # Get the "or" map of the bool_map and beam_map
        bool_map = np.logical_or(bool_map, beam_map > threshold)

    # Get pixels of skymap where corresponding mask value (bool_map) is true 
    # Count the number of "1" pixels in bool_map
    num_pixels = np.sum(bool_map)
    print(f"Number of covered pixels: {num_pixels}")

    # Get the pixel indices of the "1" pixels:
    pixel_indices = np.where(bool_map)[0]

    beam_proj = np.zeros((ntime, num_pixels))

    for ti in range(ntime):
        # Compute angular separation between each pixel and the beam center
        cos_sep = np.cos(theta) * np.cos(theta_c[ti]) + np.sin(theta) * np.sin(theta_c[ti]) * np.cos(phi - phi_c[ti])
        cos_sep = np.clip(cos_sep, -1, 1)  # Ensure within valid range
        angular_sep = np.arccos(cos_sep)  # Separation in radians
        # Compute Gaussian beam response centered at (RA_center, Dec_center)
        beam_map = np.exp(-0.5 * (angular_sep / sigma_rad) ** 2)
        beam_proj[ti] = beam_map[pixel_indices]

    # Normalize the beam
    norm=np.sum(beam_proj, axis=1)
    beam_proj/=norm[:,None]

    return beam_proj, pixel_indices

def Tsky_params(pixel_indices, freq, NSIDE=128):
    gsm = GlobalSkyModel()
    gsm.nside =NSIDE
    skymap = gsm.generate(freq)
    true_Tsky = skymap[pixel_indices]
    return true_Tsky

def Tsky_healpix_map(vals, pixel_indices, NSIDE=128):
    skymap = np.zeros(hp.nside2npix(NSIDE))
    skymap[pixel_indices] = vals
    return skymap



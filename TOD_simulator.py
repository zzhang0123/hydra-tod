


# ---- Define start and end times in SAST ----
start_time_sast = "2024-02-23 19:54:07.397"
# ---- Convert to UTC (SAST = UTC+2) ----
start_time = Time(start_time_sast) - TimeDelta(2 * u.hour)

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

    t_list = np.arange(ntime) * dtime 
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

# generate a vector of length ntime, every 10 elements there is a 1, the rest is 0
def generate_vector(ntime, nd_period=10):
    vector = np.zeros(ntime)
    for i in range(0, ntime, nd_period):
        vector[i] = 1
    return vector

ndiode_proj = generate_vector(ntime)

T_ndiode = 10.0
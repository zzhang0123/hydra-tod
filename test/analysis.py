import sys
sys.path.append('../hydra_tod/')

import numpy as np
import healpy as hp

import pickle
from simulation import TODSimulation, MultiTODSimulation

# Load the simulation data from a pickle file
with open('tod_simulation_single.pkl', 'rb') as f:
    tod_sim = pickle.load(f)
with open('multi_tod_simulation_data.pkl', 'rb') as f:
    multi_tod_sim = pickle.load(f)


def intersect_with_indices(arr1, arr2):
    """
    Returns the intersection of arr1 and arr2, along with the indices in arr1 and arr2.
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    intersection, idx_arr1, idx_arr2 = np.intersect1d(arr1, arr2, return_indices=True)
    return intersection, idx_arr1, idx_arr2

def get_subset_indices(pixel_indices, pixel_indices_setting):
    """
    Returns the indices in pixel_indices corresponding to each value in pixel_indices_setting.
    Assumes all elements of pixel_indices_setting are present in pixel_indices.
    """
    pixel_indices = np.asarray(pixel_indices)
    pixel_indices_setting = np.asarray(pixel_indices_setting)
    # Create a mapping from value to index for fast lookup
    value_to_index = {val: idx for idx, val in enumerate(pixel_indices)}
    # Get the indices for each value in pixel_indices_setting
    indices = [value_to_index[val] for val in pixel_indices_setting]
    return np.array(indices)

def calculate_coverage_maps(samples, true_sky):

    """Calculate coverage violation maps"""
    
    # Calculate percentiles
    p16 = np.percentile(samples, 16, axis=0)
    p84 = np.percentile(samples, 84, axis=0)
    ci_68_width = p84 - p16
    p2p5 = np.percentile(samples, 2.5, axis=0)
    p97p5 = np.percentile(samples, 97.5, axis=0)
    median = np.median(samples, axis=0)
    residual = median - true_sky
    
    # Coverage checks
    within_68 = (true_sky >= p16) & (true_sky <= p84)
    within_95 = (true_sky >= p2p5) & (true_sky <= p97p5)
    
    # Create violation maps with different levels
    # 0 = within 68%, 1 = outside 68% but within 90%, 2 = outside 90% but within 95%, 3 = outside 95%
    coverage_map = np.zeros_like(true_sky)
    coverage_map[~within_68] = 1  # Outside 68%
    coverage_map[~within_95] = 2  # Outside 95%

    return median, residual, ci_68_width, coverage_map

def common_pixel_indices():
    """Find common pixel indices between setting and rising scans."""
    pixel_indices_setting = np.where(multi_tod_sim.bool_map_setting)[0]
    pixel_indices_rising = np.where(multi_tod_sim.bool_map_rising)[0]
    common_pix, _, _ = intersect_with_indices(pixel_indices_setting, pixel_indices_rising)
    print(len(common_pix))
    inds_common_pix_in_setting = get_subset_indices(pixel_indices_setting, common_pix)
    inds_common_pix_in_db = get_subset_indices(multi_tod_sim.pixel_indices, common_pix)
    return pixel_indices_setting, multi_tod_sim.pixel_indices, inds_common_pix_in_setting, inds_common_pix_in_db

def true_sky_params():
    return tod_sim.sky_params, multi_tod_sim.sky_params

param_names_compact = [
    r'$p_{{\rm g},0}$', r'$p_{{\rm g},1}$', r'$p_{{\rm g},2}$', r'$p_{{\rm g},3}$',  # 4 gain parameters
    r'$p_{{\rm loc},0}$', r'$p_{{\rm loc},1}$', r'$p_{{\rm loc},2}$', r'$p_{{\rm loc},3}$', r'$p_{{\rm loc},4}$',  # 5 receiver parameters
    r'$\log_{10} f_0$', r'$\alpha$'  # 2 noise parameters
]

param_names_compact_2 = [
 '$p^{(0)}_{{\\rm g},0}$',
 '$p^{(0)}_{{\\rm g},1}$',
 '$p^{(0)}_{{\\rm g},2}$',
 '$p^{(0)}_{{\\rm g},3}$',
 '$p^{(1)}_{{\\rm g},0}$',
 '$p^{(1)}_{{\\rm g},1}$',
 '$p^{(1)}_{{\\rm g},2}$',
 '$p^{(1)}_{{\\rm g},3}$',
 '$p^{(0)}_{{\\rm loc},0}$',
 '$p^{(0)}_{{\\rm loc},1}$',
 '$p^{(0)}_{{\\rm loc},2}$',
 '$p^{(0)}_{{\\rm loc},3}$',
 '$p^{(0)}_{{\\rm loc},4}$',
 '$p^{(1)}_{{\\rm loc},0}$',
 '$p^{(1)}_{{\\rm loc},1}$',
 '$p^{(1)}_{{\\rm loc},2}$',
 '$p^{(1)}_{{\\rm loc},3}$',
 '$p^{(1)}_{{\\rm loc},4}$',
 '$\\log_{10} f_0^{(0)}$',
 '$\\alpha^{(0)}$',
 '$\\log_{10} f_0^{(1)}$',
 '$\\alpha^{(1)}$']

pixel_indices_setting, pixel_indices_db, inds_common_pix_in_setting, inds_common_pix_in_db = common_pixel_indices()

sky_params_setting, sky_params_db = true_sky_params()

theta, phi = hp.pix2ang(64, pixel_indices_setting)
lon, lat = np.degrees(phi), 90 - np.degrees(theta)
lon_center, lat_center = np.median(lon), np.median(lat)

# Calculate appropriate zoom/resolution
patch_size = 100  # Add 20% margin
res = patch_size / 20  # Adjust resolution based on patch size

def explore_posterior_joint_Tsys(Tsys_samples, gain_samples, noise_samples, 
                                 true_sky,
                                 title_str,
                                 save_path,
                                 two_x_TODs=True, 
                                 warm_up=0, 
                                 cbar=False
                                 ):
    assert Tsys_samples.ndim == 2
    assert gain_samples.ndim == 3
    assert noise_samples.ndim == 3

    if two_x_TODs:
        pixel_indices = pixel_indices_db
        inds_common_pix = inds_common_pix_in_db
        true_sky = sky_params_db
    else:
        pixel_indices = pixel_indices_setting
        inds_common_pix = inds_common_pix_in_setting
        true_sky = sky_params_setting

    n_loc = 10 if two_x_TODs else 5
    Tsky_s = Tsys_samples[warm_up:, :-n_loc]
    Tsky_mean = np.mean(Tsky_s, axis=0)
    Tsky_std = np.std(Tsky_s, axis=0)
    Tsky_mean_res = Tsky_mean - true_sky
    Tsky_zscore = Tsky_mean_res / Tsky_std
    Tsky_mean_res_comm = Tsky_mean_res[inds_common_pix]
    Tsky_med, Tsky_med_res, Tsky_ci_68, Tsky_coverage = calculate_coverage_maps(Tsky_s, true_sky)
    Tsky_med_res_comm = Tsky_med_res[inds_common_pix]
    Tsky_zscore_med = Tsky_med_res / Tsky_std
    Dict = {
        "mean": (Tsky_mean, Tsky_mean_res, Tsky_mean_res_comm),
        "std": Tsky_std,
        "median": (Tsky_med, Tsky_med_res, Tsky_med_res_comm),
        "ci_68": Tsky_ci_68,
        "coverage": Tsky_coverage,
        "zscore": np.abs(Tsky_mean_res / Tsky_std),
        "bias": np.abs(Tsky_mean_res)
    }

    from visualisation import plot_residual_histogram, gnomview_patch


    append_1 =  "_mean_res.pdf"
    append_2 =  "_median_res.pdf"

    plot_residual_histogram(
        title_str,
        Tsky_mean_res,
        Tsky_mean_res_comm,
        save_path= save_path + append_1,
        kde=False,
        figsize=(14, 9),
        xlim=[-1.1, 1.1]
    )
    plot_residual_histogram(
        title_str,
        Tsky_med_res,
        Tsky_med_res_comm,
        save_path= save_path + append_2,
        kde=False,
        figsize=(14, 9),
        xlim=[-1.1, 1.1]
    )


    gnomview_patch(Tsky_mean,
               pixel_indices,
               lon_center,
               lat_center,
               res,
               5, 
               14,
               cbar=cbar,
            #    xtick=False,
            #    xlabel=None,
            #    ytick=True,
               save_path=save_path + "_mean_map.pdf"
               )
    
    gnomview_patch(Tsky_std, 
               pixel_indices, 
               lon_center, 
               lat_center,
               res,
               0, 
               0.2,
               cbar=cbar,
               xtick=False,
               xlabel=None,
               ytick=False,
               ylabel=None,
               save_path=save_path + "_std_map.pdf"
               )
    
    gnomview_patch(Tsky_mean_res, 
               pixel_indices,
               lon_center, 
               lat_center,
               res,
               -0.2, 
               0.2,
               cmap='RdBu_r',
               cbar=cbar,
               xtick=False,
               xlabel=None,
               ytick=False,
               ylabel=None,
               save_path=save_path + "_err_map.pdf"
               )

    gnomview_patch(Tsky_zscore, 
               pixel_indices, 
               lon_center, 
               lat_center,
               res,
               -3.0, 
               3.0,
               unit='',
               cmap='RdBu_r',
               cbar=cbar,
               xtick=False,
               xlabel=None,
               ytick=False,
               ylabel=None,
               save_path=save_path + "_zscore_map.pdf"
               )
    
    gnomview_patch(Tsky_zscore_med, 
               pixel_indices, 
               lon_center, 
               lat_center,
               res,
               -3.0, 
               3.0,
               unit='',
               cmap='RdBu_r',
               cbar=cbar,
               xtick=False,
               xlabel=None,
               ytick=False,
               ylabel=None,
               save_path=save_path + "_med_zscore_map.pdf"
               )

    gnomview_patch(Tsky_med, 
               pixel_indices, 
               lon_center, 
               lat_center,
               res,
               5, 
               14,
               cbar=cbar,
               save_path=save_path + "_median_map.pdf"
               )
    
    gnomview_patch(Tsky_ci_68, 
               pixel_indices, 
               lon_center, 
               lat_center,
               res,
               0, 
               0.4,
               unit='',
               cbar=cbar,
               xtick=False,
               xlabel=None,
               ytick=False,
               ylabel=None,
               save_path=save_path + "_ci_map.pdf"
               )
    
    gnomview_patch(Tsky_med_res, 
               pixel_indices, 
               lon_center, 
               lat_center,
               res,
               -0.2, 
               0.2,
               cmap='RdBu_r',
               cbar=cbar,
               xtick=False,
               xlabel=None,
               ytick=False,
               ylabel=None,
               save_path=save_path + "_med_err_map.pdf"
               )
    
    gnomview_patch(Tsky_coverage, 
               pixel_indices, 
               lon_center, 
               lat_center,
               res,
               0.0, 
               2.0,
               unit='',
               cmap='RdYlGn_r',
               cbar=cbar,
               xtick=False,
               xlabel=None,
               ytick=False,
               ylabel=None,
               save_path=save_path + "_coverage_map.pdf"
               )
    return Dict


import numpy as np
import matplotlib.pyplot as plt
import pickle, os

from pygdsm import GlobalSkyModel

from flicker_model import flicker_cov

from full_Gibbs_sampler import full_Gibbs_sampler_multi_TODS

import mpiutil

# Antenna position: Latitude: -30.7130° S; Longitude: 21.4430° E.

n_elevation = mpiutil.size # set the number of elevations to be the number of processes
#n_sets = 2*n_elevation # set the number of TOD sets to be twice the number of elevations, then each process can do two TOD sets

# set the random prameters for simulation
local_rec_params_list = [np.random.uniform(low=0.0, high=1.0, size=3) for i in range(2)]
local_gain_params_list = [np.random.uniform(low=0.0, high=1.0, size=4) + np.array([6., 0., 0., 0.]) 
                        for i in range(2)] # add 6 to the first element to make the gain positive


logf0_c, logfc_c, alpha_c =  np.log10(1e-4), np.log10(2e-5), 2.2
center_values = np.array([logf0_c, logfc_c, alpha_c])
local_noise_params_list = [np.random.uniform(low=-0.25, high=0.25, size=3) + center_values for i in range(2)]

# Use a GSM model to generate a sky map
gsm = GlobalSkyModel()
gsm.nside = 64
skymap = gsm.generate(408)

from TOD_simulator import TOD_sim

local_TOD = TOD_sim()
local_TOD.generate(n_elevation, local_rec_params_list, local_gain_params_list, local_noise_params_list, skymap)

# Save the "local_TOD" objects
savepath = "/scratch3/users/zzhang/TOD_simulations"
TOD_savename = "local_TOD_{}.pkl".format(mpiutil.rank)

# combind the savepath and savename
TOD_savepath = os.path.join(savepath, TOD_savename)

with open(TOD_savepath, "wb") as f:
    pickle.dump(local_TOD, f)

init_Tsky_params = local_TOD.Tsky + 0.2
init_Trec_params = np.zeros(3)
init_noise_params = [-4., -5., 2.]

Tsky_samples, all_gain_samples, all_noise_samples, all_rec_samples = \
    full_Gibbs_sampler_multi_TODS(local_TOD.local_TOD_list,
                                  local_TOD.local_t_list,
                                  local_TOD.local_TOD_ndiode_list,
                                  local_TOD.local_gain_proj_list,
                                  local_TOD.local_Tsky_proj_list,
                                  local_TOD.local_rec_proj_list,
                                  init_Tsky_params,
                                  init_Trec_params, 
                                  init_noise_params, 
                                  wnoise_var=2.5e-6,
                                  Tsky_prior_cov_inv=None,
                                  Tsky_prior_mean=None,
                                  Trec_prior_cov_inv=None,
                                  Trec_prior_mean=None,
                                  gain_prior_cov_inv=None,
                                  gain_prior_mean=None,
                                  noise_prior_func=None,
                                  n_samples=2,
                                  tol=1e-15,
                                  linear_solver=None,
                                  root=None,
                                  )

# Save the samples (numpy arraies) to the disk
if mpiutil.rank0:
    savename = "Tsky_samples.npy"
    Tsky_savepath = os.path.join(savepath, savename)
    np.save(savepath, Tsky_samples)

    savename = "gain_samples.npy"
    gain_savepath = os.path.join(savepath, savename)
    np.save(gain_savepath, np.concatenate(all_gain_samples, axis=0))

    savename = "noise_samples.npy"
    noise_savepath = os.path.join(savepath, savename)
    np.save(noise_savepath, np.concatenate(all_noise_samples, axis=0))

    savename = "rec_samples.npy"
    rec_savepath = os.path.join(savepath, savename)
    np.save(rec_savepath, np.concatenate(all_rec_samples, axis=0))

    print("All samples have been saved to the disk.")

# Done!








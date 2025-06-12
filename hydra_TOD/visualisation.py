import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import corner
import healpy as hp


def view_patch_map(map, pixel_indices, nside=64, 
                   title=" ", 
                   unit="K", cmap='RdBu_r', 
                   map_min=20, map_max=70, save_path=None):
    # Create a new map with just the patch (other pixels set to UNSEEN)
    patch_only_map = np.full(len(map), hp.UNSEEN)
    patch_only_map[pixel_indices] = map[pixel_indices]

    # Get pixel coordinates
    theta, phi = hp.pix2ang(nside, pixel_indices)
    lon, lat = np.degrees(phi), 90 - np.degrees(theta)
    lon_center, lat_center = np.median(lon), np.median(lat)

    # Calculate appropriate zoom/resolution
    patch_size = 100  # Add 20% margin
    res = patch_size / 20  # Adjust resolution based on patch size

    hp.gnomview(patch_only_map, rot=(lon_center, lat_center), 
           xsize=520, ysize=350, reso=res, 
           title=title, 
           unit=unit, cmap=cmap, min=map_min, max=map_max,
           coord=['C'], 
           cbar=True, notext=False, badcolor='gray')

    hp.graticule(dpar=10, dmer=10, coord=['C'], local=True)  
    plt.gca().set_facecolor('gray')  # Set background to white
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_residual_histogram(residuals, 
                            binwidth=0.05, # Adjust this value as needed for your applicatio
                            kde=False,
                            save_path=None):
    """
    Plots a styled histogram with KDE of residuals,
    including mean and 16th–84th percentile annotations.

    Parameters:
        residuals (array-like): Array of residual values.
    """
    # Set up the plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style='whitegrid')

    # Plot histogram with KDE
    ax = sns.histplot(residuals, bins=50, binwidth=binwidth, # Adjust binwidth as needed for your applicatio
                      kde=kde,
                      color='#2ecc71', edgecolor='w',
                      linewidth=0.9, alpha=0.8)

    # Compute statistics
    mean_res = np.mean(residuals)
    p16 = np.percentile(residuals, 16)
    p84 = np.percentile(residuals, 84)
    print("The mean residual is: ", mean_res)
    print("16th percentile: ", p16)
    print("84th percentile: ", p84)

    # Add vertical lines
    # plt.axvline(mean_res, color='#e74c3c', linestyle='--', lw=2, label='Mean')
    plt.axvline(p16, color='#3498db', linestyle=':', lw=2.5, label='16th–84th percentile')
    plt.axvline(p84, color='#3498db', linestyle=':', lw=2.5)
    plt.xlim(-1.5, 1.5)

    # Annotate plot
    plt.xlabel(r'$T_{\mathrm{residual}} = \langle T^{\mathrm{sample}}_{\mathrm{sky}} \rangle - T_{\mathrm{sky}}^{\mathrm{true}}$ [K]', fontsize=16)
    plt.ylabel('Probability Density', fontsize=16)
    plt.grid(alpha=0.2)
    plt.text(0.05, 0.95,
             f'16th–84th pct = [{p16:.2f}, {p84:.2f}] K',
             transform=ax.transAxes, va='top',
             bbox=dict(facecolor='white', alpha=0.8), fontsize=16)

    # Set the size of the tick labels
    plt.tick_params(axis='both', which='major', labelsize=14)
    

    plt.tight_layout()

    # Save plot if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()



def plot_corner(samples, labels=[r'$a_0$', r'$a_1$', r'$a_2$',r'$a_3$', r'$T_{\mathrm{nd}}$',
                                r'$c_0$', r'$c_1$', r'$c_2$',r'$c_3$',
                                r'$\text{log}_{10} f_0$',r'$\alpha$'], 
                truths=None, quantiles=[0.16, 0.5, 0.84], 
                show_titles=True, title_kwargs={"fontsize": 15}, 
                label_kwargs={"fontsize": 15}, smooth=1.0, smooth1d=1.0,
                truth_color='orangered', fill_contours=True, 
                hist_kwargs={'color': 'navy'}, 
                save_path=None, **kwargs):
    """
    Generates and optionally saves a corner plot from MCMC samples.

    Parameters:
    -----------
    samples : numpy.ndarray
        The MCMC samples array of shape (n_samples, n_params).
    labels : list of str, optional
        A list of names for the parameters (e.g., ['p1', 'p2', ...]). 
        Defaults to None (corner.py might generate default labels).
    # ... (other parameters as before)
    """
    
    if not isinstance(samples, np.ndarray) or samples.ndim != 2:
        raise ValueError("Input 'samples' must be a 2D NumPy array.")

    if labels is not None and len(labels) != samples.shape[1]:
        raise ValueError(f"Number of labels ({len(labels)}) must match number of parameters in samples ({samples.shape[1]}).")

    figure = corner.corner(
        samples,
        labels=labels, # This is where your list of parameter names goes
        truths=truths,
        quantiles=quantiles,
        show_titles=show_titles,
        title_kwargs=title_kwargs,
        label_kwargs=label_kwargs,
        smooth=smooth,
        smooth1d=smooth1d,
        truth_color=truth_color,
        fill_contours=fill_contours,
        hist_kwargs=hist_kwargs,
        **kwargs
    )

    figure.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Corner plot saved to {save_path}")
        plt.show()
    else:
        plt.show()
        
    return figure

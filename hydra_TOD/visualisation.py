import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import corner
import healpy as hp
import os



def combine_pdfs_to_panels(pdf_files, rows, cols, figsize=(16, 12), 
                          output_file=None, 
                          titles=None, suptitle=None, dpi=300, 
                          common_xlabel=None, common_ylabel=None, 
                          add_panel_labels=True, panel_label_fontsize=16,
                          ay=1., by=0., ly=0.0,
                          ax=1., bx=0., lx=0.0):
    """
    Combine multiple PDF files into a single PDF with panels arranged in rows and columns.
    
    Parameters:
    -----------
    pdf_files : list of str
        List of PDF file paths to combine
    output_file : str
        Output PDF file path
    rows : int
        Number of rows in the panel layout
    cols : int
        Number of columns in the panel layout
    figsize : tuple
        Figure size (width, height) in inches
    titles : list of str, optional
        List of titles for each panel
    suptitle : str, optional
        Main title for the entire figure
    dpi : int
        DPI for the output PDF
    common_xlabel : str, optional
        Common x-axis label for all panels
    common_ylabel : str, optional
        Common y-axis label for all panels
    label_fontsize : int
        Font size for common axis labels
    add_panel_labels : bool
        Whether to add panel labels (a, b, c, etc.)
    panel_label_fontsize : int
        Font size for panel labels
    """
    
    # Verify we have the right number of files
    expected_panels = rows * cols
    if len(pdf_files) != expected_panels:
        raise ValueError(f"Expected {expected_panels} PDF files for {rows}x{cols} layout, got {len(pdf_files)}")
    
    # Convert PDFs to images first (using pdf2image library)
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError("Please install pdf2image: pip install pdf2image")
    
    # Create the combined figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Process each PDF file
    for i, pdf_file in enumerate(pdf_files):
        if not os.path.exists(pdf_file):
            print(f"Warning: File {pdf_file} not found, skipping...")
            continue
            
        # Convert PDF to image
        try:
            images = convert_from_path(pdf_file, dpi=200, first_page=1, last_page=1)
            img_array = np.array(images[0])
            
            # Display the image
            axes[i].imshow(img_array)
            axes[i].axis('off')
            
            # Add title if provided
            if titles and i < len(titles) and titles[i]:
                axes[i].set_title(titles[i], fontsize=12, pad=10)
                
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            axes[i].text(0.5, 0.5, f"Error loading\n{os.path.basename(pdf_file)}", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(len(pdf_files), len(axes)):
        axes[i].axis('off')
    
    # Add main title if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=0.95)
    
    # Adjust layout to make room for common labels and panel labels
    plt.tight_layout()
    if suptitle:
        plt.subplots_adjust(top=0.9)
    
    # Adjust margins for labels
    left_margin = 0.08 if common_ylabel else 0.05
    bottom_margin = 0.08 if common_xlabel else 0.05
    
    # Extra space for panel labels
    if add_panel_labels:
        left_margin += 0.03  # Extra space for row labels
        bottom_margin += 0.03  # Extra space for column labels
    
    plt.subplots_adjust(left=left_margin, bottom=bottom_margin)
    
    # Add common axis labels
    # if common_xlabel:
    #     fig.text(0.5, 0.02, common_xlabel, ha='center', va='bottom', 
    #             fontsize=label_fontsize, weight='bold')
    
    # if common_ylabel:
    #     fig.text(0.02, 0.5, common_ylabel, ha='center', va='center', 
    #             rotation='vertical', fontsize=label_fontsize, weight='bold')
    
    # Add panel labels
    if add_panel_labels:
        
        # Row labels  
        row_labels = [common_ylabel] * rows  # Use common x-label as row label
        for i, label in enumerate(row_labels):
            y_pos =ay *( 1 - (i + 0.5) / rows ) + by # Center vertically for each row
            fig.text(left_margin - ly, y_pos, f'{label}', 
                    ha='center', va='center', 
                    fontsize=panel_label_fontsize, weight='bold', rotation='vertical')
        
        # Column labels (bottom edge) - numbers: 1, 2, 3, ... or letters
        col_labels = [common_xlabel] * cols  # Use common y-label as column label

        
        for j, label in enumerate(col_labels):
            x_pos = ax * ((j + 0.5) / cols) + bx  # Center horizontally for each column
            fig.text(x_pos, bottom_margin - lx, f'{label}', 
                    ha='center', va='center', 
                    fontsize=panel_label_fontsize, weight='bold')
    
    # Save the combined figure
    if output_file is not None:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    pass

def plot_residual_histogram(type_str,
                            residuals, 
                            residuals_common,
                            binwidth=0.05,
                            fts=24,  # Increased from 19 to 24
                            density=False,
                            kde=False,
                            save_path=None,
                            figsize=(12, 7),
                            show_mean=False,
                            show_std=True,
                            colors=None,
                            title=None,
                            print_xlabel=False,
                            print_ylabel=False,
                            xlim=None):
    """
    Plots two histograms: one of all residuals as the gray background and, on top of it, one of common residuals in red,
    including mean and 16th–84th percentile annotations.

    Parameters:
        residuals (array-like): Array of all residual values (in gray).
        residuals_common (array-like): Array of common residual values (in red).
        binwidth (float): Width of histogram bins
        fts (int): Font size for labels
        kde (bool): Whether to show KDE overlay
        save_path (str): Path to save the figure
        figsize (tuple): Figure size (width, height)
        show_mean (bool): Whether to show mean lines
        show_std (bool): Whether to show standard deviation in text
        colors (dict): Custom color scheme
        title (str): Plot title
        xlim (tuple): x-axis limits
    """
    # Default colors
    if colors is None:
        colors = {
            'all': '#7f8c8d',      # Sophisticated gray
            'common': '#e74c3c',    # Red
            'all_lines': '#2980b9', # Blue
            'common_lines': '#c0392b', # Dark red
            'mean_all': '#34495e',   # Dark gray
            'mean_common': '#a93226' # Dark red
        }
    
    # Set up the plot with better styling
    plt.figure(figsize=figsize)
    sns.set_theme(style='whitegrid')
    
    # Calculate common bin edges for consistent comparison
    all_data = np.concatenate([residuals, residuals_common])
    if xlim is None:
        data_range = np.max(all_data) - np.min(all_data)
        margin = 0.1 * data_range
        xlim = (np.min(all_data) - margin, np.max(all_data) + margin)
    
    # Calculate bins
    n_bins = int((xlim[1] - xlim[0]) / binwidth)
    bins = np.linspace(xlim[0], xlim[1], n_bins)

    # Plot histogram of all residuals as background
    ax = plt.gca()
    n_all, bins_all, patches_all = plt.hist(residuals, bins=bins, 
                                            color=colors['all'], 
                                            alpha=0.8, 
                                            density=density,
                                            edgecolor='white', 
                                            linewidth=0.5,
                                            label='All residuals')

    # Plot histogram of common residuals on top
    n_common, bins_common, patches_common = plt.hist(residuals_common, bins=bins,
                                                     color=colors['common'], 
                                                     alpha=0.5, 
                                                     density=density,
                                                     edgecolor='white', 
                                                     linewidth=0.5,
                                                     label='Common residuals')

    # Add KDE if requested
    if kde:
        from scipy.stats import gaussian_kde
        
        # KDE for all residuals
        kde_all = gaussian_kde(residuals)
        x_kde = np.linspace(xlim[0], xlim[1], 200)
        y_kde_all = kde_all(x_kde)
        plt.plot(x_kde, y_kde_all, color=colors['all'], linestyle='-', 
                linewidth=2, alpha=0.8)
        
        # KDE for common residuals
        kde_common = gaussian_kde(residuals_common)
        y_kde_common = kde_common(x_kde)
        plt.plot(x_kde, y_kde_common, color=colors['common'], linestyle='-', 
                linewidth=2, alpha=0.9)

    # Compute comprehensive statistics
    stats_all = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'median': np.median(residuals),
        'p16': np.percentile(residuals, 16),
        'p84': np.percentile(residuals, 84),
        'rms': np.sqrt(np.mean(residuals**2))
    }
    
    stats_common = {
        'mean': np.mean(residuals_common),
        'std': np.std(residuals_common),
        'median': np.median(residuals_common),
        'p16': np.percentile(residuals_common, 16),
        'p84': np.percentile(residuals_common, 84),
        'rms': np.sqrt(np.mean(residuals_common**2))
    }

    # Print statistics
    print(f"All residuals - Mean: {stats_all['mean']:.4f}, Std: {stats_all['std']:.4f}, RMS: {stats_all['rms']:.4f}")
    print(f"All residuals - 16th-84th percentile: [{stats_all['p16']:.4f}, {stats_all['p84']:.4f}]")
    print(f"Common residuals - Mean: {stats_common['mean']:.4f}, Std: {stats_common['std']:.4f}, RMS: {stats_common['rms']:.4f}")
    print(f"Common residuals - 16th-84th percentile: [{stats_common['p16']:.4f}, {stats_common['p84']:.4f}]")

    # Add vertical lines for percentiles
    plt.axvline(stats_all['p16'], color=colors['all_lines'], linestyle=':', 
               lw=3, alpha=0.8, label='All: 16th–84th percentile')  # Increased linewidth
    plt.axvline(stats_all['p84'], color=colors['all_lines'], linestyle=':', lw=3, alpha=0.8)
    
    plt.axvline(stats_common['p16'], color=colors['common_lines'], linestyle='--', 
               lw=3, alpha=0.9, label='Common: 16th–84th percentile')  # Increased linewidth
    plt.axvline(stats_common['p84'], color=colors['common_lines'], linestyle='--', lw=3, alpha=0.9)
    
    # Add mean lines if requested
    if show_mean:
        plt.axvline(stats_all['mean'], color=colors['mean_all'], linestyle='-', 
                   lw=3, alpha=0.7, label='All: Mean')  # Increased linewidth
        plt.axvline(stats_common['mean'], color=colors['mean_common'], linestyle='-', 
                   lw=3, alpha=0.8, label='Common: Mean')  # Increased linewidth

    # Set axis limits
    plt.xlim(xlim)
    
    # Enhanced annotations with larger font sizes
    if print_xlabel:
        plt.xlabel(r'$T_{\mathrm{residual}} = \langle T^{\mathrm{sample}}_{\mathrm{sky}} \rangle - T_{\mathrm{sky}}^{\mathrm{true}}$ [K]', 
                fontsize=fts)
    if print_ylabel:
        plt.ylabel('Histogram', fontsize=fts)

    if title:
        plt.title(title, fontsize=fts+4, pad=20)  # Increased from +2 to +4
    
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Enhanced text annotation with larger font size
    text_lines = [
        f'$\\mathbf{{{type_str}}}$',
        f'',
        f'All residuals (n={len(residuals)}):',
        f'  16th–84th pct: [{stats_all["p16"]:.3f}, {stats_all["p84"]:.3f}] K',
        f'  Mean ± Std: {stats_all["mean"]:.3f} ± {stats_all["std"]:.3f} K',
        f'',
        f'Common residuals (n={len(residuals_common)}):',
        f'  16th–84th pct: [{stats_common["p16"]:.3f}, {stats_common["p84"]:.3f}] K',
        f'  Mean ± Std: {stats_common["mean"]:.3f} ± {stats_common["std"]:.3f} K'
    ]
    
    text_str = '\n'.join(text_lines)
    
    plt.text(0.02, 0.95, text_str,
             transform=ax.transAxes, va='top', ha='left',
             bbox=dict(facecolor='white', alpha=0.9, pad=10, edgecolor='gray'),  # Increased padding
             fontsize=fts-5, fontfamily='monospace')  # Changed from fts-4 to fts-6, but still larger than before

    # Enhanced legend with larger font size
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate labels
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
              loc='lower left', fontsize=fts-4,  # Changed from fts-3 to fts-4, but still larger
              frameon=False, # framealpha=0.9, edgecolor='gray'
              )

    # Set tick parameters with larger font sizes
    plt.tick_params(axis='both', which='major', labelsize=fts-2)  # Increased tick label size
    plt.tick_params(axis='both', which='minor', labelsize=fts-4)

    plt.tight_layout()

    # Save plot if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Plot saved to: {save_path}")

    plt.show()
    
    # Return statistics for further analysis
    # return {'all': stats_all, 'common': stats_common}
    pass


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


def view_patch_map(map, pixel_indices):
    # Create a new map with just the patch (other pixels set to UNSEEN)
    patch_only_map = np.full(len(map), hp.UNSEEN)
    patch_only_map[pixel_indices] = map[pixel_indices]
    return patch_only_map


def generate_post_map(sim, Tsys_samples, save_root='outputs/GS1'):

    pixel_indices = sim.pixel_indices
    nside = sim.nside

    # Get pixel coordinates
    theta, phi = hp.pix2ang(nside, pixel_indices)
    lon, lat = np.degrees(phi), 90 - np.degrees(theta)
    lon_center, lat_center = np.median(lon), np.median(lat)

    # Calculate appropriate zoom/resolution
    patch_size = 100  # Add 20% margin
    res = patch_size / 20  # Adjust resolution based on patch size

    num_pixels = len(pixel_indices)
    Tsky_samples = Tsys_samples[:, :num_pixels]
    Tmean = np.mean(Tsky_samples, axis=0)
    Tstd = np.std(Tsky_samples, axis=0)

    NPIX = hp.nside2npix(nside)

    true_map = np.zeros(NPIX, dtype=float)
    sample_mean_map = np.zeros(NPIX, dtype=float)
    sample_std_map = np.zeros(NPIX, dtype=float)

    true_map[pixel_indices] = sim.sky_params
    sample_mean_map[pixel_indices] = Tmean
    sample_std_map[pixel_indices] = Tstd

    patch_true_map = view_patch_map(true_map, pixel_indices)

    hp.gnomview(patch_true_map, rot=(lon_center, lat_center), 
           xsize=520, ysize=350, reso=res, 
           title=" ", 
           unit="K", cmap='jet', # min=sky_min, max=sky_max,
           #notext=True,
           coord=['C'], 
           cbar=True, notext=False, badcolor='gray')
    #plt.grid(True)
    hp.graticule(dpar=10, dmer=10, coord=['C'], local=True)  
    #plt.grid(color='gray', linestyle=':', alpha=0.5)  # Custom grid style
    plt.gca().set_facecolor('gray')  # Set background to white
    plt.savefig(save_root + '/true_map.pdf', bbox_inches='tight', 
                pad_inches=0.1)
    
    patch_mean_map = view_patch_map(sample_mean_map, pixel_indices)

    hp.gnomview(patch_mean_map, rot=(lon_center, lat_center), 
            xsize=520, ysize=350, reso=res, title=" ", 
            unit="K", cmap='jet', #min=sky_min, max=sky_max,
            #notext=True,
            coord=['C'], 
            cbar=True, notext=False, badcolor='gray')
    #plt.grid(True)
    hp.graticule(dpar=10, dmer=10, coord=['C'], local=True)  
    #plt.grid(color='gray', linestyle=':', alpha=0.5)  # Custom grid style
    plt.gca().set_facecolor('gray')  # Set background to white
    plt.savefig(save_root + "/mean_map.pdf", bbox_inches='tight', 
                pad_inches=0.1)
    
    patch_error_map = view_patch_map(sample_mean_map-true_map, pixel_indices)
    # plt.figure(figsize=(10, 6))
    hp.gnomview(patch_error_map, rot=(lon_center, lat_center), 
            xsize=520, ysize=350, reso=res, title=" ", 
            unit="K", cmap='RdBu_r', #min=-0.1, max=0.1,
            #notext=True,
            coord=['C'], 
            cbar=True, notext=False,
            badcolor='gray')
    #plt.grid(True)
    hp.graticule(dpar=10, dmer=10, coord=['C'], local=True)  
    #plt.grid(color='gray', linestyle=':', alpha=0.5)  # Custom grid style
    plt.gca().set_facecolor('gray')  # Set background to white
    plt.savefig(save_root + "/error_map.pdf", bbox_inches='tight', 
                pad_inches=0.1)
    
    patch_std_map = view_patch_map(sample_std_map, pixel_indices)

    hp.gnomview(patch_std_map, rot=(lon_center, lat_center), 
            xsize=520, ysize=350, reso=res, title=None, 
            unit="K", cmap='jet', 
            notext=False,
            coord=['C'], 
            cbar=True, 
            badcolor='gray',
            #min=0, max=1,
            #norm='log'
            #sub=(2, 1, 1),  # Proper subplot specification
            #margins=(0.05, 0.15, 0.05, 0.15)
            )
    hp.graticule(dpar=10, dmer=10, coord=['C'], local=True)  
    plt.gca().set_facecolor('gray')  # Set background to white
    plt.savefig(save_root + "/std_map.pdf", bbox_inches='tight', 
                pad_inches=0.1)

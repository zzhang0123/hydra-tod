#!/usr/bin/env python3
"""
Script to generate individual residual histogram plots and combine them into a single figure.
This creates 6 panels (3 rows x 2 columns) with shared x and y labels only on edge panels.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import os


def plot_residual_histogram_no_labels(residuals, residuals_common, binwidth=0.05, kde=True, 
                                    colors=None, figsize=(6, 4), xlim=None):
    """
    Generate a residual histogram plot without axis labels for combining into subplots.
    
    Parameters:
    -----------
    residuals : array-like
        Array of all residual values (background, gray)
    residuals_common : array-like  
        Array of common residual values (foreground, red)
    binwidth : float
        Width of histogram bins
    kde : bool
        Whether to show KDE overlay
    colors : dict or None
        Custom color scheme
    figsize : tuple
        Figure size (width, height)
    xlim : tuple or None
        x-axis limits
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Default colors
    if colors is None:
        colors = {
            'all': 'gray',
            'common': 'red',
            'all_lines': 'darkgray',
            'common_lines': 'darkred'
        }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Combine data to determine range if xlim not specified
    all_data = np.concatenate([residuals, residuals_common])
    if xlim is None:
        data_range = np.max(all_data) - np.min(all_data)
        margin = 0.1 * data_range
        xlim = (np.min(all_data) - margin, np.max(all_data) + margin)
    
    # Calculate bins
    n_bins = int((xlim[1] - xlim[0]) / binwidth)
    bins = np.linspace(xlim[0], xlim[1], n_bins)

    # Plot histogram of all residuals as background
    ax.hist(residuals, bins=bins, color=colors['all'], alpha=0.6, density=True,
            edgecolor='white', linewidth=0.5)

    # Plot histogram of common residuals on top
    ax.hist(residuals_common, bins=bins, color=colors['common'], alpha=0.8, density=True,
            edgecolor='white', linewidth=0.5)

    # Add KDE if requested
    if kde:
        x_kde = np.linspace(xlim[0], xlim[1], 200)
        
        # KDE for all residuals
        kde_all = gaussian_kde(residuals)
        y_kde_all = kde_all(x_kde)
        ax.plot(x_kde, y_kde_all, color=colors['all'], linestyle='-', 
                linewidth=2, alpha=0.8)
        
        # KDE for common residuals
        kde_common = gaussian_kde(residuals_common)
        y_kde_common = kde_common(x_kde)
        ax.plot(x_kde, y_kde_common, color=colors['common'], linestyle='-', 
                linewidth=2, alpha=0.9)

    # Compute statistics
    stats_all = {
        'p16': np.percentile(residuals, 16),
        'p84': np.percentile(residuals, 84),
        'mean': np.mean(residuals),
        'std': np.std(residuals)
    }
    
    stats_common = {
        'p16': np.percentile(residuals_common, 16),
        'p84': np.percentile(residuals_common, 84),
        'mean': np.mean(residuals_common),
        'std': np.std(residuals_common)
    }

    # Add vertical lines for percentiles
    ax.axvline(stats_all['p16'], color=colors['all_lines'], linestyle=':', 
               lw=3, alpha=0.8)
    ax.axvline(stats_all['p84'], color=colors['all_lines'], linestyle=':', 
               lw=3, alpha=0.8)
    ax.axvline(stats_common['p16'], color=colors['common_lines'], linestyle=':', 
               lw=3, alpha=0.9)
    ax.axvline(stats_common['p84'], color=colors['common_lines'], linestyle=':', 
               lw=3, alpha=0.9)

    # Set limits and grid
    ax.set_xlim(xlim)
    ax.grid(True, alpha=0.2)
    
    # Remove labels (will be added to combined figure)
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    
    return fig, ax, stats_all, stats_common


def create_combined_residual_figure(data_dict, titles, save_path=None, figsize=(16, 12), 
                                  binwidth=0.05, kde=True):
    """
    Create a combined figure with 6 residual histogram panels (3 rows x 2 columns).
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing residual data for each panel.
        Keys should be: 'GS1', 'GS1_db', 'GS5', 'GS5_db', 'GSF5', 'GSF5_db'
        Each value is a tuple: (residuals, residuals_common)
    titles : list
        List of 6 title strings for each panel
    save_path : str or None
        Path to save the combined figure
    figsize : tuple
        Overall figure size
    binwidth : float
        Histogram bin width
    kde : bool
        Whether to show KDE overlay
    """
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.15)
    
    # Panel positions and corresponding data keys
    panel_positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    panel_keys = ['GS1', 'GS1_db', 'GS5', 'GS5_db', 'GSF5', 'GSF5_db']
    
    # Determine global x-limits for consistency
    all_residuals = []
    for key in panel_keys:
        if key in data_dict:
            residuals, residuals_common = data_dict[key]
            all_residuals.extend(residuals)
            all_residuals.extend(residuals_common)
    
    all_residuals = np.array(all_residuals)
    data_range = np.max(all_residuals) - np.min(all_residuals)
    margin = 0.1 * data_range
    global_xlim = (np.min(all_residuals) - margin, np.max(all_residuals) + margin)
    
    axes = []
    
    # Create each subplot
    for i, ((row, col), key, title) in enumerate(zip(panel_positions, panel_keys, titles)):
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)
        
        if key not in data_dict:
            ax.text(0.5, 0.5, f'No data for {key}', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(title, fontsize=14, fontweight='bold')
            continue
            
        residuals, residuals_common = data_dict[key]
        
        # Calculate bins
        n_bins = int((global_xlim[1] - global_xlim[0]) / binwidth)
        bins = np.linspace(global_xlim[0], global_xlim[1], n_bins)
        
        # Colors
        colors = {
            'all': 'gray',
            'common': 'red', 
            'all_lines': 'darkgray',
            'common_lines': 'darkred'
        }

        # Plot histograms
        ax.hist(residuals, bins=bins, color=colors['all'], alpha=0.6, density=True,
                edgecolor='white', linewidth=0.5, label='All residuals')
        ax.hist(residuals_common, bins=bins, color=colors['common'], alpha=0.8, density=True,
                edgecolor='white', linewidth=0.5, label='Common residuals')

        # Add KDE if requested
        if kde:
            x_kde = np.linspace(global_xlim[0], global_xlim[1], 200)
            
            # KDE for all residuals
            kde_all = gaussian_kde(residuals)
            y_kde_all = kde_all(x_kde)
            ax.plot(x_kde, y_kde_all, color=colors['all'], linestyle='-', 
                    linewidth=2, alpha=0.8)
            
            # KDE for common residuals
            kde_common = gaussian_kde(residuals_common)
            y_kde_common = kde_common(x_kde)
            ax.plot(x_kde, y_kde_common, color=colors['common'], linestyle='-', 
                    linewidth=2, alpha=0.9)

        # Add percentile lines
        p16_all = np.percentile(residuals, 16)
        p84_all = np.percentile(residuals, 84)
        p16_common = np.percentile(residuals_common, 16)
        p84_common = np.percentile(residuals_common, 84)
        
        ax.axvline(p16_all, color=colors['all_lines'], linestyle=':', lw=2.5, alpha=0.8)
        ax.axvline(p84_all, color=colors['all_lines'], linestyle=':', lw=2.5, alpha=0.8)
        ax.axvline(p16_common, color=colors['common_lines'], linestyle=':', lw=2.5, alpha=0.9)
        ax.axvline(p84_common, color=colors['common_lines'], linestyle=':', lw=2.5, alpha=0.9)

        # Set title
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set limits and grid
        ax.set_xlim(global_xlim)
        ax.grid(True, alpha=0.2)
        
        # Only show x-labels for bottom row
        if row == 2:  # Bottom row
            ax.set_xlabel(r'$T_{\mathrm{residual}} = \langle T^{\mathrm{sample}}_{\mathrm{sky}} \rangle - T_{\mathrm{sky}}^{\mathrm{true}}$ [K]', 
                         fontsize=12)
        else:
            ax.set_xticklabels([])
            
        # Only show y-labels for left column
        if col == 0:  # Left column
            ax.set_ylabel('Probability Density', fontsize=12)
        else:
            ax.set_yticklabels([])
            
        # Add legend only to top-right panel
        if row == 0 and col == 1:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add overall title
    fig.suptitle('Residual Analysis: Temperature Sky Reconstruction', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        print(f"Combined figure saved to: {save_path}")
    
    return fig, axes


def main():
    """
    Main function to demonstrate usage.
    Replace this with your actual data loading.
    """
    
    # Example usage - replace with your actual data variables
    # You would load your data here, for example:
    # GS1_Tsky_residual = ...
    # GS1_Tsky_residual_common = ...
    # etc.
    
    # For demonstration, create dummy data
    np.random.seed(42)
    
    # Simulate residual data (replace with your actual data)
    data_dict = {
        'GS1': (np.random.normal(0, 0.1, 1000), np.random.normal(0, 0.08, 500)),
        'GS1_db': (np.random.normal(0, 0.09, 1000), np.random.normal(0, 0.07, 500)),
        'GS5': (np.random.normal(0, 0.08, 1000), np.random.normal(0, 0.06, 500)),
        'GS5_db': (np.random.normal(0, 0.07, 1000), np.random.normal(0, 0.05, 500)),
        'GSF5': (np.random.normal(0, 0.06, 1000), np.random.normal(0, 0.04, 500)),
        'GSF5_db': (np.random.normal(0, 0.05, 1000), np.random.normal(0, 0.03, 500)),
    }
    
    # Panel titles
    titles = [
        r'1 $\times$ TOD; 1 CalSrc',           # (0,0)
        r'2 $\times$ TOD; 1 CalSrc',           # (0,1) 
        r'1 $\times$ TOD; 5 CalSrc',           # (1,0)
        r'2 $\times$ TOD; 5 CalSrc',           # (1,1)
        r'1 $\times$ TOD; 5 CalSrc + 1/f prior',  # (2,0)
        r'2 $\times$ TOD; 5 CalSrc + 1/f prior',  # (2,1)
    ]
    
    # Create the combined figure
    fig, axes = create_combined_residual_figure(
        data_dict=data_dict,
        titles=titles,
        save_path='figures/combined_residual_analysis.pdf',
        figsize=(16, 12),
        binwidth=0.05,
        kde=True
    )
    
    plt.show()


if __name__ == "__main__":
    main()

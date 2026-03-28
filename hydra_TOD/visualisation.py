from __future__ import annotations

"""
Visualisation utilities for posterior samples, residual histograms, and sky maps.

Provides convenience wrappers around matplotlib, seaborn, corner, and healpy
for common diagnostic and publication-quality plots used during Gibbs-sampler
analysis.

References
----------
Zhang et al. (2026), RASTI, rzag024.
"""

import numpy as np
from numpy.typing import NDArray  # noqa: F401 — used in type comments
import matplotlib.pyplot as plt
import seaborn as sns
import corner
import healpy as hp
import os


def view_samples(
    p_samples: NDArray[np.floating],
    true_values: NDArray[np.floating],
) -> None:
    """
    Plot histograms of posterior samples against true parameter values.

    For each parameter, displays a normalised histogram of the samples
    together with vertical lines for the true value and the sample mean,
    and prints a numerical summary (mean, std, relative error).

    Parameters
    ----------
    p_samples : NDArray[np.floating]
        Posterior samples, shape ``(n_samples, n_params)``.
    true_values : NDArray[np.floating]
        Ground-truth parameter values, shape ``(n_params,)``.
    """
    n_params = p_samples.shape[1]
    mean = np.mean(p_samples, axis=0)
    std = np.std(p_samples, axis=0)

    # Create subplots for four parameters
    # Set figure size according to number of parameters
    fig, axes = plt.subplots(n_params, 1, figsize=(8, 4 * n_params))
    axes = axes.ravel()

    for i in range(n_params):
        # Plot histogram of samples for each parameter
        axes[i].hist(p_samples[:, i], bins=50, density=True, alpha=0.6, label="Samples")

        # Plot true value line
        axes[i].axvline(
            x=true_values[i],
            color="r",
            linestyle="-",
            label="True Value",
            linewidth=2,
            alpha=0.7,
        )

        # Plot mean value line
        axes[i].axvline(x=mean[i], color="g", linestyle="--", label="Mean")

        # Add labels and title
        axes[i].set_xlabel("Coefficient")
        axes[i].set_ylabel("Density")
        axes[i].set_title(f"Parameter {i+1}")
        axes[i].legend()

        # Print numerical comparison for each parameter
        print(f"\n Parameter {i+1}:")
        print(f"True value: {true_values[i]:.6f}")
        print(f"Mean sampled: {mean[i]:.6f}")
        print(f"Standard deviation: {std[i]:.6f}")
        print(
            f"Relative error: {abs(mean[i] - true_values[i])/true_values[i]*100:.2f}%"
        )

    plt.tight_layout()
    plt.show()


def combine_pdfs_to_panels(
    pdf_files: list[str],
    rows: int,
    cols: int,
    figsize: tuple[float, float] = (16, 12),
    output_file: str | None = None,
    titles: list[str] | None = None,
    suptitle: str | None = None,
    dpi: int = 300,
    common_xlabel: str | None = None,
    common_ylabel: str | None = None,
    add_panel_labels: bool = True,
    panel_label_fontsize: int = 16,
    ay: float = 1.0,
    by: float = 0.0,
    ly: float = 0.0,
    ax: float = 1.0,
    bx: float = 0.0,
    lx: float = 0.0,
) -> None:
    """
    Combine multiple PDF files into a single figure with panel layout.

    Each PDF is rasterised and placed into a grid of ``rows x cols``
    subplots.  Optionally adds row / column labels and a super-title.

    Parameters
    ----------
    pdf_files : list of str
        Paths to the PDF files to combine.  Must contain exactly
        ``rows * cols`` entries.
    rows : int
        Number of rows in the panel layout.
    cols : int
        Number of columns in the panel layout.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches. Default is ``(16, 12)``.
    output_file : str or None, optional
        Path to save the combined figure.  If ``None``, the figure is only
        displayed.
    titles : list of str or None, optional
        Individual titles for each panel.
    suptitle : str or None, optional
        Main title for the entire figure.
    dpi : int, optional
        DPI for the output file. Default is 300.
    common_xlabel : str or None, optional
        Common x-axis label replicated across columns.
    common_ylabel : str or None, optional
        Common y-axis label replicated across rows.
    add_panel_labels : bool, optional
        Whether to add row/column labels. Default is ``True``.
    panel_label_fontsize : int, optional
        Font size for panel labels. Default is 16.
    ay, by, ly : float, optional
        Affine positioning parameters for row labels.
    ax, bx, lx : float, optional
        Affine positioning parameters for column labels.
    """

    # Verify we have the right number of files
    expected_panels = rows * cols
    if len(pdf_files) != expected_panels:
        raise ValueError(
            f"Expected {expected_panels} PDF files for {rows}x{cols} layout, got {len(pdf_files)}"
        )

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
            axes[i].axis("off")

            # Add title if provided
            if titles and i < len(titles) and titles[i]:
                axes[i].set_title(titles[i], fontsize=12, pad=10)

        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            axes[i].text(
                0.5,
                0.5,
                f"Error loading\n{os.path.basename(pdf_file)}",
                ha="center",
                va="center",
                transform=axes[i].transAxes,
            )
            axes[i].axis("off")

    # Hide any unused subplots
    for i in range(len(pdf_files), len(axes)):
        axes[i].axis("off")

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
            y_pos = ay * (1 - (i + 0.5) / rows) + by  # Center vertically for each row
            fig.text(
                left_margin - ly,
                y_pos,
                f"{label}",
                ha="center",
                va="center",
                fontsize=panel_label_fontsize,
                weight="bold",
                rotation="vertical",
            )

        # Column labels (bottom edge) - numbers: 1, 2, 3, ... or letters
        col_labels = [common_xlabel] * cols  # Use common y-label as column label

        for j, label in enumerate(col_labels):
            x_pos = ax * ((j + 0.5) / cols) + bx  # Center horizontally for each column
            fig.text(
                x_pos,
                bottom_margin - lx,
                f"{label}",
                ha="center",
                va="center",
                fontsize=panel_label_fontsize,
                weight="bold",
            )

    # Save the combined figure
    if output_file is not None:
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.show()

    pass


def plot_residual_histogram_old(
    type_str: str,
    residuals: NDArray[np.floating],
    residuals_common: NDArray[np.floating],
    binwidth: float = 0.05,
    fts: int = 24,  # Increased from 19 to 24
    density: bool = False,
    kde: bool = False,
    save_path: str | None = None,
    figsize: tuple[float, float] = (12, 7),
    show_mean: bool = False,
    show_std: bool = True,
    colors: dict[str, str] | None = None,
    title: str | None = None,
    print_xlabel: bool = False,
    print_ylabel: bool = False,
    xlim: tuple[float, float] | None = None,
) -> None:
    """
    Plot overlaid histograms of all and common residuals (old version).

    Displays a grey background histogram of all residuals with a red
    overlay for common (interior) residuals, annotated with 16th--84th
    percentile lines and summary statistics.

    Parameters
    ----------
    type_str : str
        Label string displayed in the annotation box (e.g. scan name).
    residuals : NDArray[np.floating]
        All residual values (plotted in grey).
    residuals_common : NDArray[np.floating]
        Common (interior pixel) residual values (plotted in red).
    binwidth : float, optional
        Histogram bin width in data units. Default is 0.05.
    fts : int, optional
        Base font size for labels. Default is 24.
    density : bool, optional
        If ``True``, normalise histograms to a density. Default is ``False``.
    kde : bool, optional
        If ``True``, overlay a KDE curve. Default is ``False``.
    save_path : str or None, optional
        File path to save the figure.
    figsize : tuple of float, optional
        Figure size ``(width, height)``. Default is ``(12, 7)``.
    show_mean : bool, optional
        Whether to show vertical mean lines. Default is ``False``.
    show_std : bool, optional
        Whether to include standard deviation in text. Default is ``True``.
    colors : dict or None, optional
        Custom colour mapping with keys ``'all'``, ``'common'``,
        ``'all_lines'``, ``'common_lines'``, ``'mean_all'``,
        ``'mean_common'``.
    title : str or None, optional
        Plot title.
    print_xlabel : bool, optional
        Whether to print the x-axis label. Default is ``False``.
    print_ylabel : bool, optional
        Whether to print the y-axis label. Default is ``False``.
    xlim : tuple of float or None, optional
        Explicit x-axis limits ``(xmin, xmax)``.

    Notes
    -----
    Superseded by :func:`plot_residual_histogram` which uses a cleaner
    style.  Kept for backward compatibility.
    """
    # Default colors
    if colors is None:
        colors = {
            "all": "#7f8c8d",  # Sophisticated gray
            "common": "#e74c3c",  # Red
            "all_lines": "#2980b9",  # Blue
            "common_lines": "#c0392b",  # Dark red
            "mean_all": "#34495e",  # Dark gray
            "mean_common": "#a93226",  # Dark red
        }

    # Set up the plot with better styling
    plt.figure(figsize=figsize)
    sns.set_theme(style="whitegrid")

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
    n_all, bins_all, patches_all = plt.hist(
        residuals,
        bins=bins,
        color=colors["all"],
        alpha=0.8,
        density=density,
        edgecolor="white",
        linewidth=0.5,
        label="All residuals",
    )

    # Plot histogram of common residuals on top
    n_common, bins_common, patches_common = plt.hist(
        residuals_common,
        bins=bins,
        color=colors["common"],
        alpha=0.5,
        density=density,
        edgecolor="white",
        linewidth=0.5,
        label="Common residuals",
    )

    # Add KDE if requested
    if kde:
        from scipy.stats import gaussian_kde

        # KDE for all residuals
        kde_all = gaussian_kde(residuals)
        x_kde = np.linspace(xlim[0], xlim[1], 200)
        y_kde_all = kde_all(x_kde)
        plt.plot(
            x_kde, y_kde_all, color=colors["all"], linestyle="-", linewidth=2, alpha=0.8
        )

        # KDE for common residuals
        kde_common = gaussian_kde(residuals_common)
        y_kde_common = kde_common(x_kde)
        plt.plot(
            x_kde,
            y_kde_common,
            color=colors["common"],
            linestyle="-",
            linewidth=2,
            alpha=0.9,
        )

    # Compute comprehensive statistics
    stats_all = {
        "mean": np.mean(residuals),
        "std": np.std(residuals),
        "median": np.median(residuals),
        "p16": np.percentile(residuals, 16),
        "p84": np.percentile(residuals, 84),
        "rms": np.sqrt(np.mean(residuals**2)),
    }

    stats_common = {
        "mean": np.mean(residuals_common),
        "std": np.std(residuals_common),
        "median": np.median(residuals_common),
        "p16": np.percentile(residuals_common, 16),
        "p84": np.percentile(residuals_common, 84),
        "rms": np.sqrt(np.mean(residuals_common**2)),
    }

    # Print statistics
    print(
        f"All residuals - Mean: {stats_all['mean']:.4f}, Std: {stats_all['std']:.4f}, RMS: {stats_all['rms']:.4f}"
    )
    print(
        f"All residuals - 16th-84th percentile: [{stats_all['p16']:.4f}, {stats_all['p84']:.4f}]"
    )
    print(
        f"Common residuals - Mean: {stats_common['mean']:.4f}, Std: {stats_common['std']:.4f}, RMS: {stats_common['rms']:.4f}"
    )
    print(
        f"Common residuals - 16th-84th percentile: [{stats_common['p16']:.4f}, {stats_common['p84']:.4f}]"
    )

    # Add vertical lines for percentiles
    plt.axvline(
        stats_all["p16"],
        color=colors["all_lines"],
        linestyle=":",
        lw=3,
        alpha=0.8,
        label="All: 16th–84th percentile",
    )  # Increased linewidth
    plt.axvline(
        stats_all["p84"], color=colors["all_lines"], linestyle=":", lw=3, alpha=0.8
    )

    plt.axvline(
        stats_common["p16"],
        color=colors["common_lines"],
        linestyle="--",
        lw=3,
        alpha=0.9,
        label="Common: 16th–84th percentile",
    )  # Increased linewidth
    plt.axvline(
        stats_common["p84"],
        color=colors["common_lines"],
        linestyle="--",
        lw=3,
        alpha=0.9,
    )

    # Add mean lines if requested
    if show_mean:
        plt.axvline(
            stats_all["mean"],
            color=colors["mean_all"],
            linestyle="-",
            lw=3,
            alpha=0.7,
            label="All: Mean",
        )  # Increased linewidth
        plt.axvline(
            stats_common["mean"],
            color=colors["mean_common"],
            linestyle="-",
            lw=3,
            alpha=0.8,
            label="Common: Mean",
        )  # Increased linewidth

    # Set axis limits
    plt.xlim(xlim)

    # Enhanced annotations with larger font sizes
    if print_xlabel:
        plt.xlabel(
            r"$T_{\mathrm{residual}} = \langle T^{\mathrm{sample}}_{\mathrm{sky}} \rangle - T_{\mathrm{sky}}^{\mathrm{true}}$ [K]",
            fontsize=fts,
        )
    if print_ylabel:
        plt.ylabel("Histogram", fontsize=fts)

    if title:
        plt.title(title, fontsize=fts + 4, pad=20)  # Increased from +2 to +4

    plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Enhanced text annotation with larger font size
    text_lines = [
        f"$\\mathbf{{{type_str}}}$",
        f"",
        f"All residuals (n={len(residuals)}):",
        f'  16th–84th pct: [{stats_all["p16"]:.3f}, {stats_all["p84"]:.3f}] K',
        f'  Mean ± Std: {stats_all["mean"]:.3f} ± {stats_all["std"]:.3f} K',
        f"",
        f"Common residuals (n={len(residuals_common)}):",
        f'  16th–84th pct: [{stats_common["p16"]:.3f}, {stats_common["p84"]:.3f}] K',
        f'  Mean ± Std: {stats_common["mean"]:.3f} ± {stats_common["std"]:.3f} K',
    ]

    text_str = "\n".join(text_lines)

    plt.text(
        0.02,
        0.95,
        text_str,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(
            facecolor="white", alpha=0.9, pad=10, edgecolor="gray"
        ),  # Increased padding
        fontsize=fts - 5,
        fontfamily="monospace",
    )  # Changed from fts-4 to fts-6, but still larger than before

    # Enhanced legend with larger font size
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate labels
    by_label = dict(zip(labels, handles))
    plt.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower left",
        fontsize=fts - 4,  # Changed from fts-3 to fts-4, but still larger
        frameon=False,  # framealpha=0.9, edgecolor='gray'
    )

    # Set tick parameters with larger font sizes
    plt.tick_params(
        axis="both", which="major", labelsize=fts - 2
    )  # Increased tick label size
    plt.tick_params(axis="both", which="minor", labelsize=fts - 4)

    plt.tight_layout()

    # Save plot if save_path is provided
    if save_path is not None:
        plt.savefig(
            save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Plot saved to: {save_path}")

    plt.show()

    # Return statistics for further analysis
    # return {'all': stats_all, 'common': stats_common}
    pass


def plot_corner(
    samples: NDArray[np.floating],
    labels: list[str] | None = None,
    truths: list[float] | NDArray[np.floating] | None = None,
    quantiles: list[float] = [0.16, 0.5, 0.84],
    show_titles: bool = True,
    title_kwargs: dict[str, object] = {"fontsize": 10, "loc": "left"},
    label_kwargs: dict[str, object] = {"fontsize": 12},
    smooth: float = 1.0,
    smooth1d: float = 1.0,
    truth_color: str = "orangered",
    fill_contours: bool = True,
    hist_kwargs: dict[str, object] = {"color": "navy"},
    max_n_ticks: int = 3,
    title_fmt: str = ".3f",
    hspace: float = 0.04,
    wspace: float = 0.04,
    save_path: str | None = None,
    **kwargs: object,
) -> plt.Figure:
    """
    Generate a corner plot from MCMC samples.

    Wraps ``corner.corner`` with sensible defaults for publication-quality
    figures, including rotated tick labels and tight panel spacing.

    Parameters
    ----------
    samples : NDArray[np.floating]
        MCMC samples of shape ``(n_samples, n_params)``.
    labels : list of str or None, optional
        Parameter names for axis labels.  If ``None``, uses default
        labels from the corner package.
    truths : list of float or NDArray or None, optional
        True parameter values to mark on the plot.
    quantiles : list of float, optional
        Quantiles to display on the 1-D histograms.
        Default is ``[0.16, 0.5, 0.84]``.
    show_titles : bool, optional
        Whether to display quantile-based titles. Default is ``True``.
    title_kwargs : dict, optional
        Keyword arguments for title text.
    label_kwargs : dict, optional
        Keyword arguments for axis labels.
    smooth : float, optional
        Gaussian kernel width for 2-D smoothing. Default is 1.0.
    smooth1d : float, optional
        Gaussian kernel width for 1-D smoothing. Default is 1.0.
    truth_color : str, optional
        Colour for truth lines. Default is ``'orangered'``.
    fill_contours : bool, optional
        Whether to fill 2-D contours. Default is ``True``.
    hist_kwargs : dict, optional
        Keyword arguments for 1-D histograms.
    max_n_ticks : int, optional
        Maximum number of ticks per axis. Default is 3.
    title_fmt : str, optional
        Format string for title values. Default is ``'.3f'``.
    hspace : float, optional
        Vertical spacing between panels. Default is 0.04.
    wspace : float, optional
        Horizontal spacing between panels. Default is 0.04.
    save_path : str or None, optional
        Path to save the figure.
    **kwargs
        Additional keyword arguments passed to ``corner.corner``.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The corner-plot figure object.

    Raises
    ------
    ValueError
        If *samples* is not a 2-D array, or if the number of labels does
        not match the number of parameters.
    """

    if not isinstance(samples, np.ndarray) or samples.ndim != 2:
        raise ValueError("Input 'samples' must be a 2D NumPy array.")

    if labels is not None and len(labels) != samples.shape[1]:
        raise ValueError(
            f"Number of labels ({len(labels)}) must match number of parameters in samples ({samples.shape[1]})."
        )

    figure = corner.corner(
        samples,
        labels=labels,
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
        max_n_ticks=max_n_ticks,
        title_fmt=title_fmt,
        **kwargs,
    )

    label_fontsize = label_kwargs.get("fontsize", 12)
    tick_fontsize = max(label_fontsize - 3, 6)

    for ax in figure.get_axes():
        # Re-apply axis label fontsize (corner sometimes misses this)
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize, labelpad=4)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize, labelpad=4)
        # Tick labels: slightly smaller, x-ticks rotated to avoid overlap with labels
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, pad=2)
        ax.tick_params(axis="x", labelrotation=45)
        for t in ax.get_xticklabels():
            t.set_ha("right")

    # Tight panel spacing — avoids the large gaps tight_layout() introduces
    figure.subplots_adjust(hspace=hspace, wspace=wspace)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Corner plot saved to {save_path}")
        plt.show()
    else:
        plt.show()

    return figure


def generate_post_map(
    sim: object,
    Tsys_samples: NDArray[np.floating],
    save_root: str = "outputs/GS1",
) -> None:
    """
    Generate and save posterior sky-map diagnostics.

    Produces four HEALPix gnomonic-projection maps (true sky, posterior mean,
    error = mean - true, and posterior standard deviation) and saves them as
    PDFs to the specified directory.

    Parameters
    ----------
    sim : object
        Simulation object carrying ``pixel_indices``, ``nside``, and
        ``sky_params`` attributes (e.g. :class:`TODSimulation`).
    Tsys_samples : NDArray[np.floating]
        Posterior samples of the system temperature parameters, shape
        ``(n_samples, n_params)``.  The first ``len(pixel_indices)``
        columns correspond to sky temperature pixels.
    save_root : str, optional
        Directory path for output PDFs. Default is ``'outputs/GS1'``.
    """

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

    hp.gnomview(
        patch_true_map,
        rot=(lon_center, lat_center),
        xsize=520,
        ysize=350,
        reso=res,
        title=" ",
        unit="K",
        cmap="jet",  # min=sky_min, max=sky_max,
        # notext=True,
        coord=["C"],
        cbar=True,
        notext=False,
        badcolor="gray",
    )
    # plt.grid(True)
    hp.graticule(dpar=10, dmer=10, coord=["C"], local=True)
    # plt.grid(color='gray', linestyle=':', alpha=0.5)  # Custom grid style
    plt.gca().set_facecolor("gray")  # Set background to white
    plt.savefig(save_root + "/true_map.pdf", bbox_inches="tight", pad_inches=0.1)

    patch_mean_map = view_patch_map(sample_mean_map, pixel_indices)

    hp.gnomview(
        patch_mean_map,
        rot=(lon_center, lat_center),
        xsize=520,
        ysize=350,
        reso=res,
        title=" ",
        unit="K",
        cmap="jet",  # min=sky_min, max=sky_max,
        # notext=True,
        coord=["C"],
        cbar=True,
        notext=False,
        badcolor="gray",
    )
    # plt.grid(True)
    hp.graticule(dpar=10, dmer=10, coord=["C"], local=True)
    # plt.grid(color='gray', linestyle=':', alpha=0.5)  # Custom grid style
    plt.gca().set_facecolor("gray")  # Set background to white
    plt.savefig(save_root + "/mean_map.pdf", bbox_inches="tight", pad_inches=0.1)

    patch_error_map = view_patch_map(sample_mean_map - true_map, pixel_indices)
    # plt.figure(figsize=(10, 6))
    hp.gnomview(
        patch_error_map,
        rot=(lon_center, lat_center),
        xsize=520,
        ysize=350,
        reso=res,
        title=" ",
        unit="K",
        cmap="RdBu_r",  # min=-0.1, max=0.1,
        # notext=True,
        coord=["C"],
        cbar=True,
        notext=False,
        badcolor="gray",
    )
    # plt.grid(True)
    hp.graticule(dpar=10, dmer=10, coord=["C"], local=True)
    # plt.grid(color='gray', linestyle=':', alpha=0.5)  # Custom grid style
    plt.gca().set_facecolor("gray")  # Set background to white
    plt.savefig(save_root + "/error_map.pdf", bbox_inches="tight", pad_inches=0.1)

    patch_std_map = view_patch_map(sample_std_map, pixel_indices)

    hp.gnomview(
        patch_std_map,
        rot=(lon_center, lat_center),
        xsize=520,
        ysize=350,
        reso=res,
        title=None,
        unit="K",
        cmap="jet",
        notext=False,
        coord=["C"],
        cbar=True,
        badcolor="gray",
        # min=0, max=1,
        # norm='log'
        # sub=(2, 1, 1),  # Proper subplot specification
        # margins=(0.05, 0.15, 0.05, 0.15)
    )
    hp.graticule(dpar=10, dmer=10, coord=["C"], local=True)
    plt.gca().set_facecolor("gray")  # Set background to white
    plt.savefig(save_root + "/std_map.pdf", bbox_inches="tight", pad_inches=0.1)


def view_patch_map(
    map: NDArray[np.floating],
    pixel_indices: NDArray[np.integer],
) -> NDArray[np.floating]:
    """
    Mask a full-sky HEALPix map to show only a selected patch.

    Pixels outside the patch are set to ``hp.UNSEEN`` so that HEALPix
    plotting routines render them as the background colour.

    Parameters
    ----------
    map : NDArray[np.floating]
        Full-sky HEALPix map of length ``NPIX``.
    pixel_indices : NDArray[np.integer]
        Indices of the observed pixels to keep.

    Returns
    -------
    patch_only_map : NDArray[np.floating]
        Map with unobserved pixels set to ``hp.UNSEEN``.
    """
    # Create a new map with just the patch (other pixels set to UNSEEN)
    patch_only_map = np.full(len(map), hp.UNSEEN)
    patch_only_map[pixel_indices] = map[pixel_indices]
    return patch_only_map


def gnomview_patch(
    map: NDArray[np.floating],
    pixel_indices: NDArray[np.integer],
    lon_center: float,
    lat_center: float,
    res: float,
    sky_min: float,
    sky_max: float,
    title: str = " ",
    save_path: str | None = None,
    cmap: str = "jet",
    cbar: bool = True,
    xtick: bool = False,
    ytick: bool = False,
    unit: str = "K",
    turn_into_map: bool = True,
    fts: int = 16,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """
    Plot a HEALPix sky patch using gnomonic projection.

    Wraps ``healpy.gnomview`` with additional formatting: graticule lines,
    custom axis labels, colorbar font sizes, and optional tick annotations.

    Parameters
    ----------
    map : NDArray[np.floating]
        Pixel values for the patch (or full-sky map if ``turn_into_map``
        is ``False``).
    pixel_indices : NDArray[np.integer]
        HEALPix pixel indices of the observed patch.
    lon_center : float
        Longitude (RA) of the projection centre in degrees.
    lat_center : float
        Latitude (Dec) of the projection centre in degrees.
    res : float
        Angular resolution of the gnomonic projection in arcmin/pixel.
    sky_min : float
        Minimum value for the colour scale.
    sky_max : float
        Maximum value for the colour scale.
    title : str, optional
        Plot title.  Default is ``' '`` (blank).
    save_path : str or None, optional
        Path to save the figure.
    cmap : str, optional
        Matplotlib colormap name. Default is ``'jet'``.
    cbar : bool, optional
        Whether to display a colorbar. Default is ``True``.
    xtick : bool, optional
        Whether to annotate the x-axis with ``lon_center``.
    ytick : bool, optional
        Whether to annotate the y-axis with ``lat_center``.
    unit : str, optional
        Colorbar unit label. Default is ``'K'``.
    turn_into_map : bool, optional
        If ``True``, *map* is treated as patch values and inserted into a
        full-sky map at *pixel_indices*. Default is ``True``.
    fts : int, optional
        Base font size. Default is 16.
    xlabel : str or None, optional
        Custom x-axis label (overrides ``lon_center`` tick).
    ylabel : str or None, optional
        Custom y-axis label (overrides ``lat_center`` tick).
    """
    NPIX = hp.nside2npix(64)
    if turn_into_map:
        aux_map = np.zeros(NPIX, dtype=float)
        aux_map[pixel_indices] = map
    else:
        aux_map = map
    patch_only_map = view_patch_map(aux_map, pixel_indices)
    hp.gnomview(
        patch_only_map,
        rot=(lon_center, lat_center),
        xsize=520,
        ysize=220,
        reso=res,
        title=title,
        unit=unit,
        cmap=cmap,
        min=sky_min,
        max=sky_max,
        notext=True,
        coord=["C"],
        cbar=cbar,
        badcolor="gray",
    )
    cb = plt.gcf().axes[-1]  # Get the colorbar axis (usually the last one)
    cb.tick_params(labelsize=fts)  # Set the font size to 18 (adjust as needed)
    hp.graticule(
        dpar=10, dmer=10, coord=["C"], local=True
    )  # Add graticule lines; separation in degrees
    plt.gca().set_facecolor("gray")  # Set background to gray

    # Add axis labels using plt.text
    fig = plt.gcf()
    ax = plt.gca()
    if title and title.strip():  # Only if title is not empty
        ax.set_title(title, fontsize=fts - 1, pad=5)

    if cbar:
        if xtick:
            fig.text(0.5, 0.185, str(lon_center)[:7], ha="center", fontsize=fts - 1)
        if ytick:
            fig.text(
                0.045,
                0.37,
                str(lat_center)[:5],
                va="center",
                rotation="vertical",
                fontsize=fts - 1,
            )
        if xlabel is not None:
            fig.text(0.5, 0.155, xlabel, ha="center", fontsize=fts - 1)
        if ylabel is not None:
            fig.text(
                0.01, 0.4, ylabel, va="center", rotation="vertical", fontsize=fts - 1
            )
    else:
        if xtick:
            fig.text(0.5, 0.31, str(lon_center)[:7], ha="center", fontsize=fts - 1)
        if ytick:
            fig.text(
                0.045,
                0.5,
                str(lat_center)[:5],
                va="center",
                rotation="vertical",
                fontsize=fts - 1,
            )
        if xlabel is not None:
            fig.text(0.5, 0.28, xlabel, ha="center", fontsize=fts - 1)
        if ylabel is not None:
            fig.text(
                0.01, 0.5, ylabel, va="center", rotation="vertical", fontsize=fts - 1
            )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    pass


def cartview_patch(
    map_values: NDArray[np.floating],
    pixel_indices: NDArray[np.integer],
    sky_min: float,
    sky_max: float,
    nside: int = 64,
    grid_res: int = 200,
    title: str = " ",
    save_path: str | None = None,
    cmap: str = "jet",
    cbar: bool = True,
    xtick: bool = False,
    ytick: bool = False,
    unit: str = "K",
    turn_into_map: bool = True,
    fts: int = 16,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """
    Plot a HEALPix sky patch in a Cartesian RA/Dec frame using pcolormesh.

    Drop-in replacement for :func:`gnomview_patch` where the projection
    centre and resolution are automatically inferred from *pixel_indices*.
    An optional *figsize* parameter allows matching panel proportions to
    other plots (e.g. scan-pattern diagrams).

    Parameters
    ----------
    map_values : NDArray[np.floating]
        Pixel values for the patch (or full-sky map if ``turn_into_map``
        is ``False``).
    pixel_indices : NDArray[np.integer]
        HEALPix pixel indices of the observed patch.
    sky_min : float
        Minimum value for the colour scale.
    sky_max : float
        Maximum value for the colour scale.
    nside : int, optional
        HEALPix ``NSIDE``. Default is 64.
    grid_res : int, optional
        Number of grid points along each RA/Dec axis. Default is 200.
    title : str, optional
        Plot title. Default is ``' '``.
    save_path : str or None, optional
        Path to save the figure.
    cmap : str, optional
        Matplotlib colormap name. Default is ``'jet'``.
    cbar : bool, optional
        Whether to display a colorbar. Default is ``True``.
    xtick : bool, optional
        Whether to label the x-axis. Default is ``False``.
    ytick : bool, optional
        Whether to label the y-axis. Default is ``False``.
    unit : str, optional
        Colorbar unit label. Default is ``'K'``.
    turn_into_map : bool, optional
        If ``True``, *map_values* contains only patch values to be
        inserted at *pixel_indices*. Default is ``True``.
    fts : int, optional
        Base font size. Default is 16.
    xlabel : str or None, optional
        Custom x-axis label.
    ylabel : str or None, optional
        Custom y-axis label.
    figsize : tuple of float or None, optional
        Figure size ``(width, height)`` in inches.  If ``None``, computed
        automatically from the RA/Dec extent.
    """
    # Build full HEALPix map
    npix = hp.nside2npix(nside)
    if turn_into_map:
        full_map = np.zeros(npix, dtype=float)
        full_map[pixel_indices] = map_values
    else:
        full_map = np.array(map_values, dtype=float)
    # Mask unobserved pixels
    observed = np.zeros(npix, dtype=bool)
    observed[pixel_indices] = True

    # RA/Dec range from observed pixels
    theta_px, phi_px = hp.pix2ang(nside, pixel_indices)
    ra_pix = np.degrees(phi_px)
    dec_pix = 90.0 - np.degrees(theta_px)
    pad = 0.5
    ra_range = (ra_pix.min() - pad, ra_pix.max() + pad)
    dec_range = (dec_pix.min() - pad, dec_pix.max() + pad)

    # Regular RA/Dec grid
    ra_grid = np.linspace(ra_range[0], ra_range[1], grid_res)
    dec_grid = np.linspace(dec_range[0], dec_range[1], grid_res)
    RA, DEC = np.meshgrid(ra_grid, dec_grid)

    # Nearest-neighbour lookup into HEALPix map
    pix_grid = hp.ang2pix(nside, np.radians(90.0 - DEC), np.radians(RA))
    grid_vals = full_map[pix_grid].astype(float)
    grid_vals[~observed[pix_grid]] = np.nan

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.ticker as mticker

    # Width fixed at 10 to match scan-pattern panel; colorbar lives inside the
    # figure via make_axes_locatable so total figure size never changes.
    if figsize is None:
        ra_span = ra_range[1] - ra_range[0]
        dec_span = dec_range[1] - dec_range[0]
        w = 10.0
        h = w * (dec_span / ra_span) * 1.7
        figsize = (w, max(h, 3.0))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    im = ax.pcolormesh(
        RA, DEC, grid_vals, vmin=sky_min, vmax=sky_max, cmap=cmap, shading="auto"
    )
    ax.set_facecolor("lightgray")
    ax.invert_xaxis()

    if title and title.strip():
        ax.set_title(title, fontsize=fts - 1, pad=5)

    if xtick:
        ax.set_xlabel(xlabel if xlabel is not None else "RA (deg)", fontsize=fts + 1)
        ax.tick_params(axis="x", labelsize=fts - 1)
    else:
        ax.set_xticklabels([])

    if ytick:
        ax.set_ylabel(ylabel if ylabel is not None else "Dec (deg)", fontsize=fts + 1)
        ax.tick_params(axis="y", labelsize=fts - 1)
    else:
        ax.set_yticklabels([])

    plt.tight_layout()

    if cbar:
        # Place colorbar manually after layout — full control over length
        fig.canvas.draw()
        ax_pos = ax.get_position()
        cbar_frac = 0.7  # colorbar length as fraction of map width
        cbar_h = 0.09  # colorbar height in figure coordinates
        cbar_w = ax_pos.width * cbar_frac
        cbar_x = ax_pos.x0 + (ax_pos.width - cbar_w) / 2
        cbar_y = ax_pos.y0 - cbar_h - 0.12
        cax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=fts + 13)
        cb.locator = mticker.MaxNLocator(nbins=4)
        cb.update_ticks()
        if unit:
            cax.text(
                1.02,
                0.5,
                unit,
                transform=cax.transAxes,
                va="center",
                ha="left",
                fontsize=fts + 13,
            )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    pass


def cartview_patch_v2(
    map_values: NDArray[np.floating],
    pixel_indices: NDArray[np.integer],
    lon_center: float,
    lat_center: float,
    res: float,
    sky_min: float,
    sky_max: float,
    nside: int = 64,
    grid_res: int = 200,
    title: str = " ",
    save_path: str | None = None,
    cmap: str = "jet",
    cbar: bool = True,
    xtick: bool = False,
    ytick: bool = False,
    unit: str = "K",
    turn_into_map: bool = True,
    fts: int = 16,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """
    Drop-in replacement for :func:`gnomview_patch` using Cartesian projection.

    Accepts the same positional signature as :func:`gnomview_patch`
    (``lon_center``, ``lat_center``, ``res`` are accepted for API
    compatibility but ignored -- the extent is inferred from
    *pixel_indices*).  Delegates to :func:`cartview_patch`.

    Parameters
    ----------
    map_values : NDArray[np.floating]
        Pixel values for the patch.
    pixel_indices : NDArray[np.integer]
        HEALPix pixel indices of the observed patch.
    lon_center : float
        Ignored (kept for API compatibility).
    lat_center : float
        Ignored (kept for API compatibility).
    res : float
        Ignored (kept for API compatibility).
    sky_min : float
        Minimum value for the colour scale.
    sky_max : float
        Maximum value for the colour scale.
    nside : int, optional
        HEALPix ``NSIDE``. Default is 64.
    grid_res : int, optional
        Grid resolution. Default is 200.
    title : str, optional
        Plot title.
    save_path : str or None, optional
        Path to save the figure.
    cmap : str, optional
        Colormap name. Default is ``'jet'``.
    cbar : bool, optional
        Whether to display a colorbar.
    xtick : bool, optional
        Whether to label the x-axis.
    ytick : bool, optional
        Whether to label the y-axis.
    unit : str, optional
        Colorbar unit label.
    turn_into_map : bool, optional
        Whether *map_values* is a patch-only array.
    fts : int, optional
        Base font size.
    xlabel : str or None, optional
        Custom x-axis label.
    ylabel : str or None, optional
        Custom y-axis label.
    figsize : tuple of float or None, optional
        Figure size in inches.
    """
    cartview_patch(
        map_values,
        pixel_indices,
        sky_min,
        sky_max,
        nside=nside,
        grid_res=grid_res,
        title=title,
        save_path=save_path,
        cmap=cmap,
        cbar=cbar,
        xtick=xtick,
        ytick=ytick,
        unit=unit,
        turn_into_map=turn_into_map,
        fts=fts,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
    )


def plot_residual_histogram(
    type_str: str,
    residuals: NDArray[np.floating],
    residuals_common: NDArray[np.floating],
    binwidth: float = 0.05,
    fts: int = 28,  # Increased from 19 to 27
    kde: bool = False,
    save_path: str | None = None,
    figsize: tuple[float, float] = (12, 7),
    show_mean: bool = False,
    show_std: bool = True,
    colors: dict[str, str] | None = None,
    title: str | None = None,
    print_xlabel: bool = False,
    print_ylabel: bool = False,
    xlim: tuple[float, float] | None = None,
    legend: bool = False,
) -> None:
    """
    Plot overlaid histograms of all-pixel and internal-pixel residuals.

    Displays a grey background histogram of all residuals with a red
    overlay for internal (common) residuals, annotated with 16th--84th
    percentile lines and summary statistics.

    Parameters
    ----------
    type_str : str
        Label string displayed in the annotation box (e.g. scan name).
    residuals : NDArray[np.floating]
        All-pixel residual values (plotted in grey).
    residuals_common : NDArray[np.floating]
        Internal-pixel residual values (plotted in red).
    binwidth : float, optional
        Histogram bin width in data units. Default is 0.05.
    fts : int, optional
        Base font size for labels. Default is 28.
    kde : bool, optional
        If ``True``, overlay a KDE curve. Default is ``False``.
    save_path : str or None, optional
        File path to save the figure.
    figsize : tuple of float, optional
        Figure size ``(width, height)``. Default is ``(12, 7)``.
    show_mean : bool, optional
        Whether to show vertical mean lines. Default is ``False``.
    show_std : bool, optional
        Whether to include standard deviation in the annotation text.
        Default is ``True``.
    colors : dict or None, optional
        Custom colour mapping with keys ``'all'``, ``'common'``,
        ``'all_lines'``, ``'common_lines'``, ``'mean_all'``,
        ``'mean_common'``.
    title : str or None, optional
        Plot title.
    print_xlabel : bool, optional
        Whether to print the x-axis label. Default is ``False``.
    print_ylabel : bool, optional
        Whether to print the y-axis label. Default is ``False``.
    xlim : tuple of float or None, optional
        Explicit x-axis limits ``(xmin, xmax)``.
    legend : bool, optional
        Whether to display the legend. Default is ``False``.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    # Default colors
    if colors is None:
        colors = {
            "all": "#7f8c8d",  # Sophisticated gray
            "common": "#e74c3c",  # Red
            "all_lines": "#2980b9",  # Blue
            "common_lines": "#c0392b",  # Dark red
            "mean_all": "#34495e",  # Dark gray
            "mean_common": "#a93226",  # Dark red
        }

    # Set up the plot with better styling
    plt.figure(figsize=figsize)
    sns.set_theme(style="white")  # Changed from 'whitegrid' to 'white' to remove grid

    # Calculate common bin edges for consistent comparison
    all_data = np.concatenate([residuals, residuals_common])
    if xlim is None:
        data_range = np.max(all_data) - np.min(all_data)
        margin = 0.1 * data_range
        xlim = (np.min(all_data) - margin, np.max(all_data) + margin)

    # Calculate bins
    n_bins = int((xlim[1] - xlim[0]) / binwidth)
    bins = np.linspace(xlim[0], xlim[1], n_bins)

    # Plot histogram of all pixels as background
    ax = plt.gca()
    n_all, bins_all, patches_all = plt.hist(
        residuals,
        bins=bins,
        color=colors["all"],
        alpha=0.8,
        density=False,
        edgecolor="white",
        linewidth=0.5,
        label="All pixels",
    )

    # Plot histogram of common residuals on top
    n_common, bins_common, patches_common = plt.hist(
        residuals_common,
        bins=bins,
        color=colors["common"],
        alpha=0.5,
        density=False,
        edgecolor="white",
        linewidth=0.5,
        label="Internal pixels",
    )

    # Add KDE if requested
    if kde:
        from scipy.stats import gaussian_kde

        # KDE for all residuals
        kde_all = gaussian_kde(residuals)
        x_kde = np.linspace(xlim[0], xlim[1], 300)
        y_kde_all = kde_all(x_kde)
        plt.plot(
            x_kde, y_kde_all, color=colors["all"], linestyle="-", linewidth=2, alpha=0.8
        )

        # KDE for common residuals
        kde_common = gaussian_kde(residuals_common)
        y_kde_common = kde_common(x_kde)
        plt.plot(
            x_kde,
            y_kde_common,
            color=colors["common"],
            linestyle="-",
            linewidth=2,
            alpha=0.9,
        )

    # Compute comprehensive statistics
    stats_all = {
        "mean": np.mean(residuals),
        "std": np.std(residuals),
        "median": np.median(residuals),
        "p16": np.percentile(residuals, 16),
        "p84": np.percentile(residuals, 84),
        "rms": np.sqrt(np.mean(residuals**2)),
    }

    stats_common = {
        "mean": np.mean(residuals_common),
        "std": np.std(residuals_common),
        "median": np.median(residuals_common),
        "p16": np.percentile(residuals_common, 16),
        "p84": np.percentile(residuals_common, 84),
        "rms": np.sqrt(np.mean(residuals_common**2)),
    }

    # Print statistics
    print(
        f"All pixels - Mean: {stats_all['mean']:.4f}, Std: {stats_all['std']:.4f}, RMS: {stats_all['rms']:.4f}"
    )
    print(
        f"All pixels - 16th-84th percentile: [{stats_all['p16']:.4f}, {stats_all['p84']:.4f}]"
    )
    print(
        f"Internal pixels - Mean: {stats_common['mean']:.4f}, Std: {stats_common['std']:.4f}, RMS: {stats_common['rms']:.4f}"
    )
    print(
        f"Internal pixels - 16th-84th percentile: [{stats_common['p16']:.4f}, {stats_common['p84']:.4f}]"
    )

    # Add vertical lines for percentiles
    plt.axvline(
        stats_all["p16"],
        color=colors["all_lines"],
        linestyle=":",
        lw=3,
        alpha=0.8,
        label="All: 16th–84th percentile",
    )  # Increased linewidth
    plt.axvline(
        stats_all["p84"], color=colors["all_lines"], linestyle=":", lw=3, alpha=0.8
    )

    plt.axvline(
        stats_common["p16"],
        color=colors["common_lines"],
        linestyle="--",
        lw=3,
        alpha=0.9,
        label="Internal: 16th–84th percentile",
    )  # Increased linewidth
    plt.axvline(
        stats_common["p84"],
        color=colors["common_lines"],
        linestyle="--",
        lw=3,
        alpha=0.9,
    )

    # Add mean lines if requested
    if show_mean:
        plt.axvline(
            stats_all["mean"],
            color=colors["mean_all"],
            linestyle="-",
            lw=3,
            alpha=0.7,
            label="All: Mean",
        )  # Increased linewidth
        plt.axvline(
            stats_common["mean"],
            color=colors["mean_common"],
            linestyle="-",
            lw=3,
            alpha=0.8,
            label="Internal: Mean",
        )  # Increased linewidth

    # Set axis limits
    plt.xlim(xlim)

    # Enhanced annotations with larger font sizes
    if print_xlabel:
        plt.xlabel(
            r"$T_{\mathrm{residual}} = \langle T^{\mathrm{sample}}_{\mathrm{sky}} \rangle - T_{\mathrm{sky}}^{\mathrm{true}}$ [K]",
            fontsize=fts,
        )
    if print_ylabel:
        plt.ylabel("Histogram", fontsize=fts)

    if title:
        plt.title(title, fontsize=fts + 4, pad=20)  # Increased from +2 to +4

    # plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Enhanced text annotation with larger font size
    text_lines = [
        f"{type_str}",
        f"",
        f"All pixels (n={len(residuals)}):",
        f'   16th–84th pct: [{stats_all["p16"]:.3f}, {stats_all["p84"]:.3f}] K',
        f'   Mean ± Std: {stats_all["mean"]:.3f} ± {stats_all["std"]:.3f} K',
        f"",
        f"Internal pixels (n={len(residuals_common)}):",
        f'   16th–84th pct: [{stats_common["p16"]:.3f}, {stats_common["p84"]:.3f}] K',
        f'   Mean ± Std: {stats_common["mean"]:.3f} ± {stats_common["std"]:.3f} K',
    ]

    text_str = "\n".join(text_lines)

    plt.text(
        0.01,
        0.97,
        text_str,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(
            facecolor="white", alpha=0.5, pad=8
        ),  # Reduced opacity and padding, removed frame
        fontsize=fts - 4,
        fontfamily="sans-serif",
        color="#555555",
    )  # Changed to sans-serif font and gray text color

    # Enhanced legend with larger font size
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate labels
    by_label = dict(zip(labels, handles))
    if legend:
        plt.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper right",
            fontsize=fts - 3,  # Changed from fts-3 to fts-4, but still larger
            frameon=False,  # framealpha=0.9, edgecolor='gray'
        )

    # Set tick parameters with larger font sizes
    plt.tick_params(
        axis="both", which="major", labelsize=fts - 2
    )  # Increased tick label size
    plt.tick_params(axis="both", which="minor", labelsize=fts - 4)

    plt.tight_layout()

    # Save plot if save_path is provided
    if save_path is not None:
        plt.savefig(
            save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Plot saved to: {save_path}")

    plt.show()

    # Return statistics for further analysis
    pass

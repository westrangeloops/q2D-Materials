#!/usr/bin/env python3
"""
Publication-quality plotting style module.
Provides consistent styling for all plots in the Examples directory.
Based on plot_ldos_publication.py style.
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Color schemes
HALOGEN_COLORS = {
    "I": "#6B9BD3",   # Blue for Iodine
    "Br": "#E7845F",  # Orange for Bromine
    "Cl": "#4C8B68",  # Green for Chlorine
}
ORGANIC_COLOR = "#A58DC1"  # Purple for Molecule

# Vibrant LDOS colormap colors (for heatmaps)
LDOS_COLORS = [
    '#020F31',  # Dark blue (matches background, zero DOS)
    '#0d47a1',  # Deep blue
    '#1565c0',  # Blue
    '#1976d2',  # Bright blue
    '#1e88e5',  # Light blue
    '#42a5f5',  # Sky blue
    '#64b5f6',  # Light sky blue
    '#90caf9',  # Pale blue
    '#bbdefb',  # Very pale blue
    '#e3f2fd',  # Almost white blue
    '#fff9c4',  # Light yellow
    '#ffeb3b',  # Yellow
    '#ffc107',  # Amber
    '#ff9800',  # Orange
    '#ff6f00',  # Deep orange
    '#ff5722',  # Red-orange
    '#f44336',  # Red
    '#d32f2f'   # Deep red
]

# Default DPI for publication quality
PUBLICATION_DPI = 600

# Standard figure sizes
FIGURE_SIZES = {
    'standard': (10, 6),
    'wide': (14, 6),
    'tall': (8, 10),
    'square': (8, 8),
    'large': (12, 8),
}


def apply_publication_style(dpi=None, font_size=None):
    """
    Apply publication-quality styling to matplotlib.
    
    Parameters
    ----------
    dpi : int, optional
        DPI setting (default: PUBLICATION_DPI = 600)
    font_size : int, optional
        Base font size (default: 16)
    
    Returns
    -------
    dict
        Dictionary of applied settings for reference
    """
    if dpi is None:
        dpi = PUBLICATION_DPI
    
    if font_size is None:
        font_size = 16
    
    # Font configuration - sans-serif bold fonts (Inter-like)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    
    # Font sizes scaled for high DPI
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.titlesize'] = font_size + 2
    plt.rcParams['axes.labelsize'] = font_size
    plt.rcParams['xtick.labelsize'] = font_size - 2
    plt.rcParams['ytick.labelsize'] = font_size - 2
    plt.rcParams['legend.fontsize'] = font_size - 2
    plt.rcParams['figure.titlesize'] = font_size + 4
    
    # Line and marker settings
    plt.rcParams['lines.linewidth'] = 3.0
    plt.rcParams['lines.markersize'] = 8.0
    plt.rcParams['axes.linewidth'] = 2.0
    
    # Grid settings
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linewidth'] = 1.5
    
    # Tick settings
    plt.rcParams['xtick.major.width'] = 2.0
    plt.rcParams['ytick.major.width'] = 2.0
    plt.rcParams['xtick.minor.width'] = 1.5
    plt.rcParams['ytick.minor.width'] = 1.5
    
    # Save settings
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    
    return {
        'dpi': dpi,
        'font_size': font_size,
        'font_family': plt.rcParams['font.family'],
        'figure_dpi': plt.rcParams['savefig.dpi']
    }


def get_ldos_colormap():
    """
    Get the vibrant LDOS colormap for heatmaps.
    
    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Custom colormap for LDOS visualization
    """
    return LinearSegmentedColormap.from_list('vibrant_ldos', LDOS_COLORS, N=512)


def style_axes(ax, spine_color='black', spine_width=2.0, tick_color='black'):
    """
    Apply consistent styling to axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to style
    spine_color : str, default='black'
        Color for axis spines (borders)
    spine_width : float, default=2.0
        Width of axis spines
    tick_color : str, default='black'
        Color for tick labels and marks
    """
    # Set spine colors and widths
    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(spine_width)
        spine.set_visible(True)
    
    # Set tick colors
    ax.tick_params(colors=tick_color, width=spine_width)


def create_figure_with_style(figsize='standard', dpi=None, title=None, title_x=0.025):
    """
    Create a figure with publication-quality styling.
    
    Parameters
    ----------
    figsize : str or tuple, default='standard'
        Figure size key or (width, height) tuple
    dpi : int, optional
        DPI setting (default: PUBLICATION_DPI)
    title : str, optional
        Figure title
    title_x : float, default=0.025
        X position for title (0-1, left-aligned)
    
    Returns
    -------
    matplotlib.figure.Figure
        Styled figure object
    """
    if dpi is None:
        dpi = PUBLICATION_DPI
    
    # Get figure size
    if isinstance(figsize, str):
        size = FIGURE_SIZES.get(figsize, FIGURE_SIZES['standard'])
    else:
        size = figsize
    
    # Apply style
    apply_publication_style(dpi=dpi)
    
    # Create figure
    fig = plt.figure(figsize=size, dpi=dpi)
    
    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=plt.rcParams['figure.titlesize'], 
                    fontweight='bold', y=0.995, x=title_x, ha='left')
    
    return fig


def save_figure(fig, path, dpi=None, **kwargs):
    """
    Save figure with publication-quality settings.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    path : str or Path
        Output path
    dpi : int, optional
        DPI for saving (default: PUBLICATION_DPI)
    **kwargs
        Additional arguments passed to plt.savefig()
    """
    if dpi is None:
        dpi = PUBLICATION_DPI
    
    # Ensure output directory exists
    from pathlib import Path
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with high DPI
    fig.savefig(str(output_path), dpi=dpi, bbox_inches='tight', 
                pad_inches=0.1, **kwargs)
    plt.close(fig)


# Apply style by default when module is imported
apply_publication_style()


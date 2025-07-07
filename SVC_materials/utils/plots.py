"""
Scientific plotting utilities with beautiful design principles.
Inspired by Federica Fragapane's approach to elegant data visualization.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Rectangle
import seaborn as sns

# Set global style for beautiful scientific plots
plt.style.use('default')  # Start with clean default
sns.set_palette("husl")  # Use beautiful color palette

# Define elegant color palette
ELEMENT_COLORS = {
    'H': '#FF6B6B',   # Soft red
    'C': '#4ECDC4',   # Teal
    'N': '#45B7D1',   # Sky blue
    'O': '#F7DC6F',   # Soft yellow
    'Pb': '#BB8FCE',  # Soft purple
    'I': '#F8C471',   # Soft orange
    'Cl': '#82E0AA', # Soft green
    'Br': '#F1948A', # Soft pink
    'S': '#85C1E9',  # Light blue
    'P': '#D7BDE2',  # Lavender
}

def get_element_color(element):
    """Get a beautiful color for an element."""
    return ELEMENT_COLORS.get(element, '#95A5A6')  # Default soft gray

def setup_beautiful_plot(figsize=(12, 8), dpi=300):
    """
    Set up a beautiful plot with Federica Fragapane-inspired aesthetics.
    
    Parameters:
    figsize (tuple): Figure size in inches
    dpi (int): Resolution for saved plots
    
    Returns:
    fig, ax: matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Clean white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make remaining spines lighter
    ax.spines['left'].set_color('#E0E0E0')
    ax.spines['bottom'].set_color('#E0E0E0')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Light grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#E0E0E0')
    ax.set_axisbelow(True)
    
    return fig, ax

def add_elegant_title(ax, title, subtitle=None, element=None):
    """
    Add an elegant title with proper typography.
    
    Parameters:
    ax: matplotlib axis
    title (str): Main title
    subtitle (str, optional): Subtitle
    element (str, optional): Element symbol for color accent
    """
    # Main title
    title_color = get_element_color(element) if element else '#2C3E50'
    ax.text(0.02, 0.98, title, transform=ax.transAxes, 
            fontsize=16, fontweight='bold', color=title_color,
            verticalalignment='top', horizontalalignment='left')
    
    # Subtitle
    if subtitle:
        ax.text(0.02, 0.94, subtitle, transform=ax.transAxes, 
                fontsize=12, color='#7F8C8D', style='italic',
                verticalalignment='top', horizontalalignment='left')

def add_stats_box(ax, stats_dict, position='top_right'):
    """
    Add a beautiful statistics box to the plot.
    
    Parameters:
    ax: matplotlib axis
    stats_dict (dict): Dictionary of statistics to display
    position (str): Position of the box ('top_right', 'top_left', etc.)
    """
    # Position mapping
    positions = {
        'top_right': (0.98, 0.98, 'right', 'top'),
        'top_left': (0.02, 0.98, 'left', 'top'),
        'bottom_right': (0.98, 0.02, 'right', 'bottom'),
        'bottom_left': (0.02, 0.02, 'left', 'bottom'),
    }
    
    x, y, ha, va = positions.get(position, positions['top_right'])
    
    # Format statistics text
    stats_text = []
    for key, value in stats_dict.items():
        if isinstance(value, float):
            stats_text.append(f"{key}: {value:.3f}")
        else:
            stats_text.append(f"{key}: {value}")
    
    # Create text box
    text_str = '\n'.join(stats_text)
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor='white', 
                     edgecolor='#BDC3C7', alpha=0.9, linewidth=1)
    
    ax.text(x, y, text_str, transform=ax.transAxes,
            verticalalignment=va, horizontalalignment=ha,
            bbox=bbox_props, fontsize=10, color='#2C3E50')

def plot_gaussian_projection(z_range, kernel_density, z_coords, element, 
                           structure_name, c_vector_length, sigma, 
                           fitted_gaussian=None, fit_params=None, ionic_radius=None):
    """
    Create a beautiful gaussian projection plot.
    
    Parameters:
    z_range (array): Z-coordinate range for plotting
    kernel_density (array): Gaussian kernel density values
    z_coords (array): Individual atom z-coordinates
    element (str): Element symbol
    structure_name (str): Name of the structure
    c_vector_length (float): Length of c-axis
    sigma (float): Kernel sigma value
    fitted_gaussian (array, optional): Fitted gaussian curve
    fit_params (tuple, optional): Fitting parameters (amp, mean, std)
    ionic_radius (float, optional): Ionic radius of the element
    
    Returns:
    fig: matplotlib figure object
    """
    # Set up beautiful plot
    fig, ax = setup_beautiful_plot()
    
    # Element color
    element_color = get_element_color(element)
    
    # Plot individual atom positions as elegant vertical lines
    for z_pos in z_coords:
        ax.axvline(x=z_pos, color=element_color, alpha=0.2, linewidth=0.8)
    
    # Plot gaussian kernel density with gradient-like fill
    ax.fill_between(z_range, kernel_density, alpha=0.3, color=element_color, 
                   label=f'{element} density distribution')
    ax.plot(z_range, kernel_density, color=element_color, linewidth=2.5,
           label=f'{element} gaussian kernel')
    
    # Plot fitted gaussian if available
    if fitted_gaussian is not None and fit_params is not None:
        ax.plot(z_range, fitted_gaussian, color='#E74C3C', linewidth=2, 
               linestyle='--', alpha=0.8, label='Fitted gaussian')
    
    # Plot atom positions as elegant scatter points
    y_positions = np.interp(z_coords, z_range, kernel_density)
    ax.scatter(z_coords, y_positions, color=element_color, s=60, alpha=0.8,
              edgecolors='white', linewidth=1.5, zorder=5,
              label=f'{element} atoms ({len(z_coords)})')
    
    # Elegant title and subtitle
    main_title = f'Gaussian Projection Analysis • {element}'
    if ionic_radius is not None:
        subtitle = f'Structure: {structure_name} • c-axis: {c_vector_length:.2f} Å • r_ionic: {ionic_radius:.2f} Å'
    else:
        subtitle = f'Structure: {structure_name} • c-axis: {c_vector_length:.2f} Å'
    add_elegant_title(ax, main_title, subtitle, element)
    
    # Beautiful labels
    ax.set_xlabel('Distance along c-axis (Å)', fontsize=12, color='#2C3E50', fontweight='500')
    ax.set_ylabel('Gaussian Density', fontsize=12, color='#2C3E50', fontweight='500')
    
    # Statistics box
    stats_dict = {
        'Atoms': len(z_coords),
        'Z-range': f'{z_coords.min():.2f} – {z_coords.max():.2f} Å',
        'Peak density': f'{kernel_density.max():.2f}'
    }
    
    if ionic_radius is not None:
        base_width = 6 * sigma  # Base width = 2 × ionic_radius
        stats_dict['Ionic radius'] = f'{ionic_radius:.3f} Å'
        stats_dict['Kernel σ'] = f'{sigma:.3f} Å'
        stats_dict['Base width'] = f'{base_width:.3f} Å'
    else:
        stats_dict['Kernel σ'] = f'{sigma:.3f} Å'
    
    if fit_params is not None:
        stats_dict['Fitted μ'] = f'{fit_params[1]:.2f} Å'
        stats_dict['Fitted σ'] = f'{fit_params[2]:.2f} Å'
    
    add_stats_box(ax, stats_dict, 'top_right')
    
    # Beautiful legend
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, 
                      shadow=False, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#BDC3C7')
    legend.get_frame().set_linewidth(1)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    return fig

def plot_multi_element_comparison(element_data_dict, structure_name, c_vector_length, 
                                z_range, sigma=None):
    """
    Create a beautiful multi-element comparison plot.
    
    Parameters:
    element_data_dict (dict): Dictionary with element data
    structure_name (str): Name of the structure
    c_vector_length (float): Length of c-axis
    z_range (array): Z-coordinate range
    sigma (float, optional): Global sigma value (if None, uses individual sigmas)
    
    Returns:
    fig: matplotlib figure object
    """
    fig, ax = setup_beautiful_plot(figsize=(14, 10))
    
    # Plot each element with slight vertical offset for clarity
    y_offset = 0
    max_density = 0
    
    for i, (element, data) in enumerate(element_data_dict.items()):
        kernel_density = data['kernel_density']
        z_coords = data['z_coords']
        element_color = get_element_color(element)
        
        # Get ionic radius info if available
        ionic_radius = data.get('ionic_radius', None)
        element_sigma = data.get('sigma', sigma)
        
        # Offset for visual separation
        offset_density = kernel_density + y_offset
        
        # Create label with ionic radius info
        if ionic_radius is not None:
            label = f'{element} ({len(z_coords)} atoms, r_ionic={ionic_radius:.2f} Å)'
        else:
            label = f'{element} ({len(z_coords)} atoms)'
        
        # Fill area
        ax.fill_between(z_range, y_offset, offset_density, alpha=0.4, 
                       color=element_color, label=label)
        
        # Main line
        ax.plot(z_range, offset_density, color=element_color, linewidth=2.5)
        
        # Atom positions
        y_positions = np.interp(z_coords, z_range, offset_density)
        ax.scatter(z_coords, y_positions, color=element_color, s=40, alpha=0.8,
                  edgecolors='white', linewidth=1, zorder=5)
        
        # Update offset and max density for next element
        y_offset += kernel_density.max() * 1.2
        max_density = max(max_density, offset_density.max())
    
    # Elegant title
    main_title = f'Multi-Element Gaussian Projection'
    subtitle = f'Structure: {structure_name} • c-axis: {c_vector_length:.2f} Å • base width = 2 × ionic radius'
    add_elegant_title(ax, main_title, subtitle)
    
    # Labels
    ax.set_xlabel('Distance along c-axis (Å)', fontsize=12, color='#2C3E50', fontweight='500')
    ax.set_ylabel('Gaussian Density (stacked)', fontsize=12, color='#2C3E50', fontweight='500')
    
    # Statistics box
    total_atoms = sum(len(data['z_coords']) for data in element_data_dict.values())
    stats_dict = {
        'Total atoms': total_atoms,
        'Elements': len(element_data_dict),
        'Base width': '2 × ionic radius',
    }
    
    # Add average ionic radius info
    ionic_radii = [data.get('ionic_radius', 0) for data in element_data_dict.values() if data.get('ionic_radius')]
    if ionic_radii:
        avg_ionic_radius = np.mean(ionic_radii)
        stats_dict['Avg r_ionic'] = f'{avg_ionic_radius:.3f} Å'
    
    add_stats_box(ax, stats_dict, 'top_right')
    
    # Beautiful legend
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, 
                      shadow=False, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#BDC3C7')
    legend.get_frame().set_linewidth(1)
    
    plt.tight_layout()
    return fig

def save_beautiful_plot(fig, filepath, dpi=300):
    """
    Save a plot with beautiful settings.
    
    Parameters:
    fig: matplotlib figure object
    filepath (str): Path to save the plot
    dpi (int): Resolution for saved plot
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save with high quality
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none',
                pad_inches=0.2, format='png')
    
    # Close figure to free memory
    plt.close(fig)

def create_summary_plot(structures_summary, output_path):
    """
    Create a beautiful summary plot comparing all structures.
    
    Parameters:
    structures_summary (dict): Summary data for all structures
    output_path (str): Path to save the summary plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    
    # Main title
    fig.suptitle('Gaussian Projection Analysis Summary', 
                fontsize=20, fontweight='bold', color='#2C3E50', y=0.95)
    
    # Plot 1: Atom counts per structure
    ax1 = axes[0, 0]
    structures = list(structures_summary.keys())
    total_atoms = [structures_summary[s]['total_atoms'] for s in structures]
    
    bars = ax1.bar(structures, total_atoms, color=['#3498DB', '#E74C3C', '#2ECC71'])
    ax1.set_title('Total Atoms per Structure', fontweight='bold', color='#2C3E50')
    ax1.set_ylabel('Number of Atoms')
    
    # Add value labels on bars
    for bar, count in zip(bars, total_atoms):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Element distribution
    ax2 = axes[0, 1]
    all_elements = set()
    for s in structures_summary.values():
        all_elements.update(s['elements'].keys())
    
    element_counts = {elem: [structures_summary[s]['elements'].get(elem, 0) 
                            for s in structures] for elem in all_elements}
    
    x = np.arange(len(structures))
    width = 0.8 / len(all_elements)
    
    for i, (elem, counts) in enumerate(element_counts.items()):
        ax2.bar(x + i * width, counts, width, label=elem, 
               color=get_element_color(elem), alpha=0.8)
    
    ax2.set_title('Element Distribution', fontweight='bold', color='#2C3E50')
    ax2.set_ylabel('Number of Atoms')
    ax2.set_xticks(x + width * (len(all_elements) - 1) / 2)
    ax2.set_xticklabels(structures)
    ax2.legend()
    
    # Plot 3: C-axis lengths
    ax3 = axes[1, 0]
    c_lengths = [structures_summary[s]['c_axis_length'] for s in structures]
    
    bars = ax3.bar(structures, c_lengths, color=['#9B59B6', '#F39C12', '#1ABC9C'])
    ax3.set_title('C-axis Lengths', fontweight='bold', color='#2C3E50')
    ax3.set_ylabel('Length (Å)')
    
    for bar, length in zip(bars, c_lengths):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{length:.1f} Å', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Z-ranges
    ax4 = axes[1, 1]
    z_ranges = [structures_summary[s]['z_range'] for s in structures]
    
    bars = ax4.bar(structures, z_ranges, color=['#E67E22', '#95A5A6', '#34495E'])
    ax4.set_title('Z-coordinate Ranges', fontweight='bold', color='#2C3E50')
    ax4.set_ylabel('Range (Å)')
    
    for bar, z_range in zip(bars, z_ranges):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{z_range:.1f} Å', ha='center', va='bottom', fontweight='bold')
    
    # Style all subplots
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E0E0E0')
        ax.spines['bottom'].set_color('#E0E0E0')
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    save_beautiful_plot(fig, output_path)
    
    return fig

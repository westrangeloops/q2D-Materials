import json
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for high-quality plots (following Federica Fragapane design principles)
plt.style.use('default')
sns.set_palette("husl")

def extract_lattice_data(base_dir):
    """
    Extract lattice parameters and penetration data from all JSON files in the analysis directories.
    
    Returns:
        pandas.DataFrame: Combined data with lattice parameters, halogen type, octahedra count, and penetration data
    """
    data_list = []
    
    # Find all analysis directories
    analysis_dirs = [d for d in os.listdir(base_dir) if d.endswith('_analysis')]
    
    print(f"Found {len(analysis_dirs)} analysis directories")
    
    for dir_name in analysis_dirs:
        try:
            # Extract halogen and n from directory name
            parts = dir_name.split('_')
            material = parts[0]  # e.g., MAPbBr3, MAPbCl3, MAPbI3
            n_value = parts[1]   # e.g., n1, n2, n3
            
            # Extract halogen from material name
            if 'Br3' in material:
                halogen = 'Br'
            elif 'Cl3' in material:
                halogen = 'Cl'
            elif 'I3' in material:
                halogen = 'I'
            else:
                continue
                
            # Extract n value
            n_octahedra = int(n_value[1:])  # Remove 'n' prefix
            
            # Look for the lattice parameters JSON file
            lattice_json_file = None
            penetration_json_file = None
            dir_path = os.path.join(base_dir, dir_name)
            
            for file in os.listdir(dir_path):
                if file.endswith('_layers_ontology.json'):
                    lattice_json_file = os.path.join(dir_path, file)
                elif file.endswith('_penetration_analysis.json'):
                    penetration_json_file = os.path.join(dir_path, file)
            
            if lattice_json_file is None:
                continue
                
            # Read lattice parameters JSON data
            with open(lattice_json_file, 'r') as f:
                lattice_data = json.load(f)
            
            # Extract lattice parameters
            lattice_params = lattice_data['cell_properties']['lattice_parameters']
            composition = lattice_data['cell_properties']['composition']
            
            # Verify number of octahedra matches
            actual_octahedra = composition['number_of_octahedra']
            
            # Extract penetration data if available
            low_to_high_plane = np.nan
            if penetration_json_file is not None:
                try:
                    with open(penetration_json_file, 'r') as f:
                        pen_data = json.load(f)
                    
                    penetration_analysis = pen_data.get('penetration_analysis', {})
                    if 'error' not in penetration_analysis:
                        # Get the first available molecule's low_plane_to_high_plane distance
                        for mol_key in penetration_analysis.keys():
                            if mol_key.startswith('molecule_'):
                                mol_data = penetration_analysis[mol_key]
                                if mol_data and 'penetration_segments' in mol_data:
                                    segments = mol_data['penetration_segments']
                                    if segments and 'low_plane_to_high_plane' in segments:
                                        low_to_high_plane = segments['low_plane_to_high_plane']
                                        break  # Use the first valid measurement
                except Exception as e:
                    print(f"Error reading penetration data for {dir_name}: {e}")
            
            # Create data entry
            entry = {
                'experiment': dir_name,
                'halogen_X': halogen,
                'n_octahedra': n_octahedra,
                'actual_octahedra': actual_octahedra,
                'A': lattice_params['A'],
                'B': lattice_params['B'], 
                'C': lattice_params['C'],
                'Alpha': lattice_params['Alpha'],
                'Beta': lattice_params['Beta'],
                'Gamma': lattice_params['Gamma'],
                'cell_volume': lattice_data['cell_properties']['structure_info']['cell_volume'],
                'low_to_high_plane': low_to_high_plane
            }
            
            data_list.append(entry)
            
        except Exception as e:
            print(f"Error processing {dir_name}: {e}")
            continue
    
    df = pd.DataFrame(data_list)
    
    # Calculate derived parameters
    df['a_over_b'] = df['A'] / df['B']
    df['alpha_over_beta'] = df['Alpha'] / df['Beta']
    
    print(f"Successfully processed {len(df)} experiments")
    print(f"Penetration data available for {len(df[~df['low_to_high_plane'].isna()])} experiments")
    
    return df

def create_halogen_specific_plots(df, output_dir):
    """
    Create specific plots for each halogen with the requested layout:
    Row 1: a/b ratio, Row 2: alpha/beta ratio, Row 3: Low→High Plane distance
    Columns: n=1, n=2, n=3
    """
    
    # Set up the plotting style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 300
    })
    
    # Define color palette for n values
    n_colors = {
        1: '#FF6B6B',  # Red
        2: '#4ECDC4',  # Teal
        3: '#45B7D1'   # Blue
    }
    
    # Define halogen colors for titles
    halogen_colors = {
        'Cl': '#2E8B57',  # Sea green
        'Br': '#FF6B35',  # Orange-red  
        'I': '#4A90E2'    # Blue
    }
    
    # Define the parameters to plot
    parameters = [
        ('a_over_b', 'a/b Ratio', ''),
        ('alpha_over_beta', 'α/β Ratio', ''),
        ('low_to_high_plane', 'Low → High Plane Distance', 'Å')
    ]
    
    # Get unique n values
    n_values = sorted(df['n_octahedra'].unique())
    
    # Create plots for each halogen
    for halogen in ['Cl', 'Br', 'I']:
        halogen_data = df[df['halogen_X'] == halogen]
        
        if len(halogen_data) == 0:
            print(f"No data found for {halogen}")
            continue
            
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'{halogen} - Lattice and Penetration Analysis\nRows: a/b, α/β, Low→High Plane | Columns: n=1, n=2, n=3', 
                     fontsize=16, fontweight='bold', y=0.95, color=halogen_colors[halogen])
        
        for row_idx, (param, param_title, unit) in enumerate(parameters):
            for col_idx, n_val in enumerate(n_values):
                ax = axes[row_idx, col_idx]
                
                # Get data for this specific n value and halogen
                subset = halogen_data[halogen_data['n_octahedra'] == n_val]
                
                if len(subset) == 0 or (param == 'low_to_high_plane' and subset[param].isna().all()):
                    # No data available
                    ax.text(0.5, 0.5, 'No Data\nAvailable', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=14, style='italic',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                    ax.set_title(f'{param_title}\nn={n_val}', fontweight='bold', pad=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # For penetration data, filter out NaN values
                if param == 'low_to_high_plane':
                    plot_data = subset[param].dropna().values
                else:
                    plot_data = subset[param].values
                
                if len(plot_data) == 0:
                    ax.text(0.5, 0.5, 'No Data\nAvailable', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=14, style='italic',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                    ax.set_title(f'{param_title}\nn={n_val}', fontweight='bold', pad=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # Create violin plot with box plot overlay
                violin_parts = ax.violinplot([plot_data], positions=[1], widths=0.6, showmeans=True)
                
                # Style the violin plot
                for pc in violin_parts['bodies']:
                    pc.set_facecolor(n_colors[n_val])
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('black')
                
                # Style other violin plot elements
                for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                    if partname in violin_parts:
                        violin_parts[partname].set_edgecolor('black')
                        violin_parts[partname].set_linewidth(1.5)
                
                # Add box plot overlay for quartiles
                box_plot = ax.boxplot([plot_data], positions=[1], widths=0.3, 
                                     patch_artist=True, 
                                     boxprops=dict(facecolor='white', alpha=0.8, linewidth=1.5),
                                     medianprops=dict(color='red', linewidth=2))
                
                # Set labels and title
                ax.set_title(f'{param_title}\nn={n_val}', fontweight='bold', pad=10)
                
                # Set y-axis label with unit
                if unit:
                    ax.set_ylabel(f'{param_title} ({unit})', fontweight='bold')
        else:
                    ax.set_ylabel(f'{param_title}', fontweight='bold')
            
                ax.set_xticks([1])
                ax.set_xticklabels([f'{halogen}'])
        ax.grid(True, alpha=0.3)
        
                # Add statistical information
                if len(plot_data) > 0:
                    mean_val = np.mean(plot_data)
                    std_val = np.std(plot_data)
                    n_samples = len(plot_data)
                    
                    # Add text box with statistics
                    stats_text = f'n={n_samples}\nmean={mean_val:.3f}\nstd={std_val:.3f}'
                    ax.text(0.02, 0.98, stats_text, 
                           transform=ax.transAxes, va='top', ha='left', 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=9, family='monospace')
                
                # Set appropriate y-axis limits
                if param == 'low_to_high_plane':
                    ax.set_ylim(0, 25)  # Up to 25 Å for plane distance
                elif param in ['a_over_b', 'alpha_over_beta']:
                    # Set reasonable limits for ratios
                    data_min, data_max = np.min(plot_data), np.max(plot_data)
                    margin = (data_max - data_min) * 0.1
                    ax.set_ylim(max(0.8, data_min - margin), data_max + margin)
    
    plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{halogen}_lattice_penetration_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
        print(f"Created analysis plot for {halogen}")
    
    # Statistical summary
    print("\n" + "="*80)
    print("LATTICE AND PENETRATION ANALYSIS SUMMARY")
    print("="*80)
    
    for halogen in ['Cl', 'Br', 'I']:
        halogen_data = df[df['halogen_X'] == halogen]
        if len(halogen_data) == 0:
            continue
            
        print(f"\n{halogen} Analysis:")
        print("-" * 40)
        
        for n_val in sorted(halogen_data['n_octahedra'].unique()):
            subset = halogen_data[halogen_data['n_octahedra'] == n_val]
            print(f"\n  n={n_val} (samples={len(subset)}):")
            
            # a/b ratio
            if len(subset) > 0:
                mean_ab = subset['a_over_b'].mean()
                std_ab = subset['a_over_b'].std()
                print(f"    a/b ratio: {mean_ab:.3f} ± {std_ab:.3f}")
            
            # alpha/beta ratio
            if len(subset) > 0:
                mean_alpha_beta = subset['alpha_over_beta'].mean()
                std_alpha_beta = subset['alpha_over_beta'].std()
                print(f"    α/β ratio: {mean_alpha_beta:.3f} ± {std_alpha_beta:.3f}")
            
            # Low to high plane distance
            penetration_data = subset['low_to_high_plane'].dropna()
            if len(penetration_data) > 0:
                mean_pen = penetration_data.mean()
                std_pen = penetration_data.std()
                print(f"    Low→High plane: {mean_pen:.3f} ± {std_pen:.3f} Å (n={len(penetration_data)})")
            else:
                print(f"    Low→High plane: No data available")

def main():
    # Set base directory
    base_dir = '/home/dotempo/Documents/REPO/SVC-Materials/Graphs/BULKS_RESULTS'
    output_dir = os.path.join(base_dir, 'lattice_parameter_analysis')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting lattice and penetration analysis...")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    # Extract data
    df = extract_lattice_data(base_dir)
    
    if df.empty:
        print("No data found!")
        return
    
    # Data processing complete
    print(f"Processed {len(df)} experiments")
    
    # Create plots
    print("Creating visualizations...")
    create_halogen_specific_plots(df, output_dir)
    
    # Save the dataframe for further analysis
    df.to_csv(os.path.join(output_dir, 'lattice_penetration_data.csv'), index=False)
    
    print(f"\nAnalysis complete! Check the '{output_dir}' directory for results.")
    print("Generated files:")
    print("- Cl_lattice_penetration_analysis.png")
    print("- Br_lattice_penetration_analysis.png") 
    print("- I_lattice_penetration_analysis.png")
    print("- lattice_penetration_data.csv")

if __name__ == "__main__":
    main() 
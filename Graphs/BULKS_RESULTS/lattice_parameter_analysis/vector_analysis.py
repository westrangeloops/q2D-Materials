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

# Set up matplotlib for high-quality plots
plt.style.use('default')
sns.set_palette("husl")

def extract_vector_data(base_dir):
    """
    Extract vector analysis data from all JSON files in the analysis directories.
    
    Returns:
        pandas.DataFrame: Combined data with vector analysis results
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
            
            # Look for the vector analysis JSON file
            vector_file = None
            dir_path = os.path.join(base_dir, dir_name)
            for file in os.listdir(dir_path):
                if file.endswith('_vector_analysis.json'):
                    vector_file = os.path.join(dir_path, file)
                    break
            
            if vector_file is None:
                continue
                
            # Read JSON data
            with open(vector_file, 'r') as f:
                data = json.load(f)
            
            # Extract vector analysis data
            vector_results = data['vector_analysis']['vector_analysis_results']
            low_plane = data['vector_analysis']['plane_analysis']['low_plane']
            high_plane = data['vector_analysis']['plane_analysis']['high_plane']
            
            # Create data entry
            entry = {
                'experiment': dir_name,
                'halogen_X': halogen,
                'n_octahedra': n_octahedra,
                'angle_between_planes': vector_results['angle_between_planes_degrees'],
                'distance_between_planes': vector_results['distance_between_plane_centers_angstrom'],
                'low_plane_z_angle': vector_results['angle_between_low_plane_and_z'],
                'high_plane_z_angle': vector_results['angle_between_high_plane_and_z'],
                'total_atoms': data['salt_structure_info']['total_atoms'],
                'low_plane_normal_x': low_plane['normal_vector'][0],
                'low_plane_normal_y': low_plane['normal_vector'][1],
                'low_plane_normal_z': low_plane['normal_vector'][2],
                'high_plane_normal_x': high_plane['normal_vector'][0],
                'high_plane_normal_y': high_plane['normal_vector'][1],
                'high_plane_normal_z': high_plane['normal_vector'][2]
            }
            
            data_list.append(entry)
            
        except Exception as e:
            print(f"Error processing {dir_name}: {e}")
            continue
    
    df = pd.DataFrame(data_list)
    print(f"Successfully processed {len(df)} experiments")
    return df

def create_vector_analysis_plots(df, output_dir):
    """
    Create comprehensive vector analysis visualizations
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
    
    # Define color palettes
    halogen_colors = {
        'Cl': '#2E8B57',  # Sea green
        'Br': '#FF6B35',  # Orange-red  
        'I': '#4A90E2'    # Blue
    }
    
    n_colors = {
        1: '#FF6B6B',  # Red
        2: '#4ECDC4',  # Teal
        3: '#45B7D1'   # Blue
    }
    
    # 1. Box plots for plane angles vs z-axis by halogen
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Plane Orientation Analysis by Halogen Type\nAngles vs Z-axis and Inter-plane Relationships', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Low plane angles
    ax = axes[0]
    box_plot = ax.boxplot([df[df['halogen_X'] == halogen]['low_plane_z_angle'].values for halogen in ['Cl', 'Br', 'I']], 
                         labels=['Cl', 'Br', 'I'], patch_artist=True)
    for patch, halogen in zip(box_plot['boxes'], ['Cl', 'Br', 'I']):
        patch.set_facecolor(halogen_colors[halogen])
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_plot[element], color='black', linewidth=1)
    ax.set_title('Low Plane Angle vs Z-axis', fontweight='bold')
    ax.set_ylabel('Angle (°)', fontweight='bold')
    ax.set_xlabel('Halogen', fontweight='bold')
    
    # High plane angles  
    ax = axes[1]
    box_plot = ax.boxplot([df[df['halogen_X'] == halogen]['high_plane_z_angle'].values for halogen in ['Cl', 'Br', 'I']], 
                         labels=['Cl', 'Br', 'I'], patch_artist=True)
    for patch, halogen in zip(box_plot['boxes'], ['Cl', 'Br', 'I']):
        patch.set_facecolor(halogen_colors[halogen])
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_plot[element], color='black', linewidth=1)
    ax.set_title('High Plane Angle vs Z-axis', fontweight='bold')
    ax.set_ylabel('Angle (°)', fontweight='bold')
    ax.set_xlabel('Halogen', fontweight='bold')
    
    # Angle between planes
    ax = axes[2]
    box_plot = ax.boxplot([df[df['halogen_X'] == halogen]['angle_between_planes'].values for halogen in ['Cl', 'Br', 'I']], 
                         labels=['Cl', 'Br', 'I'], patch_artist=True)
    for patch, halogen in zip(box_plot['boxes'], ['Cl', 'Br', 'I']):
        patch.set_facecolor(halogen_colors[halogen])
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_plot[element], color='black', linewidth=1)
    ax.set_title('Angle Between Planes', fontweight='bold')
    ax.set_ylabel('Angle (°)', fontweight='bold')
    ax.set_xlabel('Halogen', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plane_angles_by_halogen.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plots by number of layers
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Plane Orientation Analysis by Number of Layers\nStructural Distortion vs Layer Count', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    n_values = sorted(df['n_octahedra'].unique())
    
    # Low plane angles by n
    ax = axes[0]
    box_plot = ax.boxplot([df[df['n_octahedra'] == n]['low_plane_z_angle'].values for n in n_values], 
                         labels=[f'n={n}' for n in n_values], patch_artist=True)
    for patch, n in zip(box_plot['boxes'], n_values):
        patch.set_facecolor(n_colors[n])
        patch.set_alpha(0.8)
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_plot[element], color='black', linewidth=1)
    ax.set_title('Low Plane Angle vs Z-axis', fontweight='bold')
    ax.set_ylabel('Angle (°)', fontweight='bold')
    ax.set_xlabel('Number of Layers', fontweight='bold')
    
    # High plane angles by n
    ax = axes[1]
    box_plot = ax.boxplot([df[df['n_octahedra'] == n]['high_plane_z_angle'].values for n in n_values], 
                         labels=[f'n={n}' for n in n_values], patch_artist=True)
    for patch, n in zip(box_plot['boxes'], n_values):
        patch.set_facecolor(n_colors[n])
        patch.set_alpha(0.8)
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_plot[element], color='black', linewidth=1)
    ax.set_title('High Plane Angle vs Z-axis', fontweight='bold')
    ax.set_ylabel('Angle (°)', fontweight='bold')
    ax.set_xlabel('Number of Layers', fontweight='bold')
    
    # Angle between planes by n
    ax = axes[2]
    box_plot = ax.boxplot([df[df['n_octahedra'] == n]['angle_between_planes'].values for n in n_values], 
                         labels=[f'n={n}' for n in n_values], patch_artist=True)
    for patch, n in zip(box_plot['boxes'], n_values):
        patch.set_facecolor(n_colors[n])
        patch.set_alpha(0.8)
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_plot[element], color='black', linewidth=1)
    ax.set_title('Angle Between Planes', fontweight='bold')
    ax.set_ylabel('Angle (°)', fontweight='bold')
    ax.set_xlabel('Number of Layers', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plane_angles_by_layers.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmaps for vector analysis parameters
    vector_parameters = [
        ('low_plane_z_angle', 'Low Plane Angle vs Z-axis (°)', '.1f'),
        ('high_plane_z_angle', 'High Plane Angle vs Z-axis (°)', '.1f'),
        ('angle_between_planes', 'Angle Between Planes (°)', '.1f'),
        ('distance_between_planes', 'Distance Between Planes (Å)', '.1f')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Vector Analysis Parameters by Halogen and Number of Layers\nPlane Orientations and Distortions', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    for i, (param, title, fmt) in enumerate(vector_parameters):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Create a pivot table for mean values
        pivot_data = df.groupby(['halogen_X', 'n_octahedra'])[param].mean().unstack()
        pivot_data = pivot_data.reindex(['Cl', 'Br', 'I'])
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt=fmt, cmap='viridis', 
                    ax=ax, cbar_kws={'label': title})
        
        ax.set_title(f'{title}', fontweight='bold', pad=10)
        ax.set_xlabel('Number of Layers', fontweight='bold')
        ax.set_ylabel('Halogen', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vector_analysis_heatmaps.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    
    # 5. Polar plots for plane orientations
    create_polar_plots(df, output_dir, halogen_colors, n_colors)
    

def create_polar_plots(df, output_dir, halogen_colors, n_colors):
    """
    Create polar plots showing plane angles with lattice parameters as radii
    """
    
    # First, we need to get lattice parameters - let's read from the lattice analysis files
    lattice_data = extract_lattice_data_for_polar(df)
    
    # Merge with vector data
    df_merged = pd.merge(df, lattice_data, on='experiment', how='inner')
    
    # Create polar plots by halogen - limit to 25 degrees (semicircle)
    fig = plt.figure(figsize=(20, 12))
    
    # Create a 2x3 grid of polar subplots
    polar_params = [
        ('low_plane_z_angle', 'B', 'Low Plane Angle vs Lattice B'),
        ('low_plane_z_angle', 'C', 'Low Plane Angle vs Lattice C'),
        ('high_plane_z_angle', 'B', 'High Plane Angle vs Lattice B'),
        ('high_plane_z_angle', 'C', 'High Plane Angle vs Lattice C'),
        ('angle_between_planes', 'B', 'Inter-plane Angle vs Lattice B'),
        ('angle_between_planes', 'C', 'Inter-plane Angle vs Lattice C')
    ]
    
    for i, (angle_param, radius_param, title) in enumerate(polar_params):
        ax = plt.subplot(2, 3, i+1, projection='polar')
        
        # Set theta limits to semicircle (0 to 25 degrees)
        ax.set_thetamin(0)
        ax.set_thetamax(25)
        
        # Plot by halogen
        for halogen in ['Cl', 'Br', 'I']:
            subset = df_merged[df_merged['halogen_X'] == halogen]
            if len(subset) > 0:
                theta = np.radians(subset[angle_param])  # Convert to radians
                radius = subset[radius_param]
                
                ax.scatter(theta, radius, c=halogen_colors[halogen], 
                          label=halogen, alpha=0.7, s=60)
        
        ax.set_title(title, fontweight='bold', pad=5)
        ax.set_ylabel(f'{radius_param} (Å)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add degree labels
        ax.set_thetagrids(np.arange(0, 26, 5))
        
        if i == 0:  # Add legend only to first plot
            ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))
    
    plt.suptitle('Polar Analysis: Plane Orientations vs Lattice Parameters by Halogen\nAngle (θ) vs Lattice Parameter (r)', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'polar_analysis_by_halogen.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create polar plots by number of layers
    fig = plt.figure(figsize=(20, 12))
    
    for i, (angle_param, radius_param, title) in enumerate(polar_params):
        ax = plt.subplot(2, 3, i+1, projection='polar')
        
        # Set theta limits to semicircle (0 to 25 degrees)
        ax.set_thetamin(0)
        ax.set_thetamax(25)
        
        # Plot by number of layers
        n_values = sorted(df_merged['n_octahedra'].unique())
        for n in n_values:
            subset = df_merged[df_merged['n_octahedra'] == n]
            if len(subset) > 0:
                theta = np.radians(subset[angle_param])  # Convert to radians
                radius = subset[radius_param]
                
                ax.scatter(theta, radius, c=n_colors[n], 
                          label=f'n={n}', alpha=0.7, s=60)
        
        ax.set_title(title, fontweight='bold', pad=5)
        ax.set_ylabel(f'{radius_param} (Å)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add degree labels
        ax.set_thetagrids(np.arange(0, 26, 5))
        
        if i == 0:  # Add legend only to first plot
            ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))
    
    plt.suptitle('Polar Analysis: Plane Orientations vs Lattice Parameters by Layers\nAngle (θ) vs Lattice Parameter (r)', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'polar_analysis_by_layers.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a special polar plot showing distance between planes as radius
    fig = plt.figure(figsize=(16, 8))
    
    # Plot 1: Low plane angle vs distance between planes
    ax1 = plt.subplot(1, 2, 1, projection='polar')
    ax1.set_thetamin(0)
    ax1.set_thetamax(25)
    
    n_values = sorted(df_merged['n_octahedra'].unique())
    for n in n_values:
        subset = df_merged[df_merged['n_octahedra'] == n]
        if len(subset) > 0:
            theta = np.radians(subset['low_plane_z_angle'])
            radius = subset['distance_between_planes']
            ax1.scatter(theta, radius, c=n_colors[n], 
                       label=f'n={n}', alpha=0.7, s=60)
    
    ax1.set_title('Low Plane Angle vs Inter-plane Distance', fontweight='bold', pad=5)
    ax1.set_ylabel('Distance (Å)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_thetagrids(np.arange(0, 26, 5))
    ax1.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))
    
    # Plot 2: High plane angle vs distance between planes
    ax2 = plt.subplot(1, 2, 2, projection='polar')
    ax2.set_thetamin(0)
    ax2.set_thetamax(25)
    
    for n in n_values:
        subset = df_merged[df_merged['n_octahedra'] == n]
        if len(subset) > 0:
            theta = np.radians(subset['high_plane_z_angle'])
            radius = subset['distance_between_planes']
            ax2.scatter(theta, radius, c=n_colors[n], 
                       label=f'n={n}', alpha=0.7, s=60)
    
    ax2.set_title('High Plane Angle vs Inter-plane Distance', fontweight='bold', pad=5)
    ax2.set_ylabel('Distance (Å)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_thetagrids(np.arange(0, 26, 5))
    
    plt.suptitle('Polar Analysis: Plane Orientations vs Inter-plane Distances\nAngle (θ) vs Distance (r)', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'polar_analysis_distances.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def extract_lattice_data_for_polar(df_vector):
    """
    Extract lattice parameters from the lattice analysis JSON files for polar plots
    """
    lattice_data = []
    
    base_dir = '/home/dotempo/Documents/BULKS_RESULTS'
    
    for _, row in df_vector.iterrows():
        experiment = row['experiment']
        
        # Look for the lattice JSON file in the experiment directory
        dir_path = os.path.join(base_dir, experiment)
        lattice_file = None
        
        for file in os.listdir(dir_path):
            if file.endswith('_layers_ontology.json'):
                lattice_file = os.path.join(dir_path, file)
                break
        
        if lattice_file:
            try:
                with open(lattice_file, 'r') as f:
                    data = json.load(f)
                
                lattice_params = data['cell_properties']['lattice_parameters']
                
                lattice_entry = {
                    'experiment': experiment,
                    'B': lattice_params['B'],
                    'C': lattice_params['C']
                }
                
                lattice_data.append(lattice_entry)
                
            except Exception as e:
                print(f"Error reading lattice data for {experiment}: {e}")
                continue
    
    return pd.DataFrame(lattice_data)

def main():
    # Set base directory
    base_dir = '/home/dotempo/Documents/BULKS_RESULTS'
    output_dir = os.path.join(base_dir, 'lattice_parameter_analysis')
    
    print("Starting vector analysis...")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    # Extract data
    df = extract_vector_data(base_dir)
    
    if df.empty:
        print("No vector data found!")
        return
    
    # Data processing complete
    print(f"Processed {len(df)} experiments")
    
    # Create plots
    print("Creating vector analysis visualizations...")
    create_vector_analysis_plots(df, output_dir)
    
    print(f"\nVector analysis complete! Check the '{output_dir}' directory for results.")
    print("Generated files:")
    print("- plane_angles_by_halogen.png")
    print("- plane_angles_by_layers.png") 
    print("- vector_analysis_heatmaps.png")
    print("- vector_analysis_correlations.png")
    print("- polar_analysis_distances.png")

if __name__ == "__main__":
    main() 
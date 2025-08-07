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

def load_energy_data(base_dir):
    """
    Load energy data from perovskites.csv and match with analysis directories.
    Handles truncated analysis directory names by fuzzy matching.
    
    Returns:
        pandas.DataFrame: Energy data with experiment names as keys
    """
    import pandas as pd
    
    csv_path = '/home/dotempo/Documents/REPO/SVC-Materials/Graphs/perovskites.csv'
    print(f"Loading energy data from CSV: {csv_path}")
    
    # Read the CSV file
    try:
        df_csv = pd.read_csv(csv_path)
        print(f"Loaded {len(df_csv)} energy records from CSV")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return pd.DataFrame()
    
    # Get analysis directories
    analysis_dirs = [d for d in os.listdir(base_dir) if d.endswith('_analysis')]
    utility_dirs = ['energy_analysis', 'lattice_parameter_analysis', 'penetration_depth']
    analysis_dirs = [d for d in analysis_dirs if d not in utility_dirs]
    
    print(f"Found {len(analysis_dirs)} experiment analysis directories")
    
    data_list = []
    matched_count = 0
    unmatched_count = 0
    
    for analysis_dir in analysis_dirs:
        try:
            # Get the experiment name (remove _analysis suffix)
            experiment_name = analysis_dir.replace('_analysis', '')
            
            # Try exact match first
            csv_match = df_csv[df_csv['Experiment'] == experiment_name]
            
            if csv_match.empty:
                # Try fuzzy matching for truncated names
                # Find CSV experiments that start with the (potentially truncated) experiment_name
                possible_matches = df_csv[df_csv['Experiment'].str.startswith(experiment_name)]
                
                if len(possible_matches) == 1:
                    csv_match = possible_matches
                    print(f"Fuzzy matched truncated '{experiment_name}' to full '{csv_match.iloc[0]['Experiment']}'")
                elif len(possible_matches) > 1:
                    # If multiple matches, take the shortest one (most likely match)
                    csv_match = possible_matches.loc[[possible_matches['Experiment'].str.len().idxmin()]]
                    print(f"Multiple matches for '{experiment_name}', using shortest: '{csv_match.iloc[0]['Experiment']}'")
            
            if not csv_match.empty:
                row = csv_match.iloc[0]
                
                # Extract halogen and n_layers from the full experiment name
                experiment_full = row['Experiment']
                
                # Extract halogen
                halogen = row['Halogen']
                
                # Extract n_layers
                n_layers = int(row['N_Slab'])
                
                entry = {
                    'experiment_key': analysis_dir,  # Use analysis dir name as key
                    'experiment_full': experiment_full,  # Store full name from CSV
                    'halogen_X': halogen,
                    'n_layers': n_layers,
                    'energy_per_atom_eV': row['NormEnergy'],  # This is our main energy metric
                    'total_energy_eV': row['Eslab'],
                    'n_total_atoms': int(row['TotalAtoms']),
                    'chemical_formula': f"{row['Perovskite']}_{row['Molecule']}",
                    'molecule_name': row['Molecule']
                }
                
                data_list.append(entry)
                matched_count += 1
            else:
                print(f"No CSV match found for: {experiment_name}")
                unmatched_count += 1
                
        except Exception as e:
            print(f"Error processing {analysis_dir}: {e}")
            unmatched_count += 1
            continue
    
    df = pd.DataFrame(data_list)
    print(f"Successfully matched {matched_count} experiments, {unmatched_count} unmatched")
    
    if not df.empty:
        print(f"Loaded energy data for {len(df)} experiments")
        print(f"Unique halogens: {df['halogen_X'].unique()}")
        print(f"Unique n_layers: {sorted(df['n_layers'].unique())}")
        print(f"NormEnergy range: {df['energy_per_atom_eV'].min():.3f} to {df['energy_per_atom_eV'].max():.3f}")
    
    return df

def load_penetration_data(base_dir):
    """
    Load penetration analysis data from JSON files.
    
    Returns:
        pandas.DataFrame: Penetration data with calculated metrics
    """
    print("Loading penetration analysis data...")
    
    data_list = []
    analysis_dirs = [d for d in os.listdir(base_dir) if d.endswith('_analysis')]
    
    for dir_name in analysis_dirs:
        try:
            # Look for penetration analysis file
            dir_path = os.path.join(base_dir, dir_name)
            pen_file = None
            
            for file in os.listdir(dir_path):
                if file.endswith('_penetration_analysis.json'):
                    pen_file = os.path.join(dir_path, file)
                    break
            
            if pen_file is None:
                continue
                
            # Read penetration data
            with open(pen_file, 'r') as f:
                pen_data = json.load(f)
            
            penetration_analysis = pen_data.get('penetration_analysis', {})
            if 'error' in penetration_analysis:
                continue
            
            # Extract data for all available molecules
            molecule_keys = [key for key in penetration_analysis.keys() if key.startswith('molecule_')]
            
            for mol_key in molecule_keys:
                if mol_key in penetration_analysis:
                    mol_data = penetration_analysis[mol_key]
                    segments = mol_data.get('penetration_segments', {})
                    
                    if segments:
                        # Calculate derived metrics
                        n1_low = segments.get('n1_to_low_plane', np.nan)
                        low_high = segments.get('low_plane_to_high_plane', np.nan)
                        high_n2 = segments.get('high_plane_to_n2', np.nan)
                        
                        penetration_percentage = np.nan
                        penetration_asymmetry = np.nan
                        
                        if not np.isnan(n1_low) and not np.isnan(high_n2) and not np.isnan(low_high) and low_high > 0:
                            penetration_percentage = ((n1_low + high_n2) / low_high) * 100
                            penetration_asymmetry = n1_low - high_n2
                        
                        entry = {
                            'experiment_key': dir_name,
                            'molecule_key': mol_key,
                            'n1_to_low_plane': n1_low,
                            'low_plane_to_high_plane': low_high,
                            'high_plane_to_n2': high_n2,
                            'penetration_percentage': penetration_percentage,
                            'penetration_asymmetry': penetration_asymmetry
                        }
                        
                        data_list.append(entry)
        
        except Exception as e:
            print(f"Error processing penetration data for {dir_name}: {e}")
            continue
    
    df = pd.DataFrame(data_list)
    print(f"Loaded penetration data for {len(df)} molecule entries from {len(df['experiment_key'].unique())} experiments")
    
    return df

def load_lattice_data(base_dir):
    """
    Load lattice parameter data from JSON files.
    
    Returns:
        pandas.DataFrame: Lattice data with calculated ratios
    """
    print("Loading lattice parameter data...")
    
    data_list = []
    analysis_dirs = [d for d in os.listdir(base_dir) if d.endswith('_analysis')]
    
    for dir_name in analysis_dirs:
        try:
            # Look for lattice parameters file
            dir_path = os.path.join(base_dir, dir_name)
            lattice_file = None
            
            for file in os.listdir(dir_path):
                if file.endswith('_layers_ontology.json'):
                    lattice_file = os.path.join(dir_path, file)
                    break
            
            if lattice_file is None:
                continue
                
            # Read lattice data
            with open(lattice_file, 'r') as f:
                lattice_data = json.load(f)
            
            # Extract lattice parameters
            lattice_params = lattice_data['cell_properties']['lattice_parameters']
            
            # Calculate derived parameters
            a_over_b = lattice_params['A'] / lattice_params['B']
            alpha_over_beta = lattice_params['Alpha'] / lattice_params['Beta']
            
            entry = {
                'experiment_key': dir_name,
                'A': lattice_params['A'],
                'B': lattice_params['B'], 
                'C': lattice_params['C'],
                'Alpha': lattice_params['Alpha'],
                'Beta': lattice_params['Beta'],
                'Gamma': lattice_params['Gamma'],
                'a_over_b': a_over_b,
                'alpha_over_beta': alpha_over_beta
            }
            
            data_list.append(entry)
            
        except Exception as e:
            print(f"Error processing lattice data for {dir_name}: {e}")
            continue
    
    df = pd.DataFrame(data_list)
    print(f"Loaded lattice data for {len(df)} experiments")
    
    return df

def combine_all_data(energy_df, penetration_df, lattice_df):
    """
    Combine energy, penetration, and lattice data into a unified dataset.
    
    Returns:
        pandas.DataFrame: Combined dataset
    """
    print("Combining all datasets...")
    
    # For penetration data, take average values per experiment (since we have multiple molecules per experiment)
    penetration_avg = penetration_df.groupby('experiment_key').agg({
        'penetration_percentage': 'mean',
        'penetration_asymmetry': 'mean',
        'low_plane_to_high_plane': 'mean'
    }).reset_index()
    
    # Merge all datasets
    combined = energy_df.copy()
    combined = combined.merge(penetration_avg, on='experiment_key', how='left')
    combined = combined.merge(lattice_df, on='experiment_key', how='left')
    
    print(f"Combined dataset has {len(combined)} records")
    print(f"Records with penetration data: {len(combined[~combined['penetration_percentage'].isna()])}")
    print(f"Records with lattice data: {len(combined[~combined['a_over_b'].isna()])}")
    
    return combined

def create_energy_analysis_plots(df, output_dir):
    """
    Create comprehensive energy analysis plots.
    """
    # Set up plotting style
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
    
    # Color palettes
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
    
    # 1. Energy by Number of Layers and Halogen
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Normalized Energy Analysis\nNormalized Energy (eV/atom) vs Structural Parameters', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Plot 1a: Energy by Halogen
    ax1 = axes[0]
    box_data_halogen = []
    labels_halogen = []
    colors_halogen = []
    
    for halogen in ['Cl', 'Br', 'I']:
        data = df[df['halogen_X'] == halogen]['energy_per_atom_eV'].dropna().values
        if len(data) > 0:
            box_data_halogen.append(data)
            labels_halogen.append(halogen)
            colors_halogen.append(halogen_colors[halogen])
    
    if box_data_halogen:
        box_plot = ax1.boxplot(box_data_halogen, labels=labels_halogen, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors_halogen):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box_plot[element], color='black', linewidth=1)
    
    ax1.set_title('Normalized Energy (eV/atom) by Halogen', fontweight='bold', pad=10)
    ax1.set_ylabel('Normalized Energy (eV/atom)', fontweight='bold')
    ax1.set_xlabel('Halogen', fontweight='bold')
    ax1.set_ylim(-2.5, 0)
    ax1.grid(True, alpha=0.3)
    
    # Plot 1b: Energy by Number of Layers
    ax2 = axes[1]
    box_data_layers = []
    labels_layers = []
    colors_layers = []
    
    for n in sorted(df['n_layers'].unique()):
        data = df[df['n_layers'] == n]['energy_per_atom_eV'].dropna().values
        if len(data) > 0:
            box_data_layers.append(data)
            labels_layers.append(f'n={n}')
            colors_layers.append(n_colors[n])
    
    if box_data_layers:
        box_plot = ax2.boxplot(box_data_layers, labels=labels_layers, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors_layers):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box_plot[element], color='black', linewidth=1)
    
    ax2.set_title('Normalized Energy (eV/atom) by Number of Layers', fontweight='bold', pad=10)
    ax2.set_ylabel('Normalized Energy (eV/atom)', fontweight='bold')
    ax2.set_xlabel('Number of Layers', fontweight='bold')
    ax2.set_ylim(-2.5, 0)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_by_structure.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Energy vs Penetration Analysis
    penetration_data = df[~df['penetration_percentage'].isna()].copy()
    
    if len(penetration_data) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Energy vs Penetration Depth Analysis\nStructural-Energy Correlations (Grouped by Halogen and Layers)', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Define markers for different n values
        n_markers = {1: 'o', 2: 's', 3: '^'}  # circle, square, triangle
        
        # Plot 2a: Energy vs Penetration Percentage
        ax1 = axes[0]

        for halogen in ['Cl', 'Br', 'I']:
            for n in sorted(penetration_data['n_layers'].unique()):
                subset = penetration_data[(penetration_data['halogen_X'] == halogen) & 
                                        (penetration_data['n_layers'] == n)]
                if len(subset) > 0:
                    ax1.scatter(subset['penetration_percentage'], subset['energy_per_atom_eV'], 
                               c=halogen_colors[halogen], marker=n_markers.get(n, 'o'),
                               label=f'{halogen}-n{n}', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        ax1.set_xlabel('Penetration Percentage (%)', fontweight='bold')
        ax1.set_ylabel('Normalized Energy (eV/atom)', fontweight='bold')
        ax1.set_title('Energy vs Penetration Percentage', fontweight='bold', pad=10)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-2.5, 0) # Added y-axis limit
        
        # Plot 2b: Energy vs Penetration Asymmetry
        ax2 = axes[1]
        
        for halogen in ['Cl', 'Br', 'I']:
            for n in sorted(penetration_data['n_layers'].unique()):
                subset = penetration_data[(penetration_data['halogen_X'] == halogen) & 
                                        (penetration_data['n_layers'] == n)]
                if len(subset) > 0:
                    ax2.scatter(subset['penetration_asymmetry'], subset['energy_per_atom_eV'], 
                               c=halogen_colors[halogen], marker=n_markers.get(n, 'o'),
                               label=f'{halogen}-n{n}', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('Penetration Asymmetry (Å)', fontweight='bold')
        ax2.set_ylabel('Normalized Energy (eV/atom)', fontweight='bold')
        ax2.set_title('Energy vs Penetration Asymmetry', fontweight='bold', pad=10)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_ylim(-2.5, 0) # Added y-axis limit
        ax2.set_xlim(-1.2, 1.2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'energy_vs_penetration.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Energy vs Lattice Parameters
    lattice_data = df[~df['a_over_b'].isna()].copy()
    
    if len(lattice_data) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Energy vs Lattice Parameter Analysis\nStructural Anisotropy-Energy Correlations (Grouped by Halogen and Layers)', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Define markers for different n values
        n_markers = {1: 'o', 2: 's', 3: '^'}  # circle, square, triangle
        
        # Plot 3a: Energy vs a/b ratio
        ax1 = axes[0]
        
        for halogen in ['Cl', 'Br', 'I']:
            for n in sorted(lattice_data['n_layers'].unique()):
                subset = lattice_data[(lattice_data['halogen_X'] == halogen) & 
                                    (lattice_data['n_layers'] == n)]
                if len(subset) > 0:
                    ax1.scatter(subset['a_over_b'], subset['energy_per_atom_eV'], 
                               c=halogen_colors[halogen], marker=n_markers.get(n, 'o'),
                               label=f'{halogen}-n{n}', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        ax1.set_xlabel('a/b Ratio', fontweight='bold')
        ax1.set_ylabel('Normalized Energy (eV/atom)', fontweight='bold')
        ax1.set_title('Energy vs Lattice a/b Ratio', fontweight='bold', pad=10)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=1, color='black', linestyle='--', alpha=0.5, label='Perfect Square (a=b)')
        ax1.set_ylim(-2.5, 0)
        
        # Plot 3b: Energy vs α/β ratio
        ax2 = axes[1]
        
        for halogen in ['Cl', 'Br', 'I']:
            for n in sorted(lattice_data['n_layers'].unique()):
                subset = lattice_data[(lattice_data['halogen_X'] == halogen) & 
                                    (lattice_data['n_layers'] == n)]
                if len(subset) > 0:
                    ax2.scatter(subset['alpha_over_beta'], subset['energy_per_atom_eV'], 
                               c=halogen_colors[halogen], marker=n_markers.get(n, 'o'),
                               label=f'{halogen}-n{n}', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('α/β Ratio', fontweight='bold')
        ax2.set_ylabel('Normalized Energy (eV/atom)', fontweight='bold')
        ax2.set_title('Energy vs Angular α/β Ratio', fontweight='bold', pad=10)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=1, color='black', linestyle='--', alpha=0.5, label='Perfect Match (α=β)')
        ax2.set_ylim(-2.5, 0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'energy_vs_lattice.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Double Grouping Plots: Energy vs Halogen and Layers Combined
    # Create 4 plots showing double grouping (3 halogens × 3 layers = 9 groups each)
    
    # Plot 4a: Box plot with double grouping
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Prepare data for double grouping
    double_group_data = []
    double_group_labels = []
    double_group_colors = []
    
    halogens = ['Cl', 'Br', 'I']
    layers = sorted(df['n_layers'].unique())
    
    for halogen in halogens:
        for n in layers:
            subset = df[(df['halogen_X'] == halogen) & (df['n_layers'] == n)]
            energy_data = subset['energy_per_atom_eV'].dropna().values
            
            if len(energy_data) > 0:
                double_group_data.append(energy_data)
                double_group_labels.append(f'{halogen}-n{n}')
                double_group_colors.append(halogen_colors[halogen])
    
    if double_group_data:
        box_plot = ax.boxplot(double_group_data, labels=double_group_labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], double_group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box_plot[element], color='black', linewidth=1)
    
    ax.set_title('Normalized Energy by Halogen and Layers (Double Grouping)', fontweight='bold', pad=15)
    ax.set_ylabel('Normalized Energy (eV/atom)', fontweight='bold')
    ax.set_xlabel('Halogen-Layer Combination', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    ax.set_ylim(-2.5, 0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_double_grouping_boxplot.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4b: Grouped violin plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Prepare data for violin plot
    violin_data = []
    violin_positions = []
    violin_colors = []
    violin_labels = []
    
    bar_width = 0.25
    x_positions = np.arange(len(layers))
    
    for i, halogen in enumerate(halogens):
        for j, n in enumerate(layers):
            subset = df[(df['halogen_X'] == halogen) & (df['n_layers'] == n)]
            energy_data = subset['energy_per_atom_eV'].dropna().values
            
            if len(energy_data) > 0:
                violin_data.append(energy_data)
                position = x_positions[j] + (i - 1) * bar_width
                violin_positions.append(position)
                violin_colors.append(halogen_colors[halogen])
                violin_labels.append(f'{halogen}-n{n}')
    
    if violin_data:
        # Create violin plot
        parts = ax.violinplot(violin_data, positions=violin_positions, widths=bar_width*0.8, 
                             showmeans=True, showmedians=True)
        
        # Color the violins
        for pc, color in zip(parts['bodies'], violin_colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
        
        # Style other elements
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                parts[partname].set_color('black')
                parts[partname].set_linewidth(1)
    
    ax.set_title('Normalized Energy by Halogen and Layers (Distribution)', fontweight='bold', pad=15)
    ax.set_ylabel('Normalized Energy (eV/atom)', fontweight='bold')
    ax.set_xlabel('Number of Layers', fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'n={n}' for n in layers])
    ax.set_ylim(-2.5, 0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Create custom legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=halogen_colors[halogen], alpha=0.7, label=halogen) 
                      for halogen in halogens]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_double_grouping_violinplot.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4c: Heatmap showing interaction
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create matrix for heatmap
    heatmap_data = np.zeros((len(halogens), len(layers)))
    
    for i, halogen in enumerate(halogens):
        for j, n in enumerate(layers):
            subset = df[(df['halogen_X'] == halogen) & (df['n_layers'] == n)]
            energy_data = subset['energy_per_atom_eV'].dropna()
            
            if len(energy_data) > 0:
                heatmap_data[i, j] = energy_data.mean()
            else:
                heatmap_data[i, j] = np.nan
    
    # Create heatmap
    im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Energy (eV/atom)', fontweight='bold')
    
    # Set labels
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'n={n}' for n in layers])
    ax.set_yticks(range(len(halogens)))
    ax.set_yticklabels(halogens)
    
    # Add text annotations
    for i in range(len(halogens)):
        for j in range(len(layers)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Normalized Energy Heatmap\nHalogen vs Number of Layers', fontweight='bold', pad=15)
    ax.set_xlabel('Number of Layers', fontweight='bold')
    ax.set_ylabel('Halogen', fontweight='bold')
    
    # Set colorbar limits
    im.set_clim(-2.5, 0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_double_grouping_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4d: Line plot showing trends
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for halogen in halogens:
        halogen_means = []
        halogen_stds = []
        
        for n in layers:
            subset = df[(df['halogen_X'] == halogen) & (df['n_layers'] == n)]
            energy_data = subset['energy_per_atom_eV'].dropna()
            
            if len(energy_data) > 0:
                halogen_means.append(energy_data.mean())
                halogen_stds.append(energy_data.std())
            else:
                halogen_means.append(np.nan)
                halogen_stds.append(np.nan)
        
        # Plot line with error bars
        ax.errorbar(layers, halogen_means, yerr=halogen_stds, 
                   marker='o', linewidth=2, markersize=8, capsize=5,
                   label=halogen, color=halogen_colors[halogen])
    
    ax.set_title('Normalized Energy Trends\nHalogen vs Number of Layers', fontweight='bold', pad=15)
    ax.set_ylabel('Normalized Energy (eV/atom)', fontweight='bold')
    ax.set_xlabel('Number of Layers', fontweight='bold')
    ax.set_xticks(layers)
    ax.set_xticklabels([f'n={n}' for n in layers])
    ax.set_ylim(-2.5, 0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_double_grouping_lineplot.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Comprehensive Correlation Heatmap
    # Select numerical columns for correlation analysis
    correlation_columns = ['energy_per_atom_eV', 'total_energy_eV', 'n_layers', 'n_total_atoms']
    
    if 'penetration_percentage' in df.columns:
        correlation_columns.extend(['penetration_percentage', 'penetration_asymmetry', 'low_plane_to_high_plane'])
    
    if 'a_over_b' in df.columns:
        correlation_columns.extend(['a_over_b', 'alpha_over_beta', 'A', 'B', 'C'])
    
    # Calculate correlations
    corr_data = df[correlation_columns].corr()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Create correlation heatmap
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    sns.heatmap(corr_data, mask=mask, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
    
    ax.set_title('Energy-Structure Correlation Matrix\nNormalized Energy vs All Parameters', 
                 fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_correlation_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Statistical Summary
    print("\n" + "="*80)
    print("ENERGY ANALYSIS STATISTICAL SUMMARY")
    print("="*80)
    
    print("\nNormalized Energy (eV/atom) by Halogen:")
    print("-" * 40)
    for halogen in ['Cl', 'Br', 'I']:
        subset = df[df['halogen_X'] == halogen]['energy_per_atom_eV'].dropna()
        if len(subset) > 0:
            print(f"{halogen} (n={len(subset)}): {subset.mean():.3f} ± {subset.std():.3f} eV")
    
    print("\nNormalized Energy (ev/atom) by Number of Layers:")
    print("-" * 50)
    for n in sorted(df['n_layers'].unique()):
        subset = df[df['n_layers'] == n]['energy_per_atom_eV'].dropna()
        if len(subset) > 0:
            print(f"n={n} (n={len(subset)}): {subset.mean():.3f} ± {subset.std():.3f} eV")
    
    if len(penetration_data) > 0:
        print("\nCorrelations with Penetration Analysis:")
        print("-" * 50)
        corr_pen_perc = penetration_data['energy_per_atom_eV'].corr(penetration_data['penetration_percentage'])
        corr_pen_asym = penetration_data['energy_per_atom_eV'].corr(penetration_data['penetration_asymmetry'])
        print(f"Energy vs Penetration Percentage: r = {corr_pen_perc:.3f}")
        print(f"Energy vs Penetration Asymmetry: r = {corr_pen_asym:.3f}")
    
    if len(lattice_data) > 0:
        print("\nCorrelations with Lattice Parameters:")
        print("-" * 50)
        corr_ab = lattice_data['energy_per_atom_eV'].corr(lattice_data['a_over_b'])
        corr_alpha_beta = lattice_data['energy_per_atom_eV'].corr(lattice_data['alpha_over_beta'])
        print(f"Energy vs a/b ratio: r = {corr_ab:.3f}")
        print(f"Energy vs α/β ratio: r = {corr_alpha_beta:.3f}")

def main():
    # Set paths
    base_dir = '/home/dotempo/Documents/REPO/SVC-Materials/Graphs/BULKS_RESULTS'
    output_dir = os.path.join(base_dir, 'energy_analysis')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting comprehensive energy analysis...")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load all datasets
    energy_df = load_energy_data(base_dir)
    if energy_df.empty:
        print("Failed to load energy data. Exiting.")
        return
    
    penetration_df = load_penetration_data(base_dir)
    lattice_df = load_lattice_data(base_dir)
    
    # Combine all data
    combined_df = combine_all_data(energy_df, penetration_df, lattice_df)
    
    if combined_df.empty:
        print("No combined data available. Exiting.")
        return
    
    # Create plots
    print("Creating energy analysis visualizations...")
    create_energy_analysis_plots(combined_df, output_dir)
    
    # Save combined dataset
    combined_df.to_csv(os.path.join(output_dir, 'combined_energy_data.csv'), index=False)
    
    print(f"\nEnergy analysis complete! Check the '{output_dir}' directory for results.")
    print("Generated files:")
    print("- energy_by_structure.png")
    print("- energy_vs_penetration.png")
    print("- energy_vs_lattice.png")
    print("- energy_double_grouping_boxplot.png")
    print("- energy_double_grouping_violinplot.png")
    print("- energy_double_grouping_heatmap.png")
    print("- energy_double_grouping_lineplot.png")
    print("- energy_correlation_matrix.png")
    print("- combined_energy_data.csv")

if __name__ == "__main__":
    main() 
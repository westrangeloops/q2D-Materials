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

# Set up matplotlib for publication-ready plots (matching lattice_analysis.py style)
plt.style.use('default')
sns.set_palette("husl")

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 16,  # Increased base font size
    'axes.linewidth': 1.5,  # Thicker axes
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 1.0,
    'figure.dpi': 300,
    'axes.labelsize': 18,  # Axis labels
    'axes.titlesize': 20,  # Subplot titles
    'xtick.labelsize': 16,  # X-axis tick labels
    'ytick.labelsize': 16,  # Y-axis tick labels
    'legend.fontsize': 16,  # Legend text
    'figure.titlesize': 24,  # Main figure title
    'lines.linewidth': 2.0,  # Line thickness
    'lines.markersize': 8,   # Marker size
    'patch.linewidth': 1.5   # Box plot line width
})

def load_dos_end_energy_data():
    """
    Load energy and structural data from the comprehensive DOS_END dataset.
    
    Returns:
        pandas.DataFrame: Combined energy and structural data
    """
    print("Loading DOS_END comprehensive dataset...")
    
    csv_path = '/home/dotempo/Documents/SVC-Materials/Graphs/DOS_END_RESULTS/comprehensive_dos_end_dataset.csv'
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records from comprehensive dataset")
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return pd.DataFrame()
    
    # Load pre-calculated formation energies from analysis directories
    print("Loading pre-calculated formation energies from analysis directories...")
    
    base_dir = '/home/dotempo/Documents/SVC-Materials/Graphs/DOS_END_RESULTS'
    energy_data = []
    
    # Find all analysis directories
    analysis_dirs = [d for d in os.listdir(base_dir) if d.endswith('_analysis')]
    
    for dir_name in analysis_dirs:
        try:
            dir_path = os.path.join(base_dir, dir_name)
            structure_name = dir_name.replace('_analysis', '')
            
            # Look for pre-calculated energy data
            energy_json_path = os.path.join(dir_path, 'energy_properties.json')
            energy_txt_path = os.path.join(dir_path, 'energy.txt')
            
            formation_energy_per_atom = None
            total_atoms = None
            
            # First try to read from energy_properties.json (most complete)
            if os.path.exists(energy_json_path):
                try:
                    with open(energy_json_path, 'r') as f:
                        energy_props = json.load(f)
                        formation_energy_per_atom = energy_props.get('formation_energy_per_atom_eV')
                        total_atoms = energy_props.get('total_atoms')
                        
                        if formation_energy_per_atom is not None and total_atoms is not None:
                            energy_data.append({
                                'structure_name': structure_name,
                                'formation_energy_eV': formation_energy_per_atom,
                                'total_atoms': total_atoms
                            })
                            continue
                except Exception as e:
                    print(f"Error reading energy_properties.json for {dir_name}: {e}")
            
            # Fallback to energy.txt if available
            if os.path.exists(energy_txt_path):
                try:
                    with open(energy_txt_path, 'r') as f:
                        energy_content = f.read().strip()
                        formation_energy_per_atom = float(energy_content)
                        
                        # Get total_atoms from the main dataset
                        structure_rows = df[df['structure_name'] == structure_name]
                        if len(structure_rows) > 0:
                            total_atoms = structure_rows.iloc[0]['total_atoms']
                            
                            energy_data.append({
                                'structure_name': structure_name,
                                'formation_energy_eV': formation_energy_per_atom,
                                'total_atoms': total_atoms
                            })
                            continue
                except Exception as e:
                    print(f"Error reading energy.txt for {dir_name}: {e}")
            
            print(f"No pre-calculated energy found for {structure_name}")
        
        except Exception as e:
            print(f"Error processing energy for {dir_name}: {e}")
            continue
    
    # Create energy dataframe
    energy_df = pd.DataFrame(energy_data)
    
    if len(energy_df) > 0:
        print(f"Successfully extracted energy data for {len(energy_df)} structures")
        
        # Merge with the main dataset
        # For octahedra, use the first octahedron of each structure
        octahedra_df = df[df['entity_type'] == 'octahedron'].groupby('structure_name').first().reset_index()
        
        # Merge energy data
        combined_df = octahedra_df.merge(energy_df, on='structure_name', how='inner', suffixes=('', '_energy'))
        
        print(f"Combined dataset has {len(combined_df)} structures with energy data")
        print(f"Formation energy range: {combined_df['formation_energy_eV'].min():.4f} to {combined_df['formation_energy_eV'].max():.4f} eV/atom")
        
        # Filter for only stable structures (formation energy < 0)
        stable_structures = combined_df[combined_df['formation_energy_eV'] < 0].copy()
        print(f"Filtering for stable structures (formation energy < 0): {len(stable_structures)} structures")
        
        if len(stable_structures) > 0:
            print(f"Stable formation energy range: {stable_structures['formation_energy_eV'].min():.4f} to {stable_structures['formation_energy_eV'].max():.4f} eV/atom")
            return stable_structures
        else:
            print("Warning: No stable structures found! Using all data.")
            return combined_df
    else:
        print("No pre-calculated formation energies found!")
        print("Please run the 'extract_dos_end_energies.py' script first to calculate formation energies.")
        return pd.DataFrame()

def create_energy_structure_plots(df, output_dir):
    """
    Create comprehensive energy analysis plots showing energy vs various structural parameters.
    Similar style to lattice_analysis.py but with energy as the dependent variable.
    """
    
    # Define color palette for halogens (matching lattice_analysis.py)
    halogen_colors = {
        'Cl': '#2E8B57',  # Sea green
        'Br': '#FF6B35',  # Orange-red  
        'I': '#4A90E2'    # Blue
    }
    
    # Define the parameters to plot against energy
    parameters = [
        # Basic structural parameters
        ('halogen_X', 'Halogen Type', '', 'categorical'),
        ('n_slab', 'Number of Layers (n)', '', 'categorical'),
        
        # Lattice parameters
        ('lattice_a', 'Lattice Parameter a', 'Å', 'continuous'),
        ('lattice_b', 'Lattice Parameter b', 'Å', 'continuous'),
        ('lattice_c', 'Lattice Parameter c', 'Å', 'continuous'),
        ('alpha', 'Alpha Angle (α)', '°', 'continuous'),
        ('beta', 'Beta Angle (β)', '°', 'continuous'),
        ('gamma', 'Gamma Angle (γ)', '°', 'continuous'),
        
        # Octahedral distortion parameters (with normalization)
        ('zeta', 'Zeta (ζ) Distortion', '', 'octahedral'),
        ('delta', 'Delta (δ) Distortion', '', 'octahedral'),
        ('sigma', 'Sigma (σ) Distortion', '°', 'octahedral'),
        ('theta_mean', 'Mean Theta Angle', '°', 'octahedral'),
        
        # Derived parameters
        ('cell_volume', 'Cell Volume', 'Å³', 'continuous'),
        ('mean_bond_distance', 'Mean Pb-X Bond Distance', 'Å', 'continuous'),
        
        # Penetration analysis (if available)
        ('vector_distance_between_centers_angstrom', 'Distance Between Plane Centers', 'Å', 'continuous'),
        ('vector_angle_between_planes_deg', 'Angle Between Planes', '°', 'continuous'),
    ]
    
    # Get unique values
    halogens = ['Cl', 'Br', 'I']
    n_values = sorted(df['n_slab'].unique())
    
    print(f"Creating energy analysis plots for {len(df)} structures...")
    print(f"Halogens: {halogens}")
    print(f"N-layers: {n_values}")
    
    # Create plots for each parameter
    for param, param_title, unit, plot_type in parameters:
        if param not in df.columns:
            print(f"Skipping {param} - not found in dataset")
            continue
            
        # Skip if all values are NaN
        if df[param].isna().all():
            print(f"Skipping {param} - all values are NaN")
            continue
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        if plot_type == 'categorical':
            # Create box plots for categorical variables
            if param == 'halogen_X':
                # Energy vs Halogen
                plot_data = []
                plot_labels = []
                plot_colors = []
                
                for halogen in halogens:
                    subset = df[df['halogen_X'] == halogen]
                    if len(subset) > 0:
                        plot_data.append(subset['formation_energy_eV'].values)
                        plot_labels.append(halogen)
                        plot_colors.append(halogen_colors[halogen])
                
                if plot_data:
                    positions = list(range(1, len(plot_data) + 1))
                    violin_parts = ax.violinplot(plot_data, positions=positions, widths=0.6, 
                                               showmeans=True, showmedians=True)
                    
                    # Color the violin plots
                    for i, (patch, color) in enumerate(zip(violin_parts['bodies'], plot_colors)):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                        patch.set_edgecolor('black')
                        patch.set_linewidth(1.5)
                    
                    ax.set_xticks(positions)
                    ax.set_xticklabels(plot_labels)
                    
            elif param == 'n_slab':
                # Energy vs Number of Layers
                plot_data = []
                plot_labels = []
                plot_colors = []
                
                n_layer_colors = {1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1'}
                
                for n in n_values:
                    subset = df[df['n_slab'] == n]
                    if len(subset) > 0:
                        plot_data.append(subset['formation_energy_eV'].values)
                        plot_labels.append(f'n={n}')
                        plot_colors.append(n_layer_colors.get(n, '#666666'))
                
                if plot_data:
                    positions = list(range(1, len(plot_data) + 1))
                    violin_parts = ax.violinplot(plot_data, positions=positions, widths=0.6,
                                               showmeans=True, showmedians=True)
                    
                    # Color the violin plots
                    for i, (patch, color) in enumerate(zip(violin_parts['bodies'], plot_colors)):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                        patch.set_edgecolor('black')
                        patch.set_linewidth(1.5)
                    
                    ax.set_xticks(positions)
                    ax.set_xticklabels(plot_labels)
        
        elif plot_type == 'octahedral':
            # Special handling for octahedral distortion parameters
            # Create 3 subplots for n=1, n=2, n=3 with same y-axis scale
            plt.close()  # Close the single plot
            
            # Apply normalization as in vector_analysis.py and lattice_analysis.py
            df_plot = df.copy()
            if param == 'sigma':
                # Divide sigma by 12 (as in vector_analysis.py line 177)
                df_plot[param] = df_plot[param] / 12
                param_title += ' (normalized by 12)'
            elif param in ['theta_mean', 'theta_min', 'theta_max']:
                # Divide theta by 24 (as in vector_analysis.py line 181)
                df_plot[param] = df_plot[param] / 24
                param_title += ' (normalized by 24)'
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
            fig.suptitle(f'Formation Energy vs {param_title}', fontweight='bold', fontsize=24, y=0.95)
            
            # Define markers for different halogens (keep shapes as requested)
            halogen_markers = {'Cl': 'o', 'Br': 's', 'I': '^'}
            
            # Calculate global y-axis limits for consistency
            energy_min = df_plot['formation_energy_eV'].min()
            energy_max = df_plot['formation_energy_eV'].max()
            energy_range = energy_max - energy_min
            margin = energy_range * 0.1
            y_limits = (energy_min - margin, energy_max + margin)
            
            # Create one subplot for each n-value
            for i, n_val in enumerate(sorted(n_values)):
                ax = axes[i]
                n_data = df_plot[df_plot['n_slab'] == n_val]
                
                if len(n_data) == 0:
                    ax.set_title(f'n={n_val} (No Data)', fontweight='bold', fontsize=20)
                    continue
                
                # Plot different halogens with different markers and colors
                for halogen in halogens:
                    halogen_data = n_data[n_data['halogen_X'] == halogen]
                    
                    if len(halogen_data) > 0:
                        # Remove NaN values
                        valid_data = halogen_data[[param, 'formation_energy_eV']].dropna()
                        
                        if len(valid_data) > 0:
                            ax.scatter(valid_data[param], valid_data['formation_energy_eV'],
                                     c=halogen_colors[halogen], marker=halogen_markers[halogen],
                                     label=f'{halogen}', alpha=0.8, s=100, 
                                     edgecolors='black', linewidth=1)
                
                # Style each subplot
                ax.set_title(f'n={n_val}', fontweight='bold', fontsize=20)
                ax.set_xlabel(f'{param_title.split("(")[0].strip()} {unit}', fontweight='bold', fontsize=16)
                if i == 0:  # Only label y-axis on the first subplot
                    ax.set_ylabel('Formation Energy per Atom (eV/atom)', fontweight='bold', fontsize=16)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.set_ylim(y_limits)
                
                # Set specific x-axis limits for certain parameters
                if param == 'theta_mean':
                    ax.set_xlim(right=15)  # Set maximum x-value to 15
                elif param == 'vector_distance_between_centers_angstrom':
                    ax.set_xlim(right=13)  # Set maximum x-value to 13
                
                # Add legend only to the first subplot
                if i == 0:
                    ax.legend(fontsize=14, title='Halogen', title_fontsize=16)
            
            plt.tight_layout()
            
            # Save with descriptive filename
            filename = f'energy_vs_{param}_analysis.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created octahedral plot for {param_title}: {filename}")
            continue  # Skip the rest of the single-plot logic
        
        else:  # continuous variables
            # Create 3 subplots for n=1, n=2, n=3 with same y-axis scale
            plt.close()  # Close the single plot
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
            fig.suptitle(f'Formation Energy vs {param_title}', fontweight='bold', fontsize=24, y=0.95)
            
            # Define markers for different halogens
            halogen_markers = {'Cl': 'o', 'Br': 's', 'I': '^'}
            
            # Calculate global y-axis limits for consistency
            energy_min = df['formation_energy_eV'].min()
            energy_max = df['formation_energy_eV'].max()
            energy_range = energy_max - energy_min
            margin = energy_range * 0.1
            y_limits = (energy_min - margin, energy_max + margin)
            
            # Create one subplot for each n-value
            for i, n_val in enumerate(sorted(n_values)):
                ax = axes[i]
                n_data = df[df['n_slab'] == n_val]
                
                if len(n_data) == 0:
                    ax.set_title(f'n={n_val} (No Data)', fontweight='bold', fontsize=20)
                    continue
                
                # Plot different halogens with different markers and colors
                for halogen in halogens:
                    halogen_data = n_data[n_data['halogen_X'] == halogen]
                    
                    if len(halogen_data) > 0:
                        # Remove NaN values
                        valid_data = halogen_data[[param, 'formation_energy_eV']].dropna()
                        
                        if len(valid_data) > 0:
                            ax.scatter(valid_data[param], valid_data['formation_energy_eV'],
                                     c=halogen_colors[halogen], marker=halogen_markers[halogen],
                                     label=f'{halogen}', alpha=0.8, s=100, 
                                     edgecolors='black', linewidth=1)
            
                # Style each subplot
                ax.set_title(f'n={n_val}', fontweight='bold', fontsize=20)
                ax.set_xlabel(f'{param_title} {unit}', fontweight='bold', fontsize=16)
                if i == 0:  # Only label y-axis on the first subplot
                    ax.set_ylabel('Formation Energy per Atom (eV/atom)', fontweight='bold', fontsize=16)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.set_ylim(y_limits)
                
                # Add legend only to the first subplot
                if i == 0:
                    ax.legend(fontsize=14, title='Halogen', title_fontsize=16)
            
            plt.tight_layout()
            
            # Save with descriptive filename
            filename = f'energy_vs_{param}_analysis.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created continuous plot for {param_title}: {filename}")
            continue  # Skip the rest of the single-plot logic
        
        # This section is now only reached by categorical plots
        # Style other violin plot elements (for categorical plots)
        if plot_type == 'categorical' and 'violin_parts' in locals():
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
                if partname in violin_parts:
                    violin_parts[partname].set_edgecolor('black')
                    violin_parts[partname].set_linewidth(2.0)
            
            # Set title and labels for categorical plots
            ax.set_title(f'Formation Energy vs {param_title}', fontweight='bold', fontsize=24, pad=20)
            ax.set_ylabel('Formation Energy per Atom (eV/atom)', fontweight='bold', fontsize=20)
            
            if unit:
                ax.set_xlabel(f'{param_title} ({unit})', fontweight='bold', fontsize=20)
            else:
                ax.set_xlabel(f'{param_title}', fontweight='bold', fontsize=20)
            
            # Style the plot
            ax.grid(True, alpha=0.3, linewidth=1.0)
            ax.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=6)
            
            # Set appropriate y-axis limits (formation energy can be positive or negative)
            energy_min = df['formation_energy_eV'].min()
            energy_max = df['formation_energy_eV'].max()
            energy_range = energy_max - energy_min
            margin = energy_range * 0.1
            ax.set_ylim(energy_min - margin, energy_max + margin)
            
            plt.tight_layout()
            
            # Save with descriptive filename
            filename = f'energy_vs_{param}_analysis.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created categorical plot for {param_title}: {filename}")
    
    # Create comprehensive correlation heatmap
    create_energy_correlation_heatmap(df, output_dir)
    
    # Create combined halogen-n_slab analysis
    create_combined_halogen_n_analysis(df, output_dir)
    
    # Create comprehensive molecular family analysis (from energy_analysis.py)
    create_molecular_family_analysis(df, output_dir)
    
    # Create original molecular family energy plots
    create_molecular_family_energy_plots(df, output_dir)
    
    # Print statistical summary
    print_energy_statistical_summary(df)

def create_energy_correlation_heatmap(df, output_dir):
    """Create a correlation heatmap showing energy correlations with all structural parameters."""
    
    print("Creating energy correlation heatmap...")
    
    # Select relevant numerical columns for correlation analysis (only formation energy and structural parameters)
    correlation_columns = [
        'formation_energy_eV',
        'lattice_a', 'lattice_b', 'lattice_c', 'alpha', 'beta', 'gamma',
        'cell_volume', 'n_slab', 'total_atoms',
        'zeta', 'delta', 'sigma', 'theta_mean', 'theta_min', 'theta_max',
        'mean_bond_distance', 'bond_distance_variance',
        'cis_angle_mean', 'trans_angle_mean',
        'vector_distance_between_centers_angstrom', 'vector_angle_between_planes_deg'
    ]
    
    # Filter to only columns that exist and have some non-NaN values
    available_columns = [col for col in correlation_columns 
                        if col in df.columns and not df[col].isna().all()]
    
    # Calculate correlations
    corr_data = df[available_columns].corr()
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Create correlation heatmap
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    sns.heatmap(corr_data, mask=mask, annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
    
    ax.set_title('Formation Energy Correlation Matrix: Formation Energy vs All Structural Parameters',
                 fontweight='bold', fontsize=20, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_correlation_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created energy correlation heatmap")

def extract_molecular_info(structure_name):
    """
    Extract molecular information from structure name (enhanced version from energy_analysis.py)
    Returns: (chemical_formula, family, carbon_count)
    """
    # Extract the molecular part from structure name
    # Format: MAPbBr3_n1_[molecular_formula]
    parts = structure_name.split('_')
    if len(parts) >= 3:
        # Join all parts after n1/n2/n3
        molecular_parts = []
        for i in range(2, len(parts)):
            molecular_parts.append(parts[i])
        
        molecular_formula = '_'.join(molecular_parts)
        
        # Enhanced family determination with better pattern matching
        if 'C1=CC=C' in molecular_formula or 'C=C' in molecular_formula:
            # Aromatic families - benzene rings and aromatic systems
            family = 'Aromatic'
            carbon_count = molecular_formula.count('C')
        elif 'C1CCC' in molecular_formula or 'CCC(' in molecular_formula:
            # Cyclic aliphatic systems
            family = 'Cyclic'
            carbon_count = molecular_formula.count('C')
        elif 'CC(C)' in molecular_formula or '(C)' in molecular_formula:
            # Branched alkyl chains - contains branching points
            family = 'Branched'
            carbon_count = molecular_formula.count('C')
        else:
            # Linear alkyl chains [NH3+]CCC[NH3+] etc.
            family = 'Linear'
            # Count carbons - improved counting
            if '[NH3+]' in molecular_formula:
                # Extract the carbon chain between NH3+ groups
                import re
                # Find continuous carbon chains
                carbon_matches = re.findall(r'C+', molecular_formula)
                carbon_count = sum(len(match) for match in carbon_matches)
            else:
                carbon_count = molecular_formula.count('C')
        
        return molecular_formula, family, carbon_count
    
    return 'Unknown', 'Unknown', 0

def create_molecular_family_analysis(df, output_dir):
    """
    Create comprehensive molecular family analysis plots (from energy_analysis.py)
    """
    print("Creating molecular family analysis...")
    
    # Add molecular information to dataframe
    df_with_molecules = df.copy()
    molecular_info = df_with_molecules['structure_name'].apply(extract_molecular_info)
    df_with_molecules['molecular_formula'] = molecular_info.apply(lambda x: x[0])
    df_with_molecules['family'] = molecular_info.apply(lambda x: x[1])
    df_with_molecules['carbon_count'] = molecular_info.apply(lambda x: x[2])
    
    # Filter out unknown families
    df_clean = df_with_molecules[df_with_molecules['family'] != 'Unknown'].copy()
    
    if len(df_clean) == 0:
        print("No valid molecular data found for family analysis")
        return
    
    # Define color palette for families
    family_colors = {
        'Linear': '#1f77b4',     # Blue
        'Branched': '#ff7f0e',   # Orange
        'Cyclic': '#2ca02c',     # Green
        'Aromatic': '#d62728'    # Red
    }
    
    # Define halogen colors
    halogen_colors = {
        'Cl': '#2E8B57',  # Sea green
        'Br': '#FF6B35',  # Orange-red  
        'I': '#4A90E2'    # Blue
    }
    
    # 1. Family Energy Box Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    family_order = ['Linear', 'Branched', 'Cyclic', 'Aromatic']
    plot_data = []
    plot_labels = []
    plot_colors = []
    
    for family in family_order:
        family_data = df_clean[df_clean['family'] == family]
        if len(family_data) > 0:
            plot_data.append(family_data['formation_energy_eV'].values)
            plot_labels.append(f'{family}\n(n={len(family_data)})')
            plot_colors.append(family_colors[family])
    
    if plot_data:
        box_plot = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box_plot[element], color='black', linewidth=1)
    
    ax.set_title('Formation Energy per Atom by Molecular Family', fontweight='bold', fontsize=24, pad=20)
    ax.set_ylabel('Formation Energy per Atom (eV/atom)', fontweight='bold', fontsize=20)
    ax.set_xlabel('Molecular Family', fontweight='bold', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_by_molecular_family.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Energy vs Carbon Count by Family
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Define markers for different families
    family_markers = {'Linear': 'o', 'Branched': 's', 'Cyclic': '^', 'Aromatic': 'D'}
    
    for family in family_order:
        family_data = df_clean[df_clean['family'] == family]
        if len(family_data) > 0:
            ax.scatter(family_data['carbon_count'], family_data['formation_energy_eV'],
                      c=family_colors[family], marker=family_markers.get(family, 'o'),
                      label=f'{family} (n={len(family_data)})', alpha=0.7, s=60,
                      edgecolors='black', linewidth=0.5)
    
    ax.set_title('Formation Energy vs Carbon Count by Molecular Family', fontweight='bold', fontsize=24, pad=20)
    ax.set_ylabel('Formation Energy per Atom (eV/atom)', fontweight='bold', fontsize=20)
    ax.set_xlabel('Number of Carbon Atoms', fontweight='bold', fontsize=20)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_vs_carbon_count_by_family.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Family-Halogen Combined Analysis
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    
    # Prepare data for grouped violin plots
    plot_data_all = []
    plot_positions_all = []
    plot_colors_all = []
    plot_labels = []
    plot_positions_unique = []
    
    position = 1
    
    # Process each family
    for family in family_order:
        family_data = df_clean[df_clean['family'] == family]
        
        if len(family_data) == 0:
            continue
        
        # For each family, create separate violins for each halogen
        for halogen in ['Cl', 'Br', 'I']:
            halogen_data = family_data[family_data['halogen_X'] == halogen]
            
            if len(halogen_data) > 0:
                values = halogen_data['formation_energy_eV'].values
                plot_data_all.append(values)
                plot_positions_all.append(position)
                plot_colors_all.append(halogen_colors[halogen])
        
        # Store unique positions and labels (only once per family)
        plot_positions_unique.append(position)
        plot_labels.append(f'{family}\n(n={len(family_data)})')
        position += 1
    
    if len(plot_data_all) > 0:
        # Create violin plot with overlapping halogens
        violin_parts = ax.violinplot(plot_data_all, positions=plot_positions_all, widths=0.6,
                                   showmeans=True, showmedians=True)
        
        # Color the violin plots by halogen with transparency
        for patch, color in zip(violin_parts['bodies'], plot_colors_all):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.0)
        
        # Style other violin plot elements
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
            if partname in violin_parts:
                violin_parts[partname].set_edgecolor('black')
                violin_parts[partname].set_linewidth(1.5)
        
        # Set x-axis labels using unique positions
        ax.set_xticks(plot_positions_unique)
        ax.set_xticklabels(plot_labels)
        
        # Add family background shading
        family_boundaries = []
        for i, family in enumerate(family_order):
            if i < len(plot_positions_unique):
                ax.axvspan(i + 0.5, i + 1.5, facecolor=family_colors[family], alpha=0.1, zorder=0)
        
        # Add dual legend for both families and halogens
        halogen_legend_elements = [plt.Rectangle((0,0),1,1, facecolor=halogen_colors[halogen], 
                                                alpha=0.6, label=f'{halogen} (violins)') 
                                 for halogen in ['Cl', 'Br', 'I'] if halogen in df_clean['halogen_X'].values]
        
        ax.legend(handles=halogen_legend_elements, loc='upper right', fontsize=14, 
                 title='Halogens', title_fontsize=16)
    
    ax.set_title('Formation Energy by Molecular Family and Halogen', fontweight='bold', fontsize=24, pad=20)
    ax.set_ylabel('Formation Energy per Atom (eV/atom)', fontweight='bold', fontsize=20)
    ax.set_xlabel('Molecular Family', fontweight='bold', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_molecular_families_by_halogen.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create family summary table
    create_family_summary_table(df_clean, output_dir)
    
    print("Molecular family analysis complete!")

def create_family_summary_table(df, output_dir):
    """
    Create comprehensive molecular family summary table (from energy_analysis.py)
    """
    print("\nCreating molecular family summary table...")
    print("\n" + "="*80)
    print("MOLECULAR FAMILY ENERGY SUMMARY")
    print("="*80)
    
    # Define family order
    family_order = ['Linear', 'Branched', 'Cyclic', 'Aromatic']
    summary_data = []
    
    for family in family_order:
        family_data = df[df['family'] == family]
        if len(family_data) == 0:
            continue
            
        # Get carbon count range for this family
        carbon_counts = sorted(family_data['carbon_count'].unique())
        carbon_range = f"C{min(carbon_counts)}-C{max(carbon_counts)}"
        
        # Calculate energy statistics
        energy_values = family_data['formation_energy_eV'].dropna()
        
        if len(energy_values) > 0:
            row_data = {
                'Family': family,
                'Carbon Range': carbon_range,
                'N Samples': len(family_data),
                'Energy Mean': energy_values.mean(),
                'Energy Std': energy_values.std(),
                'Energy Min': energy_values.min(),
                'Energy Max': energy_values.max()
            }
            summary_data.append(row_data)
    
    # Create DataFrame for easier handling
    summary_df = pd.DataFrame(summary_data)
    
    if len(summary_df) == 0:
        print("No family data available for summary table")
        return None
    
    # Print formatted summary table
    print(f"\n{'Family':<10} {'Range':<8} {'N':<3} {'Formation Energy (eV/atom)':<25} {'Range':<15}")
    print("-" * 75)
    
    for _, row in summary_df.iterrows():
        family = row['Family']
        carbon_range = row['Carbon Range']
        n_samples = int(row['N Samples'])
        
        energy_mean = row['Energy Mean']
        energy_std = row['Energy Std']
        energy_min = row['Energy Min']
        energy_max = row['Energy Max']
        
        energy_str = f"{energy_mean:.4f}±{energy_std:.4f}"
        range_str = f"{energy_min:.4f} to {energy_max:.4f}"
        
        print(f"{family:<10} {carbon_range:<8} {n_samples:<3} {energy_str:<25} {range_str:<15}")
    
    # Save detailed CSV
    if len(summary_df) > 0:
        csv_filename = os.path.join(output_dir, 'molecular_family_energy_summary.csv')
        summary_df.to_csv(csv_filename, index=False, float_format='%.6f')
        print(f"\nDetailed energy summary saved to: {csv_filename}")
    
    # Detailed family analysis with energy statistics by halogen
    print(f"\n{'='*80}")
    print("DETAILED FAMILY ENERGY ANALYSIS (Formation Energy per Atom)")
    print(f"{'='*80}")
    
    for family in family_order:
        family_data = df[df['family'] == family]
        if len(family_data) == 0:
            continue
            
        print(f"\n{family.upper()} FAMILY ({len(family_data)} total samples):")
        print("-" * 60)
        
        # Overall family statistics
        energy_values = family_data['formation_energy_eV'].dropna()
        if len(energy_values) > 0:
            print(f"  OVERALL FORMATION ENERGY: {energy_values.mean():.4f} ± {energy_values.std():.4f} eV/atom")
        
        # By halogen within family
        print("  BY HALOGEN:")
        for halogen in ['Cl', 'Br', 'I']:
            halogen_family_data = family_data[family_data['halogen_X'] == halogen]
            if len(halogen_family_data) > 0:
                energy_values = halogen_family_data['formation_energy_eV'].dropna()
                if len(energy_values) > 0:
                    print(f"    {halogen} ({len(halogen_family_data)} samples): {energy_values.mean():.4f} ± {energy_values.std():.4f} eV/atom")
        
        # By n_slab within family
        print("  BY NUMBER OF LAYERS:")
        for n_slab in sorted(family_data['n_slab'].unique()):
            n_family_data = family_data[family_data['n_slab'] == n_slab]
            if len(n_family_data) > 0:
                energy_values = n_family_data['formation_energy_eV'].dropna()
                if len(energy_values) > 0:
                    print(f"    n={n_slab} ({len(n_family_data)} samples): {energy_values.mean():.4f} ± {energy_values.std():.4f} eV/atom")
    
    return summary_df

def create_combined_halogen_n_analysis(df, output_dir):
    """Create combined analysis plots showing energy vs halogen-n_slab combinations."""
    
    print("Creating combined halogen-n_slab energy analysis...")
    
    halogen_colors = {'Cl': '#2E8B57', 'Br': '#FF6B35', 'I': '#4A90E2'}
    halogens = ['Cl', 'Br', 'I']
    n_values = sorted(df['n_slab'].unique())
    
    # 1. Box plot with double grouping
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    plot_data = []
    plot_labels = []
    plot_colors = []
    
    for halogen in halogens:
        for n in n_values:
            subset = df[(df['halogen_X'] == halogen) & (df['n_slab'] == n)]
            if len(subset) > 0:
                plot_data.append(subset['formation_energy_eV'].values)
                plot_labels.append(f'{halogen}-n{n}')
                plot_colors.append(halogen_colors[halogen])
    
    if plot_data:
        positions = list(range(1, len(plot_data) + 1))
        violin_parts = ax.violinplot(plot_data, positions=positions, widths=0.6,
                                   showmeans=True, showmedians=True)
        
        # Color the violin plots
        for i, (patch, color) in enumerate(zip(violin_parts['bodies'], plot_colors)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        # Style other elements
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
            if partname in violin_parts:
                violin_parts[partname].set_edgecolor('black')
                violin_parts[partname].set_linewidth(2.0)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(plot_labels)
    
    ax.set_title('Formation Energy by Halogen and Number of Layers (Combined Analysis)',
                 fontweight='bold', fontsize=24, pad=20)
    ax.set_ylabel('Formation Energy per Atom (eV/atom)', fontweight='bold', fontsize=20)
    ax.set_xlabel('Halogen-Layer Combination', fontweight='bold', fontsize=20)
    ax.grid(True, alpha=0.3, linewidth=1.0)
    ax.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=6)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add legend for halogens
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=halogen_colors[hal], alpha=0.7, label=hal)
                      for hal in halogens]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=16, title='Halogen', title_fontsize=18)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_combined_halogen_n_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created combined halogen-n_slab energy analysis")

def create_molecular_family_energy_plots(df, output_dir):
    """
    Create energy analysis plots by molecular family, similar to vector_analysis.py
    """
    print("Creating molecular family energy analysis...")
    
    # Add molecular information to dataframe
    df_with_molecules = df.copy()
    molecular_info = df_with_molecules['structure_name'].apply(extract_molecular_info)
    df_with_molecules['molecular_formula'] = molecular_info.apply(lambda x: x[0])
    df_with_molecules['family'] = molecular_info.apply(lambda x: x[1])
    df_with_molecules['carbon_count'] = molecular_info.apply(lambda x: x[2])
    
    # Filter out unknown families
    df_clean = df_with_molecules[df_with_molecules['family'] != 'Unknown'].copy()
    
    if len(df_clean) == 0:
        print("No valid molecular data found for energy plotting")
        return
    
    # Define color palette for families
    family_colors = {
        'Linear': '#1f77b4',     # Blue
        'Branched': '#ff7f0e',   # Orange
        'Cyclic': '#2ca02c',     # Green
        'Aromatic': '#d62728'    # Red
    }
    
    # Define halogen colors (with transparency)
    halogen_colors = {
        'Cl': '#2E8B57',  # Sea green
        'Br': '#FF6B35',  # Orange-red  
        'I': '#4A90E2'    # Blue
    }
    
    # Define family order
    family_order = ['Linear', 'Branched', 'Cyclic', 'Aromatic']
    
    # Create energy vs molecular family plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    
    # Prepare data for violin plots - ordered by family and carbon count
    plot_data_all = []
    plot_positions_all = []
    plot_colors_all = []
    plot_labels = []
    plot_positions_unique = []
    
    position = 1
    
    # Process each family in order
    for family in family_order:
        family_data = df_clean[df_clean['family'] == family]
        
        if len(family_data) == 0:
            continue
        
        # Get unique carbon counts for this family, sorted
        carbon_counts = sorted(family_data['carbon_count'].unique())
        
        # Process each carbon count within the family
        for carbon_count in carbon_counts:
            carbon_data = family_data[family_data['carbon_count'] == carbon_count]
            
            if len(carbon_data) > 0:
                # Create separate violin for each halogen at the same position
                for halogen in ['Cl', 'Br', 'I']:
                    halogen_data = carbon_data[carbon_data['halogen_X'] == halogen]
                    
                    if len(halogen_data) > 0:
                        values = halogen_data['formation_energy_eV'].values
                        plot_data_all.append(values)
                        plot_positions_all.append(position)
                        plot_colors_all.append(halogen_colors[halogen])
                
                # Store unique positions and labels (only once per family-carbon combination)
                plot_positions_unique.append(position)
                plot_labels.append(f'{family}\nC{carbon_count}')
                position += 1
    
    if len(plot_data_all) == 0:
        print("No data to plot for molecular family energy analysis")
        plt.close()
        return
    
    # Create violin plot with overlapping halogens
    violin_parts = ax.violinplot(plot_data_all, positions=plot_positions_all, widths=0.5, 
                               showmeans=True, showmedians=True)
    
    # Color the violin plots by halogen with transparency
    for patch, color in zip(violin_parts['bodies'], plot_colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)  # Increased transparency for overlapping
        patch.set_edgecolor('black')
        patch.set_linewidth(1.0)  # Thinner lines for less visual clutter
    
    # Style other violin plot elements with transparency
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        if partname in violin_parts:
            violin_parts[partname].set_edgecolor('black')
            violin_parts[partname].set_linewidth(1.5)
            violin_parts[partname].set_alpha(0.8)
    
    # Set x-axis labels using unique positions
    ax.set_xticks(plot_positions_unique)
    ax.set_xticklabels(plot_labels, rotation=45, ha='right')
    
    # Set title and labels
    ax.set_title('Formation Energy per Atom by Molecular Family and Carbon Count', 
                 fontweight='bold', fontsize=24, pad=20)
    ax.set_ylabel('Formation Energy per Atom (eV/atom)', fontweight='bold', fontsize=20)
    ax.set_xlabel('Molecular Family & Carbon Count', fontweight='bold', fontsize=20)
    
    # Style the plot
    ax.grid(True, alpha=0.3, linewidth=1.0)
    ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
    
    # Add vertical lines to separate families
    family_boundaries = []
    current_pos = 1
    for family in family_order:
        family_data = df_clean[df_clean['family'] == family]
        if len(family_data) > 0:
            carbon_counts = sorted(family_data['carbon_count'].unique())
            current_pos += len(carbon_counts)
            if current_pos <= len(plot_positions_unique):
                family_boundaries.append(current_pos - 0.5)
    
    # Add family background shading
    family_start = 1
    for i, family in enumerate(family_order):
        family_data = df_clean[df_clean['family'] == family]
        if len(family_data) > 0:
            carbon_counts = sorted(family_data['carbon_count'].unique())
            family_end = family_start + len(carbon_counts)
            
            # Add background shading for this family
            ax.axvspan(family_start - 0.5, family_end - 0.5, 
                      facecolor=family_colors[family], alpha=0.1, zorder=0)
            
            family_start = family_end
    
    # Draw family boundary lines
    for boundary in family_boundaries:  # Don't draw line after last family
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add dual legend for both families and halogens
    # Halogen legend (violin colors)
    halogen_legend_elements = [plt.Rectangle((0,0),1,1, facecolor=halogen_colors[halogen], 
                                            alpha=0.6, label=f'{halogen} (violins)') 
                             for halogen in ['Cl', 'Br', 'I'] if halogen in df_clean['halogen_X'].values]
    
    # Combine legends
    ax.legend(handles=halogen_legend_elements, loc='upper right', fontsize=14, 
             title='Halogens', title_fontsize=16, ncol=1)
    
    plt.tight_layout()
    
    # Save with descriptive filename
    filename = 'energy_molecular_families_by_carbon.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created molecular family energy plot: {filename}")
    
    # Create family summary table
    create_family_energy_summary_table(df_clean, output_dir)

def create_family_energy_summary_table(df, output_dir):
    """
    Create comprehensive energy summary table by molecular family
    """
    print("\nCreating molecular family energy summary table...")
    print("\n" + "="*80)
    print("MOLECULAR FAMILY ENERGY SUMMARY")
    print("="*80)
    
    # Define family order
    family_order = ['Linear', 'Branched', 'Cyclic', 'Aromatic']
    summary_data = []
    
    for family in family_order:
        family_data = df[df['family'] == family]
        if len(family_data) == 0:
            continue
            
        # Get carbon count range for this family
        carbon_counts = sorted(family_data['carbon_count'].unique())
        carbon_range = f"C{min(carbon_counts)}-C{max(carbon_counts)}"
        
        # Calculate energy statistics
        energy_values = family_data['formation_energy_eV'].dropna()
        
        if len(energy_values) > 0:
            row_data = {
                'Family': family,
                'Carbon Range': carbon_range,
                'N Samples': len(family_data),
                'Energy Mean': energy_values.mean(),
                'Energy Std': energy_values.std(),
                'Energy Min': energy_values.min(),
                'Energy Max': energy_values.max()
            }
            summary_data.append(row_data)
    
    # Create DataFrame for easier handling
    summary_df = pd.DataFrame(summary_data)
    
    if len(summary_df) == 0:
        print("No family data available for energy summary table")
        return None
    
    # Print formatted summary table
    print(f"\n{'Family':<10} {'Range':<8} {'N':<3} {'Energy (eV/atom)':<20} {'Range':<15}")
    print("-" * 70)
    
    for _, row in summary_df.iterrows():
        family = row['Family']
        carbon_range = row['Carbon Range']
        n_samples = int(row['N Samples'])
        
        energy_mean = row['Energy Mean']
        energy_std = row['Energy Std']
        energy_min = row['Energy Min']
        energy_max = row['Energy Max']
        
        energy_str = f"{energy_mean:.3f}±{energy_std:.3f}"
        range_str = f"{energy_min:.3f} to {energy_max:.3f}"
        
        print(f"{family:<10} {carbon_range:<8} {n_samples:<3} {energy_str:<20} {range_str:<15}")
    
    # Save detailed CSV
    if len(summary_df) > 0:
        csv_filename = os.path.join(output_dir, 'molecular_family_energy_summary.csv')
        summary_df.to_csv(csv_filename, index=False, float_format='%.6f')
        print(f"\nDetailed energy summary saved to: {csv_filename}")
    
    # Detailed family analysis with energy statistics by halogen
    print(f"\n{'='*80}")
    print("DETAILED FAMILY ENERGY ANALYSIS (Mean ± Standard Deviation)")
    print(f"{'='*80}")
    
    for family in family_order:
        family_data = df[df['family'] == family]
        if len(family_data) == 0:
            continue
            
        print(f"\n{family.upper()} FAMILY ({len(family_data)} total samples):")
        print("-" * 60)
        
        # Overall family statistics
        energy_values = family_data['formation_energy_eV'].dropna()
        if len(energy_values) > 0:
            print(f"  OVERALL FORMATION ENERGY: {energy_values.mean():.4f} ± {energy_values.std():.4f} eV/atom")
        
        # By halogen within family
        print("  BY HALOGEN:")
        for halogen in ['Cl', 'Br', 'I']:
            halogen_family_data = family_data[family_data['halogen_X'] == halogen]
            if len(halogen_family_data) > 0:
                energy_values = halogen_family_data['formation_energy_eV'].dropna()
                if len(energy_values) > 0:
                    print(f"    {halogen} ({len(halogen_family_data)} samples): {energy_values.mean():.4f} ± {energy_values.std():.4f} eV/atom")
        
        # By n_slab within family
        print("  BY NUMBER OF LAYERS:")
        for n_slab in sorted(family_data['n_slab'].unique()):
            n_family_data = family_data[family_data['n_slab'] == n_slab]
            if len(n_family_data) > 0:
                energy_values = n_family_data['formation_energy_eV'].dropna()
                if len(energy_values) > 0:
                    print(f"    n={n_slab} ({len(n_family_data)} samples): {energy_values.mean():.4f} ± {energy_values.std():.4f} eV/atom")
    
    # Family comparison summary
    print(f"\n{'='*80}")
    print("FAMILY ENERGY COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print("\nFormation Energy per Atom (eV/atom):")
    print("-" * 50)
    
    for family in family_order:
        family_values = df[df['family'] == family]['formation_energy_eV'].dropna()
        if len(family_values) > 0:
            print(f"  {family:<10}: {family_values.mean():.4f} ± {family_values.std():.4f} (n={len(family_values)})")
    
    print(f"\n{'='*80}")
    return summary_df

def print_energy_statistical_summary(df):
    """Print comprehensive statistical summary of energy analysis."""
    
    print("\n" + "="*80)
    print("FORMATION ENERGY ANALYSIS STATISTICAL SUMMARY")
    print("="*80)
    
    # Overall energy statistics
    print(f"\nOverall Formation Energy Statistics:")
    print("-" * 50)
    print(f"Total structures: {len(df)}")
    print(f"Formation energy range: {df['formation_energy_eV'].min():.4f} to {df['formation_energy_eV'].max():.4f} eV/atom")
    print(f"Mean formation energy: {df['formation_energy_eV'].mean():.4f} ± {df['formation_energy_eV'].std():.4f} eV/atom")
    
    # Energy by halogen
    print(f"\nFormation Energy by Halogen:")
    print("-" * 50)
    for halogen in ['Cl', 'Br', 'I']:
        subset = df[df['halogen_X'] == halogen]
        if len(subset) > 0:
            mean_energy = subset['formation_energy_eV'].mean()
            std_energy = subset['formation_energy_eV'].std()
            print(f"{halogen} (n={len(subset)}): {mean_energy:.4f} ± {std_energy:.4f} eV/atom")
    
    # Energy by number of layers
    print(f"\nFormation Energy by Number of Layers:")
    print("-" * 55)
    for n in sorted(df['n_slab'].unique()):
        subset = df[df['n_slab'] == n]
        if len(subset) > 0:
            mean_energy = subset['formation_energy_eV'].mean()
            std_energy = subset['formation_energy_eV'].std()
            print(f"n={n} (n={len(subset)}): {mean_energy:.4f} ± {std_energy:.4f} eV/atom")
    
    # Combined analysis
    print(f"\nCombined Halogen-Layer Analysis:")
    print("-" * 50)
    for halogen in ['Cl', 'Br', 'I']:
        halogen_data = df[df['halogen_X'] == halogen]
        if len(halogen_data) == 0:
            continue
            
        print(f"\n{halogen} Analysis:")
        for n in sorted(halogen_data['n_slab'].unique()):
            subset = halogen_data[halogen_data['n_slab'] == n]
            if len(subset) > 0:
                mean_energy = subset['formation_energy_eV'].mean()
                std_energy = subset['formation_energy_eV'].std()
                print(f"  n={n} (samples={len(subset)}): {mean_energy:.4f} ± {std_energy:.4f} eV/atom")
    
    # Correlation analysis
    print(f"\nKey Energy-Structure Correlations:")
    print("-" * 50)
    
    # Check correlations with key structural parameters
    key_params = [
        ('lattice_a', 'Lattice parameter a'),
        ('lattice_c', 'Lattice parameter c'),
        ('cell_volume', 'Cell volume'),
        ('sigma', 'Octahedral distortion (σ)'),
        ('mean_bond_distance', 'Mean bond distance'),
        ('vector_distance_between_centers_angstrom', 'Plane separation distance')
    ]
    
    for param, param_name in key_params:
        if param in df.columns:
            valid_data = df[[param, 'formation_energy_eV']].dropna()
            if len(valid_data) > 5:
                correlation = valid_data[param].corr(valid_data['formation_energy_eV'])
                print(f"Formation energy vs {param_name}: r = {correlation:.3f}")
    
    # Most stable and least stable structures
    print(f"\nMost Stable Structures (Lowest Formation Energy):")
    print("-" * 60)
    most_stable = df.nsmallest(5, 'formation_energy_eV')[['structure_name', 'halogen_X', 'n_slab', 'formation_energy_eV']]
    for _, row in most_stable.iterrows():
        print(f"{row['structure_name'][:50]}: {row['formation_energy_eV']:.4f} eV/atom ({row['halogen_X']}-n{row['n_slab']})")
    
    print(f"\nLeast Stable Structures (Highest Formation Energy):")
    print("-" * 60)
    least_stable = df.nlargest(5, 'formation_energy_eV')[['structure_name', 'halogen_X', 'n_slab', 'formation_energy_eV']]
    for _, row in least_stable.iterrows():
        print(f"{row['structure_name'][:50]}: {row['formation_energy_eV']:.4f} eV/atom ({row['halogen_X']}-n{row['n_slab']})")

def main():
    # Set base directory
    base_dir = '/home/dotempo/Documents/SVC-Materials/Graphs/DOS_END_RESULTS'
    output_dir = os.path.join(base_dir, 'energy_analysis')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting comprehensive DOS_END energy analysis...")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print("\nThis script uses pre-calculated formation energies from extract_dos_end_energies.py")
    print("If no energy data is found, please run extract_dos_end_energies.py first.")
    
    # Load data
    df = load_dos_end_energy_data()
    
    if df.empty:
        print("\n❌ No energy data found!")
        print("Please run the following command first:")
        print("python extract_dos_end_energies.py")
        return
    
    print(f"Processed {len(df)} structures with energy data")
    
    # Create plots
    print("Creating energy analysis visualizations...")
    create_energy_structure_plots(df, output_dir)
    
    # Save the dataframe for further analysis
    df.to_csv(os.path.join(output_dir, 'formation_energy_structural_data.csv'), index=False)
    
    print(f"\nFormation Energy Analysis Complete! Check the '{output_dir}' directory for results.")
    print("="*80)
    print("Generated files (all using Formation Energy per Atom < 0 eV/atom):")
    print("- energy_vs_halogen_X_analysis.png")
    print("- energy_vs_n_slab_analysis.png")
    print("- energy_vs_lattice_*_analysis.png (a, b, c, angles)")
    print("- energy_vs_*_distortion_analysis.png (zeta, delta, sigma)")
    print("- energy_vs_cell_volume_analysis.png")
    print("- energy_vs_mean_bond_distance_analysis.png")
    print("- energy_vs_vector_*_analysis.png (plane analysis)")
    print("- energy_correlation_heatmap.png")
    print("- energy_combined_halogen_n_analysis.png")
    print("- energy_by_molecular_family.png")
    print("- energy_vs_carbon_count_by_family.png")
    print("- energy_molecular_families_by_halogen.png")
    print("- energy_molecular_families_by_carbon.png")
    print("- molecular_family_energy_summary.csv")
    print("- formation_energy_structural_data.csv")
    print("\nAll plots use pre-calculated formation energies from extract_dos_end_energies.py")
    print("Formation energy formula: E_formation = E_total - (NX * EX + NB * EPb + NMA * EMA + 2 * EMOL)")
    print("Where EMOL is the molecular energy from molecule_data.csv")
    print("Values are normalized per atom for comparison across different structures.")
    print("✅ Using actual DFT formation energies with molecular contributions!")

if __name__ == "__main__":
    main()

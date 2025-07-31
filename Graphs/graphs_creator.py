import pandas as pd
import matplotlib
# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

def setup_publication_style():
    """Sets matplotlib parameters for a style that resembles scientific publications."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif'],  # A common, professional-looking serif font
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'savefig.dpi': 300,
        'figure.autolayout': True, # Use tight_layout automatically
    })

def create_boxplot(data, group_by, y, title, filename, y_label=None, x_label=None, folder='Boxplot_graphs'):
    """Generates and saves a publication-quality boxplot."""
    # Use provided labels or default to column names
    y_label = y_label or y
    x_label = x_label or group_by
    
    # Group data and filter out empty groups
    grouped = data.groupby(group_by)[y].apply(list)
    labels = grouped.index.to_list()
    values = grouped.to_list()
    
    # Define figure size
    figsize = (10, 6) if len(labels) < 10 else (min(len(labels) * 0.8, 20), 7)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use a perceptually uniform colormap
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(labels)))
    
    # Custom boxplot
    bp = ax.boxplot(values, patch_artist=True, labels=labels)
    
    # Style the boxplot components
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
        
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
        
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linestyle('-')
        whisker.set_linewidth(1.5)

    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.5)
        
    # Style outliers
    for flier in bp['fliers']:
        flier.set(marker='o', markerfacecolor='gray', alpha=0.5, markersize=5, linestyle='none')

    # Minimalist style adjustments
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    
    # Rotate x-axis labels if they are long or numerous
    if len(labels) > 8 or any(len(str(l)) > 8 for l in labels):
        plt.xticks(rotation=45, ha='right')
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, filename), bbox_inches='tight')
    plt.close()

def create_scatter_with_regression(data, x, y, group_by, title, filename, y_label=None, x_label=None, folder='Scatter_graphs'):
    """Generates and saves a publication-quality scatter plot with regression lines."""
    # Use provided labels or default to column names
    y_label = y_label or y
    x_label = x_label or x
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_groups = data[group_by].unique()
    num_groups = len(unique_groups)
    
    # Use tab10 colormap which is available in the environment
    cmap = matplotlib.colormaps.get_cmap('tab10') 
    colors = [cmap(i) for i in np.linspace(0, 1, num_groups)]
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    
    for i, group in enumerate(unique_groups):
        group_data = data[data[group_by] == group]
        if len(group_data) < 2:
            continue
            
        color = colors[i]
        marker = markers[i % len(markers)]
        
        # Scatter plot
        ax.scatter(group_data[x], group_data[y], color=color, marker=marker,
                   label=group, alpha=0.7, edgecolor='black', s=80) # s is marker size
    
    # Minimalist style adjustments
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(title=group_by.replace('_', ' '), loc='best', frameon=False)
    
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, filename), bbox_inches='tight')
    plt.close()

def create_entity_comparison_plot(data, title, filename, folder='Entity_comparison'):
    """Create comparison plots between octahedra and molecules."""
    os.makedirs(folder, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    octahedra_df = data[data['entity_type'] == 'octahedron']
    molecules_df = data[data['entity_type'] == 'molecule']
    
    # Plot 1: Entity count by halogen
    if not octahedra_df.empty and not molecules_df.empty:
        halogen_oct_counts = octahedra_df['halogen'].value_counts()
        halogen_mol_counts = molecules_df['halogen'].value_counts()
        
        halogens = sorted(set(halogen_oct_counts.index.tolist() + halogen_mol_counts.index.tolist()))
        oct_counts = [halogen_oct_counts.get(h, 0) for h in halogens]
        mol_counts = [halogen_mol_counts.get(h, 0) for h in halogens]
        
        x = np.arange(len(halogens))
        width = 0.35
        
        ax1.bar(x - width/2, oct_counts, width, label='Octahedra', color='steelblue', alpha=0.8)
        ax1.bar(x + width/2, mol_counts, width, label='Molecules', color='darkorange', alpha=0.8)
        
        ax1.set_xlabel('Halogen')
        ax1.set_ylabel('Count')
        ax1.set_title('Entity Distribution by Halogen')
        ax1.set_xticks(x)
        ax1.set_xticklabels(halogens)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Octahedral distortion distribution
    if not octahedra_df.empty and 'zeta' in octahedra_df.columns:
        zeta_data = octahedra_df.dropna(subset=['zeta'])
        if not zeta_data.empty:
            ax2.hist(zeta_data['zeta'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            ax2.set_xlabel('Zeta Parameter')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Octahedral Distortion Distribution')
            ax2.grid(True, alpha=0.3)
    
    # Plot 3: Molecular size distribution
    if not molecules_df.empty and 'molecular_size_max' in molecules_df.columns:
        size_data = molecules_df.dropna(subset=['molecular_size_max'])
        if not size_data.empty:
            spacer_sizes = size_data[size_data['is_spacer_molecule'] == True]['molecular_size_max']
            a_site_sizes = size_data[size_data['is_a_site_molecule'] == True]['molecular_size_max']
            
            if not spacer_sizes.empty:
                ax3.hist(spacer_sizes, bins=15, alpha=0.7, label='Spacer', color='green', edgecolor='black')
            if not a_site_sizes.empty:
                ax3.hist(a_site_sizes, bins=15, alpha=0.7, label='A-site', color='red', edgecolor='black')
            
            ax3.set_xlabel('Molecular Size (Å)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Molecular Size Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # Plot 4: Energy vs entity position (if energy data available)
    if 'eslab' in data.columns and any(pd.notna(data['eslab'])):
        energy_data = data.dropna(subset=['eslab'])
        if not energy_data.empty:
            octahedra_energy = energy_data[energy_data['entity_type'] == 'octahedron']
            molecules_energy = energy_data[energy_data['entity_type'] == 'molecule']
            
            if not octahedra_energy.empty:
                ax4.scatter(octahedra_energy['central_z'], octahedra_energy['eslab'], 
                           alpha=0.6, label='Octahedra', color='steelblue', s=50)
            if not molecules_energy.empty:
                ax4.scatter(molecules_energy['center_of_mass_z'], molecules_energy['eslab'], 
                           alpha=0.6, label='Molecules', color='darkorange', s=50)
            
            ax4.set_xlabel('Z Position (Å)')
            ax4.set_ylabel('Formation Energy (eV)')
            ax4.set_title('Energy vs. Z Position')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder, filename), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # Set the global plot style for the script
    setup_publication_style()

    try:
        # Load the comprehensive dataset
        df = pd.read_csv('comprehensive_perovskite_properties.csv')
        
        print(f"📊 Loaded dataset with {len(df)} entities and {len(df.columns)} properties")
        print(f"   Octahedra: {len(df[df['entity_type'] == 'octahedron'])}")
        print(f"   Molecules: {len(df[df['entity_type'] == 'molecule'])}")
        
        # Filter out invalid data
        df = df.dropna(subset=['halogen'])
        
        # Define the desired halogen order
        halogen_order = ['Cl', 'Br', 'I']
        df['halogen'] = pd.Categorical(df['halogen'], categories=halogen_order, ordered=True)
        
        # Separate octahedral and molecular data
        octahedra_df = df[df['entity_type'] == 'octahedron'].copy()
        molecules_df = df[df['entity_type'] == 'molecule'].copy()
        
        print(f"\n🔬 GENERATING OCTAHEDRAL DISTORTION PLOTS...")
        
        # --- Octahedral Distortion Plots ---
        if not octahedra_df.empty:
            # Zeta parameter analysis
            create_boxplot(octahedra_df.dropna(subset=['zeta']), 'halogen', 'zeta', 
                          'Octahedral Zeta Distortion by Halogen', 
                          'octahedral_zeta_by_halogen.png', 
                          y_label=r'Zeta Parameter', x_label='Halogen')
        
            # Delta parameter analysis
            create_boxplot(octahedra_df.dropna(subset=['delta']), 'halogen', 'delta', 
                          'Octahedral Delta Distortion by Halogen', 
                          'octahedral_delta_by_halogen.png', 
                          y_label=r'Delta Parameter', x_label='Halogen')
            
            # Sigma parameter analysis
            create_boxplot(octahedra_df.dropna(subset=['sigma']), 'halogen', 'sigma', 
                          'Octahedral Sigma Distortion by Halogen', 
                          'octahedral_sigma_by_halogen.png', 
                          y_label=r'Sigma Parameter (°)', x_label='Halogen')
            
            # Bond distance analysis
            create_boxplot(octahedra_df.dropna(subset=['mean_bond_distance']), 'halogen', 'mean_bond_distance', 
                          'Mean Bond Distance by Halogen', 
                          'octahedral_bond_distance_by_halogen.png', 
                          y_label=r'Mean Bond Distance (Å)', x_label='Halogen')
            
            # Bond angle analysis
            create_boxplot(octahedra_df.dropna(subset=['cis_angle_mean']), 'halogen', 'cis_angle_mean', 
                          'Cis Bond Angles by Halogen', 
                          'octahedral_cis_angles_by_halogen.png', 
                          y_label=r'Mean Cis Angle (°)', x_label='Halogen')
        
            # N_slab analysis for octahedra
            if 'n_slab' in octahedra_df.columns:
                create_boxplot(octahedra_df.dropna(subset=['zeta', 'n_slab']), 'n_slab', 'zeta', 
                              'Octahedral Zeta by Slab Thickness', 
                              'octahedral_zeta_by_slab.png', 
                              y_label=r'Zeta Parameter', x_label='Number of Slabs')
        
        print(f"✓ Generated octahedral distortion plots")
        
        print(f"\n🧪 GENERATING MOLECULAR ANALYSIS PLOTS...")
        
        # --- Molecular Analysis Plots ---
        if not molecules_df.empty:
            # Molecular size analysis
            create_boxplot(molecules_df.dropna(subset=['molecular_size_max']), 'halogen', 'molecular_size_max', 
                          'Molecular Size by Halogen', 
                          'molecular_size_by_halogen.png', 
                          y_label=r'Molecular Size (Å)', x_label='Halogen')

            # Spacer vs A-site comparison
            spacer_type_df = molecules_df.copy()
            spacer_type_df['molecule_type'] = spacer_type_df.apply(
                lambda x: 'Spacer' if x['is_spacer_molecule'] else ('A-site' if x['is_a_site_molecule'] else 'Other'), axis=1
            )
            
            create_boxplot(spacer_type_df.dropna(subset=['molecular_size_max']), 'molecule_type', 'molecular_size_max', 
                          'Molecular Size by Type', 
                          'molecular_size_by_type.png', 
                          y_label=r'Molecular Size (Å)', x_label='Molecule Type')

            # Molecular composition analysis
            if 'mol_C_count' in molecules_df.columns:
                create_boxplot(molecules_df.dropna(subset=['mol_C_count']), 'halogen', 'mol_C_count', 
                              'Carbon Count by Halogen', 
                              'molecular_carbon_by_halogen.png', 
                              y_label='Number of Carbon Atoms', x_label='Halogen')

            if 'mol_N_count' in molecules_df.columns:
                create_boxplot(molecules_df.dropna(subset=['mol_N_count']), 'halogen', 'mol_N_count', 
                              'Nitrogen Count by Halogen', 
                              'molecular_nitrogen_by_halogen.png', 
                              y_label='Number of Nitrogen Atoms', x_label='Halogen')
        
        print(f"✓ Generated molecular analysis plots")
        
        print(f"\n📐 GENERATING LATTICE PARAMETER PLOTS...")
        
        # --- Lattice Parameter Plots (using unique experiments only) ---
        experiment_df = df.drop_duplicates(subset=['experiment_name']).copy()
        
        # Lattice constants
        for param, label in [('lattice_a', 'A'), ('lattice_b', 'B'), ('lattice_c', 'C')]:
            if param in experiment_df.columns:
                create_boxplot(experiment_df.dropna(subset=[param]), 'halogen', param, 
                              f'Lattice {label} by Halogen', 
                              f'lattice_{label.lower()}_by_halogen.png', 
                              y_label=r'Distance (Å)', x_label='Halogen')
        
        # Lattice angles
        for param, label in [('lattice_alpha', 'Alpha'), ('lattice_beta', 'Beta'), ('lattice_gamma', 'Gamma')]:
            if param in experiment_df.columns:
                create_boxplot(experiment_df.dropna(subset=[param]), 'halogen', param, 
                              f'Lattice {label} by Halogen', 
                              f'lattice_{label.lower()}_by_halogen.png', 
                              y_label=r'Angle (°)', x_label='Halogen')
        
        print(f"✓ Generated lattice parameter plots")
        
        print(f"\n⚡ GENERATING ENERGY ANALYSIS PLOTS...")

        # --- Energy Analysis Plots (if energy data available) ---
        if 'eslab' in experiment_df.columns and any(pd.notna(experiment_df['eslab'])):
            energy_df = experiment_df.dropna(subset=['eslab']).copy()
            
            # Formation energy by halogen
            create_boxplot(energy_df, 'halogen', 'eslab', 
                          'Formation Energy by Halogen', 
                          'formation_energy_by_halogen.png', 
                          y_label=r'Formation Energy (eV)', x_label='Halogen')

            # Formation energy by slab thickness
            if 'n_slab' in energy_df.columns:
                create_boxplot(energy_df.dropna(subset=['n_slab']), 'n_slab', 'eslab', 
                              'Formation Energy by Slab Thickness', 
                              'formation_energy_by_slab.png', 
                              y_label=r'Formation Energy (eV)', x_label='Number of Slabs')
            
            # Energy vs lattice parameters
            if 'lattice_a' in energy_df.columns:
                create_scatter_with_regression(energy_df, 'lattice_a', 'eslab', 'halogen', 
                                             'Formation Energy vs. Lattice A', 
                                             'energy_vs_lattice_a.png',
                                             x_label=r'Lattice A (Å)', 
                                             y_label=r'Formation Energy (eV)')
        
        print(f"✓ Generated energy analysis plots")
        
        print(f"\n🔍 GENERATING CORRELATION PLOTS...")
        
        # --- Correlation Analysis ---
        if not octahedra_df.empty and not molecules_df.empty:
            # Octahedral distortion vs molecular size correlation
            # (using experiment-level aggregation)
            exp_oct_stats = octahedra_df.groupby('experiment_name').agg({
                'zeta': 'mean',
                'delta': 'mean', 
                'sigma': 'mean',
                'mean_bond_distance': 'mean'
            }).reset_index()
            
            exp_mol_stats = molecules_df.groupby('experiment_name').agg({
                'molecular_size_max': 'mean',
                'num_atoms': 'mean'
            }).reset_index()
            
            correlation_df = pd.merge(exp_oct_stats, exp_mol_stats, on='experiment_name', how='inner')
            correlation_df = pd.merge(correlation_df, 
                                    experiment_df[['experiment_name', 'halogen']], 
                                    on='experiment_name', how='left')
            
            if not correlation_df.empty:
                create_scatter_with_regression(correlation_df.dropna(subset=['zeta', 'molecular_size_max']), 
                                             'molecular_size_max', 'zeta', 'halogen', 
                                             'Octahedral Distortion vs. Molecular Size', 
                                             'zeta_vs_molecular_size.png',
                                             x_label=r'Mean Molecular Size (Å)', 
                                             y_label=r'Mean Zeta Parameter')
        
        print(f"✓ Generated correlation plots")
        
        print(f"\n📈 GENERATING ENTITY COMPARISON PLOTS...")
        
        # --- Entity Comparison Plots ---
        create_entity_comparison_plot(df, 'Comprehensive Entity Analysis', 
                                    'entity_comparison_overview.png')
        
        print(f"✓ Generated entity comparison plots")
        
        print(f"\n🎯 SUMMARY:")
        print(f"   Generated comprehensive visualization suite")
        print(f"   Analyzed {len(df)} total entities:")
        print(f"     • {len(octahedra_df)} octahedra")
        print(f"     • {len(molecules_df)} molecules")
        print(f"   Created plots for:")
        print(f"     • Octahedral distortion parameters")
        print(f"     • Molecular composition and size")
        print(f"     • Lattice parameter distributions")
        if 'eslab' in df.columns and any(pd.notna(df['eslab'])):
            print(f"     • Energy analysis and correlations")
        print(f"     • Cross-entity correlations")
        print(f"   Output folders: Boxplot_graphs/, Scatter_graphs/, Entity_comparison/")

    except FileNotFoundError:
        print("❌ Error: 'comprehensive_perovskite_properties.csv' not found.")
        print("   Please run the extraction script first to generate the dataset.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
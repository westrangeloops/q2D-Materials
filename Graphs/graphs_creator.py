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
    
    # --- FIX ---
    # Replaced 'seaborn-colorblind' with 'tab10', which is available in your environment.
    # Updated the deprecated `plt.cm.get_cmap` to `matplotlib.colormaps.get_cmap`.
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
        
        # Linear regression line
        #slope, intercept = np.polyfit(group_data[x], group_data[y], 1)
        #x_fit = np.linspace(group_data[x].min(), group_data[x].max(), 100)
        #y_fit = slope * x_fit + intercept
        #ax.plot(x_fit, y_fit, color=color, linewidth=2.5)
    
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

if __name__ == '__main__':
    # Set the global plot style for the script
    setup_publication_style()

    try:
        df = pd.read_csv('perovskites.csv')
        df['NormEnergy'] = pd.to_numeric(df['NormEnergy'], errors='coerce')
        df = df.dropna(subset=['NormEnergy'])
        # Define the desired order
        halogen_order = ['Cl', 'Br', 'I']

        # Convert the 'Halogen' column to a categorical type with the specified order
        df['Halogen'] = pd.Categorical(df['Halogen'], categories=halogen_order, ordered=True)
        
        # --- Create Boxplots Energy ---
        # Note the use of LaTeX for axis labels (the 'r' prefix is important)
        y_axis_label = r'Normalized Energy ($E_{norm}$ / eV)'
        
        create_boxplot(df, 'Halogen', 'NormEnergy', 'Energy Distribution by Halogen', 
                       'norm_energy_by_halogen.png', y_label=y_axis_label, folder='Boxplot_graphs')

        create_boxplot(df.sort_values('N_Slab'), 'N_Slab', 'NormEnergy', 'Energy Distribution by Slab Thickness', 
                       'norm_energy_by_slab.png', y_label=y_axis_label, x_label='Number of Slabs')
        
        # Corrected filename for the family plot
        create_boxplot(df.sort_values('NCarbonsMol'), 'FamilyMol', 'NormEnergy', 'Energy Distribution by Molecular Family',
                       'norm_energy_by_family.png', y_label=y_axis_label, x_label='Molecular Family')
        
        create_boxplot(df, 'Molecule', 'NormEnergy', 'Energy Distribution by Molecule',
                       'norm_energy_by_molecule.png', y_label=y_axis_label)
        
        # --- Create Boxplots Geometry ---
        y_axis_label = r'Distance ($\AA$)'
        # Boxplot of the lattice constants A,B,C,Alpha,Beta,Gamma per Halogen and ordered x axis [Cl, Br, I]
        create_boxplot(df.sort_values('Halogen'), 'Halogen', 'A', 'Lattice A by Halogen',
                       'norm_energy_by_lattice_constant_a.png', y_label=y_axis_label)
        
        create_boxplot(df.sort_values('Halogen'), 'Halogen', 'B', 'Lattice B by Halogen', 
                       'norm_energy_by_lattice_constant_b.png', y_label=y_axis_label)
        
        create_boxplot(df.sort_values('Halogen'), 'Halogen', 'C', 'Lattice C by Halogen',
                       'norm_energy_by_lattice_constant_c.png', y_label=y_axis_label)
        
        y_axis_label = r'Angle ($\degree$)'
        create_boxplot(df.sort_values('Halogen'), 'Halogen', 'Alpha', 'Lattice Alpha by Halogen',
                       'norm_energy_by_lattice_constant_alpha.png', y_label=y_axis_label)
        
        create_boxplot(df.sort_values('Halogen'), 'Halogen', 'Beta', 'Lattice Beta by Halogen',
                       'norm_energy_by_lattice_constant_beta.png', y_label=y_axis_label)
        
        create_boxplot(df.sort_values('Halogen'), 'Halogen', 'Gamma', 'Lattice Gamma by Halogen', 
                       'norm_energy_by_lattice_constant_gamma.png', y_label=y_axis_label)

        # --- Create Scatter Plot Energy vs Slab, Family, Molecule ---
        create_scatter_with_regression(df, 'NormEmol', 'NormEnergy', 'N_Slab', 
                                     r'Formation Energy vs. Molecular Energy', 
                                     'NormEmol_vs_NormEnergy_by_Slab.png',
                                     x_label=r'Molecular Energy ($E_{mol}$ / eV)',
                                     y_label=r'Formation Energy ($E_{form}$ / eV)')

        # --- Create Scatter Plot ---
        create_scatter_with_regression(df, 'NormEmol', 'NormEnergy', 'FamilyMol', 
                                     r'Formation Energy vs. Molecular Energy', 
                                     'NormEmol_vs_NormEnergy_by_Family.png',
                                     x_label=r'Molecular Energy ($E_{mol}$ / eV)',
                                     y_label=r'Formation Energy ($E_{form}$ / eV)')

        # --- Create Scatter Plot ---
        create_scatter_with_regression(df, 'NormEmol', 'NormEnergy', 'Molecule', 
                                     r'Formation Energy vs. Molecular Energy', 
                                     'NormEmol_vs_NormEnergy_by_Molecule.png',
                                     x_label=r'Molecular Energy ($E_{mol}$ / eV)',
                                     y_label=r'Formation Energy ($E_{form}$ / eV)')
        
        # NormEnergy Vs NCarbonsMol grouper by FamilyMol
        create_scatter_with_regression(df, 'NCarbonsMol', 'NormEnergy', 'FamilyMol', 
                                     r'Formation Energy vs. Molecular Energy', 
                                     'NormEnergy_vs_NCarbonsMol_by_Family.png',
                                     x_label=r'Number of carbons',
                                     y_label=r'Formation Energy ($E_{form}$ / eV)')
                                     
        # NormEnergy Vs NCarbonsMol grouper by Molecule
        create_scatter_with_regression(df, 'NCarbonsMol', 'NormEnergy', 'Molecule', 
                                     r'Formation Energy vs. Molecular Energy', 
                                     'NormEnergy_vs_NCarbonsMol_by_Molecule.png',
                                     x_label=r'Number of carbons',
                                     y_label=r'Formation Energy ($E_{form}$ / eV)')

        # NormEnergy Vs NCarbonsMol grouper by Slab
        create_scatter_with_regression(df, 'NCarbonsMol', 'NormEnergy', 'N_Slab', 
                                     r'Formation Energy vs. Molecular Energy', 
                                     'NormEnergy_vs_NCarbonsMol_by_Slab.png',
                                     x_label=r'Number of carbons',
                                     y_label=r'Formation Energy ($E_{form}$ / eV)')

        # NormEmol Vs NCarbonsMol grouper by FamilyMol
        create_scatter_with_regression(df, 'NCarbonsMol', 'NormEmol', 'FamilyMol', 
                                     r'Formation Energy vs. Molecular Energy', 
                                     'NormEmol_vs_NCarbonsMol_by_Family.png',
                                     x_label=r'Number of carbons',
                                     y_label=r'Formation Energy ($E_{form}$ / eV)')

        # --- Create Scatter Plot Energy vs Lattice Constants ---
        create_scatter_with_regression(df, 'A', 'NormEnergy', 'Halogen', 
                                     r'Formation Energy vs. Lattice Constant A', 
                                     'NormEnergy_vs_Lattice_Constant_A_by_Halogen.png',
                                     x_label=r'Lattice Constant A ($\AA$)',
                                     y_label=r'Formation Energy ($E_{form}$ / eV)')
        create_scatter_with_regression(df, 'B', 'NormEnergy', 'Halogen', 
                                     r'Formation Energy vs. Lattice Constant B', 
                                     'NormEnergy_vs_Lattice_Constant_B_by_Halogen.png',
                                     x_label=r'Lattice Constant B ($\AA$)',
                                     y_label=r'Formation Energy ($E_{form}$ / eV)')
        create_scatter_with_regression(df, 'C', 'NormEnergy', 'Halogen', 
                                     r'Formation Energy vs. Lattice Constant C', 
                                     'NormEnergy_vs_Lattice_Constant_C_by_Halogen.png',
                                     x_label=r'Lattice Constant C ($\AA$)',
                                     y_label=r'Formation Energy ($E_{form}$ / eV)')
        create_scatter_with_regression(df, 'Alpha', 'NormEnergy', 'Halogen',
                                     r'Formation Energy vs. Lattice Constant Alpha', 
                                     'NormEnergy_vs_Lattice_Constant_Alpha_by_Halogen.png',
                                     x_label=r'Angle Alpha ($\degree$)',
                                     y_label=r'Formation Energy ($E_{form}$ / eV)')
        create_scatter_with_regression(df, 'Beta', 'NormEnergy', 'Halogen', 
                                     r'Formation Energy vs. Lattice Constant Beta', 
                                     'NormEnergy_vs_Lattice_Constant_Beta_by_Halogen.png', 
                                     x_label=r'Angle Beta ($\degree$)',
                                     y_label=r'Formation Energy ($E_{form}$ / eV)')
        create_scatter_with_regression(df, 'Gamma', 'NormEnergy', 'Halogen', 
                                     r'Formation Energy vs. Lattice Constant Gamma', 
                                     'NormEnergy_vs_Lattice_Constant_Gamma_by_Halogen.png',
                                     x_label=r'Angle Gamma ($\degree$)',
                                     y_label=r'Formation Energy ($E_{form}$ / eV)')



        print("All graphs have been generated successfully.")

    except FileNotFoundError:
        print("Error: 'perovskites.csv' not found. Please ensure the data file is in the correct directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
#!/usr/bin/env python3
"""
Real Batch Analysis for q2D Materials.

This script performs batch analysis on real MAPbBr3 structures with different
spacer molecules from the DOS_END directory.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q2D_Materials.core.batch_analysis import BatchAnalyzer, export_comprehensive_data_to_csv
from q2D_Materials.core.analyzer import q2D_analyzer

def read_band_edges_csv(csv_path):
    """Read band edges from CSV file and return structured data."""
    try:
        df = pd.read_csv(csv_path)
        data = {}
        for _, row in df.iterrows():
            component = row['Component']
            data[component] = {
                'VBM': row['VBM (eV)'],
                'CBM': row['CBM (eV)'],
                'Band_Gap': row['Band Gap (eV)'],
                'Has_DOS_at_Fermi': row['Has DOS at Fermi']
            }
        return data
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def load_band_edges_data(dos_end_dir, batch_analyzer):
    """Load band edges data from all CSV files in subdirectories."""
    print("\n" + "=" * 60)
    print("LOADING BAND EDGES DATA FROM CSV FILES")
    print("=" * 60)
    
    band_edges_data = {}
    loaded_count = 0
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(dos_end_dir) 
              if os.path.isdir(os.path.join(dos_end_dir, d))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(dos_end_dir, subdir)
        
        # Look for CSV files with band edges
        csv_files = list(Path(subdir_path).glob("*band_edges.csv"))
        
        if csv_files:
            csv_file = csv_files[0]  # Use the first CSV file found
            band_data = read_band_edges_csv(csv_file)
            
            if band_data:
                # Extract layer thickness and X-site from directory name
                layer_thickness = None
                x_site = None
                
                # Extract layer thickness (n) from directory name
                if '_n' in subdir:
                    try:
                        n_part = subdir.split('_n')[1].split('_')[0]
                        layer_thickness = int(n_part)
                    except (ValueError, IndexError):
                        pass
                
                # Extract X-site from directory name
                if 'MAPbBr3' in subdir:
                    x_site = 'Br'
                elif 'MAPbI3' in subdir:
                    x_site = 'I'
                elif 'MAPbCl3' in subdir:
                    x_site = 'Cl'
                
                if layer_thickness is not None and x_site is not None:
                    band_edges_data[subdir] = {
                        'data': band_data,
                        'layer_thickness': layer_thickness,
                        'x_site': x_site,
                        'directory': subdir
                    }
                    loaded_count += 1
                    print(f"  ✓ Loaded: {subdir} (X={x_site}, n={layer_thickness})")
                else:
                    print(f"  ⚠ Could not extract layer thickness or X-site from {subdir}")
            else:
                print(f"  ✗ Failed to read band edges from {subdir}")
        else:
            print(f"  ⚠ No band edges CSV found in {subdir}")
    
    print(f"\nSuccessfully loaded band edges data from {loaded_count} directories")
    return band_edges_data

def create_band_edges_plots(band_edges_data, save=True):
    """Create plots for CBM, VBM, HOMO, LUMO grouped by n and halogen."""
    print("\n" + "=" * 60)
    print("CREATING BAND EDGES PLOTS")
    print("=" * 60)
    
    if not band_edges_data:
        print("No band edges data available for plotting")
        return []
    
    # Prepare data for plotting
    plot_data = []
    for exp_name, exp_data in band_edges_data.items():
        layer_thickness = exp_data['layer_thickness']
        x_site = exp_data['x_site']
        band_data = exp_data['data']
        
        # Add spacer data
        if 'SPACER' in band_data:
            spacer_data = band_data['SPACER']
            plot_data.append({
                'Experiment': exp_name,
                'Layer_Thickness': layer_thickness,
                'X_Site': x_site,
                'Component': 'SPACER',
                'VBM': spacer_data['VBM'],
                'CBM': spacer_data['CBM'],
                'Band_Gap': spacer_data['Band_Gap'],
                'HOMO': spacer_data['VBM'],  # HOMO = VBM for organic molecules
                'LUMO': spacer_data['CBM']   # LUMO = CBM for organic molecules
            })
        
        # Add slab data
        if 'SLAB' in band_data:
            slab_data = band_data['SLAB']
            plot_data.append({
                'Experiment': exp_name,
                'Layer_Thickness': layer_thickness,
                'X_Site': x_site,
                'Component': 'SLAB',
                'VBM': slab_data['VBM'],
                'CBM': slab_data['CBM'],
                'Band_Gap': slab_data['Band_Gap'],
                'HOMO': slab_data['VBM'],  # HOMO = VBM for inorganic slab
                'LUMO': slab_data['CBM']   # LUMO = CBM for inorganic slab
            })
    
    if not plot_data:
        print("No valid band edges data for plotting")
        return []
    
    df = pd.DataFrame(plot_data)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    figures = []
    
    # Properties to plot
    properties = ['VBM', 'CBM', 'HOMO', 'LUMO', 'Band_Gap']
    property_labels = {
        'VBM': 'Valence Band Maximum (eV)',
        'CBM': 'Conduction Band Minimum (eV)', 
        'HOMO': 'Highest Occupied Molecular Orbital (eV)',
        'LUMO': 'Lowest Unoccupied Molecular Orbital (eV)',
        'Band_Gap': 'Band Gap (eV)'
    }
    
    # Create plots grouped by X-site
    print("Creating band edges plots grouped by X-site...")
    
    # Define consistent ordering
    halogen_order = ['Cl', 'Br', 'I']
    halogen_colors = {'Cl': '#419667', 'Br': '#FD7949', 'I': '#5A9AE4'}
    component_colors = {'SPACER': '#FF6B6B', 'SLAB': '#4ECDC4'}  # Colors for Component hue
    
    for prop in properties:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: By X-site (ordered)
        sns.boxplot(data=df, x='X_Site', y=prop, hue='Component', ax=axes[0], 
                    order=halogen_order, palette=component_colors)
        axes[0].set_title(f'{property_labels[prop]} by X-Site', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X-Site (Halogen)', fontsize=12)
        axes[0].set_ylabel(property_labels[prop], fontsize=12)
        axes[0].legend(title='Component', loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: By Layer Thickness
        sns.boxplot(data=df, x='Layer_Thickness', y=prop, hue='Component', ax=axes[1], 
                    palette=component_colors)
        axes[1].set_title(f'{property_labels[prop]} by Layer Thickness', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Layer Thickness (n)', fontsize=12)
        axes[1].set_ylabel(property_labels[prop], fontsize=12)
        axes[1].legend(title='Component', loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f'MAPbX3_{prop.lower()}_by_X_site_and_layer.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {filename}")
        
        figures.append(fig)
    
    # Create combined comparison plots
    print("Creating combined band edges comparison plots...")
    
    # Combined VBM/CBM plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # VBM by X-site (ordered)
    sns.boxplot(data=df, x='X_Site', y='VBM', hue='Component', ax=axes[0,0], 
                order=halogen_order, palette=component_colors)
    axes[0,0].set_title('VBM by X-Site', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('VBM (eV)', fontsize=12)
    axes[0,0].legend(title='Component')
    axes[0,0].grid(True, alpha=0.3)
    
    # CBM by X-site (ordered)
    sns.boxplot(data=df, x='X_Site', y='CBM', hue='Component', ax=axes[0,1], 
                order=halogen_order, palette=component_colors)
    axes[0,1].set_title('CBM by X-Site', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('CBM (eV)', fontsize=12)
    axes[0,1].legend(title='Component')
    axes[0,1].grid(True, alpha=0.3)
    
    # VBM by Layer Thickness
    sns.boxplot(data=df, x='Layer_Thickness', y='VBM', hue='Component', ax=axes[1,0], 
                palette=component_colors)
    axes[1,0].set_title('VBM by Layer Thickness', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Layer Thickness (n)', fontsize=12)
    axes[1,0].set_ylabel('VBM (eV)', fontsize=12)
    axes[1,0].legend(title='Component')
    axes[1,0].grid(True, alpha=0.3)
    
    # CBM by Layer Thickness
    sns.boxplot(data=df, x='Layer_Thickness', y='CBM', hue='Component', ax=axes[1,1], 
                palette=component_colors)
    axes[1,1].set_title('CBM by Layer Thickness', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Layer Thickness (n)', fontsize=12)
    axes[1,1].set_ylabel('CBM (eV)', fontsize=12)
    axes[1,1].legend(title='Component')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = 'MAPbX3_vbm_cbm_combined_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
    
    figures.append(fig)
    
    # Create HOMO/LUMO comparison for organic spacers
    spacer_df = df[df['Component'] == 'SPACER'].copy()
    if not spacer_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # HOMO by X-site (ordered)
        sns.boxplot(data=spacer_df, x='X_Site', y='HOMO', ax=axes[0], 
                    order=halogen_order, palette=halogen_colors)
        axes[0].set_title('HOMO (Spacer) by X-Site', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X-Site (Halogen)', fontsize=12)
        axes[0].set_ylabel('HOMO (eV)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # LUMO by X-site (ordered)
        sns.boxplot(data=spacer_df, x='X_Site', y='LUMO', ax=axes[1], 
                    order=halogen_order, palette=halogen_colors)
        axes[1].set_title('LUMO (Spacer) by X-Site', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X-Site (Halogen)', fontsize=12)
        axes[1].set_ylabel('LUMO (eV)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = 'MAPbX3_homo_lumo_spacer_analysis.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {filename}")
        
        figures.append(fig)
    
    print(f"✓ Created {len(figures)} band edges plots")
    return figures

def create_correlation_analysis_from_csv(csv_file="unified_octahedral_molecular_dataset.csv"):
    """
    Create correlation analysis from the comprehensive CSV file.
    
    Args:
        csv_file: Path to the comprehensive CSV file
    """
    print("\n" + "=" * 60)
    print("CREATING CORRELATION ANALYSIS FROM CSV")
    print("=" * 60)
    
    try:
        # Load data
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded data from {csv_file}")
        print(f"  Total experiments: {len(df)}")
        print(f"  Total variables: {len(df.columns)}")
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_data = df[numeric_cols].dropna()
        
        if len(correlation_data) < 2:
            print("Not enough numeric data for correlation analysis")
            return None
        
        print(f"  Numeric variables: {len(numeric_cols)}")
        print(f"  Valid data points: {len(correlation_data)}")
        
        # Create correlation matrices
        pearson_corr = correlation_data.corr(method='pearson')
        spearman_corr = correlation_data.corr(method='spearman')
        
        # Create heatmaps
        figures = []
        
        # Pearson correlation heatmap
        fig, ax = plt.subplots(figsize=(16, 14))
        mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
        sns.heatmap(pearson_corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.3f', cbar_kws={"shrink": .8}, ax=ax)
        ax.set_title('Pearson Correlation Matrix (Comprehensive Dataset)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = 'MAPbX3_correlation_pearson_heatmap.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Pearson correlation heatmap saved to: {filename}")
        figures.append(fig)
        
        # Spearman correlation heatmap  
        fig, ax = plt.subplots(figsize=(16, 14))
        mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
        sns.heatmap(spearman_corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.3f', cbar_kws={"shrink": .8}, ax=ax)
        ax.set_title('Spearman Correlation Matrix (Comprehensive Dataset)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = 'MAPbX3_correlation_spearman_heatmap.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Spearman correlation heatmap saved to: {filename}")
        figures.append(fig)
        
        # Comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # Pearson
        mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
        sns.heatmap(pearson_corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                    square=True, cbar_kws={"shrink": .8}, ax=ax1)
        ax1.set_title('Pearson (Linear)', fontsize=14, fontweight='bold')
        
        # Spearman
        mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
        sns.heatmap(spearman_corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                    square=True, cbar_kws={"shrink": .8}, ax=ax2)
        ax2.set_title('Spearman (Non-Linear)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        filename = 'MAPbX3_correlation_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Correlation comparison plot saved to: {filename}")
        figures.append(fig)
        
        # Save correlation data to text files
        # Pearson correlations
        pearson_filename = 'MAPbX3_correlation_pearson_correlations.txt'
        with open(pearson_filename, 'w') as f:
            f.write("PEARSON LINEAR CORRELATIONS\n")
            f.write("=" * 50 + "\n\n")
            f.write("Correlation Matrix:\n")
            f.write(pearson_corr.to_string())
            f.write("\n\n")
            
            # Find strongest correlations
            f.write("STRONGEST CORRELATIONS (|r| > 0.5):\n")
            f.write("-" * 40 + "\n")
            
            # Get upper triangle of correlation matrix
            upper_tri = pearson_corr.where(np.triu(np.ones(pearson_corr.shape), k=1).astype(bool))
            
            # Flatten and sort by absolute value
            corr_pairs = []
            for i in range(len(upper_tri.columns)):
                for j in range(i+1, len(upper_tri.columns)):
                    corr_val = upper_tri.iloc[i, j]
                    if not pd.isna(corr_val) and abs(corr_val) > 0.5:
                        corr_pairs.append((upper_tri.columns[i], upper_tri.columns[j], corr_val))
            
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            for var1, var2, corr_val in corr_pairs:
                f.write(f"{var1} vs {var2}: r = {corr_val:.4f}\n")
            
            f.write(f"\nTotal data points: {len(correlation_data)}\n")
            f.write(f"Variables analyzed: {len(numeric_cols)}\n")
        
        print(f"Pearson correlations saved to: {pearson_filename}")
        
        # Spearman correlations
        spearman_filename = 'MAPbX3_correlation_spearman_correlations.txt'
        with open(spearman_filename, 'w') as f:
            f.write("SPEARMAN NON-LINEAR CORRELATIONS\n")
            f.write("=" * 50 + "\n\n")
            f.write("Correlation Matrix:\n")
            f.write(spearman_corr.to_string())
            f.write("\n\n")
            
            # Find strongest correlations
            f.write("STRONGEST CORRELATIONS (|ρ| > 0.5):\n")
            f.write("-" * 40 + "\n")
            
            # Get upper triangle of correlation matrix
            upper_tri = spearman_corr.where(np.triu(np.ones(spearman_corr.shape), k=1).astype(bool))
            
            # Flatten and sort by absolute value
            corr_pairs = []
            for i in range(len(upper_tri.columns)):
                for j in range(i+1, len(upper_tri.columns)):
                    corr_val = upper_tri.iloc[i, j]
                    if not pd.isna(corr_val) and abs(corr_val) > 0.5:
                        corr_pairs.append((upper_tri.columns[i], upper_tri.columns[j], corr_val))
            
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            for var1, var2, corr_val in corr_pairs:
                f.write(f"{var1} vs {var2}: ρ = {corr_val:.4f}\n")
            
            f.write(f"\nTotal data points: {len(correlation_data)}\n")
            f.write(f"Variables analyzed: {len(numeric_cols)}\n")
        
        print(f"Spearman correlations saved to: {spearman_filename}")
        
        # Combined analysis file
        combined_filename = 'MAPbX3_correlation_combined_analysis.txt'
        with open(combined_filename, 'w') as f:
            f.write("COMPREHENSIVE CORRELATION ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("ANALYSIS SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total experiments: {len(df)}\n")
            f.write(f"Valid data points: {len(correlation_data)}\n")
            f.write(f"Variables analyzed: {len(numeric_cols)}\n")
            f.write(f"Missing data points: {len(df) - len(correlation_data)}\n\n")
            
            f.write("VARIABLES INCLUDED:\n")
            f.write("-" * 20 + "\n")
            for i, var in enumerate(numeric_cols, 1):
                f.write(f"{i:2d}. {var}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("PEARSON LINEAR CORRELATIONS\n")
            f.write("=" * 60 + "\n")
            f.write(pearson_corr.to_string())
            
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("SPEARMAN NON-LINEAR CORRELATIONS\n")
            f.write("=" * 60 + "\n")
            f.write(spearman_corr.to_string())
        
        print(f"Combined analysis saved to: {combined_filename}")
        
        return figures
        
    except Exception as e:
        print(f"✗ Error creating correlation analysis: {e}")
        return None

def classify_molecule_family(smiles):
    """
    Classify a molecule SMILES into a family based on its structure.
    
    Args:
        smiles: SMILES string of the molecule
        
    Returns:
        tuple: (family, carbon_count, display_name)
    """
    import re
    
    # Count carbon atoms (C not in brackets)
    carbon_count = len(re.findall(r'C(?![A-Z])', smiles))
    
    # Classify based on structure patterns
    if 'C1=CC=C' in smiles or 'C1=CC=CC=C1' in smiles:
        # Aromatic (benzene-like)
        family = 'Aromatic'
        display_name = f'Aromatic C{carbon_count}'
    elif 'C1CCC' in smiles and 'C1=CC' not in smiles:
        # Cyclic (hexane-like, no double bonds)
        family = 'Cyclic'
        display_name = f'Cyclic C{carbon_count}'
    elif 'C(C)' in smiles or 'CC(C)' in smiles or 'C(C)(C)' in smiles:
        # Branched (methylated)
        family = 'Branched'
        display_name = f'Branched C{carbon_count}'
    else:
        # Linear (straight chain)
        family = 'Linear'
        display_name = f'Linear C{carbon_count}'
    
    return family, carbon_count, display_name

def create_molecular_family_trend_plots(csv_file="unified_octahedral_molecular_dataset.csv"):
    """
    Create trend plots for molecular families with family + carbon count on x-axis.
    
    Args:
        csv_file: Path to the comprehensive CSV file
    """
    print("\n" + "=" * 60)
    print("CREATING MOLECULAR FAMILY TREND PLOTS")
    print("=" * 60)
    
    try:
        # Load data
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded data from {csv_file}")
        print(f"  Total experiments: {len(df)}")
        
        # Classify molecules into families
        if 'Molecule' in df.columns:
            print("Classifying molecules into families...")
            df['Family'] = df['Molecule'].apply(lambda x: classify_molecule_family(x)[0])
            df['Carbon_Count'] = df['Molecule'].apply(lambda x: classify_molecule_family(x)[1])
            df['Display_Name'] = df['Molecule'].apply(lambda x: classify_molecule_family(x)[2])
            
            # Create family + carbon count labels for x-axis
            df['Family_Carbon'] = df['Family'] + ' C' + df['Carbon_Count'].astype(str)
            
            # Define family colors
            family_colors = {
                'Linear': '#1976D2',      # Blue
                'Branched': '#F57C00',    # Orange
                'Cyclic': '#388E3C',      # Green
                'Aromatic': '#7B1FA2'     # Purple
            }
            
            family_bg_colors = {
                'Linear': '#E3F2FD',      # Light blue
                'Branched': '#FFF3E0',    # Light orange
                'Cyclic': '#E8F5E8',      # Light green
                'Aromatic': '#F3E5F5'     # Light purple
            }
            
            # Properties to plot
            properties = ['Bandgap', 'VBM', 'CBM', 'Spacer_VBM', 'Spacer_CBM', 'Spacer_Band_Gap']
            property_labels = {
                'Bandgap': 'Band Gap (eV)',
                'VBM': 'Valence Band Maximum (eV)',
                'CBM': 'Conduction Band Minimum (eV)',
                'Spacer_VBM': 'Spacer VBM (eV)',
                'Spacer_CBM': 'Spacer CBM (eV)',
                'Spacer_Band_Gap': 'Spacer Band Gap (eV)'
            }
            
            figures = []
            
            for prop in properties:
                if prop not in df.columns:
                    continue
                    
                # Filter data with valid values
                plot_data = df[df[prop].notna() & df['Family'].notna()].copy()
                
                if len(plot_data) < 5:
                    print(f"  ⚠ Not enough data for {prop}")
                    continue
                
                # Sort by family and carbon count
                family_order = ['Linear', 'Branched', 'Cyclic', 'Aromatic']
                plot_data = plot_data.sort_values(['Family', 'Carbon_Count'])
                
                # Create figure
                fig, ax = plt.subplots(figsize=(16, 8))
                
                # Get unique molecules first, sorted by family and carbon count
                unique_molecules = plot_data.drop_duplicates('Molecule').sort_values(['Family', 'Carbon_Count'])
                
                # Create numerical x-positions for unique molecules only
                x_labels = []
                family_positions = {}
                mol_to_x_pos = {}
                current_pos = 0
                
                # Define family abbreviations
                family_abbrev = {
                    'Linear': 'L',
                    'Branched': 'B', 
                    'Cyclic': 'C',
                    'Aromatic': 'A'
                }
                
                # First pass: determine positions and labels for unique molecules
                for family in family_order:
                    family_molecules = unique_molecules[unique_molecules['Family'] == family]
                    if len(family_molecules) > 0:
                        family_start = current_pos
                        # Sort by carbon count to get proper order
                        family_molecules_sorted = family_molecules.sort_values('Carbon_Count')
                        family_counter = 1
                        for _, row in family_molecules_sorted.iterrows():
                            mol_to_x_pos[row['Molecule']] = current_pos
                            x_labels.append(f"{family_abbrev[family]}{family_counter}")
                            family_counter += 1
                            current_pos += 1
                        family_positions[family] = (family_start, current_pos - 1)
                
                # Add x_position column to plot_data
                plot_data_with_positions = plot_data.copy()
                plot_data_with_positions['x_position'] = plot_data_with_positions['Molecule'].map(mol_to_x_pos)
                
                # Define marker shapes for layer thickness
                layer_markers = {1: 'o', 2: 's', 3: '^'}  # circles, squares, triangles
                layer_labels = {1: 'n=1', 2: 'n=2', 3: 'n=3'}
                
                # Create scatter plot using numerical positions with different markers for layer thickness
                legend_handles = []
                legend_labels = []
                
                for family in family_order:
                    family_data = plot_data_with_positions[plot_data_with_positions['Family'] == family]
                    if len(family_data) > 0:
                        # Plot each layer thickness with different markers
                        for layer_n in sorted(family_data['Layer_Thickness'].unique()):
                            layer_data = family_data[family_data['Layer_Thickness'] == layer_n]
                            if len(layer_data) > 0:
                                scatter = ax.scatter(layer_data['x_position'], layer_data[prop], 
                                                   c=family_colors[family], s=120, alpha=0.7, 
                                                   marker=layer_markers.get(layer_n, 'o'),
                                                   edgecolors='black', linewidth=1.0)
                        
                        # Add family to legend (we'll create a custom legend)
                        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                       markerfacecolor=family_colors[family], 
                                                       markersize=10, markeredgecolor='black',
                                                       label=family))
                
                # Add layer thickness markers to legend
                for layer_n, marker in layer_markers.items():
                    if layer_n in plot_data_with_positions['Layer_Thickness'].values:
                        legend_handles.append(plt.Line2D([0], [0], marker=marker, color='w', 
                                                       markerfacecolor='gray', markersize=10, 
                                                       markeredgecolor='black',
                                                       label=layer_labels[layer_n]))
                
                # Add background colors for families
                for family, (start, end) in family_positions.items():
                    if family in family_bg_colors:
                        ax.axvspan(start - 0.5, end + 0.5, alpha=0.3, 
                                 color=family_bg_colors[family], zorder=0)
                
                # Add vertical lines between families
                for i, family in enumerate(family_order[1:], 1):
                    if family in family_positions:
                        prev_family = family_order[i-1]
                        if prev_family in family_positions:
                            separator_x = family_positions[prev_family][1] + 0.5
                            ax.axvline(x=separator_x, color='black', linestyle='--', 
                                      alpha=0.7, linewidth=2, zorder=1)
                
                # Set x-axis labels
                ax.set_xticks(range(len(x_labels)))
                ax.set_xticklabels(x_labels, rotation=45, ha='right')
                
                # Customize plot
                ax.set_title(f'{property_labels[prop]} by Molecular Family & Carbon Count', 
                           fontsize=16, fontweight='bold')
                ax.set_xlabel('Molecular Family & Carbon Count', fontsize=14, fontweight='bold')
                ax.set_ylabel(property_labels[prop], fontsize=14, fontweight='bold')
                
                # Add custom legend with both families and layer thickness
                ax.legend(handles=legend_handles, fontsize=12, loc='best', 
                         title='Family & Layer Thickness', title_fontsize=12)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                filename = f'MAPbX3_{prop.lower()}_by_molecular_family_trend.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  ✓ Saved: {filename}")
                figures.append(fig)
            
            print(f"\n✓ Created {len(figures)} molecular family trend plots")
            return figures
            
        else:
            print("  ⚠ No Molecule column found, skipping family trend plots")
            return []
            
    except Exception as e:
        print(f"✗ Error creating molecular family trend plots: {e}")
        return []

def create_high_correlation_scatter_plots(csv_file="unified_octahedral_molecular_dataset.csv"):
    """
    Create scatter plots for the highest correlations found in the analysis.
    
    Args:
        csv_file: Path to the comprehensive CSV file
    """
    print("\n" + "=" * 60)
    print("CREATING HIGH CORRELATION SCATTER PLOTS")
    print("=" * 60)
    
    try:
        # Load data
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded data from {csv_file}")
        print(f"  Total experiments: {len(df)}")
        
        figures = []
        
        # Define interesting correlations to plot (excluding trivial ones like VBM=HOMO)
        correlations_to_plot = [
            # Strong positive correlations (scientifically interesting)
            ('Goldschmidt_Tolerance', 'Bandgap', 'Goldschmidt Tolerance vs Bandgap', 0.80),
            ('Delta', 'Goldschmidt_Tolerance', 'Delta (Δ) vs Goldschmidt Tolerance', 0.61),
            ('Layer_Thickness', 'Cell_Volume', 'Layer Thickness vs Cell Volume', 0.84),
            ('Spacer_CBM', 'Spacer_Band_Gap', 'Spacer CBM vs Spacer Band Gap', 0.84),
            ('Lambda_3', 'Lambda_2', 'λ₃ vs λ₂', 0.76),
            
            # Strong negative correlations (scientifically interesting)
            ('Cell_Volume', 'Bandgap', 'Cell Volume vs Bandgap', -0.70),
            ('Bandgap', 'VBM', 'Bandgap vs VBM', -0.64),
            ('Delta', 'Axial_Central_Axial_Mean', 'Delta (Δ) vs Axial-Central-Axial Angle', -0.61),
            ('Central_Axial_Central_Mean', 'Bandgap', 'B-X-B Angle vs Bandgap', 0.45),
            ('Goldschmidt_Tolerance', 'Spacer_VBM', 'Goldschmidt Tolerance vs Spacer VBM', -0.62),
            ('Spacer_VBM', 'Spacer_Band_Gap', 'Spacer VBM vs Spacer Band Gap', -0.72),
        ]
        
        # Classify molecules into families
        if 'Molecule' in df.columns:
            print("Classifying molecules into families...")
            df['Family'] = df['Molecule'].apply(lambda x: classify_molecule_family(x)[0])
            df['Carbon_Count'] = df['Molecule'].apply(lambda x: classify_molecule_family(x)[1])
            df['Display_Name'] = df['Molecule'].apply(lambda x: classify_molecule_family(x)[2])
            
            # Get unique families and create color mapping
            unique_families = df['Family'].unique()
            print(f"Found {len(unique_families)} molecular families: {unique_families}")
            
            # Define family colors (similar to the reference image)
            family_colors = {
                'Linear': '#E3F2FD',      # Light blue
                'Branched': '#FFF3E0',    # Light orange
                'Cyclic': '#E8F5E8',      # Light green
                'Aromatic': '#F3E5F5'     # Light purple
            }
            
            # Create molecule number mapping for clean legend
            sorted_molecules = sorted(df['Molecule'].unique())
            molecule_numbers = {mol: i+1 for i, mol in enumerate(sorted_molecules)}
        else:
            # Fallback to X-site grouping if Molecule not available
            unique_families = df['X_Site'].unique()
            family_colors = {'Br': '#FD7949', 'I': '#5A9AE4', 'Cl': '#419667'}
            molecule_numbers = {mol: i+1 for i, mol in enumerate(unique_families)}
        
        for var1, var2, title, expected_corr in correlations_to_plot:
            if var1 in df.columns and var2 in df.columns:
                # Filter data with both variables
                if 'Molecule' in df.columns:
                    plot_data = df[[var1, var2, 'Molecule', 'Family', 'Carbon_Count', 'Display_Name']].dropna()
                    group_col = 'Molecule'
                else:
                    plot_data = df[[var1, var2, 'X_Site']].dropna()
                    group_col = 'X_Site'
                
                if len(plot_data) > 5:  # Need enough points for meaningful plot
                    try:
                        fig, ax = plt.subplots(figsize=(14, 10))
                        
                        if 'Molecule' in df.columns:
                            # Create family-ordered plot with background colors
                            # Sort by family and carbon count
                            family_order = ['Linear', 'Branched', 'Cyclic', 'Aromatic']
                            plot_data_sorted = plot_data.sort_values(['Family', 'Carbon_Count'])
                            
                            # Create background regions for families
                            family_positions = {}
                            current_pos = 0
                            
                            for family in family_order:
                                family_data = plot_data_sorted[plot_data_sorted['Family'] == family]
                                if len(family_data) > 0:
                                    family_positions[family] = (current_pos, current_pos + len(family_data))
                                    current_pos += len(family_data)
                            
                            # Add background colors for families
                            for family, (start, end) in family_positions.items():
                                if family in family_colors:
                                    ax.axvspan(start - 0.5, end - 0.5, alpha=0.3, 
                                             color=family_colors[family], zorder=0)
                            
                            # Create scatter plot - use var1 for x-axis, var2 for y-axis
                            for i, (_, row) in enumerate(plot_data_sorted.iterrows()):
                                molecule = row['Molecule']
                                molecule_num = molecule_numbers.get(molecule, '?')
                                family = row['Family']
                                
                                # Use family-specific colors for points
                                point_colors = {
                                    'Linear': '#1976D2',      # Blue
                                    'Branched': '#F57C00',    # Orange
                                    'Cyclic': '#388E3C',      # Green
                                    'Aromatic': '#7B1FA2'     # Purple
                                }
                                
                                ax.scatter(row[var1], row[var2], 
                                          c=point_colors.get(family, '#888888'), 
                                          s=100, alpha=0.8, 
                                          edgecolors='black', linewidth=1.0,
                                          zorder=2)
                            
                            # Add family legend
                            from matplotlib.patches import Patch
                            legend_elements = []
                            for family in family_order:
                                if family in family_positions:
                                    legend_elements.append(Patch(facecolor=point_colors.get(family, '#888888'), 
                                                               label=family))
                            
                            ax.legend(handles=legend_elements, loc='best', fontsize=12)
                            
                            # Set axis labels
                            var1_label = var1.replace('_', ' ').replace('Goldschmidt Tolerance', 'Goldschmidt Tolerance Factor')
                            var2_label = var2.replace('_', ' ').replace('Goldschmidt Tolerance', 'Goldschmidt Tolerance Factor')
                            ax.set_xlabel(var1_label, fontsize=14, fontweight='bold')
                            ax.set_ylabel(var2_label, fontsize=14, fontweight='bold')
                            
                        else:
                            # Fallback to simple grouping
                            for molecule in plot_data[group_col].unique():
                                molecule_data = plot_data[plot_data[group_col] == molecule]
                                molecule_num = molecule_numbers.get(molecule, '?')
                                ax.scatter(molecule_data[var1], molecule_data[var2], 
                                          s=80, alpha=0.7, 
                                          label=f'{molecule_num} (n={len(molecule_data)})',
                                          edgecolors='black', linewidth=0.8)
                        
                        # Add trend line
                        x_vals = plot_data[var1].values
                        y_vals = plot_data[var2].values
                        
                        if len(x_vals) > 1:
                            # Calculate actual correlation
                            pearson_corr = np.corrcoef(x_vals, y_vals)[0, 1]
                            spearman_corr = pd.Series(x_vals).corr(pd.Series(y_vals), method='spearman')
                            
                            # Add trend line
                            z = np.polyfit(x_vals, y_vals, 1)
                            p = np.poly1d(z)
                            x_trend = np.linspace(min(x_vals), max(x_vals), 100)
                            ax.plot(x_trend, p(x_trend), "k--", alpha=0.8, linewidth=2.5, 
                                   label=f'Trend line')
                            
                            # Add correlation info to plot
                            textstr = f'Pearson r = {pearson_corr:.3f}\nSpearman ρ = {spearman_corr:.3f}\nData points: {len(plot_data)}'
                            props = dict(boxstyle='round', facecolor='white', alpha=0.8, 
                                       edgecolor='black', linewidth=1)
                            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                                   verticalalignment='top', bbox=props)
                        
                        # Set appropriate axis limits based on parameter types
                        x_vals = plot_data[var1].values
                        y_vals = plot_data[var2].values
                        
                        # Define expected ranges for specific parameters
                        param_ranges = {
                            'Goldschmidt_Tolerance': (0.8, 1.2),
                            'Octahedral_Tolerance': (0.3, 0.8),
                            'Delta': (0, 0.1),
                            'Sigma': (0, 100),
                            'Lambda_2': (1.0, 1.15),
                            'Lambda_3': (1.0, 1.15),
                            'Cis_Angle_Mean': (0, 180),
                            'Trans_Angle_Mean': (60, 220),
                            'Axial_Central_Axial_Mean': (140, 220),
                            'Central_Axial_Central_Mean': (140, 220),
                            'Bandgap': (0, 4),
                            'VBM': (-8, 0),
                            'CBM': (-2, 4),
                            'C1ll_Volume': (500, 5000),
                            'Layer_Thickness': (0.5, 3.5),
                            'Spacer_VBM': (-8, 0),
                            'Spacer_CBM': (-2, 4),
                            'Spacer_Band_Gap': (0, 8)
                        }
                        
                        # Set x-axis limits
                        if len(x_vals) > 0:
                            x_min, x_max = min(x_vals), max(x_vals)
                            if var1 in param_ranges:
                                expected_min, expected_max = param_ranges[var1]
                                ax.set_xlim(max(x_min * 0.95, expected_min), min(x_max * 1.05, expected_max))
                            else:
                                padding = (x_max - x_min) * 0.05 if x_max != x_min else 0.1
                                ax.set_xlim(x_min - padding, x_max + padding)
                        
                        # Set y-axis limits
                        if len(y_vals) > 0:
                            y_min, y_max = min(y_vals), max(y_vals)
                            if var2 in param_ranges:
                                expected_min, expected_max = param_ranges[var2]
                                ax.set_ylim(max(y_min * 0.95, expected_min), min(y_max * 1.05, expected_max))
                            else:
                                padding = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
                                ax.set_ylim(y_min - padding, y_max + padding)
                        
                        # Customize plot
                        var1_label = var1.replace('_', ' ').replace('Goldschmidt Tolerance', 'Goldschmidt Tolerance Factor')
                        var2_label = var2.replace('_', ' ').replace('Goldschmidt Tolerance', 'Goldschmidt Tolerance Factor')
                        
                        ax.set_xlabel(var1_label, fontsize=16, fontweight='bold', color='#333333')
                        ax.set_ylabel(var2_label, fontsize=16, fontweight='bold', color='#333333')
                        ax.set_title(title, fontsize=18, fontweight='bold', color='#222222', pad=20)
                        
                        # Clean axis styling
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_color('#666666')
                        ax.spines['bottom'].set_color('#666666')
                        ax.spines['left'].set_linewidth(2)
                        ax.spines['bottom'].set_linewidth(2)
                        
                        # Make tick labels bigger
                        ax.tick_params(axis='both', which='major', labelsize=14, colors='#333333', 
                                      width=2, length=6)
                        
                        # Add legend
                        ax.legend(fontsize=12, frameon=True, fancybox=False, shadow=False, 
                                 framealpha=0.9, facecolor='white', edgecolor='black',
                                 loc='best')
                        
                        # Add grid
                        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                        
                        plt.tight_layout()
                        
                        # Save plot
                        filename = f'MAPbX3_correlation_{var1.lower()}_vs_{var2.lower()}.png'
                        plt.savefig(filename, dpi=300, bbox_inches='tight')
                        print(f"  ✓ Saved: {filename}")
                        figures.append(fig)
                    
                    except Exception as e:
                        print(f"  ⚠ Error creating plot for {var1} vs {var2}: {e}")
                else:
                    print(f"  ⚠ Not enough data points for {var1} vs {var2} ({len(plot_data)} points)")
            else:
                missing_vars = []
                if var1 not in df.columns:
                    missing_vars.append(var1)
                if var2 not in df.columns:
                    missing_vars.append(var2)
                print(f"  ⚠ Missing variables: {', '.join(missing_vars)}")
        
        # Create molecular family legend image
        if 'Molecule' in df.columns and len(unique_molecules) > 1:
            create_molecular_family_legend_image(df)
        
        print(f"\n✓ Created {len(figures)} correlation scatter plots")
        return figures
        
    except Exception as e:
        print(f"✗ Error creating correlation scatter plots: {e}")
        return []

def create_molecular_family_legend_image(df, filename="molecular_families_legend.png"):
    """
    Create a legend image showing molecular families with their molecules.
    
    Args:
        df: DataFrame with molecular data including Family, Carbon_Count, Display_Name columns
        filename: Output filename for the legend image
    """
    print(f"\nCreating molecular family legend image: {filename}")
    
    try:
        # Get unique molecules sorted by family and carbon count
        family_order = ['Linear', 'Branched', 'Cyclic', 'Aromatic']
        df_sorted = df.sort_values(['Family', 'Carbon_Count']).drop_duplicates('Molecule')
        
        # Define colors
        family_colors = {
            'Linear': '#1976D2',      # Blue
            'Branched': '#F57C00',    # Orange
            'Cyclic': '#388E3C',      # Green
            'Aromatic': '#7B1FA2'     # Purple
        }
        
        family_bg_colors = {
            'Linear': '#E3F2FD',      # Light blue
            'Branched': '#FFF3E0',    # Light orange
            'Cyclic': '#E8F5E8',      # Light green
            'Aromatic': '#F3E5F5'     # Light purple
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, max(10, len(df_sorted) * 0.3)))
        
        # Group molecules by family
        family_groups = {}
        for family in family_order:
            family_data = df_sorted[df_sorted['Family'] == family]
            if len(family_data) > 0:
                family_groups[family] = family_data
        
        # Create legend entries
        y_start = 0.95
        y_step = 0.08
        current_y = y_start
        
        for family in family_order:
            if family in family_groups:
                family_data = family_groups[family]
                
                # Add family header
                ax.text(0.05, current_y, f"{family} Family", 
                       transform=ax.transAxes, fontsize=16, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', 
                               facecolor=family_bg_colors[family], 
                               alpha=0.8, edgecolor=family_colors[family], linewidth=2))
                current_y -= 0.03
                
                # Add molecules in this family
                for _, row in family_data.iterrows():
                    molecule = row['Molecule']
                    display_name = row['Display_Name']
                    carbon_count = row['Carbon_Count']
                    
                    # Add colored circle
                    ax.scatter(0.08, current_y, c=[family_colors[family]], s=150, 
                              edgecolors='black', linewidth=1.5, 
                              transform=ax.transAxes, zorder=3)
                    
                    # Add molecule info
                    ax.text(0.15, current_y, f"{display_name}", 
                           transform=ax.transAxes, fontsize=12, fontweight='bold', 
                           va='center', ha='left')
                    
                    # Add SMILES (truncated)
                    display_smiles = molecule
                    if len(display_smiles) > 50:
                        display_smiles = display_smiles[:47] + "..."
                    
                    ax.text(0.15, current_y - 0.015, display_smiles, 
                           transform=ax.transAxes, fontsize=10, 
                           va='center', ha='left', fontfamily='monospace',
                           style='italic', color='#666666')
                    
                    current_y -= y_step
                
                current_y -= 0.02  # Extra space between families
        
        # Customize plot
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Molecular Families Legend', fontsize=20, fontweight='bold', 
                    transform=ax.transAxes, pad=30)
        
        # Add summary
        total_molecules = len(df_sorted)
        family_counts = {family: len(family_groups.get(family, [])) for family in family_order}
        summary_text = f"Total molecules: {total_molecules} | " + " | ".join([f"{family}: {count}" for family, count in family_counts.items() if count > 0])
        
        ax.text(0.5, 0.02, summary_text, 
               transform=ax.transAxes, fontsize=12, ha='center', 
               style='italic', color='#666666')
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close()
        
    except Exception as e:
        print(f"  ⚠ Error creating molecular family legend: {e}")

def create_molecule_legend_image(unique_molecules, molecule_colors, filename="molecules_legend.png"):
    """
    Create a legend image showing molecule numbers and their SMILES.
    
    Args:
        unique_molecules: List of unique molecule SMILES
        molecule_colors: Dictionary mapping molecule SMILES to colors
        filename: Output filename for the legend image
    """
    print(f"\nCreating molecule legend image: {filename}")
    
    try:
        # Sort molecules for consistent numbering
        sorted_molecules = sorted(unique_molecules)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(8, len(sorted_molecules) * 0.4)))
        
        # Create legend entries
        y_positions = np.linspace(0.95, 0.05, len(sorted_molecules))
        
        for i, molecule in enumerate(sorted_molecules):
            y_pos = y_positions[i]
            color = molecule_colors.get(molecule, '#888888')
            
            # Add colored circle
            ax.scatter(0.05, y_pos, c=[color], s=200, edgecolors='black', linewidth=1.5, 
                      transform=ax.transAxes, zorder=3)
            
            # Add molecule number
            ax.text(0.12, y_pos, f"{i+1:2d}.", transform=ax.transAxes, 
                   fontsize=14, fontweight='bold', va='center', ha='left')
            
            # Add molecule SMILES (truncated if too long)
            display_smiles = molecule
            if len(display_smiles) > 60:
                display_smiles = display_smiles[:57] + "..."
            
            ax.text(0.18, y_pos, display_smiles, transform=ax.transAxes, 
                   fontsize=12, va='center', ha='left', fontfamily='monospace')
        
        # Customize plot
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Molecules Legend', fontsize=18, fontweight='bold', 
                    transform=ax.transAxes, pad=20)
        ax.text(0.5, 0.02, f'Total molecules: {len(sorted_molecules)}', 
               transform=ax.transAxes, fontsize=12, ha='center', 
               style='italic', color='#666666')
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close()
        
    except Exception as e:
        print(f"  ⚠ Error creating molecule legend: {e}")

def load_formation_energy_data(csv_file="batch_analysis_plots/formation_energy_structural_data.csv"):
    """
    Load formation energy data from CSV file and return structured data.
    
    Args:
        csv_file: Path to the formation energy CSV file
        
    Returns:
        dict: Dictionary mapping experiment names to formation energy data
    """
    print("\n" + "=" * 60)
    print("LOADING FORMATION ENERGY DATA FROM CSV")
    print("=" * 60)
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded formation energy data from {csv_file}")
        print(f"  Total entries: {len(df)}")
        
        formation_energy_data = {}
        
        for _, row in df.iterrows():
            # Construct experiment name: perovskite_name_n{number}_molecule_name
            exp_name = f"{row['perovskite_name']}_n{row['n_slab']}_{row['molecule_name']}"
            
            formation_energy_data[exp_name] = {
                'formation_energy': row['formation_energy_eV'],
                'perovskite_name': row['perovskite_name'],
                'halogen': row['halogen'],
                'n_slab': row['n_slab'],
                'molecule_name': row['molecule_name']
            }
        
        print(f"✓ Successfully loaded formation energy data for {len(formation_energy_data)} experiments")
        return formation_energy_data
        
    except Exception as e:
        print(f"✗ Error loading formation energy data: {e}")
        return {}

def create_formation_energy_scatter_plots(batch_analyzer=None, formation_energy_data=None, 
                                        csv_file=None, save=True, filename_prefix='MAPbX3_formation_energy_vs'):
    """
    Create formation energy scatter plots similar to bandgap plots.
    
    Args:
        batch_analyzer: BatchAnalyzer instance with experiment data (optional)
        formation_energy_data: Dictionary with formation energy data (optional)
        csv_file: Path to CSV file with structural data (optional)
        save: Whether to save plots
        filename_prefix: Prefix for saved filenames
        
    Returns:
        list: List of matplotlib figures
    """
    print("\n" + "=" * 60)
    print("CREATING FORMATION ENERGY SCATTER PLOTS")
    print("=" * 60)
    
    # Load formation energy data if not provided
    if formation_energy_data is None:
        formation_energy_data = load_formation_energy_data()
    
    if not formation_energy_data:
        print("No formation energy data available for plotting")
        return []
    
    # Load structural data from CSV if batch_analyzer not available
    if batch_analyzer is None and csv_file is not None:
        print(f"Loading structural data from CSV: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            print(f"✓ Loaded {len(df)} structural data entries")
            
            # Convert CSV data to comparison data format
            comparison_data = {}
            for _, row in df.iterrows():
                exp_name = row['Experiment']
                comparison_data[exp_name] = {
                    'X_site': row.get('X_Site', 'Unknown'),
                    'layer_thickness': row.get('Layer_Thickness', 1),
                    'delta': row.get('Delta'),
                    'sigma': row.get('Sigma'),
                    'lambda_3': row.get('Lambda_3'),
                    'lambda_2': row.get('Lambda_2'),
                    'cis_angle_mean': row.get('Cis_Angle_Mean'),
                    'trans_angle_mean': row.get('Trans_Angle_Mean'),
                    'axial_central_axial_mean': row.get('Axial_Central_Axial_Mean'),
                    'central_axial_central_mean': row.get('Central_Axial_Central_Mean')
                }
        except Exception as e:
            print(f"✗ Error loading CSV data: {e}")
            return []
    elif batch_analyzer is not None:
        # Extract comparison data from batch analyzer
        comparison_data = batch_analyzer.extract_comparison_data()
    else:
        print("✗ No structural data source provided")
        return []
    
    # Prepare scatter data with formation energy
    scatter_data = []
    for exp_name, exp_data in comparison_data.items():
        if exp_name in formation_energy_data:
            data_point = exp_data.copy()
            data_point['formation_energy'] = formation_energy_data[exp_name]['formation_energy']
            data_point['halogen'] = formation_energy_data[exp_name]['halogen']
            data_point['n_slab'] = formation_energy_data[exp_name]['n_slab']
            scatter_data.append(data_point)
    
    if not scatter_data:
        print("No matching experiments found between structural and formation energy data")
        print(f"Structural experiments: {len(comparison_data)}")
        print(f"Formation energy experiments: {len(formation_energy_data)}")
        
        # Show some examples for debugging
        print("\nFirst 3 structural experiments:")
        for i, exp_name in enumerate(list(comparison_data.keys())[:3]):
            print(f"  {i+1}. {exp_name}")
        
        print("\nFirst 3 formation energy experiments:")
        for i, exp_name in enumerate(list(formation_energy_data.keys())[:3]):
            print(f"  {i+1}. {exp_name}")
        return []
    
    print(f"Found {len(scatter_data)} experiments with both structural and formation energy data")
    
    # Parameters to plot against formation energy
    params = [
        ('delta', 'Δ (Distortion Parameter)'),
        ('sigma', 'σ (Distortion Parameter)'),
        ('lambda_3', 'λ₃ (Distortion Parameter)'),
        ('lambda_2', 'λ₂ (Distortion Parameter)'),
        ('cis_angle_mean', 'Cis Angle Mean'),
        ('trans_angle_mean', 'Trans Angle Mean'),
        ('axial_central_axial_mean', 'Axial-Central-Axial Mean'),
        ('central_axial_central_mean', 'B-X-B Mean')
    ]
    
    figures = []
    
    for param, param_label in params:
        # Filter out None values
        valid_data = [d for d in scatter_data if d.get(param) is not None]
        
        if not valid_data:
            print(f"No valid data for {param}")
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Separate data by X-site for coloring
        x_sites = list(set(d['X_site'] for d in valid_data))
        colors = {'Br': '#FD7949', 'I': '#5A9AE4', 'Cl': '#419667'}
        
        for x_site in x_sites:
            site_data = [d for d in valid_data if d['X_site'] == x_site]
            
            if not site_data:
                continue
            
            x_values = [d[param] for d in site_data]
            y_values = [d['formation_energy'] for d in site_data]
            
            # Create scatter plot
            ax.scatter(x_values, y_values, 
                      c=colors.get(x_site, '#888888'), 
                      s=100, alpha=0.7, 
                      label=f'{x_site} (n={len(site_data)})',
                      edgecolors='black', linewidth=0.5)
        
        # Add trend line
        if len(valid_data) > 1:
            x_vals = [d[param] for d in valid_data]
            y_vals = [d['formation_energy'] for d in valid_data]
            
            # Calculate correlation
            try:
                correlation = np.corrcoef(x_vals, y_vals)[0, 1]
            except:
                correlation = 0.0
            
            # Add trend line (with error handling for numerical issues)
            try:
                # Check if x_vals have sufficient variation
                x_range = max(x_vals) - min(x_vals)
                if x_range > 1e-10:  # Avoid fitting when all x values are essentially the same
                    z = np.polyfit(x_vals, y_vals, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(x_vals), max(x_vals), 100)
                    ax.plot(x_trend, p(x_trend), "k--", alpha=0.8, linewidth=2)
                else:
                    # If no variation in x, just show the mean y value
                    mean_y = np.mean(y_vals)
                    ax.axhline(y=mean_y, color='k', linestyle='--', alpha=0.8, linewidth=2)
            except (np.linalg.LinAlgError, ValueError, RuntimeWarning):
                # If polyfit fails, skip the trend line
                pass
            
            # Add correlation info
            ax.text(0.05, 0.95, f'r = {correlation:.3f}', 
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Customize plot
        ax.set_xlabel(param_label, fontsize=14, fontweight='bold')
        ax.set_ylabel('Formation Energy (eV)', fontsize=14, fontweight='bold')
        ax.set_title(f'Formation Energy vs {param_label}', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f'{filename_prefix}_{param}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {filename}")
        
        figures.append(fig)
    
    print(f"✓ Created {len(figures)} formation energy scatter plots")
    return figures

def create_formation_energy_by_layer_thickness_plots(batch_analyzer=None, formation_energy_data=None,
                                                   csv_file=None, save=True, filename_prefix='MAPbX3_formation_energy_by_layer_thickness'):
    """
    Create formation energy plots grouped by layer thickness.
    
    Args:
        batch_analyzer: BatchAnalyzer instance with experiment data (optional)
        formation_energy_data: Dictionary with formation energy data (optional)
        csv_file: Path to CSV file with structural data (optional)
        save: Whether to save plots
        filename_prefix: Prefix for saved filenames
        
    Returns:
        list: List of matplotlib figures
    """
    print("\n" + "=" * 60)
    print("CREATING FORMATION ENERGY BY LAYER THICKNESS PLOTS")
    print("=" * 60)
    
    # Load formation energy data if not provided
    if formation_energy_data is None:
        formation_energy_data = load_formation_energy_data()
    
    if not formation_energy_data:
        print("No formation energy data available for plotting")
        return []
    
    # Load structural data from CSV if batch_analyzer not available
    if batch_analyzer is None and csv_file is not None:
        print(f"Loading structural data from CSV: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            print(f"✓ Loaded {len(df)} structural data entries")
            
            # Convert CSV data to comparison data format
            comparison_data = {}
            for _, row in df.iterrows():
                exp_name = row['Experiment']
                comparison_data[exp_name] = {
                    'X_site': row.get('X_Site', 'Unknown'),
                    'layer_thickness': row.get('Layer_Thickness', 1),
                    'delta': row.get('Delta'),
                    'sigma': row.get('Sigma'),
                    'lambda_3': row.get('Lambda_3'),
                    'lambda_2': row.get('Lambda_2'),
                    'cis_angle_mean': row.get('Cis_Angle_Mean'),
                    'trans_angle_mean': row.get('Trans_Angle_Mean'),
                    'axial_central_axial_mean': row.get('Axial_Central_Axial_Mean'),
                    'central_axial_central_mean': row.get('Central_Axial_Central_Mean')
                }
        except Exception as e:
            print(f"✗ Error loading CSV data: {e}")
            return []
    elif batch_analyzer is not None:
        # Extract comparison data from batch analyzer
        comparison_data = batch_analyzer.extract_comparison_data()
    else:
        print("✗ No structural data source provided")
        return []
    
    # Prepare data with formation energy
    plot_data = []
    for exp_name, exp_data in comparison_data.items():
        if exp_name in formation_energy_data:
            data_point = exp_data.copy()
            data_point['formation_energy'] = formation_energy_data[exp_name]['formation_energy']
            data_point['halogen'] = formation_energy_data[exp_name]['halogen']
            data_point['n_slab'] = formation_energy_data[exp_name]['n_slab']
            plot_data.append(data_point)
    
    if not plot_data:
        print("No matching experiments found between structural and formation energy data")
        return []
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame(plot_data)
    
    figures = []
    
    # Box plot by layer thickness
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors for halogens
    colors = {'Br': '#FD7949', 'I': '#5A9AE4', 'Cl': '#419667'}
    
    # Create box plot grouped by layer thickness and colored by halogen
    layer_thicknesses = sorted(df['layer_thickness'].unique())
    
    for i, layer_thickness in enumerate(layer_thicknesses):
        layer_data = df[df['layer_thickness'] == layer_thickness]
        
        # Separate by halogen for coloring
        for halogen in ['Cl', 'Br', 'I']:
            halogen_data = layer_data[layer_data['halogen'] == halogen]
            if len(halogen_data) > 0:
                x_pos = i + (['Cl', 'Br', 'I'].index(halogen) - 1) * 0.25
                ax.scatter([x_pos] * len(halogen_data), halogen_data['formation_energy'],
                          c=colors[halogen], s=80, alpha=0.7, 
                          label=f'{halogen} (n={len(halogen_data)})' if i == 0 else "",
                          edgecolors='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Layer Thickness (n)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Formation Energy (eV)', fontsize=14, fontweight='bold')
    ax.set_title('Formation Energy by Layer Thickness', fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(layer_thicknesses)))
    ax.set_xticklabels([f'n={n}' for n in layer_thicknesses])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = f'{filename_prefix}_scatter.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
    
    figures.append(fig)
    
    # Box plot version
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create box plot
    sns.boxplot(data=df, x='layer_thickness', y='formation_energy', hue='halogen', 
                palette=colors, ax=ax)
    
    ax.set_xlabel('Layer Thickness (n)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Formation Energy (eV)', fontsize=14, fontweight='bold')
    ax.set_title('Formation Energy by Layer Thickness (Box Plot)', fontsize=16, fontweight='bold')
    ax.legend(title='Halogen', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = f'{filename_prefix}_boxplot.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
    
    figures.append(fig)
    
    print(f"✓ Created {len(figures)} formation energy by layer thickness plots")
    return figures

def main():
    """Main function to analyze real MAPbBr3 structures."""
    
    print("=" * 80)
    print("q2D Materials - Real Batch Analysis")
    print("Analyzing MAPbBr3 structures with different spacers")
    print("=" * 80)
    
    # Check if comprehensive CSV already exists
    csv_file = "unified_octahedral_molecular_dataset.csv"
    
    if os.path.exists(csv_file):
        print(f"\n✓ Found existing comprehensive dataset: {csv_file}")
        print("Choose option:")
        print("  1. Use existing CSV for analysis")
        print("  2. Regenerate CSV from scratch")
        print("  3. Exit")
        
        while True:
            try:
                choice = input("\nEnter choice (1/2/3): ").strip()
                if choice in ['1', '2', '3']:
                    break
                print("Please enter 1, 2, or 3")
            except KeyboardInterrupt:
                print("\nExiting...")
                return
        
        if choice == '3':
            print("Exiting...")
            return
        elif choice == '1':
            print(f"\n✓ Using existing CSV: {csv_file}")
            
            # Load band edges data for additional plots
            dos_end_dir = os.path.expanduser("~/Documents/DOS_END")
            if os.path.exists(dos_end_dir):
                print(f"\n✓ Loading band edges data from: {dos_end_dir}")
                batch_analyzer = BatchAnalyzer()  # Create minimal instance for band edges loading
                band_edges_data = load_band_edges_data(dos_end_dir, batch_analyzer)
                
                # Create band edges plots
                if band_edges_data:
                    print("\n" + "=" * 60)
                    print("CREATING BAND EDGES PLOTS")
                    print("=" * 60)
                    band_edges_figures = create_band_edges_plots(band_edges_data, save=True)
                    if band_edges_figures:
                        print(f"✓ Created {len(band_edges_figures)} band edges plots")
            
            # Create correlation analysis from CSV
            correlation_figures = create_correlation_analysis_from_csv(csv_file)
            if correlation_figures:
                print(f"✓ Created {len(correlation_figures)} correlation plots")
            
            # Create high correlation scatter plots
            scatter_figures = create_high_correlation_scatter_plots(csv_file)
            if scatter_figures:
                print(f"✓ Created {len(scatter_figures)} high correlation scatter plots")
            
            # Create molecular family trend plots
            family_trend_figures = create_molecular_family_trend_plots(csv_file)
            if family_trend_figures:
                print(f"✓ Created {len(family_trend_figures)} molecular family trend plots")
            
            # Load formation energy data and create formation energy plots
            print("\n" + "=" * 60)
            print("CREATING FORMATION ENERGY PLOTS")
            print("=" * 60)
            
            formation_energy_data = load_formation_energy_data()
            
            if formation_energy_data:
                # Create formation energy scatter plots
                formation_energy_figures = create_formation_energy_scatter_plots(
                    batch_analyzer=None, formation_energy_data=formation_energy_data,
                    csv_file=csv_file, save=True,
                    filename_prefix='MAPbX3_formation_energy_vs'
                )
                
                if formation_energy_figures:
                    print(f"✓ Created {len(formation_energy_figures)} formation energy scatter plots")
                else:
                    print("✗ No formation energy scatter plots created")
                
                # Create formation energy plots grouped by layer thickness
                formation_energy_by_n_figures = create_formation_energy_by_layer_thickness_plots(
                    batch_analyzer=None, formation_energy_data=formation_energy_data,
                    csv_file=csv_file, save=True,
                    filename_prefix='MAPbX3_formation_energy_by_layer_thickness'
                )
                
                if formation_energy_by_n_figures:
                    print(f"✓ Created {len(formation_energy_by_n_figures)} formation energy plots by layer thickness")
                else:
                    print("✗ No formation energy plots by layer thickness created")
            else:
                print("✗ No formation energy data available")
            
            print("\n" + "=" * 80)
            print("ANALYSIS FROM CSV COMPLETED")
            print("=" * 80)
            return
    
    # If we reach here, we need to generate the CSV
    print(f"\n✓ Generating comprehensive dataset: {csv_file}")
    
    # Initialize batch analyzer
    batch_analyzer = BatchAnalyzer()
    
    # Load experiments from DOS_END directory
    dos_end_dir = os.path.expanduser("~/Documents/DOS_END")
    print(f"Loading experiments from: {dos_end_dir}")
    
    if not os.path.exists(dos_end_dir):
        print(f"Error: Directory {dos_end_dir} does not exist!")
        return
    
    # Load all VASP files from subdirectories
    print("Scanning for VASP files in subdirectories...")
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(dos_end_dir) 
              if os.path.isdir(os.path.join(dos_end_dir, d))]
    
    print(f"Found {len(subdirs)} subdirectories:")
    for subdir in subdirs:
        print(f"  - {subdir}")
    
    # Load VASP files from each subdirectory
    loaded_count = 0
    for subdir in subdirs:
        subdir_path = os.path.join(dos_end_dir, subdir)
        
        # Look for POSCAR files (VASP format)
        vasp_files = []
        for filename in os.listdir(subdir_path):
            if filename.upper() in ['POSCAR', 'CONTCAR']:
                vasp_files.append(os.path.join(subdir_path, filename))
        
        if vasp_files:
            # Use the first VASP file found in each directory
            vasp_file = vasp_files[0]
            try:
                # Extract experiment name from directory name
                exp_name = subdir
                
                # Extract layer thickness (n) from directory name
                # Expected format: MAPbX3_n{number}_spacer
                layer_thickness = None
                if '_n' in subdir:
                    try:
                        n_part = subdir.split('_n')[1].split('_')[0]
                        layer_thickness = int(n_part)
                    except (ValueError, IndexError):
                        print(f"  ⚠ Could not extract layer thickness from {subdir}")
                
                # Extract spacer SMILES from directory name
                # Expected format: MAPbX3_n{number}_{spacer_smiles}
                spacer_smiles = None
                parts = subdir.split('_')
                if len(parts) >= 3:
                    spacer_smiles = '_'.join(parts[2:])  # Everything after MAPbX3_n{number}_
                else:
                    spacer_smiles = 'Unknown'
                
                # Auto-detect X-site from structure
                from ase.io import read
                atoms = read(vasp_file)
                symbols = atoms.get_chemical_symbols()
                
                # Look for halogen atoms (Br, I, Cl)
                x_site = None
                for symbol in symbols:
                    if symbol in ['Br', 'I', 'Cl']:
                        x_site = symbol
                        break
                
                if x_site is None:
                    print(f"Warning: No halogen found in {vasp_file}, using default 'Cl'")
                    x_site = 'Cl'
                
                # Initialize analyzer with auto-detected X-site
                analyzer = q2D_analyzer(
                    file_path=vasp_file,
                    b='Pb',  # B-site is Pb
                    x=x_site,  # Auto-detected from structure
                    cutoff_ref_ligand=4.0
                )
                
                # Store layer thickness in analyzer for later use
                analyzer.layer_thickness = layer_thickness
                
                # Read bandgap from the same directory
                bandgap = batch_analyzer.read_bandgap_from_directory(subdir_path, verbose=True)
                if bandgap is not None:
                    batch_analyzer.bandgap_data[exp_name] = bandgap
                    print(f"  ✓ Bandgap: {bandgap:.3f} eV")
                else:
                    print(f"  ⚠ No bandgap data found")
                
                # Read VBM/CBM from the same directory
                vbm_cbm_data = batch_analyzer.read_vbm_cbm_from_directory(subdir_path, verbose=True)
                if vbm_cbm_data is not None:
                    # Store VBM/CBM data in a new dictionary
                    if not hasattr(batch_analyzer, 'vbm_cbm_data'):
                        batch_analyzer.vbm_cbm_data = {}
                    batch_analyzer.vbm_cbm_data[exp_name] = vbm_cbm_data
                    print(f"  ✓ VBM: {vbm_cbm_data['vbm']:.3f} eV, CBM: {vbm_cbm_data['cbm']:.3f} eV")
                else:
                    print(f"  ⚠ No VBM/CBM data found")
                
                batch_analyzer.add_experiment(exp_name, analyzer)
                loaded_count += 1
                print(f"  ✓ Loaded: {exp_name} (X={x_site}, n={layer_thickness})")
                
            except Exception as e:
                print(f"  ✗ Failed to load {subdir}: {str(e)}")
        else:
            print(f"  ⚠ No VASP files found in {subdir}")
    
    if not batch_analyzer.experiments:
        print("No experiments loaded. Please check the DOS_END directory structure.")
        return
    
    print(f"\nSuccessfully loaded {loaded_count} experiments")
    
    # Extract comparison data
    print("\nExtracting comparison data...")
    comparison_data = batch_analyzer.extract_comparison_data()
    
    # Print batch summary
    batch_analyzer.print_batch_summary()
    
    # Print bandgap summary
    batch_analyzer.print_bandgap_summary()
    
    # Create bond angle distribution comparisons
    print("\n" + "=" * 60)
    print("CREATING BOND ANGLE DISTRIBUTION COMPARISONS")
    print("=" * 60)
    
    # Group by X-site (Br, I, Cl) - All angle types
    print("Creating cis angles distribution plot grouped by X-site...")
    fig1 = batch_analyzer.create_bond_angle_distribution_comparison(
        group_by='X_site',
        angle_type='cis_angles',
        show_mean_std=True,
        save=True,
        filename='MAPbX3_cis_angles_by_X_site.png'
    )
    
    if fig1:
        print("✓ Cis angles by X-site plot created successfully")
    else:
        print("✗ Failed to create cis angles by X-site plot")
    
    print("Creating trans angles distribution plot grouped by X-site...")
    fig2 = batch_analyzer.create_bond_angle_distribution_comparison(
        group_by='X_site',
        angle_type='trans_angles',
        show_mean_std=True,
        save=True,
        filename='MAPbX3_trans_angles_by_X_site.png'
    )
    
    if fig2:
        print("✓ Trans angles by X-site plot created successfully")
    else:
        print("✗ Failed to create trans angles by X-site plot")
    
    print("Creating axial-central-axial angles distribution plot grouped by X-site...")
    fig3 = batch_analyzer.create_bond_angle_distribution_comparison(
        group_by='X_site',
        angle_type='axial_central_axial',
        show_mean_std=True,
        save=True,
        filename='MAPbX3_axial_central_axial_by_X_site.png'
    )
    
    if fig3:
        print("✓ Axial-central-axial angles by X-site plot created successfully")
    else:
        print("✗ Failed to create axial-central-axial angles by X-site plot")
    
    print("Creating B-X-B (central-axial-central) angles distribution plot grouped by X-site...")
    fig4 = batch_analyzer.create_bond_angle_distribution_comparison(
        group_by='X_site',
        angle_type='central_axial_central',
        show_mean_std=True,
        save=True,
        filename='MAPbX3_b_x_b_angles_by_X_site.png'
    )
    
    if fig4:
        print("✓ B-X-B angles by X-site plot created successfully")
    else:
        print("✗ Failed to create B-X-B angles by X-site plot")
    
    # Group by layer thickness (n=1, n=2, n=3) - All angle types
    print("Creating cis angles distribution plot grouped by layer thickness...")
    fig5 = batch_analyzer.create_bond_angle_distribution_comparison(
        group_by='layer_thickness',
        angle_type='cis_angles',
        show_mean_std=True,
        save=True,
        filename='MAPbX3_cis_angles_by_layer_thickness.png'
    )
    
    if fig5:
        print("✓ Cis angles by layer thickness plot created successfully")
    else:
        print("✗ Failed to create cis angles by layer thickness plot")
    
    print("Creating trans angles distribution plot grouped by layer thickness...")
    fig6 = batch_analyzer.create_bond_angle_distribution_comparison(
        group_by='layer_thickness',
        angle_type='trans_angles',
        show_mean_std=True,
        save=True,
        filename='MAPbX3_trans_angles_by_layer_thickness.png'
    )
    
    if fig6:
        print("✓ Trans angles by layer thickness plot created successfully")
    else:
        print("✗ Failed to create trans angles by layer thickness plot")
    
    print("Creating axial-central-axial angles distribution plot grouped by layer thickness...")
    fig7 = batch_analyzer.create_bond_angle_distribution_comparison(
        group_by='layer_thickness',
        angle_type='axial_central_axial',
        show_mean_std=True,
        save=True,
        filename='MAPbX3_axial_central_axial_by_layer_thickness.png'
    )
    
    if fig7:
        print("✓ Axial-central-axial angles by layer thickness plot created successfully")
    else:
        print("✗ Failed to create axial-central-axial angles by layer thickness plot")
    
    print("Creating B-X-B (central-axial-central) angles distribution plot grouped by layer thickness...")
    fig8 = batch_analyzer.create_bond_angle_distribution_comparison(
        group_by='layer_thickness',
        angle_type='central_axial_central',
        show_mean_std=True,
        save=True,
        filename='MAPbX3_b_x_b_angles_by_layer_thickness.png'
    )
    
    if fig8:
        print("✓ B-X-B angles by layer thickness plot created successfully")
    else:
        print("✗ Failed to create B-X-B angles by layer thickness plot")
    
    # Create distortion parameter comparisons
    print("\n" + "=" * 60)
    print("CREATING DISTORTION PARAMETER COMPARISONS")
    print("=" * 60)
    
    # Distortion parameters by X-site
    distortion_params = ['delta', 'sigma', 'lambda_3', 'lambda_2']
    for param in distortion_params:
        print(f"Creating {param} comparison plot grouped by X-site...")
        fig = batch_analyzer.create_distortion_comparison_plot(
            group_by='X_site',
            distortion_param=param,
            show_mean_std=True,
            save=True,
            filename=f'MAPbX3_{param}_by_X_site.png'
        )
        if fig:
            print(f"✓ {param} by X-site plot created successfully")
        else:
            print(f"✗ Failed to create {param} by X-site plot")
    
    # Distortion parameters by layer thickness
    for param in distortion_params:
        print(f"Creating {param} comparison plot grouped by layer thickness...")
        fig = batch_analyzer.create_distortion_comparison_plot(
            group_by='layer_thickness',
            distortion_param=param,
            show_mean_std=True,
            save=True,
            filename=f'MAPbX3_{param}_by_layer_thickness.png'
        )
        if fig:
            print(f"✓ {param} by layer thickness plot created successfully")
        else:
            print(f"✗ Failed to create {param} by layer thickness plot")
    
    # Create bandgap scatter plots
    print("\n" + "=" * 60)
    print("CREATING BANDGAP SCATTER PLOTS")
    print("=" * 60)
    
    bandgap_figures = batch_analyzer.create_bandgap_scatter_plots(
        save=True,
        filename_prefix='MAPbX3_bandgap_vs'
    )
    
    if bandgap_figures:
        print(f"✓ Created {len(bandgap_figures)} bandgap scatter plots")
    else:
        print("✗ No bandgap scatter plots created (no bandgap data available)")
    
    # Create bandgap plots grouped by layer thickness (n=1,2,3)
    print("\nCreating bandgap plots grouped by layer thickness...")
    bandgap_by_n_figures = batch_analyzer.create_bandgap_by_layer_thickness_plots(
        save=True,
        filename_prefix='MAPbX3_bandgap_by_layer_thickness'
    )
    
    if bandgap_by_n_figures:
        print(f"✓ Created {len(bandgap_by_n_figures)} bandgap plots by layer thickness")
    else:
        print("✗ No bandgap plots by layer thickness created")
    
    # Load formation energy data and create formation energy plots
    print("\n" + "=" * 60)
    print("CREATING FORMATION ENERGY PLOTS")
    print("=" * 60)
    
    formation_energy_data = load_formation_energy_data()
    
    if formation_energy_data:
        # Create formation energy scatter plots
        formation_energy_figures = create_formation_energy_scatter_plots(
            batch_analyzer, formation_energy_data,
            save=True,
            filename_prefix='MAPbX3_formation_energy_vs'
        )
        
        if formation_energy_figures:
            print(f"✓ Created {len(formation_energy_figures)} formation energy scatter plots")
        else:
            print("✗ No formation energy scatter plots created")
        
        # Create formation energy plots grouped by layer thickness
        formation_energy_by_n_figures = create_formation_energy_by_layer_thickness_plots(
            batch_analyzer, formation_energy_data,
            save=True,
            filename_prefix='MAPbX3_formation_energy_by_layer_thickness'
        )
        
        if formation_energy_by_n_figures:
            print(f"✓ Created {len(formation_energy_by_n_figures)} formation energy plots by layer thickness")
        else:
            print("✗ No formation energy plots by layer thickness created")
    else:
        print("✗ No formation energy data available")
    
    # Create VBM and CBM plots
    print("\nCreating VBM and CBM plots...")
    vbm_cbm_figures = batch_analyzer.create_vbm_cbm_plots(
        save=True,
        filename_prefix='MAPbX3_vbm_cbm'
    )
    
    if vbm_cbm_figures:
        print(f"✓ Created {len(vbm_cbm_figures)} VBM/CBM plots")
    else:
        print("✗ No VBM/CBM plots created")
    
    # Load band edges data from CSV files
    band_edges_data = load_band_edges_data(dos_end_dir, batch_analyzer)
    
    # Export comprehensive data to CSV
    print("\n" + "=" * 60)
    print("EXPORTING COMPREHENSIVE DATA TO CSV")
    print("=" * 60)
    
    comprehensive_df = export_comprehensive_data_to_csv(
        batch_analyzer, 
        band_edges_data, 
        output_file=csv_file
    )
    
    if comprehensive_df is not None:
        print(f"✓ Comprehensive dataset exported successfully")
        print(f"  Total experiments: {len(comprehensive_df)}")
        print(f"  Total variables: {len(comprehensive_df.columns)}")
        
        # Now create all plots from the CSV
        print(f"\n✓ Now creating analysis plots from {csv_file}")
        
        # Create band edges plots
        if band_edges_data:
            print("\n" + "=" * 60)
            print("CREATING BAND EDGES PLOTS")
            print("=" * 60)
            band_edges_figures = create_band_edges_plots(band_edges_data, save=True)
            if band_edges_figures:
                print(f"✓ Created {len(band_edges_figures)} band edges plots")
        
        # Create correlation analysis from the CSV
        print("\n" + "=" * 60)
        print("CREATING CORRELATION ANALYSIS")
        print("=" * 60)
        
        correlation_figures = create_correlation_analysis_from_csv(csv_file)
        if correlation_figures:
            print(f"✓ Created {len(correlation_figures)} correlation plots")
        else:
            print("✗ No correlation analysis created")
        
        # Create high correlation scatter plots
        scatter_figures = create_high_correlation_scatter_plots(csv_file)
        if scatter_figures:
            print(f"✓ Created {len(scatter_figures)} high correlation scatter plots")
        else:
            print("✗ No high correlation scatter plots created")
        
        # Create molecular family trend plots
        family_trend_figures = create_molecular_family_trend_plots(csv_file)
        if family_trend_figures:
            print(f"✓ Created {len(family_trend_figures)} molecular family trend plots")
        else:
            print("✗ No molecular family trend plots created")
    else:
        print("✗ Failed to export comprehensive dataset")
    
    print("\n" + "=" * 80)
    print("REAL BATCH ANALYSIS COMPLETED")
    print("=" * 80)
    print("Generated files:")
    print(f"  Primary dataset:")
    print(f"    - {csv_file} (comprehensive dataset)")
    print("  Band edges analysis:")
    print("    - MAPbX3_vbm_by_X_site_and_layer.png")
    print("    - MAPbX3_cbm_by_X_site_and_layer.png")
    print("    - MAPbX3_homo_by_X_site_and_layer.png")
    print("    - MAPbX3_lumo_by_X_site_and_layer.png")
    print("    - MAPbX3_band_gap_by_X_site_and_layer.png")
    print("    - MAPbX3_vbm_cbm_combined_analysis.png")
    print("    - MAPbX3_homo_lumo_spacer_analysis.png")
    print("  Formation energy analysis:")
    print("    - MAPbX3_formation_energy_vs_delta.png")
    print("    - MAPbX3_formation_energy_vs_sigma.png")
    print("    - MAPbX3_formation_energy_vs_lambda_3.png")
    print("    - MAPbX3_formation_energy_vs_lambda_2.png")
    print("    - MAPbX3_formation_energy_vs_cis_angle_mean.png")
    print("    - MAPbX3_formation_energy_vs_trans_angle_mean.png")
    print("    - MAPbX3_formation_energy_vs_axial_central_axial_mean.png")
    print("    - MAPbX3_formation_energy_vs_central_axial_central_mean.png")
    print("    - MAPbX3_formation_energy_by_layer_thickness_scatter.png")
    print("    - MAPbX3_formation_energy_by_layer_thickness_boxplot.png")
    print("  Correlation analysis:")
    print("    - MAPbX3_correlation_pearson_heatmap.png")
    print("    - MAPbX3_correlation_spearman_heatmap.png")
    print("    - MAPbX3_correlation_comparison.png")
    print("    - MAPbX3_correlation_pearson_correlations.txt")
    print("    - MAPbX3_correlation_spearman_correlations.txt")
    print("    - MAPbX3_correlation_combined_analysis.txt")
    print("  High correlation scatter plots (grouped by molecular families):")
    print("    - MAPbX3_correlation_goldschmidt_tolerance_vs_bandgap.png")
    print("    - MAPbX3_correlation_delta_vs_goldschmidt_tolerance.png")
    print("    - MAPbX3_correlation_layer_thickness_vs_cell_volume.png")
    print("    - MAPbX3_correlation_spacer_cbm_vs_spacer_band_gap.png")
    print("    - MAPbX3_correlation_lambda_3_vs_lambda_2.png")
    print("    - MAPbX3_correlation_cell_volume_vs_bandgap.png")
    print("    - MAPbX3_correlation_bandgap_vs_vbm.png")
    print("    - MAPbX3_correlation_delta_vs_axial_central_axial_mean.png")
    print("    - MAPbX3_correlation_goldschmidt_tolerance_vs_spacer_vbm.png")
    print("    - MAPbX3_correlation_spacer_vbm_vs_spacer_band_gap.png")
    print("    - molecular_families_legend.png (molecular families legend)")
    print("  Molecular family trend plots:")
    print("    - MAPbX3_bandgap_by_molecular_family_trend.png")
    print("    - MAPbX3_vbm_by_molecular_family_trend.png")
    print("    - MAPbX3_cbm_by_molecular_family_trend.png")
    print("    - MAPbX3_spacer_vbm_by_molecular_family_trend.png")
    print("    - MAPbX3_spacer_cbm_by_molecular_family_trend.png")
    print("    - MAPbX3_spacer_band_gap_by_molecular_family_trend.png")
    print("\nAnalysis summary:")
    if 'loaded_count' in locals():
        print(f"  ✓ Analyzed {loaded_count} MAPbX3 structures")
    else:
        print(f"  ✓ Analyzed structures from {csv_file}")
    print("  ✓ Auto-detected X-site (Br, I, Cl) from structures")
    print("  ✓ Extracted layer thickness (n=1, n=2, n=3) from directory names")
    print("  ✓ Different spacer molecules compared")
    print("  ✓ Comprehensive structural and electronic data collected")
    print("  ✓ Band edges data (VBM, CBM, HOMO, LUMO) loaded from CSV files")
    print("  ✓ Band edges analysis grouped by X-site and layer thickness")
    print("  ✓ Spacer vs Slab band edges comparison")
    print("  ✓ Formation energy analysis from CSV data")
    print("  ✓ Formation energy vs structural parameters scatter plots")
    print("  ✓ Formation energy grouped by layer thickness analysis")
    print("  ✓ Linear (Pearson) and non-linear (Spearman) correlations calculated")
    print("  ✓ Comprehensive correlation matrices generated")
    print("  ✓ Statistical comparisons generated")
    print("  ✓ Publication-ready plots created")
    print(f"\nNext time: Run this script again and choose option 1 to use {csv_file} directly!")

if __name__ == "__main__":
    main()

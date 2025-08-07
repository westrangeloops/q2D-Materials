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

def extract_penetration_data(base_dir):
    """
    Extract penetration depth data from all JSON files in the analysis directories.
    
    Returns:
        pandas.DataFrame: Combined data with penetration segments for both molecules
    """
    data_list = []
    
    # Find all analysis directories
    analysis_dirs = [d for d in os.listdir(base_dir) if d.endswith('_analysis')]
    
    print(f"Found {len(analysis_dirs)} analysis directories")
    
    files_processed = 0
    files_with_penetration = 0
    molecules_found = 0
    errors_encountered = 0
    
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
                print(f"Skipping {dir_name}: Unknown halogen in {material}")
                continue
                
            # Extract n value
            n_octahedra = int(n_value[1:])  # Remove 'n' prefix
            
            # Look for the penetration analysis JSON file
            json_file = None
            dir_path = os.path.join(base_dir, dir_name)
            for file in os.listdir(dir_path):
                if file.endswith('_penetration_analysis.json'):
                    json_file = os.path.join(dir_path, file)
                    break
            
            if json_file is None:
                print(f"No penetration analysis file found in {dir_name}")
                continue
                
            files_processed += 1
            
            # Read JSON data
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check if penetration analysis is available
            penetration_data = data.get('penetration_analysis', {})
            if 'error' in penetration_data:
                print(f"Penetration analysis error in {dir_name}: {penetration_data['error']}")
                errors_encountered += 1
                continue
            
            files_with_penetration += 1
            molecules_in_this_file = 0
            
            # Extract data for all molecules dynamically
            molecule_keys = [key for key in penetration_data.keys() if key.startswith('molecule_')]
            
            if len(molecule_keys) != 2:
                print(f"WARNING: Expected 2 molecules in {dir_name}, found {len(molecule_keys)} molecules: {molecule_keys}")
            
            for mol_idx, mol_key in enumerate(sorted(molecule_keys)):
                if mol_key in penetration_data:
                    mol_data = penetration_data[mol_key]
                    segments = mol_data.get('penetration_segments', {})
                    
                    if segments:
                        # Determine molecule number for labeling (use order: first molecule = 1, second = 2)
                        mol_number = mol_idx + 1
                        
                        # Extract basic measurements
                        n1_low = segments.get('n1_to_low_plane', np.nan)
                        low_high = segments.get('low_plane_to_high_plane', np.nan)
                        high_n2 = segments.get('high_plane_to_n2', np.nan)
                        
                        # Calculate derived metrics
                        penetration_percentage = np.nan
                        penetration_asymmetry = np.nan
                        
                        if not np.isnan(n1_low) and not np.isnan(high_n2) and not np.isnan(low_high) and low_high > 0:
                            penetration_percentage = ((n1_low + high_n2) / low_high) * 100
                            penetration_asymmetry = n1_low - high_n2
                        
                        # Create data entry
                        entry = {
                            'experiment': dir_name,
                            'halogen_X': halogen,
                            'n_octahedra': n_octahedra,
                            'molecule_number': mol_number,
                            'molecule_key': mol_key,  # Add the original key for debugging
                            'molecule_formula': mol_data.get('formula', 'Unknown'),
                            'n1_to_low_plane': n1_low,
                            'low_plane_to_high_plane': low_high,
                            'high_plane_to_n2': high_n2,
                            'total_length': segments.get('molecular_length', np.nan),
                            'penetration_percentage': penetration_percentage,
                            'penetration_asymmetry': penetration_asymmetry
                        }
                        
                        data_list.append(entry)
                        molecules_found += 1
                        molecules_in_this_file += 1
                    else:
                        print(f"No penetration segments found for {mol_key} in {dir_name}")
                else:
                    print(f"Missing {mol_key} in {dir_name}")
            
            # Debug: Check if we got the expected molecules
            if molecules_in_this_file != len(molecule_keys):
                print(f"WARNING: Expected {len(molecule_keys)} molecules in {dir_name}, found {molecules_in_this_file}")
            
        except Exception as e:
            print(f"Error processing {dir_name}: {e}")
            errors_encountered += 1
            continue
    
    df = pd.DataFrame(data_list)
    
    # Print detailed summary
    print(f"\n=== EXTRACTION SUMMARY ===")
    print(f"Analysis directories found: {len(analysis_dirs)}")
    print(f"Files processed: {files_processed}")
    print(f"Files with valid penetration data: {files_with_penetration}")
    print(f"Total molecules found: {molecules_found}")
    print(f"Errors encountered: {errors_encountered}")
    print(f"Expected molecules (2 per valid file): {files_with_penetration * 2}")
    print(f"Missing molecules: {files_with_penetration * 2 - molecules_found}")
    
    if len(df) > 0:
        print(f"\n=== DATA BREAKDOWN ===")
        mol1_count = len(df[df['molecule_number'] == 1])
        mol2_count = len(df[df['molecule_number'] == 2])
        print(f"Molecule 1 entries: {mol1_count}")
        print(f"Molecule 2 entries: {mol2_count}")
        print(f"Unique experiments: {len(df['experiment'].unique())}")
        print(f"Halogens found: {df['halogen_X'].unique()}")
        print(f"N values found: {sorted(df['n_octahedra'].unique())}")
        print(f"Formulas found: {df['molecule_formula'].unique()}")
    
    return df

def create_penetration_depth_plots(df, output_dir):
    """
    Create side-by-side box plots for penetration depth segments of molecules 1 and 2.
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
    
    # Define color palette for molecules
    mol_colors = {
        1: '#FF6B6B',  # Red for Molecule 1
        2: '#4ECDC4'   # Teal for Molecule 2
    }
    
    # Define color palette for halogens
    halogen_colors = {
        'Cl': '#2E8B57',  # Sea green
        'Br': '#FF6B35',  # Orange-red  
        'I': '#4A90E2'    # Blue
    }
    
    # Define color palette for n values
    n_colors = {
        1: '#FF6B6B',  # Red
        2: '#4ECDC4',  # Teal
        3: '#45B7D1'   # Blue
    }
    
    # 1. Main plot: Three penetration segments for both molecules side by side
    segments = ['n1_to_low_plane', 'low_plane_to_high_plane', 'high_plane_to_n2']
    segment_labels = ['N1 → Low Plane', 'Low → High Plane', 'High Plane → N2']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Penetration Depth Segments: Molecule 1 vs Molecule 2\nSpacer Molecule Analysis', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    for i, (segment, label) in enumerate(zip(segments, segment_labels)):
        ax = axes[i]
        
        # Prepare data for side-by-side box plots
        mol1_data = df[df['molecule_number'] == 1][segment].dropna().values
        mol2_data = df[df['molecule_number'] == 2][segment].dropna().values
        
        # Create box plot
        box_plot = ax.boxplot([mol1_data, mol2_data], 
                             labels=['Molecule 1', 'Molecule 2'],
                             patch_artist=True)
        
        # Style the boxes
        box_plot['boxes'][0].set_facecolor(mol_colors[1])
        box_plot['boxes'][0].set_alpha(0.7)
        box_plot['boxes'][1].set_facecolor(mol_colors[2])
        box_plot['boxes'][1].set_alpha(0.7)
        
        # Style other elements
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box_plot[element], color='black', linewidth=1)
        
        ax.set_title(f'{label}', fontweight='bold', pad=10)
        ax.set_ylabel('Distance (Å)', fontweight='bold')
        ax.set_xlabel('Molecule', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits based on segment type
        if segment in ['n1_to_low_plane', 'high_plane_to_n2']:
            ax.set_ylim(0, 2.0)  # Penetration segments limited to 2.0 Å
        elif segment == 'low_plane_to_high_plane':
            ax.set_ylim(0, 25.0)  # Distance between planes limited to 25 Å
        
        # Add sample sizes
        n_mol1 = len(mol1_data)
        n_mol2 = len(mol2_data)
        ax.text(1, ax.get_ylim()[0], f'n={n_mol1}', 
               ha='center', va='top', fontsize=9, style='italic')
        ax.text(2, ax.get_ylim()[0], f'n={n_mol2}', 
               ha='center', va='top', fontsize=9, style='italic')
        
        # Add statistical information
        if len(mol1_data) > 0 and len(mol2_data) > 0:
            mol1_mean = np.mean(mol1_data)
            mol1_std = np.std(mol1_data)
            mol2_mean = np.mean(mol2_data)
            mol2_std = np.std(mol2_data)
            
            # Add mean values as text
            ax.text(0.02, 0.98, f'Mol 1: {mol1_mean:.3f}±{mol1_std:.3f} Å', 
                   transform=ax.transAxes, va='top', ha='left', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)
            ax.text(0.02, 0.85, f'Mol 2: {mol2_mean:.3f}±{mol2_std:.3f} Å', 
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'penetration_segments_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plots by halogen type for each segment
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Penetration Depth Segments by Halogen Type\nMolecule 1 vs Molecule 2', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    for i, (segment, label) in enumerate(zip(segments, segment_labels)):
        ax = axes[i]
        
        # Prepare data grouped by halogen
        halogens = ['Cl', 'Br', 'I']
        positions = []
        box_data = []
        colors_list = []
        labels_list = []
        
        for j, halogen in enumerate(halogens):
            for mol_num in [1, 2]:
                data = df[(df['halogen_X'] == halogen) & (df['molecule_number'] == mol_num)][segment].dropna().values
                if len(data) > 0:
                    box_data.append(data)
                    positions.append(j*3 + mol_num)  # Group by halogen, separate molecules
                    colors_list.append(mol_colors[mol_num])
                    labels_list.append(f'{halogen}\nMol {mol_num}')
        
        if box_data:
            # Create box plot
            box_plot = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
            
            # Style the boxes
            for patch, color in zip(box_plot['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Style other elements
            for element in ['whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(box_plot[element], color='black', linewidth=1)
            
            # Set labels and ticks
            ax.set_xticks(positions)
            ax.set_xticklabels(labels_list, rotation=0, ha='center')
            
        ax.set_title(f'{label}', fontweight='bold', pad=10)
        ax.set_ylabel('Distance (Å)', fontweight='bold')
        ax.set_xlabel('Halogen / Molecule', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits based on segment type
        if segment in ['n1_to_low_plane', 'high_plane_to_n2']:
            ax.set_ylim(0, 2.0)  # Penetration segments limited to 2.0 Å
        elif segment == 'low_plane_to_high_plane':
            ax.set_ylim(0, 25.0)  # Distance between planes limited to 25 Å
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'penetration_segments_by_halogen.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Box plots by number of layers for each segment
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Penetration Depth Segments by Number of Layers\nMolecule 1 vs Molecule 2', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    for i, (segment, label) in enumerate(zip(segments, segment_labels)):
        ax = axes[i]
        
        # Prepare data grouped by n_octahedra
        n_values = sorted(df['n_octahedra'].unique())
        positions = []
        box_data = []
        colors_list = []
        labels_list = []
        
        for j, n in enumerate(n_values):
            for mol_num in [1, 2]:
                data = df[(df['n_octahedra'] == n) & (df['molecule_number'] == mol_num)][segment].dropna().values
                if len(data) > 0:
                    box_data.append(data)
                    positions.append(j*3 + mol_num)  # Group by n, separate molecules
                    colors_list.append(mol_colors[mol_num])
                    labels_list.append(f'n={n}\nMol {mol_num}')
        
        if box_data:
            # Create box plot
            box_plot = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
            
            # Style the boxes
            for patch, color in zip(box_plot['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Style other elements
            for element in ['whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(box_plot[element], color='black', linewidth=1)
            
            # Set labels and ticks
            ax.set_xticks(positions)
            ax.set_xticklabels(labels_list, rotation=0, ha='center')
            
        ax.set_title(f'{label}', fontweight='bold', pad=10)
        ax.set_ylabel('Distance (Å)', fontweight='bold')
        ax.set_xlabel('Layers / Molecule', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits based on segment type
        if segment in ['n1_to_low_plane', 'high_plane_to_n2']:
            ax.set_ylim(0, 2.0)  # Penetration segments limited to 2.0 Å
        elif segment == 'low_plane_to_high_plane':
            ax.set_ylim(0, 25.0)  # Distance between planes limited to 25 Å
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'penetration_segments_by_layers.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Penetration Percentage by Halogen
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Penetration Percentage by Halogen Type\n[(N1→Low + High→N2) / (Low→High)] × 100', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Prepare data grouped by halogen
    halogens = ['Cl', 'Br', 'I']
    positions = []
    box_data = []
    colors_list = []
    labels_list = []
    
    for j, halogen in enumerate(halogens):
        for mol_num in [1, 2]:
            data = df[(df['halogen_X'] == halogen) & (df['molecule_number'] == mol_num)]['penetration_percentage'].dropna().values
            if len(data) > 0:
                box_data.append(data)
                positions.append(j*3 + mol_num)  # Group by halogen, separate molecules
                colors_list.append(mol_colors[mol_num])
                labels_list.append(f'{halogen}\nMol {mol_num}')
    
    if box_data:
        # Create box plot
        box_plot = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
        
        # Style the boxes
        for patch, color in zip(box_plot['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Style other elements
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box_plot[element], color='black', linewidth=1)
        
        # Set labels and ticks
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_list, rotation=0, ha='center')
    
    ax.set_title('Penetration Percentage by Halogen', fontweight='bold', pad=10)
    ax.set_ylabel('Penetration Percentage (%)', fontweight='bold')
    ax.set_xlabel('Halogen / Molecule', fontweight='bold')
    ax.set_ylim(0, 50)  # Limit to 50% for readability
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'penetration_percentage_by_halogen.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Penetration Percentage by Number of Layers
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Penetration Percentage by Number of Layers\n[(N1→Low + High→N2) / (Low→High)] × 100', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Prepare data grouped by n_octahedra
    n_values = sorted(df['n_octahedra'].unique())
    positions = []
    box_data = []
    colors_list = []
    labels_list = []
    
    for j, n in enumerate(n_values):
        for mol_num in [1, 2]:
            data = df[(df['n_octahedra'] == n) & (df['molecule_number'] == mol_num)]['penetration_percentage'].dropna().values
            if len(data) > 0:
                box_data.append(data)
                positions.append(j*3 + mol_num)  # Group by n, separate molecules
                colors_list.append(mol_colors[mol_num])
                labels_list.append(f'n={n}\nMol {mol_num}')
    
    if box_data:
        # Create box plot
        box_plot = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
        
        # Style the boxes
        for patch, color in zip(box_plot['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Style other elements
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box_plot[element], color='black', linewidth=1)
        
        # Set labels and ticks
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_list, rotation=0, ha='center')
    
    ax.set_title('Penetration Percentage by Number of Layers', fontweight='bold', pad=10)
    ax.set_ylabel('Penetration Percentage (%)', fontweight='bold')
    ax.set_xlabel('Layers / Molecule', fontweight='bold')
    ax.set_ylim(0, 50)  # Limit to 50% for readability
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'penetration_percentage_by_layers.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Penetration Asymmetry by Halogen
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Penetration Asymmetry by Halogen Type\n(N1→Low) - (High→N2)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Prepare data grouped by halogen
    halogens = ['Cl', 'Br', 'I']
    positions = []
    box_data = []
    colors_list = []
    labels_list = []
    
    for j, halogen in enumerate(halogens):
        for mol_num in [1, 2]:
            data = df[(df['halogen_X'] == halogen) & (df['molecule_number'] == mol_num)]['penetration_asymmetry'].dropna().values
            if len(data) > 0:
                box_data.append(data)
                positions.append(j*3 + mol_num)  # Group by halogen, separate molecules
                colors_list.append(mol_colors[mol_num])
                labels_list.append(f'{halogen}\nMol {mol_num}')
    
    if box_data:
        # Create box plot
        box_plot = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
        
        # Style the boxes
        for patch, color in zip(box_plot['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Style other elements
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box_plot[element], color='black', linewidth=1)
        
        # Set labels and ticks
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_list, rotation=0, ha='center')
    
    ax.set_title('Penetration Asymmetry by Halogen', fontweight='bold', pad=10)
    ax.set_ylabel('Asymmetry (Å)', fontweight='bold')
    ax.set_xlabel('Halogen / Molecule', fontweight='bold')
    ax.set_ylim(-2.0, 2.0)  # Symmetric range around 0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)  # Reference line at 0
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'penetration_asymmetry_by_halogen.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Penetration Asymmetry by Number of Layers
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Penetration Asymmetry by Number of Layers\n(N1→Low) - (High→N2)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Prepare data grouped by n_octahedra
    n_values = sorted(df['n_octahedra'].unique())
    positions = []
    box_data = []
    colors_list = []
    labels_list = []
    
    for j, n in enumerate(n_values):
        for mol_num in [1, 2]:
            data = df[(df['n_octahedra'] == n) & (df['molecule_number'] == mol_num)]['penetration_asymmetry'].dropna().values
            if len(data) > 0:
                box_data.append(data)
                positions.append(j*3 + mol_num)  # Group by n, separate molecules
                colors_list.append(mol_colors[mol_num])
                labels_list.append(f'n={n}\nMol {mol_num}')
    
    if box_data:
        # Create box plot
        box_plot = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
        
        # Style the boxes
        for patch, color in zip(box_plot['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Style other elements
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box_plot[element], color='black', linewidth=1)
        
        # Set labels and ticks
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_list, rotation=0, ha='center')
    
    ax.set_title('Penetration Asymmetry by Number of Layers', fontweight='bold', pad=10)
    ax.set_ylabel('Asymmetry (Å)', fontweight='bold')
    ax.set_xlabel('Layers / Molecule', fontweight='bold')
    ax.set_ylim(-2.0, 2.0)  # Symmetric range around 0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)  # Reference line at 0
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'penetration_asymmetry_by_layers.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Statistical summary
    print("\n" + "="*80)
    print("PENETRATION DEPTH STATISTICAL SUMMARY")
    print("="*80)
    
    print("\nPenetration Segments by Molecule:")
    print("-"*50)
    for mol_num in [1, 2]:
        subset = df[df['molecule_number'] == mol_num]
        print(f"\nMolecule {mol_num} (n={len(subset)}):")
        for segment, label in zip(segments, segment_labels):
            data = subset[segment].dropna()
            if len(data) > 0:
                mean_val = data.mean()
                std_val = data.std()
                print(f"  {label}: {mean_val:.3f} ± {std_val:.3f} Å")
            else:
                print(f"  {label}: No data")
        
        # Add penetration percentage and asymmetry
        pen_perc_data = subset['penetration_percentage'].dropna()
        pen_asym_data = subset['penetration_asymmetry'].dropna()
        
        if len(pen_perc_data) > 0:
            print(f"  Penetration Percentage: {pen_perc_data.mean():.2f} ± {pen_perc_data.std():.2f} %")
        else:
            print(f"  Penetration Percentage: No data")
            
        if len(pen_asym_data) > 0:
            print(f"  Penetration Asymmetry: {pen_asym_data.mean():.3f} ± {pen_asym_data.std():.3f} Å")
        else:
            print(f"  Penetration Asymmetry: No data")
    
    print("\nPenetration Segments by Halogen:")
    print("-"*50)
    for halogen in ['Cl', 'Br', 'I']:
        print(f"\n{halogen}:")
        for mol_num in [1, 2]:
            subset = df[(df['halogen_X'] == halogen) & (df['molecule_number'] == mol_num)]
            print(f"  Molecule {mol_num} (n={len(subset)}):")
            for segment, label in zip(segments, segment_labels):
                data = subset[segment].dropna()
                if len(data) > 0:
                    mean_val = data.mean()
                    std_val = data.std()
                    print(f"    {label}: {mean_val:.3f} ± {std_val:.3f} Å")
    
    print("\nPenetration Segments by Number of Layers:")
    print("-"*60)
    for n in sorted(df['n_octahedra'].unique()):
        print(f"\nn={n}:")
        for mol_num in [1, 2]:
            subset = df[(df['n_octahedra'] == n) & (df['molecule_number'] == mol_num)]
            print(f"  Molecule {mol_num} (n={len(subset)}):")
            for segment, label in zip(segments, segment_labels):
                data = subset[segment].dropna()
                if len(data) > 0:
                    mean_val = data.mean()
                    std_val = data.std()
                    print(f"    {label}: {mean_val:.3f} ± {std_val:.3f} Å")

def main():
    # Set base directory
    base_dir = '/home/dotempo/Documents/REPO/SVC-Materials/Graphs/BULKS_RESULTS'
    output_dir = os.path.join(base_dir, 'penetration_depth_analysis')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting penetration depth analysis...")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    # Extract data
    df = extract_penetration_data(base_dir)
    
    if df.empty:
        print("No penetration depth data found!")
        return
    
    # Data processing complete
    print(f"Processed {len(df)} molecule entries")
    print(f"Unique experiments: {len(df['experiment'].unique())}")
    print(f"Molecule formulas found: {df['molecule_formula'].unique()}")
    
    # Create plots
    print("Creating penetration depth visualizations...")
    create_penetration_depth_plots(df, output_dir)
    
    # Save the dataframe for further analysis
    df.to_csv(os.path.join(output_dir, 'penetration_depth_data.csv'), index=False)
    
    print(f"\nPenetration depth analysis complete! Check the '{output_dir}' directory for results.")
    print("Generated files:")
    print("- penetration_segments_comparison.png")
    print("- penetration_segments_by_halogen.png") 
    print("- penetration_segments_by_layers.png")
    print("- penetration_percentage_by_halogen.png")
    print("- penetration_percentage_by_layers.png")
    print("- penetration_asymmetry_by_halogen.png")
    print("- penetration_asymmetry_by_layers.png")
    print("- penetration_depth_data.csv")

if __name__ == "__main__":
    main() 
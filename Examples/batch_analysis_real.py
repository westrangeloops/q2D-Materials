#!/usr/bin/env python3
"""
Real Batch Analysis for q2D Materials.

This script performs batch analysis on real MAPbBr3 structures with different
spacer molecules from the DOS_END directory.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q2D_Materials.core.batch_analysis import BatchAnalyzer
from q2D_Materials.core.analyzer import q2D_analyzer

def main():
    """Main function to analyze real MAPbBr3 structures."""
    
    print("=" * 80)
    print("q2D Materials - Real Batch Analysis")
    print("Analyzing MAPbBr3 structures with different spacers")
    print("=" * 80)
    
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
    
    # Group by layer thickness (n=1, n=2, n=3) - All angle types
    print("Creating cis angles distribution plot grouped by layer thickness...")
    fig4 = batch_analyzer.create_bond_angle_distribution_comparison(
        group_by='layer_thickness',
        angle_type='cis_angles',
        show_mean_std=True,
        save=True,
        filename='MAPbX3_cis_angles_by_layer_thickness.png'
    )
    
    if fig4:
        print("✓ Cis angles by layer thickness plot created successfully")
    else:
        print("✗ Failed to create cis angles by layer thickness plot")
    
    print("Creating trans angles distribution plot grouped by layer thickness...")
    fig5 = batch_analyzer.create_bond_angle_distribution_comparison(
        group_by='layer_thickness',
        angle_type='trans_angles',
        show_mean_std=True,
        save=True,
        filename='MAPbX3_trans_angles_by_layer_thickness.png'
    )
    
    if fig5:
        print("✓ Trans angles by layer thickness plot created successfully")
    else:
        print("✗ Failed to create trans angles by layer thickness plot")
    
    print("Creating axial-central-axial angles distribution plot grouped by layer thickness...")
    fig6 = batch_analyzer.create_bond_angle_distribution_comparison(
        group_by='layer_thickness',
        angle_type='axial_central_axial',
        show_mean_std=True,
        save=True,
        filename='MAPbX3_axial_central_axial_by_layer_thickness.png'
    )
    
    if fig6:
        print("✓ Axial-central-axial angles by layer thickness plot created successfully")
    else:
        print("✗ Failed to create axial-central-axial angles by layer thickness plot")
    
    print("\n" + "=" * 80)
    print("REAL BATCH ANALYSIS COMPLETED")
    print("=" * 80)
    print("Generated files:")
    print("  X-site grouped plots:")
    print("    - MAPbX3_cis_angles_by_X_site.png")
    print("    - MAPbX3_trans_angles_by_X_site.png")
    print("    - MAPbX3_axial_central_axial_by_X_site.png")
    print("  Layer thickness grouped plots:")
    print("    - MAPbX3_cis_angles_by_layer_thickness.png")
    print("    - MAPbX3_trans_angles_by_layer_thickness.png")
    print("    - MAPbX3_axial_central_axial_by_layer_thickness.png")
    print("\nAnalysis summary:")
    print(f"  ✓ Analyzed {loaded_count} MAPbX3 structures")
    print("  ✓ Auto-detected X-site (Br, I, Cl) from structures")
    print("  ✓ Extracted layer thickness (n=1, n=2, n=3) from directory names")
    print("  ✓ Different spacer molecules compared")
    print("  ✓ Bond angle distributions analyzed")
    print("  ✓ Statistical comparisons generated")
    print("  ✓ Dynamic x-axis optimization applied")

if __name__ == "__main__":
    main()

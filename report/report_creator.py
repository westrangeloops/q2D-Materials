"""
Test script for analyzing molecule deformations and structural properties.
"""

from pathlib import Path
import pandas as pd
from SVC_materials.utils.isolate_molecule import analyze_molecule_deformation
from SVC_materials.core.analyzer import q2D_analysis
from ase.io import read
import logging
import sys

def get_molecule_name(filename: str) -> str:
    """Extract molecule name from perovskite filename."""
    return filename.replace('.xyz', '')

def get_halogen(perov_type: str) -> str:
    """Extract halogen type from perovskite name."""
    if perov_type == "MAPbCl3":
        return "Cl"
    elif perov_type == "MAPbI3":
        return "I"
    elif perov_type == "MAPbBr3":
        return "Br"
    else:
        raise ValueError(f"Unknown perovskite type: {perov_type}")

def main():
    # --- USER-DEFINED PATHS ---
    OUTPUT_DIR = Path("/home/dotempo/Documents/REPOS/SVC-Materials/report/final_report")
    TEMPLATE_DIR = Path("~/Documents/DION-JACOBSON/MOLECULES/").expanduser()
    INPUT_DIR = Path("/home/dotempo/Documents/REPOS/SVC-Materials/report/data/xyz")
    BULK_DIR = Path("~/Documents/DION-JACOBSON/BULKS/").expanduser()
    
    # Print directory information
    print("\nDirectory Information:")
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Template Directory: {TEMPLATE_DIR}")
    print(f"Bulk Directory: {BULK_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Check if directories exist
    if not INPUT_DIR.exists():
        print(f"Error: Input directory does not exist: {INPUT_DIR}")
        return
    if not TEMPLATE_DIR.exists():
        print(f"Error: Template directory does not exist: {TEMPLATE_DIR}")
        return
    if not BULK_DIR.exists():
        print(f"Error: Bulk directory does not exist: {BULK_DIR}")
        return
    if not OUTPUT_DIR.exists():
        print(f"Error: Output directory does not exist: {OUTPUT_DIR}")
        return
    
    # Initialize results storage
    summary_results = []
    
    # Get list of processed molecules
    processed_files = list(INPUT_DIR.glob("*.xyz"))
    print(f"\nFound {len(processed_files)} .xyz files in input directory")
    
    if len(processed_files) == 0:
        print("No .xyz files found in input directory")
        return
    
    for extracted_file in processed_files:
        # Get full molecule name (including perovskite prefix)
        molecule_name = get_molecule_name(extracted_file.name)
        print(f"\nProcessing {molecule_name}...")
        
        # Extract perovskite type and SMILES for template matching
        try:
            parts = molecule_name.split('_', 2)
            perov_type = parts[0]  # e.g., MAPbBr3
            smiles = parts[2]  # Get everything after MAPbX3_nN_
            print(f"Perovskite type: {perov_type}")
            print(f"SMILES: {smiles}")
            
            # Get halogen type
            halogen = get_halogen(perov_type)
            print(f"Halogen type: {halogen}")
        except Exception as e:
            print(f"Error parsing molecule name {molecule_name}: {e}")
            continue
        
        # Find the template CONTCAR in the subdirectory
        template_subdir = TEMPLATE_DIR / smiles
        template_file = template_subdir / "CONTCAR"
        
        if not template_file.exists():
            print(f"Warning: Template CONTCAR not found in {template_subdir}")
            continue
        
        print(f"Using template: {template_file}")
        
        try:
            # Analyze molecule deformation
            print("Analyzing molecule deformation...")
            metrics = analyze_molecule_deformation(
                ideal_file=template_file,
                extracted_file=extracted_file,
                output_dir=OUTPUT_DIR,
                debug=True,  # Enable debug to see detailed comparison
                include_hydrogens=False
            )
            
            # Find corresponding perovskite file
            perov_file = BULK_DIR / perov_type / molecule_name / "CONTCAR"
            print(f"Looking for perovskite file: {perov_file}")
            
            if not perov_file.exists():
                print(f"Warning: Perovskite file not found: {perov_file}")
                continue
                
            # Read energy from vasprun.xml
            vasprun_file = BULK_DIR / perov_type / molecule_name / "vasprun.xml"
            print(f"Looking for vasprun.xml: {vasprun_file}")
            
            if vasprun_file.exists():
                try:
                    print("Reading energy from vasprun.xml...")
                    atoms = read(str(vasprun_file))
                    energy = atoms.get_potential_energy()
                    print(f"Energy: {energy:.6f} eV")
                except Exception as e:
                    print(f"Warning: Could not read energy from vasprun.xml: {e}")
                    energy = None
            else:
                print(f"Warning: vasprun.xml not found: {vasprun_file}")
                energy = None
                
            # Initialize analyzer for structural analysis
            print("Initializing structural analyzer...")
            analyzer = q2D_analysis(B='Pb', X=halogen, crystal=str(perov_file))
            
            # Analyze perovskite structure
            print("Analyzing perovskite structure...")
            structure_data = analyzer.analyze_perovskite_structure()
            
            # Calculate nitrogen penetration
            print("Calculating nitrogen penetration...")
            penetration_data = analyzer.calculate_n_penetration()
            
            if 'error' not in metrics and structure_data and penetration_data:
                # Store summary metrics
                result = {
                    'molecule': molecule_name,
                    'perovskite_type': perov_type,
                    'energy': energy,
                    **metrics['summary']
                }
                
                # Add structural analysis data
                if structure_data:
                    result.update({
                        'avg_axial_angle': sum(structure_data['axial_angles']) / len(structure_data['axial_angles']),
                        'avg_equatorial_angle': sum(structure_data['equatorial_angles']) / len(structure_data['equatorial_angles']),
                        'avg_axial_length': sum(structure_data['axial_lengths']) / len(structure_data['axial_lengths']),
                        'avg_equatorial_length': sum(structure_data['equatorial_lengths']) / len(structure_data['equatorial_lengths']),
                        'avg_out_of_plane_distortion': sum(structure_data['out_of_plane_distortions']) / len(structure_data['out_of_plane_distortions']),
                        'num_octahedra': len(structure_data['per_octahedron'])
                    })
                
                # Add penetration data
                if penetration_data:
                    result.update({
                        'lower_penetration': penetration_data['lower_penetration'],
                        'upper_penetration': penetration_data['upper_penetration'],
                        'total_penetration': penetration_data['total_penetration']
                    })
                
                summary_results.append(result)
                print(f"✓ Successfully analyzed {molecule_name}")
                print(f"  - Molecular deformation: {len(metrics['summary'])} metrics")
                print(f"  - Structural analysis: {len(structure_data['per_octahedron'])} octahedra analyzed")
                print(f"  - Nitrogen penetration: {penetration_data['total_penetration']:.3f} Å")
                if energy is not None:
                    print(f"  - Energy: {energy:.6f} eV")
            else:
                print(f"Error analyzing {molecule_name}: {metrics.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Error processing {molecule_name}: {str(e)}")
            import traceback
            print("Full error traceback:")
            print(traceback.format_exc())
            continue
    
    # Save summary results
    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        df_summary.to_csv(OUTPUT_DIR / "analysis_summary.csv", index=False)
        print(f"\nAnalysis summary saved to: {OUTPUT_DIR}/analysis_summary.csv")
        print(f"Successfully analyzed {len(df_summary)} molecules")
    else:
        print("\nNo molecules were successfully analyzed")

if __name__ == "__main__":
    main() 
"""
Script to process CONTCAR files and isolate molecules using the existing isolate_spacer function.
"""

import os
from pathlib import Path
import shutil
from SVC_materials.core.analyzer import q2D_analysis
from ase.io import read, write
from SVC_materials.utils.coordinate_ops import shift_structure

def process_contcars(base_dir: str, output_dir: str, shift_direction='a', shift_amount=0.5):
    """
    Process CONTCAR files from the base directory and save original and isolated molecules.
    
    Args:
        base_dir (str): Base directory containing the perovskite folders
        output_dir (str): Output directory for processed files
        shift_direction (str): Direction to shift ('a', 'b', or 'c')
        shift_amount (float): Amount to shift in fractional coordinates
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    
    # Create output directories if they don't exist
    original_dir = output_path / 'original'
    molecules_dir = output_path / 'molecules'
    original_dir.mkdir(parents=True, exist_ok=True)
    molecules_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each perovskite type folder
    for perovskite_dir in base_path.iterdir():
        if not perovskite_dir.is_dir():
            continue
            
        print(f"\nProcessing {perovskite_dir.name}...")
        
        # Process each calculation folder
        for calc_dir in perovskite_dir.iterdir():
            if not calc_dir.is_dir():
                continue
                
            contcar_path = calc_dir / 'CONTCAR'
            if not contcar_path.exists():
                print(f"No CONTCAR found in {calc_dir}")
                continue
                
            # Create output filename based on the calculation directory name
            output_name = f"{calc_dir.name}.vasp"
            
            # Read and shift the structure
            try:
                atoms = read(contcar_path)
                shifted_atoms = shift_structure(atoms, direction=shift_direction, shift=shift_amount)
                
                # Save shifted structure
                original_output = original_dir / output_name
                write(original_output, shifted_atoms, format='vasp')
                print(f"Saved shifted structure to {original_output}")
                
                # Isolate molecule
                # Initialize analyzer with appropriate B and X atoms
                if 'Br' in perovskite_dir.name:
                    analyzer = q2D_analysis(B='Pb', X='Br', crystal=str(original_output))
                elif 'Cl' in perovskite_dir.name:
                    analyzer = q2D_analysis(B='Pb', X='Cl', crystal=str(original_output))
                else:  # I
                    analyzer = q2D_analysis(B='Pb', X='I', crystal=str(original_output))
                
                # Save isolated molecule
                molecule_output = molecules_dir / output_name
                if analyzer.save_spacer(name=str(molecule_output)):
                    print(f"Saved isolated molecule to {molecule_output}")
                else:
                    print(f"Failed to isolate molecule from {contcar_path}")
                    
            except Exception as e:
                print(f"Error processing {contcar_path}: {e}")

if __name__ == "__main__":
    base_dir = "/home/dotempo/Documents/DION-JACOBSON/BULKS"
    output_dir = "SVC_materials/ml_isolator/data"
    
    # Process with a shift of 0.5 in the a direction
    process_contcars(base_dir, output_dir, shift_direction='a', shift_amount=0.5) 
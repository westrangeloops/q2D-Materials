"""
Script to read isolated molecules and apply a shift of 0.5 in the a-direction to the unit cell.
Uses ASE's built-in functionality to handle periodic boundary conditions.
"""

import os
from pathlib import Path
from ase.io import read, write
from ase.geometry import wrap_positions
import numpy as np

def shift_unit_cell(atoms, shift_vector):
    """
    Shift the unit cell by a vector in direct coordinates using ASE's functionality.
    
    Args:
        atoms: ASE Atoms object
        shift_vector (np.ndarray): Shift vector in direct coordinates [a, b, c]
        
    Returns:
        ASE Atoms object with shifted cell
    """
    # Get the cell vectors
    cell = atoms.get_cell()
    
    # Convert shift vector from direct to cartesian coordinates
    cartesian_shift = np.dot(shift_vector, cell)
    
    # Get current positions
    positions = atoms.get_positions()
    
    # Apply shift to positions
    shifted_positions = positions + cartesian_shift
    
    # Wrap positions back into the cell using ASE's wrap_positions
    wrapped_positions = wrap_positions(shifted_positions, cell, pbc=atoms.get_pbc())
    
    # Set the new positions
    atoms.set_positions(wrapped_positions)
    
    return atoms

def transform_molecules(input_dir: str):
    """
    Read isolated molecules and shift the unit cell by 0.5 in the a-direction.
    Uses ASE's functionality to handle periodic boundary conditions.
    Saves the transformed molecules with the same name, overwriting the original files.
    
    Args:
        input_dir (str): Directory containing the isolated molecules
    """
    input_path = Path(input_dir)
    
    # Define shift vector in direct coordinates [a, b, c]
    shift_vector = np.array([0.5, 0.0, 0.0])
    
    # Process each molecule file
    for molecule_file in input_path.glob('*.vasp'):
        try:
            # Read the molecule
            atoms = read(molecule_file)
            
            # Ensure periodic boundary conditions are set
            atoms.set_pbc([True, True, True])
            
            # Print original cell and positions
            print(f"\nProcessing {molecule_file.name}")
            print("Original cell:")
            print(atoms.get_cell())
            print("Original positions (first atom):")
            print(atoms.get_positions()[0])
            
            # Shift the unit cell
            shifted_atoms = shift_unit_cell(atoms, shift_vector)
            
            # Print new cell and positions
            print("Shifted cell:")
            print(shifted_atoms.get_cell())
            print("Shifted positions (first atom):")
            print(shifted_atoms.get_positions()[0])
            
            # Save the transformed molecule with the same name
            write(molecule_file, shifted_atoms)
            print(f"Saved to: {molecule_file}")
            
        except Exception as e:
            print(f"Error processing {molecule_file}: {e}")

if __name__ == "__main__":
    input_dir = "SVC_materials/ml_isolator/data/molecules"
    transform_molecules(input_dir) 
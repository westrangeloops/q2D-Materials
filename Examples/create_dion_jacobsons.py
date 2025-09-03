#!/usr/bin/env python3
"""
Create Dion-Jacobson (DJ) Perovskite Structures

Creates DJ structures for MAPbX3 (X = Cl, I, Br) using methylammonium as A-site cation.
Note: In DJ structures, MA is the A-site cation (between layers), not the spacer.
"""

import os
import numpy as np
from ase import Atoms
from ase.io import write
from q2D_Materials.utils.file_handlers import mol_load
from q2D_Materials.utils.perovskite_builder import make_dj, auto_calculate_BX_distance
from q2D_Materials.utils.common_a_sites import get_a_site_object

def main():
    """
    Create DJ structures for MAPbX3 compositions.
    """
    print("Creating DJ structures...")
    
    # Compositions and parameters
    compositions = {
        'MAPbI3': {'B': 'Pb', 'X': 'I'},
        'MAPbCl3': {'B': 'Pb', 'X': 'Cl'},
        'MAPbBr3': {'B': 'Pb', 'X': 'Br'},
        'MASnI3': {'B': 'Sn', 'X': 'I'},
        'MASnCl3': {'B': 'Sn', 'X': 'Cl'},
        'MASnBr3': {'B': 'Sn', 'X': 'Br'},
    }
    n_layers = [1, 2, 3, 4]
    b_sites = ['Pb', 'Sn']
    x_sites = ['I', 'Cl', 'Br']
    penetration_a = 0.2

    # Load A-site cation molecule (methylammonium)
    # Note: In DJ structures, MA is the A-site cation, not the spacer
    ma_xyz_file = "CamiloITM/methylammonium.xyz"
    if not os.path.exists(ma_xyz_file):
        print(f"Error: {ma_xyz_file} not found")
        return
    
    ma_df = mol_load(ma_xyz_file)
    elements = ma_df['Element'].tolist()
    positions = ma_df[['X', 'Y', 'Z']].values
    ma_a_site_atoms = Atoms(symbols=elements, positions=positions)
    
    # Load spacer files (all .xyz files except methylammonium)
    spacer_files = [f for f in os.listdir("CamiloITM") if f.endswith('.xyz') and f != 'methylammonium.xyz']
    
    if not spacer_files:
        print("Error: No spacer files found in CamiloITM/ directory")
        return
    
    print(f"Found {len(spacer_files)} spacer files: {spacer_files}")
    
    # Create structures for all combinations
    for name, comp in compositions.items():
        B_cation = comp['B']
        X_anion = comp['X']
        
        print(f"\nProcessing {name}:")
        
        # Calculate B-X distance and penetration
        bx_dist = auto_calculate_BX_distance(B_cation, X_anion)
        penetration_fraction = penetration_a / bx_dist
        
        # A-site cation (MA) that goes between inorganic layers
        A_cation = get_a_site_object("MA")
        
        # Process each spacer
        for spacer_file in spacer_files:
            print(f"  Using spacer {spacer_file}")
            
            # Load spacer molecule
            spacer_df = mol_load(f"CamiloITM/{spacer_file}")
            spacer_elements = spacer_df['Element'].tolist()
            spacer_positions = spacer_df[['X', 'Y', 'Z']].values
            spacer_name = spacer_file.split('.')[0]
            Ap_spacer = Atoms(symbols=spacer_elements, positions=spacer_positions)
            
            # Process each layer thickness
            for n in n_layers:
                # Create folder structure: MAPbX3_spacer-name_n#/
                folder_name = f"{name}_{spacer_name}_n{n}"
                structure_dir = f"CamiloITM/{folder_name}"
                os.makedirs(structure_dir, exist_ok=True)
                
                                # Create DJ structure
                dj_structure = make_dj(
                    Ap_spacer=Ap_spacer,      # A'-site spacer (surface capping)
                    A_site_cation=A_cation,   # A-site cation (between layers)
                    B_site_cation=B_cation,
                    X_site_anion=X_anion,
                    n=n,
                    BX_dist=bx_dist,
                    penet=penetration_fraction,
                    attachment_end='top',  # For DJ structures, use only one side
                    wrap=True,  # Enable wrapping to prevent molecules from disappearing
                    output=False,  # We'll save manually as POSCAR
                    output_type='vasp',
                    file_name=None
                )
                
                # Group atoms by element type for proper POSCAR format
                
                # Get all atoms and group by element
                symbols = dj_structure.get_chemical_symbols()
                positions = dj_structure.get_positions()
                cell = dj_structure.cell
                
                # Group by element type
                element_groups = {}
                for i, symbol in enumerate(symbols):
                    if symbol not in element_groups:
                        element_groups[symbol] = []
                    element_groups[symbol].append(positions[i])
                
                # Create ordered lists
                ordered_symbols = []
                ordered_positions = []
                element_counts = []
                
                # Standard order: inorganic first, then organic
                element_order = ['Br', 'Cl', 'I', 'Pb', 'Sn', 'C', 'H', 'N', 'O', 'S', 'P']
                
                for element in element_order:
                    if element in element_groups:
                        ordered_symbols.extend([element] * len(element_groups[element]))
                        ordered_positions.extend(element_groups[element])
                        element_counts.append(len(element_groups[element]))
                
                # Add any remaining elements not in the standard order
                for element in element_groups:
                    if element not in element_order:
                        ordered_symbols.extend([element] * len(element_groups[element]))
                        ordered_positions.extend(element_groups[element])
                        element_counts.append(len(element_groups[element]))
                
                # Create new structure with grouped atoms
                grouped_structure = Atoms(symbols=ordered_symbols, positions=ordered_positions, cell=cell)
                
                # Save only as POSCAR in the folder
                poscar_path = f"{structure_dir}/POSCAR"
                write(poscar_path, grouped_structure, format='vasp')

                # Debug: Check molecule positions and cell dimensions
                cell = dj_structure.cell.cellpar()
                positions = dj_structure.get_positions()
                min_pos = np.min(positions, axis=0)
                max_pos = np.max(positions, axis=0)
                
                print(f"    n={n}: {len(dj_structure)} atoms -> {folder_name}/POSCAR")
                print(f"      Cell: a={cell[0]:.2f}, b={cell[1]:.2f}, c={cell[2]:.2f} Å")
                print(f"      Position range: X[{min_pos[0]:.2f}, {max_pos[0]:.2f}], Y[{min_pos[1]:.2f}, {max_pos[1]:.2f}], Z[{min_pos[2]:.2f}, {max_pos[2]:.2f}]")
        
        print("Done.")

if __name__ == "__main__":
    main()
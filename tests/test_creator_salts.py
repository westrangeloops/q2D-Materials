import sys
from pathlib import Path
import numpy as np
# Add parent directory to path so we can import q2D_Materials
sys.path.insert(0, str(Path(__file__).parent.parent))

from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.core.structure import q2DStructure
from q2D_Materials.builders.spacer import calculate_double_spacer_nh3_distances as nh3_distance
from q2D_Materials.builders.spacer import calculate_molecule_radius as molecule_radius
from q2D_Materials.builders.optimizers import elongate_molecule
from q2D_Materials.builders.molecule_builder import smiles_to_ase_atoms
from ase.io import write, read

molecules_list = ['CC(C)(CC[NH3+])C(C)(C)CC[NH3+]', '[NH3+]CCCCCC[NH3+]', 'CC(C)(CCC[NH3+])CCC[NH3+]', 'CC(C)(CCC[NH3+])C(C)(C)CCC[NH3+]', '[NH3+]C1CCC([NH3+])CC1', '[NH3+]CCCCCCCCC[NH3+]', '[NH3+]CCCC[NH3+]', 'CC(C)(CC[NH3+])CC[NH3+]', '[NH3+]CC1=CC=C(C[NH3+])C=C1', '[NH3+]CCCCC[NH3+]', '[NH3+]CCCCCCCCCC[NH3+]', '[NH3+]CCC1CCC(CC[NH3+])CC1', '[NH3+]CCC1=CC=C(CC[NH3+])C=C1', 'CC(C)(C[NH3+])C(C)(C)C[NH3+]', '[NH3+]CC1CCC(C[NH3+])CC1', 'CC(C)(C[NH3+])C[NH3+]', 'CC(C)(CCCC[NH3+])C(C)(C)CCCC[NH3+]', '[NH3+]CCCC1CCC(CCC[NH3+])CC1', '[NH3+]CCCC1=CC=C(CCC[NH3+])C=C1', '[NH3+]C1=CC=C([NH3+])C=C1', '[NH3+]CCCCCCCC[NH3+]', 'CC(C)(CCCC[NH3+])CCCC[NH3+]', '[NH3+]CCCCCCC[NH3+]', '[NH3+]CCC[NH3+]']

q2d = q2D_creator()

molecules_list = molecules_list
# For each molecule in the list, create a structure and write it to a file.
for molecule in molecules_list:
    for halogen in ['Cl', 'Br', 'I']:
        elongated_molecule = elongate_molecule(molecule, step_size=0.01, max_iterations=1000)

        # Calculate the N - N distance:
        N_N_distance = nh3_distance(elongated_molecule)
        print(f"N - N distance: {N_N_distance} Å")

        # Calculate the molecule diameter:
        molecule_diameter = molecule_radius(elongated_molecule) * 2
        print(f"Molecule diameter: {molecule_diameter} Å")

        # The square we need is horizontal is 2 N - N distance + padding of 6 amstrongs.
        h = 2 * N_N_distance + 6
        print(f"h: {h} Å")

        # So the square we need is h^2 = A^2 + B^2, but A = B, so h^2 = 2A^2, so A = h/sqrt(2)
        denominator = 1
        if "CC(C)" in molecule:
            denominator = 0.5
        if "C1C" in molecule or "C1=C" in molecule:   
            denominator = 1.5
        else:
            denominator = 2.0

        A = (h / np.sqrt(2)) + (molecule_diameter / denominator)
        print(f"A: {A} Å")

        atoms = q2d.create_structure(
            template="salts",
            X_ions=halogen,
            spacer=elongated_molecule,
            lattice_multipliers=[A/3, A/3],
            layer_sequence=f"L1-({molecule_diameter})-L2-(0.0)",
            structure_type="bulk",
            optimizer="UFF", # Molecule is already elongated
            vacuum=0.0,
            
        )
        
        path = Path("../SALTS")
        path.mkdir(parents=True, exist_ok=True)
        write(path / f"{molecule}_{halogen}.vasp", atoms, format="vasp", sort=True)

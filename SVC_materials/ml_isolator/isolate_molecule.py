"""
Optimized script to process molecules using a supercell-based approach.
1. create_supercell: For each input crystal, create a 2x2x2 supercell to
   ensure molecules are not split across periodic boundaries.
2. extract_with_template: Finds all molecular fragments in the supercell and
   selects the first one that passes all validation checks (size, formula, N-N path).
3. XYZ conversion: Convert the extracted molecule to XYZ format.

This version saves all intermediate files for debugging purposes.
"""

import os
import shutil
from pathlib import Path
import numpy as np

# ASE is the primary library for atomic structure manipulation.
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import build_neighbor_list

# NetworkX is used for graph analysis.
import networkx as nx


# --- Explicit Bonding Cutoffs ---
# These base values are used to determine connectivity.
BOND_CUTOFFS = {
    frozenset(['C', 'H']): 1.3,
    frozenset(['C', 'C']): 1.7,
    frozenset(['C', 'N']): 1.6,
    frozenset(['N', 'H']): 1.2,
    frozenset(['O', 'H']): 1.2,
}


def get_connectivity_graph(atoms: Atoms) -> nx.Graph:
    """
    Builds a NetworkX graph from an ASE Atoms object based on bond cutoffs.
    The nodes are annotated with the element symbol.
    """
    g = nx.Graph()
    symbols = atoms.get_chemical_symbols()
    for i, symbol in enumerate(symbols):
        g.add_node(i, element=symbol)

    max_cutoff = max(BOND_CUTOFFS.values()) if BOND_CUTOFFS else 0
    # For non-periodic structures (like a wrapped fragment), get_cell() will be zero, so PBC is handled.
    nl = build_neighbor_list(atoms, [max_cutoff/2.0] * len(atoms), self_interaction=False)

    for i in range(len(atoms)):
        neighbors, offsets = nl.get_neighbors(i)
        for j, offset in zip(neighbors, offsets):
            if i >= j:
                continue

            bond_type = frozenset([symbols[i], symbols[j]])
            cutoff = BOND_CUTOFFS.get(bond_type)

            if cutoff is not None:
                distance = np.linalg.norm(atoms.positions[i] - (atoms.positions[j] + offset @ atoms.get_cell()))
                if distance <= cutoff:
                    g.add_edge(i, j)
    return g


def extract_with_template(target_supercell: Atoms, template_dir: Path, smiles_name: str) -> Atoms:
    """
    Finds and extracts a molecule from a supercell by finding all molecular
    fragments and returning the first one that passes all validation checks.

    Args:
        target_supercell: The supercell crystal structure.
        template_dir: The base directory containing folders of template calculations.
        smiles_name: The SMILES-like string that identifies the template folder.

    Returns:
        An ASE Atoms object of the isolated molecule, or None if no match is found.
    """
    # 1. Load the template molecule to get the target chemical formula.
    try:
        template_folder = template_dir / smiles_name
        template_file = template_folder / 'CONTCAR'
        if not template_file.exists():
            print(f"  - DEBUG: Template file not found at {template_file}")
            return None
        
        template_atoms = read(template_file)
        template_formula = template_atoms.get_chemical_formula()
        template_size = len(template_atoms)
        print(f"  - DEBUG: Loaded template {template_file.name}")
        print(f"  - DEBUG: Target formula: {template_formula}")
        print(f"  - DEBUG: Target size: {template_size} atoms")

    except Exception as e:
        print(f"  - ERROR: Could not read template for {smiles_name}: {e}")
        return None

    # 2. Build the connectivity graph for the entire supercell.
    target_graph = get_connectivity_graph(target_supercell)
    
    # 3. Find all connected components (i.e., all molecules and fragments).
    all_fragments_indices = list(nx.connected_components(target_graph))

    # 4. Find the first fragment that passes all validation checks
    for fragment_indices in all_fragments_indices:
        fragment_size = len(fragment_indices)
        
        # Check 1: Exact size match with template
        if fragment_size != template_size:
            print(f"  - DEBUG: Fragment size {fragment_size} does not match template size {template_size}")
            continue

        # Create fragment and check its properties
        fragment = target_supercell[list(fragment_indices)]
        
        # Check 2: Formula match
        if fragment.get_chemical_formula() != template_formula:
            print(f"  - DEBUG: Fragment formula {fragment.get_chemical_formula()} does not match template {template_formula}")
            continue
        
        # Check 3: Connectivity validation
        fragment_graph = get_connectivity_graph(fragment)
        
        # Verify the fragment is a single connected component
        if nx.number_connected_components(fragment_graph) != 1:
            print("  - DEBUG: Fragment is not fully connected")
            continue
            
        # Check 4: N-N Path validation
        n_indices_in_fragment = [a.index for a in fragment if a.symbol == 'N']
        
        if len(n_indices_in_fragment) != 2:
            print(f"  - DEBUG: Fragment has {len(n_indices_in_fragment)} nitrogens, expected 2")
            continue
            
        n1_frag_idx, n2_frag_idx = n_indices_in_fragment
        
        # Verify path exists between nitrogens
        if not nx.has_path(fragment_graph, source=n1_frag_idx, target=n2_frag_idx):
            print("  - DEBUG: No path between nitrogen atoms")
            continue
            
        # Check 5: Verify all atoms are connected to at least one other atom
        isolated_atoms = [node for node in fragment_graph.nodes() if fragment_graph.degree(node) == 0]
        if isolated_atoms:
            print(f"  - DEBUG: Found {len(isolated_atoms)} isolated atoms")
            continue
            
        # Check 6: Verify reasonable bond lengths
        has_invalid_bonds = False
        for i, j in fragment_graph.edges():
            pos_i = fragment.positions[i]
            pos_j = fragment.positions[j]
            bond_length = np.linalg.norm(pos_i - pos_j)
            
            # Get the expected bond length from BOND_CUTOFFS
            symbols = fragment.get_chemical_symbols()
            bond_type = frozenset([symbols[i], symbols[j]])
            expected_length = BOND_CUTOFFS.get(bond_type)
            
            if expected_length and bond_length > expected_length * 1.2:  # 20% tolerance
                print(f"  - DEBUG: Invalid bond length {bond_length:.2f} Å between {symbols[i]}-{symbols[j]}")
                has_invalid_bonds = True
                break
                
        if has_invalid_bonds:
            continue

        # All checks passed
        print(f"  - SUCCESS: Found valid fragment with exact size match ({fragment_size} atoms)")
        fragment.wrap()  # Ensure the final molecule is contiguous
        return fragment
    
    print("  - FAILURE: No valid fragment in the supercell passed all checks.")
    return None


def create_supercells(input_files: list, output_dir: str):
    """
    Step 1: Create a 2x2x2 supercell for each input structure.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    for molecule_file in input_files:
        print(f"\n--- Creating supercell for: {molecule_file.name} ---")
        try:
            atoms = read(molecule_file)
            # Create a 2x2x2 supercell to ensure at least one molecule is fully intact.
            supercell = atoms.repeat((2, 2, 2))
            supercell.center()
            
            output_file = output_path / f"{molecule_file.stem}_supercell.vasp"
            write(output_file, supercell, format='vasp')
            processed_files.append(output_file)
            
        except Exception as e:
            print(f"FATAL ERROR while creating supercell for {molecule_file.name}: {e}")
            
    return processed_files


def extract_molecules_with_templates(input_files: list, output_dir: str, template_dir: Path):
    """
    Step 2: Isolates the molecule from each supercell file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    for supercell_file in input_files:
        print(f"\n--- Extracting from: {supercell_file.name} ---")
        try:
            atoms = read(supercell_file)
            
            # Robustly parse the SMILES part of the filename.
            base_name = supercell_file.stem.replace('_supercell', '')
            try:
                smiles_name = '_'.join(base_name.split('_')[2:])
            except IndexError:
                print(f"  - WARNING: Could not parse SMILES name from '{base_name}'")
                continue

            isolated_molecule = extract_with_template(atoms, template_dir, smiles_name)
            
            if isolated_molecule is None:
                continue
            
            output_file = output_path / f"{base_name}_extracted.vasp"
            write(output_file, isolated_molecule, format='vasp')
            processed_files.append(output_file)
            
        except Exception as e:
            print(f"FATAL ERROR while processing {supercell_file.name}: {e}")
    
    return processed_files


def convert_to_xyz(input_files: list, output_dir: str):
    """Step 3: Convert final VASP files to XYZ format."""
    output_path = Path(output_dir) / 'xyz'
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    for molecule_file in input_files:
        try:
            atoms = read(molecule_file)
            base_name = molecule_file.stem.replace('_extracted', '')
            parts = base_name.split('_', 1)
            compound, molecule_name = parts if len(parts) == 2 else ("UNKNOWN", base_name)
            
            atoms.set_pbc(False)
            atoms.set_cell(None)
            atoms.center()
            
            xyz_path = output_path / f"{compound}_{molecule_name}.xyz"
            write(xyz_path, atoms, format='xyz')
            processed_files.append(xyz_path)
            
        except Exception as e:
            print(f"Error in convert_to_xyz for {molecule_file.name}: {e}")
    
    return processed_files


def process_all_files(input_dir: str, output_dir: str, template_dir: str):
    """Process all files through the optimized supercell pipeline."""
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    template_path = Path(template_dir)
    
    intermediates_dir = base_path / 'intermediates'
    
    input_files = sorted(Path(input_dir).glob('*.vasp'))
    print(f"--- Starting Processing ---")
    print(f"Found {len(input_files)} input files in '{input_dir}'")
    
    # Step 1: Create supercells from original files.
    supercell_dir = intermediates_dir / '1_supercells'
    supercell_files = create_supercells(input_files, str(supercell_dir))
    
    # Step 2: Use the template-based extraction on the supercells.
    extracted_dir = intermediates_dir / '2_extracted_final'
    extracted_files = extract_molecules_with_templates(supercell_files, str(extracted_dir), template_path)
    
    # Step 3: Convert the final, verified molecules to XYZ.
    xyz_files = convert_to_xyz(extracted_files, str(base_path))
    
    print(f"\n--- Final Summary ---")
    print(f"Initial input files: {len(input_files)}")
    print(f"Final output XYZ files: {len(xyz_files)}")
    print(f"\nIntermediate VASP files saved in: {intermediates_dir.resolve()}")


def isolate_molecule_from_crystal(
    crystal_file: str | Path,
    template_file: str | Path,
    output_file: str | Path,
    debug: bool = False
) -> bool:
    """
    Isolates a single molecule from a crystal structure using a template molecule.
    
    Args:
        crystal_file: Path to the input crystal structure (VASP or XYZ format)
        template_file: Path to the template molecule (VASP or XYZ format)
        output_file: Path where the isolated molecule will be saved (XYZ format)
        debug: If True, prints detailed debug information
        
    Returns:
        bool: True if molecule was successfully isolated and saved, False otherwise
        
    Example:
        >>> success = isolate_molecule_from_crystal(
        ...     "crystal.vasp",
        ...     "template/CONTCAR",
        ...     "isolated_molecule.xyz"
        ... )
    """
    try:
        # Convert paths to Path objects
        crystal_file = Path(crystal_file)
        template_file = Path(template_file)
        output_file = Path(output_file)
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load and validate template
        if debug:
            print(f"Loading template from: {template_file}")
        template_atoms = read(template_file)
        template_formula = template_atoms.get_chemical_formula()
        template_size = len(template_atoms)
        
        if debug:
            print(f"Template formula: {template_formula}")
            print(f"Template size: {template_size} atoms")
        
        # Load crystal structure
        if debug:
            print(f"Loading crystal from: {crystal_file}")
        crystal_atoms = read(crystal_file)
        
        # Create supercell
        supercell = crystal_atoms.repeat((2, 2, 2))
        supercell.center()
        
        if debug:
            print(f"Created 2x2x2 supercell with {len(supercell)} atoms")
        
        # Build connectivity graph
        target_graph = get_connectivity_graph(supercell)
        all_fragments_indices = list(nx.connected_components(target_graph))
        
        if debug:
            print(f"Found {len(all_fragments_indices)} potential fragments")
        
        # Find valid fragment
        for fragment_indices in all_fragments_indices:
            fragment_size = len(fragment_indices)
            
            # Size check
            if fragment_size != template_size:
                if debug:
                    print(f"Fragment size {fragment_size} != template size {template_size}")
                continue
            
            # Create and validate fragment
            fragment = supercell[list(fragment_indices)]
            
            # Formula check
            if fragment.get_chemical_formula() != template_formula:
                if debug:
                    print(f"Fragment formula {fragment.get_chemical_formula()} != template {template_formula}")
                continue
            
            # Connectivity check
            fragment_graph = get_connectivity_graph(fragment)
            if nx.number_connected_components(fragment_graph) != 1:
                if debug:
                    print("Fragment is not fully connected")
                continue
            
            # N-N path check
            n_indices = [a.index for a in fragment if a.symbol == 'N']
            if len(n_indices) != 2:
                if debug:
                    print(f"Found {len(n_indices)} N atoms, expected 2")
                continue
            
            if not nx.has_path(fragment_graph, source=n_indices[0], target=n_indices[1]):
                if debug:
                    print("No path between N atoms")
                continue
            
            # Bond length check
            has_invalid_bonds = False
            for i, j in fragment_graph.edges():
                pos_i = fragment.positions[i]
                pos_j = fragment.positions[j]
                bond_length = np.linalg.norm(pos_i - pos_j)
                
                symbols = fragment.get_chemical_symbols()
                bond_type = frozenset([symbols[i], symbols[j]])
                expected_length = BOND_CUTOFFS.get(bond_type)
                
                if expected_length and bond_length > expected_length * 1.2:
                    if debug:
                        print(f"Invalid bond length {bond_length:.2f} Å between {symbols[i]}-{symbols[j]}")
                    has_invalid_bonds = True
                    break
            
            if has_invalid_bonds:
                continue
            
            # All checks passed - save the molecule
            fragment.wrap()
            fragment.set_pbc(False)
            fragment.set_cell(None)
            fragment.center()
            
            write(output_file, fragment, format='xyz')
            
            if debug:
                print(f"Successfully isolated molecule and saved to: {output_file}")
            
            return True
        
        if debug:
            print("No valid fragment found in the crystal")
        return False
        
    except Exception as e:
        if debug:
            print(f"Error during molecule isolation: {str(e)}")
        return False


if __name__ == "__main__":
    # --- USER-DEFINED PATHS ---
    INPUT_DIR = "SVC_materials/ml_isolator/data/original"
    OUTPUT_DIR = "SVC_materials/ml_isolator/data/molecules"
    TEMPLATE_DIR = "~/Documents/DION-JACOBSON/MOLECULES/"
    
    # --- SCRIPT EXECUTION ---
    resolved_template_dir = str(Path(TEMPLATE_DIR).expanduser())

    if not Path(INPUT_DIR).exists():
        print(f"Error: Input directory not found at '{INPUT_DIR}'")
    elif not Path(resolved_template_dir).exists():
        print(f"Error: Template directory not found at '{resolved_template_dir}'")
    else:
        process_all_files(
            input_dir=INPUT_DIR, 
            output_dir=OUTPUT_DIR, 
            template_dir=resolved_template_dir
        )

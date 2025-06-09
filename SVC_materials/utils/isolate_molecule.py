"""
Utility module for isolating molecules from crystal structures.
This module provides functions to extract individual molecules from crystal structures
using template-based validation and connectivity analysis.
"""

import os
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import build_neighbor_list
import networkx as nx

# --- Bond Cutoffs for Connectivity Analysis ---
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
    
    Args:
        atoms: ASE Atoms object representing the structure
        
    Returns:
        NetworkX graph where nodes are atoms and edges are bonds
    """
    g = nx.Graph()
    symbols = atoms.get_chemical_symbols()
    for i, symbol in enumerate(symbols):
        g.add_node(i, element=symbol)

    max_cutoff = max(BOND_CUTOFFS.values()) if BOND_CUTOFFS else 0
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

def isolate_molecule(
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
        >>> success = isolate_molecule(
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

def isolate_molecule_from_atoms(
    crystal_atoms: Atoms,
    template_atoms: Atoms,
    debug: bool = False
) -> Atoms | None:
    """
    Isolates a single molecule from an ASE Atoms object using a template molecule.
    
    Args:
        crystal_atoms: ASE Atoms object representing the crystal structure
        template_atoms: ASE Atoms object representing the template molecule
        debug: If True, prints detailed debug information
        
    Returns:
        Atoms | None: The isolated molecule as an ASE Atoms object, or None if isolation failed
    """
    try:
        template_formula = template_atoms.get_chemical_formula()
        template_size = len(template_atoms)
        
        if debug:
            print(f"Template formula: {template_formula}")
            print(f"Template size: {template_size} atoms")
        
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
            
            # All checks passed - prepare the molecule
            fragment.wrap()
            fragment.set_pbc(False)
            fragment.set_cell(None)
            fragment.center()
            
            if debug:
                print("Successfully isolated molecule")
            
            return fragment
        
        if debug:
            print("No valid fragment found in the crystal")
        return None
        
    except Exception as e:
        if debug:
            print(f"Error during molecule isolation: {str(e)}")
        return None

def process_molecules(
    input_dir: str | Path,
    output_dir: str | Path,
    template_dir: str | Path,
    debug: bool = False
) -> dict:
    """
    High-level interface to process all molecules in a directory.
    
    Args:
        input_dir: Directory containing input VASP files
        output_dir: Directory to save isolated molecules
        template_dir: Directory containing template molecules
        debug: Whether to print detailed debug information
    
    Returns:
        dict: Processing statistics including:
            - total_files: Total number of input files
            - successful: Number of successfully processed files
            - failed: Number of failed files
            - output_dir: Path to output directory
    
    Example:
        >>> stats = process_molecules(
        ...     input_dir="data/original",
        ...     output_dir="data/molecules",
        ...     template_dir="~/templates",
        ...     debug=True
        ... )
        >>> print(f"Processed {stats['successful']}/{stats['total_files']} files")
    """
    # Convert paths to Path objects
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    template_dir = Path(template_dir).expanduser()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of input files
    input_files = sorted(input_dir.glob('*.vasp'))
    total_files = len(input_files)
    
    if debug:
        print(f"Found {total_files} input files in '{input_dir}'")
    
    # Process each file
    successful = 0
    failed = 0
    
    for input_file in input_files:
        if debug:
            print(f"\nProcessing: {input_file.name}")
        
        # Parse SMILES name from filename
        base_name = input_file.stem
        try:
            smiles_name = '_'.join(base_name.split('_')[2:])
        except IndexError:
            if debug:
                print(f"  - WARNING: Could not parse SMILES name from '{base_name}'")
            failed += 1
            continue
        
        # Construct paths
        template_file = template_dir / smiles_name / 'CONTCAR'
        output_file = output_dir / f"{base_name}_extracted.xyz"
        
        if debug:
            print(f"  Template: {template_file}")
            print(f"  Output: {output_file}")
        
        # Try to isolate molecule
        success = isolate_molecule(
            crystal_file=input_file,
            template_file=template_file,
            output_file=output_file,
            debug=debug
        )
        
        if success:
            successful += 1
            if debug:
                print(f"  - SUCCESS: Saved to {output_file}")
        else:
            failed += 1
            if debug:
                print(f"  - FAILURE: Could not isolate molecule")
    
    # Return statistics
    return {
        'total_files': total_files,
        'successful': successful,
        'failed': failed,
        'output_dir': output_dir.resolve()
    }

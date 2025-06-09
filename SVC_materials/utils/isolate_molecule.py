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
from networkx.algorithms.isomorphism import GraphMatcher
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union

# --- Bond Cutoffs for Connectivity Analysis ---
BOND_CUTOFFS = {
    frozenset(['C', 'H']): 1.3,
    frozenset(['C', 'C']): 1.7,
    frozenset(['C', 'N']): 1.6,
    frozenset(['N', 'H']): 1.2,
    frozenset(['O', 'H']): 1.2,
}

def classify_deformation(bond_mae: float, angle_mae: float, dihedral_mae: float) -> List[str]:
    """
    Classify the deformation based on the provided metrics.
    
    Args:
        bond_mae: Mean absolute error in bond lengths
        angle_mae: Mean absolute error in bond angles
        dihedral_mae: Mean absolute error in dihedral angles
        
    Returns:
        List of classifications for bond lengths, angles, and dihedrals
    """
    classifications = []
    
    # Bond length classification
    if bond_mae < 0.05:
        classifications.append("Normal bond fluctuations")
    elif bond_mae < 0.2:
        classifications.append("Significant bond strain")
    else:
        classifications.append("Severe bond strain")
    
    # Bond angle classification
    if angle_mae < 5:
        classifications.append("Normal angle flexibility")
    elif angle_mae < 15:
        classifications.append("Moderate angle strain")
    else:
        classifications.append("Severe angle strain")
    
    # Dihedral angle classification
    if dihedral_mae < 30:
        classifications.append("Local torsional oscillation")
    elif dihedral_mae < 90:
        classifications.append("Moderate conformational change")
    else:
        classifications.append("Major conformational change")
    
    return classifications

def analyze_deformation_details(metrics: Dict) -> Dict:
    """
    Analyze detailed deformation metrics and return summary statistics.
    
    Args:
        metrics: Dictionary containing deformation metrics
        
    Returns:
        Dictionary containing statistical analysis of deformations
    """
    bond_errors = [abs(ideal - extracted) for _, ideal, extracted in metrics['bond_details']]
    angle_errors = [abs(ideal - extracted) for _, ideal, extracted in metrics['angle_details']]
    dihedral_errors = [abs(ideal - extracted) for _, ideal, extracted in metrics['dihedral_details']]
    
    # Calculate statistics
    bond_stats = {
        'mean': np.mean(bond_errors),
        'std': np.std(bond_errors),
        'max': np.max(bond_errors),
        'min': np.min(bond_errors)
    }
    
    angle_stats = {
        'mean': np.mean(angle_errors),
        'std': np.std(angle_errors),
        'max': np.max(angle_errors),
        'min': np.min(angle_errors)
    }
    
    dihedral_stats = {
        'mean': np.mean(dihedral_errors),
        'std': np.std(dihedral_errors),
        'max': np.max(dihedral_errors),
        'min': np.min(dihedral_errors)
    }
    
    return {
        'bonds': bond_stats,
        'angles': angle_stats,
        'dihedrals': dihedral_stats
    }

def save_analysis_results(
    metrics: Dict,
    stats: Dict,
    classifications: List[str],
    output_dir: Union[str, Path],
    molecule_name: str
) -> None:
    """
    Save analysis results to CSV files and log file.
    
    Args:
        metrics: Dictionary containing deformation metrics
        stats: Dictionary containing statistical analysis
        classifications: List of deformation classifications
        output_dir: Directory to save results
        molecule_name: Name of the molecule being analyzed
    """
    output_dir = Path(output_dir)
    molecule_dir = output_dir / molecule_name
    molecule_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = molecule_dir / "analysis.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Log the analysis
    logger.info("\nDeformation Analysis Summary:")
    logger.info("---------------------------")
    logger.info(f"Chemical Formula: {metrics['chemical_formula']}")
    logger.info(f"Isomorphic: {metrics['is_isomorphic']}")
    
    logger.info("\nBond Length Analysis:")
    logger.info(f"  Mean Error: {stats['bonds']['mean']:.4f} ± {stats['bonds']['std']:.4f} Å")
    logger.info(f"  Range: [{stats['bonds']['min']:.4f}, {stats['bonds']['max']:.4f}] Å")
    logger.info(f"  Classification: {classifications[0]}")
    
    logger.info("\nBond Angle Analysis:")
    logger.info(f"  Mean Error: {stats['angles']['mean']:.4f} ± {stats['angles']['std']:.4f}°")
    logger.info(f"  Range: [{stats['angles']['min']:.4f}, {stats['angles']['max']:.4f}]°")
    logger.info(f"  Classification: {classifications[1]}")
    
    logger.info("\nDihedral Angle Analysis:")
    logger.info(f"  Mean Error: {stats['dihedrals']['mean']:.4f} ± {stats['dihedrals']['std']:.4f}°")
    logger.info(f"  Range: [{stats['dihedrals']['min']:.4f}, {stats['dihedrals']['max']:.4f}]°")
    logger.info(f"  Classification: {classifications[2]}")
    
    # Save detailed CSV files
    if metrics['bond_details']:
        bond_df = pd.DataFrame([
            {
                'bond': bond_name,
                'ideal_length': ideal,
                'extracted_length': extracted,
                'error': abs(ideal - extracted)
            }
            for bond_name, ideal, extracted in metrics['bond_details']
        ])
        bond_df.to_csv(molecule_dir / "bonds.csv", index=False)
    
    if metrics['angle_details']:
        angle_df = pd.DataFrame([
            {
                'angle': angle_name,
                'ideal_angle': ideal,
                'extracted_angle': extracted,
                'error': abs(ideal - extracted)
            }
            for angle_name, ideal, extracted in metrics['angle_details']
        ])
        angle_df.to_csv(molecule_dir / "angles.csv", index=False)
    
    if metrics['dihedral_details']:
        dihedral_df = pd.DataFrame([
            {
                'dihedral': dihedral_name,
                'ideal_angle': ideal,
                'extracted_angle': extracted,
                'error': abs(ideal - extracted)
            }
            for dihedral_name, ideal, extracted in metrics['dihedral_details']
        ])
        dihedral_df.to_csv(molecule_dir / "dihedrals.csv", index=False)
    
    # Remove handler to avoid duplicate logs
    logger.removeHandler(file_handler)

def get_analysis_summary(metrics: Dict, stats: Dict, classifications: List[str]) -> Dict:
    """
    Generate a summary dictionary of the analysis results.
    
    Args:
        metrics: Dictionary containing deformation metrics
        stats: Dictionary containing statistical analysis
        classifications: List of deformation classifications
        
    Returns:
        Dictionary containing summary of analysis results
    """
    return {
        'bond_length_mae': metrics['bond_length_mae'],
        'bond_length_std': stats['bonds']['std'],
        'bond_angle_mae': metrics['bond_angle_mae'],
        'bond_angle_std': stats['angles']['std'],
        'dihedral_angle_mae': metrics['dihedral_angle_mae'],
        'dihedral_angle_std': stats['dihedrals']['std'],
        'is_isomorphic': metrics['is_isomorphic'],
        'chemical_formula': metrics['chemical_formula'],
        'bond_classification': classifications[0],
        'angle_classification': classifications[1],
        'dihedral_classification': classifications[2]
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

def _analyze_molecule_deformation_base(
    ideal_file: Union[str, Path],
    extracted_file: Union[str, Path],
    debug: bool = False,
    include_hydrogens: bool = False
) -> Dict:
    """
    Base function for analyzing molecule deformation.
    This is the internal implementation that does the actual analysis.
    
    Args:
        ideal_file: Path to the ideal template molecule (XYZ or VASP format)
        extracted_file: Path to the extracted molecule (XYZ or VASP format)
        debug: Whether to print detailed debug information
        include_hydrogens: Whether to include hydrogen atoms in the analysis
        
    Returns:
        Dictionary containing deformation metrics
    """
    try:
        # Load the molecules
        ideal_atoms = read(ideal_file)
        extracted_atoms = read(extracted_file)
        
        if debug:
            print(f"Comparing '{ideal_file}' (Ideal) vs. '{extracted_file}' (Extracted)")
        
        # Ensure the molecules are chemically identical
        if ideal_atoms.get_chemical_formula() != extracted_atoms.get_chemical_formula():
            error_msg = f"Molecules have different chemical formulas:\n  - Ideal: {ideal_atoms.get_chemical_formula()}\n  - Extracted: {extracted_atoms.get_chemical_formula()}"
            if debug:
                print(error_msg)
            return {
                'error': error_msg,
                'is_isomorphic': False,
                'chemical_formula': ideal_atoms.get_chemical_formula()
            }
        
        # Build connectivity graphs
        ideal_graph = get_connectivity_graph(ideal_atoms)
        extracted_graph = get_connectivity_graph(extracted_atoms)
        
        # Check if molecules are isomorphic
        node_match = lambda n1, n2: n1['element'] == n2['element']
        matcher = GraphMatcher(extracted_graph, ideal_graph, node_match=node_match)
        
        if not matcher.is_isomorphic():
            error_msg = "The two molecules are not isomorphic (i.e., their bonding is different)"
            if debug:
                print(error_msg)
            return {
                'error': error_msg,
                'is_isomorphic': False,
                'chemical_formula': ideal_atoms.get_chemical_formula()
            }
        
        # Get atom mapping
        raw_map = matcher.mapping
        atom_map = {v: k for k, v in raw_map.items()}
        
        def get_atom_name(idx, atoms):
            """Get formatted atom name (e.g., 'C1', 'N2', etc.)"""
            symbol = atoms.get_chemical_symbols()[idx]
            if not include_hydrogens and symbol == 'H':
                return None
            # Count how many atoms of this type come before this one
            count = sum(1 for i in range(idx) if atoms.get_chemical_symbols()[i] == symbol)
            return f"{symbol}{count + 1}"
        
        # 1. Bond Length Comparison
        bond_errors = []
        bond_details = []
        for i, j in ideal_graph.edges():
            # Skip bonds involving hydrogens if not including them
            if not include_hydrogens and ('H' in [ideal_atoms.get_chemical_symbols()[i], ideal_atoms.get_chemical_symbols()[j]]):
                continue
                
            ideal_dist = ideal_atoms.get_distance(i, j)
            extracted_dist = extracted_atoms.get_distance(atom_map[i], atom_map[j])
            error = abs(ideal_dist - extracted_dist)
            bond_errors.append(error)
            
            # Create bond name
            atom1 = get_atom_name(i, ideal_atoms)
            atom2 = get_atom_name(j, ideal_atoms)
            if atom1 and atom2:  # Only include if both atoms are valid
                bond_name = f"{atom1}-{atom2}"
                bond_details.append((bond_name, ideal_dist, extracted_dist))
        
        bond_length_mae = np.mean(bond_errors) if bond_errors else 0.0
        if debug:
            print(f"  - Bond Length MAE: {bond_length_mae:.4f} Å")
        
        # 2. Bond Angle Comparison
        angle_errors = []
        angle_details = []
        for i in ideal_graph.nodes():
            # Skip angles centered on hydrogen if not including them
            if not include_hydrogens and ideal_atoms.get_chemical_symbols()[i] == 'H':
                continue
                
            neighbors = list(ideal_graph.neighbors(i))
            if len(neighbors) >= 2:
                for j1_idx in range(len(neighbors)):
                    for j2_idx in range(j1_idx + 1, len(neighbors)):
                        j1, j2 = neighbors[j1_idx], neighbors[j2_idx]
                        
                        # Skip angles involving hydrogens if not including them
                        if not include_hydrogens and ('H' in [ideal_atoms.get_chemical_symbols()[j1], ideal_atoms.get_chemical_symbols()[j2]]):
                            continue
                            
                        ideal_angle = ideal_atoms.get_angle(j1, i, j2)
                        extracted_angle = extracted_atoms.get_angle(atom_map[j1], atom_map[i], atom_map[j2])
                        error = abs(ideal_angle - extracted_angle)
                        angle_errors.append(error)
                        
                        # Create angle name
                        atom1 = get_atom_name(j1, ideal_atoms)
                        atom2 = get_atom_name(i, ideal_atoms)
                        atom3 = get_atom_name(j2, ideal_atoms)
                        if atom1 and atom2 and atom3:  # Only include if all atoms are valid
                            angle_name = f"{atom1}-{atom2}-{atom3}"
                            angle_details.append((angle_name, ideal_angle, extracted_angle))
        
        bond_angle_mae = np.mean(angle_errors) if angle_errors else 0.0
        if debug:
            print(f"  - Bond Angle MAE: {bond_angle_mae:.4f} degrees")
        
        # 3. Dihedral Angle Comparison
        dihedral_errors = []
        dihedral_details = []
        for i, j in ideal_graph.edges():
            # Skip dihedrals involving hydrogens if not including them
            if not include_hydrogens and ('H' in [ideal_atoms.get_chemical_symbols()[i], ideal_atoms.get_chemical_symbols()[j]]):
                continue
                
            i_neighbors = [n for n in ideal_graph.neighbors(i) if n != j]
            j_neighbors = [n for n in ideal_graph.neighbors(j) if n != i]
            for k in i_neighbors:
                for l in j_neighbors:
                    # Skip dihedrals involving hydrogens if not including them
                    if not include_hydrogens and ('H' in [ideal_atoms.get_chemical_symbols()[k], ideal_atoms.get_chemical_symbols()[l]]):
                        continue
                        
                    ideal_dihedral = ideal_atoms.get_dihedral(k, i, j, l)
                    extracted_dihedral = extracted_atoms.get_dihedral(atom_map[k], atom_map[i], atom_map[j], atom_map[l])
                    error = abs(ideal_dihedral - extracted_dihedral)
                    error = min(error, 360 - error)  # Handle periodicity
                    dihedral_errors.append(error)
                    
                    # Create dihedral name
                    atom1 = get_atom_name(k, ideal_atoms)
                    atom2 = get_atom_name(i, ideal_atoms)
                    atom3 = get_atom_name(j, ideal_atoms)
                    atom4 = get_atom_name(l, ideal_atoms)
                    if atom1 and atom2 and atom3 and atom4:  # Only include if all atoms are valid
                        dihedral_name = f"{atom1}-{atom2}-{atom3}-{atom4}"
                        dihedral_details.append((dihedral_name, ideal_dihedral, extracted_dihedral))
        
        dihedral_angle_mae = np.mean(dihedral_errors) if dihedral_errors else 0.0
        if debug:
            print(f"  - Dihedral Angle MAE: {dihedral_angle_mae:.4f} degrees")
        
        return {
            'bond_length_mae': bond_length_mae,
            'bond_angle_mae': bond_angle_mae,
            'dihedral_angle_mae': dihedral_angle_mae,
            'is_isomorphic': True,
            'chemical_formula': ideal_atoms.get_chemical_formula(),
            'bond_details': bond_details,
            'angle_details': angle_details,
            'dihedral_details': dihedral_details
        }
        
    except Exception as e:
        error_msg = f"Error during comparison: {str(e)}"
        if debug:
            print(error_msg)
        return {
            'error': error_msg,
            'is_isomorphic': False,
            'chemical_formula': None
        }

def analyze_molecule_deformation(
    ideal_file: Union[str, Path],
    extracted_file: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    debug: bool = False,
    include_hydrogens: bool = False
) -> Dict:
    """
    Analyzes the structural deformation of an extracted molecule compared to its ideal template.
    
    Args:
        ideal_file: Path to the ideal template molecule (XYZ or VASP format)
        extracted_file: Path to the extracted molecule (XYZ or VASP format)
        output_dir: Directory to save analysis results (optional)
        debug: Whether to print detailed debug information
        include_hydrogens: Whether to include hydrogen atoms in the analysis
        
    Returns:
        Dictionary containing deformation metrics and analysis results
    """
    try:
        # Get base metrics
        metrics = _analyze_molecule_deformation_base(
            ideal_file=ideal_file,
            extracted_file=extracted_file,
            debug=debug,
            include_hydrogens=include_hydrogens
        )
        
        if 'error' in metrics:
            return metrics
            
        # Get detailed statistics
        stats = analyze_deformation_details(metrics)
        
        # Classify the deformation
        classifications = classify_deformation(
            metrics['bond_length_mae'],
            metrics['bond_angle_mae'],
            metrics['dihedral_angle_mae']
        )
        
        # Save results if output directory is provided
        if output_dir:
            molecule_name = Path(extracted_file).stem.replace('_extracted', '')
            save_analysis_results(metrics, stats, classifications, output_dir, molecule_name)
        
        # Add analysis results to metrics
        metrics.update({
            'statistics': stats,
            'classifications': classifications,
            'summary': get_analysis_summary(metrics, stats, classifications)
        })
        
        return metrics
        
    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}"
        if debug:
            print(error_msg)
        return {
            'error': error_msg,
            'is_isomorphic': False,
            'chemical_formula': None
        }

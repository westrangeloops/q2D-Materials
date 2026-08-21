"""Structure reconstruction utilities for molecular modification."""

import numpy as np
import networkx as nx
from typing import List, Set, Optional
from ase import Atoms

from q2D_Materials.utils.properties.atomic_properties import (
    calculate_ideal_bond_length,
    align_fragment_geometry_aware
)


def reconstruct_structure(
    original_structure: Atoms,
    old_molecule_indices: List[int],
    new_molecule_atoms: Atoms,
) -> Atoms:
    """Reconstruct full structure after molecule modification.

    Replaces atoms at specified indices with new molecule atoms while
    preserving the rest of the structure (inorganic framework and other molecules).

    Parameters
    ----------
    original_structure : Atoms
        The complete original structure
    old_molecule_indices : List[int]
        Original indices of atoms in the molecule being replaced
    new_molecule_atoms : Atoms
        The new modified molecule as an ASE Atoms object

    Returns
    -------
    Atoms
        Complete new structure with the molecule replaced
    """
    old_indices_set: Set[int] = set(old_molecule_indices)

    # Build arrays for non-molecule atoms (framework + other molecules)
    kept_symbols = []
    kept_positions = []
    for i in range(len(original_structure)):
        if i not in old_indices_set:
            kept_symbols.append(original_structure[i].symbol)
            kept_positions.append(original_structure[i].position.copy())

    # Add new molecule atoms
    new_symbols = new_molecule_atoms.get_chemical_symbols()
    new_positions = new_molecule_atoms.get_positions()

    all_symbols = kept_symbols + list(new_symbols)
    if len(kept_positions) > 0:
        all_positions = np.vstack([kept_positions, new_positions])
    else:
        all_positions = new_positions

    # Create new Atoms object preserving cell and pbc
    new_structure = Atoms(
        symbols=all_symbols,
        positions=all_positions,
        cell=original_structure.get_cell(),
        pbc=original_structure.get_pbc(),
    )

    return new_structure


def calculate_fragment_positions_geometry_aware(
    target_atom_idx: int,
    target_neighbors: List[int],
    molecule_graph: nx.Graph,
    molecule_atoms: Atoms,
    fragment_atoms: Atoms,
    fragment_attachment_index: int,
    fragment_graph: nx.Graph,
    parent_structure: Optional[Atoms] = None,
    fragment_attachment_node: Optional[int] = None,
) -> np.ndarray:
    """Calculate fragment positions considering bonding geometry.

    Uses the bonding geometry (sp, sp2, sp3) of the target atom to properly
    orient and position the fragment with correct bond lengths and angles.
    Now uses stored hybridization data from the graph for improved geometry.

    Parameters
    ----------
    target_atom_idx : int
        Original index of atom being replaced
    target_neighbors : List[int]
        Original indices of neighboring atoms (only atoms to keep bonded)
    molecule_graph : nx.Graph
        Molecular graph with original indices and hybridization info
    molecule_atoms : Atoms
        Original molecule atoms object
    fragment_atoms : Atoms
        Fragment to insert
    fragment_attachment_index : int
        Index of attachment atom in fragment
    fragment_graph : nx.Graph
        Fragment molecular graph

    Returns
    -------
    np.ndarray
        Geometry-aware positions for fragment atoms, shape (n_atoms, 3)
    """
    # Get original indices mapping
    original_indices = [node for node in molecule_graph.nodes() if node in molecule_graph.nodes()]

    # Validate target atom exists - no fallbacks, must work
    if target_atom_idx not in original_indices:
        raise ValueError(
            f"Target atom index {target_atom_idx} not found in molecule graph. "
            f"Available indices: {original_indices}"
        )

    # Get target atom info
    target_node_data = molecule_graph.nodes[target_atom_idx]
    target_symbol = target_node_data.get('symbol', 'C')
    target_position = target_node_data.get('position')
    target_hybridization = target_node_data.get('hybridization', 'unknown')

    # Validate target position exists - no fallbacks
    if target_position is None:
        # Try to get from molecule_atoms positions
        original_indices_list = sorted(molecule_graph.nodes())
        if target_atom_idx in original_indices_list:
            local_idx = original_indices_list.index(target_atom_idx)
            if local_idx < len(molecule_atoms):
                target_position = molecule_atoms.get_positions()[local_idx]
        
        if target_position is None:
            raise ValueError(
                f"Target atom position not found for index {target_atom_idx}. "
                f"Cannot determine replacement position."
            )
    
    # Must have neighbors for geometry calculation - no fallbacks
    if len(target_neighbors) == 0:
        raise ValueError(
            f"Target atom {target_atom_idx} has no neighbors. "
            f"Cannot determine geometry for fragment placement."
        )

    # Get neighbor positions and calculate bond vectors (PBC-aware)
    neighbor_positions = []
    neighbor_vectors = []
    
    # Get cell and PBC for PBC-aware calculations - try parent_structure first
    cell = None
    pbc = None
    if parent_structure is not None:
        if hasattr(parent_structure, 'cell') and parent_structure.cell is not None:
            cell = parent_structure.cell
            pbc = parent_structure.pbc if hasattr(parent_structure, 'pbc') else [True, True, True]
    elif hasattr(molecule_atoms, 'cell') and molecule_atoms.cell is not None:
        cell = molecule_atoms.cell
        pbc = molecule_atoms.pbc if hasattr(molecule_atoms, 'pbc') else [True, True, True]
    
    for neighbor_idx in target_neighbors:
        if neighbor_idx in molecule_graph.nodes:
            neighbor_data = molecule_graph.nodes[neighbor_idx]
            neighbor_pos = neighbor_data.get('position')
            if neighbor_pos is not None:
                neighbor_positions.append(neighbor_pos)
                # Vector from neighbor to target (PBC-aware)
                vec = target_position - neighbor_pos
                
                # Apply PBC if cell is provided
                if cell is not None and pbc is not None and any(pbc):
                    try:
                        inv_cell = np.linalg.inv(cell)
                        # Convert to fractional coordinates
                        vec_frac = vec @ inv_cell.T
                        # Apply PBC wrapping
                        for i in range(3):
                            if pbc[i]:
                                vec_frac[i] = vec_frac[i] - np.round(vec_frac[i])
                        # Convert back to Cartesian
                        vec = vec_frac @ cell
                    except np.linalg.LinAlgError:
                        pass  # Use unwrapped vector if cell is singular
                
                vec_norm = np.linalg.norm(vec)
                if vec_norm > 1e-6:
                    neighbor_vectors.append(vec / vec_norm)

    # Calculate bond direction using hybridization-aware logic
    if len(neighbor_vectors) == 0:
        # No valid neighbor directions, use default orientation
        bond_direction = np.array([0, 0, 1])
    elif target_hybridization == 'sp3' and len(neighbor_vectors) >= 3:
        # For sp3 with 3 neighbors, the 4th bond should complete tetrahedral geometry
        # Calculate the direction that maximizes tetrahedral angles (~109.5°)
        avg_direction = np.mean(neighbor_vectors, axis=0)
        norm = np.linalg.norm(avg_direction)
        if norm > 1e-6:
            bond_direction = avg_direction / norm
        else:
            bond_direction = np.array([0, 0, 1])
    elif target_hybridization == 'sp2' and len(neighbor_vectors) >= 2:
        # For sp2 with 2 neighbors, the 3rd bond should be coplanar at ~120°
        # Calculate perpendicular in the plane defined by the two neighbors
        v1, v2 = neighbor_vectors[0], neighbor_vectors[1]
        # Average direction points away from both
        avg_direction = (v1 + v2) / 2.0
        norm = np.linalg.norm(avg_direction)
        if norm > 1e-6:
            bond_direction = avg_direction / norm
        else:
            bond_direction = np.array([0, 0, 1])
    elif target_hybridization == 'sp' and len(neighbor_vectors) >= 1:
        # For sp with 1 neighbor, the 2nd bond should be linear (180°)
        bond_direction = neighbor_vectors[0]  # Same direction (180° from neighbor)
    else:
        # Default: average direction away from all neighbors
        avg_direction = np.mean(neighbor_vectors, axis=0)
        norm = np.linalg.norm(avg_direction)
        if norm > 1e-6:
            bond_direction = avg_direction / norm
        else:
            bond_direction = np.array([0, 0, 1])

    # Get fragment attachment atom symbol
    fragment_symbols = fragment_atoms.get_chemical_symbols()
    fragment_attach_symbol = fragment_symbols[fragment_attachment_index]

    # Calculate ideal bond length
    bond_length = calculate_ideal_bond_length(target_symbol, fragment_attach_symbol)

    # Use comprehensive geometry-aware alignment that considers:
    # 1. Fragment attachment atom's hybridization
    # 2. Target atom's hybridization
    # 3. Neighbors' hybridization (to preserve their geometry)
    # 4. Periodic boundary conditions
    fragment_positions = fragment_atoms.get_positions()
    
    # Get molecule atoms positions for neighbor lookup
    # Note: molecule_graph uses original indices, molecule_atoms uses local indices
    # We need to create a mapping
    molecule_positions = molecule_atoms.get_positions()
    original_indices_list = sorted(molecule_graph.nodes())
    original_to_local = {orig_idx: local_idx for local_idx, orig_idx in enumerate(original_indices_list)}
    
    # Get cell and PBC information - try parent_structure first, then molecule_atoms
    cell = None
    pbc = None
    if parent_structure is not None:
        if hasattr(parent_structure, 'cell') and parent_structure.cell is not None:
            cell = parent_structure.cell
            pbc = parent_structure.pbc if hasattr(parent_structure, 'pbc') else [True, True, True]
    elif hasattr(molecule_atoms, 'cell') and molecule_atoms.cell is not None:
        cell = molecule_atoms.cell
        pbc = molecule_atoms.pbc if hasattr(molecule_atoms, 'pbc') else [True, True, True]
    
    # Always use geometry-aware alignment (don't silently fall back)
    aligned_positions = align_fragment_geometry_aware(
            fragment_positions=fragment_positions,
            fragment_attachment_idx=fragment_attachment_index,
            fragment_graph=fragment_graph,
            fragment_atoms_symbols=fragment_symbols,
            target_position=target_position,
            target_atom_idx=target_atom_idx,
            target_neighbors=target_neighbors,
            molecule_graph=molecule_graph,
            molecule_atoms_positions=molecule_positions,
            original_to_local_map=original_to_local,
            bond_length=bond_length,
            cell=cell,
            pbc=pbc,
            fragment_attachment_node=fragment_attachment_node
        )

    return aligned_positions

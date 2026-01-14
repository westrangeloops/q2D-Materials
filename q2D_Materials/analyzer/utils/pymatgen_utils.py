"""Pymatgen integration utilities for molecular component detection.

This module provides functions to detect and extract molecular components
from crystal structures using custom bond detection based on covalent radii.

Uses covalent_radii.json for bond detection, supporting all element pairs.
"""

import logging
from typing import List, Optional, Set, Tuple

import numpy as np
import networkx as nx
from ase import Atoms

from ...utils.properties.atomic_properties import (
    determine_hybridization,
    are_atoms_bonded,
    get_covalent_radius,
)

logger = logging.getLogger(__name__)


def _detect_bonds_from_distances(
    symbols: List[str],
    positions: np.ndarray,
    tolerance: float = 0.45,
    cell: Optional[np.ndarray] = None,
) -> List[Tuple[int, int, float]]:
    """Detect covalent bonds from atomic positions using covalent radii.
    
    Enforces chemical constraints: Hydrogen atoms can only have one bond
    (the shortest/closest one).
    
    Parameters
    ----------
    symbols : list of str
        Atomic symbols
    positions : np.ndarray
        Atomic positions (N, 3)
    tolerance : float, default=0.45
        Additional tolerance beyond sum of covalent radii
    cell : np.ndarray, optional
        3x3 unit cell matrix for PBC-aware distance calculations
        
    Returns
    -------
    list of tuple (int, int, float)
        Each tuple contains (atom_i, atom_j, bond_length)
    """
    from ...utils.geometry.geometry import _calculate_distances
    
    bonds = []
    n_atoms = len(symbols)
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Skip H-H bonds
            if symbols[i] == 'H' and symbols[j] == 'H':
                continue
            
            # Use PBC-aware distance if cell is provided
            if cell is not None:
                distance = _calculate_distances(positions[i], positions[j:j+1], cell)[0]
            else:
                distance = np.linalg.norm(positions[i] - positions[j])
            
            if are_atoms_bonded(distance, symbols[i], symbols[j], tolerance):
                bonds.append((i, j, distance))
    
    # Post-process: Ensure H atoms only have one bond (keep shortest)
    h_bonds_by_atom = {}
    non_h_bonds = []
    
    for i, j, dist in bonds:
        if symbols[i] == 'H':
            if i not in h_bonds_by_atom:
                h_bonds_by_atom[i] = []
            h_bonds_by_atom[i].append((i, j, dist))
        elif symbols[j] == 'H':
            if j not in h_bonds_by_atom:
                h_bonds_by_atom[j] = []
            h_bonds_by_atom[j].append((i, j, dist))
        else:
            # Neither atom is H, keep as-is
            non_h_bonds.append((i, j, dist))
    
    # For each H atom, keep only the shortest bond
    valid_h_bonds = []
    for h_idx, h_bond_list in h_bonds_by_atom.items():
        if h_bond_list:
            # Sort by distance (shortest first)
            h_bond_list.sort(key=lambda x: x[2])
            # Keep only the shortest bond
            valid_h_bonds.append(h_bond_list[0])
    
    # Combine non-H bonds with validated H bonds
    final_bonds = non_h_bonds + valid_h_bonds
    
    return final_bonds


def extract_molecular_components(
    atoms: Atoms,
    exclude_indices: Set[int],
    organic_elements: Optional[Set[str]] = None,
) -> List[Tuple[Set[int], Atoms]]:
    """Extract molecular components using custom bond detection.

    Identifies separate molecules by finding connected components in the
    covalent bond graph. Uses covalent radii from covalent_radii.json
    to detect bonds, supporting all element pairs.

    Parameters
    ----------
    atoms : ase.Atoms
        The full crystal structure
    exclude_indices : set of int
        Indices of atoms to exclude (e.g., atoms in octahedra)
    organic_elements : set of str, optional
        Elements to consider as organic (default: C, N, H, O, S, P)

    Returns
    -------
    list of tuple (set of int, ase.Atoms)
        Each tuple contains:
        - Set of original atom indices in this molecule
        - ASE Atoms object for the molecule
    """
    if organic_elements is None:
        organic_elements = {'C', 'N', 'H', 'O', 'S', 'P'}

    organic_data = [
        (i, atoms[i].symbol, atoms[i].position)
        for i in range(len(atoms))
        if i not in exclude_indices and atoms[i].symbol in organic_elements
    ]

    if not organic_data:
        return []

    organic_indices, organic_symbols, organic_positions = zip(*organic_data)
    organic_indices = list(organic_indices)
    organic_symbols = list(organic_symbols)
    organic_positions = np.array(organic_positions)

    # Get cell for PBC-aware distance calculations
    cell = np.array(atoms.get_cell()) if atoms.cell is not None and np.any(atoms.pbc) else None

    # Detect bonds using covalent radii with PBC support
    bonds = _detect_bonds_from_distances(organic_symbols, organic_positions, cell=cell)
    
    # Build graph from bonds
    nx_graph = nx.Graph()
    mol_idx_to_crystal_idx = {i: orig_idx for i, orig_idx in enumerate(organic_indices)}
    
    # Add nodes
    for i in range(len(organic_indices)):
        nx_graph.add_node(i)
    
    # Add edges from detected bonds
    for i, j, _ in bonds:
        nx_graph.add_edge(i, j)

    molecules = []
    for component in nx.connected_components(nx_graph):
        crystal_indices = {mol_idx_to_crystal_idx[i] for i in component}
        mol_symbols = [atoms[i].symbol for i in crystal_indices]
        mol_positions = np.array([atoms[i].position for i in crystal_indices])

        mol_atoms = Atoms(symbols=mol_symbols, positions=mol_positions)
        mol_atoms.info['original_indices'] = list(crystal_indices)
        molecules.append((crystal_indices, mol_atoms))

    return molecules


def get_covalent_bonds(
    atoms: Atoms,
    atom_indices: Optional[Set[int]] = None,
) -> List[Tuple[int, int, float]]:
    """Get covalent bonds using custom bond detection with covalent radii.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure to analyze
    atom_indices : set of int, optional
        If provided, only consider bonds among these atoms

    Returns
    -------
    list of tuple (int, int, float)
        Each tuple contains (atom_i, atom_j, bond_length)
    """
    indices_to_use = list(atom_indices) if atom_indices is not None else list(range(len(atoms)))

    if not indices_to_use:
        return []

    mol_symbols = [atoms[i].symbol for i in indices_to_use]
    mol_positions = np.array([atoms[i].position for i in indices_to_use])
    
    # Get cell for PBC-aware distance calculations
    cell = np.array(atoms.get_cell()) if atoms.cell is not None and np.any(atoms.pbc) else None
    
    # Detect bonds using covalent radii with PBC support
    bonds = _detect_bonds_from_distances(mol_symbols, mol_positions, cell=cell)
    
    # Map back to original indices
    mol_idx_to_orig = {i: orig_idx for i, orig_idx in enumerate(indices_to_use)}
    result = []
    
    for mol_i, mol_j, bond_length in bonds:
        orig_i = mol_idx_to_orig[mol_i]
        orig_j = mol_idx_to_orig[mol_j]
        result.append((orig_i, orig_j, bond_length))

    return result


def build_molecular_graph(
    atoms: Atoms,
    exclude_indices: Set[int],
    original_indices: Optional[List[int]] = None,
) -> nx.Graph:
    """Build molecular connectivity graph using custom bond detection.

    Creates a graph where nodes are original atom indices and edges represent
    covalent bonds. Uses covalent radii from covalent_radii.json to detect
    bonds, supporting all element pairs.

    Each node contains:
    - 'symbol': atomic symbol
    - 'position': 3D position as numpy array
    - 'original_index': original atom index (same as node ID, for clarity)

    Parameters
    ----------
    atoms : ase.Atoms
        The structure (can be full structure or extracted molecule)
    exclude_indices : set of int
        Indices of atoms to exclude (e.g., atoms in octahedra).
        For extracted molecules, typically pass empty set.
    original_indices : list of int, optional
        If provided, maps local indices in 'atoms' to these original indices.
        Use when 'atoms' is an extracted molecule from a larger structure.
        If None, uses indices from 'atoms' directly (0, 1, 2, ...).

    Returns
    -------
    networkx.Graph
        Graph with original atom indices as nodes and covalent bonds as edges.
        Nodes use original indices, ensuring unique mapping for modifications.
    """
    included_indices = [i for i in range(len(atoms)) if i not in exclude_indices]

    if not included_indices:
        return nx.Graph()

    # Map local indices to original indices
    if original_indices is not None:
        if len(original_indices) != len(atoms):
            raise ValueError(
                f"original_indices length ({len(original_indices)}) must match "
                f"atoms length ({len(atoms)})"
            )
        local_to_original = {local_idx: orig_idx for local_idx, orig_idx in enumerate(original_indices)}
    else:
        # Use local indices as original indices
        local_to_original = {i: i for i in range(len(atoms))}

    mol_symbols = [atoms[i].symbol for i in included_indices]
    mol_positions = np.array([atoms[i].position for i in included_indices])
    
    # Get cell for PBC-aware distance calculations
    cell = np.array(atoms.get_cell()) if atoms.cell is not None and np.any(atoms.pbc) else None
    
    # Detect bonds using covalent radii with PBC support
    bonds = _detect_bonds_from_distances(mol_symbols, mol_positions, cell=cell)
    
    # Map molecule indices to local indices
    mol_idx_to_local = {i: local_idx for i, local_idx in enumerate(included_indices)}
    mol_graph = nx.Graph()

    # First pass: Add nodes with basic attributes
    for local_idx in included_indices:
        orig_idx = local_to_original[local_idx]
        mol_graph.add_node(
            orig_idx,
            symbol=atoms[local_idx].symbol,
            position=np.array(atoms[local_idx].position),
            original_index=orig_idx,  # Explicit mapping for clarity
        )

    # Add edges representing covalent bonds
    for mol_i, mol_j, bond_length in bonds:
        local_i = mol_idx_to_local[mol_i]
        local_j = mol_idx_to_local[mol_j]
        orig_i = local_to_original[local_i]
        orig_j = local_to_original[local_j]

        mol_graph.add_edge(
            orig_i,
            orig_j,
            bond_length=bond_length,
            edge_type='covalent_bond'
        )

    # Second pass: Add hybridization information based on connectivity
    for node in mol_graph.nodes():
        neighbors = list(mol_graph.neighbors(node))
        num_neighbors = len(neighbors)

        if num_neighbors > 0:
            try:
                hybridization, ideal_vectors = determine_hybridization(num_neighbors)
                mol_graph.nodes[node]['hybridization'] = hybridization
                mol_graph.nodes[node]['num_neighbors'] = num_neighbors
                mol_graph.nodes[node]['ideal_bond_vectors'] = ideal_vectors
            except ValueError:
                # Unsupported coordination number
                mol_graph.nodes[node]['hybridization'] = 'unknown'
                mol_graph.nodes[node]['num_neighbors'] = num_neighbors
                mol_graph.nodes[node]['ideal_bond_vectors'] = None
        else:
            mol_graph.nodes[node]['hybridization'] = 'isolated'
            mol_graph.nodes[node]['num_neighbors'] = 0
            mol_graph.nodes[node]['ideal_bond_vectors'] = None

    return mol_graph


def detect_hydrogen_bonds(
    atoms: Atoms,
    donor_indices: Set[int],
    acceptor_indices: Set[int],
    max_distance: float = 3.2,
    min_angle: float = 120.0,
) -> List[Tuple[Optional[int], int, int, float]]:
    """Detect hydrogen bonds between donor and acceptor atoms.

    Kept separate from covalent bond detection to prevent hydrogen bonds
    from merging separate molecules.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure to analyze
    donor_indices : set of int
        Indices of potential hydrogen bond donors (H atoms bonded to N, O)
    acceptor_indices : set of int
        Indices of potential acceptors (halides, O, N)
    max_distance : float
        Maximum H...acceptor distance in Angstroms (default: 3.2)
    min_angle : float
        Minimum D-H...A angle in degrees (default: 120, currently unused)

    Returns
    -------
    list of tuple (int or None, int, int, float)
        Each tuple contains (donor_heavy_atom, H_atom, acceptor_atom, distance)
    """
    from ...utils.geometry.geometry import _calculate_distances

    cell = np.array(atoms.get_cell())
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    h_bonds = []

    for h_idx in donor_indices:
        if symbols[h_idx] != 'H':
            continue

        h_pos = positions[h_idx]
        for acc_idx in acceptor_indices:
            dist = _calculate_distances(h_pos, [positions[acc_idx]], cell)[0]
            if dist <= max_distance:
                h_bonds.append((None, h_idx, acc_idx, dist))

    return h_bonds

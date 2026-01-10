"""Pymatgen integration utilities for molecular component detection.

This module provides functions to detect and extract molecular components
from crystal structures using pymatgen's CovalentBondNN strategy.

Note: CovalentBondNN is designed for non-periodic Molecule objects.
For periodic structures, we extract organic atoms first, create a
non-periodic Molecule, then use MoleculeGraph.
"""

import logging
from typing import List, Optional, Set, Tuple

import numpy as np
import networkx as nx
from ase import Atoms
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import CovalentBondNN
from pymatgen.core import Molecule

logger = logging.getLogger(__name__)


def extract_molecular_components(
    atoms: Atoms,
    exclude_indices: Set[int],
    organic_elements: Optional[Set[str]] = None,
) -> List[Tuple[Set[int], Atoms]]:
    """Extract molecular components using pymatgen's CovalentBondNN.

    Identifies separate molecules by finding connected components in the
    covalent bond graph. Uses proper bond chemistry to avoid merging
    molecules that are spatially close but not covalently bonded.

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
    organic_positions = list(organic_positions)

    mol = Molecule(organic_symbols, organic_positions)
    strategy = CovalentBondNN()

    try:
        mg = MoleculeGraph.from_local_env_strategy(mol, strategy)
    except Exception as e:
        logger.warning("Could not build molecule graph: %s", e)
        return []

    nx_graph = mg.graph.to_undirected()
    mol_idx_to_crystal_idx = {i: orig_idx for i, orig_idx in enumerate(organic_indices)}

    h_h_edges = [
        edge for edge in nx_graph.edges()
        if organic_symbols[edge[0]] == 'H' and organic_symbols[edge[1]] == 'H'
    ]
    nx_graph.remove_edges_from(h_h_edges)

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
    """Get covalent bonds using pymatgen's CovalentBondNN.

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
    mol_positions = [atoms[i].position for i in indices_to_use]
    mol = Molecule(mol_symbols, mol_positions)
    strategy = CovalentBondNN()

    try:
        mg = MoleculeGraph.from_local_env_strategy(mol, strategy)
    except Exception as e:
        logger.warning("Could not build molecule graph: %s", e)
        return []

    mol_idx_to_orig = {i: orig_idx for i, orig_idx in enumerate(indices_to_use)}
    bonds = []

    for edge in mg.graph.edges(data=True):
        mol_i, mol_j = edge[0], edge[1]
        orig_i = mol_idx_to_orig[mol_i]
        orig_j = mol_idx_to_orig[mol_j]

        if atoms[orig_i].symbol == 'H' and atoms[orig_j].symbol == 'H':
            continue

        bond_length = np.linalg.norm(
            np.array(atoms[orig_i].position) - np.array(atoms[orig_j].position)
        )
        bonds.append((orig_i, orig_j, bond_length))

    return bonds


def build_molecular_graph(
    atoms: Atoms,
    exclude_indices: Set[int],
) -> nx.Graph:
    """Build molecular connectivity graph using pymatgen's CovalentBondNN.

    Creates a graph where nodes are atom indices and edges represent
    covalent bonds. Hydrogen bonds are not included.

    Parameters
    ----------
    atoms : ase.Atoms
        The full crystal structure
    exclude_indices : set of int
        Indices of atoms to exclude (e.g., atoms in octahedra)

    Returns
    -------
    networkx.Graph
        Graph with atom indices as nodes and covalent bonds as edges
    """
    included_indices = [i for i in range(len(atoms)) if i not in exclude_indices]

    if not included_indices:
        return nx.Graph()

    mol_symbols = [atoms[i].symbol for i in included_indices]
    mol_positions = [atoms[i].position for i in included_indices]
    mol = Molecule(mol_symbols, mol_positions)
    strategy = CovalentBondNN()

    try:
        mg = MoleculeGraph.from_local_env_strategy(mol, strategy)
    except Exception as e:
        logger.warning("Could not build molecule graph: %s", e)
        return nx.Graph()

    mol_idx_to_orig = {i: orig_idx for i, orig_idx in enumerate(included_indices)}
    mol_graph = nx.Graph()

    for i in included_indices:
        mol_graph.add_node(i, symbol=atoms[i].symbol)

    for edge in mg.graph.edges(data=True):
        mol_i, mol_j = edge[0], edge[1]
        orig_i = mol_idx_to_orig[mol_i]
        orig_j = mol_idx_to_orig[mol_j]

        if atoms[orig_i].symbol == 'H' and atoms[orig_j].symbol == 'H':
            continue

        bond_length = np.linalg.norm(
            np.array(atoms[orig_i].position) - np.array(atoms[orig_j].position)
        )
        mol_graph.add_edge(orig_i, orig_j, bond_length=bond_length, edge_type='covalent_bond')

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
    from ..utils.geometry import _calculate_distances

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

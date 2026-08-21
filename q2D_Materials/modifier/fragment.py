"""Fragment utilities for SMILES conversion and molecular graph construction."""

import logging
import numpy as np
import networkx as nx
from typing import Optional, Tuple
from ase import Atoms

from q2D_Materials.utils.molecules.molecule_builder import validate_smiles
from q2D_Materials.utils.molecules.graph_converter import (
    rdkit_to_graph,
    graph_to_rdkit,
    validate_smiles as validate_smiles_rdkit
)
from q2D_Materials.utils.properties.atomic_properties import get_covalent_radius

logger = logging.getLogger(__name__)

# Import RDKit for SMILES computation
from rdkit import Chem


def atoms_to_smiles(atoms: Atoms) -> Optional[str]:
    """Convert ASE Atoms to SMILES string using RDKit.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object representing a molecule

    Returns
    -------
    str or None
        SMILES string, or None if RDKit is not available or conversion fails
    """
    try:
        mol = Chem.RWMol()

        # Add atoms
        atom_map = {}
        for i, symbol in enumerate(atoms.get_chemical_symbols()):
            atom = Chem.Atom(symbol)
            atom_map[i] = mol.AddAtom(atom)

        # Add bonds based on geometry
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()

        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                dist = np.linalg.norm(positions[i] - positions[j])
                r1 = get_covalent_radius(symbols[i])
                r2 = get_covalent_radius(symbols[j])

                # Check if atoms are bonded (within 1.3 * sum of radii)
                if dist < 1.3 * (r1 + r2):
                    bond_order = 1
                    if dist < 0.9 * (r1 + r2):
                        bond_order = 2
                    if dist < 0.8 * (r1 + r2):
                        bond_order = 3

                    mol.AddBond(atom_map[i], atom_map[j], Chem.BondType(bond_order))

        mol = mol.GetMol()
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except Exception as e:
        logger.warning("Could not compute SMILES: %s", e)
        return None


def validate_fragment(smiles: str, fragment_index: int = 0) -> Tuple[bool, str]:
    """Validate that a SMILES fragment is valid and ready for use.
    
    This function checks that:
    1. The SMILES string is valid and can be parsed
    2. The fragment can be converted to a molecular graph (graph-based, no 3D coordinates)
    3. The specified fragment_index corresponds to a valid atom with bonds
    
    Note: This function does NOT check for explicit attachment point notation.
    Users should provide fragments with explicit notation (e.g., [CH0]=C) to avoid
    ambiguity. The native SMILES format supports this.
    
    Parameters
    ----------
    smiles : str
        SMILES string representing the molecular fragment.
        Should use explicit attachment point notation (e.g., [CH0] for carbon with 0 hydrogens).
    fragment_index : int, default=0
        Index of the atom in the fragment to use as attachment point
        
    Returns
    -------
    tuple of (bool, str)
        (is_valid, message)
        - is_valid: True if fragment is valid and ready for use
        - message: Explanation of validation result or error message
    """
    # Check if SMILES is valid
    if not validate_smiles(smiles):
        return False, f"Invalid SMILES string: {smiles}"
    
    try:
        # Use RDKit for SMILES parsing (no 3D coordinates needed)
        validate_smiles_rdkit(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, f"RDKit could not parse SMILES string: {smiles}"
        mol_graph = rdkit_to_graph(mol, coords=None)
        fragment_nodes = list(mol_graph.nodes())
        
        if fragment_index >= len(fragment_nodes):
            return False, (
                f"Fragment index {fragment_index} out of range. "
                f"Fragment has {len(fragment_nodes)} atoms (indices 0-{len(fragment_nodes)-1})."
            )
        
        # Check if the specified attachment atom is reasonable
        attachment_node = fragment_nodes[fragment_index]
        attachment_symbol = mol_graph.nodes[attachment_node].get('symbol', 'C')
        attachment_neighbors = list(mol_graph.neighbors(attachment_node))
        
        # Attachment point should have at least one bond (to the fragment)
        if len(attachment_neighbors) == 0:
            return False, (
                f"Fragment attachment atom at index {fragment_index} ({attachment_symbol}) "
                f"has no bonds. This is not a valid attachment point."
            )
        
        return True, "Fragment is valid and ready for use"
        
    except Exception as e:
        return False, f"Error validating fragment: {e}"


def from_smiles(smiles: str, validate: bool = True) -> nx.Graph:
    """Convert SMILES string to NetworkX graph using RDKit.

    Uses RDKit for SMILES parsing without 3D coordinate generation.
    Pattern matching is based solely on element symbols and neighbor connectivity,
    making it fast, accurate, and geometry-independent.

    Parameters
    ----------
    smiles : str
        SMILES string representing the molecular fragment.
        Should use explicit attachment point notation (e.g., [CH0] for carbon with 0 hydrogens).
    validate : bool, default=True
        If True, validate that the fragment is unambiguous before conversion.
        Set to False to skip validation (not recommended).

    Returns
    -------
    nx.Graph
        NetworkX graph with atom indices as nodes and bonds as edges.
        Each node has 'symbol' (str) attribute and optional attributes like
        'charge', 'hcount', 'aromatic', etc.
        No 3D positions are included - graph matching is based on connectivity only.
        
    Raises
    ------
    ValueError
        If validation fails and validate=True, or if SMILES is invalid
    """
    if validate:
        is_valid, message = validate_fragment(smiles, fragment_index=0)
        if not is_valid:
            raise ValueError(
                f"Fragment validation failed: {message}\n"
                f"Please use explicit SMILES notation with unambiguous attachment points "
                f"(e.g., [CH0]=C for vinyl group with explicit attachment point)."
            )
    
    # Use RDKit for SMILES parsing (no 3D coordinates - graph-based only)
    validate_smiles_rdkit(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES string: {smiles}")
    mol_graph = rdkit_to_graph(mol, coords=None)

    # Ensure 'symbol' attribute exists for all nodes (should already be set by rdkit_to_graph)
    for node in mol_graph.nodes():
        if 'symbol' not in mol_graph.nodes[node]:
            # Fallback: try to get from other attributes
            symbol = mol_graph.nodes[node].get('element', 'C')
            mol_graph.nodes[node]['symbol'] = symbol

    return mol_graph

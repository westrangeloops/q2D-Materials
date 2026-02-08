"""Unified molecular graph construction and backbone identification.

This module provides functions to create standardized molecular graphs from
various input formats (SMILES, XYZ, Atoms, or existing graphs) and identify
backbone atoms through NH3 group analysis.

The molecular graphs created here match the format used in graph_construction.py,
enabling seamless integration between structure-extracted molecules and standalone
molecular analysis.
"""

from typing import Union, Optional, List
import numpy as np
import networkx as nx
from ase import Atoms
from collections import Counter

from ...utils.molecules.graph_converter import (
    graph_to_rdkit,
    rdkit_to_graph,
    validate_smiles,
    atoms_to_graph
)

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None


def create_molecule_graph(
    molecule: Union[Atoms, str, nx.Graph],
    analyze_backbone: bool = True,
    initial_pattern: Union[str, List[str]] = None,
    final_pattern: Union[str, List[str]] = None
) -> nx.Graph:
    """Create unified molecular graph from SMILES, XYZ, Atoms, or existing graph.
    
    This function creates a standardized molecular graph format that matches
    the structure used in graph_construction.py, enabling all analysis modules
    to work identically on both structure-extracted molecules and standalone molecules.
    
    The returned graph has:
    - molecule_0 node with attributes: node_type='molecule', nh3_count, formula
    - atom nodes (0, 1, 2, ...) with attributes: node_type='atom', symbol, role
    - bonded_to edges between atoms
    - contains edges from molecule_0 to all atoms
    
    Parameters
    ----------
    molecule : Atoms, str, or nx.Graph
        Input molecule as ASE Atoms object, SMILES string, or existing NetworkX graph
    analyze_backbone : bool, default=True
        If True, perform backbone identification and mark atoms with role attribute
    initial_pattern : str or List[str], optional
        SMARTS pattern(s) for NH3 groups (default: ['[NH3+]C', '[NH2]C'])
    final_pattern : str or List[str], optional
        SMARTS pattern(s) for final NH3 groups (default: same as initial_pattern)
    
    Returns
    -------
    nx.Graph
        Unified molecular graph with molecule node and atom nodes
        
    Examples
    --------
    >>> from q2D_Materials.analyzer.molecular_processing import create_molecule_graph
    >>> 
    >>> # From SMILES
    >>> graph = create_molecule_graph("C(CC[NH3+])C[NH3+]")
    >>> 
    >>> # From Atoms object
    >>> from ase.io import read
    >>> atoms = read("molecule.xyz")
    >>> graph = create_molecule_graph(atoms)
    >>> 
    >>> # Access molecule properties
    >>> mol_data = graph.nodes['molecule_0']
    >>> print(f"NH3 count: {mol_data['nh3_count']}")
    >>> 
    >>> # Access atom properties
    >>> for node in graph.nodes():
    ...     if node != 'molecule_0':
    ...         atom_data = graph.nodes[node]
    ...         print(f"Atom {node}: {atom_data['symbol']}, role={atom_data.get('role')}")
    """
    # Convert input to molecular graph (atom-level connectivity)
    if isinstance(molecule, nx.Graph):
        # Already a graph - check if it's a full structural graph or just molecular
        if any(node.startswith('molecule_') for node in molecule.nodes()):
            # Already has molecule nodes - return as is
            return molecule
        else:
            # Just atom connectivity graph - wrap it
            mol_graph = molecule
    elif isinstance(molecule, str):
        # SMILES string
        try:
            validate_smiles(molecule)
            mol = Chem.MolFromSmiles(molecule)
            if mol is None:
                raise ValueError(f"RDKit could not parse SMILES: {molecule}")
            mol_graph = rdkit_to_graph(mol, coords=None)
        except Exception as e:
            raise ValueError(f"Invalid SMILES string: {e}")
    else:
        # ASE Atoms object
        try:
            mol_graph = atoms_to_graph(molecule, preserve_coords=True)
        except Exception as e:
            raise ValueError(f"Failed to build molecular graph: {e}")
    
    # Create unified graph with molecule node
    G = nx.Graph()
    
    # Calculate formula from atom symbols
    symbols = [mol_graph.nodes[node].get('symbol', 'C') for node in sorted(mol_graph.nodes())]
    symbol_counts = Counter(symbols)
    
    # Hill notation: C first, then H, then alphabetical
    formula_parts = []
    if 'C' in symbol_counts:
        formula_parts.append(f"C{symbol_counts['C']}" if symbol_counts['C'] > 1 else 'C')
        del symbol_counts['C']
    if 'H' in symbol_counts:
        formula_parts.append(f"H{symbol_counts['H']}" if symbol_counts['H'] > 1 else 'H')
        del symbol_counts['H']
    for symbol in sorted(symbol_counts.keys()):
        count = symbol_counts[symbol]
        formula_parts.append(f"{symbol}{count}" if count > 1 else symbol)
    formula = ''.join(formula_parts)
    
    # Add molecule node
    G.add_node(
        'molecule_0',
        node_type='molecule',
        molecule_type='organic',  # Generic type for standalone molecules
        formula=formula,
        nh3_count=0  # Will be set by identify_backbone if analyze_backbone=True
    )
    
    # Add atom nodes (renumber to match structural graph format)
    node_mapping = {}
    for idx, old_node in enumerate(sorted(mol_graph.nodes())):
        new_node = idx
        node_data = mol_graph.nodes[old_node].copy()
        node_data['node_type'] = 'atom'
        # Ensure symbol is present
        if 'symbol' not in node_data:
            node_data['symbol'] = 'C'  # Default fallback
        G.add_node(new_node, **node_data)
        node_mapping[old_node] = new_node
        
        # Add CONTAINS edge from molecule to atom
        G.add_edge('molecule_0', new_node, edge_type='contains')
    
    # Add bonded_to edges between atoms
    for old_i, old_j in mol_graph.edges():
        new_i = node_mapping[old_i]
        new_j = node_mapping[old_j]
        edge_data = mol_graph.edges[old_i, old_j].copy()
        edge_data['edge_type'] = 'bonded_to'
        
        # Add bond_type if not present
        if 'bond_type' not in edge_data:
            symbol_i = G.nodes[new_i]['symbol']
            symbol_j = G.nodes[new_j]['symbol']
            edge_data['bond_type'] = f'{symbol_i}-{symbol_j}' if symbol_i <= symbol_j else f'{symbol_j}-{symbol_i}'
        
        G.add_edge(new_i, new_j, **edge_data)
    
    # Perform backbone analysis if requested
    if analyze_backbone:
        identify_backbone(G, 'molecule_0', initial_pattern, final_pattern)
    
    return G


def identify_backbone(
    graph: nx.Graph,
    molecule_node: str,
    initial_pattern: Union[str, List[str]] = None,
    final_pattern: Union[str, List[str]] = None
) -> int:
    """Identify backbone atoms and count NH3 groups in a molecule.
    
    This function:
    1. Finds all NH3 groups using SMARTS pattern matching
    2. Stores nh3_count on the molecule node
    3. Finds the longest path between NH3 groups (or from NH3) through CHON backbone
    4. Marks atoms on the path as role='backbone'
    5. Marks other CHON atoms (not H) as role='functional_group'
    
    The algorithm avoids H atoms when finding paths (they are dead ends).
    
    Parameters
    ----------
    graph : nx.Graph
        Molecular graph with molecule node and atom nodes
    molecule_node : str
        ID of the molecule node (e.g., 'molecule_0')
    initial_pattern : str or List[str], optional
        SMARTS pattern(s) for NH3 groups (default: ['[NH3+]C', '[NH2]C'])
    final_pattern : str or List[str], optional
        SMARTS pattern(s) for final NH3 groups (default: same as initial_pattern)
    
    Returns
    -------
    int
        Number of NH3 groups found
        
    Modifies
    --------
    graph : nx.Graph
        Sets G.nodes[atom]['role'] = 'backbone' or 'functional_group' for CHON atoms
        Sets G.nodes[molecule_node]['nh3_count'] = int
        
    Examples
    --------
    >>> G = create_molecule_graph("C(CC[NH3+])C[NH3+]", analyze_backbone=False)
    >>> nh3_count = identify_backbone(G, 'molecule_0')
    >>> print(f"Found {nh3_count} NH3 groups")
    >>> 
    >>> # Check atom roles
    >>> for node in G.nodes():
    ...     if node != 'molecule_0':
    ...         role = G.nodes[node].get('role')
    ...         if role:
    ...             print(f"Atom {node}: {role}")
    """
    # Import here to avoid circular imports
    from .molecule_candidates import _parse_pattern, _find_pattern_matches
    
    # Default patterns
    if initial_pattern is None:
        initial_pattern = ['[NH3+]C', '[NH2]C']
    if final_pattern is None:
        final_pattern = initial_pattern
    
    # Convert to lists
    if isinstance(initial_pattern, str):
        initial_patterns = [initial_pattern]
    else:
        initial_patterns = initial_pattern
    
    if isinstance(final_pattern, str):
        final_patterns = [final_pattern]
    else:
        final_patterns = final_pattern
    
    # Extract molecular subgraph (only atoms, not the molecule node)
    atom_nodes = [n for n in graph.nodes() if n != molecule_node and graph.nodes[n].get('node_type') == 'atom']
    mol_subgraph = graph.subgraph(atom_nodes).copy()
    
    # Find all NH3 pattern matches
    all_nh3_matches = []
    for pattern in initial_patterns + final_patterns:
        try:
            pat_mol = _parse_pattern(pattern)
            matches = _find_pattern_matches(mol_subgraph, pat_mol)
            all_nh3_matches.extend(matches)
        except Exception:
            continue
    
    # Remove duplicate matches (same anchor)
    unique_nh3_anchors = list(set(m['anchor_idx'] for m in all_nh3_matches if m['anchor_idx'] is not None))
    nh3_count = len(unique_nh3_anchors)
    
    # Store nh3_count on molecule node
    graph.nodes[molecule_node]['nh3_count'] = nh3_count
    
    # Find backbone path
    backbone_atoms = set()
    
    if nh3_count >= 2:
        # Find longest path between any pair of NH3 groups through CHON
        longest_path = []
        
        for i, anchor1 in enumerate(unique_nh3_anchors):
            for anchor2 in unique_nh3_anchors[i+1:]:
                # Find path avoiding H atoms (dead ends)
                path = _find_path_avoiding_h(mol_subgraph, anchor1, anchor2)
                if path and len(path) > len(longest_path):
                    longest_path = path
        
        backbone_atoms = set(longest_path)
    
    elif nh3_count == 1:
        # Find longest path from the single NH3 group through CHON
        anchor = unique_nh3_anchors[0]
        longest_path = _find_longest_path_from_node(mol_subgraph, anchor)
        backbone_atoms = set(longest_path)
    
    # Mark atoms with roles
    chon_elements = {'C', 'H', 'O', 'N'}
    
    for node in atom_nodes:
        symbol = graph.nodes[node].get('symbol', '')
        
        # Only mark CHON atoms (skip H as they're not structural)
        if symbol in chon_elements and symbol != 'H':
            if node in backbone_atoms:
                graph.nodes[node]['role'] = 'backbone'
            else:
                graph.nodes[node]['role'] = 'functional_group'
    
    return nh3_count


def _find_path_avoiding_h(graph: nx.Graph, start: int, end: int) -> List[int]:
    """Find shortest path between two nodes, avoiding H atoms.
    
    Parameters
    ----------
    graph : nx.Graph
        Molecular graph with atom nodes
    start : int
        Starting node
    end : int
        Ending node
    
    Returns
    -------
    List[int]
        Path as list of node indices, or empty list if no path found
    """
    # Create subgraph without H atoms
    non_h_nodes = [n for n in graph.nodes() if graph.nodes[n].get('symbol', '') != 'H']
    subgraph = graph.subgraph(non_h_nodes)
    
    try:
        path = nx.shortest_path(subgraph, start, end)
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def _find_longest_path_from_node(graph: nx.Graph, start: int) -> List[int]:
    """Find longest simple path from a starting node, avoiding H atoms.
    
    Parameters
    ----------
    graph : nx.Graph
        Molecular graph with atom nodes
    start : int
        Starting node
    
    Returns
    -------
    List[int]
        Longest path as list of node indices
    """
    # Create subgraph without H atoms
    non_h_nodes = [n for n in graph.nodes() if graph.nodes[n].get('symbol', '') != 'H']
    subgraph = graph.subgraph(non_h_nodes)
    
    if start not in subgraph:
        return [start]
    
    # Use BFS to find all reachable nodes and their distances
    longest_path = [start]
    max_distance = 0
    
    # For each reachable node, find the path
    for target in subgraph.nodes():
        if target == start:
            continue
        
        try:
            path = nx.shortest_path(subgraph, start, target)
            if len(path) > len(longest_path):
                longest_path = path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    
    return longest_path


"""Molecule and NH3 group helpers for cavity tracing.

Graph-based helpers to get molecule atom indices and NH3 groups,
and to compute anchor positions from NH3 or reconstructed molecule data.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple


def _get_molecule_atoms(graph: nx.Graph, node_id: str) -> List[int]:
    """Get atom indices for a molecule.
    
    Returns
    -------
    List[int]
        List of atom indices (vasp_index)
    """
    mol_atoms = []
    for neighbor in graph.neighbors(node_id):
        edge_data = graph.get_edge_data(node_id, neighbor)
        if edge_data and edge_data.get('edge_type') == 'contains':
            neighbor_data = graph.nodes.get(neighbor, {})
            if neighbor_data.get('node_type') == 'atom':
                atom_idx = neighbor_data.get('vasp_index')
                if atom_idx is not None:
                    mol_atoms.append(atom_idx)
    return mol_atoms


def _find_nh3_groups(graph: nx.Graph, node_id: str) -> List[Dict[str, Any]]:
    """Find NH3 groups in molecule using graph connectivity.
    
    Returns
    -------
    List[Dict]
        List of {'n_index': int, 'h_indices': [int, int, int]}
    """
    nh3_groups = []
    
    # Get all atoms belonging to the molecule via CONTAINS edges
    molecule_atoms = [
        neighbor for neighbor in graph.neighbors(node_id)
        if graph.get_edge_data(node_id, neighbor).get('edge_type') == 'contains'
    ]
    
    # Identify Nitrogen atoms
    n_nodes = [
        node for node in molecule_atoms 
        if graph.nodes[node].get('symbol') == 'N'
    ]
    
    for n_node in n_nodes:
        h_neighbors = []
        
        # Check bonded neighbors of the Nitrogen atom
        for neighbor in graph.neighbors(n_node):
            edge_data = graph.get_edge_data(n_node, neighbor)
            
            if edge_data.get('edge_type') == 'bonded_to':
                if graph.nodes[neighbor].get('symbol') == 'H':
                    h_neighbors.append(graph.nodes[neighbor].get('vasp_index'))
        
        # Validation: Exactly 3 Hydrogen atoms bonded to 1 Nitrogen
        if len(h_neighbors) == 3:
            nh3_groups.append({
                'n_index': graph.nodes[n_node].get('vasp_index'),
                'h_indices': h_neighbors
            })
            
    return nh3_groups


def _calculate_nh3_center(nh3_group: Dict[str, Any], atom_positions: np.ndarray) -> np.ndarray:
    """Get N atom position from NH3 group (used as anchor point)."""
    return atom_positions[nh3_group['n_index']]


def _calculate_nh3_center_from_reconstructed(
    nh3_group: Dict[str, Any],
    mol_data: Dict[int, Tuple[np.ndarray, Tuple[int, int, int]]]
) -> np.ndarray:
    """Get N atom position from reconstructed molecule coordinates (used as anchor point).
    
    This function is used specifically for DJ spacer processing, where we need to find
    B atoms relative to the reconstructed (unwrapped) molecule positions. For A-site and
    RP spacer processing, use _calculate_nh3_center instead, which returns the wrapped
    N atom position from the original cell.
    
    Parameters
    ----------
    nh3_group : Dict[str, Any]
        NH3 group dictionary with 'n_index' key
    mol_data : Dict[int, Tuple[np.ndarray, Tuple[int, int, int]]]
        Reconstructed molecule data mapping atom indices to (position, image_label)
        
    Returns
    -------
    np.ndarray
        N atom position in unwrapped (reconstructed) coordinates
    """
    n_pos, _ = mol_data[nh3_group['n_index']]
    return n_pos

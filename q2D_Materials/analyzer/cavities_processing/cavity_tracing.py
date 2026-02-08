"""Streamlined cavity detection using graph-based pipeline.

This module detects cavities (A-site and spacer) by:
1. Iterating over molecule nodes in the graph
2. Finding nearest B/X atoms using PBC-aware geometry
3. Reconstructing molecules with correct PBC coordinates
4. Building isolated cavity subgraphs

Algorithm: Geometry for distances, Graph for everything else.
"""

import sys
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional

from .cavity_class import Cavity
from .weights import ASiteWeights, SpacerWeights
from .molecule_helpers import _get_molecule_atoms
from .cavity_processors import _process_a_site, _process_rp_spacer, _process_dj_spacer
from .pbc_images import _precompute_27_images, _find_nearest_from_cached_images


def detect_all_cavities(
    graph: nx.Graph,
    atom_positions: np.ndarray,
    atom_symbols: List[str],
    cell: np.ndarray,
    analyzer: Optional[Any] = None,
    a_site_weights: Optional[ASiteWeights] = None,
    spacer_weights: Optional[SpacerWeights] = None,
    max_validation_iterations: int = 10
) -> List[Cavity]:
    """Detect all cavities in the structure.

    Iterates over molecule nodes, finds enclosing B/X atoms,
    reconstructs molecules with PBC, and builds isolated subgraphs.

    Parameters
    ----------
    graph : nx.Graph
        Structural graph with molecule, octahedron, and atom nodes
    atom_positions : np.ndarray
        Atom positions (N, 3)
    atom_symbols : List[str]
        Atom symbols
    cell : np.ndarray
        Unit cell matrix (3, 3)
    analyzer : q2D_analyzer, optional
        Analyzer instance. If provided, uses cached B/X atom data for better performance.
    a_site_weights : ASiteWeights, optional
        Weight configuration for A-site X atom selection.
        If None, uses default weights (0.7 for dist_anchor, 0.3 for z_diff).
    spacer_weights : SpacerWeights, optional
        Weight configuration for spacer terminal X atom selection.
        If None, uses default weights (0.4 for dist_anchor, 0.3 for dist_b_center, 0.3 for z_diff).
    max_validation_iterations : int, optional
        Maximum number of iterations for B-X connectivity validation and repair.
        Default: 10. Increase for more exhaustive attempts to fix malformed cavities.

    Returns
    -------
    List[Cavity]
        List of detected cavities
    """
    if a_site_weights is None:
        a_site_weights = ASiteWeights()
    if spacer_weights is None:
        spacer_weights = SpacerWeights()

    print(f"  Detecting cavities from molecule nodes...", file=sys.stderr)
    
    if analyzer is not None:
        b_x_data = analyzer.get_b_x_atoms()
        b_indices = b_x_data['b_indices']
        b_positions = b_x_data['b_positions']
        x_indices = b_x_data['x_indices']
        x_positions = b_x_data['x_positions']
    else:
        b_positions, b_indices, x_positions, x_indices = _get_b_and_x_positions(
            graph, atom_positions, atom_symbols
        )
    
    print(f"  Found {len(b_indices)} B-sites and {len(x_indices)} X-sites", file=sys.stderr)
    
    print(f"  Pre-computing 27 PBC images for B and X atoms...", file=sys.stderr)
    b_image_positions, b_image_indices, b_image_labels = _precompute_27_images(
        b_positions, b_indices, cell, pbc=True
    )
    x_image_positions, x_image_indices, x_image_labels = _precompute_27_images(
        x_positions, x_indices, cell, pbc=True
    )
    print(f"  Cached {len(b_image_positions)} B images and {len(x_image_positions)} X images", file=sys.stderr)
    
    cavities = []
    cavity_id = 0
    
    for node, data in graph.nodes(data=True):
        node_type = data.get('node_type')
        if node_type not in ['a_site', 'spacer']:
            continue
        
        molecule_type = node_type
        nh3_count = data.get('nh3_count', 0)
        
        mol_indices = _get_molecule_atoms(graph, node)
        if not mol_indices:
            continue
        
        print(f"  Processing {node}: type={molecule_type}, nh3_count={nh3_count}, atoms={len(mol_indices)}", 
              file=sys.stderr)
        
        if molecule_type == 'a_site':
            cavity = _process_a_site(
                node, mol_indices, nh3_count, graph, atom_positions, atom_symbols,
                b_positions, b_indices, x_positions, x_indices, cell, cavity_id,
                b_image_positions, b_image_indices, b_image_labels,
                x_image_positions, x_image_indices, x_image_labels,
                a_site_weights, max_validation_iterations
            )
            if cavity:
                cavities.append(cavity)
                cavity_id += 1

        elif molecule_type == 'spacer':
            if nh3_count == 1:
                cavity = _process_rp_spacer(
                    node, mol_indices, graph, atom_positions, atom_symbols,
                    b_positions, b_indices, x_positions, x_indices, cell, cavity_id,
                    b_image_positions, b_image_indices, b_image_labels,
                    x_image_positions, x_image_indices, x_image_labels,
                    spacer_weights, max_validation_iterations
                )
                if cavity:
                    cavities.append(cavity)
                    cavity_id += 1

            elif nh3_count >= 2:
                cavity = _process_dj_spacer(
                    node, mol_indices, graph, atom_positions, atom_symbols,
                    b_positions, b_indices, x_positions, x_indices, cell, cavity_id,
                    b_image_positions, b_image_indices, b_image_labels,
                    x_image_positions, x_image_indices, x_image_labels,
                    spacer_weights, max_validation_iterations
                )
                if cavity:
                    cavities.append(cavity)
                    cavity_id += 1
            else:
                print(f"  WARNING: Spacer {node} has no NH3 groups, skipping", file=sys.stderr)
    
    print(f"  Detected {len(cavities)} cavities total", file=sys.stderr)
    return cavities


def _get_b_and_x_positions(
    graph: nx.Graph,
    atom_positions: np.ndarray,
    atom_symbols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract B-site and X-site positions/indices from graph.

    Uses Octahedron->B->X structure:
    - B: atoms that are CONTAINS (role='center') by an octahedron
    - X: atoms that are BONDED_TO (role='ligand') by a B atom

    Returns
    -------
    tuple
        (b_positions, b_indices, x_positions, x_indices)
    """
    b_indices = []
    x_indices_set = set()

    for node, data in graph.nodes(data=True):
        if data.get('node_type') != 'octahedron':
            continue
        for neighbor in graph.neighbors(node):
            edge_data = graph.get_edge_data(node, neighbor)
            if (edge_data and
                edge_data.get('edge_type') == 'contains' and
                edge_data.get('role') == 'center'):
                vasp_idx = graph.nodes.get(neighbor, {}).get('vasp_index')
                if vasp_idx is not None:
                    b_indices.append(vasp_idx)
                break

    for v in b_indices:
        b_node = f'atom_{v}'
        if b_node not in graph:
            continue
        for neighbor in graph.neighbors(b_node):
            edge_data = graph.get_edge_data(b_node, neighbor)
            if (edge_data and
                edge_data.get('edge_type') == 'bonded_to' and
                edge_data.get('role') == 'ligand'):
                vasp_idx = graph.nodes.get(neighbor, {}).get('vasp_index')
                if vasp_idx is not None:
                    x_indices_set.add(vasp_idx)

    b_indices = np.array(b_indices, dtype=np.int32)
    x_indices = np.array(sorted(x_indices_set), dtype=np.int32)
    b_positions = atom_positions[b_indices]
    x_positions = atom_positions[x_indices]

    return b_positions, b_indices, x_positions, x_indices

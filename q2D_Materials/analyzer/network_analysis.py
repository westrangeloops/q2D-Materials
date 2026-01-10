"""B-X network graph construction for slab analysis.

This module builds connectivity graphs where octahedra are nodes and edges
represent sharing of X-site atoms.
"""

import numpy as np
import networkx as nx


def _build_bx_network(
    octahedra_info: list,
    atom_positions: np.ndarray,
    shared_atoms: dict,
) -> nx.Graph:
    """Build B-X network graph with octahedra as nodes.

    Graph is used to identify slabs based on z-continuity:
    - Nodes: Octahedra with z-coordinate of their center
    - Edges: Octahedra that share X-site atoms, with edge weight = z-difference

    Parameters
    ----------
    octahedra_info : list
        List of octahedra dictionaries from _graph_inorganic_ontology
    atom_positions : np.ndarray
        Array of all atom positions
    shared_atoms : dict
        Dictionary mapping (oct_i, oct_j) -> list of shared atom indices

    Returns
    -------
    nx.Graph
        B-X network graph with octahedra as nodes
    """
    bx_graph = nx.Graph()

    for oct_idx, oct_data in enumerate(octahedra_info):
        central_idx = oct_data.get('central_atom_index')
        if central_idx is not None:
            z_coord = atom_positions[central_idx][2]
            bx_graph.add_node(
                oct_idx,
                node_type='octahedron',
                central_atom=central_idx,
                z_coord=z_coord,
                octahedra_data=oct_data,
            )

    for (oct_i, oct_j), shared in shared_atoms.items():
        if oct_i not in bx_graph.nodes or oct_j not in bx_graph.nodes:
            continue

        z_i = bx_graph.nodes[oct_i]['z_coord']
        z_j = bx_graph.nodes[oct_j]['z_coord']
        z_diff = abs(z_j - z_i)
        n_shared = len(shared)

        if n_shared >= 3:
            sharing_type = 'edge'
        else:
            sharing_type = 'corner'

        bx_graph.add_edge(
            oct_i, oct_j,
            z_difference=z_diff,
            n_shared_atoms=n_shared,
            shared_atoms=shared,
            sharing_type=sharing_type,
        )

    return bx_graph

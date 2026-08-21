"""Stack (slab) detection for twisted multilayer perovskite structures."""

from typing import Any, Dict, List

import networkx as nx
import numpy as np

from ..core.layer_identification import _identify_slabs_by_continuity
from ..octahedral_processing.octahedral_detection import find_shared_atoms


def detect_stacks(
    graph: nx.Graph,
    octahedra_info: List[Dict[str, Any]],
    neighbor_indices: List[List[int]],
    atom_positions: np.ndarray,
    cell: np.ndarray = None,
) -> Dict[str, Any]:
    """Detect independent perovskite stacks (slabs) in a structure graph.

    Wraps ``_identify_slabs_by_continuity`` with correct ligand neighbor data and
    maps octahedron enumeration indices to graph node IDs.

    Parameters
    ----------
    graph : nx.Graph
        Structural connectivity graph.
    octahedra_info : list of dict
        Output from ``build_octahedra_ligand_info`` (must include ``id`` key).
    neighbor_indices : list of list of int
        Ligand atom indices per octahedron (same order as octahedra_info).
    atom_positions : np.ndarray
        Cartesian atom positions (N, 3).
    cell : np.ndarray, optional
        3x3 cell matrix for minimum-image B–B distances.

    Returns
    -------
    dict
        Keys: ``slabs`` (slab_id -> list of octahedron node IDs),
        ``slab_z_ranges``, ``discontinuity_regions``, ``expected_layer_spacing``,
        ``max_z_jump_threshold``, ``n_slabs``, ``is_multi_stack``.
    """
    shared_atoms = find_shared_atoms(neighbor_indices)
    raw = _identify_slabs_by_continuity(
        octahedra_info, shared_atoms, atom_positions, cell=cell
    )

    # Map enumeration indices to octahedron node IDs
    oct_id_by_idx = [oct['id'] for oct in octahedra_info]
    slabs_with_ids: Dict[int, List[str]] = {}
    for slab_id, oct_indices in raw.get('slabs', {}).items():
        slabs_with_ids[slab_id] = [oct_id_by_idx[i] for i in oct_indices if i < len(oct_id_by_idx)]

    n_slabs = len(slabs_with_ids)
    return {
        'slabs': slabs_with_ids,
        'slab_z_ranges': raw.get('slab_z_ranges', {}),
        'discontinuity_regions': raw.get('discontinuity_regions', []),
        'expected_layer_spacing': raw.get('expected_layer_spacing', 7.0),
        'max_z_jump_threshold': raw.get('max_z_jump_threshold', 0.0),
        'n_slabs': n_slabs,
        'is_multi_stack': n_slabs > 1,
    }

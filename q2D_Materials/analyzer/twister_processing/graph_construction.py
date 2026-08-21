"""Graph enrichment with slab (stack) nodes for multi-slab structures.

Slab nodes are added whenever ``detect_stacks`` finds one or more independent
inorganic stacks. This includes ordinary RP/DJ cells with z-discontinuous
layers as well as user-declared twisters. ``is_twister`` on the structure root
is not set from slab count here; the analyzer sets it only when the structure
type was user/creator-declared as ``twister``.
"""

from typing import Any, Dict, List, Optional, Set

import networkx as nx
import numpy as np


def _get_layer_for_b(b_idx: int, parent_graph: nx.Graph) -> Optional[str]:
    """Return layer ID string (without prefix) for a B atom index."""
    b_node = f'atom_{b_idx}'
    if b_node not in parent_graph:
        return None
    for oct_node in parent_graph.neighbors(b_node):
        if not str(oct_node).startswith('octahedron_'):
            continue
        for layer_node in parent_graph.neighbors(oct_node):
            if str(layer_node).startswith('layer_'):
                return str(layer_node).replace('layer_', '')
    return None


def _get_atom_indices_for_molecule(graph: nx.Graph, mol_node: str) -> List[int]:
    """Return VASP indices of atoms contained in a molecule node."""
    indices: List[int] = []
    for neighbor in graph.neighbors(mol_node):
        edge_data = graph.get_edge_data(mol_node, neighbor)
        if not edge_data or edge_data.get('edge_type') != 'contains':
            continue
        nd = graph.nodes.get(neighbor, {})
        if nd.get('node_type') == 'atom':
            vasp_idx = nd.get('vasp_index')
            if vasp_idx is not None:
                indices.append(vasp_idx)
    return indices


def _slabs_for_z(z: float, stacks_info: Dict[str, Any]) -> Set[str]:
    """Return slab IDs associated with a z coordinate.

    - Contained in a slab z-range → that slab.
    - In the gap between two adjacent slab ranges → both (bridging cation).
    - Outside all ranges (vacuum) → nearest slab by midpoint.
    """
    z_ranges = stacks_info.get('slab_z_ranges', {})
    if not z_ranges:
        return set()

    ranges: List[tuple] = []
    contained: Set[str] = set()
    for sid, (z_min, z_max) in z_ranges.items():
        if z_min is None or z_max is None:
            continue
        sid_s = str(sid)
        ranges.append((sid_s, float(z_min), float(z_max)))
        if z_min <= z <= z_max:
            contained.add(sid_s)
    if contained:
        return contained

    ranges.sort(key=lambda r: r[1])
    for i in range(len(ranges) - 1):
        sid_lo, _, zmax_lo = ranges[i]
        sid_hi, zmin_hi, _ = ranges[i + 1]
        if zmax_lo < z < zmin_hi:
            return {sid_lo, sid_hi}

    best_sid = None
    best_dist = float('inf')
    for sid, z_min, z_max in ranges:
        mid = 0.5 * (z_min + z_max)
        dist = abs(z - mid)
        if dist < best_dist:
            best_dist = dist
            best_sid = sid
    return {best_sid} if best_sid is not None else set()


def _get_layers_for_molecule(
    graph: nx.Graph,
    mol_node: str,
) -> Set[str]:
    """Find layer IDs associated with atoms contained in a molecule node."""
    layer_ids: Set[str] = set()
    for neighbor in graph.neighbors(mol_node):
        edge_data = graph.get_edge_data(mol_node, neighbor)
        if not edge_data or edge_data.get('edge_type') != 'contains':
            continue
        nd = graph.nodes.get(neighbor, {})
        if nd.get('node_type') != 'atom':
            continue
        vasp_idx = nd.get('vasp_index')
        if vasp_idx is None:
            continue
        # Walk B neighbors for layer assignment
        atom_node = f'atom_{vasp_idx}'
        if atom_node in graph:
            for nb in graph.neighbors(atom_node):
                if str(nb).startswith('octahedron_'):
                    for layer in graph.neighbors(nb):
                        if str(layer).startswith('layer_'):
                            layer_ids.add(str(layer).replace('layer_', ''))
        # Also try direct B connection
        lid = _get_layer_for_b(vasp_idx, graph)
        if lid is not None:
            layer_ids.add(lid)
    return layer_ids


def _octahedron_to_layer(graph: nx.Graph) -> Dict[str, str]:
    """Map octahedron node ID -> layer ID string (without prefix)."""
    mapping: Dict[str, str] = {}
    for node, data in graph.nodes(data=True):
        if data.get('node_type') != 'layer':
            continue
        layer_id = str(node).replace('layer_', '')
        for neighbor in graph.neighbors(node):
            edge_data = graph.get_edge_data(node, neighbor)
            if edge_data and edge_data.get('edge_type') == 'contains':
                if str(neighbor).startswith('octahedron_'):
                    mapping[neighbor] = layer_id
    return mapping


def _build_layer_to_slab(stacks_info: Dict[str, Any], oct_to_layer: Dict[str, str]) -> Dict[str, str]:
    """Map layer_id -> slab_id string."""
    layer_to_slab: Dict[str, str] = {}
    for slab_id, oct_list in stacks_info.get('slabs', {}).items():
        for oct_id in oct_list:
            layer_id = oct_to_layer.get(oct_id)
            if layer_id is not None:
                layer_to_slab[layer_id] = str(slab_id)
    return layer_to_slab


def enrich_graph_with_slabs(
    G: nx.Graph,
    stacks_info: Dict[str, Any],
    atom_positions=None,
) -> None:
    """Add slab nodes and rewire hierarchy: structure_0 -> slab_k -> layer_i.

    Mutates ``G`` in place.

    Parameters
    ----------
    G : nx.Graph
        Structural graph after ``_graph_inorganic_ontology``.
    stacks_info : dict
        Output from ``detect_stacks``.
    atom_positions : array-like, optional
        Cartesian positions used to assign a_site/spacer nodes to slabs by z.
    """
    atom_positions = np.asarray(atom_positions) if atom_positions is not None else None

    structure_id = G.graph.get('structure_node', 'structure_0')
    if structure_id not in G:
        structure_id = 'structure_0'

    slabs = stacks_info.get('slabs', {})
    if not slabs:
        return

    oct_to_layer = _octahedron_to_layer(G)
    layer_to_slab = _build_layer_to_slab(stacks_info, oct_to_layer)

    # Collect layers per slab
    slab_layers: Dict[str, List[str]] = {str(sid): [] for sid in slabs}
    for layer_id, slab_id in layer_to_slab.items():
        if slab_id in slab_layers and layer_id not in slab_layers[slab_id]:
            slab_layers[slab_id].append(layer_id)

    # Create slab nodes and rewire layer edges
    for slab_id, oct_list in slabs.items():
        slab_key = f'slab_{slab_id}'
        z_ranges = stacks_info.get('slab_z_ranges', {})
        z_range = z_ranges.get(slab_id, (None, None))
        layer_ids = sorted(slab_layers.get(str(slab_id), []), key=lambda x: int(x) if x.isdigit() else 0)

        G.add_node(
            slab_key,
            node_type='slab',
            z_range=z_range,
            layer_count=len(layer_ids),
            octahedra_count=len(oct_list),
            layer_ids=layer_ids,
        )
        G.add_edge(structure_id, slab_key, edge_type='contains')

        for layer_id in layer_ids:
            layer_node = f'layer_{layer_id}'
            if layer_node not in G:
                continue
            # Remove direct structure -> layer edge if present
            if G.has_edge(structure_id, layer_node):
                G.remove_edge(structure_id, layer_node)
            G.add_edge(slab_key, layer_node, edge_type='contains')

    # Update structure root metadata (is_twister is declaration-only; set by analyzer)
    n_slabs = len(slabs)
    if structure_id in G:
        G.nodes[structure_id]['n_slabs'] = n_slabs
        G.nodes[structure_id]['slab_count'] = n_slabs
        G.nodes[structure_id]['is_multi_stack'] = n_slabs > 1
        if 'is_twister' not in G.nodes[structure_id]:
            G.nodes[structure_id]['is_twister'] = False

    # Reassign molecule nodes (a_site / spacer)
    for node, data in list(G.nodes(data=True)):
        node_type = data.get('node_type')
        if node_type not in ('a_site', 'spacer'):
            continue

        layer_ids = _get_layers_for_molecule(G, node)
        slab_ids: Set[str] = set()
        for lid in layer_ids:
            sid = layer_to_slab.get(lid)
            if sid is not None:
                slab_ids.add(sid)

        # Z-based fallback for atoms not linked to octahedra (e.g. Cs spacers)
        # and for cations sitting in the vdW gap (bridging).
        if not slab_ids and atom_positions is not None:
            atom_indices = _get_atom_indices_for_molecule(G, node)
            z_vals = [float(atom_positions[i][2]) for i in atom_indices if i < len(atom_positions)]
            if z_vals:
                for z in z_vals:
                    slab_ids.update(_slabs_for_z(z, stacks_info))
                if not slab_ids:
                    mean_z = sum(z_vals) / len(z_vals)
                    slab_ids.update(_slabs_for_z(mean_z, stacks_info))

        if len(slab_ids) == 1:
            sid = next(iter(slab_ids))
            slab_node = f'slab_{sid}'
            if G.has_edge(structure_id, node):
                G.remove_edge(structure_id, node)
            if slab_node in G:
                G.add_edge(slab_node, node, edge_type='contains')
        elif len(slab_ids) >= 2:
            sorted_slabs = tuple(sorted(slab_ids, key=lambda x: int(x) if str(x).isdigit() else 0))
            G.nodes[node]['bridges_slabs'] = sorted_slabs
            # Keep attached to structure_0 as interface molecule

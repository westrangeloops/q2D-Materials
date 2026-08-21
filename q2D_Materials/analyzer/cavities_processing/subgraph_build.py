"""Cavity subgraph construction with B-X connectivity and metadata.

Builds the isolated cavity subgraph (molecule + B + X nodes, B-X edges,
cage/half_cage metadata) and runs connectivity validation/repair.
"""

import os
import sys
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional

_CONNECTIVITY_DEBUG = os.environ.get("Q2D_CAVITY_CONNECTIVITY_DEBUG", "").strip().lower() in ("1", "true", "yes")

from .molecule_helpers import _find_nh3_groups
from .connectivity import (
    _get_connectivity_rules,
    _diagnose_connectivity_issues,
    _remove_excess_connections,
    _remove_bad_equatorial_edges,
    _add_missing_connections,
    _validate_bx_connectivity,
    _warn_long_bx_edges,
)


def _get_layer_for_b(b_idx, parent_graph):
    """Helper function to get layer ID for a B atom index.
    
    Parameters
    ----------
    b_idx : int
        B atom index
    parent_graph : nx.Graph
        Parent structural graph
        
    Returns
    -------
    str or None
        Layer ID (e.g., '0', '1') or None if not found
    """
    b_node = f'atom_{b_idx}'
    if b_node not in parent_graph:
        return None
    for oct in parent_graph.neighbors(b_node):
        if oct.startswith('octahedron_'):
            for layer in parent_graph.neighbors(oct):
                if layer.startswith('layer_'):
                    return layer.replace('layer_', '')
    return None


# Stricter max distance for equatorial B-X so we avoid assigning wrong PBC image (5+ Å edges)
MAX_EQUATORIAL_BOND_LENGTH = 4.0


def _build_x_candidate_pools(
    b_node_instances: List[Tuple[Tuple[int, Tuple], str]],
    subgraph: nx.Graph,
    atom_node_mapping: Dict[Tuple[int, Tuple], str],
    x_labels_in_cavity: set,
    MAX_BOND_LENGTH: float = 6.0
) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    """Build candidate X atom pools for each B atom, separated by type.
    
    For each B atom, finds all X atoms within MAX_BOND_LENGTH and categorizes
    them by type (equatorial/terminal/axial) based on their node properties.
    Equatorial candidates are further restricted to MAX_EQUATORIAL_BOND_LENGTH
    to avoid wrong PBC image assignment (long edges).
    """
    candidate_pools = {}
    
    for (b_idx, b_img_label), b_node_id in b_node_instances:
        b_node_data = subgraph.nodes[b_node_id]
        b_pos = np.array(b_node_data.get('pbc_position'))
        if b_pos is None:
            continue
        
        pools = {
            'equatorial': [],
            'terminal': [],
            'axial': []
        }
        
        for (x_idx, x_img_label) in x_labels_in_cavity:
            x_node_id = atom_node_mapping.get((x_idx, x_img_label))
            if x_node_id is None:
                continue
            
            x_node_data = subgraph.nodes.get(x_node_id, {})
            if not x_node_data.get('is_X', False):
                continue
            
            x_pos = np.array(x_node_data.get('pbc_position'))
            if x_pos is None:
                continue
            
            distance = np.linalg.norm(x_pos - b_pos)
            if distance > MAX_BOND_LENGTH:
                continue
            
            # DJ spacer: only connect B to X in the same half-cage (avoid inter-layer bonds)
            b_half = b_node_data.get('half_cage')
            x_half = x_node_data.get('half_cage')
            if b_half is not None and x_half is not None and b_half != x_half:
                continue
            
            if x_node_data.get('is_equatorial', False):
                if distance <= MAX_EQUATORIAL_BOND_LENGTH:
                    pools['equatorial'].append((x_node_id, distance))
            if x_node_data.get('is_terminal', False):
                pools['terminal'].append((x_node_id, distance))
            if x_node_data.get('is_axial', False):
                pools['axial'].append((x_node_id, distance))
        
        for x_type in pools:
            pools[x_type].sort(key=lambda x: x[1])
        # Equatorial: keep all within MAX_EQUATORIAL_BOND_LENGTH (wrong X removed by _remove_bad_equatorial_edges)
        candidate_pools[b_node_id] = pools
    
    return candidate_pools


def _add_metadata_nodes(
    subgraph: nx.Graph,
    parent_graph: nx.Graph,
    atom_node_mapping: Dict[Tuple[int, Tuple[int, int, int]], str],
    cage_info: List[Dict[str, Any]],
    molecule_node_id: str,
    node_type: str = 'a_site'
) -> None:
    """Add cage/half_cage metadata nodes to the cavity subgraph.

    Links cage/half_cage nodes to B atoms (as corners), anchor nodes, and a_site nodes.
    Does NOT connect directly to X atoms (X atoms connect via B atoms).
    """
    for cage_idx, cage_data in enumerate(cage_info):
        is_complete = cage_data.get('is_complete', True)
        
        if is_complete:
            cage_node_id = f'cage_{cage_idx}'
            cage_node_type = 'cage'
        else:
            cage_node_id = f'half_cage_{cage_idx}'
            cage_node_type = 'half_cage'

        layer_ids = [_get_layer_for_b(b, parent_graph) for b in list(cage_data['b_indices'])[:(2 if is_complete else 1)]]
        layer_property = sorted(set([l for l in layer_ids if l])) if len(layer_ids) > 1 else (layer_ids[0] if layer_ids else None)

        subgraph.add_node(
            cage_node_id,
            node_type=cage_node_type,
            cage_index=cage_idx,
            layer_id=layer_property
        )

        cage_b_indices = set(cage_data['b_indices'])

        for (atom_idx, img_label), node_id in atom_node_mapping.items():
            if atom_idx in cage_b_indices:
                subgraph.add_edge(cage_node_id, node_id, edge_type='contains', role='corner')

        if node_type == 'a_site':
            if molecule_node_id in subgraph:
                subgraph.add_edge(cage_node_id, molecule_node_id, edge_type='contains', role='a_site')
        else:
            nh3_group_idx = cage_data.get('nh3_group_idx')
            if nh3_group_idx is not None:
                anchor_node_id = f'anchor_{nh3_group_idx}'
                if anchor_node_id in subgraph:
                    subgraph.add_edge(cage_node_id, anchor_node_id, edge_type='contains', role='anchor')
            
            if not is_complete and cage_idx == 1 and len(cage_info) > 1:
                anchor_1_id = 'anchor_1'
                if anchor_1_id in subgraph:
                    subgraph.add_edge(anchor_1_id, cage_node_id, edge_type='contains', role='half_cage')


def _build_subgraph(
    mol_data: Dict[int, Tuple[np.ndarray, Tuple[int, int, int]]],
    b_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    x_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    graph: nx.Graph,
    cell: np.ndarray,
    molecule_node_id: str,
    cage_info: List[Dict[str, Any]] = None,
    max_validation_iterations: int = 10
) -> Optional[nx.Graph]:
    """Build isolated subgraph with PBC-unwrapped coordinates and direct B-X connectivity.

    The subgraph contains:
    - Molecule node and molecule atoms
    - B-atoms (cavity corners)
    - X-atoms (cavity edges)
    - Cage metadata nodes
    - Direct B-X edges (via edge_type='bonded_to')
    - No octahedron nodes (used only in parent graph for topology)
    - No X-X edges (faces are identified via path minimization through B-atoms)

    Node IDs: atom_{idx}_img_{i}_{j}_{k} for uniqueness
    Inherits all properties from parent graph nodes
    Adds edges based on parent graph connectivity (for organic molecules)
    """
    subgraph = nx.Graph()
    atom_node_mapping = {}

    mol_node_data = graph.nodes[molecule_node_id]
    node_type = mol_node_data.get('node_type', 'a_site')  # 'a_site' or 'spacer'
    
    if node_type == 'a_site':
        subgraph.add_node(molecule_node_id, **mol_node_data)

    nh3_groups = _find_nh3_groups(graph, molecule_node_id) if node_type == 'spacer' else []
    
    anchor_atom_mapping = {}
    nh3_atom_indices = set()
    for nh3_group in nh3_groups:
        nh3_atom_indices.add(nh3_group['n_index'])
        nh3_atom_indices.update(nh3_group['h_indices'])

    if len(mol_data) > 0:
        print(f"    Adding {len(mol_data)} molecule atoms to cavity subgraph", file=sys.stderr)

    for atom_idx, (pos, img_label) in mol_data.items():
        node_id = f'atom_{atom_idx}_img_{img_label[0]}_{img_label[1]}_{img_label[2]}'
        atom_node_mapping[(atom_idx, img_label)] = node_id

        parent_node = f'atom_{atom_idx}'
        parent_data = graph.nodes.get(parent_node, {})

        node_attrs = {k: v for k, v in parent_data.items() if k not in ('x', 'y', 'z', 'position')}
        node_attrs['original_index'] = atom_idx
        node_attrs['image_label'] = img_label
        node_attrs['pbc_position'] = pos

        subgraph.add_node(node_id, **node_attrs)

        if node_type == 'a_site':
            subgraph.add_edge(molecule_node_id, node_id, edge_type='contains')
    
    if node_type == 'spacer' and nh3_groups:
        for anchor_idx, nh3_group in enumerate(nh3_groups):
            anchor_node_id = f'anchor_{anchor_idx}'
            subgraph.add_node(
                anchor_node_id,
                node_type='anchor',
                anchor_index=anchor_idx
            )
            
            nh3_atom_list = [nh3_group['n_index']] + nh3_group['h_indices']
            for atom_idx in nh3_atom_list:
                for (idx, img_label), node_id in atom_node_mapping.items():
                    if idx == atom_idx:
                        subgraph.add_edge(anchor_node_id, node_id, edge_type='contains')
            
            anchor_atom_mapping[anchor_idx] = nh3_atom_list
        
        if node_type == 'spacer' and len(nh3_groups) == 2:
            pass
        elif node_type == 'spacer' and len(nh3_groups) == 1:
            pass
    
    n_cages = len(cage_info) if cage_info else 0
    if n_cages >= 2:
        b0, x0 = cage_info[0]['b_indices'], cage_info[0]['x_indices']
        n_b_cage0 = int(b0.shape[0]) if hasattr(b0, 'shape') else len(b0)
        n_x_cage0 = int(x0.shape[0]) if hasattr(x0, 'shape') else len(x0)
    else:
        n_b_cage0 = n_x_cage0 = 0

    b_indices, b_positions, b_distances, b_labels = b_data
    for i in range(len(b_indices)):
        b_idx = int(b_indices[i])
        b_pos = b_positions[i]
        img_label = tuple(b_labels[i])

        node_id = f'atom_{b_idx}_img_{img_label[0]}_{img_label[1]}_{img_label[2]}'
        atom_node_mapping[(b_idx, img_label)] = node_id

        parent_node = f'atom_{b_idx}'
        parent_data = graph.nodes.get(parent_node, {})

        node_attrs = {k: v for k, v in parent_data.items() if k not in ('x', 'y', 'z', 'position')}
        node_attrs['original_index'] = b_idx
        node_attrs['image_label'] = img_label
        node_attrs['pbc_position'] = b_pos
        node_attrs['is_B'] = True
        if n_cages >= 2:
            node_attrs['half_cage'] = 0 if i < n_b_cage0 else 1

        subgraph.add_node(node_id, **node_attrs)
    
    x_indices, x_positions, x_distances, x_labels = x_data
    for i in range(len(x_indices)):
        x_idx = int(x_indices[i])
        x_pos = x_positions[i]
        img_label = tuple(x_labels[i])

        node_id = f'atom_{x_idx}_img_{img_label[0]}_{img_label[1]}_{img_label[2]}'
        atom_node_mapping[(x_idx, img_label)] = node_id

        parent_node = f'atom_{x_idx}'
        parent_data = graph.nodes.get(parent_node, {})

        node_attrs = {k: v for k, v in parent_data.items() if k not in ('x', 'y', 'z', 'position')}
        node_attrs['original_index'] = x_idx
        node_attrs['image_label'] = img_label
        node_attrs['pbc_position'] = x_pos
        node_attrs['is_X'] = True
        if n_cages >= 2:
            node_attrs['half_cage'] = 0 if i < n_x_cage0 else 1

        subgraph.add_node(node_id, **node_attrs)

    if cage_info is None:
        cage_info = [{
            'nh3_group_idx': None,
            'b_indices': b_data[0],
            'x_indices': x_data[0],
            'is_complete': True
        }]
    
    b_indices_set = set(int(b_indices[i]) for i in range(len(b_indices)))
    b_node_instances = [
        ((b_idx, b_img_label), b_node_id)
        for (b_idx, b_img_label), b_node_id in atom_node_mapping.items()
        if b_idx in b_indices_set
    ]
    
    x_labels_in_cavity = set()
    for cage_data in cage_info:
        cage_x_indices = set(int(idx) for idx in cage_data['x_indices'])
        for (x_idx, x_img_label) in atom_node_mapping.keys():
            if x_idx in cage_x_indices:
                x_labels_in_cavity.add((x_idx, x_img_label))
    
    connectivity_rules = _get_connectivity_rules(cage_info)
    
    MAX_BOND_LENGTH = 6.0
    candidate_pools = _build_x_candidate_pools(
        b_node_instances, subgraph, atom_node_mapping, 
        x_labels_in_cavity, MAX_BOND_LENGTH
    )
    
    print(f"    Performing initial B-X edge assignment using connectivity rules: {connectivity_rules}", 
          file=sys.stderr)
    
    for x_type, required_count in connectivity_rules.items():
        if required_count == 0:
            continue
        # Process B with fewest candidates first so shared X (e.g. equatorial in DJ ring)
        # are assigned to the most constrained B before others take both slots.
        ordered_b = sorted(
            candidate_pools.items(),
            key=lambda item: len(item[1].get(x_type, []))
        )
        for b_node_id, pools in ordered_b:
            candidates = pools.get(x_type, [])
            connected = 0

            for x_node_id, distance in candidates:
                if connected >= required_count:
                    break
                
                if subgraph.has_edge(b_node_id, x_node_id):
                    continue
                
                x_geom_count = 0
                for b_neighbor in subgraph.neighbors(x_node_id):
                    if not subgraph.nodes[b_neighbor].get('is_B', False):
                        continue
                    edge_data = subgraph.get_edge_data(b_neighbor, x_node_id)
                    if edge_data and edge_data.get('geometry') == x_type:
                        x_geom_count += 1
                
                if x_type == 'equatorial':
                    max_x_connections = 2
                elif x_type == 'axial':
                    max_x_connections = 2
                else:
                    max_x_connections = 1
                
                if x_geom_count >= max_x_connections:
                    continue
                
                subgraph.add_edge(
                    b_node_id, x_node_id,
                    edge_type='bonded_to',
                    role='ligand',
                    geometry=x_type
                )
                connected += 1
    
    validation_success = False
    
    for iteration in range(max_validation_iterations):
        is_valid = _validate_bx_connectivity(
            subgraph, b_node_instances, connectivity_rules,
            x_labels_in_cavity, atom_node_mapping
        )
        if _CONNECTIVITY_DEBUG:
            print(f"    [CONNECTIVITY_DEBUG] Re-validation: {'OK' if is_valid else 'FAILED'}", file=sys.stderr)
        
        if is_valid:
            validation_success = True
            print(f"    ✓ Connectivity validated successfully after {iteration} iteration(s)", file=sys.stderr)
            _warn_long_bx_edges(subgraph, b_node_instances, max_bond_length=4.0)
            break

        # Remove equatorial X that are not in top-2 for both of their B (wrong X → eliminate, repair will replace)
        removed_bad = _remove_bad_equatorial_edges(subgraph, b_node_instances)
        
        under_connected_b, over_connected_b = _diagnose_connectivity_issues(
            subgraph, b_node_instances, connectivity_rules
        )
        
        if not under_connected_b and not over_connected_b:
            print(f"    ERROR: Validation failed but no B atom issues found (iteration {iteration})", 
                  file=sys.stderr)
            break
        
        removed = _remove_excess_connections(subgraph, over_connected_b)
        added = _add_missing_connections(subgraph, under_connected_b, candidate_pools)
        
        print(f"    Iteration {iteration}: removed {removed_bad + removed} edges (bad equatorial: {removed_bad}), added {added} edges", file=sys.stderr)
        if _CONNECTIVITY_DEBUG:
            n_bx = sum(
                1 for u, v in subgraph.edges()
                if (subgraph.nodes[u].get('is_B') and subgraph.nodes[v].get('is_X'))
                or (subgraph.nodes[u].get('is_X') and subgraph.nodes[v].get('is_B'))
            )
            print(f"    [CONNECTIVITY_DEBUG] After repair: {n_bx} B-X edges in subgraph", file=sys.stderr)
        
        if removed == 0 and added == 0:
            print(f"    ERROR: Cannot fix connectivity issues (iteration {iteration})", file=sys.stderr)
            break
    
    if not validation_success:
        print(f"    ERROR: Failed to fix B-X connectivity after {max_validation_iterations} iterations. "
              f"Graph topology is malformed. Rejecting cavity.", file=sys.stderr)
        return None

    cavity_atoms_with_labels = {}
    for (atom_idx, img_label), node_id in atom_node_mapping.items():
        if atom_idx not in cavity_atoms_with_labels:
            cavity_atoms_with_labels[atom_idx] = []
        cavity_atoms_with_labels[atom_idx].append(img_label)

    for atom_i in cavity_atoms_with_labels:
        for atom_j in cavity_atoms_with_labels:
            if atom_i >= atom_j:
                continue

            if graph.has_edge(f'atom_{atom_i}', f'atom_{atom_j}'):
                edge_data = graph.get_edge_data(f'atom_{atom_i}', f'atom_{atom_j}')
                img_label_i = cavity_atoms_with_labels[atom_i][0]
                img_label_j = cavity_atoms_with_labels[atom_j][0]
                node_i = atom_node_mapping[(atom_i, img_label_i)]
                node_j = atom_node_mapping[(atom_j, img_label_j)]

                node_i_is_b = subgraph.nodes[node_i].get('is_B', False)
                node_i_is_x = subgraph.nodes[node_i].get('is_X', False)
                node_j_is_b = subgraph.nodes[node_j].get('is_B', False)
                node_j_is_x = subgraph.nodes[node_j].get('is_X', False)
                # Never copy B-X edges from parent; subgraph B-X comes only from validation/repair
                if (node_i_is_b and node_j_is_x) or (node_i_is_x and node_j_is_b):
                    continue

                if not subgraph.has_edge(node_i, node_j):
                    subgraph.add_edge(node_i, node_j, **edge_data)

    _add_metadata_nodes(subgraph, graph, atom_node_mapping, cage_info, molecule_node_id, node_type)

    # Defensive final re-validation: ensure the subgraph we return actually passes
    final_valid = _validate_bx_connectivity(
        subgraph, b_node_instances, connectivity_rules,
        x_labels_in_cavity, atom_node_mapping
    )
    if not final_valid:
        print(f"    ERROR: Final re-validation failed after post-processing. Rejecting cavity.", file=sys.stderr)
        return None

    return subgraph

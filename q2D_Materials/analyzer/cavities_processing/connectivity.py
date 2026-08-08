"""B-X connectivity validation and repair for cavity subgraphs.

Provides rules and helpers to ensure each B atom has the correct number
of equatorial/terminal/axial X connections and to fix under/over-connection.

Debug: set Q2D_CAVITY_CONNECTIVITY_DEBUG=1 (or true/yes) to print detailed
validation and repair info to stderr (e.g. when debugging no-DJ-cavity cases).
"""

import os
import sys
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple

# Enable verbose debug when env var is set (e.g. for EXTRACT.py --folder runs)
_CONNECTIVITY_DEBUG = os.environ.get("Q2D_CAVITY_CONNECTIVITY_DEBUG", "").strip().lower() in ("1", "true", "yes")
# When set, validation fails if any B-X edge length exceeds max (catches wrong PBC image)
_STRICT_BX_DISTANCE = os.environ.get("Q2D_CAVITY_STRICT_BX_DISTANCE", "").strip().lower() in ("1", "true", "yes")


def _get_connectivity_rules(cage_info: List[Dict[str, Any]]) -> Dict[str, int]:
    """Return connectivity rules based on cavity type.
    
    Parameters
    ----------
    cage_info : List[Dict[str, Any]]
        Information about cage(s) in this cavity
        Each dict: {'nh3_group_idx': int, 'b_indices': array, 'x_indices': array, 'is_complete': bool}
        
    Returns
    -------
    Dict[str, int]
        {'equatorial': 2, 'terminal': 1, 'axial': 0}  # for antiprism (spacer)
        or
        {'equatorial': 2, 'axial': 1, 'terminal': 0}  # for cuboctahedron (A-site)
    """
    # Check if this is a complete cage (cuboctahedron) or half-cage (antiprism)
    is_complete = cage_info[0].get('is_complete', False) if cage_info else False
    
    if is_complete:
        # A-site cuboctahedron: 2 equatorial + 1 axial
        return {'equatorial': 2, 'axial': 1, 'terminal': 0}
    else:
        # Spacer antiprism: 2 equatorial + 1 terminal
        return {'equatorial': 2, 'terminal': 1, 'axial': 0}


def _diagnose_connectivity_issues(
    subgraph: nx.Graph,
    b_node_instances: List[Tuple[Tuple[int, Tuple], str]],
    connectivity_rules: Dict[str, int]
) -> Tuple[List[Dict], List[Dict]]:
    """Identify which B atoms have missing or excess connections.
    
    Parameters
    ----------
    subgraph : nx.Graph
        The cavity subgraph being built
    b_node_instances : List[Tuple[Tuple[int, Tuple], str]]
        List of ((b_idx, b_img_label), b_node_id) tuples
    connectivity_rules : Dict[str, int]
        Expected connection counts by type (e.g., {'equatorial': 2, 'terminal': 1})
        
    Returns
    -------
    Tuple[List[Dict], List[Dict]]
        (under_connected_b, over_connected_b)
        
        under_connected_b: [
            {
                'b_node_id': 'atom_50_img_0_0_0',
                'b_idx': 50,
                'b_img_label': (0, 0, 0),
                'missing': {'equatorial': 1, 'terminal': 0}  # needs 1 more equatorial
            },
            ...
        ]
        
        over_connected_b: [
            {
                'b_node_id': 'atom_51_img_0_0_0',
                'b_idx': 51,
                'b_img_label': (0, 0, 0),
                'excess': {'equatorial': 1, 'terminal': 0},  # has 1 extra equatorial
                'excess_nodes': {'equatorial': [x_node_id], 'terminal': []}
            },
            ...
        ]
    """
    under_connected_b = []
    over_connected_b = []
    
    for (b_idx, b_img_label), b_node_id in b_node_instances:
        # Count current connections by type USING EDGE GEOMETRY (not node properties)
        # This avoids issues with X atoms that have multiple type flags set
        current_connections = {
            'equatorial': [],
            'terminal': [],
            'axial': []
        }
        
        for x_neighbor in subgraph.neighbors(b_node_id):
            edge_data = subgraph.get_edge_data(b_node_id, x_neighbor)
            geometry = edge_data.get('geometry', None) if edge_data else None
            
            if geometry == 'equatorial':
                current_connections['equatorial'].append(x_neighbor)
            elif geometry == 'terminal':
                current_connections['terminal'].append(x_neighbor)
            elif geometry == 'axial':
                current_connections['axial'].append(x_neighbor)
        
        # Check for missing connections
        missing = {}
        for x_type, required_count in connectivity_rules.items():
            current_count = len(current_connections[x_type])
            if current_count < required_count:
                missing[x_type] = required_count - current_count
        
        if missing:
            under_connected_b.append({
                'b_node_id': b_node_id,
                'b_idx': b_idx,
                'b_img_label': b_img_label,
                'missing': missing
            })
        
        # Check for excess connections
        excess = {}
        excess_nodes = {}
        for x_type, required_count in connectivity_rules.items():
            current_count = len(current_connections[x_type])
            if current_count > required_count:
                excess[x_type] = current_count - required_count
                excess_nodes[x_type] = current_connections[x_type]
        
        if excess:
            over_connected_b.append({
                'b_node_id': b_node_id,
                'b_idx': b_idx,
                'b_img_label': b_img_label,
                'excess': excess,
                'excess_nodes': excess_nodes
            })
    
    if _CONNECTIVITY_DEBUG and (under_connected_b or over_connected_b):
        print(f"    [CONNECTIVITY_DEBUG] _diagnose: under_connected_b={len(under_connected_b)}, over_connected_b={len(over_connected_b)}", file=sys.stderr)
        for u in under_connected_b:
            print(f"    [CONNECTIVITY_DEBUG]   under: B{u['b_idx']} missing={u['missing']}", file=sys.stderr)
        for o in over_connected_b:
            print(f"    [CONNECTIVITY_DEBUG]   over: B{o['b_idx']} excess={o['excess']}", file=sys.stderr)

    return under_connected_b, over_connected_b


def _remove_excess_connections(
    subgraph: nx.Graph,
    over_connected_b: List[Dict]
) -> int:
    """Remove excess B-X edges (keep only closest X atoms).
    
    Parameters
    ----------
    subgraph : nx.Graph
        The cavity subgraph being built
    over_connected_b : List[Dict]
        List of B atoms with excess connections (from _diagnose_connectivity_issues)
        
    Returns
    -------
    int
        Number of edges removed
    """
    edges_removed = 0
    
    for b_data in over_connected_b:
        b_node_id = b_data['b_node_id']
        b_pos = np.array(subgraph.nodes[b_node_id].get('pbc_position'))
        if b_pos is None:
            continue
        
        excess = b_data['excess']
        excess_nodes = b_data['excess_nodes']
        
        for x_type, excess_count in excess.items():
            if excess_count == 0:
                continue
            
            x_node_list = excess_nodes.get(x_type, [])
            if not x_node_list:
                continue
            
            # Calculate distances to all X atoms of this type
            x_distances = []
            for x_node_id in x_node_list:
                x_pos = np.array(subgraph.nodes[x_node_id].get('pbc_position'))
                if x_pos is not None:
                    distance = np.linalg.norm(x_pos - b_pos)
                    x_distances.append((x_node_id, distance))
            
            # Sort by distance and identify which to remove (furthest ones)
            x_distances.sort(key=lambda x: x[1])
            to_remove = x_distances[len(x_distances) - excess_count:]
            
            # Remove edges
            for x_node_id, distance in to_remove:
                if subgraph.has_edge(b_node_id, x_node_id):
                    subgraph.remove_edge(b_node_id, x_node_id)
                    edges_removed += 1
                    print(f"    REMOVED: B{b_data['b_idx']} - {x_type} X (distance={distance:.3f} Å)", 
                          file=sys.stderr)
    
    return edges_removed


def _top2_equatorial_x_per_b(
    subgraph: nx.Graph,
    b_node_instances: List[Tuple[Tuple[int, Tuple], str]]
) -> Dict[str, set]:
    """For each B node, return the set of (at most) 2 closest equatorial X node_ids (by distance).
    Used to enforce: an equatorial X must be one of the two closest to each of its two B.
    """
    b_to_top2: Dict[str, set] = {}
    for (_b_idx, _b_img), b_node_id in b_node_instances:
        b_node_data = subgraph.nodes[b_node_id]
        b_pos = np.array(b_node_data.get("pbc_position"))
        b_half = b_node_data.get("half_cage")
        if b_pos is None:
            b_to_top2[b_node_id] = set()
            continue
        candidates = []
        for node_id in subgraph.nodes():
            nd = subgraph.nodes[node_id]
            if not nd.get("is_X") or not nd.get("is_equatorial", False):
                continue
            x_half = nd.get("half_cage")
            if b_half is not None and x_half is not None and b_half != x_half:
                continue
            x_pos = nd.get("pbc_position")
            if x_pos is None:
                continue
            d = float(np.linalg.norm(np.asarray(x_pos) - b_pos))
            candidates.append((node_id, d))
        candidates.sort(key=lambda x: x[1])
        b_to_top2[b_node_id] = {node_id for node_id, _ in candidates[:2]}
    return b_to_top2


def _remove_bad_equatorial_edges(
    subgraph: nx.Graph,
    b_node_instances: List[Tuple[Tuple[int, Tuple], str]]
) -> int:
    """Remove equatorial B-X edges where X is not one of the two closest equatorial X for both of its B.
    Such X are incorrect (not 'between' two B); removing edges lets repair assign the correct X.
    Returns number of edges removed.
    """
    b_top2 = _top2_equatorial_x_per_b(subgraph, b_node_instances)
    edges_removed = 0
    seen_x = set()
    for node_id in subgraph.nodes():
        nd = subgraph.nodes[node_id]
        if not nd.get("is_X") or node_id in seen_x:
            continue
        b_neighbors = [
            n for n in subgraph.neighbors(node_id)
            if subgraph.nodes[n].get("is_B")
        ]
        if len(b_neighbors) != 2:
            continue
        edge1 = subgraph.get_edge_data(b_neighbors[0], node_id) or {}
        edge2 = subgraph.get_edge_data(b_neighbors[1], node_id) or {}
        if edge1.get("geometry") != "equatorial" or edge2.get("geometry") != "equatorial":
            continue
        b1, b2 = b_neighbors[0], b_neighbors[1]
        if node_id in b_top2.get(b1, set()) and node_id in b_top2.get(b2, set()):
            continue
        seen_x.add(node_id)
        for b_node_id in (b1, b2):
            if subgraph.has_edge(b_node_id, node_id):
                subgraph.remove_edge(b_node_id, node_id)
                edges_removed += 1
                if _CONNECTIVITY_DEBUG:
                    bidx = subgraph.nodes[b_node_id].get("original_index", "?")
                    print(
                        f"    [CONNECTIVITY_DEBUG] Removed bad equatorial: B{bidx}-X{nd.get('original_index', '?')} "
                        "(X not in top-2 for both B)",
                        file=sys.stderr,
                    )
    return edges_removed


def _add_missing_connections(
    subgraph: nx.Graph,
    under_connected_b: List[Dict],
    candidate_pools: Dict[str, Dict[str, List[Tuple[str, float]]]]
) -> int:
    """Add missing B-X edges using candidate pools.
    
    Parameters
    ----------
    subgraph : nx.Graph
        The cavity subgraph being built
    under_connected_b : List[Dict]
        List of B atoms with missing connections (from _diagnose_connectivity_issues)
    candidate_pools : Dict[str, Dict[str, List[Tuple[str, float]]]]
        Candidate X atoms for each B atom (from _build_x_candidate_pools)
        
    Returns
    -------
    int
        Number of edges added
    """
    edges_added = 0
    
    for b_data in under_connected_b:
        b_node_id = b_data['b_node_id']
        missing = b_data['missing']
        
        # Get candidate pool for this B atom
        pools = candidate_pools.get(b_node_id, {})
        
        for x_type, missing_count in missing.items():
            if missing_count == 0:
                continue
            
            # Get candidates for this X type
            candidates = pools.get(x_type, [])
            if _CONNECTIVITY_DEBUG and missing_count > 0:
                print(f"    [CONNECTIVITY_DEBUG] _add_missing: B{b_data['b_idx']} needs {missing_count} {x_type}, "
                      f"pool has {len(candidates)} candidates", file=sys.stderr)
            
            # Try to add missing connections
            added_for_this_type = 0
            for x_node_id, distance in candidates:
                if added_for_this_type >= missing_count:
                    break
                
                # Skip if already connected
                if subgraph.has_edge(b_node_id, x_node_id):
                    if _CONNECTIVITY_DEBUG and added_for_this_type == 0:
                        pass  # only log first skip per type below
                    continue
                
                # Check if this X atom is already fully connected
                # Equatorial X should have max 2 B connections
                # Axial X should have max 2 B connections (for interlayer bridging)
                # Terminal X should have max 1 B connection
                x_node_data = subgraph.nodes[x_node_id]
                b_neighbors = [n for n in subgraph.neighbors(x_node_id) 
                              if subgraph.nodes[n].get('is_B', False)]
                
                if x_type == 'equatorial':
                    max_b_connections = 2
                elif x_type == 'axial':
                    max_b_connections = 2  # Axial can bridge 2 B atoms
                else:  # terminal
                    max_b_connections = 1
                
                if len(b_neighbors) >= max_b_connections:
                    if _CONNECTIVITY_DEBUG and added_for_this_type == 0:
                        print(f"    [CONNECTIVITY_DEBUG] _add_missing: skip X {x_node_data.get('original_index', '?')} "
                              f"(node {x_node_id}): already has {len(b_neighbors)} B neighbors (max={max_b_connections})", file=sys.stderr)
                    continue  # This X is already fully connected
                
                # Add edge
                subgraph.add_edge(
                    b_node_id, x_node_id,
                    edge_type='bonded_to',
                    role='ligand',
                    geometry=x_type
                )
                edges_added += 1
                added_for_this_type += 1
                print(f"    ADDED: B{b_data['b_idx']} - {x_type} X (distance={distance:.3f} Å)", 
                      file=sys.stderr)
            
            if added_for_this_type < missing_count:
                print(f"    WARNING: Could not add all missing {x_type} connections for B{b_data['b_idx']} "
                      f"(added {added_for_this_type}/{missing_count})", file=sys.stderr)
                if _CONNECTIVITY_DEBUG:
                    if len(candidates) == 0:
                        print(f"    [CONNECTIVITY_DEBUG] Reason: {x_type} pool for this B has 0 candidates "
                              f"(B node_id={b_node_id})", file=sys.stderr)
                    else:
                        print(f"    [CONNECTIVITY_DEBUG] Reason: all {len(candidates)} candidates were either "
                              f"already connected to this B or their X node already has 2 B neighbors", file=sys.stderr)
    
    return edges_added


def _validate_bx_connectivity(
    subgraph: nx.Graph,
    b_node_instances: List[Tuple[Tuple[int, Tuple], str]],
    connectivity_rules: Dict[str, int],
    x_labels_in_cavity: set,
    atom_node_mapping: Dict[Tuple[int, Tuple], str]
) -> bool:
    """Validate that all B atoms have correct connectivity (read-only check).
    
    Validates:
    1. Each B atom has the correct number of connections by type (equatorial/terminal/axial)
    2. Each equatorial X atom has exactly 2 B connections
    3. Each terminal/axial X atom has exactly 1 B connection
    
    Parameters
    ----------
    subgraph : nx.Graph
        The cavity subgraph being validated
    b_node_instances : List[Tuple[Tuple[int, Tuple], str]]
        List of ((b_idx, b_img_label), b_node_id) tuples
    connectivity_rules : Dict[str, int]
        Expected connection counts by type (e.g., {'equatorial': 2, 'terminal': 1})
    x_labels_in_cavity : set
        Set of (x_idx, image_label) tuples for X atoms in cavity
    atom_node_mapping : Dict[Tuple[int, Tuple], str]
        Mapping from (atom_idx, image_label) to node_id
        
    Returns
    -------
    bool
        True if all connectivity is valid, False otherwise
    """
    if _CONNECTIVITY_DEBUG:
        print(f"    [CONNECTIVITY_DEBUG] _validate_bx_connectivity: rules={connectivity_rules}, n_B={len(b_node_instances)}, n_X_in_cavity={len(x_labels_in_cavity)}", file=sys.stderr)

    # Validate B atoms using EDGE GEOMETRY (not node properties)
    for (b_idx, b_img_label), b_node_id in b_node_instances:
        # Count current connections by type using edge geometry
        current_connections = {
            'equatorial': 0,
            'terminal': 0,
            'axial': 0
        }
        
        for x_neighbor in subgraph.neighbors(b_node_id):
            edge_data = subgraph.get_edge_data(b_node_id, x_neighbor)
            geometry = edge_data.get('geometry', None) if edge_data else None
            
            if geometry == 'equatorial':
                current_connections['equatorial'] += 1
            elif geometry == 'terminal':
                current_connections['terminal'] += 1
            elif geometry == 'axial':
                current_connections['axial'] += 1
        
        # Check against rules
        for x_type, required_count in connectivity_rules.items():
            if current_connections[x_type] != required_count:
                print(f"    VALIDATION FAILED: B{b_idx} has {current_connections[x_type]} {x_type} "
                      f"connections (expected {required_count})", file=sys.stderr)
                if _CONNECTIVITY_DEBUG:
                    print(f"    [CONNECTIVITY_DEBUG] B{b_idx} full counts: eq={current_connections['equatorial']} term={current_connections['terminal']} axial={current_connections['axial']}", file=sys.stderr)
                return False
    
    # Validate ALL X atoms in the subgraph using EDGE GEOMETRY
    # Count how many B connections each X has for each geometry type
    for node_id in subgraph.nodes():
        node_data = subgraph.nodes[node_id]
        
        # Skip non-X atoms
        if not node_data.get('is_X', False):
            continue
        
        x_idx = node_data.get('original_index', 'unknown')
        
        # Count B connections by edge geometry type
        b_connections_by_geom = {
            'equatorial': 0,
            'terminal': 0,
            'axial': 0
        }
        
        for b_neighbor in subgraph.neighbors(node_id):
            if not subgraph.nodes[b_neighbor].get('is_B', False):
                continue
            
            edge_data = subgraph.get_edge_data(b_neighbor, node_id)
            geometry = edge_data.get('geometry', None) if edge_data else None
            
            if geometry == 'equatorial':
                b_connections_by_geom['equatorial'] += 1
            elif geometry == 'terminal':
                b_connections_by_geom['terminal'] += 1
            elif geometry == 'axial':
                b_connections_by_geom['axial'] += 1
        
        # Validate based on geometry type
        # Equatorial edges: X should have exactly 2 B connections
        if b_connections_by_geom['equatorial'] > 0:
            if b_connections_by_geom['equatorial'] != 2:
                print(f"    VALIDATION FAILED: X{x_idx} (node {node_id}) has {b_connections_by_geom['equatorial']} "
                      f"equatorial B connections (expected 2)", file=sys.stderr)
                if _CONNECTIVITY_DEBUG:
                    print(f"    [CONNECTIVITY_DEBUG] X{x_idx} full: eq={b_connections_by_geom['equatorial']} term={b_connections_by_geom['terminal']} axial={b_connections_by_geom['axial']}", file=sys.stderr)
                return False
        
        # Terminal edges: X should have exactly 1 B connection
        if b_connections_by_geom['terminal'] > 0:
            if b_connections_by_geom['terminal'] != 1:
                print(f"    VALIDATION FAILED: X{x_idx} (node {node_id}) has {b_connections_by_geom['terminal']} "
                      f"terminal B connections (expected 1)", file=sys.stderr)
                if _CONNECTIVITY_DEBUG:
                    print(f"    [CONNECTIVITY_DEBUG] X{x_idx} full: eq={b_connections_by_geom['equatorial']} term={b_connections_by_geom['terminal']} axial={b_connections_by_geom['axial']}", file=sys.stderr)
                return False
        
        # Axial edges: X can have 1 or 2 B connections (for interlayer bridging)
        if b_connections_by_geom['axial'] > 0:
            if b_connections_by_geom['axial'] < 1 or b_connections_by_geom['axial'] > 2:
                print(f"    VALIDATION FAILED: X{x_idx} (node {node_id}) has {b_connections_by_geom['axial']} "
                      f"axial B connections (expected 1-2)", file=sys.stderr)
                if _CONNECTIVITY_DEBUG:
                    print(f"    [CONNECTIVITY_DEBUG] X{x_idx} full: eq={b_connections_by_geom['equatorial']} term={b_connections_by_geom['terminal']} axial={b_connections_by_geom['axial']}", file=sys.stderr)
                return False
    
    if _STRICT_BX_DISTANCE and not _validate_bx_distances(subgraph, b_node_instances, max_bond_length=4.0):
        return False
    
    return True


def _validate_bx_distances(
    subgraph: nx.Graph,
    b_node_instances: List[Tuple[Tuple[int, Tuple], str]],
    max_bond_length: float = 4.0
) -> bool:
    """Return False if any B-X edge length exceeds max_bond_length (wrong PBC image).
    Used when Q2D_CAVITY_STRICT_BX_DISTANCE=1 to fail validation and allow repair to fix.
    """
    for (_b_idx, _b_img), b_node_id in b_node_instances:
        b_pos = subgraph.nodes[b_node_id].get("pbc_position")
        if b_pos is None:
            continue
        b_pos = np.asarray(b_pos)
        for x_node_id in subgraph.neighbors(b_node_id):
            if not subgraph.nodes[x_node_id].get("is_X", False):
                continue
            x_pos = subgraph.nodes[x_node_id].get("pbc_position")
            if x_pos is None:
                continue
            x_pos = np.asarray(x_pos)
            dist = float(np.linalg.norm(x_pos - b_pos))
            if dist > max_bond_length:
                x_idx = subgraph.nodes[x_node_id].get("original_index", "?")
                edge_data = subgraph.get_edge_data(b_node_id, x_node_id) or {}
                geom = edge_data.get("geometry", "?")
                print(
                    f"    VALIDATION FAILED (strict distance): B{_b_idx}-X{x_idx} ({geom}) "
                    f"distance = {dist:.3f} Å (max {max_bond_length} Å).",
                    file=sys.stderr,
                )
                if _CONNECTIVITY_DEBUG:
                    print(f"    [CONNECTIVITY_DEBUG] Wrong PBC image; repair may remove and re-add.", file=sys.stderr)
                return False
    return True


def _warn_long_bx_edges(
    subgraph: nx.Graph,
    b_node_instances: List[Tuple[Tuple[int, Tuple], str]],
    max_bond_length: float = 4.0
) -> None:
    """Log a warning for any B-X edge whose length exceeds max_bond_length.
    Helps catch wrong PBC image assignments (correct count but far X position).
    """
    for (_b_idx, _b_img), b_node_id in b_node_instances:
        b_pos = subgraph.nodes[b_node_id].get("pbc_position")
        if b_pos is None:
            continue
        b_pos = np.asarray(b_pos)
        for x_node_id in subgraph.neighbors(b_node_id):
            if not subgraph.nodes[x_node_id].get("is_X", False):
                continue
            x_pos = subgraph.nodes[x_node_id].get("pbc_position")
            if x_pos is None:
                continue
            x_pos = np.asarray(x_pos)
            dist = float(np.linalg.norm(x_pos - b_pos))
            if dist > max_bond_length:
                x_idx = subgraph.nodes[x_node_id].get("original_index", "?")
                edge_data = subgraph.get_edge_data(b_node_id, x_node_id) or {}
                geom = edge_data.get("geometry", "?")
                print(
                    f"    WARNING: B{_b_idx}-X{x_idx} ({geom}) distance = {dist:.3f} Å "
                    f"(max {max_bond_length} Å). Check PBC image assignment.",
                    file=sys.stderr,
                )

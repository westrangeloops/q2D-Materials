"""Structure-level geometric features derived from the connectivity graph.

This module provides functions to calculate structure-level features that use
the entire structure graph rather than cavity-specific information. These features
characterize global properties of the material structure.

Coordinate System
-----------------
All atom positions in the graph are stored as Cartesian coordinates (x, y, z).
For non-orthogonal cells (angles ≠ 90°), Cartesian Z-coordinates can exceed
the c cell parameter because the c-vector is tilted. This module uses:
- Actual Z-extent from cell matrix (not c parameter) for distance calculations
- Fractional coordinates for PBC wrapping/unwrapping
- PBC-aware distance functions for all inter-atomic distances

Functions
---------
calculate_global_interplane_distance
    Calculate distance between terminal atom planes using layer information
"""

import numpy as np
import networkx as nx
from typing import Dict, Any, List, Tuple

# Helper functions to get data directly from graph (no external dependencies)
def _get_cell_matrix(graph: nx.Graph) -> np.ndarray:
    """Get cell matrix from graph metadata."""
    return graph.graph.get('cell_matrix', None)

def _get_structure_node(graph: nx.Graph) -> Dict:
    """Get structure node data from graph."""
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'structure':
            return dict(data)
    return None

def _get_molecule_atoms(graph: nx.Graph, molecule_node: str) -> list:
    """Get atom IDs belonging to a molecule node."""
    atom_ids = []
    for neighbor in graph.neighbors(molecule_node):
        edge_data = graph.get_edge_data(molecule_node, neighbor)
        if edge_data and edge_data.get('edge_type') == 'contains':
            node_data = graph.nodes.get(neighbor, {})
            if node_data.get('node_type') == 'atom':
                atom_ids.append(neighbor)
    return atom_ids


def _calculate_xy_area(cell: np.ndarray) -> float:
    """Calculate XY cross-sectional area from cell matrix.
    
    Calculates the area of the XY plane (a*b*sin(gamma)) where gamma is the angle
    between the a and b cell vectors. This handles non-orthogonal cells correctly.
    
    Parameters
    ----------
    cell : np.ndarray
        Cell matrix (3x3 array) where cell[0] is a vector, cell[1] is b vector
        
    Returns
    -------
    float
        XY area in Å²
    """
    a = np.linalg.norm(cell[0])  # Length of a vector
    b = np.linalg.norm(cell[1])  # Length of b vector
    
    # Calculate angle between a and b vectors
    cos_gamma = np.dot(cell[0], cell[1]) / (a * b)
    # Clamp to avoid numerical errors
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    sin_gamma = np.sqrt(1.0 - cos_gamma**2)
    
    # XY area
    xy_area = a * b * sin_gamma
    
    return float(xy_area)


def calculate_centroid(terminal_node_ids: List[str], all_node_data: Dict) -> np.ndarray:
    """Calculate centroid based on number of terminals: 1, 2, or 3+ (SVD).
    
    Parameters
    ----------
    terminal_node_ids : List[str]
        List of terminal node IDs
    all_node_data : Dict
        Dictionary of all node data from graph
        
    Returns
    -------
    np.ndarray
        Centroid position as [x, y, z]
    """
    positions = []
    for node_id in terminal_node_ids:
        node_data = all_node_data.get(node_id, {})
        positions.append(np.array([node_data['x'], node_data['y'], node_data['z']]))
    
    if len(positions) == 0:
        raise ValueError("No valid terminal positions found")
    
    positions = np.array(positions)
    n = len(positions)
    
    if n == 1:
        # Single atom: centroid = atom position
        return positions[0]
    elif n == 2:
        # Two atoms: centroid = midpoint
        return np.mean(positions, axis=0)
    else:  # n >= 3
        # Three or more atoms: use SVD to fit plane, centroid = mean
        centroid = np.mean(positions, axis=0)
        centered = positions - centroid
        
        # Use SVD to find the best-fit plane
        try:
            u, s, vh = np.linalg.svd(centered, full_matrices=False)
            # The plane normal is the last row of vh (smallest singular value)
            # But we only need the centroid for distance calculation
            return centroid
        except:
            # Fallback to simple mean if SVD fails
            return centroid


def _get_positions(node_ids: List[str], all_node_data: Dict) -> np.ndarray:
    """Get positions for a list of node IDs."""
    positions = []
    for node_id in node_ids:
        node_data = all_node_data.get(node_id, {})
        positions.append([node_data['x'], node_data['y'], node_data['z']])
    return np.array(positions)


def one_layer_distance(
    terminal_node_ids: List[str],
    b_atom_ids: List[str],
    all_node_data: Dict,
    cell_inv: np.ndarray,
    cell_c_perp: float,
    xy_area: float,
    debug: bool = False
) -> Dict[str, Any]:
    """Calculate slab thickness and interplane distance for single layer (n=1) structures.
    
    Parameters
    ----------
    terminal_node_ids : List[str]
        List of terminal node IDs (all from same layer)
    b_atom_ids : List[str]
        List of B-atom IDs for this layer
    all_node_data : Dict
        Dictionary of all node data from graph
    cell_inv : np.ndarray
        Inverse cell matrix for fractional coordinate conversion
    cell_c_perp : float
        Perpendicular cell c dimension
    xy_area : float
        XY cross-sectional area in Å²
    debug : bool
        If True, print debugging information
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with interplane_distance, slab_thickness, volumes, and related info
    """
    # Get positions for terminals and B-atoms
    terminal_positions = _get_positions(terminal_node_ids, all_node_data)
    b_positions = _get_positions(b_atom_ids, all_node_data)

    # We need to divide the terminal atoms in two, the one "below" and the one "above" the B-atoms
    # But this is not trivial as both X can be above or below due to the PBC.
    # So we reconstruct them in PBC-aware distances to measure from the B 

    # Convert to fractional coordinates
    terminal_frac = terminal_positions @ cell_inv
    terminal_frac_c = terminal_frac[:, 2]
    b_frac = b_positions @ cell_inv
    b_frac_c = b_frac[:, 2]

    # Use B-atoms mean as reference point for PBC-aware unwrapping
    b_mean_frac_c = float(np.mean(b_frac_c))

    # Unwrap each terminal relative to B-atoms reference using minimum image convention
    # This gives us the signed distance from B to terminal in fractional space
    terminal_unwrapped_frac_c = []
    for t_frac_c in terminal_frac_c:
        # Calculate difference from B reference
        delta = t_frac_c - b_mean_frac_c
        # Apply minimum image convention: wrap to [-0.5, 0.5] range
        delta_wrapped = delta - np.round(delta)
        # Unwrapped position relative to B reference
        terminal_unwrapped_frac_c.append(b_mean_frac_c + delta_wrapped)

    terminal_unwrapped_frac_c = np.array(terminal_unwrapped_frac_c)

    # For n=1, group terminals into two groups based on their unwrapped positions
    # Use median of unwrapped positions as the dividing line to ensure balanced groups
    median_unwrapped = float(np.median(terminal_unwrapped_frac_c))
    
    terminals_below = []
    terminals_above = []
    for i, unwrapped_frac_c in enumerate(terminal_unwrapped_frac_c):
        if unwrapped_frac_c < median_unwrapped:
            terminals_below.append(terminal_node_ids[i])
        else:
            terminals_above.append(terminal_node_ids[i])

    # Calculate two centroids: one for terminals below B and one for terminals above B
    if len(terminals_below) == 0 or len(terminals_above) == 0:
        raise ValueError(f"Need terminals both below and above B-atoms, found below={len(terminals_below)}, above={len(terminals_above)}")
    
    centroid_below = calculate_centroid(terminals_below, all_node_data)
    centroid_above = calculate_centroid(terminals_above, all_node_data)
    
    # Convert centroids to fractional coordinates
    centroid_below_frac = centroid_below @ cell_inv
    centroid_above_frac = centroid_above @ cell_inv
    
    # Unwrap centroids relative to B reference (same as terminals)
    delta_below = centroid_below_frac[2] - b_mean_frac_c
    delta_above = centroid_above_frac[2] - b_mean_frac_c
    delta_below_wrapped = delta_below - np.round(delta_below)
    delta_above_wrapped = delta_above - np.round(delta_above)
    
    centroid_below_unwrapped_frac_c = b_mean_frac_c + delta_below_wrapped
    centroid_above_unwrapped_frac_c = b_mean_frac_c + delta_above_wrapped
    
    # Distance between centroids in fractional space (already unwrapped)
    # For n=1: This is the SLAB (octahedral layer between terminals)
    slab_thickness_frac = abs(centroid_above_unwrapped_frac_c - centroid_below_unwrapped_frac_c)
    
    # Convert to perpendicular distance
    slab_thickness = slab_thickness_frac * cell_c_perp
    
    # Interplane distance is the remainder (space between terminals where molecules are)
    interplane_distance = cell_c_perp - slab_thickness
    
    if debug:
        print(f"  Single layer (n=1): {len(terminals_below)} terminals below, {len(terminals_above)} terminals above, {len(b_atom_ids)} B-atoms")
        print(f"  SLAB: {slab_thickness:.3f} Å (distance between centroids)")
        print(f"  INTERPLANE: {interplane_distance:.3f} Å (cell_c - slab)")
        print(f"  Cell c perpendicular: {cell_c_perp:.3f} Å")
    
    # Get positions for return values
    below_positions = _get_positions(terminals_below, all_node_data)
    above_positions = _get_positions(terminals_above, all_node_data)
    
    # Validation print
    sum_value = slab_thickness + interplane_distance
    difference = abs(sum_value - cell_c_perp)
    print(f"  INTERPLANE: {interplane_distance:.6f} + SLAB: {slab_thickness:.6f} = {sum_value:.6f}, expected: {cell_c_perp:.6f}")
    if difference > 0.01:
        print(f"  ⚠ WARNING: Difference of {difference:.6f} Å exceeds tolerance")
    
    # Calculate z ranges from unwrapped positions
    min_unwrapped_frac_c = float(np.min(terminal_unwrapped_frac_c))
    max_unwrapped_frac_c = float(np.max(terminal_unwrapped_frac_c))
    min_terminal_z = min_unwrapped_frac_c * cell_c_perp
    max_terminal_z = max_unwrapped_frac_c * cell_c_perp
    
    # Calculate volumes
    interplane_volume = xy_area * interplane_distance
    slab_volume = xy_area * slab_thickness
    
    return {
        'interplane_distance': float(interplane_distance),
        'slab_thickness': float(slab_thickness),
        'interplane_volume': float(interplane_volume),
        'slab_volume': float(slab_volume),
        'top_centroid': centroid_above.tolist(),
        'bottom_centroid': centroid_below.tolist(),
        'top_atom_count': len(above_positions),
        'bottom_atom_count': len(below_positions),
        'molecule_mean_z': None,
        'molecule_centroid': None,
        'top_z_range': [float(np.min(above_positions[:, 2])), float(np.max(above_positions[:, 2]))],
        'bottom_z_range': [float(np.min(below_positions[:, 2])), float(np.max(below_positions[:, 2]))],
        'terminal_z_range': [float(min_terminal_z), float(max_terminal_z)],
        'spacer_info': [],
    }


def multilayer_distance(
    terminals_list_1: List[str],
    terminals_list_2: List[str],
    b_atom_ids_list_1: List[str],
    b_atom_ids_list_2: List[str],
    all_node_data: Dict,
    cell_inv: np.ndarray,
    cell_c_perp: float,
    cell_volume: float,
    xy_area: float,
    debug: bool = False
) -> Dict[str, Any]:
    """Calculate slab thickness and interplane distance for multi-layer structures.
    
    Parameters
    ----------
    terminals_list_1 : List[str]
        First list of terminal node IDs
    terminals_list_2 : List[str]
        Second list of terminal node IDs
    b_atom_ids_list_1 : List[str]
        B-atom IDs for first layer
    b_atom_ids_list_2 : List[str]
        B-atom IDs for second layer
    all_node_data : Dict
        Dictionary of all node data from graph
    cell_inv : np.ndarray
        Inverse cell matrix for fractional coordinate conversion
    cell_c_perp : float
        Perpendicular cell c dimension
    cell_volume : float
        Cell volume for debug output
    xy_area : float
        XY cross-sectional area in Å²
    debug : bool
        If True, print debugging information
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with interplane_distance, slab_thickness, volumes, and related info
    """
    # Calculate centroids for each layer
    centroid_1 = calculate_centroid(terminals_list_1, all_node_data)
    centroid_2 = calculate_centroid(terminals_list_2, all_node_data)
    
    # Identify X1 (lower Z) and X2 (higher Z) layers by centroid Z-coordinate
    if centroid_1[2] < centroid_2[2]:
        terminals_x1 = terminals_list_1
        terminals_x2 = terminals_list_2
        b_atom_ids_x1 = b_atom_ids_list_1
        b_atom_ids_x2 = b_atom_ids_list_2
        centroid_x1 = centroid_1
        centroid_x2 = centroid_2
    else:
        terminals_x1 = terminals_list_2
        terminals_x2 = terminals_list_1
        b_atom_ids_x1 = b_atom_ids_list_2
        b_atom_ids_x2 = b_atom_ids_list_1
        centroid_x1 = centroid_2
        centroid_x2 = centroid_1
    
    # Combine all terminals and B-atoms for zone calculation
    all_terminal_atoms = terminals_list_1 + terminals_list_2
    all_b_atom_ids = b_atom_ids_list_1 + b_atom_ids_list_2
    
    # Get positions for all terminals and B-atoms
    all_terminal_positions = _get_positions(all_terminal_atoms, all_node_data)
    b_positions = _get_positions(all_b_atom_ids, all_node_data)
    
    # Convert to fractional c-coordinates
    terminal_frac = all_terminal_positions @ cell_inv
    terminal_frac_c = terminal_frac[:, 2]
    b_frac = b_positions @ cell_inv
    b_frac_c = b_frac[:, 2]
    
    # Find zone boundaries (min and max terminal frac_c)
    min_terminal_frac = float(np.min(terminal_frac_c))
    max_terminal_frac = float(np.max(terminal_frac_c))
    
    # Convert to perpendicular distances along c
    min_terminal_z = min_terminal_frac * cell_c_perp
    max_terminal_z = max_terminal_frac * cell_c_perp
    
    if debug:
        term_info = []
        for node_id in all_terminal_atoms[:10]:
            node_data = all_node_data.get(node_id, {})
            symbol = node_data.get('symbol', '?')
            vasp_idx = node_data.get('vasp_index', '?')
            term_info.append(f"{symbol}{vasp_idx}")
        print(f"  Terminal atoms: {', '.join(term_info)}{'...' if len(all_terminal_atoms) > 10 else ''} ({len(all_terminal_atoms)} total)")
        print(f"  Terminal frac_c range: [{min_terminal_frac:.4f}, {max_terminal_frac:.4f}]")
        print(f"  B-atoms frac_c range: [{np.min(b_frac_c):.4f}, {np.max(b_frac_c):.4f}]")
    
    # Calculate 3 zone distances (using perpendicular distance)
    zone_0_to_min = min_terminal_frac * cell_c_perp  # [0 → min_X]
    zone_min_to_max = (max_terminal_frac - min_terminal_frac) * cell_c_perp  # [min_X → max_X]
    zone_max_to_end = (1.0 - max_terminal_frac) * cell_c_perp  # [max_X → cell_c]
    
    # Determine which zone contains B-atoms (that's the slab)
    b_mean_frac_c = float(np.mean(b_frac_c))
    b_in_central_zone = (b_mean_frac_c >= min_terminal_frac) and (b_mean_frac_c <= max_terminal_frac)
    
    if b_in_central_zone:
        # B-atoms are between terminals → central zone is interplane, boundaries are slab
        interplane_distance = zone_min_to_max
        slab_thickness = zone_0_to_min + zone_max_to_end
        if debug:
            print(f"  B-atoms in central zone → INTERPLANE: {interplane_distance:.3f} Å (zone [{min_terminal_frac:.3f}→{max_terminal_frac:.3f}])")
            print(f"  SLAB: {slab_thickness:.3f} Å (boundary zones [0→{min_terminal_frac:.3f}] + [{max_terminal_frac:.3f}→1])")
    else:
        # B-atoms are in boundaries → boundaries are interplane, central is slab
        interplane_distance = zone_0_to_min + zone_max_to_end
        slab_thickness = zone_min_to_max
        if debug:
            print(f"  B-atoms in boundary zones → INTERPLANE: {interplane_distance:.3f} Å (boundary zones [0→{min_terminal_frac:.3f}] + [{max_terminal_frac:.3f}→1])")
            print(f"  SLAB: {slab_thickness:.3f} Å (central zone [{min_terminal_frac:.3f}→{max_terminal_frac:.3f}])")
    
    if debug:
        print(f"  Zones: [0→{min_terminal_frac:.3f}]={zone_0_to_min:.3f}Å, [{min_terminal_frac:.3f}→{max_terminal_frac:.3f}]={zone_min_to_max:.3f}Å, [{max_terminal_frac:.3f}→1]={zone_max_to_end:.3f}Å")
        print(f"  Cell c perpendicular distance: {cell_c_perp:.6f} Å (volume={cell_volume:.3f} Å³)")
    
    # Get positions for return values
    bottom_positions = _get_positions(terminals_x1, all_node_data)
    top_positions = _get_positions(terminals_x2, all_node_data)
    
    # Validation: print the sum check
    sum_value = slab_thickness + interplane_distance
    difference = abs(sum_value - cell_c_perp)
    print(f"  INTERPLANE: {interplane_distance:.6f} + SLAB: {slab_thickness:.6f} = {sum_value:.6f}, expected: {cell_c_perp:.6f}")
    if difference > 0.01:
        print(f"  ⚠ WARNING: Difference of {difference:.6f} Å exceeds tolerance")
    
    # Calculate volumes
    interplane_volume = xy_area * interplane_distance
    slab_volume = xy_area * slab_thickness
    
    return {
        'interplane_distance': float(interplane_distance),
        'slab_thickness': float(slab_thickness),
        'interplane_volume': float(interplane_volume),
        'slab_volume': float(slab_volume),
        'top_centroid': centroid_x2.tolist(),
        'bottom_centroid': centroid_x1.tolist(),
        'top_atom_count': len(top_positions),
        'bottom_atom_count': len(bottom_positions),
        'molecule_mean_z': None,
        'molecule_centroid': None,
        'top_z_range': [float(np.min(top_positions[:, 2])), float(np.max(top_positions[:, 2]))],
        'bottom_z_range': [float(np.min(bottom_positions[:, 2])), float(np.max(bottom_positions[:, 2]))],
        'terminal_z_range': [float(min_terminal_z), float(max_terminal_z)],
        'spacer_info': [],
    }


def calculate_global_interplane_distance(graph: nx.Graph, debug: bool = False) -> Dict[str, Any]:
    """Calculate slab thickness and interplane distance using terminal atoms and B-atoms.
    
    Simple algorithm using 3 zones:
    1. Terminal atoms (is_terminal=True) define zone boundaries: min_X and max_X
    2. This creates 3 zones: [0 → min_X], [min_X → max_X], [max_X → cell_c]
    3. B-atoms (octahedral centers) determine which zone is the slab
    4. The zone containing B-atoms = slab_thickness
    5. The remaining zone(s) = interplane_distance
    
    Parameters
    ----------
    graph : nx.Graph
        Structure graph with atoms and cell matrix.
    debug : bool
        If True, print debugging information
        
    Returns
    -------
    dict with 'interplane_distance', 'slab_thickness', and related info
    """
    # Get all node data
    all_node_data = dict(graph.nodes(data=True))
    
    # Get cell matrix
    cell = _get_cell_matrix(graph)
    if cell is None:
        raise ValueError("Cell matrix not found in graph.")
    
    # Calculate cell volume (scalar triple product)
    cell_volume = abs(np.dot(cell[0], np.cross(cell[1], cell[2])))
    
    # Calculate the cross product of a and b vectors (normal to the ab-plane)
    ab_cross = np.cross(cell[0], cell[1])
    ab_cross_norm = np.linalg.norm(ab_cross)
    
    # Perpendicular distance along c (the "true" c-spacing)
    cell_c_perp = cell_volume / ab_cross_norm
    
    # Calculate XY cross-sectional area
    xy_area = _calculate_xy_area(cell)
    
    # For converting positions to fractional coordinates
    cell_inv = np.linalg.inv(cell)
    
    # Step 1: Get ALL terminal atoms (is_terminal=True)
    # Filter to only atom nodes (exclude layer nodes and other structural nodes)
    attr_dict = nx.get_node_attributes(graph, "is_terminal")
    terminal_atoms = [node for node, is_terminal in attr_dict.items() 
                      if is_terminal and node.startswith('atom_')]
    
    # Step 2: Group terminals by layer (layer_0 vs layer_n)
    list_x_terminal_atoms_layer_0 = []
    list_x_terminal_atoms_layer_n = []
    list_b_layer_0 = []
    list_b_layer_n = []
    for x_terminal_atom in terminal_atoms:
        # Find B-atom: terminal is bonded to B, and B is connected to octahedra with role='center'
        b_attached_to_terminal = None
        octahedra_node = None
        
        for neighbor in graph.neighbors(x_terminal_atom):
            edge_data = graph.get_edge_data(x_terminal_atom, neighbor)
            if edge_data and edge_data.get('edge_type') == 'bonded_to':
                # Check if this neighbor is a B-atom (connected to octahedra with role='center')
                for oct_neighbor in graph.neighbors(neighbor):
                    oct_edge_data = graph.get_edge_data(neighbor, oct_neighbor)
                    if oct_edge_data and oct_edge_data.get('edge_type') == 'contains' and oct_edge_data.get('role') == 'center':
                        b_attached_to_terminal = neighbor
                        octahedra_node = oct_neighbor
                        break
                if b_attached_to_terminal is not None:
                    break
        
        if b_attached_to_terminal is None:
            continue  # Skip if no B-atom found
        
        # Octahedra are connected to both a layer node and the center B atom
        # We need to find the layer node specifically (not just take the first neighbor)
        layer_node_attached_to_octahedra = None
        for neighbor in graph.neighbors(octahedra_node):
            neighbor_data = graph.nodes.get(neighbor, {})
            if neighbor_data.get('node_type') == 'layer':
                layer_node_attached_to_octahedra = neighbor
                break
        
        if layer_node_attached_to_octahedra is None:
            continue  # Skip if no layer found
        
        # Group terminals based on layer
        if layer_node_attached_to_octahedra.startswith('layer_0'):
            list_x_terminal_atoms_layer_0.append(x_terminal_atom)
            list_b_layer_0.append(b_attached_to_terminal)
        else:
            list_x_terminal_atoms_layer_n.append(x_terminal_atom)
            list_b_layer_n.append(b_attached_to_terminal)
    
    # Step 3: Check if single layer (n=1) or multi-layer
    if len(list_x_terminal_atoms_layer_n) == 0: # n=1 case
        # Single layer case: use all terminals
        return one_layer_distance(
            list_x_terminal_atoms_layer_0,
            list_b_layer_0,
            all_node_data,
            cell_inv,
            cell_c_perp,
            xy_area,
            debug
        )
    
    # Multi-layer case: need terminals in both layers
    return multilayer_distance(
        list_x_terminal_atoms_layer_0,
        list_x_terminal_atoms_layer_n,
        list_b_layer_0,
        list_b_layer_n,
        all_node_data,
        cell_inv,
        cell_c_perp,
        cell_volume,
        xy_area,
        debug
    )

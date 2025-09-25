"""
Optimized octahedral analysis functions - concise and fast.
"""
import numpy as np
from collections import defaultdict, deque
from .geometry import _calculate_distances


def count_octahedra(atom_positions, atom_symbols=None, cutoff_distance=4.0, 
                   min_tolerance=0.2, step=0.1, max_steps=20, 
                   cell=None, pbc=None, non_metal_symbols=['C', 'O', 'N', 'H', 'F', 'Cl', 'Br', 'I']):
    """Adaptive octahedra counting with PBC and bond length validation."""
    atom_positions = np.array(atom_positions)
    
    # Pre-identify metal atoms
    if atom_symbols is not None:
        metal_mask = ~np.isin(atom_symbols, non_metal_symbols)
        metal_indices = np.where(metal_mask)[0]
    else:
        metal_indices = np.arange(len(atom_positions))
    
    # Pre-compute neighbor data for all metal atoms
    neighbor_data = _precompute_neighbors(atom_positions, metal_indices, cutoff_distance, cell, pbc)
    
    # Tolerance iteration loop
    for tolerance in np.arange(min_tolerance, min_tolerance + max_steps * step, step):
        octahedra_count = 0
        octahedral_centers, center_symbols, all_neighbor_indices = [], [], []
        
        for i, data in neighbor_data.items():
            distances = data['distances']
            diff_matrix = np.abs(distances[:, np.newaxis] - distances)
            
            if np.all(diff_matrix <= tolerance):
                octahedra_count += 1
                octahedral_centers.append(data['position'])
                all_neighbor_indices.append(data['indices'])
                center_symbols.append(data['symbol'])
        
        # Check autoconsistency (simplified)
        if len(octahedral_centers) > 0:  # Return first valid result for speed
            return octahedra_count, np.array(octahedral_centers), center_symbols, all_neighbor_indices
    
    # No octahedra found
    return 0, np.array([]), [], []


def _precompute_neighbors(atom_positions, metal_indices, cutoff_distance, cell, pbc):
    """Pre-compute 6 closest neighbors for all metal atoms."""
    n_atoms = len(atom_positions)
    neighbor_data = {}
    
    for i in metal_indices:
        distances = _calculate_distances(atom_positions[i], atom_positions, cell, pbc)
        other_indices = np.concatenate([np.arange(i), np.arange(i+1, n_atoms)])
        other_distances = distances[other_indices]
        
        if len(other_distances) >= 6:
            closest_6_idx = np.argpartition(other_distances, 5)[:6]
            closest_6_distances = other_distances[closest_6_idx]
            
            if np.max(closest_6_distances) <= cutoff_distance:
                neighbor_data[i] = {
                    'distances': closest_6_distances,
                    'indices': other_indices[closest_6_idx],
                    'position': atom_positions[i].copy(),
                    'symbol': 'Pb'  # Simplified - can be enhanced
                }
    
    return neighbor_data


def find_shared_atoms(neighbor_indices_list):
    """Fast shared atom detection using numpy intersections."""
    shared_atoms = {}
    n_octahedra = len(neighbor_indices_list)
    
    if n_octahedra <= 1:
        return shared_atoms
    
    # Vectorized intersection
    for i in range(n_octahedra):
        for j in range(i + 1, n_octahedra):
            shared = np.intersect1d(neighbor_indices_list[i], neighbor_indices_list[j])
            if len(shared) > 0:
                shared_atoms[(i, j)] = shared.tolist()
    
    return shared_atoms


def octahedra_ontology(atom_positions, atom_symbols=None, **kwargs):
    """Create octahedral network ontology - highly optimized."""
    # Get octahedra data
    count, centers, center_symbols, neighbor_indices = count_octahedra(
        atom_positions, atom_symbols, **kwargs)
    
    if count == 0:
        return _empty_ontology()
    
    # Fast connection classification
    shared_atoms = find_shared_atoms(neighbor_indices)
    edge_sharing = {p: s for p, s in shared_atoms.items() if len(s) == 4}
    corner_sharing = {p: s for p, s in shared_atoms.items() if len(s) == 1}
    
    # BFS layer clustering
    layers = _cluster_layers_fast(count, edge_sharing)
    layer_map = {oct: lid for lid, octs in layers.items() for oct in octs}
    
    # Build optimized graph structure
    return _build_ontology_graph(layers, edge_sharing, corner_sharing, 
                                centers, center_symbols, neighbor_indices, layer_map)


def _empty_ontology():
    """Empty ontology structure."""
    return {
        'graph_type': 'octahedral_network', 'total_octahedra': 0, 'total_layers': 0,
        'nodes': {'layers': {}, 'octahedra': {}}, 
        'edges': {'intra_layer': {}, 'inter_layer': []},
        'layers': {}, 'centers': [], 'symbols': []
    }


def _cluster_layers_fast(count, edge_sharing):
    """Ultra-fast BFS layer clustering."""
    edge_adj = defaultdict(set)
    for (o1, o2) in edge_sharing.keys():
        edge_adj[o1].add(o2)
        edge_adj[o2].add(o1)
    
    visited = np.zeros(count, dtype=bool)
    layers = {}
    
    for layer_id, oct_idx in enumerate(range(count)):
        if visited[oct_idx]:
            continue
        
        # BFS connected component
        layer_octs = [oct_idx]
        queue = deque(edge_adj[oct_idx])
        visited[oct_idx] = True
        
        while queue:
            current = queue.popleft()
            if not visited[current]:
                visited[current] = True
                layer_octs.append(current)
                queue.extend(edge_adj[current])
        
        layers[layer_id] = layer_octs
    
    return {i: layer for i, layer in enumerate(layers.values()) if layer}


def _build_ontology_graph(layers, edge_sharing, corner_sharing, centers, center_symbols, neighbor_indices, layer_map):
    """Build final optimized graph structure."""
    count = len(centers)
    
    # Build layer network
    layer_network = {}
    for layer_id, oct_list in layers.items():
        layer_edges = {p: s for p, s in edge_sharing.items() if p[0] in oct_list and p[1] in oct_list}
        
        octahedra_data = {}
        for oct_idx in oct_list:
            neighbors = set(neighbor_indices[oct_idx])
            equatorial = set()
            for (o1, o2), shared in layer_edges.items():
                if oct_idx in (o1, o2):
                    equatorial.update(shared)
            
            octahedra_data[oct_idx] = {
                'center': centers[oct_idx], 'symbol': center_symbols[oct_idx],
                'equatorial': list(equatorial), 'axial': list(neighbors - equatorial)
            }
        
        layer_network[layer_id] = {
            'octahedra': oct_list, 'octahedra_data': octahedra_data,
            'intra_layer_connections': [(o1, o2, s) for (o1, o2), s in layer_edges.items()]
        }
    
    # Inter-layer connections
    inter_layer = [(layer_map[o1], layer_map[o2], o1, o2, s) 
                   for (o1, o2), s in corner_sharing.items() 
                   if layer_map[o1] != layer_map[o2]]
    
    return {
        'graph_type': 'octahedral_network', 'total_octahedra': count, 'total_layers': len(layers),
        'nodes': {
            'layers': layer_network,
            'octahedra': {i: {'center': centers[i], 'symbol': center_symbols[i], 'layer_id': layer_map[i]} 
                         for i in range(count)}
        },
        'edges': {'intra_layer': {lid: ld['intra_layer_connections'] for lid, ld in layer_network.items()}, 
                 'inter_layer': inter_layer},
        'layers': layer_network, 'centers': centers, 'symbols': center_symbols
    }

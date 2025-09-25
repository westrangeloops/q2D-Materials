"""
Simple geometry function for distance calculations.
"""

import numpy as np
from collections import defaultdict, deque

def _calculate_distances(reference_atom, atom_list, cell, pbc=None):
    """
    Calculate PBC-aware distances between one reference atom and multiple atoms.
    Optimized for crystal structures - always uses periodic boundary conditions.
    
    Parameters:
    reference_atom: [x, y, z] coordinates of the reference atom
    atom_list: list of [x, y, z] coordinates of atoms to calculate distances to
    cell: 3x3 array of unit cell vectors (required)
    pbc: list of 3 booleans for periodic boundary conditions (optional, defaults to [True, True, True])
    
    Returns:
    numpy array: PBC-aware distances from reference_atom to each atom in atom_list
    """
    # Convert inputs to numpy arrays for efficiency
    ref_coord = np.asarray(reference_atom, dtype=np.float64)
    atom_coords = np.asarray(atom_list, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    
    # Always use PBC for crystal structures
    inv_cell = np.linalg.inv(cell)
    
    # Calculate all differences at once (vectorized)
    diff_vectors = atom_coords - ref_coord  # Shape: (n_atoms, 3)
    
    # Apply PBC to all vectors at once
    diff_cell_coords = diff_vectors @ inv_cell.T  # More efficient matrix multiplication
    diff_cell_coords = diff_cell_coords - np.round(diff_cell_coords)
    pbc_diff_vectors = diff_cell_coords @ cell  # Shape: (n_atoms, 3)
    
    # Calculate distances for all atoms at once
    distances = np.linalg.norm(pbc_diff_vectors, axis=1)
    
    return distances
    

def _count_octahedra(atom_positions, atom_symbols=None, cutoff_distance=4.0, 
                    min_tolerance=0.2, step=0.1, max_steps=20, 
                    cell=None, non_metal_symbols=['C', 'O', 'N', 'H', 'F', 'Cl', 'Br', 'I']):
    """
    Adaptive octahedra counting with automatic tolerance optimization.
    
    Parameters:
    atom_positions: list of [x, y, z] coordinates
    atom_symbols: list of atomic symbols (optional, for filtering central atoms)
    cutoff_distance: maximum distance for neighbors
    min_tolerance: starting bond length tolerance (Å)
    step: increment step for tolerance optimization
    max_steps: maximum steps before stopping
    cell: 3x3 array of unit cell vectors (required for PBC)
    pbc: list of 3 booleans for periodic boundary conditions (required for PBC)
    
    Returns:
    tuple: (count, centers_positions, center_symbols, neighbor_indices) where:
        - count: number of octahedra
        - centers_positions: numpy array of shape (n_octahedra, 3) 
        - center_symbols: list of element symbols for octahedral centers
        - neighbor_indices: list of lists, each containing 6 atom indices that form the octahedron
    """
    import numpy as np
    
    atom_positions = np.array(atom_positions)
    
    # History to track autoconsistency
    count_history = []
    tolerance = min_tolerance
    
    # Pre-identify metal atoms to avoid repeated filtering
    if atom_symbols is not None:
        non_metal_mask = np.array([sym in non_metal_symbols for sym in atom_symbols])
        metal_indices = np.where(~non_metal_mask)[0]
    else:
        metal_indices = np.arange(len(atom_positions))
    
    # PRE-COMPUTE: Calculate all distances for all metal atoms once
    n_atoms = len(atom_positions)
    all_distances = {}
    all_neighbor_data = {}
    
    for i in metal_indices:
        # Calculate PBC-aware distances to all other atoms
        distances = _calculate_distances(atom_positions[i], atom_positions, cell)
        
        # Get indices of all other atoms (excluding self) - ultra optimized
        other_indices = np.concatenate([np.arange(i), np.arange(i+1, n_atoms)])
        other_distances = distances[other_indices]
        
        if len(other_distances) >= 6:
            # Get indices of 6 closest neighbors
            closest_6_idx = np.argpartition(other_distances, 5)[:6]
            closest_6_distances = other_distances[closest_6_idx]
            closest_6_atom_indices = other_indices[closest_6_idx]
            
            # Check if the 6th closest neighbor is within reasonable octahedral range
            if np.max(closest_6_distances) <= cutoff_distance:
                all_neighbor_data[i] = {
                    'distances': closest_6_distances,
                    'indices': closest_6_atom_indices,
                    'position': atom_positions[i].copy(),
                    'symbol': atom_symbols[i] if atom_symbols is not None else 'Unknown'
                }
    
    # MAIN TOLERANCE ITERATION LOOP - now much faster
    for step_num in range(max_steps):
        octahedra_count = 0
        octahedral_centers = []
        center_symbols = []
        all_neighbor_indices = []
        center_atom_indices = []  # Track actual atom indices
        
        # Process pre-computed data
        for i, data in all_neighbor_data.items():
            # Vectorized pairwise difference check
            closest_6_distances = data['distances']
            diff_matrix = np.abs(closest_6_distances[:, np.newaxis] - closest_6_distances)
            
            # Check if all pairwise differences are within tolerance
            if np.all(diff_matrix <= tolerance):
                octahedra_count += 1
                octahedral_centers.append(data['position'])
                all_neighbor_indices.append(data['indices'].tolist())
                center_symbols.append(data['symbol'])
                center_atom_indices.append(i)  # Store actual metal atom index
        
        # Add to history
        count_history.append(octahedra_count)
        
        # Check for autoconsistency (last 3 values are the same)
        if len(count_history) >= 3:
            if count_history[-1] == count_history[-2] == count_history[-3]:
                # Found stable solution
                return octahedra_count, np.array(octahedral_centers), center_symbols, all_neighbor_indices, center_atom_indices
        
        # Increment tolerance for next iteration
        tolerance += step
    
    # If no convergence, return the last result with warning
    print("Warning: No convergence found after maximum steps. Returning last result.")
    return octahedra_count, np.array(octahedral_centers), center_symbols, all_neighbor_indices, center_atom_indices


def find_shared_atoms(neighbor_indices_list):
    """
    Find which octahedra share atoms - optimized with numpy.
    
    Parameters:
    neighbor_indices_list: list of lists, each containing 6 atom indices for an octahedron
    
    Returns:
    dict: mapping of octahedra pairs to their shared atom indices
          e.g., {(0,1): [2, 5], (0,2): [3]} means octahedra 0&1 share atoms 2&5
    """
    import numpy as np
    
    shared_atoms = {}
    n_octahedra = len(neighbor_indices_list)
    
    if n_octahedra <= 1:
        return shared_atoms
    
    # Convert to numpy arrays for faster operations
    neighbor_arrays = [np.array(indices, dtype=np.int32) for indices in neighbor_indices_list]
    
    # Vectorized intersection finding
    for i in range(n_octahedra):
        for j in range(i + 1, n_octahedra):
            # Fast numpy intersection
            shared = np.intersect1d(neighbor_arrays[i], neighbor_arrays[j])
            if len(shared) > 0:
                shared_atoms[(i, j)] = shared.tolist()
    
    return shared_atoms


def _octahedra_ontology(atom_positions, atom_symbols=None, cutoff_distance=4.0, 
                       min_tolerance=0.2, step=0.1, max_steps=20, 
                       cell=None, non_metal_symbols=['C', 'O', 'N', 'H', 'F', 'Cl', 'Br', 'I']):
    """
    Create a clean entity-relationship graph for octahedral networks.
    
    Returns a graph with well-defined entities (atoms, octahedra, layers) and their relationships.
    No redundant information - each entity is defined once with clear references.
    
    Parameters:
    Same as _count_octahedra
    
    Returns:
    dict: Clean entity-relationship graph:
        {
            'entities': {
                'atoms': {atom_id: {'index': int, 'symbol': str, 'position': [x,y,z]}},
                'octahedra': {oct_id: {'center_atom': atom_id, 'neighbors': [atom_ids], 'layer': layer_id}},
                'layers': {layer_id: {'octahedra': [oct_ids]}}
            },
            'relationships': {
                'edge_sharing': [(oct1, oct2, [shared_atom_ids])],
                'corner_sharing': [(oct1, oct2, [shared_atom_ids])],
                'layer_membership': {oct_id: layer_id}
            },
            'metadata': {
                'total_atoms': int,
                'total_octahedra': int, 
                'total_layers': int
            }
        }
    """
    
    # First, get all octahedra data using the optimized function
    count, centers, center_symbols, neighbor_indices, center_atom_indices = _count_octahedra(
        atom_positions, atom_symbols, cutoff_distance, min_tolerance, 
        step, max_steps, cell, non_metal_symbols
    )
    
    if count == 0:
        return {
            'entities': {'atoms': {}, 'octahedra': {}, 'layers': {}},
            'relationships': {'edge_sharing': [], 'corner_sharing': [], 'layer_membership': {}},
            'metadata': {'total_atoms': len(atom_positions), 'total_octahedra': 0, 'total_layers': 0}
        }
    
    # ENTITIES: Define each entity type once with no redundancy
    
    # 1. ATOMS - All unique atoms involved in octahedra
    atoms_in_octahedra = set()
    for oct_neighbors in neighbor_indices:
        atoms_in_octahedra.update(oct_neighbors)
    for center_idx in center_atom_indices:
        atoms_in_octahedra.add(center_idx)
    
    atoms_entity = {}
    for atom_idx in atoms_in_octahedra:
        atoms_entity[atom_idx] = {
            'index': int(atom_idx) + 1,  # 1-based for VASP compatibility
            'symbol': atom_symbols[atom_idx] if atom_symbols else 'Unknown',
            'position': atom_positions[atom_idx].tolist()
        }
    
    # 2. OCTAHEDRA - Define once with references to atoms
    octahedra_entity = {}
    for oct_id in range(count):
        octahedra_entity[oct_id] = {
            'center_atom': center_atom_indices[oct_id],
            'neighbors': neighbor_indices[oct_id],
            'layer': None  # Will be set during layer clustering
        }
    
    # RELATIONSHIPS: Define connections between entities
    
    # Get shared atoms between octahedra
    shared_atoms = find_shared_atoms(neighbor_indices)
    
    # Classify sharing relationships
    edge_sharing_relationships = []
    corner_sharing_relationships = []
    
    for (oct1, oct2), shared in shared_atoms.items():
        num_shared = len(shared)
        if num_shared >= 3:  # Edge-sharing (same layer)
            edge_sharing_relationships.append((oct1, oct2, shared))
        elif num_shared <= 2:  # Corner-sharing (inter-layer)
            corner_sharing_relationships.append((oct1, oct2, shared))
    
    # 3. LAYERS - Cluster octahedra using edge-sharing
    visited = np.zeros(count, dtype=bool)
    layers_entity = {}
    layer_membership = {}
    layer_id = 0
    
    # Build edge adjacency for clustering
    edge_adjacency = defaultdict(set)
    for oct1, oct2, _ in edge_sharing_relationships:
        edge_adjacency[oct1].add(oct2)
        edge_adjacency[oct2].add(oct1)
    
    # BFS clustering into layers
    for oct_idx in range(count):
        if visited[oct_idx]:
            continue
            
        layer_octahedra = []
        queue = deque([oct_idx])
        visited[oct_idx] = True
        
        while queue:
            current_oct = queue.popleft()
            layer_octahedra.append(current_oct)
            layer_membership[current_oct] = layer_id
            octahedra_entity[current_oct]['layer'] = layer_id
            
            for neighbor_oct in edge_adjacency[current_oct]:
                if not visited[neighbor_oct]:
                    visited[neighbor_oct] = True
                    queue.append(neighbor_oct)
        
        layers_entity[layer_id] = {
            'octahedra': layer_octahedra
        }
        layer_id += 1
    
    # ENHANCED: Classify atoms by their sharing behavior
    # Axial = inter-layer shared (corner-sharing)
    # Equatorial = intra-layer shared (edge-sharing) 
    # Terminal = not shared (unique to octahedron)
    
    # Build sharing classification for each octahedron
    for oct_id in range(count):
        oct_neighbors = set(neighbor_indices[oct_id])
        
        # Find atoms that are edge-shared (equatorial)
        equatorial_atoms = set()
        for oct1, oct2, shared in edge_sharing_relationships:
            if oct_id in (oct1, oct2):
                equatorial_atoms.update(shared)
        
        # Find atoms that are corner-shared (axial)
        axial_atoms = set()
        for oct1, oct2, shared in corner_sharing_relationships:
            if oct_id in (oct1, oct2):
                axial_atoms.update(shared)
        
        # Remaining atoms are terminal (not shared)
        terminal_atoms = oct_neighbors - equatorial_atoms - axial_atoms
        
        # Add classification to octahedra entity
        octahedra_entity[oct_id]['atom_classification'] = {
            'equatorial': list(equatorial_atoms),  # Intra-layer shared
            'axial': list(axial_atoms),           # Inter-layer shared
            'terminal': list(terminal_atoms)       # Not shared
        }

    # Return clean entity-relationship graph
    return {
        'entities': {
            'atoms': atoms_entity,
            'octahedra': octahedra_entity,
            'layers': layers_entity
        },
        'relationships': {
            'edge_sharing': edge_sharing_relationships,
            'corner_sharing': corner_sharing_relationships,
            'layer_membership': layer_membership
        },
        'metadata': {
            'total_atoms': len(atoms_entity),
            'total_octahedra': count,
            'total_layers': len(layers_entity)
        }
    }

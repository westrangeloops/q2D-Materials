"""
Simple geometry function for distance calculations.
"""
import networkx as nx
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

# The main idea here ir to build a graph, first we part for the unit cell:
# Properties of the unit cell node: lattice constants, angles.
# Connectivity: unit cell to layer node from unit cell to layer node.
# Then we create the layer node, properties: position: Surface or Central.
# Connectivity: layer to octahedra node from layer to octahedra node.
# Then we create the octahedra node, properties: Central_Atom, Terminal_Atoms (Depends if is surface or not), Interlayer_Atoms (Shared by between layers atoms), Intralayer_Atoms (Shared by within the samelayer atoms).
# Connectivity: Central_Atom to atom, Terminal_Atoms to atom, Interlayer_Atoms to atom, Intralayer_Atoms to atom.
# Then we create the atoms nodes, properties: Vasp_index, Symbol, Direct_coordinates: [x, y, z], 


def _graph_inorganic_ontology(atom_positions, atom_symbols, cell, cutoff_distance=4.0, 
                             min_tolerance=0.2, step=0.1, max_steps=20,
                             non_metal_symbols=['C', 'O', 'N', 'H', 'F', 'Cl', 'Br', 'I']):
    """
    Build a comprehensive graph-based inorganic ontology for 2D materials.
    
    The graph structure follows this hierarchy:
    1. Unit Cell Node: Contains lattice parameters and angles
    2. Layer Nodes: Classified as Surface or Central based on terminal atoms
    3. Octahedra Nodes: Each octahedron with classified atoms
    4. Atom Nodes: Individual atoms with properties and classifications
    
    Parameters:
    atom_positions: list of [x, y, z] coordinates
    atom_symbols: list of atomic symbols
    cell: 3x3 array of unit cell vectors
    cutoff_distance: maximum distance for octahedral neighbors
    min_tolerance: starting bond length tolerance for octahedra detection
    step: increment step for tolerance optimization
    max_steps: maximum steps for tolerance optimization
    non_metal_symbols: list of non-metal symbols for filtering central atoms
    
    Returns:
    networkx.Graph: Complete inorganic ontology graph
    """
    import numpy as np
    
    # Initialize the graph
    G = nx.Graph()
    
    # 1. This part is managed by the cell_analysis.py module.
    
    # 2. DETECT OCTAHEDRA
    octahedra_count, centers_positions, center_symbols, neighbor_indices, center_atom_indices = _count_octahedra(
        atom_positions, atom_symbols, cutoff_distance, min_tolerance, step, max_steps, cell, non_metal_symbols
    )
    
    # 3. FIND SHARED ATOMS BETWEEN OCTAHEDRA
    shared_atoms = find_shared_atoms(neighbor_indices)
    
    # 4. CLASSIFY ATOMS BASED ON SHARING
    atom_classifications = _classify_atoms(atom_positions, atom_symbols, neighbor_indices, shared_atoms)
    
    # 5. IDENTIFY LAYERS BASED ON EDGE-SHARING RELATIONSHIPS
    layers = _identify_layers(neighbor_indices, shared_atoms)
    
    # 6. CREATE LAYER NODES
    for layer_id, layer_info in layers.items():
        G.add_node(f'layer_{layer_id}',
                   node_type='layer',
                   position=layer_info['position'])  # 'surface' or 'central'
        
        # Note: Unit cell connections will be handled by the analyzer
    
    # 7. CREATE OCTAHEDRA NODES
    for i, (center_pos, center_sym, neighbors, center_idx) in enumerate(zip(
        centers_positions, center_symbols, neighbor_indices, center_atom_indices)):
        
        # Classify atoms in this octahedron
        terminal_atoms = []
        interlayer_atoms = []
        intralayer_atoms = []
        
        for neighbor_idx in neighbors:
            atom_class = atom_classifications[neighbor_idx]['classification']
            if atom_class == 'terminal':
                terminal_atoms.append(neighbor_idx)
            elif atom_class in ['axial', 'mixed']:  # Interlayer atoms
                interlayer_atoms.append(neighbor_idx)
            elif atom_class in ['equatorial']:  # Intralayer atoms
                intralayer_atoms.append(neighbor_idx)
        
        # Determine which layer this octahedron belongs to
        layer_id = _get_octahedron_layer(i, layers)
        
        G.add_node(f'octahedron_{i}',
                   node_type='octahedron',
                   central_atom=center_idx,
                   terminal_atoms=terminal_atoms,
                   interlayer_atoms=interlayer_atoms,
                   intralayer_atoms=intralayer_atoms)
        
        # Connect octahedron to its layer
        G.add_edge(f'layer_{layer_id}', f'octahedron_{i}', edge_type='contains')
    
    # 8. CREATE ATOM NODES
    for i, (pos, symbol) in enumerate(zip(atom_positions, atom_symbols)):
        G.add_node(f'atom_{i}',
                   node_type='atom',
                   vasp_index=i,
                   symbol=symbol,
                   direct_coordinates=pos.tolist())
    
    # 9. CREATE OCTAHEDRA-TO-OCTAHEDRA CONNECTIONS (for shared atoms)
    for (oct_i, oct_j), shared_atom_indices in shared_atoms.items():
        G.add_edge(f'octahedron_{oct_i}', f'octahedron_{oct_j}', 
                   edge_type='shares_atoms', 
                   shared_atoms=shared_atom_indices)
    
    # 10. CREATE OCTAHEDRA-TO-ATOM CONNECTIONS
    for i, neighbors in enumerate(neighbor_indices):
        # Connect octahedron to all its neighbor atoms
        for neighbor_idx in neighbors:
            G.add_edge(f'octahedron_{i}', f'atom_{neighbor_idx}', 
                       edge_type='contains_atom')
        
        # Connect octahedron to its central atom
        center_idx = center_atom_indices[i]
        G.add_edge(f'octahedron_{i}', f'atom_{center_idx}', 
                   edge_type='has_center')
    
    # 11. CREATE CENTRAL ATOM TO OCTAHEDRA CONNECTIONS
    # Each central atom (divalent) is connected to all octahedra it belongs to
    for i, center_idx in enumerate(center_atom_indices):
        G.add_edge(f'atom_{center_idx}', f'octahedron_{i}', 
                   edge_type='is_center_of')
    
    # 12. CREATE MOLECULAR CONNECTIONS (H-N-C bonds and H-halogen bonds)
    _create_molecular_connections(G, atom_positions, atom_symbols, neighbor_indices, cell)
    
    return G


def _create_molecular_connections(G, atom_positions, atom_symbols, neighbor_indices, cell,
                                bond_cutoff_multiplier=1.2):
    """
    Create comprehensive molecular connections using covalent radii and PBC-aware distances.
    Similar to VESTA's bond identification approach.
    
    Parameters:
    G: networkx graph
    atom_positions: list of atom coordinates
    atom_symbols: list of atomic symbols
    neighbor_indices: list of neighbor indices for each octahedron
    cell: unit cell vectors for PBC calculations
    bond_cutoff_multiplier: multiplier for covalent radii to determine bonds
    """
    import numpy as np
    
    # Standard covalent radii in Angstroms (from ASE and literature)
    covalent_radii = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
        'S': 1.05, 'Cl': 0.99, 'Br': 1.20, 'I': 1.39, 'P': 1.07,
        'Si': 1.11, 'B': 0.84, 'Al': 1.21, 'Mg': 1.41, 'Ca': 1.76,
        'Na': 1.66, 'K': 2.03, 'Li': 1.28, 'Be': 0.96, 'Ne': 0.58,
        'Ar': 1.06, 'Kr': 1.16, 'Xe': 1.40, 'Rn': 1.50, 'He': 0.28,
        'Pb': 1.75, 'Sn': 1.40, 'Ge': 1.20, 'Ti': 1.60, 'Zr': 1.75,
        'Hf': 1.75, 'V': 1.53, 'Nb': 1.64, 'Ta': 1.70, 'Cr': 1.39,
        'Mo': 1.54, 'W': 1.62, 'Mn': 1.39, 'Fe': 1.32, 'Co': 1.26,
        'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'In': 1.42,
        'Tl': 1.70, 'Bi': 1.70, 'Sb': 1.40, 'As': 1.19, 'Se': 1.20,
        'Te': 1.38, 'Po': 1.50
    }
    
    # Find atoms that are not part of any octahedron (isolated atoms)
    atoms_in_octahedra = set()
    for oct_neighbors in neighbor_indices:
        atoms_in_octahedra.update(oct_neighbors)
    
    isolated_atoms = []
    for i, symbol in enumerate(atom_symbols):
        if i not in atoms_in_octahedra:
            isolated_atoms.append(i)
    
    if not isolated_atoms:
        return
    
    # Find halogen atoms in the inorganic structure
    halogen_symbols = ['F', 'Cl', 'Br', 'I']
    halogen_atoms = []
    for i, symbol in enumerate(atom_symbols):
        if i in atoms_in_octahedra and symbol in halogen_symbols:
            halogen_atoms.append(i)
    
    # Create comprehensive adjacency matrix for molecular atoms
    n_atoms = len(atom_symbols)
    adjacency_matrix = np.zeros((n_atoms, n_atoms), dtype=bool)
    
    # Check all pairs of atoms for covalent bonds
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Calculate PBC-aware distance
            distance = _calculate_distances(atom_positions[i], [atom_positions[j]], cell)[0]
            
            # Get covalent radii
            symbol_i = atom_symbols[i]
            symbol_j = atom_symbols[j]
            
            radius_i = covalent_radii.get(symbol_i, 1.5)  # Default fallback
            radius_j = covalent_radii.get(symbol_j, 1.5)
            
            # Calculate bond cutoff
            bond_cutoff = (radius_i + radius_j) * bond_cutoff_multiplier
            
            # Check if atoms are bonded
            if distance <= bond_cutoff:
                adjacency_matrix[i, j] = True
                adjacency_matrix[j, i] = True
    
    # Add covalent bonds to graph
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if adjacency_matrix[i, j]:
                distance = _calculate_distances(atom_positions[i], [atom_positions[j]], cell)[0]
                
                # Determine bond type
                symbol_i = atom_symbols[i]
                symbol_j = atom_symbols[j]
                
                if symbol_i == 'H' and symbol_j == 'N':
                    bond_type = 'H-N'
                elif symbol_i == 'N' and symbol_j == 'H':
                    bond_type = 'H-N'
                elif symbol_i == 'N' and symbol_j == 'C':
                    bond_type = 'N-C'
                elif symbol_i == 'C' and symbol_j == 'N':
                    bond_type = 'N-C'
                elif symbol_i == 'H' and symbol_j == 'C':
                    bond_type = 'H-C'
                elif symbol_i == 'C' and symbol_j == 'H':
                    bond_type = 'H-C'
                elif symbol_i == 'C' and symbol_j == 'C':
                    bond_type = 'C-C'
                elif symbol_i == 'H' and symbol_j == 'H':
                    bond_type = 'H-H'
                else:
                    bond_type = f'{symbol_i}-{symbol_j}'
                
                G.add_edge(f'atom_{i}', f'atom_{j}', 
                          edge_type='covalent_bond', bond_type=bond_type, distance=distance)
    
    # Add hydrogen bonds (H to halogens)
    for h_atom_idx in isolated_atoms:
        if atom_symbols[h_atom_idx] == 'H':
            h_position = atom_positions[h_atom_idx]
            
            for halogen_idx in halogen_atoms:
                distance = _calculate_distances(h_position, [atom_positions[halogen_idx]], cell)[0]
                
                # Hydrogen bond cutoff (typically 2.5-3.5 Å)
                hbond_cutoff = 3.0
                
                if distance <= hbond_cutoff:
                    G.add_edge(f'atom_{h_atom_idx}', f'atom_{halogen_idx}', 
                              edge_type='hydrogen_bond', distance=distance)


def _classify_atoms(atom_positions, atom_symbols, neighbor_indices, shared_atoms):
    """
    Classify atoms as terminal, equatorial (intralayer), or axial (interlayer) based on sharing patterns.
    Based on the previous code logic but adapted for graph structure.
    
    Parameters:
    atom_positions: list of atom coordinates
    atom_symbols: list of atomic symbols  
    neighbor_indices: list of neighbor indices for each octahedron
    shared_atoms: dict mapping octahedra pairs to shared atom indices
    
    Returns:
    dict: atom index -> classification info
    """
    import numpy as np
    
    atom_classifications = {}
    
    # Count how many octahedra each atom belongs to
    atom_octahedra_count = {}
    atom_octahedra_list = {}
    
    for atom_idx in range(len(atom_positions)):
        atom_octahedra_count[atom_idx] = 0
        atom_octahedra_list[atom_idx] = []
    
    for oct_idx, neighbors in enumerate(neighbor_indices):
        for neighbor_idx in neighbors:
            atom_octahedra_count[neighbor_idx] += 1
            atom_octahedra_list[neighbor_idx].append(oct_idx)
    
    # Classify sharing relationships
    edge_sharing_relationships = []
    corner_sharing_relationships = []
    
    for (oct1, oct2), shared in shared_atoms.items():
        num_shared = len(shared)
        if num_shared >= 3:  # Edge-sharing (same layer)
            edge_sharing_relationships.append((oct1, oct2, shared))
        elif num_shared <= 2:  # Corner-sharing (inter-layer)
            corner_sharing_relationships.append((oct1, oct2, shared))
    
    # Classify atoms based on sharing behavior
    for atom_idx in range(len(atom_positions)):
        octahedra_count = atom_octahedra_count[atom_idx]
        octahedra_list = atom_octahedra_list[atom_idx]
        
        if octahedra_count == 1:
            classification = 'terminal'
        elif octahedra_count > 1:
            # Check if this atom is involved in edge-sharing (equatorial) or corner-sharing (axial)
            is_edge_shared = any(atom_idx in shared for oct1, oct2, shared in edge_sharing_relationships)
            is_corner_shared = any(atom_idx in shared for oct1, oct2, shared in corner_sharing_relationships)
            
            if is_edge_shared and not is_corner_shared:
                classification = 'equatorial'  # Intra-layer shared
            elif is_corner_shared and not is_edge_shared:
                classification = 'axial'       # Inter-layer shared
            elif is_edge_shared and is_corner_shared:
                classification = 'mixed'      # Both edge and corner shared
            else:
                classification = 'intralayer'  # Fallback
        else:
            classification = 'isolated'
        
        atom_classifications[atom_idx] = {
            'classification': classification,
            'shared_by_octahedra': octahedra_list,
            'octahedra_count': octahedra_count
        }
    
    return atom_classifications


def _identify_layers(neighbor_indices, shared_atoms):
    """
    Identify layers by clustering octahedra using edge-sharing relationships.
    Based on the previous code logic but adapted for graph structure.
    
    Parameters:
    neighbor_indices: list of neighbor indices for each octahedron
    shared_atoms: dict mapping octahedra pairs to shared atom indices
    
    Returns:
    dict: layer_id -> layer_info
    """
    import numpy as np
    from collections import defaultdict, deque
    
    n_octahedra = len(neighbor_indices)
    if n_octahedra == 0:
        return {}
    
    # Classify sharing relationships
    edge_sharing_relationships = []
    corner_sharing_relationships = []
    
    for (oct1, oct2), shared in shared_atoms.items():
        num_shared = len(shared)
        if num_shared >= 3:  # Edge-sharing (same layer)
            edge_sharing_relationships.append((oct1, oct2, shared))
        elif num_shared <= 2:  # Corner-sharing (inter-layer)
            corner_sharing_relationships.append((oct1, oct2, shared))
    
    # Build edge adjacency for clustering
    edge_adjacency = defaultdict(set)
    for oct1, oct2, _ in edge_sharing_relationships:
        edge_adjacency[oct1].add(oct2)
        edge_adjacency[oct2].add(oct1)
    
    # BFS clustering into layers
    visited = np.zeros(n_octahedra, dtype=bool)
    layers = {}
    layer_id = 0
    
    for oct_idx in range(n_octahedra):
        if visited[oct_idx]:
            continue
            
        layer_octahedra = []
        queue = deque([oct_idx])
        visited[oct_idx] = True
        
        while queue:
            current_oct = queue.popleft()
            layer_octahedra.append(current_oct)
            
            for neighbor_oct in edge_adjacency[current_oct]:
                if not visited[neighbor_oct]:
                    visited[neighbor_oct] = True
                    queue.append(neighbor_oct)
        
        # Count terminal atoms in this layer (atoms not shared between octahedra)
        layer_atoms = set()
        for oct_id in layer_octahedra:
            layer_atoms.update(neighbor_indices[oct_id])
        
        # Count terminal atoms (atoms that appear in only one octahedron in this layer)
        terminal_count = 0
        for atom_idx in layer_atoms:
            atom_octahedra_count = sum(1 for oct_id in layer_octahedra 
                                     if atom_idx in neighbor_indices[oct_id])
            if atom_octahedra_count == 1:
                terminal_count += 1
        
        # Determine if this is a surface layer (has terminal atoms) or central layer
        position = 'surface' if terminal_count > 0 else 'central'
        
        layers[layer_id] = {
            'position': position,
            'octahedra': layer_octahedra,
            'terminal_atoms_count': terminal_count,
            'octahedra_count': len(layer_octahedra),
            'edge_sharing_relationships': [rel for rel in edge_sharing_relationships 
                                          if rel[0] in layer_octahedra or rel[1] in layer_octahedra]
        }
        layer_id += 1
    
    return layers


def _get_octahedron_layer(octahedron_id, layers):
    """
    Determine which layer an octahedron belongs to based on layer membership.
    
    Parameters:
    octahedron_id: index of the octahedron
    layers: dict of layer information
    
    Returns:
    int: layer_id
    """
    for layer_id, layer_info in layers.items():
        if octahedron_id in layer_info['octahedra']:
            return layer_id
    
    # If not found, return 0 as default
    return 0


    
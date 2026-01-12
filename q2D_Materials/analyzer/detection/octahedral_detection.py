"""
Octahedral structure detection and classification for perovskites.

This module contains functions to identify octahedral units (BX6) in crystal structures
and classify their constituent atoms based on sharing patterns.

Functions
---------
_count_octahedra
    Detect octahedral centers and neighbors with adaptive tolerance
find_shared_atoms
    Identify which octahedra share atoms
_classify_atoms
    Classify atoms as terminal, equatorial, or axial based on sharing
_classify_x_atoms_by_z
    Classify X-site atoms using z-coordinate grouping
_calculate_avg_bx_distance
    Calculate average B-X bond distance from octahedra
"""

import numpy as np
from collections import defaultdict
from ..utils.perovskite_constants import get_bond_cutoff
from ...utils.geometry.geometry import _calculate_distances


def _count_octahedra(atom_positions, atom_symbols=None, cutoff_distance=4.0, 
                    min_tolerance=0.2, step=0.1, max_steps=20, 
                    cell=None, non_metal_symbols=['C', 'O', 'N', 'H', 'F', 'Cl', 'Br', 'I']):
    """
    Adaptive octahedra counting with automatic tolerance optimization.
    
    Handles both large supercells and minimal unit cells where octahedral vertices
    may be periodic images of the same atom.
    
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
        - neighbor_indices: list of lists, each containing atom indices that form the octahedron
    """
    import numpy as np
    
    atom_positions = np.array(atom_positions)
    
    # Common X-site anions in perovskites (atoms that form octahedral vertices)
    x_site_elements = {'O', 'F', 'Cl', 'Br', 'I', 'S', 'Se', 'Te'}
    
    # History to track autoconsistency
    count_history = []
    tolerance = min_tolerance
    
    # Pre-identify metal atoms (potential B-sites) to avoid repeated filtering
    if atom_symbols is not None:
        non_metal_mask = np.array([sym in non_metal_symbols for sym in atom_symbols])
        metal_indices = np.where(~non_metal_mask)[0]
        
        # Also identify X-site indices
        x_site_mask = np.array([sym in x_site_elements for sym in atom_symbols])
        x_site_indices = set(np.where(x_site_mask)[0])
    else:
        metal_indices = np.arange(len(atom_positions))
        x_site_indices = set()
    
    # PRE-COMPUTE: Calculate all distances for all metal atoms once
    n_atoms = len(atom_positions)
    all_neighbor_data = {}
    
    for i in metal_indices:
        # Calculate PBC-aware distances to all other atoms
        distances = _calculate_distances(atom_positions[i], atom_positions, cell)
        
        # Get indices of all other atoms (excluding self)
        other_indices = np.concatenate([np.arange(i), np.arange(i+1, n_atoms)])
        other_distances = distances[other_indices]
        
        # IMPROVED: Filter to only X-site elements for perovskite octahedra
        if atom_symbols is not None and x_site_indices:
            x_site_mask_local = np.array([idx in x_site_indices for idx in other_indices])
            x_site_other_indices = other_indices[x_site_mask_local]
            x_site_other_distances = other_distances[x_site_mask_local]
            
            # First, filter by cutoff distance
            within_cutoff = x_site_other_distances <= cutoff_distance
            x_site_close_indices = x_site_other_indices[within_cutoff]
            x_site_close_distances = x_site_other_distances[within_cutoff]
            
            # Check if we have enough X-site neighbors within cutoff
            n_close = len(x_site_close_distances)
            if n_close >= 4:  # Allow 4+ for small cells with PBC
                # Use all close neighbors (up to 6)
                n_neighbors = min(6, n_close)
                
                # Get the closest ones if we have more than 6
                if n_close > 6:
                    closest_idx = np.argpartition(x_site_close_distances, 5)[:6]
                    closest_distances = x_site_close_distances[closest_idx]
                    closest_atom_indices = x_site_close_indices[closest_idx]
                else:
                    closest_distances = x_site_close_distances
                    closest_atom_indices = x_site_close_indices
                
                all_neighbor_data[i] = {
                    'distances': closest_distances,
                    'indices': closest_atom_indices,
                    'position': atom_positions[i].copy(),
                    'symbol': atom_symbols[i] if atom_symbols is not None else 'Unknown',
                    'n_neighbors': n_neighbors
                }
        
        # Fallback for structures without clear X-sites
        if i not in all_neighbor_data and len(other_distances) >= 6:
            closest_6_idx = np.argpartition(other_distances, 5)[:6]
            closest_6_distances = other_distances[closest_6_idx]
            closest_6_atom_indices = other_indices[closest_6_idx]
            
            if np.max(closest_6_distances) <= cutoff_distance:
                all_neighbor_data[i] = {
                    'distances': closest_6_distances,
                    'indices': closest_6_atom_indices,
                    'position': atom_positions[i].copy(),
                    'symbol': atom_symbols[i] if atom_symbols is not None else 'Unknown',
                    'n_neighbors': 6
                }
    
    # MAIN TOLERANCE ITERATION LOOP
    for step_num in range(max_steps):
        octahedra_count = 0
        octahedral_centers = []
        center_symbols = []
        all_neighbor_indices = []
        center_atom_indices = []
        
        for i, data in all_neighbor_data.items():
            closest_distances = data['distances']
            n_neighbors = data['n_neighbors']
            
            # Vectorized pairwise difference check
            diff_matrix = np.abs(closest_distances[:, np.newaxis] - closest_distances)
            
            # Check if all pairwise differences are within tolerance
            # For small cells with 4-5 visible neighbors, use looser criteria
            if np.all(diff_matrix <= tolerance):
                octahedra_count += 1
                octahedral_centers.append(data['position'])
                all_neighbor_indices.append(data['indices'].tolist())
                center_symbols.append(data['symbol'])
                center_atom_indices.append(i)
            elif n_neighbors >= 4 and n_neighbors < 6:
                # Small cell case: check if neighbors are reasonably close
                avg_dist = np.mean(closest_distances)
                max_dev = np.max(np.abs(closest_distances - avg_dist))
                if max_dev <= tolerance * 1.5:  # Slightly looser for small cells
                    octahedra_count += 1
                    octahedral_centers.append(data['position'])
                    all_neighbor_indices.append(data['indices'].tolist())
                    center_symbols.append(data['symbol'])
                    center_atom_indices.append(i)
        
        count_history.append(octahedra_count)
        
        # Check for autoconsistency (last 3 values are the same)
        if len(count_history) >= 3:
            if count_history[-1] == count_history[-2] == count_history[-3]:
                return octahedra_count, np.array(octahedral_centers) if octahedral_centers else np.array([]).reshape(0, 3), center_symbols, all_neighbor_indices, center_atom_indices
        
        tolerance += step
    
    # If no convergence, return the last result
    return octahedra_count, np.array(octahedral_centers) if octahedral_centers else np.array([]).reshape(0, 3), center_symbols, all_neighbor_indices, center_atom_indices


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

def _calculate_avg_bx_distance(
    neighbor_indices: list,
    atom_positions: np.ndarray,
    center_atom_indices: list,
    max_bx_distance: float = 5.0,
) -> float:
    """
    Calculate average B-X distance from octahedra geometry.

    Parameters
    ----------
    neighbor_indices : list
        List of neighbor atom indices for each octahedron
    atom_positions : np.ndarray
        Array of atom positions
    center_atom_indices : list
        List of central atom indices for each octahedron
    max_bx_distance : float
        Maximum reasonable B-X distance to filter outliers

    Returns
    -------
    float
        Average B-X distance in Angstroms
    """
    bx_distances = []
    for oct_idx, neighbors in enumerate(neighbor_indices):
        if oct_idx >= len(center_atom_indices):
            continue
        center_idx = center_atom_indices[oct_idx]
        center_pos = atom_positions[center_idx]
        for neighbor_idx in neighbors:
            neighbor_pos = atom_positions[neighbor_idx]
            dist = np.linalg.norm(center_pos - neighbor_pos)
            if dist < max_bx_distance:
                bx_distances.append(dist)

    return np.mean(bx_distances) if bx_distances else 3.2  # Default for Pb-I



def _classify_x_atoms_by_z(
    neighbor_indices: list,
    shared_atoms: dict,
    atom_positions: np.ndarray,
    center_atom_indices: list,
) -> dict:
    """
    Classify X-atoms as axial, interlayer, or intralayer based on Z-differences.

    Classification criteria:
    - AXIAL (terminal): Connected to only 1 octahedron (surface atoms)
    - INTERLAYER: Connected to 2+ octahedra with large Z-difference between their centers
    - INTRALAYER (equatorial): Connected to 2+ octahedra with small Z-difference

    Parameters
    ----------
    neighbor_indices : list
        List of neighbor atom indices for each octahedron
    shared_atoms : dict
        Dict mapping (oct_i, oct_j) -> list of shared atom indices
    atom_positions : np.ndarray
        Array of atom positions
    center_atom_indices : list
        List of central B-site atom indices for each octahedron

    Returns
    -------
    dict
        {atom_idx: {'type': 'axial'|'interlayer'|'intralayer',
                   'connected_octahedra': [oct_indices],
                   'z_coord': float}}
    """
    # Calculate average B-X distance to determine Z threshold
    avg_bx = _calculate_avg_bx_distance(neighbor_indices, atom_positions, center_atom_indices)
    # Z-threshold: half the expected layer spacing (B-X distance)
    z_threshold = avg_bx * 0.7  # Slightly less than 1 B-X distance

    # Build atom -> octahedra mapping
    atom_to_octahedra = defaultdict(list)
    for oct_idx, neighbors in enumerate(neighbor_indices):
        for atom_idx in neighbors:
            atom_to_octahedra[atom_idx].append(oct_idx)

    # Get Z-coordinates of octahedra centers
    oct_z_coords = {}
    for oct_idx, center_idx in enumerate(center_atom_indices):
        oct_z_coords[oct_idx] = atom_positions[center_idx][2]

    # Classify each X-atom
    x_atom_classifications = {}

    for atom_idx, octahedra_list in atom_to_octahedra.items():
        atom_z = atom_positions[atom_idx][2]
        n_octahedra = len(octahedra_list)

        if n_octahedra == 1:
            # Connected to only 1 octahedron = AXIAL (terminal/surface)
            classification = 'axial'
        elif n_octahedra >= 2:
            # Check Z-difference between connected octahedra
            z_coords = [oct_z_coords[oct_idx] for oct_idx in octahedra_list if oct_idx in oct_z_coords]
            if len(z_coords) >= 2:
                z_diff = max(z_coords) - min(z_coords)
                if z_diff > z_threshold:
                    # Large Z-difference = INTERLAYER (connects floors)
                    classification = 'interlayer'
                else:
                    # Small Z-difference = INTRALAYER (equatorial, same floor)
                    classification = 'intralayer'
            else:
                classification = 'intralayer'  # Default
        else:
            classification = 'unknown'

        x_atom_classifications[atom_idx] = {
            'type': classification,
            'connected_octahedra': octahedra_list,
            'z_coord': atom_z,
        }

    return x_atom_classifications



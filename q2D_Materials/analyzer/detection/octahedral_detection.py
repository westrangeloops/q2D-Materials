"""
Octahedral structure detection using 27-image PBC neighbor finding.

This module detects octahedral units (BX6) in crystal structures using the
straightforward 27-image approach for periodic boundary conditions.

Key insights:
1. 27-image PBC: Simple, direct Euclidean distances in Angstroms
2. Topology-based classification: atoms are classified by their ROLE in the structure
   (shared vs terminal, intra-layer vs inter-layer) not by local angles

Functions
---------
_find_neighbors_27img
    Find 6 nearest X-site neighbors using 27 periodic images
classify_atoms_by_topology
    Classify atoms by their topological role (terminal/equatorial/axial)
_count_octahedra
    Main function to detect all octahedra in a structure
find_shared_atoms
    Identify which octahedra share atoms
"""

import numpy as np
import sys


def _get_27_image_offsets():
    """Get the 27 offset vectors for periodic images."""
    offsets = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                offsets.append([i, j, k])
    return np.array(offsets, dtype=np.float64)


def _find_neighbors_27img(
    b_atom_idx: int,
    atom_positions: np.ndarray,
    atom_symbols: list,
    x_site_indices: set,
    valid_halogen: list,
    cell: np.ndarray,
    target_neighbors: int = 6,
) -> tuple:
    """
    Find neighbors using 27-image periodic boundary conditions.

    For each X-site candidate, considers all 27 periodic images and
    finds the closest one to the B-site center. Returns the N nearest.

    Parameters
    ----------
    b_atom_idx : int
        Index of B-site atom
    atom_positions : np.ndarray
        All atom positions (3D)
    atom_symbols : list
        All atom symbols
    x_site_indices : set
        Set of X-site atom indices
    valid_halogen : list
        List of valid halogen symbols
    cell : np.ndarray
        Unit cell matrix (3x3) - rows are cell vectors
    target_neighbors : int
        Target number of neighbors (default: 6 for octahedra)

    Returns
    -------
    tuple
        (neighbor_indices, neighbor_distances) or (None, None) if not enough neighbors
        Distances are in Angstroms (directly interpretable)
    """
    offsets = _get_27_image_offsets()

    # Get B-site position
    center_pos = atom_positions[b_atom_idx]

    # Get all X-site candidates
    x_site_positions = []
    x_site_atom_indices = []
    for idx in x_site_indices:
        if idx != b_atom_idx:
            if atom_symbols is None or atom_symbols[idx] in valid_halogen:
                x_site_positions.append(atom_positions[idx])
                x_site_atom_indices.append(idx)

    if not x_site_positions or len(x_site_atom_indices) < target_neighbors:
        return None, None

    x_site_positions = np.array(x_site_positions)
    x_site_atom_indices = np.array(x_site_atom_indices)
    n_candidates = len(x_site_positions)

    # For each candidate, find minimum distance across all 27 images
    min_distances = np.full(n_candidates, np.inf)

    for i, pos in enumerate(x_site_positions):
        for offset in offsets:
            # Translate position by offset * cell vectors
            translated = pos + offset @ cell
            dist = np.linalg.norm(translated - center_pos)
            if dist < min_distances[i]:
                min_distances[i] = dist

    # Sort by distance and take N nearest
    sorted_idx = np.argsort(min_distances)
    final_indices = x_site_atom_indices[sorted_idx[:target_neighbors]]
    final_distances = min_distances[sorted_idx[:target_neighbors]]

    return final_indices, final_distances


def find_shared_atoms(neighbor_indices_list):
    """
    Find which octahedra share atoms.

    Parameters
    ----------
    neighbor_indices_list : list of lists
        Each inner list contains atom indices for an octahedron

    Returns
    -------
    dict
        Mapping of octahedra pairs to their shared atom indices
        e.g., {(0,1): [2, 5], (0,2): [3]}
    """
    shared_atoms = {}
    n_octahedra = len(neighbor_indices_list)

    if n_octahedra <= 1:
        return shared_atoms

    neighbor_arrays = [np.array(indices, dtype=np.int32) for indices in neighbor_indices_list]

    for i in range(n_octahedra):
        for j in range(i + 1, n_octahedra):
            shared = np.intersect1d(neighbor_arrays[i], neighbor_arrays[j])
            if len(shared) > 0:
                shared_atoms[(i, j)] = shared.tolist()

    return shared_atoms


def classify_atoms_by_topology(neighbor_indices_list):
    """
    Classify atoms using pure topology - no coordinates needed.

    This is the key insight: in perovskites, atom classification depends on
    their STRUCTURAL ROLE, not local geometry:
    - Terminal atoms: belong to only 1 octahedron (the "free" axials)
    - Equatorial atoms: shared by octahedra in the SAME layer
    - Axial interlayer atoms: shared by octahedra in DIFFERENT layers

    The layer structure is determined by the sharing pattern:
    - High sharing count (typically 4) = same layer (corner-sharing in plane)
    - Low sharing count (typically 2) = different layers (axial connection)

    Parameters
    ----------
    neighbor_indices_list : list of lists
        Each inner list contains atom indices for an octahedron

    Returns
    -------
    tuple
        (atom_classification, layer_membership, octahedra_geometries)
        - atom_classification: dict mapping atom_idx -> 'terminal' | 'equatorial' | 'axial_interlayer'
        - layer_membership: dict mapping oct_idx -> layer_idx
        - octahedra_geometries: list of dicts mapping neighbor_idx -> geometry label
    """
    n_octahedra = len(neighbor_indices_list)

    if n_octahedra == 0:
        return {}, {}, []

    # Step 1: Build atom -> octahedra membership
    atom_to_octahedra = {}
    for oct_idx, neighbors in enumerate(neighbor_indices_list):
        for atom_idx in neighbors:
            atom_idx = int(atom_idx)
            if atom_idx not in atom_to_octahedra:
                atom_to_octahedra[atom_idx] = set()
            atom_to_octahedra[atom_idx].add(oct_idx)

    # Step 2: Find sharing between octahedra pairs
    pair_sharing = {}
    for atom_idx, octahedra in atom_to_octahedra.items():
        if len(octahedra) == 2:
            pair = tuple(sorted(octahedra))
            if pair not in pair_sharing:
                pair_sharing[pair] = []
            pair_sharing[pair].append(atom_idx)

    # Step 3: Detect 1x1 unit cell case - octahedra share atoms with themselves through PBC
    # In 1x1 cells, high sharing counts (4-5 atoms) between octahedra are due to
    # equatorial atoms being shared with self through PBC (only 2 unique equatorial atoms)
    # Real sharing is only through axial connections (1 atom between layers)
    
    # Heuristic: if max sharing is >= 4 and we have few octahedra, likely 1x1 cell
    is_1x1_cell = False
    if pair_sharing:
        sharing_counts = {pair: len(atoms) for pair, atoms in pair_sharing.items()}
        max_sharing = max(sharing_counts.values())
        
        # 1x1 signature: very high sharing count (>=4) due to PBC self-sharing
        if max_sharing >= 4 and n_octahedra <= 8:  # Small system likely to be 1x1
            is_1x1_cell = True
            print(f"  INFO: Detected 1x1 unit cell (max_sharing={max_sharing}, n_oct={n_octahedra})", file=sys.stderr)
            print(f"  INFO: Ignoring self-shared equatorial atoms, using only axial connections", file=sys.stderr)
    
    if not pair_sharing:
        # Single octahedron or no sharing - all atoms are terminal
        atom_classification = {int(atom_idx): 'terminal' for atom_idx in atom_to_octahedra}
        layer_membership = {oct_idx: oct_idx for oct_idx in range(n_octahedra)}
        octahedra_geometries = []
        for oct_idx, neighbors in enumerate(neighbor_indices_list):
            geometry = {int(n): 'terminal' for n in neighbors}
            octahedra_geometries.append(geometry)
        return atom_classification, layer_membership, octahedra_geometries

    # Find the maximum sharing count to distinguish intra-layer from inter-layer
    sharing_counts = {pair: len(atoms) for pair, atoms in pair_sharing.items()}
    max_sharing = max(sharing_counts.values()) if not is_1x1_cell else 1

    same_layer_pairs = set()
    inter_layer_pairs = set()

    if is_1x1_cell:
        # For 1x1 cells: only count sharing of 1 atom as inter-layer (axial bridge)
        # Ignore high sharing counts (4-5) which are equatorial atoms shared with self
        for pair, count in sharing_counts.items():
            if count == 1:
                # Single shared atom = axial interlayer connection
                inter_layer_pairs.add(pair)
            # Ignore count >= 2: these are self-shared equatorial atoms through PBC
    else:
        # Normal case: use max sharing heuristic
        for pair, count in sharing_counts.items():
            if count >= max_sharing:
                same_layer_pairs.add(pair)
            else:
                inter_layer_pairs.add(pair)

    # Step 4: Assign layer membership using union-find
    layer_membership = {}
    next_layer = 0

    for pair in same_layer_pairs:
        oct_i, oct_j = pair
        if oct_i in layer_membership and oct_j in layer_membership:
            if layer_membership[oct_i] != layer_membership[oct_j]:
                old_layer = layer_membership[oct_j]
                new_layer = layer_membership[oct_i]
                for oct, layer in list(layer_membership.items()):
                    if layer == old_layer:
                        layer_membership[oct] = new_layer
        elif oct_i in layer_membership:
            layer_membership[oct_j] = layer_membership[oct_i]
        elif oct_j in layer_membership:
            layer_membership[oct_i] = layer_membership[oct_j]
        else:
            layer_membership[oct_i] = next_layer
            layer_membership[oct_j] = next_layer
            next_layer += 1

    for oct_idx in range(n_octahedra):
        if oct_idx not in layer_membership:
            layer_membership[oct_idx] = next_layer
            next_layer += 1

    # Step 5: Classify atoms globally
    atom_classification = {}
    
    # For 1x1 cells: identify which atoms are truly shared vs self-shared equatorial
    if is_1x1_cell:
        # Collect atoms that are in inter-layer pairs (truly shared axials)
        truly_shared_atoms = set()
        for pair in inter_layer_pairs:
            if pair in pair_sharing:
                truly_shared_atoms.update(pair_sharing[pair])
        
        for atom_idx, octahedra in atom_to_octahedra.items():
            atom_idx = int(atom_idx)
            if len(octahedra) == 1:
                # Belongs to only one octahedron = terminal axial
                atom_classification[atom_idx] = 'terminal'
            elif len(octahedra) == 2:
                oct_list = list(octahedra)
                # Check if this atom is in the truly shared set (axial bridge)
                if atom_idx in truly_shared_atoms:
                    # True inter-layer sharing (1 atom between octahedra)
                    atom_classification[atom_idx] = 'axial_interlayer'
                else:
                    # High-count sharing in 1x1 = equatorial self-shared through PBC
                    # These are equatorial atoms belonging to same layer (shared with self)
                    atom_classification[atom_idx] = 'equatorial'
            else:
                atom_classification[atom_idx] = 'multi_shared'
    else:
        # Normal classification for larger cells
        for atom_idx, octahedra in atom_to_octahedra.items():
            atom_idx = int(atom_idx)
            if len(octahedra) == 1:
                atom_classification[atom_idx] = 'terminal'
            elif len(octahedra) == 2:
                oct_list = list(octahedra)
                if layer_membership[oct_list[0]] == layer_membership[oct_list[1]]:
                    atom_classification[atom_idx] = 'equatorial'
                else:
                    atom_classification[atom_idx] = 'axial_interlayer'
            else:
                atom_classification[atom_idx] = 'multi_shared'

    # Step 6: Build per-octahedron geometry dictionaries
    octahedra_geometries = []

    for oct_idx, neighbors in enumerate(neighbor_indices_list):
        geometry = {}

        for neighbor_idx in neighbors:
            neighbor_idx = int(neighbor_idx)
            global_class = atom_classification.get(neighbor_idx, 'unknown')

            if global_class == 'equatorial':
                geometry[neighbor_idx] = 'equatorial'
            elif global_class == 'terminal':
                geometry[neighbor_idx] = 'axial_terminal'
            elif global_class == 'axial_interlayer':
                geometry[neighbor_idx] = 'axial_interlayer'
            else:
                geometry[neighbor_idx] = 'unknown'

        octahedra_geometries.append(geometry)

    return atom_classification, layer_membership, octahedra_geometries


def _count_octahedra(
    atom_positions,
    atom_symbols=None,
    cutoff_distance=None,
    min_tolerance=0.2,
    step=0.1,
    max_steps=20,
    cell=None,
    non_metal_symbols=None,
    octahedra_edges=None,
    octahedra_centers=None,
    valid_halogen=None,
    valid_molecule=None,
):
    """
    Detect octahedra using 27-image PBC + topology classification.

    For each B-site atom, finds the 6 nearest X-site neighbors using
    simple Euclidean distances with 27 periodic images.
    Then classifies atoms using topology (sharing patterns).

    Parameters
    ----------
    atom_positions : array-like
        List of [x, y, z] coordinates
    atom_symbols : list, optional
        List of atomic symbols
    cell : np.ndarray
        3x3 unit cell matrix (REQUIRED)
    octahedra_centers : list, optional
        Valid B-site atoms (default: ['Pb', 'Sn'])
    valid_halogen : list, optional
        Valid halogen symbols (default: ['Cl', 'Br', 'I'])

    Returns
    -------
    tuple
        (count, centers_positions, center_symbols, neighbor_indices,
         center_atom_indices, octahedra_geometries)
    """
    if cell is None:
        raise ValueError("cell is REQUIRED for PBC neighbor finding")

    atom_positions = np.array(atom_positions)
    cell = np.asarray(cell, dtype=np.float64)

    # Defaults
    if non_metal_symbols is None:
        non_metal_symbols = ['C', 'O', 'N', 'H', 'F', 'Cl', 'Br', 'I']
    if octahedra_edges is None:
        octahedra_edges = [6]
    if octahedra_centers is None:
        octahedra_centers = ['Pb', 'Sn']
    if valid_halogen is None:
        valid_halogen = ['Cl', 'Br', 'I']
    if valid_molecule is None:
        valid_molecule = ['C', 'H', 'O', 'N', 'S']

    # X-site elements
    x_site_elements = set(valid_halogen)
    x_site_elements.update({'S', 'Se', 'Te', 'O', 'F'})

    # Identify B-site and X-site atoms
    b_site_elements = set(octahedra_centers)

    if atom_symbols is not None:
        b_site_mask = np.array([sym in b_site_elements for sym in atom_symbols])
        b_site_indices = np.where(b_site_mask)[0]

        x_site_mask = np.array([sym in x_site_elements for sym in atom_symbols])
        x_site_indices = set(np.where(x_site_mask)[0])
    else:
        b_site_indices = np.arange(len(atom_positions))
        x_site_indices = set()

    print(f"INFO: Detecting octahedra with 27-image PBC", file=sys.stderr)
    print(f"INFO: Found {len(b_site_indices)} B-site atoms ({octahedra_centers})", file=sys.stderr)
    print(f"INFO: Found {len(x_site_indices)} X-site atoms ({valid_halogen})", file=sys.stderr)

    # Find neighbors for each B-site atom
    target_neighbors = max(octahedra_edges)
    all_neighbor_data = {}

    for b_idx in b_site_indices:
        neighbor_indices, neighbor_distances = _find_neighbors_27img(
            b_idx, atom_positions, atom_symbols, x_site_indices, valid_halogen,
            cell, target_neighbors
        )

        if neighbor_indices is not None and len(neighbor_indices) >= min(octahedra_edges):
            all_neighbor_data[b_idx] = {
                'distances': neighbor_distances,
                'indices': neighbor_indices,
                'position': atom_positions[b_idx].copy(),
                'symbol': atom_symbols[b_idx] if atom_symbols else 'Unknown',
                'n_neighbors': len(neighbor_indices),
            }
        else:
            sym = atom_symbols[b_idx] if atom_symbols else '?'
            print(f"  WARNING: B-site atom {b_idx} ({sym}) - insufficient neighbors", file=sys.stderr)

    print(f"INFO: Found {len(all_neighbor_data)}/{len(b_site_indices)} B-site atoms with {target_neighbors} neighbors", file=sys.stderr)

    # Build output lists
    octahedra_count = 0
    octahedral_centers_list = []
    center_symbols = []
    all_neighbor_indices = []
    center_atom_indices = []

    for i, data in all_neighbor_data.items():
        n_neighbors = data['n_neighbors']

        if n_neighbors not in octahedra_edges:
            continue

        octahedra_count += 1
        octahedral_centers_list.append(data['position'])
        all_neighbor_indices.append(data['indices'].tolist())
        center_symbols.append(data['symbol'])
        center_atom_indices.append(i)

    # Use topology-based classification
    atom_classification, layer_membership, octahedra_geometries = classify_atoms_by_topology(
        all_neighbor_indices
    )

    centers_array = np.array(octahedral_centers_list) if octahedral_centers_list else np.array([]).reshape(0, 3)
    return (octahedra_count, centers_array, center_symbols,
            all_neighbor_indices, center_atom_indices, octahedra_geometries)


def _calculate_avg_bx_distance(
    neighbor_indices: list,
    atom_positions: np.ndarray,
    center_atom_indices: list,
    cell: np.ndarray,
    max_bx_distance: float = 5.0,
) -> float:
    """
    Calculate average B-X distance using 27-image PBC.

    Parameters
    ----------
    neighbor_indices : list
        List of neighbor atom indices for each octahedron
    atom_positions : np.ndarray
        Array of atom positions
    center_atom_indices : list
        List of central atom indices for each octahedron
    cell : np.ndarray
        Unit cell matrix
    max_bx_distance : float
        Maximum reasonable B-X distance to filter outliers

    Returns
    -------
    float
        Average B-X distance in Angstroms
    """
    offsets = _get_27_image_offsets()

    bx_distances = []
    for oct_idx, neighbors in enumerate(neighbor_indices):
        if oct_idx >= len(center_atom_indices):
            continue
        center_idx = center_atom_indices[oct_idx]
        center_pos = atom_positions[center_idx]

        for neighbor_idx in neighbors:
            neighbor_pos = atom_positions[neighbor_idx]
            # Find minimum distance across 27 images
            min_dist = np.inf
            for offset in offsets:
                translated = neighbor_pos + offset @ cell
                dist = np.linalg.norm(translated - center_pos)
                if dist < min_dist:
                    min_dist = dist

            if min_dist < max_bx_distance:
                bx_distances.append(float(min_dist))

    return np.mean(bx_distances) if bx_distances else 3.2

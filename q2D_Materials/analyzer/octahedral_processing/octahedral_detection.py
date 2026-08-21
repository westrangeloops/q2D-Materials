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
    Classify atoms by topological role (terminal/equatorial/axial); axial = along c, equatorial = ab-plane
_count_octahedra
    Main function to detect all octahedra in a structure
find_shared_atoms
    Identify which octahedra share atoms
"""

from typing import Dict, List, Optional, Tuple

import networkx as nx
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

    if not x_site_positions:
        return None, None

    x_site_positions = np.array(x_site_positions)
    x_site_atom_indices = np.array(x_site_atom_indices)

    # Generate ALL 27 images of ALL candidate atoms and calculate distances
    # This allows the same atom to appear multiple times through PBC (self-sharing)
    all_distances = []
    all_indices = []
    
    for i, pos in enumerate(x_site_positions):
        atom_idx = x_site_atom_indices[i]
        for offset in offsets:
            # Translate position by offset * cell vectors
            translated = pos + offset @ cell
            dist = np.linalg.norm(translated - center_pos)
            all_distances.append(dist)
            all_indices.append(atom_idx)

    # Convert to arrays for sorting
    all_distances = np.array(all_distances)
    all_indices = np.array(all_indices)
    
    # Sort by distance and take N nearest (may include same atom multiple times)
    # We assume if we can find 6 neighbors, it's an octahedron
    sorted_idx = np.argsort(all_distances)
    
    # Check if we have enough total images to get target_neighbors
    if len(all_distances) < target_neighbors:
        return None, None
    
    final_indices = all_indices[sorted_idx[:target_neighbors]]
    final_distances = all_distances[sorted_idx[:target_neighbors]]

    return final_indices, final_distances


def build_octahedra_ligand_info(
    graph: nx.Graph,
    atom_symbols: Optional[List[str]] = None,
) -> Tuple[List[Dict], List[List[int]]]:
    """Build octahedra info and ligand neighbor lists from a structural graph.

    Walks octahedron --contains(role=center)--> B atom --bonded_to(role=ligand)--> X
    and classifies ligands using atom node flags (is_terminal, is_interlayer, etc.).

    Parameters
    ----------
    graph : nx.Graph
        Structural connectivity graph from ``_graph_inorganic_ontology``.
    atom_symbols : list of str, optional
        Chemical symbols for all atoms (for central_atom_symbol in output).

    Returns
    -------
    tuple
        (octahedra_info, neighbor_indices) where:
        - octahedra_info: list of dicts with id, central_atom_index, terminal_atoms,
          interlayer_atoms, intralayer_atoms, ligand_atoms
        - neighbor_indices: list of ligand index lists (one per octahedron), same order
    """
    octahedra_info: List[Dict] = []
    neighbor_indices: List[List[int]] = []

    for node, data in graph.nodes(data=True):
        if data.get('node_type') != 'octahedron':
            continue

        central_idx = None
        b_atom_node = None

        for neighbor in graph.neighbors(node):
            edge_data = graph.get_edge_data(node, neighbor)
            if (
                edge_data
                and edge_data.get('edge_type') == 'contains'
                and edge_data.get('role') == 'center'
            ):
                neighbor_data = graph.nodes.get(neighbor, {})
                if neighbor_data.get('node_type') == 'atom':
                    atom_idx = neighbor_data.get('vasp_index')
                    if atom_idx is not None:
                        central_idx = atom_idx
                        b_atom_node = neighbor
                        break

        terminal_atoms: List[int] = []
        interlayer_atoms: List[int] = []
        intralayer_atoms: List[int] = []

        if b_atom_node:
            for neighbor in graph.neighbors(b_atom_node):
                edge_data = graph.get_edge_data(b_atom_node, neighbor)
                if (
                    edge_data
                    and edge_data.get('edge_type') == 'bonded_to'
                    and edge_data.get('role') == 'ligand'
                ):
                    neighbor_data = graph.nodes.get(neighbor, {})
                    if neighbor_data.get('node_type') == 'atom':
                        atom_idx = neighbor_data.get('vasp_index')
                        if atom_idx is not None:
                            if neighbor_data.get('is_terminal', False):
                                terminal_atoms.append(atom_idx)
                            elif neighbor_data.get('is_interlayer', False):
                                interlayer_atoms.append(atom_idx)
                            else:
                                intralayer_atoms.append(atom_idx)

        ligand_atoms = terminal_atoms + interlayer_atoms + intralayer_atoms

        oct_info = {
            'id': node,
            'central_atom_index': central_idx,
            'terminal_atoms': terminal_atoms,
            'interlayer_atoms': interlayer_atoms,
            'intralayer_atoms': intralayer_atoms,
            'ligand_atoms': ligand_atoms,
        }
        if atom_symbols is not None and central_idx is not None:
            oct_info['central_atom_symbol'] = atom_symbols[central_idx]
        else:
            oct_info['central_atom_symbol'] = None

        octahedra_info.append(oct_info)
        neighbor_indices.append(ligand_atoms)

    return octahedra_info, neighbor_indices


def find_shared_atoms(neighbor_indices_list):
    """
    Find which octahedra share atoms.

    Parameters
    ----------
    neighbor_indices_list : list of lists
        Each inner list contains atom indices for an octahedron
        May contain duplicate indices when same atom appears through PBC

    Returns
    -------
    dict
        Mapping of octahedra pairs to their shared atom indices
        e.g., {(0,1): [2, 5], (0,2): [3]}
        For self-sharing through PBC: {(0,0): [shared_atoms]} when octahedron 0
        has the same atom appearing multiple times in its neighbor list
    """
    shared_atoms = {}
    n_octahedra = len(neighbor_indices_list)

    if n_octahedra == 0:
        return shared_atoms

    neighbor_arrays = [np.array(indices, dtype=np.int32) for indices in neighbor_indices_list]

    # Detect self-sharing: when the same atom appears multiple times in an octahedron's neighbor list
    # This happens when an octahedron shares atoms with itself through PBC
    for i in range(n_octahedra):
        neighbors = neighbor_arrays[i]
        # Find atoms that appear multiple times (self-sharing through PBC)
        unique, counts = np.unique(neighbors, return_counts=True)
        self_shared = unique[counts > 1].tolist()
        if len(self_shared) > 0:
            # Mark as self-sharing: (i, i) means octahedron i shares with itself
            shared_atoms[(i, i)] = self_shared

    # Find sharing between different octahedra
    for i in range(n_octahedra):
        for j in range(i + 1, n_octahedra):
            # Use unique values to find shared atoms between different octahedra
            unique_i = np.unique(neighbor_arrays[i])
            unique_j = np.unique(neighbor_arrays[j])
            shared = np.intersect1d(unique_i, unique_j)
            if len(shared) > 0:
                shared_atoms[(i, j)] = shared.tolist()

    return shared_atoms


def classify_atoms_by_topology(neighbor_indices_list, atom_positions=None, center_atom_indices=None, cell=None):
    """
    Classify atoms using B-site offset along the cell c-axis to identify axial vs equatorial.

    Convention (3D structures): axial = along c (stacking direction), equatorial = in ab-plane (XY).
    Uses the cell's c-vector so classification is correct for any cell orientation in Cartesian space.

    Algorithm:
    1. For each octahedra, rank its 6 X neighbors by |offset along c| from B-site
    2. The 2 with LARGEST |offset along c| = axial (along c, above/below B-site)
    3. The 4 with SMALLEST |offset along c| = equatorial (in ab-plane with B-site)
    4. No threshold - just relative ranking within each octahedra
    5. Propagate within slab via equatorial connections (same layer)
    6. Follow axial connections to next layer, repeat propagation

    Classification:
    - Terminal atoms: belong to only 1 octahedron (axial at surface)
    - Equatorial atoms: shared by octahedra in the SAME layer
    - Axial interlayer atoms: shared by octahedra in DIFFERENT layers

    Parameters
    ----------
    neighbor_indices_list : list of lists
        Each inner list contains atom indices for an octahedron
    atom_positions : np.ndarray
        Array of atom positions (Cartesian, required for offset calculation)
    center_atom_indices : list
        List of B-site atom indices (required for Z-offset calculation)
    cell : np.ndarray
        Unit cell matrix (3x3, rows = a, b, c). Required for PBC and c-axis direction.

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

    if atom_positions is None or cell is None or center_atom_indices is None:
        raise ValueError("atom_positions, center_atom_indices, and cell are required")

    atom_positions = np.asarray(atom_positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)

    # =========================================================================
    # STEP 1: Get index-based shared atoms and build helper structures
    # =========================================================================
    shared_atoms = find_shared_atoms(neighbor_indices_list)

    # Collect all X atom indices
    all_x_indices = set()
    for neighbors in neighbor_indices_list:
        for n in neighbors:
            all_x_indices.add(int(n))

    # Build atom_to_octahedra: which octahedra contain each atom index
    atom_to_octahedra = {}
    for atom_idx in all_x_indices:
        atom_to_octahedra[atom_idx] = {
            oct_idx for oct_idx in range(n_octahedra)
            if atom_idx in neighbor_indices_list[oct_idx]
        }

    # Build oct_neighbors: which octahedra share at least one atom
    oct_neighbors = {}
    for (i, j), shared in shared_atoms.items():
        if i == j or not shared:
            continue
        oct_neighbors.setdefault(i, set()).add(j)
        oct_neighbors.setdefault(j, set()).add(i)

    # =========================================================================
    # STEP 2: Classify X atoms as axial/equatorial by offset along cell c-axis
    # Axial = along c (stacking direction), equatorial = in ab-plane (XY).
    # Use c-vector from cell so convention holds for any cell orientation.
    # =========================================================================
    offsets = _get_27_image_offsets()
    c_vec = np.asarray(cell[2], dtype=np.float64)
    c_norm = np.linalg.norm(c_vec)
    if c_norm < 1e-10:
        c_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # fallback
    else:
        c_hat = c_vec / c_norm

    # Track which atoms are classified as axial by each octahedra
    axial_votes = {}  # atom_idx -> count of octahedra that classify it as axial

    for oct_idx, neighbors in enumerate(neighbor_indices_list):
        b_site_idx = center_atom_indices[oct_idx]
        b_site_pos = atom_positions[b_site_idx]

        # For each neighbor: minimum-image B→X vector, then |projection onto c|
        neighbor_axial_offsets = []
        for neighbor_idx in neighbors:
            neighbor_idx = int(neighbor_idx)
            neighbor_pos = atom_positions[neighbor_idx]

            min_dist = np.inf
            best_vec = None
            for offset in offsets:
                translated = neighbor_pos + offset @ cell
                vec = translated - b_site_pos
                dist = np.linalg.norm(vec)
                if dist < min_dist:
                    min_dist = dist
                    best_vec = vec

            if best_vec is not None:
                # Offset along c (absolute): axial = along c, equatorial = in ab-plane
                axial_offset = abs(np.dot(best_vec, c_hat))
                neighbor_axial_offsets.append((neighbor_idx, axial_offset))

        # Sort by axial offset (ascending): smallest = equatorial, largest = axial
        neighbor_axial_offsets.sort(key=lambda x: x[1])

        # The 2 with LARGEST offset along c are axial
        n_axial = min(2, len(neighbor_axial_offsets))
        axial_neighbors = [x[0] for x in neighbor_axial_offsets[-n_axial:]]

        for atom_idx in axial_neighbors:
            axial_votes[atom_idx] = axial_votes.get(atom_idx, 0) + 1

    # An atom is globally axial if ANY octahedra classifies it as axial
    # (this ensures interlayer atoms connecting different octahedra are caught)
    axial_atoms = set(axial_votes.keys())

    # =========================================================================
    # STEP 3: Define connection types
    # =========================================================================
    def _is_equatorial_connection(oct_a, oct_b):
        """
        True if oct_a and oct_b are connected within the same layer.
        Same layer if they share at least one equatorial atom (or ≥2 atoms).
        """
        key = (min(oct_a, oct_b), max(oct_a, oct_b))
        shared_list = shared_atoms.get(key, [])

        if len(shared_list) >= 2:
            return True

        # Single shared atom: equatorial if NOT classified as axial
        for atom_idx in shared_list:
            if atom_idx not in axial_atoms:
                return True
        return False

    def _is_axial_connection(oct_a, oct_b):
        """True if oct_a and oct_b share exactly one axial atom."""
        key = (min(oct_a, oct_b), max(oct_a, oct_b))
        shared_list = shared_atoms.get(key, [])
        if len(shared_list) != 1:
            return False
        return shared_list[0] in axial_atoms

    # =========================================================================
    # STEP 4: Handle edge case - no sharing between octahedra
    # =========================================================================
    pairs_between_octahedra = [(i, j) for (i, j) in shared_atoms if i != j]
    if not pairs_between_octahedra:
        atom_classification = {}
        for atom_idx in all_x_indices:
            atom_idx = int(atom_idx)
            num_oct = len(atom_to_octahedra.get(atom_idx, set()))
            atom_classification[atom_idx] = 'terminal' if num_oct == 1 else 'equatorial'

        layer_membership = {oct_idx: 0 for oct_idx in range(n_octahedra)}

        octahedra_geometries = []
        for oct_idx, neighbors in enumerate(neighbor_indices_list):
            geometry = {}
            for neighbor_idx in neighbors:
                neighbor_idx = int(neighbor_idx)
                if atom_classification.get(neighbor_idx) == 'terminal':
                    geometry[neighbor_idx] = 'axial_terminal'
                else:
                    geometry[neighbor_idx] = 'equatorial'
            octahedra_geometries.append(geometry)

        return atom_classification, layer_membership, octahedra_geometries

    # =========================================================================
    # STEP 5: Layer propagation algorithm
    # =========================================================================
    layer_membership = {}
    unassigned = set(range(n_octahedra))
    current_layer_id = 0

    def _propagate_within_layer(seed_octahedra, layer_id):
        """
        Propagate layer assignment via equatorial connections.
        Returns all octahedra in the same layer.
        """
        layer_octs = set(seed_octahedra)
        frontier = set(seed_octahedra)

        while frontier:
            new_frontier = set()
            for oct_idx in frontier:
                for neighbor in oct_neighbors.get(oct_idx, set()):
                    if neighbor in layer_membership:
                        continue
                    if neighbor in layer_octs:
                        continue
                    if _is_equatorial_connection(oct_idx, neighbor):
                        layer_membership[neighbor] = layer_id
                        layer_octs.add(neighbor)
                        new_frontier.add(neighbor)
            frontier = new_frontier

        return layer_octs

    # Main loop: seed a layer, propagate, follow axial to next layer, repeat
    max_iterations = 100
    iteration = 0

    while unassigned and iteration < max_iterations:
        # Pick a seed from unassigned octahedra
        seed = next(iter(unassigned))
        layer_membership[seed] = current_layer_id
        unassigned.discard(seed)

        # Propagate within this layer
        current_layer_octs = _propagate_within_layer({seed}, current_layer_id)
        unassigned -= current_layer_octs

        # Follow axial connections to next layer
        while True:
            axial_next = set()
            for oct_idx in current_layer_octs:
                for neighbor in oct_neighbors.get(oct_idx, set()):
                    if neighbor in layer_membership:
                        continue
                    if _is_axial_connection(oct_idx, neighbor):
                        axial_next.add(neighbor)

            if not axial_next:
                break

            # Assign next layer
            current_layer_id += 1
            for oct in axial_next:
                layer_membership[oct] = current_layer_id
                unassigned.discard(oct)

            # Propagate within new layer
            current_layer_octs = _propagate_within_layer(axial_next, current_layer_id)
            unassigned -= current_layer_octs

        # Move to next layer for any remaining unassigned
        current_layer_id += 1
        iteration += 1

    # Assign any remaining octahedra to their own layers
    for oct_idx in unassigned:
        layer_membership[oct_idx] = current_layer_id
        current_layer_id += 1

    # =========================================================================
    # STEP 6: Merge layers that are actually connected by equatorial
    # =========================================================================
    parent = {}

    def _find(lid):
        if lid not in parent:
            parent[lid] = lid
        if parent[lid] != lid:
            parent[lid] = _find(parent[lid])
        return parent[lid]

    def _union(lid_a, lid_b):
        pa, pb = _find(lid_a), _find(lid_b)
        if pa != pb:
            parent[pa] = pb

    for (i, j) in pairs_between_octahedra:
        if _is_equatorial_connection(i, j) and i in layer_membership and j in layer_membership:
            _union(layer_membership[i], layer_membership[j])

    # Relabel to contiguous layer IDs
    for oct_idx in range(n_octahedra):
        layer_membership[oct_idx] = _find(layer_membership[oct_idx])

    unique_ids = sorted(set(layer_membership.values()))
    remap = {old: new for new, old in enumerate(unique_ids)}
    for oct_idx in range(n_octahedra):
        layer_membership[oct_idx] = remap[layer_membership[oct_idx]]

    # =========================================================================
    # STEP 7: Classify atoms based on layer membership
    # =========================================================================
    atom_classification = {}
    for atom_idx in all_x_indices:
        atom_idx = int(atom_idx)
        octahedra = atom_to_octahedra.get(atom_idx, set())
        num_oct = len(octahedra)

        if num_oct == 1:
            atom_classification[atom_idx] = 'terminal'
        elif num_oct == 2:
            oct_list = list(octahedra)
            layer_0 = layer_membership.get(oct_list[0])
            layer_1 = layer_membership.get(oct_list[1])
            if layer_0 != layer_1:
                atom_classification[atom_idx] = 'axial_interlayer'
            else:
                atom_classification[atom_idx] = 'equatorial'
        else:
            # 3+ octahedra: check if all same layer
            layers = {layer_membership.get(o) for o in octahedra}
            atom_classification[atom_idx] = 'axial_interlayer' if len(layers) > 1 else 'equatorial'

    # =========================================================================
    # STEP 8: Build per-octahedron geometry dictionaries
    # =========================================================================
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
        all_neighbor_indices,
        atom_positions=atom_positions,
        center_atom_indices=center_atom_indices,
        cell=cell
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

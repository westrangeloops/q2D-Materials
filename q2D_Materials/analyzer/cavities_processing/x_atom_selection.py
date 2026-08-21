"""X atom selection for cavity cages (cuboctahedron and antiprism).

Selects which X atoms form cage edges around B atoms, using topology and
weighted scoring with PBC-aware geometry.
"""

import sys
import numpy as np
import networkx as nx
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

from .weights import ASiteWeights, SpacerWeights
from ...utils.geometry.pbc_distances import find_nearest_image_positions


def _get_cuboctahedron_x_atoms(
    graph: nx.Graph,
    b_indices: np.ndarray,
    atom_positions: np.ndarray,
    cell: np.ndarray,
    anchor_position: np.ndarray,
    weights: ASiteWeights,
    b_positions_unwrapped: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract X atoms that form cuboctahedron edges from B atoms.

    For a cuboctahedron, each of the 8 B-atoms connects to 3 X atoms (not all 6 ligands).
    X atoms are selected topologically: those shared by 2+ B atoms in the cage.
    Positions are calculated relative to anchor (A-site center or NH3 N atom).

    Parameters
    ----------
    graph : nx.Graph
        Parent graph with B-X connectivity
    b_indices : np.ndarray
        Indices of the 8 B atoms forming cuboctahedron corners
    atom_positions : np.ndarray
        All atom positions (wrapped)
    cell : np.ndarray
        Unit cell matrix
    anchor_position : np.ndarray
        Anchor position (A-site center or NH3 N atom). X atoms are selected
        and positioned relative to this anchor.
    weights : ASiteWeights
        Weight configuration for X atom scoring/selection.
    b_positions_unwrapped : np.ndarray, optional
        PBC unwrapped B atom positions. Used for initial collection only.
        Shape: (len(b_indices), 3). If None, uses wrapped positions.

    Returns
    -------
    tuple
        (x_indices, x_positions, x_distances, x_labels) for cuboctahedron X atoms
    """
    # Collect all X-ligands from the 8 B atoms
    x_atom_occurrences = Counter()  # Count how many B atoms share each X
    x_atom_indices_set = set()  # Track unique X atoms

    for b_idx in b_indices:
        b_node = f'atom_{b_idx}'
        if b_node not in graph:
            continue

        # Find all X-ligands bonded to this B atom
        for neighbor in graph.neighbors(b_node):
            edge_data = graph.get_edge_data(b_node, neighbor)
            if (edge_data and
                edge_data.get('edge_type') == 'bonded_to' and
                edge_data.get('role') == 'ligand'):

                neighbor_data = graph.nodes.get(neighbor, {})
                if neighbor_data.get('node_type') == 'atom':
                    x_idx = neighbor_data.get('vasp_index')
                    if x_idx is not None:
                        # Count this X atom
                        x_atom_occurrences[x_idx] += 1
                        x_atom_indices_set.add(x_idx)

    # Select X atoms that are shared by 2+ B atoms (these form cuboctahedron edges)
    # In a perfect cuboctahedron, each edge X is shared by 2 corner B atoms
    cuboctahedron_x_indices = [
        x_idx for x_idx, count in x_atom_occurrences.items()
        if count >= 2
    ]

    print(f"    Found {len(cuboctahedron_x_indices)} shared X atoms (expected 12)", file=sys.stderr)

    # ALWAYS apply weighted scoring to select the best 12 X atoms from all 27 PBC images
    # This ensures we get geometrically correct X atoms even when topology gives us wrong candidates
    if True:  # Always apply weighted scoring for A-sites
        print(f"    Applying weighted scoring to all {len(cuboctahedron_x_indices)} candidates", file=sys.stderr)

        # Get anchor Z for Z-difference calculation
        anchor_z = anchor_position[2]

        # Explore ALL 27 PBC images of each candidate X atom and score them
        all_x_candidates = []
        for x_idx in cuboctahedron_x_indices:
            x_pos_array = np.array([atom_positions[x_idx]])
            x_idx_array = np.array([x_idx])
            # Get ALL 27 PBC images
            result = find_nearest_image_positions(
                anchor_position, x_pos_array, x_idx_array, cell, n_neighbors=27
            )
            if result[0] is not None and len(result[0]) > 0:
                # Score each PBC image
                for j in range(len(result[0])):
                    pos = result[1][j]
                    dist_anchor = result[2][j]
                    label = tuple(result[3][j])
                    # Z-difference for scoring: prefer atoms in same plane as anchor
                    # Note: This is a heuristic for scoring, not a precise distance.
                    # For non-orthogonal cells, this approximates planarity preference.
                    z_diff = abs(pos[2] - anchor_z)

                    # Weighted score for A-sites using configurable weights
                    # B centroid not needed for A-sites (symmetric cuboctahedron)
                    score = weights.w_dist_anchor * dist_anchor + weights.w_z_diff * z_diff
                    all_x_candidates.append((score, x_idx, dist_anchor, pos, label, z_diff))

        # Deduplicate by (x_idx, pbc_label) to avoid duplicate (atom, PBC) pairs
        seen_x_images = {}
        for candidate in all_x_candidates:
            score, x_idx, dist_anchor, pos, label, z_diff = candidate
            key = (x_idx, label)
            if key not in seen_x_images or score < seen_x_images[key][0]:
                seen_x_images[key] = candidate

        all_x_candidates = list(seen_x_images.values())

        # Sort by score and select top 12
        all_x_candidates.sort(key=lambda x: x[0])
        top_12 = all_x_candidates[:min(12, len(all_x_candidates))]

        # Extract indices, positions, distances, and labels from top 12
        cuboctahedron_x_indices = [x[1] for x in top_12]
        x_positions_final = [x[3] for x in top_12]
        x_distances_final = [x[2] for x in top_12]
        x_labels_final = [x[4] for x in top_12]

        print(f"    Selected 12 X atoms using weighted scoring ({weights.w_dist_anchor}*dist_anchor + {weights.w_z_diff}*z_diff):", file=sys.stderr)
        for idx, candidate in enumerate(top_12):
            score, x_idx, dist_anchor, pos, label, z_diff = candidate
            print(f"      {idx+1}. X {x_idx} (PBC {label}): score={score:.3f} [dist_anchor={dist_anchor:.3f}, z_diff={z_diff:.3f}]", file=sys.stderr)

    # Build result arrays - use pre-calculated positions/distances/labels from weighted scoring
    x_indices = np.array(cuboctahedron_x_indices, dtype=np.int32)
    x_positions = np.array(x_positions_final)
    x_distances = np.array(x_distances_final)
    x_labels = np.array(x_labels_final)

    return x_indices, x_positions, x_distances, x_labels


def _get_antiprism_x_atoms(
    graph: nx.Graph,
    b_indices: np.ndarray,
    atom_positions: np.ndarray,
    cell: np.ndarray,
    anchor_position: np.ndarray,
    weights: SpacerWeights,
    b_positions_unwrapped: Optional[np.ndarray] = None
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Extract X atoms that form square antiprism edges from B atoms.

    For a square antiprism (half-cuboctahedron), each of the 4 B-atoms connects to
    2-4 X atoms. X atoms are selected topologically:
    - 4 terminal X atoms (1 per B atom, selected from each B's terminal ligands)
    - 4 equatorial X atoms (shared between B atoms in the cage)
    Positions are calculated relative to anchor (NH3 N atom).

    NEW APPROACH: For each B atom, find ALL its terminal X ligands, then select
    the one closest to the anchor. This ensures topological correctness first,
    then optimizes for geometry (avoiding off-center NH3 issues).

    Parameters
    ----------
    graph : nx.Graph
        Parent graph with B-X connectivity
    b_indices : np.ndarray
        Indices of the 4 B atoms forming antiprism corners
    atom_positions : np.ndarray
        All atom positions (wrapped)
    cell : np.ndarray
        Unit cell matrix
    anchor_position : np.ndarray
        NH3 anchor position. X atoms are selected and positioned relative to this anchor.
    weights : SpacerWeights
        Weight configuration for terminal X atom scoring/selection.
    b_positions_unwrapped : np.ndarray, optional
        PBC unwrapped B atom positions. Used for initial collection only.
        Shape: (len(b_indices), 3). If None, uses wrapped positions.

    Returns
    -------
    tuple or None
        (x_indices, x_positions, x_distances, x_labels) for antiprism X atoms,
        or None if validation fails
    """
    # Collect equatorial X atoms (shared between B atoms)
    x_equatorial_with_distances = []
    x_atom_seen_equatorial = set()

    # Collect all terminal X ligands from the unique B atoms in the cage
    terminal_x_per_b_all = {}

    unique_b_indices = np.unique(b_indices)
    
    if len(unique_b_indices) < len(b_indices):
        print(f"    DEBUG: All B atoms are clones (unique: {len(unique_b_indices)}, total: {len(b_indices)})", file=sys.stderr)
    
    for b_idx in unique_b_indices:
        b_node = f'atom_{b_idx}'
        if b_node not in graph:
            continue

        terminal_x_per_b_all[int(b_idx)] = []
        terminal_count = 0
        equatorial_count = 0
        other_count = 0

        for neighbor in graph.neighbors(b_node):
            edge_data = graph.get_edge_data(b_node, neighbor)
            if (edge_data and
                edge_data.get('edge_type') == 'bonded_to' and
                edge_data.get('role') == 'ligand'):

                neighbor_data = graph.nodes.get(neighbor, {})
                if neighbor_data.get('node_type') == 'atom':
                    x_idx = neighbor_data.get('vasp_index')
                    if x_idx is None:
                        continue

                    is_terminal = neighbor_data.get('is_terminal', False)
                    is_equatorial_node = neighbor_data.get('is_equatorial', False)
                    is_equatorial_edge = edge_data.get('geometry') == 'equatorial'
                    is_equatorial = is_equatorial_node or is_equatorial_edge
                    
                    if is_equatorial:
                        equatorial_count += 1
                    elif is_terminal:
                        terminal_count += 1
                    else:
                        other_count += 1

                    if is_equatorial:
                        if x_idx not in x_atom_seen_equatorial:
                            x_atom_seen_equatorial.add(x_idx)
                            x_pos_array = np.array([atom_positions[x_idx]])
                            x_idx_array = np.array([x_idx])
                            result = find_nearest_image_positions(
                                anchor_position, x_pos_array, x_idx_array, cell, n_neighbors=27
                            )
                            if result[0] is not None and len(result[0]) > 0:
                                for j in range(len(result[0])):
                                    distance = result[2][j]
                                    unwrapped_pos = result[1][j]
                                    label = tuple(result[3][j])
                                    x_equatorial_with_distances.append((x_idx, distance, unwrapped_pos, label))
                    elif is_terminal:
                        terminal_x_per_b_all[int(b_idx)].append(x_idx)
        
        if len(unique_b_indices) < len(b_indices):
            print(f"    DEBUG: B atom {b_idx} has {terminal_count} terminal, {equatorial_count} equatorial, {other_count} other X neighbors", file=sys.stderr)

    n_unique_b = len(np.unique(b_indices))
    is_n1 = (n_unique_b <= 2)
    anchor_z = anchor_position[2]

    all_terminal_candidates = []

    for i, b_idx in enumerate(b_indices):
        terminal_x_candidates = terminal_x_per_b_all.get(int(b_idx), [])

        for x_idx in terminal_x_candidates:
            x_pos_array = np.array([atom_positions[x_idx]])
            x_idx_array = np.array([x_idx])
            result = find_nearest_image_positions(
                anchor_position, x_pos_array, x_idx_array, cell, n_neighbors=27
            )

            if result[0] is None or len(result[0]) == 0:
                continue

            for j in range(len(result[0])):
                unwrapped_pos = result[1][j]
                distance_to_anchor = np.linalg.norm(unwrapped_pos - anchor_position)
                label = tuple(result[3][j])
                z_diff = abs(unwrapped_pos[2] - anchor_z)
                all_terminal_candidates.append((x_idx, distance_to_anchor, unwrapped_pos, label, z_diff, b_idx, i))

    if len(all_terminal_candidates) < 4:
        print(f"    WARNING: Found only {len(all_terminal_candidates)} terminal candidates", file=sys.stderr)
        return None

    print(f"    Collected {len(all_terminal_candidates)} terminal X candidates from 27 PBC images (before deduplication)", file=sys.stderr)

    seen_x_images = {}
    for candidate in all_terminal_candidates:
        x_idx, dist, pos, label, z_diff, b_idx, b_pos = candidate
        key = (x_idx, label)
        if key not in seen_x_images:
            seen_x_images[key] = candidate

    all_terminal_candidates = list(seen_x_images.values())
    print(f"    After deduplication: {len(all_terminal_candidates)} unique terminal X candidates", file=sys.stderr)

    b_positions_for_center = []
    for b_idx in b_indices:
        b_pos = atom_positions[b_idx]
        result = find_nearest_image_positions(
            anchor_position, np.array([b_pos]), np.array([b_idx]), cell, n_neighbors=1
        )
        if result[0] is not None and len(result[0]) > 0:
            b_positions_for_center.append(result[1][0])

    if len(b_positions_for_center) == 0:
        print(f"    ERROR: Could not calculate B atom geometric center", file=sys.stderr)
        return None

    b_geometric_center = np.mean(b_positions_for_center, axis=0)
    print(f"    B geometric center: [{b_geometric_center[0]:.3f}, {b_geometric_center[1]:.3f}, {b_geometric_center[2]:.3f}]", file=sys.stderr)

    print(f"    Using weighted scoring: {weights.w_dist_anchor}*dist_anchor + {weights.w_dist_b_center}*dist_b_center + {weights.w_z_diff}*z_diff", file=sys.stderr)

    scored_candidates = []
    for candidate in all_terminal_candidates:
        x_idx, dist_anchor, pos, label, z_diff, b_idx, b_pos = candidate
        dist_to_b_center = np.linalg.norm(pos - b_geometric_center)
        score = (weights.w_dist_anchor * dist_anchor +
                 weights.w_dist_b_center * dist_to_b_center +
                 weights.w_z_diff * z_diff)
        scored_candidates.append((score, x_idx, dist_anchor, pos, label, z_diff, dist_to_b_center))

    scored_candidates.sort(key=lambda x: x[0])
    n_candidates_to_consider = 8 if is_n1 else 12
    top_candidates = scored_candidates[:min(n_candidates_to_consider, len(scored_candidates))]
    terminal_validated = [(x[1], x[2], x[3], x[4]) for x in top_candidates[:4]]

    print(f"    Selected 4 terminal X atoms from top {n_candidates_to_consider} scored candidates:", file=sys.stderr)
    for idx, candidate in enumerate(top_candidates[:4]):
        score, x_idx, dist_anchor, pos, label, z_diff, dist_b_center = candidate
        print(f"      {idx+1}. Terminal X {x_idx} (PBC {label}): score={score:.3f} [dist_anchor={dist_anchor:.3f}, dist_b_center={dist_b_center:.3f}, z_diff={z_diff:.3f}]", file=sys.stderr)

    if len(terminal_validated) < 4:
        print(f"    WARNING: Found only {len(terminal_validated)} terminals (expected 4)", file=sys.stderr)
        return None

    if len(terminal_validated) != 4:
        print(f"    WARNING: Expected 4 terminal X atoms but got {len(terminal_validated)}", file=sys.stderr)
        return None

    print(f"    Collected {len(x_equatorial_with_distances)} equatorial X candidates (before deduplication)", file=sys.stderr)
    seen_equatorial = {}
    for eq_candidate in x_equatorial_with_distances:
        x_idx, dist, pos, label = eq_candidate
        key = (x_idx, label)
        if key not in seen_equatorial:
            seen_equatorial[key] = eq_candidate
    x_equatorial_with_distances = list(seen_equatorial.values())
    print(f"    After deduplication: {len(x_equatorial_with_distances)} unique equatorial X candidates", file=sys.stderr)

    x_equatorial_with_distances.sort(key=lambda x: x[1])

    if len(x_equatorial_with_distances) < 4:
        print(f"    WARNING: Found only {len(x_equatorial_with_distances)} equatorial X atoms (expected 4)", file=sys.stderr)
        print(f"    Structure not currently supported: insufficient equatorial X atoms for half-cage", file=sys.stderr)
        return None

    equatorial_validated = x_equatorial_with_distances[:4]
    print(f"    Selected 4 equatorial X atoms (closest to anchor)", file=sys.stderr)

    if len(terminal_validated) + len(equatorial_validated) != 8:
        print(f"    WARNING: Expected 8 X atoms total (4 terminal + 4 equatorial) but got {len(terminal_validated) + len(equatorial_validated)}", file=sys.stderr)
        return None

    antiprism_x_data = terminal_validated + equatorial_validated

    x_indices = np.array([x[0] for x in antiprism_x_data], dtype=np.int32)
    x_positions = np.array([x[2] for x in antiprism_x_data])
    x_distances = np.array([x[1] for x in antiprism_x_data])
    x_labels = np.array([x[3] for x in antiprism_x_data])

    print(f"    ✓ Selected 4 terminal X atoms (1 per B) + 4 equatorial X atoms", file=sys.stderr)
    return x_indices, x_positions, x_distances, x_labels

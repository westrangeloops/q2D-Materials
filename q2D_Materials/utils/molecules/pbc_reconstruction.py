"""PBC-aware molecule reconstruction using graph connectivity.

This module reconstructs molecules with correct periodic boundary condition (PBC)
coordinates by traversing graph bonds and finding the nearest PBC image for each
bonded atom. This ensures molecular connectivity is preserved even when molecules
span cell boundaries.

Algorithm:
1. Start from an anchor point (e.g., NH3 center)
2. Place anchor atoms using nearest PBC image
3. Traverse graph bonds (BFS) to connected atoms
4. For each bonded atom, find its nearest PBC image relative to the already-placed atom
5. Build complete molecule with consistent PBC coordinates

This approach is more robust than geometric center methods, which fail when
molecules cross cell boundaries.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Union
from collections import deque

from ..geometry.pbc_distances import (
    calculate_pbc_distances,
    find_nearest_image_positions,
)


def reconstruct_molecule_pbc(
    molecule_indices: List[int],
    anchor_position: np.ndarray,
    graph: nx.Graph,
    atom_positions: np.ndarray,
    atom_symbols: List[str],
    cell: np.ndarray,
    pbc: Union[bool, List[bool]] = True,
    anchor_atom_indices: Optional[List[int]] = None,
) -> Dict[int, Tuple[np.ndarray, Tuple[int, int, int]]]:
    """Reconstruct molecule with correct PBC coordinates using graph connectivity.
    
    Starting from an anchor position (e.g., NH3 center), this function traverses
    the graph bonds to place each atom at its nearest PBC image relative to its
    bonded neighbors. This ensures correct molecular connectivity even when the
    molecule spans periodic boundaries.
    
    Parameters
    ----------
    molecule_indices : list of int
        Atom indices (VASP indices) of all atoms in the molecule
    anchor_position : np.ndarray
        Anchor position (3D) to start reconstruction from. Typically an NH3 center
        or geometric center of a molecular group.
    graph : nx.Graph
        Structural graph with BONDED_TO edges connecting atoms within molecules
    atom_positions : np.ndarray
        All atom positions in the structure. Shape: (N_atoms, 3)
    atom_symbols : list of str
        All atom symbols
    cell : np.ndarray
        Unit cell matrix. Shape: (3, 3)
    pbc : bool or list of bool, default=True
        Periodic boundary conditions. True for full 3D PBC, [True, True, False]
        for XY-only, etc.
    anchor_atom_indices : list of int or None, default=None
        Specific atom indices to use as initial anchors. If None, automatically
        finds atoms near anchor_position. Useful for NH3 groups where you want
        to start from N and 3H atoms.
        
    Returns
    -------
    dict
        Mapping from atom index to (cartesian_position, image_label):
        {
            atom_idx: (np.ndarray([x, y, z]), (i, j, k))
        }
        where image_label (i, j, k) indicates which PBC image was used.
        [0, 0, 0] is the original unit cell.
    
    Examples
    --------
    Reconstruct an A-site molecule starting from NH3 center:
    
    >>> import numpy as np
    >>> import networkx as nx
    >>> from q2D_Materials.utils.molecules.pbc_reconstruction import reconstruct_molecule_pbc
    >>> 
    >>> # Get molecule indices and graph
    >>> mol_indices = [10, 11, 12, 13, 14]  # Example molecule
    >>> nh3_center = np.array([5.0, 5.0, 5.0])
    >>> 
    >>> # Reconstruct
    >>> positions = reconstruct_molecule_pbc(
    ...     mol_indices, nh3_center, graph, atom_positions, atom_symbols, cell
    ... )
    >>> 
    >>> # Access positions
    >>> atom_10_pos, img_label = positions[10]
    >>> print(f"Atom 10 at {atom_10_pos} in image {img_label}")
    
    Notes
    -----
    - Uses BFS traversal to ensure all atoms are placed relative to their
      bonded neighbors, maintaining correct bond lengths
    - Leverages numba-accelerated PBC distance calculations for speed
    - Works for any molecule shape/orientation, including those crossing
      cell boundaries
    """
    # Convert inputs to numpy arrays
    molecule_indices = list(molecule_indices)
    anchor_position = np.asarray(anchor_position, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    
    # Initialize reconstruction mapping
    # Maps atom_idx -> (cartesian_position, image_label)
    reconstructed = {}
    
    # Set of atom indices in molecule for fast lookup
    molecule_set = set(molecule_indices)
    
    # Step 1: Place anchor atoms (starting points)
    if anchor_atom_indices is not None:
        # Use provided anchor atoms
        anchor_atoms = [idx for idx in anchor_atom_indices if idx in molecule_set]
    else:
        # Find atoms near anchor position (within 2 Å)
        anchor_atoms = _find_atoms_near_position(
            anchor_position, molecule_indices, atom_positions, cell, pbc, cutoff=2.0
        )
    
    if not anchor_atoms:
        # Fallback: use first atom in molecule
        anchor_atoms = [molecule_indices[0]]
    
    # Place anchor atoms using nearest PBC images to anchor_position
    for anchor_idx in anchor_atoms:
        if anchor_idx in reconstructed:
            continue
        
        # Find nearest image of this atom to anchor position
        anchor_pos = atom_positions[anchor_idx]
        anchor_pos_array = np.array([anchor_pos])
        anchor_idx_array = np.array([anchor_idx])
        
        # Get nearest image
        nearest_indices, nearest_positions, _, nearest_labels = find_nearest_image_positions(
            anchor_position,
            anchor_pos_array,
            anchor_idx_array,
            cell,
            n_neighbors=1,
            pbc=pbc
        )
        
        if len(nearest_indices) > 0:
            reconstructed[anchor_idx] = (
                nearest_positions[0],
                tuple(nearest_labels[0])
            )
    
    # Step 2: BFS traversal to place remaining atoms
    # Queue: (atom_idx, parent_atom_idx)
    # We place each atom relative to its bonded parent
    queue = deque([(atom_idx, None) for atom_idx in anchor_atoms])
    
    while queue:
        current_idx, parent_idx = queue.popleft()
        
        # Get current atom's position (should already be placed)
        if current_idx not in reconstructed:
            continue
        
        current_pos, current_img = reconstructed[current_idx]
        
        # Find bonded neighbors in graph
        current_node = f'atom_{current_idx}'
        if current_node not in graph.nodes:
            continue
        
        # Get neighbors connected by BONDED_TO edges
        neighbors = []
        for neighbor_node in graph.neighbors(current_node):
            edge_data = graph.get_edge_data(current_node, neighbor_node)
            if edge_data and edge_data.get('edge_type') == 'bonded_to':
                # Extract atom index from node ID
                if neighbor_node.startswith('atom_'):
                    neighbor_idx = graph.nodes[neighbor_node].get('vasp_index')
                    if neighbor_idx is not None and neighbor_idx in molecule_set:
                        neighbors.append(neighbor_idx)
        
        # Place each unplaced neighbor
        for neighbor_idx in neighbors:
            if neighbor_idx in reconstructed:
                continue
            
            # Find nearest PBC image of neighbor relative to current atom
            neighbor_pos = atom_positions[neighbor_idx]
            neighbor_pos_array = np.array([neighbor_pos])
            neighbor_idx_array = np.array([neighbor_idx])
            
            # Use calculate_pbc_distances with return_vectors to get the actual
            # position of the nearest image
            distances, vectors = calculate_pbc_distances(
                current_pos,
                neighbor_pos_array,
                cell,
                pbc=pbc,
                mode='extended',
                return_vectors=True
            )
            
            # The nearest image position is: current_pos + vector
            nearest_image_pos = current_pos + vectors[0]
            
            # Calculate which image label this corresponds to
            # We need to find which of the 27 images gives this position
            image_label = _find_image_label(
                neighbor_pos, nearest_image_pos, cell
            )
            
            # Store reconstructed position
            reconstructed[neighbor_idx] = (nearest_image_pos, image_label)
            
            # Add to queue to process its neighbors
            queue.append((neighbor_idx, current_idx))
    
    # Step 3: Verify all atoms were placed
    missing = [idx for idx in molecule_indices if idx not in reconstructed]
    if missing:
        # Fallback: place missing atoms using nearest image to anchor
        for missing_idx in missing:
            missing_pos = atom_positions[missing_idx]
            missing_pos_array = np.array([missing_pos])
            missing_idx_array = np.array([missing_idx])
            
            nearest_indices, nearest_positions, _, nearest_labels = find_nearest_image_positions(
                anchor_position,
                missing_pos_array,
                missing_idx_array,
                cell,
                n_neighbors=1,
                pbc=pbc
            )
            
            if len(nearest_indices) > 0:
                reconstructed[missing_idx] = (
                    nearest_positions[0],
                    tuple(nearest_labels[0])
                )
    
    return reconstructed


def _find_atoms_near_position(
    position: np.ndarray,
    candidate_indices: List[int],
    atom_positions: np.ndarray,
    cell: np.ndarray,
    pbc: Union[bool, List[bool]],
    cutoff: float = 2.0
) -> List[int]:
    """Find atom indices near a given position within cutoff distance.
    
    Parameters
    ----------
    position : np.ndarray
        Reference position. Shape: (3,)
    candidate_indices : list of int
        Atom indices to search
    atom_positions : np.ndarray
        All atom positions
    cell : np.ndarray
        Unit cell matrix
    pbc : bool or list of bool
        Periodic boundary conditions
    cutoff : float, default=2.0
        Distance cutoff in Angstroms
        
    Returns
    -------
    list of int
        Atom indices within cutoff distance
    """
    if not candidate_indices:
        return []
    
    candidate_positions = np.array([atom_positions[idx] for idx in candidate_indices])
    candidate_indices_array = np.array(candidate_indices)
    
    # Calculate PBC distances
    distances = calculate_pbc_distances(
        position,
        candidate_positions,
        cell,
        pbc=pbc,
        mode='extended'
    )
    
    # Find atoms within cutoff
    near_indices = [
        candidate_indices[i] for i, dist in enumerate(distances)
        if dist < cutoff
    ]
    
    return near_indices


def _find_image_label(
    original_pos: np.ndarray,
    image_pos: np.ndarray,
    cell: np.ndarray
) -> Tuple[int, int, int]:
    """Find the PBC image label (i, j, k) for a translated position.
    
    Given an original position and its translated image position, determine
    which periodic image (i, j, k) was used.
    
    Parameters
    ----------
    original_pos : np.ndarray
        Original atom position. Shape: (3,)
    image_pos : np.ndarray
        Translated image position. Shape: (3,)
    cell : np.ndarray
        Unit cell matrix. Shape: (3, 3)
        
    Returns
    -------
    tuple of int
        Image label (i, j, k) where each is in {-1, 0, 1}
    """
    # Calculate displacement
    displacement = image_pos - original_pos
    
    # Convert to fractional coordinates
    inv_cell = np.linalg.inv(cell)
    displacement_frac = displacement @ inv_cell.T
    
    # Round to nearest integer (should be -1, 0, or 1 for 27-image search)
    image_frac = np.round(displacement_frac).astype(int)
    
    # Clamp to [-1, 1] range
    image_frac = np.clip(image_frac, -1, 1)
    
    return tuple(image_frac.tolist())


def reconstruct_molecule_from_nh3(
    molecule_indices: List[int],
    nh3_center: np.ndarray,
    graph: nx.Graph,
    atom_positions: np.ndarray,
    atom_symbols: List[str],
    cell: np.ndarray,
    pbc: Union[bool, List[bool]] = True,
) -> Dict[int, Tuple[np.ndarray, Tuple[int, int, int]]]:
    """Reconstruct molecule starting from NH3 group.
    
    This function reconstructs the molecule starting from the N atom and the 3 closest
    H atoms. The reconstruction begins with the N atom as the primary anchor, then
    finds the 3 closest H atoms to that N (within bond cutoff), and uses all 4 atoms
    (N + 3H) as initial anchors for BFS traversal.
    
    For DJ spacers, this ensures the molecule is properly reconstructed starting from
    the N atom with the closest H3 group, maintaining correct molecular connectivity.
    
    Parameters
    ----------
    molecule_indices : list of int
        Atom indices in the molecule
    nh3_center : np.ndarray
        NH3 anchor position (typically the N atom position, not center of mass)
    graph : nx.Graph
        Structural graph
    atom_positions : np.ndarray
        All atom positions
    atom_symbols : list of str
        All atom symbols
    cell : np.ndarray
        Unit cell matrix
    pbc : bool or list of bool, default=True
        Periodic boundary conditions
        
    Returns
    -------
    dict
        Same as reconstruct_molecule_pbc
    """
    # Find NH3 atoms: N atom closest to nh3_center, then 3 closest H atoms to that N
    nh3_atoms = _find_nh3_atoms(
        nh3_center, molecule_indices, atom_positions, atom_symbols, cell, pbc
    )
    
    return reconstruct_molecule_pbc(
        molecule_indices,
        nh3_center,
        graph,
        atom_positions,
        atom_symbols,
        cell,
        pbc=pbc,
        anchor_atom_indices=nh3_atoms
    )


def _find_nh3_atoms(
    nh3_center: np.ndarray,
    molecule_indices: List[int],
    atom_positions: np.ndarray,
    atom_symbols: List[str],
    cell: np.ndarray,
    pbc: Union[bool, List[bool]],
    nh_bond_cutoff: float = 1.2
) -> List[int]:
    """Find atom indices belonging to NH3 group near nh3_center.
    
    This function finds the N atom closest to nh3_center, then finds the 3 closest
    H atoms to that N atom (within bond cutoff). This ensures proper reconstruction
    starting from the N atom with the closest H3 group.
    
    Parameters
    ----------
    nh3_center : np.ndarray
        NH3 anchor position (typically the N atom position)
    molecule_indices : list of int
        Atom indices in molecule
    atom_positions : np.ndarray
        All atom positions
    atom_symbols : list of str
        All atom symbols
    cell : np.ndarray
        Unit cell matrix
    pbc : bool or list of bool
        Periodic boundary conditions
    nh_bond_cutoff : float, default=1.2
        N-H bond distance cutoff in Angstroms
        
    Returns
    -------
    list of int
        Atom indices: [N_idx, H1_idx, H2_idx, H3_idx] where H atoms are sorted
        by distance to N (closest first)
    """
    # Step 1: Find N atom closest to nh3_center
    n_candidates = [idx for idx in molecule_indices if atom_symbols[idx] == 'N']
    
    if not n_candidates:
        return []
    
    # Find N closest to nh3_center
    n_positions = np.array([atom_positions[idx] for idx in n_candidates])
    n_indices_array = np.array(n_candidates)
    
    distances = calculate_pbc_distances(
        nh3_center, n_positions, cell, pbc=pbc, mode='extended'
    )
    
    closest_n_idx = n_candidates[np.argmin(distances)]
    n_pos = atom_positions[closest_n_idx]
    
    # Step 2: Find the 3 closest H atoms to this N atom
    # Calculate distances to all H candidates and select the 3 closest
    h_candidates = [idx for idx in molecule_indices if atom_symbols[idx] == 'H']
    nh3_atoms = [closest_n_idx]
    
    if not h_candidates:
        return nh3_atoms
    
    # Calculate distances from N to all H candidates
    h_positions = np.array([atom_positions[idx] for idx in h_candidates])
    h_distances = calculate_pbc_distances(
        n_pos, h_positions, cell, pbc=pbc, mode='extended'
    )
    
    # Create list of (distance, h_idx) pairs for H atoms within bond cutoff
    h_with_distances = [
        (h_distances[i], h_candidates[i])
        for i in range(len(h_candidates))
        if h_distances[i] < nh_bond_cutoff
    ]
    
    # Sort by distance and take the 3 closest H atoms
    # This ensures we start reconstruction with the N atom and its 3 closest H neighbors
    h_with_distances.sort(key=lambda x: x[0])
    closest_h_indices = [h_idx for _, h_idx in h_with_distances[:3]]
    
    # Add the 3 closest H atoms to nh3_atoms (ordered by distance to N)
    nh3_atoms.extend(closest_h_indices)
    
    return nh3_atoms


"""
Glazer pattern detection for perovskite structures.

This module provides functions to detect Glazer notation from structures by
analyzing octahedral tilting patterns. This is the inverse operation of
glazer_tilting.py in the builders module.

Functions
---------
_detect_glazer_pattern
    Detect Glazer notation from structure by analyzing octahedral tilting
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from scipy.spatial.transform import Rotation
from ...utils.geometry.geometry import _calculate_distances
from ..octahedral_processing.octahedral_detection import find_shared_atoms
from ..octahedral_processing.tilt_calculations import (
    calculate_euler_angles_from_bonds,
    get_reference_axes,
)
from ...builders.glazer_notation import get_space_group_from_notation, get_conventional_pattern
from ..utils.geometry_helpers import (
    apply_pbc_to_vector,
    get_all_x_atoms_from_octahedron,
)

# Default tolerance constants for Glazer pattern detection
DEFAULT_TOLERANCE = 0.1
DEFAULT_TILT_SIGNIFICANCE_THRESHOLD = 0.1
DEFAULT_ZERO_TILT_THRESHOLD = 3.0
DEFAULT_ZERO_TILT_RELATIVE_THRESHOLD = 0.3
DEFAULT_MAGNITUDE_EQUIVALENCE_THRESHOLD = 0.1
DEFAULT_BOND_SELECTION_THRESHOLD = 0.1
DEFAULT_NUMERICAL_TOLERANCE = 1e-6


def _abs_sqrt(m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the sign-preserving square root of a number or array.
    
    This preserves magnitude information while maintaining sign, which is
    important for Glazer pattern detection where both magnitude and sign
    of correlations matter.
    
    Parameters
    ----------
    m : float or np.ndarray
        Input value(s)
    
    Returns
    -------
    float or np.ndarray
        sqrt(|m|) * sign(m)
    """
    return np.sqrt(np.abs(m)) * np.sign(m)


def _apply_pbc_to_vectors(vectors: np.ndarray, cell: np.ndarray, inv_cell: np.ndarray) -> np.ndarray:
    """
    Apply periodic boundary conditions to vectors.
    
    This is a wrapper around the shared apply_pbc_to_vector utility that accepts
    pre-computed inv_cell for performance in Glazer detection.
    
    Parameters
    ----------
    vectors : np.ndarray
        Vectors to wrap (N, 3) or (N, M, 3)
    cell : np.ndarray
        Cell matrix (3, 3)
    inv_cell : np.ndarray
        Inverse cell matrix (3, 3) - pre-computed for performance
    
    Returns
    -------
    np.ndarray
        Vectors with PBC applied, same shape as input
    """
    return apply_pbc_to_vector(vectors, cell, inv_cell=inv_cell)


def _build_benv_matrix(
    b_positions: np.ndarray, 
    cell: np.ndarray, 
    inv_cell: np.ndarray,
    estimated_supercell: np.ndarray
) -> Optional[np.ndarray]:
    """
    Build B-site environment matrix (Benv) identifying nearest neighbors along each axis.
    
    This matches the monolith's approach:
    1. Find all first nearest neighbors using distance threshold
    2. Order neighbors by axis using geometric criteria
    3. For 3 neighbors (2x2x2): organize as [+x, +y, +z]
    4. For 6 neighbors (larger supercells): organize as [+x, -x, +y, -y, +z, -z]
    
    Parameters
    ----------
    b_positions : np.ndarray
        B-site positions (N, 3)
    cell : np.ndarray
        Cell matrix (3, 3)
    inv_cell : np.ndarray
        Inverse cell matrix (3, 3)
    estimated_supercell : np.ndarray
        Estimated supercell dimensions (3,)
    
    Returns
    -------
    np.ndarray or None
        Benv matrix:
        - For 3 neighbors: (N, 3) where Benv[i, j] is the +j neighbor index
        - For 6 neighbors: (N, 6) where Benv[i, :] = [+x, -x, +y, -y, +z, -z] neighbor indices
        Returns None if detection fails.
    """
    n_oct = len(b_positions)
    if n_oct < 2:
        return None
    
    # Calculate distance matrix (matching monolith approach)
    # Build full distance matrix
    r0 = np.zeros((n_oct, n_oct))
    for i in range(n_oct):
        distances = _calculate_distances(b_positions[i], b_positions, cell)
        r0[i, :] = distances
    
    # Find first nearest neighbor threshold
    # For perovskites, first NN is typically ~6-7 Å, second NN is ~8-9 Å
    # Use a simple approach: find the gap between first and second NN
    all_dists = r0[r0 > 0.1]  # Exclude self-distances
    if len(all_dists) == 0:
        return None
    
    # Sort distances and find gap
    sorted_dists = np.sort(all_dists)
    # Look for gap between first and second NN populations
    # First NN typically < 7.5 Å, second NN > 8.0 Å
    search_NN1 = 7.5  # Default threshold for first NN
    if len(sorted_dists) > 10:
        # Try to find gap automatically
        diffs = np.diff(sorted_dists)
        gap_idx = np.argmax(diffs)
        if gap_idx > 0 and diffs[gap_idx] > 0.5:  # Significant gap found
            search_NN1 = sorted_dists[gap_idx] + 0.2
    
    # Find all first nearest neighbors (matching monolith: res=np.where(np.logical_and(r0<search_NN1,r0>0.1)))
    res = np.where(np.logical_and(r0 < search_NN1, r0 > 0.1))
    
    # Build Benv as list of lists (matching monolith approach)
    Benv = [[] for _ in range(n_oct)]
    for i in range(res[0].shape[0]):
        Benv[res[0][i]].append(res[1][i])
    
    # Convert to numpy array - handle variable number of neighbors
    max_neighbors = max(len(neighbors) for neighbors in Benv) if Benv else 0
    
    if max_neighbors == 0:
        return None
    
    # For 2x2x2 supercell, expect 3 neighbors; for larger, expect 6
    # If we have 3 or 6 neighbors, we can order them by axis
    if max_neighbors == 3:
        # 2x2x2 supercell - order neighbors by x, y, z axes
        # First, convert Benv to numpy array (pad if needed)
        benv_array = np.zeros((n_oct, max_neighbors), dtype=int)
        for i in range(n_oct):
            neighbors = Benv[i]
            if len(neighbors) == max_neighbors:
                benv_array[i, :] = neighbors
            elif len(neighbors) > 0:
                # Pad with first neighbor if needed
                benv_array[i, :len(neighbors)] = neighbors
                benv_array[i, len(neighbors):] = neighbors[0]  # Repeat first neighbor
        
        # Now order neighbors by axis (matching monolith exactly)
        # monolith: orders = np.argmax(np.abs(Bpos[0,Benv[i,:],:] - Bpos[0,i,:]), axis=0)
        benv_ordered = np.zeros((n_oct, 3), dtype=int)
        for i in range(n_oct):
            if len(Benv[i]) == 3:
                # Get relative positions of neighbors
                neighbors = benv_array[i, :3]
                # Calculate relative positions (matching monolith approach)
                rel_pos = b_positions[neighbors] - b_positions[i]
                # Apply PBC using minimum image convention
                rel_cart = _apply_pbc_to_vectors(rel_pos, cell, inv_cell)
                
                # Find which neighbor has max abs difference along each axis
                # This matches monolith: orders = np.argmax(np.abs(...), axis=0)
                # But we also need to ensure we select the neighbor in the positive direction
                orders = np.argmax(np.abs(rel_cart), axis=0)
                
                # Verify that selected neighbors are in positive direction
                # If not, we might need to select the opposite neighbor
                for axis_idx in range(3):
                    selected_neighbor = neighbors[orders[axis_idx]]
                    rel_vec = rel_cart[orders[axis_idx]]
                    # Check if the selected neighbor is in positive direction
                    if rel_vec[axis_idx] < 0:
                        # Find neighbor in positive direction along this axis
                        pos_mask = rel_cart[:, axis_idx] > 0.1
                        if np.any(pos_mask):
                            pos_indices = np.where(pos_mask)[0]
                            # Select the one with largest projection along this axis
                            best_idx = pos_indices[np.argmax(rel_cart[pos_indices, axis_idx])]
                            orders[axis_idx] = best_idx
                
                benv_ordered[i, :] = neighbors[orders]
            else:
                # Fallback: use first 3 neighbors or pad
                neighbors = np.array(Benv[i][:3] if len(Benv[i]) >= 3 else Benv[i])
                if len(neighbors) < 3:
                    # Pad with first neighbor if needed
                    neighbors = np.pad(neighbors, (0, 3 - len(neighbors)), mode='edge')
                benv_ordered[i, :] = neighbors[:3]
        
        return benv_ordered
    
    elif max_neighbors == 6:
        # Larger supercell - need to order 6 neighbors into pairs for each axis
        # Use reference octahedron pattern matching (matching monolith approach)
        
        # First, convert Benv to numpy array
        benv_array = np.zeros((n_oct, 6), dtype=int)
        for i in range(n_oct):
            neighbors = Benv[i]
            if len(neighbors) == 6:
                benv_array[i, :] = neighbors
            elif len(neighbors) > 0:
                # Pad if needed
                benv_array[i, :len(neighbors)] = neighbors[:6]
                if len(neighbors) < 6:
                    # Repeat neighbors to fill
                    while len(neighbors) < 6:
                        neighbors = np.concatenate([neighbors, Benv[i][:min(6-len(neighbors), len(Benv[i]))]])
                    benv_array[i, :] = neighbors[:6]
        
        # Calculate relative position vectors (Bcoordenv)
        Bcoordenv = np.empty((n_oct, 6, 3))
        for i in range(n_oct):
            if len(Benv[i]) == 6:
                Bcoordenv[i, :] = b_positions[benv_array[i, :]] - b_positions[i]
            else:
                # Fallback: use available neighbors
                available = Benv[i][:min(6, len(Benv[i]))]
                for j in range(6):
                    if j < len(available):
                        Bcoordenv[i, j] = b_positions[available[j]] - b_positions[i]
                    else:
                        # Pad with zero vector if not enough neighbors
                        Bcoordenv[i, j] = np.zeros(3)
        
        # Apply PBC to vectors
        Bcoordenv = _apply_pbc_to_vectors(Bcoordenv, cell, inv_cell)
        
        # Define reference octahedron pattern: [+x, -x, +y, -y, +z, -z]
        ref_octa = np.array([
            [1, 0, 0],   # +x direction
            [-1, 0, 0],  # -x direction
            [0, 1, 0],   # +y direction
            [0, -1, 0],  # -y direction
            [0, 0, 1],   # +z direction
            [0, 0, -1]   # -z direction
        ], dtype=float)
        
        # Match neighbors to reference directions
        benv_ordered = np.zeros((n_oct, 6), dtype=int)
        for i in range(n_oct):
            if len(Benv[i]) == 6:
                orders = np.zeros(6, dtype=int)
                for j in range(6):
                    # Find which neighbor best matches reference direction j
                    # Using dot product to find best alignment
                    dots = np.dot(Bcoordenv[i, :, :], ref_octa[j, :])
                    orders[j] = np.argmax(dots)
                
                # Reorder Benv according to orders
                benv_ordered[i, :] = benv_array[i, :][orders]
            else:
                # Fallback: use first 6 neighbors or pad
                neighbors = np.array(Benv[i][:6] if len(Benv[i]) >= 6 else Benv[i])
                if len(neighbors) < 6:
                    # Pad with first neighbor if needed
                    neighbors = np.pad(neighbors, (0, 6 - len(neighbors)), mode='edge')
                benv_ordered[i, :] = neighbors[:6]
        
        return benv_ordered
    
    else:
        # Unexpected number of neighbors
        return None


def _calculate_phase_relationship(
    shift_i: np.ndarray, 
    shift_j: np.ndarray, 
    axis: int, 
    estimated_supercell: np.ndarray
) -> float:
    """
    Calculate expected phase difference between two octahedra based on shifts.
    
    The builder applies phase factors: phase = (-1)^dot(shift, kvec)
    For "-" patterns, kvec = [1,1,1], so if shift_diff[axis] == 1, phases differ by sign.
    
    Parameters
    ----------
    shift_i : np.ndarray
        Shift indices for octahedron i (3,)
    shift_j : np.ndarray
        Shift indices for octahedron j (3,)
    axis : int
        Axis along which to check phase relationship (0=x, 1=y, 2=z)
    estimated_supercell : np.ndarray
        Estimated supercell dimensions (3,)
    
    Returns
    -------
    float
        Expected phase difference: (-1) ** shift_diff[axis]
        For shift_diff[axis] == 0: returns 1 (same phase)
        For shift_diff[axis] == 1: returns -1 (opposite phase)
    """
    shift_diff = shift_j - shift_i
    # Handle PBC wrapping
    shift_diff = (shift_diff + estimated_supercell) % estimated_supercell
    # For "-" pattern, if shift_diff[axis] == 1, phases differ by sign
    return (-1) ** shift_diff[axis]


# Functions _calculate_euler_angles_from_bonds and _get_reference_axes
# have been moved to octahedral_processing.tilt_calculations
# and are now imported as calculate_euler_angles_from_bonds and get_reference_axes


def _detect_glazer_pattern(
    analyzer, 
    tolerance: float = DEFAULT_TOLERANCE,
    tilt_significance_threshold: Optional[float] = None,
    zero_tilt_threshold: Optional[float] = None,
    magnitude_equivalence_threshold: Optional[float] = None,
    bond_selection_threshold: float = DEFAULT_BOND_SELECTION_THRESHOLD,
) -> Dict[str, Union[str, List[str], List[float], Optional[str], Dict]]:
    """
    Detect Glazer pattern from structure by analyzing octahedral tilting.

    **IMPORTANT**: This function is only valid for 3D bulk perovskites.
    For quasi-2D perovskite structures (Ruddlesden-Popper, Dion-Jacobson,
    or monolayer), use alternative distortion metrics instead.

    Analyzes the rotation of octahedra relative to the unit cell axes
    to determine the Glazer notation.

    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzer instance with analyzed structure
    tolerance : float, default=0.1
        Base tolerance for angle comparisons (degrees). Used as default for
        all thresholds if specific thresholds are not provided.
    tilt_significance_threshold : float, optional
        Minimum tilt angle (degrees) to consider a tilt significant for
        correlation calculations. Default: tolerance
    zero_tilt_threshold : float, optional
        Maximum tilt angle (degrees) to classify as zero tilt (pattern "0").
        Default: tolerance
    magnitude_equivalence_threshold : float, optional
        Maximum difference (degrees) between tilt angles to consider them
        equivalent (same magnitude letter: a, b, or c).
        Default: tolerance
    bond_selection_threshold : float, default=0.1
        Minimum projection value for bond selection along reference axes.
        Ensures consistent sign convention. Units: dimensionless (normalized).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'notation': Full Glazer notation string (e.g., "a-b+a-")
        - 'tilt_pattern': List of tilt phases ['+', '-', '+']
        - 'tilt_angles': List of detected angles [omega_x, omega_y, omega_z] in degrees
        - 'magnitudes': List of magnitude symbols ['a', 'b', 'a']
        - 'space_group': Inferred space group (if available)
        - 'tolerance_info': dict with tolerance values used (for documentation)

    Raises
    ------
    ValueError
        If the structure is identified as a quasi-2D perovskite (Ruddlesden-Popper,
        Dion-Jacobson, or monolayer). Glazer notation is only defined for 3D bulk
        perovskites with infinite corner-sharing octahedral networks.

    Notes
    -----
    **Tolerance Sensitivity:**
    
    There is no specified numeric tolerance in the original Glazer notation
    (Howard & Stokes, 1998) for deciding sign of tilts, equivalence of tilt
    magnitudes, or mapping continuous rotations to discrete a/b/c+/–/0 symbols.
    Different reasonable tolerances will label marginal structures differently.
    
    The tolerances control three key classification decisions:
    
    1. **Sign of tilts (+/-)**: Determined by correlation sign. Very small tilts
       near `tilt_significance_threshold` may be excluded from correlation
       calculations, leading to uncertain pattern detection.
    
    2. **Equivalence of tilt magnitudes**: Angles within
       `magnitude_equivalence_threshold` are considered equal (same letter:
       a, b, or c). Marginal cases near this threshold may be labeled as
       "aac" vs "abc" depending on the exact value.
    
    3. **Zero tilt classification**: Tilts below `zero_tilt_threshold` are
       labeled "0". Structures with very small tilts may be classified as
       "a0a0a0" vs "a+a+a+" depending on this threshold.
    
    **Recommended values:**
    - For high-precision structures: tolerance = 0.05-0.1 degrees
    - For experimental/optimized structures: tolerance = 0.1-0.5 degrees
    - For noisy or approximate structures: tolerance = 0.5-1.0 degrees
    
    **Example tolerance sensitivity:**
    - A structure with tilts [10.0°, 10.1°, 15.0°] and tolerance=0.05:
      → "abc" (x and y are different: 0.1° > 0.05)
    - Same structure with tolerance=0.2:
      → "aac" (x and y are equal: 0.1° < 0.2)
    """
    # Set defaults based on base tolerance
    # If specific thresholds not provided, use tolerance value (which defaults to module constant)
    tilt_sig = tilt_significance_threshold if tilt_significance_threshold is not None else tolerance
    zero_thresh = zero_tilt_threshold if zero_tilt_threshold is not None else tolerance
    mag_eq_thresh = magnitude_equivalence_threshold if magnitude_equivalence_threshold is not None else tolerance
    
    # Store tolerance info for documentation
    tolerance_info = {
        'base_tolerance': tolerance,
        'tilt_significance_threshold': tilt_sig,
        'zero_tilt_threshold': zero_thresh,
        'magnitude_equivalence_threshold': mag_eq_thresh,
        'bond_selection_threshold': bond_selection_threshold,
    }

    # VALIDATE: Glazer notation is only valid for 3D bulk structures
    structure_type = analyzer.structure_type
    if structure_type in ('dj', 'rp', 'monolayer'):
        # Get number of layers for error message
        layers = analyzer.get_layers()
        num_layers = len(layers) if layers else 'unknown'

        # Build structure type description
        type_descriptions = {
            'dj': 'Dion-Jacobson (DJ) quasi-2D',
            'rp': 'Ruddlesden-Popper (RP) quasi-2D',
            'monolayer': 'monolayer quasi-2D',
        }
        type_desc = type_descriptions.get(structure_type, 'quasi-2D')

        raise ValueError(
            f"Glazer notation is not applicable to {type_desc} perovskites "
            f"(detected {num_layers} inorganic layers). Glazer notation was designed "
            f"for 3D bulk perovskites with infinite corner-sharing octahedral networks.\n\n"
            f"For q2D structures, consider using alternative distortion metrics:\n"
            f"  - analyzer.compute_delta(group_by='layer')  # Distortion parameter Δ\n"
            f"  - analyzer.compute_sigma(group_by='layer')  # Variance σ²\n"
            f"  - analyzer.compute_lambda(group_by='layer') # Elongation λ\n"
            f"  - analyzer.get_bxb_angles(group_by='layer') # B-X-B bond angles\n"
            f"\n"
            f"A dedicated q2D octahedral tilting characterization metric is under development."
        )

    # Get octahedra and structure data
    octahedra = analyzer.get_octahedra()
    if not octahedra:
        return {
            "notation": "a0a0a0",
            "tilt_pattern": ["0", "0", "0"],
            "tilt_angles": [0.0, 0.0, 0.0],
            "magnitudes": ["a", "a", "a"],
            "space_group": "Pm-3m",
        }

    atom_positions = analyzer.cell.get_positions()
    cell = np.array(analyzer.cell.get_cell())
    inv_cell = np.linalg.inv(cell)
    
    # Get reference axes (pseudo-cubic basis)
    ref_axes = get_reference_axes(atom_positions, cell, octahedra)

    # 1. Calculate local tilts for each octahedron
    local_tilts = []  # List of [alpha, beta, gamma] for each oct

    for oct_data in octahedra:
        b_idx = oct_data.get("central_atom_index")
        if b_idx is None:
            local_tilts.append([0.0, 0.0, 0.0])
            continue

        b_pos = atom_positions[b_idx]

        # Get all X neighbors using shared utility
        x_indices = get_all_x_atoms_from_octahedron(oct_data)

        # Vectors to neighbors with PBC using shared utility
        vectors = []
        for x_idx in x_indices:
            v = atom_positions[x_idx] - b_pos
            # Apply PBC using shared utility
            v = apply_pbc_to_vector(v, cell, inv_cell=inv_cell)
            vectors.append(v)

        # Project vectors onto reference axes to identify them
        # CRITICAL: Use consistent sign convention - always select bonds pointing
        # in the POSITIVE direction along each axis to ensure global sign consistency
        v_x, v_y, v_z = None, None, None
        v_x_proj, v_y_proj, v_z_proj = -np.inf, -np.inf, -np.inf  # Start with -inf to require positive

        for v in vectors:
            # Project onto ref axes
            projs = [np.dot(v, ref_axes[i]) for i in range(3)]
            abs_projs = np.abs(projs)
            max_idx = np.argmax(abs_projs)

            # CRITICAL: Only select bonds with POSITIVE projection to ensure consistent sign
            # This matches the monolith's approach of using a global reference frame
            if max_idx == 0 and projs[0] > bond_selection_threshold and projs[0] > v_x_proj:
                v_x = v
                v_x_proj = projs[0]
            elif max_idx == 1 and projs[1] > bond_selection_threshold and projs[1] > v_y_proj:
                v_y = v
                v_y_proj = projs[1]
            elif max_idx == 2 and projs[2] > bond_selection_threshold and projs[2] > v_z_proj:
                v_z = v
                v_z_proj = projs[2]

        # Use proper Euler angle decomposition to extract individual axis tilts
        # This correctly handles the coupled rotations applied by the builder:
        # R_total = R_z(gamma) @ R_y(beta) @ R_x(alpha)
        #
        # The Kabsch algorithm finds the best rotation matrix that transforms
        # ideal bond vectors to actual bond vectors, then scipy's Rotation
        # decomposes this into Euler angles using the matching convention.
        tilts, _ = calculate_euler_angles_from_bonds(v_x, v_y, v_z, ref_axes)
        local_tilts.append(tilts)

    # Safety check: ensure local_tilts has the same length as octahedra
    if len(local_tilts) != len(octahedra):
        raise ValueError(
            f"Mismatch between number of octahedra ({len(octahedra)}) "
            f"and local_tilts ({len(local_tilts)}). "
            f"This indicates a bug in the tilt calculation loop."
        )

    # 2. Analyze patterns and phases
    
    # Build neighbor indices for find_shared_atoms using shared utility
    neighbor_indices = []
    for oct in octahedra:
        all_neighbors = get_all_x_atoms_from_octahedron(oct)
        neighbor_indices.append(all_neighbors)

    shared_atoms = find_shared_atoms(neighbor_indices)
    
    # Calculate shift indices for each octahedron (matching builder's calculation)
    # This is needed to understand phase relationships
    # For a 2x2x2 supercell, shifts should be in [0,1] for each axis
    # We estimate supercell size from the structure
    # Try to infer supercell from B-site positions
    b_positions = np.array([atom_positions[oct["central_atom_index"]] for oct in octahedra])
    
    # Estimate supercell by looking at B-site distribution
    # For a 2x2x2 supercell, we expect 8 B-sites in the unit cell
    # Find the smallest repeating unit
    b_frac = b_positions @ inv_cell.T
    
    # Estimate supercell dimensions from B-site distribution
    # Try to detect actual supercell size from structure
    b_frac = b_positions @ inv_cell.T
    
    # Find the smallest repeating unit by looking at fractional coordinate distribution
    # For a 2x2x2 supercell, we expect 8 B-sites, fractional coords should cluster at 0, 0.5
    # For larger supercells, adjust accordingly
    n_oct = len(b_positions)
    
    # Try to estimate supercell size
    # For cubic-like cells, supercell_size^3 should equal number of octahedra
    # But we need to account for the fact that we might have a 2x2x2 supercell
    estimated_supercell = np.array([2, 2, 2], dtype=int)  # Default assumption
    
    # Try to detect from fractional coordinates
    # For a 2x2x2 supercell, frac coords should be at multiples of 0.5
    # For a 3x3x3, multiples of 1/3, etc.
    for axis in range(3):
        frac_coords = b_frac[:, axis]
        # Remove PBC wrapping
        frac_coords = frac_coords % 1.0
        # Look for clustering - for 2x2x2, should see clusters at 0, 0.5
        # For 3x3x3, clusters at 0, 0.33, 0.67, etc.
        # Simple heuristic: count distinct "bins"
        bins = np.round(frac_coords * 2) / 2  # Round to nearest 0.5
        unique_bins = len(np.unique(bins))
        if unique_bins >= 2:
            estimated_supercell[axis] = unique_bins
        else:
            # Fallback: try to estimate from number of octahedra
            # For a cubic supercell, n_oct = supercell_size^3
            # So supercell_size = n_oct^(1/3)
            est_size = int(np.round(n_oct ** (1/3)))
            if est_size >= 2:
                estimated_supercell[axis] = est_size
            else:
                estimated_supercell[axis] = 2  # Default
    
    # Calculate shift indices for each octahedron
    oct_shifts = []
    for oct in octahedra:
        b_idx = oct["central_atom_index"]
        b_pos = atom_positions[b_idx]
        
        # Convert to fractional coordinates
        b_frac = b_pos @ inv_cell.T
        
        # Scale by estimated supercell and get integer indices
        # This matches the builder's calculation: unit_cell_coords = frac_coords * supercell
        unit_cell_coords = b_frac * estimated_supercell.astype(float)
        indices = np.floor(unit_cell_coords + DEFAULT_NUMERICAL_TOLERANCE).astype(int)
        # Apply modulo for PBC
        indices = indices % estimated_supercell
        oct_shifts.append(indices)
    
    oct_shifts = np.array(oct_shifts)
    
    # Try to build Benv matrix for better neighbor identification
    # This matches the monolith's approach and ensures we compare the right pairs
    benv = _build_benv_matrix(b_positions, cell, inv_cell, estimated_supercell)
    use_benv = benv is not None and benv.shape[0] == len(octahedra)
    has_6_neighbors = use_benv and benv.shape[1] == 6
    
    # Store correlations: axis -> list of products (t_i * t_j)
    # Glazer notation describes the correlation of tilts ABOUT an axis
    # when octahedra are connected ALONG that axis (longitudinal correlations)
    # + means same sense, - means opposite sense
    correlations = {0: [], 1: [], 2: []} 

    # Use Benv if available, otherwise fall back to shared_atoms approach
    if use_benv:
        if has_6_neighbors:
            # For 6 neighbors: Benv[i, :] = [+x, -x, +y, -y, +z, -z] neighbor indices
            # Use both positive and negative neighbors for correlation (matching monolith)
            # But prioritize positive direction for vector convention
            # Index mapping: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z
            axis_to_benv_idx = {0: (0, 1), 1: (2, 3), 2: (4, 5)}  # x->(0,1), y->(2,3), z->(4,5)
            
            for oct_i in range(len(octahedra)):
                for axis in range(3):
                    pos_idx, neg_idx = axis_to_benv_idx[axis]
                    
                    # Use positive direction neighbor for vector convention
                    oct_j = benv[oct_i, pos_idx]
                    
                    if oct_j == oct_i or oct_j >= len(octahedra) or oct_j < 0:
                        continue
                    
                    # Safety check: ensure indices are valid
                    if (oct_i >= len(local_tilts) or oct_j >= len(local_tilts) or
                        oct_i >= len(oct_shifts) or oct_j >= len(oct_shifts)):
                        continue
                    
                    # Get shift indices to check if octahedra are successive
                    shift_i = oct_shifts[oct_i]
                    shift_j = oct_shifts[oct_j]
                    shift_diff = shift_j - shift_i
                    shift_diff = (shift_diff + estimated_supercell) % estimated_supercell
                    
                    # Check if octahedra are successive along this axis (shift_diff[axis] == 1)
                    # For successive octahedra, the phase relationship depends on the pattern
                    # For "-" patterns: phase alternates (shift_diff[axis] == 1 → phase_diff = -1)
                    # For "+" patterns: phase is same (shift_diff[axis] == 1 → phase_diff = 1, but kvec[axis] = 0)
                    # We can't know the pattern yet, but we can use the fact that for "-" patterns,
                    # successive octahedra should have opposite-signed tilts
                    is_successive = (shift_diff[axis] == 1 or 
                                    shift_diff[axis] == estimated_supercell[axis] - 1)
                    
                    t_i = local_tilts[oct_i][axis]
                    t_j = local_tilts[oct_j][axis]
                    
                    # Only measure correlation if both tilts are significant
                    if abs(t_i) > tilt_sig and abs(t_j) > tilt_sig:
                        # Calculate geometric correlation
                        correlation = _abs_sqrt(t_i * t_j)
                        
                        # For successive octahedra, if we detect same-signed tilts but they're successive,
                        # it might indicate a reference frame mismatch. However, we can't definitively
                        # know if it's a "+" or "-" pattern yet. Instead, we'll use the raw correlation
                        # and let the aggregation logic determine the pattern based on majority.
                        # The key is that we're using the correct neighbor pairs (vector convention).
                        correlations[axis].append(correlation)
                    
                    # Also use negative neighbor for additional correlation data (matching monolith)
                    oct_j_neg = benv[oct_i, neg_idx]
                    if (oct_j_neg != oct_i and oct_j_neg < len(octahedra) and oct_j_neg >= 0 and
                        oct_i < len(local_tilts) and oct_j_neg < len(local_tilts)):
                        t_j_neg = local_tilts[oct_j_neg][axis]
                        if abs(t_i) > tilt_sig and abs(t_j_neg) > tilt_sig:
                            correlation_neg = _abs_sqrt(t_i * t_j_neg)
                            correlations[axis].append(correlation_neg)
        else:
            # For 3 neighbors (2x2x2): Benv[i, :] = [+x, +y, +z] neighbor indices
            for oct_i in range(len(octahedra)):
                for axis in range(3):
                    oct_j = benv[oct_i, axis]
                    if oct_j == oct_i or oct_j >= len(octahedra) or oct_j < 0:
                        continue
                    
                    # Safety check: ensure indices are valid
                    if (oct_i >= len(local_tilts) or oct_j >= len(local_tilts) or
                        oct_i >= len(oct_shifts) or oct_j >= len(oct_shifts)):
                        continue
                    
                    # Get shift indices to check if octahedra are successive
                    shift_i = oct_shifts[oct_i]
                    shift_j = oct_shifts[oct_j]
                    shift_diff = shift_j - shift_i
                    shift_diff = (shift_diff + estimated_supercell) % estimated_supercell
                    
                    # Check if octahedra are successive along this axis
                    is_successive = (shift_diff[axis] == 1 or 
                                    shift_diff[axis] == estimated_supercell[axis] - 1)
                    
                    t_i = local_tilts[oct_i][axis]
                    t_j = local_tilts[oct_j][axis]
                    
                    # Only measure correlation if both tilts are significant
                    if abs(t_i) > tilt_sig and abs(t_j) > tilt_sig:
                        # Calculate geometric correlation
                        correlation = _abs_sqrt(t_i * t_j)
                        
                        # For successive octahedra, check if the correlation sign matches
                        # the expected pattern. If octahedra are successive (shift_diff[axis] == 1),
                        # and we detect same-signed tilts (positive correlation), but the structure
                        # was built with a "-" pattern, the builder would have applied opposite phases.
                        # This suggests our reference frame might be inverted. However, we can't
                        # know the pattern yet, so we'll use the raw correlation and let aggregation decide.
                        # The key is that we're using the correct neighbor pairs (vector convention).
                        correlations[axis].append(correlation)
    else:
        # Fall back to shared_atoms approach
        for (oct_i, oct_j), _ in shared_atoms.items():
            if oct_i >= len(octahedra) or oct_j >= len(octahedra):
                continue
            
            if oct_i >= len(oct_shifts) or oct_j >= len(oct_shifts):
                continue
                
            # Get shift indices for these octahedra
            shift_i = oct_shifts[oct_i]
            shift_j = oct_shifts[oct_j]
            
            # Determine direction of connection
            b_i_pos = atom_positions[octahedra[oct_i]["central_atom_index"]]
            b_j_pos = atom_positions[octahedra[oct_j]["central_atom_index"]]
            
            delta_frac = (b_j_pos - b_i_pos) @ inv_cell.T
            delta_frac -= np.round(delta_frac)
            
            bond_axis = np.argmax(np.abs(delta_frac))
            
            # Only consider octahedra that are "successive" along the bond axis
            # Successive means their shift indices differ by 1 (mod supercell) along that axis
            # This matches the reference implementation's concept of adjacent cells
            shift_diff = shift_j - shift_i
            # Handle PBC wrapping
            shift_diff = (shift_diff + estimated_supercell) % estimated_supercell
            
            # Check if they differ by exactly 1 along the bond axis
            # (allowing for PBC, so difference could be supercell_size - 1)
            axis_diff = shift_diff[bond_axis]
            is_successive = (axis_diff == 1 or axis_diff == estimated_supercell[bond_axis] - 1)
            
            # Also check if other axes are the same (they should be for successive octahedra)
            other_axes_same = all(shift_diff[ax] == 0 or shift_diff[ax] == estimated_supercell[ax] - 1 
                                  for ax in range(3) if ax != bond_axis)
            
            if not (is_successive and other_axes_same):
                # Not successive along this axis - skip
                continue
            
            # Check correlations for the tilt axis matching the bond axis
            # In Glazer notation, the pattern along axis X (e.g., a+) refers to
            # the correlation of tilts about X when octahedra are connected along X
            tilt_axis = bond_axis
            
            # Calculate expected phase relationship based on shift difference
            # This matches the builder's phase convention: phase = (-1)^dot(shift, kvec)
            expected_phase_diff = _calculate_phase_relationship(
                shift_i, shift_j, tilt_axis, estimated_supercell
            )
            
            # Get connection vector to ensure positive direction convention
            connection_vec = b_j_pos - b_i_pos
            connection_vec_frac = connection_vec @ inv_cell.T
            connection_vec_frac -= np.round(connection_vec_frac)
            connection_vec_cart = connection_vec_frac @ cell
            
            # Ensure we're using positive direction along the bond axis
            # If connection vector points in negative direction, we should still compare
            # but the phase relationship accounts for the direction
            if connection_vec_cart[bond_axis] < 0:
                # Connection is in negative direction, but we still compare tilts
                # The phase relationship already accounts for this
                pass
            
            t_i = local_tilts[oct_i][tilt_axis]
            t_j = local_tilts[oct_j][tilt_axis]
            
            # Only measure correlation if both tilts are significant
            # This avoids noise from near-zero tilts affecting correlation
            if abs(t_i) > tilt_sig and abs(t_j) > tilt_sig:
                # Use abs_sqrt to preserve magnitude information while maintaining sign
                # This matches the monolith implementation and improves pattern detection
                # abs_sqrt(t_i * t_j) = sqrt(|t_i * t_j|) * sign(t_i * t_j)
                # This preserves the correlation strength while maintaining the sign relationship
                correlation = _abs_sqrt(t_i * t_j)
                correlations[tilt_axis].append(correlation)

    # 3. Aggregate results
    final_tilts = []
    final_patterns = []

    # First, calculate all axis magnitudes for relative comparison
    all_axis_avg_mags = []
    for axis in range(3):
        mags = [abs(t[axis]) for t in local_tilts]
        all_axis_avg_mags.append(np.mean(mags) if mags else 0.0)
    max_axis_mag = max(all_axis_avg_mags) if all_axis_avg_mags else 0.0

    for axis in range(3):
        # Calculate magnitude and check for zero tilt
        # Use both absolute values and raw values to better detect zero tilts
        mags = [abs(t[axis]) for t in local_tilts]
        raw_tilts = [t[axis] for t in local_tilts]
        avg_mag = np.mean(mags) if mags else 0.0
        max_mag = np.max(mags) if mags else 0.0

        # Improved zero-tilt detection with both absolute and relative thresholds:
        # 1. Absolute: Average magnitude should be small (< zero_thresh)
        # 2. Maximum magnitude should also be small (no outliers)
        # 3. Standard deviation should be small (consistent near-zero)
        # 4. RELATIVE: Magnitude should be small compared to other axes
        #    This handles spurious angles from Euler decomposition coupling
        std_mag = np.std(mags) if len(mags) > 1 else 0.0

        # Absolute threshold check
        is_zero_absolute = (avg_mag < zero_thresh and
                           max_mag < zero_thresh * 2 and
                           std_mag < zero_thresh)

        # Relative threshold check: if this axis magnitude is much smaller
        # than the maximum axis magnitude, it's likely a spurious angle from
        # Euler decomposition coupling. Use DEFAULT_ZERO_TILT_RELATIVE_THRESHOLD.
        relative_threshold = DEFAULT_ZERO_TILT_RELATIVE_THRESHOLD
        is_zero_relative = (max_axis_mag > zero_thresh * 3 and  # There are real tilts
                           avg_mag < max_axis_mag * relative_threshold)

        is_zero_tilt = is_zero_absolute or is_zero_relative
        
        if is_zero_tilt:
            # Zero tilt
            final_tilts.append(0.0)
            final_patterns.append("0")
        else:
            # There is a tilt - determine phase pattern
            final_tilts.append(avg_mag)
            corrs = correlations[axis]
            
            if not corrs:
                # No correlations measured - this could mean:
                # 1. Insufficient octahedra connections (isolated layers)
                # 2. Detection issue
                # Check if there are actually tilts on this axis
                if avg_mag >= zero_thresh:
                    # There are tilts but no correlations - might be isolated
                    # Default to "+" but this is uncertain
                    final_patterns.append("+")
                else:
                    # Should have been caught by zero-tilt check, but fallback
                    final_patterns.append("0")
            else:
                # Verify sign convention:
                # In glazer_tilting.py, phase = (-1)^dot(shift, kvec)
                # For "+" pattern: kvec[i] = 0, so phase depends on other axes
                # For "-" pattern: kvec = [1,1,1], so phase alternates
                # When two octahedra have same-signed tilts → same sense → "+"
                # When two octahedra have opposite-signed tilts → opposite sense → "-"
                # 
                # Using abs_sqrt(t_i * t_j) preserves magnitude while maintaining sign:
                #   Positive value if both positive or both negative (same sense) → "+"
                #   Negative value if one positive, one negative (opposite sense) → "-"
                # This matches the monolith implementation and improves detection.
                
                # Use magnitude-preserving correlations for better pattern detection
                # The sign of each correlation indicates the pattern (+ or -)
                # The magnitude indicates the strength of the correlation
                num_corrs = len(corrs)
                if num_corrs > 0:
                    # Count positive vs negative correlations
                    # Positive correlations indicate "+" pattern (same sense)
                    # Negative correlations indicate "-" pattern (opposite sense)
                    positive_count = sum(1 for c in corrs if c > 0)
                    negative_count = sum(1 for c in corrs if c < 0)
                    
                    # Use weighted voting based on correlation magnitudes
                    # This gives more weight to stronger correlations
                    positive_weight = sum(c for c in corrs if c > 0)
                    negative_weight = abs(sum(c for c in corrs if c < 0))
                    
                    # Require clear majority (at least 60% agreement)
                    # This helps with noisy measurements
                    positive_ratio = positive_count / num_corrs
                    weight_ratio = positive_weight / (positive_weight + negative_weight + 1e-9)
                    
                    # Use both count and weight ratios for robust detection
                    if positive_ratio >= 0.6 and weight_ratio >= 0.5:
                        final_patterns.append("+")
                    elif positive_ratio <= 0.4 and weight_ratio <= 0.5:
                        final_patterns.append("-")
                    else:
                        # Ambiguous - use weighted majority
                        if positive_weight >= negative_weight:
                            final_patterns.append("+")
                        else:
                            final_patterns.append("-")
                else:
                    # Shouldn't happen, but fallback
                    final_patterns.append("+")

    # Assign magnitude letters based on angle similarities
    # Following Glazer notation conventions:
    # - aaa: all equal
    # - aac: x=y, z different  (NOT aab!)
    # - abb: x different, y=z
    # - abc: all different
    angle_x, angle_y, angle_z = final_tilts

    x_eq_y = abs(angle_x - angle_y) < mag_eq_thresh
    x_eq_z = abs(angle_x - angle_z) < mag_eq_thresh
    y_eq_z = abs(angle_y - angle_z) < mag_eq_thresh

    if x_eq_y and x_eq_z:
        # All equal
        final_magnitudes = ["a", "a", "a"]
    elif x_eq_y:
        # x=y, z different → aac
        final_magnitudes = ["a", "a", "c"]
    elif x_eq_z:
        # x=z, y different → aba
        final_magnitudes = ["a", "b", "a"]
    elif y_eq_z:
        # y=z, x different → abb
        final_magnitudes = ["a", "b", "b"]
    else:
        # All different → abc
        final_magnitudes = ["a", "b", "c"]

    # Build notation string
    notation_parts = []
    for mag, phase in zip(final_magnitudes, final_patterns):
        notation_parts.append(mag)
        notation_parts.append(phase)
    notation = "".join(notation_parts)

    # Try to parse and get space group
    try:
        space_group = get_space_group_from_notation(notation)
    except (ValueError, KeyError):
        space_group = None

    # Validation: Check consistency
    # 1. If all phases are "0", all magnitudes should be effectively zero
    all_zero = all(p == "0" for p in final_patterns)
    if all_zero:
        # Verify magnitudes are actually zero
        if any(abs(t) > zero_thresh for t in final_tilts):
            # Inconsistency: phases say zero but magnitudes don't
            # This shouldn't happen with improved zero-tilt detection, but log it
            pass
    
    # 2. If phase is "0", magnitude should be zero (enforced by zero-tilt check above)
    # 3. Validate notation format
    if len(notation) != 6:
        # Shouldn't happen, but validate
        notation = "a0a0a0"  # Fallback
    
    # 4. Check for domain equivalence hints
    # If detected space group matches expected but notation differs,
    # this might indicate domain equivalence (handled by caller)
    
    # 5. Convert to conventional pattern using Howard & Stokes (1998) classification
    conventional_notation = get_conventional_pattern(notation)
    
    return {
        "notation": conventional_notation,  # Return conventional pattern
        "tilt_pattern": final_patterns,
        "tilt_angles": final_tilts,
        "magnitudes": final_magnitudes,
        "space_group": space_group,
        "tolerance_info": tolerance_info,
    }


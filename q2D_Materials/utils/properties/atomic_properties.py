"""Atomic properties utilities for bonding geometry and covalent radii.

This module provides centralized access to atomic properties and bonding
geometry calculations used throughout q2D-Materials.
"""

import json
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

# Load covalent radii from JSON
_COVALENT_RADII_CACHE: Optional[Dict[str, float]] = None
_COVALENT_RADII_BOND_ORDER_CACHE: Optional[Dict[str, Dict[str, Optional[float]]]] = None
_ATOMIC_VALENCE_CACHE: Optional[Dict[str, int]] = None
_ATOMIC_VALENCE_DATA_CACHE: Optional[Dict[str, Dict[str, Union[int, float]]]] = None
_OXIDATION_STATE_CACHE: Optional[Dict[str, float]] = None


def get_covalent_radii() -> Dict[str, float]:
    """Get covalent radii for all elements.

    Returns
    -------
    dict
        Dictionary mapping element symbols to covalent radii in Angstroms
    """
    global _COVALENT_RADII_CACHE

    if _COVALENT_RADII_CACHE is None:
        # Load from central data/tables directory
        # Path: q2D_Materials/utils/properties/atomic_properties.py -> q2D_Materials/data/tables/
        data_dir = Path(__file__).parent.parent.parent / "data" / "tables"
        radii_file = data_dir / "covalent_radii.json"

        with open(radii_file, 'r') as f:
            data = json.load(f)

        # Filter out metadata keys starting with underscore
        _COVALENT_RADII_CACHE = {k: v for k, v in data.items() if not k.startswith('_')}

    return _COVALENT_RADII_CACHE


def get_covalent_radius(element: str, default: float = 1.0) -> float:
    """Get covalent radius for a specific element.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'C', 'N', 'H')
    default : float, default=1.0
        Default radius if element not found

    Returns
    -------
    float
        Covalent radius in Angstroms (single bond radius)
    """
    radii = get_covalent_radii()
    return radii.get(element, default)


def get_covalent_radii_by_bond_order() -> Dict[str, Dict[str, Optional[float]]]:
    """Get covalent radii for elements by bond order (single, double, triple).
    
    Returns bond-order-specific radii from first-principle calculations.
    Falls back to single bond radii from main JSON if not available.
    
    Returns
    -------
    dict
        Dictionary mapping element symbols to dicts with 'single', 'double', 'triple' keys.
        Values are radii in Angstroms or None if not available for that bond order.
    """
    global _COVALENT_RADII_BOND_ORDER_CACHE
    
    if _COVALENT_RADII_BOND_ORDER_CACHE is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "tables"
        bond_order_file = data_dir / "covalent_radii_bond_order.json"
        
        if bond_order_file.exists():
            with open(bond_order_file, 'r') as f:
                data = json.load(f)
            _COVALENT_RADII_BOND_ORDER_CACHE = {
                k: v for k, v in data.items() if not k.startswith('_')
            }
        else:
            # Fallback: create from single bond radii
            single_radii = get_covalent_radii()
            _COVALENT_RADII_BOND_ORDER_CACHE = {
                elem: {'single': radius, 'double': None, 'triple': None}
                for elem, radius in single_radii.items()
            }
    
    return _COVALENT_RADII_BOND_ORDER_CACHE


def get_covalent_radius_by_bond_order(
    element: str,
    bond_order: int = 1,
    default: float = 1.0
) -> float:
    """Get covalent radius for a specific element and bond order.
    
    Parameters
    ----------
    element : str
        Element symbol (e.g., 'C', 'N', 'H')
    bond_order : int, default=1
        Bond order (1=single, 2=double, 3=triple)
    default : float, default=1.0
        Default radius if element/bond order not found
        
    Returns
    -------
    float
        Covalent radius in Angstroms for the specified bond order.
        Falls back to single bond radius if bond order not available.
    """
    bond_order_radii = get_covalent_radii_by_bond_order()
    
    if element not in bond_order_radii:
        return default
    
    element_data = bond_order_radii[element]
    
    # Map bond order to key
    order_key = {1: 'single', 2: 'double', 3: 'triple'}.get(bond_order, 'single')
    
    # Get radius for bond order, fallback to single if not available
    radius = element_data.get(order_key)
    if radius is not None:
        return radius
    
    # Fallback to single bond radius
    if element_data.get('single') is not None:
        return element_data['single']
    
    # Final fallback to default
    return default


def estimate_bond_order(
    distance: float,
    element1: str,
    element2: str,
    tolerance: float = 0.15
) -> int:
    """Estimate bond order from interatomic distance.
    
    Uses bond-order-specific covalent radii to estimate whether a bond
    is single, double, or triple based on distance.
    
    Parameters
    ----------
    distance : float
        Distance between atoms in Angstroms
    element1 : str
        First element symbol
    element2 : str
        Second element symbol
    tolerance : float, default=0.15
        Tolerance for bond order determination
        
    Returns
    -------
    int
        Estimated bond order (1, 2, or 3)
    """
    # Try triple bond first (shortest)
    r1_triple = get_covalent_radius_by_bond_order(element1, bond_order=3, default=None)
    r2_triple = get_covalent_radius_by_bond_order(element2, bond_order=3, default=None)
    
    if r1_triple is not None and r2_triple is not None:
        triple_length = r1_triple + r2_triple + tolerance
        if distance <= triple_length:
            return 3
    
    # Try double bond
    r1_double = get_covalent_radius_by_bond_order(element1, bond_order=2, default=None)
    r2_double = get_covalent_radius_by_bond_order(element2, bond_order=2, default=None)
    
    if r1_double is not None and r2_double is not None:
        double_length = r1_double + r2_double + tolerance
        if distance <= double_length:
            return 2
    
    # Default to single bond
    return 1


def _load_atomic_valence_data() -> Dict[str, Dict[str, Union[int, float]]]:
    """Load atomic valence data (including oxidation states) from JSON.

    Returns
    -------
    dict
        Dictionary mapping element symbols to dict with 'valence' and 'oxidation_state'
    """
    global _ATOMIC_VALENCE_DATA_CACHE

    if _ATOMIC_VALENCE_DATA_CACHE is None:
        # Load from central data/tables directory
        # Path: q2D_Materials/utils/properties/atomic_properties.py -> q2D_Materials/data/tables/
        data_dir = Path(__file__).parent.parent.parent / "data" / "tables"
        valence_file = data_dir / "atomic_valence.json"

        with open(valence_file, 'r') as f:
            data = json.load(f)

        # Filter out metadata keys starting with underscore
        _ATOMIC_VALENCE_DATA_CACHE = {k: v for k, v in data.items() if not k.startswith('_')}

    return _ATOMIC_VALENCE_DATA_CACHE


def get_atomic_valences() -> Dict[str, int]:
    """Get typical valence values for common elements.

    Returns
    -------
    dict
        Dictionary mapping element symbols to typical valence (max bonds)
    """
    global _ATOMIC_VALENCE_CACHE

    if _ATOMIC_VALENCE_CACHE is None:
        data = _load_atomic_valence_data()
        # Extract just valence values, handling both old format (int) and new format (dict)
        _ATOMIC_VALENCE_CACHE = {}
        for element, value in data.items():
            if isinstance(value, dict):
                _ATOMIC_VALENCE_CACHE[element] = value.get('valence', 4)
            else:
                # Backward compatibility with old format
                _ATOMIC_VALENCE_CACHE[element] = value

    return _ATOMIC_VALENCE_CACHE


def get_oxidation_states() -> Dict[str, float]:
    """Get typical oxidation states for common elements in perovskite structures.

    Returns
    -------
    dict
        Dictionary mapping element symbols to typical oxidation state
    """
    global _OXIDATION_STATE_CACHE

    if _OXIDATION_STATE_CACHE is None:
        data = _load_atomic_valence_data()
        # Extract oxidation states, handling both old format (int) and new format (dict)
        _OXIDATION_STATE_CACHE = {}
        for element, value in data.items():
            if isinstance(value, dict):
                _OXIDATION_STATE_CACHE[element] = value.get('oxidation_state', 0.0)
            else:
                # Backward compatibility: infer oxidation state from valence for old format
                # This is a fallback - new format should always be used
                _OXIDATION_STATE_CACHE[element] = 0.0

    return _OXIDATION_STATE_CACHE


def get_oxidation_state(element: str, default: float = 0.0) -> float:
    """Get typical oxidation state for an element in perovskite structures.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'Pb', 'I', 'Cs')
    default : float, default=0.0
        Default oxidation state if element not found

    Returns
    -------
    float
        Typical oxidation state for this element in perovskites
    """
    oxidation_states = get_oxidation_states()
    return oxidation_states.get(element, default)


def get_valence(element: str, default: int = 4) -> int:
    """Get typical valence for a specific element.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'C', 'N', 'O')
    default : int, default=4
        Default valence if element not found

    Returns
    -------
    int
        Typical maximum number of bonds for this element
    """
    valences = get_atomic_valences()
    return valences.get(element, default)


def calculate_ideal_bond_length(
    element1: str,
    element2: str,
    scale: float = 1.0,
    bond_order: Optional[int] = None
) -> float:
    """Calculate ideal bond length between two elements.

    Parameters
    ----------
    element1 : str
        First element symbol
    element2 : str
        Second element symbol
    scale : float, default=1.0
        Scaling factor for bond length
    bond_order : int, optional
        Bond order (1=single, 2=double, 3=triple).
        If provided, uses bond-order-specific radii.
        If None, uses single bond radii.

    Returns
    -------
    float
        Ideal bond length in Angstroms
    """
    if bond_order is not None:
        r1 = get_covalent_radius_by_bond_order(element1, bond_order=bond_order)
        r2 = get_covalent_radius_by_bond_order(element2, bond_order=bond_order)
    else:
        r1 = get_covalent_radius(element1)
        r2 = get_covalent_radius(element2)
    return (r1 + r2) * scale


def are_atoms_bonded(
    distance: float,
    element1: str,
    element2: str,
    tolerance: float = 0.45
) -> bool:
    """Check if two atoms are bonded based on distance and covalent radii.

    Parameters
    ----------
    distance : float
        Distance between atoms in Angstroms
    element1 : str
        First element symbol
    element2 : str
        Second element symbol
    tolerance : float, default=0.45
        Additional tolerance beyond sum of covalent radii

    Returns
    -------
    bool
        True if atoms are likely bonded
    """
    r1 = get_covalent_radius(element1)
    r2 = get_covalent_radius(element2)
    bond_cutoff = r1 + r2 + tolerance
    return distance <= bond_cutoff


def determine_hybridization(num_neighbors: int) -> Tuple[str, np.ndarray]:
    """Determine hybridization and ideal bond vectors from number of neighbors.

    Parameters
    ----------
    num_neighbors : int
        Number of bonded neighbors (1, 2, 3, or 4)

    Returns
    -------
    tuple of (str, np.ndarray)
        - Hybridization type ('sp', 'sp2', 'sp3', 'sp3d' etc.)
        - Ideal bond direction vectors as array of shape (num_neighbors, 3)

    Notes
    -----
    Bond vectors are normalized unit vectors pointing from central atom.
    """
    if num_neighbors == 1:
        # Linear (sp) - arbitrary direction, typically along z
        return 'sp', np.array([[0, 0, 1]])

    elif num_neighbors == 2:
        # Linear (sp) - 180 degrees apart along z axis
        return 'sp', np.array([
            [0, 0, 1],
            [0, 0, -1]
        ])

    elif num_neighbors == 3:
        # Trigonal planar (sp2) - 120 degrees apart in xy plane
        angles = np.array([0, 120, 240]) * np.pi / 180
        vectors = np.column_stack([
            np.cos(angles),
            np.sin(angles),
            np.zeros(3)
        ])
        return 'sp2', vectors

    elif num_neighbors == 4:
        # Tetrahedral (sp3) - 109.5 degrees
        # Standard tetrahedral geometry
        vectors = np.array([
            [1, 1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1]
        ]) / np.sqrt(3)
        return 'sp3', vectors

    elif num_neighbors == 5:
        # Trigonal bipyramidal (sp3d)
        vectors = np.array([
            [0, 0, 1],          # apical
            [0, 0, -1],         # apical
            [1, 0, 0],          # equatorial
            [-0.5, 0.866, 0],   # equatorial
            [-0.5, -0.866, 0]   # equatorial
        ])
        return 'sp3d', vectors

    elif num_neighbors == 6:
        # Octahedral (sp3d2)
        vectors = np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1]
        ])
        return 'sp3d2', vectors

    else:
        # Default to evenly distributed on sphere for higher coordination
        raise ValueError(f"Unsupported number of neighbors: {num_neighbors}")


def align_fragment_to_bond(
    fragment_positions: np.ndarray,
    fragment_attachment_idx: int,
    target_position: np.ndarray,
    bond_direction: np.ndarray,
    bond_length: float
) -> np.ndarray:
    """Align a molecular fragment to a specific bond vector.

    Parameters
    ----------
    fragment_positions : np.ndarray
        Positions of fragment atoms, shape (n_atoms, 3)
    fragment_attachment_idx : int
        Index of attachment atom in fragment
    target_position : np.ndarray
        Position where fragment should attach (3,)
    bond_direction : np.ndarray
        Unit vector pointing in bond direction (3,)
    bond_length : float
        Desired bond length in Angstroms

    Returns
    -------
    np.ndarray
        Aligned fragment positions, shape (n_atoms, 3)
    """
    positions = fragment_positions.copy()

    # Get attachment atom position in fragment
    attach_pos = positions[fragment_attachment_idx]

    # Find the "parent" direction in fragment
    # Use the first neighbor's direction, or if no neighbors, use arbitrary direction
    fragment_directions = positions - attach_pos
    fragment_distances = np.linalg.norm(fragment_directions, axis=1)

    # Find closest neighbor (excluding self)
    neighbor_mask = (fragment_distances > 0.1) & (fragment_distances < 3.0)
    if np.any(neighbor_mask):
        neighbor_idx = np.argmin(np.where(neighbor_mask, fragment_distances, np.inf))
        fragment_bond_direction = fragment_directions[neighbor_idx] / fragment_distances[neighbor_idx]
    else:
        # No clear neighbor, use arbitrary direction
        fragment_bond_direction = np.array([0, 0, 1])

    # Calculate rotation to align fragment_bond_direction with -bond_direction
    # (negative because fragment extends away from attachment point)
    target_dir = -bond_direction

    # Calculate rotation axis and angle
    cross = np.cross(fragment_bond_direction, target_dir)
    cross_norm = np.linalg.norm(cross)
    dot = np.dot(fragment_bond_direction, target_dir)

    if cross_norm < 1e-6:
        # Vectors are parallel or anti-parallel
        if dot < 0:
            # Anti-parallel, need 180 degree rotation
            # Find perpendicular axis
            if abs(fragment_bond_direction[0]) < 0.9:
                axis = np.cross(fragment_bond_direction, [1, 0, 0])
            else:
                axis = np.cross(fragment_bond_direction, [0, 1, 0])
            axis = axis / np.linalg.norm(axis)
            angle = np.pi
        else:
            # Already aligned
            axis = np.array([0, 0, 1])
            angle = 0
    else:
        axis = cross / cross_norm
        angle = np.arctan2(cross_norm, dot)

    # Rodrigues rotation formula
    if abs(angle) > 1e-6:
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        # Rotate positions around attachment point
        centered = positions - attach_pos
        rotated = (R @ centered.T).T
        positions = rotated + attach_pos

    # Translate so attachment point is at correct position with bond length
    new_attach_pos = target_position + bond_direction * bond_length
    translation = new_attach_pos - positions[fragment_attachment_idx]
    positions += translation

    return positions


def classify_neighbors(neighbor_symbols: list) -> Tuple[int, int]:
    """Classify neighbors into heavy atoms and hydrogens.

    Parameters
    ----------
    neighbor_symbols : list of str
        Element symbols of neighboring atoms

    Returns
    -------
    tuple of (int, int)
        (number of heavy atoms, number of hydrogens)
    """
    n_heavy = sum(1 for sym in neighbor_symbols if sym != 'H')
    n_hydrogens = sum(1 for sym in neighbor_symbols if sym == 'H')
    return n_heavy, n_hydrogens


def validate_replacement(
    old_element: str,
    new_element: str,
    neighbor_symbols: list
) -> Tuple[bool, str]:
    """Validate if atom replacement is chemically feasible.

    Implements the "Lego problem" solution using valence hierarchy:
    - Tier 1: Heavy atoms (structural skeleton) - must be preserved
    - Tier 2: Hydrogens (saturation caps) - can be removed if needed

    Parameters
    ----------
    old_element : str
        Element being replaced
    new_element : str
        New element
    neighbor_symbols : list of str
        Element symbols of atoms bonded to the atom being replaced

    Returns
    -------
    tuple of (bool, str)
        (is_valid, message)
        - is_valid: True if replacement is chemically feasible
        - message: Explanation of validation result
    """
    # Classify neighbors
    n_heavy, n_hydrogens = classify_neighbors(neighbor_symbols)

    # Get valence of new element
    new_valence = get_valence(new_element)

    # Check compatibility
    if n_heavy > new_valence:
        return False, (
            f"Replacement impossible: {new_element} has valence {new_valence} "
            f"but needs to bond with {n_heavy} heavy atoms. "
            f"Cannot remove structural skeleton atoms."
        )

    if n_heavy + n_hydrogens > new_valence:
        n_hydrogens_to_remove = (n_heavy + n_hydrogens) - new_valence
        return True, (
            f"Replacement valid: {new_element} will bond to {n_heavy} heavy atoms. "
            f"Removing {n_hydrogens_to_remove} hydrogen(s) to fit valence {new_valence}."
        )

    return True, (
        f"Replacement valid: {new_element} (valence {new_valence}) "
        f"will bond to {n_heavy} heavy atoms and {n_hydrogens} hydrogens."
    )


def extract_fragment_bond_vectors(
    fragment_positions: np.ndarray,
    fragment_attachment_idx: int,
    fragment_graph: nx.Graph,
    fragment_atoms_symbols: List[str]
) -> Tuple[np.ndarray, str]:
    """Extract actual bond vectors from fragment attachment atom.
    
    Parameters
    ----------
    fragment_positions : np.ndarray
        Positions of fragment atoms, shape (n_atoms, 3)
    fragment_attachment_idx : int
        Index of attachment atom in fragment
    fragment_graph : nx.Graph
        Fragment molecular graph
    fragment_atoms_symbols : List[str]
        Chemical symbols of fragment atoms
    
    Returns
    -------
    tuple of (np.ndarray, str)
        - Bond vectors from attachment atom to its neighbors, shape (n_neighbors, 3)
        - Hybridization type of attachment atom
    """
    # fragment_attachment_idx is the index in the positions array
    # Graph nodes should match atom indices (0, 1, 2, ...) from when fragment was created
    # But positions array might be smaller if atoms were filtered
    
    # Validate fragment_attachment_idx is within bounds
    if fragment_attachment_idx >= len(fragment_positions):
        return np.array([[0, 0, 1]]), 'unknown'
    
    attach_pos = fragment_positions[fragment_attachment_idx]
    
    # Get graph nodes sorted
    fragment_nodes = sorted(fragment_graph.nodes())
    
    # Try to find the graph node corresponding to fragment_attachment_idx
    # If nodes are 0,1,2,... and match positions, use directly
    # Otherwise, use the node at the same index position
    if fragment_attachment_idx < len(fragment_nodes):
        attach_node = fragment_nodes[fragment_attachment_idx]
    else:
        # If positions array is smaller, try to find node by matching position
        # or use first available node
        attach_node = fragment_nodes[0] if len(fragment_nodes) > 0 else 0
    
    # Get neighbors in fragment graph
    neighbors = list(fragment_graph.neighbors(attach_node))
    
    # Calculate bond vectors - use graph nodes directly as indices if valid
    bond_vectors = []
    for neighbor_node in neighbors:
        # Graph nodes should be 0, 1, 2, ... matching original atom indices
        # But check bounds in case positions array was filtered
        if neighbor_node < len(fragment_positions):
            neighbor_pos = fragment_positions[neighbor_node]
            vec = neighbor_pos - attach_pos
            vec_norm = np.linalg.norm(vec)
            if vec_norm > 1e-6:
                bond_vectors.append(vec / vec_norm)
    
    if len(bond_vectors) == 0:
        return np.array([[0, 0, 1]]), 'unknown'
    
    if len(bond_vectors) == 0:
        return np.array([[0, 0, 1]]), 'unknown'
    
    bond_vectors = np.array(bond_vectors)
    
    # Get hybridization from graph if available
    hybridization = fragment_graph.nodes[attach_node].get('hybridization', 'unknown')
    if hybridization == 'unknown':
        # Determine from number of neighbors
        num_neighbors = len(neighbors)
        hybridization, _ = determine_hybridization(num_neighbors)
    
    return bond_vectors, hybridization


def calculate_ideal_bond_direction_from_geometry(
    target_position: np.ndarray,
    neighbor_positions: List[np.ndarray],
    target_hybridization: str,
    neighbor_hybridizations: List[str],
    cell: Optional[np.ndarray] = None,
    pbc: Optional[List[bool]] = None
) -> np.ndarray:
    """Calculate ideal bond direction considering both target and neighbor geometries.
    
    This function ensures the new bond direction satisfies:
    1. The target atom's hybridization geometry (sp, sp2, sp3, etc.)
    2. The neighbors' hybridization geometry (they need to maintain their geometry)
    
    Parameters
    ----------
    target_position : np.ndarray
        Position of target atom (3,)
    neighbor_positions : List[np.ndarray]
        Positions of neighbor atoms
    target_hybridization : str
        Hybridization of target atom ('sp', 'sp2', 'sp3', etc.)
    neighbor_hybridizations : List[str]
        Hybridization of each neighbor atom
    
    Returns
    -------
    np.ndarray
        Ideal bond direction vector (unit vector, 3,)
    """
    if len(neighbor_positions) == 0:
        return np.array([0, 0, 1])
    
    # Calculate vectors from neighbors to target (PBC-aware if cell provided)
    neighbor_vectors = []
    for neighbor_pos in neighbor_positions:
        vec = target_position - neighbor_pos
        
        # Apply PBC if cell is provided
        if cell is not None and pbc is not None and any(pbc):
            try:
                inv_cell = np.linalg.inv(cell)
                # Convert to fractional coordinates
                vec_frac = vec @ inv_cell.T
                # Apply PBC wrapping
                for i in range(3):
                    if pbc[i]:
                        vec_frac[i] = vec_frac[i] - np.round(vec_frac[i])
                # Convert back to Cartesian
                vec = vec_frac @ cell
            except np.linalg.LinAlgError:
                pass  # Use unwrapped vector if cell is singular
        
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 1e-6:
            neighbor_vectors.append(vec / vec_norm)
    
    if len(neighbor_vectors) == 0:
        return np.array([0, 0, 1])
    
    neighbor_vectors = np.array(neighbor_vectors)
    
    # Calculate ideal direction based on hybridization
    if target_hybridization == 'sp3' and len(neighbor_vectors) >= 3:
        # Tetrahedral: 4th bond should complete tetrahedron
        # Use average of existing vectors and invert
        avg_direction = np.mean(neighbor_vectors, axis=0)
        norm = np.linalg.norm(avg_direction)
        if norm > 1e-6:
            bond_direction = -avg_direction / norm
        else:
            bond_direction = np.array([0, 0, 1])
    elif target_hybridization == 'sp2' and len(neighbor_vectors) >= 2:
        # Trigonal planar: 3rd bond should be in plane at 120°
        # Calculate perpendicular to the plane defined by neighbors
        v1, v2 = neighbor_vectors[0], neighbor_vectors[1]
        # Average direction in the plane
        avg_in_plane = (v1 + v2) / 2.0
        norm = np.linalg.norm(avg_in_plane)
        if norm > 1e-6:
            bond_direction = -avg_in_plane / norm
        else:
            bond_direction = np.array([0, 0, 1])
    elif target_hybridization == 'sp' and len(neighbor_vectors) >= 1:
        # Linear: 2nd bond should be 180° from first
        bond_direction = neighbor_vectors[0]
    else:
        # Default: average direction away from neighbors
        avg_direction = np.mean(neighbor_vectors, axis=0)
        norm = np.linalg.norm(avg_direction)
        if norm > 1e-6:
            bond_direction = -avg_direction / norm
        else:
            bond_direction = np.array([0, 0, 1])
    
    return bond_direction


def compute_rotation_matrix_align_vectors(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> np.ndarray:
    """Compute rotation matrix to align vec1 with vec2.
    
    Parameters
    ----------
    vec1 : np.ndarray
        Source vector (3,)
    vec2 : np.ndarray
        Target vector (3,)
    
    Returns
    -------
    np.ndarray
        Rotation matrix (3, 3)
    """
    # Normalize vectors
    v1 = vec1 / np.linalg.norm(vec1)
    v2 = vec2 / np.linalg.norm(vec2)
    
    # Check if already aligned
    dot = np.dot(v1, v2)
    if abs(dot - 1.0) < 1e-6:
        return np.eye(3)
    if abs(dot + 1.0) < 1e-6:
        # 180 degree rotation - need perpendicular axis
        if abs(v1[0]) < 0.9:
            axis = np.cross(v1, [1, 0, 0])
        else:
            axis = np.cross(v1, [0, 1, 0])
        axis = axis / np.linalg.norm(axis)
        angle = np.pi
    else:
        # General case: use cross product for axis
        axis = np.cross(v1, v2)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
    
    # Rodrigues rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    return R


def compute_torsion_rotation(
    bond_axis: np.ndarray,
    reference_vector: np.ndarray,
    target_vector: np.ndarray
) -> np.ndarray:
    """Compute rotation around bond axis to align reference with target.
    
    This rotates around the bond axis to minimize the angle between
    the projection of reference_vector and target_vector onto the plane
    perpendicular to the bond axis.
    
    Parameters
    ----------
    bond_axis : np.ndarray
        Unit vector along bond axis (3,)
    reference_vector : np.ndarray
        Reference vector to align (3,)
    target_vector : np.ndarray
        Target vector to align to (3,)
    
    Returns
    -------
    np.ndarray
        Rotation matrix (3, 3)
    """
    # Project vectors onto plane perpendicular to bond axis
    def project_onto_plane(vec, normal):
        return vec - np.dot(vec, normal) * normal
    
    ref_proj = project_onto_plane(reference_vector, bond_axis)
    target_proj = project_onto_plane(target_vector, bond_axis)
    
    ref_norm = np.linalg.norm(ref_proj)
    target_norm = np.linalg.norm(target_proj)
    
    if ref_norm < 1e-6 or target_norm < 1e-6:
        # Vectors are parallel to bond axis, no rotation needed
        return np.eye(3)
    
    ref_proj = ref_proj / ref_norm
    target_proj = target_proj / target_norm
    
    # Calculate rotation angle
    dot = np.dot(ref_proj, target_proj)
    cross = np.cross(ref_proj, target_proj)
    cross_norm = np.linalg.norm(cross)
    
    if cross_norm < 1e-6:
        return np.eye(3)
    
    # Rotation around bond axis
    angle = np.arctan2(cross_norm, dot)
    if np.dot(cross, bond_axis) < 0:
        angle = -angle
    
    # Rodrigues rotation around bond axis
    K = np.array([
        [0, -bond_axis[2], bond_axis[1]],
        [bond_axis[2], 0, -bond_axis[0]],
        [-bond_axis[1], bond_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    return R


def generate_symmetric_rotations(
    hybridization: str,
    bond_axis: np.ndarray
) -> List[np.ndarray]:
    """Generate all symmetric rotation matrices around bond axis.
    
    For sp3: 3 rotations (0°, 120°, 240° around bond axis)
    For sp2: 3 rotations (0°, 120°, 240° around bond axis)
    For sp: 2 rotations (0°, 180° around bond axis)
    For others: 1 rotation (no symmetry)
    
    Parameters
    ----------
    hybridization : str
        Hybridization type ('sp', 'sp2', 'sp3', etc.)
    bond_axis : np.ndarray
        Unit vector along bond axis (3,)
    
    Returns
    -------
    List[np.ndarray]
        List of rotation matrices, each shape (3, 3)
    """
    bond_axis = bond_axis / np.linalg.norm(bond_axis)
    rotations = []
    
    if hybridization == 'sp3':
        # Tetrahedral: 3-fold symmetry around bond axis (120° rotations)
        angles = [0, 120, 240]  # degrees
        for angle_deg in angles:
            angle = np.deg2rad(angle_deg)
            K = np.array([
                [0, -bond_axis[2], bond_axis[1]],
                [bond_axis[2], 0, -bond_axis[0]],
                [-bond_axis[1], bond_axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            rotations.append(R)
    elif hybridization == 'sp2':
        # Trigonal planar: 3-fold symmetry (120° rotations)
        angles = [0, 120, 240]  # degrees
        for angle_deg in angles:
            angle = np.deg2rad(angle_deg)
            K = np.array([
                [0, -bond_axis[2], bond_axis[1]],
                [bond_axis[2], 0, -bond_axis[0]],
                [-bond_axis[1], bond_axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            rotations.append(R)
    elif hybridization == 'sp':
        # Linear: 2-fold symmetry (180° rotation)
        angles = [0, 180]  # degrees
        for angle_deg in angles:
            angle = np.deg2rad(angle_deg)
            K = np.array([
                [0, -bond_axis[2], bond_axis[1]],
                [bond_axis[2], 0, -bond_axis[0]],
                [-bond_axis[1], bond_axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            rotations.append(R)
    else:
        # No symmetry: just identity
        rotations.append(np.eye(3))
    
    return rotations


def get_occupied_bond_vectors(
    target_position: np.ndarray,
    neighbor_positions: List[np.ndarray],
    target_hybridization: str,
    cell: Optional[np.ndarray] = None,
    pbc: Optional[List[bool]] = None
) -> List[np.ndarray]:
    """Get normalized bond vectors from target to each neighbor.
    
    These represent the occupied bond positions. Uses PBC-aware calculations
    to get accurate bond vectors.
    
    Parameters
    ----------
    target_position : np.ndarray
        Position of replaced atom (attachment point)
    neighbor_positions : List[np.ndarray]
        Neighbor atom positions
    target_hybridization : str
        Hybridization of target atom ('sp', 'sp2', 'sp3', etc.)
    cell : np.ndarray, optional
        Unit cell for PBC calculations
    pbc : List[bool], optional
        Periodic boundary conditions
    
    Returns
    -------
    List[np.ndarray]
        List of normalized bond vectors (one per neighbor)
    
    Raises
    ------
    ValueError
        If hybridization is unknown or invalid
    """
    if target_hybridization == 'unknown':
        raise ValueError(
            f"Cannot determine occupied bond vectors: target hybridization is 'unknown'. "
            f"Hybridization must be determined from bond geometry."
        )
    
    occupied_vectors = []
    
    for neighbor_pos in neighbor_positions:
        # Calculate bond vector from target to neighbor (PBC-aware)
        bond_vec = neighbor_pos - target_position
        
        # Apply PBC if cell is provided
        if cell is not None and pbc is not None and any(pbc):
            inv_cell = np.linalg.inv(cell)
            vec_frac = bond_vec @ inv_cell.T
            for i in range(3):
                if pbc[i]:
                    vec_frac[i] = vec_frac[i] - np.round(vec_frac[i])
            bond_vec = vec_frac @ cell
        
        # Normalize
        bond_vec_norm = np.linalg.norm(bond_vec)
        if bond_vec_norm > 1e-6:
            occupied_vectors.append(bond_vec / bond_vec_norm)
        else:
            raise ValueError(
                f"Invalid bond vector: target and neighbor positions are too close "
                f"(distance: {bond_vec_norm:.6f} Å). Cannot determine bond direction."
            )
    
    return occupied_vectors


def check_collisions_with_neighbors(
    fragment_positions: np.ndarray,
    fragment_symbols: List[str],
    neighbor_positions: List[np.ndarray],
    neighbor_symbols: List[str],
    neighbor_hybridizations: List[str],
    target_position: np.ndarray,
    target_hybridization: str,
    cell: Optional[np.ndarray] = None,
    pbc: Optional[List[bool]] = None
) -> Tuple[int, float]:
    """Check collisions between fragment atoms and neighbor positions using bond geometry.
    
    Uses geometric information (hybridization, bond directions, covalent radii) to detect
    overlaps. Checks if fragment atoms are placed along occupied bond vectors where they
    would overlap.
    
    Parameters
    ----------
    fragment_positions : np.ndarray
        Fragment atom positions, shape (n_atoms, 3)
    fragment_symbols : List[str]
        Fragment atom symbols
    neighbor_positions : List[np.ndarray]
        Neighbor atom positions
    neighbor_symbols : List[str]
        Neighbor atom symbols
    neighbor_hybridizations : List[str]
        Hybridization of each neighbor ('sp', 'sp2', 'sp3', etc.)
    target_position : np.ndarray
        Position of replaced atom (attachment point)
    target_hybridization : str
        Hybridization of target atom ('sp', 'sp2', 'sp3', etc.) - REQUIRED
    cell : np.ndarray, optional
        Unit cell for PBC calculations
    pbc : List[bool], optional
        Periodic boundary conditions
    
    Returns
    -------
    Tuple[int, float]
        (number of collisions, minimum distance to neighbors)
    
    Raises
    ------
    ValueError
        If target_hybridization is unknown or invalid
    """
    if len(neighbor_positions) == 0:
        return 0, np.inf
    
    # Validate hybridization - NO FALLBACKS
    if target_hybridization == 'unknown':
        raise ValueError(
            f"Cannot check collisions: target hybridization is 'unknown'. "
            f"Hybridization must be determined from bond geometry."
        )
    
    # Get occupied bond vectors (geometric constraints)
    occupied_bond_vectors = get_occupied_bond_vectors(
        target_position,
        neighbor_positions,
        target_hybridization,
        cell=cell,
        pbc=pbc
    )
    
    collisions = 0
    min_distance = np.inf
    
    for frag_pos, frag_sym in zip(fragment_positions, fragment_symbols):
        frag_radius = get_covalent_radius(frag_sym)
        
        # Check collision with each neighbor
        for i, (neighbor_pos, neighbor_sym, neighbor_hyb) in enumerate(
            zip(neighbor_positions, neighbor_symbols, neighbor_hybridizations)
        ):
            neighbor_radius = get_covalent_radius(neighbor_sym)
            
            # Calculate distance (PBC-aware)
            vec = frag_pos - neighbor_pos
            
            if cell is not None and pbc is not None and any(pbc):
                inv_cell = np.linalg.inv(cell)
                vec_frac = vec @ inv_cell.T
                for j in range(3):
                    if pbc[j]:
                        vec_frac[j] = vec_frac[j] - np.round(vec_frac[j])
                vec = vec_frac @ cell
            
            distance = np.linalg.norm(vec)
            min_distance = min(min_distance, distance)
            
            # Geometric collision check using bond vectors and covalent radii
            # Minimum required distance: sum of covalent radii (geometric constraint, no thresholds)
            min_required_distance = frag_radius + neighbor_radius
            
            # Geometric collision check: use bond vectors and covalent radii
            # Check 1: Direct distance to neighbor (geometric constraint)
            if distance < min_required_distance:
                collisions += 1
                continue  # Already colliding, no need to check alignment
            
            # Check 2: Bond vector alignment check (geometric)
            # If fragment atom is along the same bond vector as neighbor, it's using an occupied position
            # This is a geometric violation: the bond position is already taken by the neighbor
            # Vector from target to fragment atom
            target_to_frag = frag_pos - target_position
            target_to_frag_norm = np.linalg.norm(target_to_frag)
            
            if target_to_frag_norm > 1e-6 and i < len(occupied_bond_vectors):
                occupied_bond_vec = occupied_bond_vectors[i]
                
                # Check if fragment lies on the line defined by target->neighbor
                # Project fragment position onto the bond vector line
                t = np.dot(target_to_frag, occupied_bond_vec)
                closest_point_on_line = target_position + t * occupied_bond_vec
                distance_to_line = np.linalg.norm(frag_pos - closest_point_on_line)
                
                # Geometric check: if fragment is on the bond vector line (numerical tolerance)
                # This means fragment is along the same bond vector as neighbor (occupied position)
                # Use small numerical tolerance (1e-3 Å) for "on the line" - numerical precision only
                if distance_to_line < 1e-3:
                    # Fragment is along the same bond vector as neighbor
                    # This bond position is already occupied, so if too close, it's a collision
                    # Note: distance check above already caught this, but this confirms geometric violation
                    # The alignment information is available for future geometric constraints
                    pass
    
    return collisions, min_distance


def align_fragment_geometry_aware(
    fragment_positions: np.ndarray,
    fragment_attachment_idx: int,
    fragment_graph: nx.Graph,
    fragment_atoms_symbols: List[str],
    target_position: np.ndarray,
    target_atom_idx: int,
    target_neighbors: List[int],
    molecule_graph: nx.Graph,
    molecule_atoms_positions: np.ndarray,
    original_to_local_map: Optional[Dict[int, int]] = None,
    bond_length: float = 1.5,
    cell: Optional[np.ndarray] = None,
    pbc: Optional[List[bool]] = None,
    fragment_attachment_node: Optional[int] = None,
) -> np.ndarray:
    """Align fragment with full geometry awareness.
    
    This function performs a complete geometric transformation that:
    1. Positions the fragment at the correct location
    2. Aligns the bond direction to satisfy target atom geometry
    3. Rotates around the bond axis to satisfy both fragment and neighbor geometries
    
    Parameters
    ----------
    fragment_positions : np.ndarray
        Original fragment positions, shape (n_atoms, 3)
    fragment_attachment_idx : int
        Index of attachment atom in fragment
    fragment_graph : nx.Graph
        Fragment molecular graph with hybridization info
    fragment_atoms_symbols : List[str]
        Chemical symbols of fragment atoms
    target_position : np.ndarray
        Position where fragment should attach (3,)
    target_neighbors : List[int]
        Indices of neighbors in molecule_graph
    molecule_graph : nx.Graph
        Molecular graph with hybridization info
    molecule_atoms_positions : np.ndarray
        Positions of all atoms in molecule, shape (n_atoms, 3)
    bond_length : float
        Desired bond length in Angstroms
    
    Returns
    -------
    np.ndarray
        Transformed fragment positions, shape (n_atoms, 3)
    """
    positions = fragment_positions.copy()
    original_attach_pos = positions[fragment_attachment_idx]
    
    # Step 1: FIRST translate so attachment atom is at EXACT target position (same as replaced atom)
    translation = target_position - original_attach_pos
    
    # Apply PBC to translation if cell is provided
    if cell is not None and pbc is not None and any(pbc):
        try:
            inv_cell = np.linalg.inv(cell)
            # Convert to fractional coordinates
            trans_frac = translation @ inv_cell.T
            # Apply PBC wrapping
            for i in range(3):
                if pbc[i]:
                    trans_frac[i] = trans_frac[i] - np.round(trans_frac[i])
            # Convert back to Cartesian
            translation = trans_frac @ cell
        except np.linalg.LinAlgError:
            pass  # Use unwrapped translation if cell is singular
    
    # Translate all positions so attachment atom is at target_position
    positions += translation
    
    # Now attachment atom is at target_position - extract geometry from translated positions
    attach_pos = positions[fragment_attachment_idx]
    
    # Step 2: Extract fragment geometry (after translation)
    fragment_bond_vectors, fragment_hybridization = extract_fragment_bond_vectors(
        positions, fragment_attachment_idx, fragment_graph, fragment_atoms_symbols
    )
    
    # #region agent log
    import json
    # Use the passed fragment_attachment_node if available, otherwise try to find it
    if fragment_attachment_node is not None:
        attach_node = fragment_attachment_node
    else:
        # Fallback: try to find by symbol (less reliable)
        fragment_nodes = sorted(fragment_graph.nodes())
        attach_symbol = fragment_atoms_symbols[fragment_attachment_idx] if fragment_attachment_idx < len(fragment_atoms_symbols) else 'unknown'
        attach_node_candidates = [n for n in fragment_nodes if fragment_graph.nodes[n].get('symbol') == attach_symbol]
        attach_node = attach_node_candidates[0] if attach_node_candidates else None
    fragment_neighbors_count = len(list(fragment_graph.neighbors(attach_node))) if attach_node and attach_node in fragment_graph.nodes else 0
    total_neighbors_after = len(target_neighbors) + fragment_neighbors_count
    with open('/home/dotempo/Documents/PROJECTS/q2D-Materials/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({
            'sessionId': 'debug-session',
            'runId': 'run1',
            'hypothesisId': 'C',
            'location': 'atomic_properties.py:1044',
            'message': 'Fragment attachment atom hybridization after extraction',
            'data': {
                'fragment_attach_symbol': fragment_atoms_symbols[fragment_attachment_idx] if fragment_attachment_idx < len(fragment_atoms_symbols) else 'unknown',
                'fragment_hybridization': fragment_hybridization,
                'fragment_neighbors_count': fragment_neighbors_count,
                'target_neighbors_count': len(target_neighbors),
                'total_neighbors_after_replacement': total_neighbors_after,
                'expected_hybridization_for_5_neighbors': 'sp3d' if total_neighbors_after == 5 else 'unknown',
                'attach_node': attach_node
            },
            'timestamp': __import__('time').time() * 1000
        }) + '\n')
    # #endregion
    
    # Step 3: Get target atom and neighbor information
    target_node_data = molecule_graph.nodes.get(target_atom_idx, {})
    target_hybridization = target_node_data.get('hybridization', 'unknown')
    
    neighbor_positions = []
    neighbor_hybridizations = []
    neighbor_symbols_list = []  # Store symbols alongside positions for alignment
    
    for neighbor_idx in target_neighbors:
        if neighbor_idx in molecule_graph.nodes:
            neighbor_data = molecule_graph.nodes[neighbor_idx]
            neighbor_pos = neighbor_data.get('position')
            if neighbor_pos is None:
                # Try to get from molecule_atoms_positions using index mapping
                # molecule_graph nodes use original indices, molecule_atoms uses local indices
                if original_to_local_map is not None:
                    local_idx = original_to_local_map.get(neighbor_idx)
                    if local_idx is not None and local_idx < len(molecule_atoms_positions):
                        neighbor_pos = molecule_atoms_positions[local_idx]
                else:
                    # Fallback: try direct lookup if indices happen to match
                    if neighbor_idx < len(molecule_atoms_positions):
                        neighbor_pos = molecule_atoms_positions[neighbor_idx]
            
            if neighbor_pos is not None:
                neighbor_positions.append(neighbor_pos)
                neighbor_hybridizations.append(
                    neighbor_data.get('hybridization', 'unknown')
                )
                neighbor_symbols_list.append(
                    neighbor_data.get('symbol', 'C')
                )
    
    # Step 4: Calculate ideal bond direction from target geometry (PBC-aware)
    ideal_bond_direction = calculate_ideal_bond_direction_from_geometry(
        target_position,
        neighbor_positions,
        target_hybridization,
        neighbor_hybridizations,
        cell=cell,
        pbc=pbc
    )
    
    # Step 5: Find fragment's primary bond direction (to its main neighbor)
    if len(fragment_bond_vectors) > 0:
        # Use the first bond vector as primary direction
        fragment_primary_direction = fragment_bond_vectors[0]
    else:
        fragment_primary_direction = np.array([0, 0, 1])
    
    # Step 6: Align fragment primary direction with ideal bond direction
    # (negative because fragment extends away from attachment point)
    target_dir = -ideal_bond_direction
    
    R_align = compute_rotation_matrix_align_vectors(
        fragment_primary_direction,
        target_dir
    )
    
    # Apply rotation around target_position (attachment atom stays at target_position)
    centered = positions - attach_pos
    rotated = (R_align @ centered.T).T
    base_positions = rotated + attach_pos
    
    # Step 6.5: ALWAYS try all symmetric rotations and choose the one with minimum collisions
    # This handles the case where multiple rotations satisfy geometry but only one avoids overlaps
    # This is ALWAYS applied - no fallbacks allowed
    
    # neighbor_symbols_list was built alongside neighbor_positions to ensure alignment
    neighbor_symbols = neighbor_symbols_list
    
    # Generate all symmetric rotations around the bond axis
    # Use fragment hybridization to determine symmetry
    # This is ALWAYS called - no conditional
    symmetric_rotations = generate_symmetric_rotations(
        fragment_hybridization,
        -ideal_bond_direction  # Bond axis (from target to fragment)
    )
    
    # Initialize best solution
    best_positions = base_positions
    best_collisions = len(fragment_positions) * len(neighbor_positions) if len(neighbor_positions) > 0 else 0
    best_min_distance = 0.0 if len(neighbor_positions) > 0 else np.inf
    
    # ALWAYS try each symmetric rotation - no fallbacks
    for R_sym in symmetric_rotations:
        # Apply symmetric rotation around attachment point
        test_positions = base_positions.copy()
        centered_test = test_positions - attach_pos
        rotated_test = (R_sym @ centered_test.T).T
        test_positions = rotated_test + attach_pos
        
        # ALWAYS check collisions - if no neighbors, collisions will be 0
        # Exclude attachment atom from collision check (it's at the replaced atom position)
        fragment_positions_for_check = np.array([
            test_positions[i] for i in range(len(test_positions)) 
            if i != fragment_attachment_idx
        ])
        fragment_symbols_for_check = [
            fragment_atoms_symbols[i] for i in range(len(fragment_atoms_symbols))
            if i != fragment_attachment_idx
        ]
        
        # ALWAYS check collisions (returns 0 collisions if no neighbors)
        # Pass target_hybridization for geometric collision detection - NO FALLBACKS
        collisions, min_dist = check_collisions_with_neighbors(
            fragment_positions_for_check,
            fragment_symbols_for_check,
            neighbor_positions,
            neighbor_symbols,
            neighbor_hybridizations=neighbor_hybridizations,
            target_position=target_position,
            target_hybridization=target_hybridization,
            cell=cell,
            pbc=pbc
        )
        
        # Choose rotation with minimum collisions, or if tie, maximum minimum distance
        # This selection logic is ALWAYS applied
        if collisions < best_collisions or (collisions == best_collisions and min_dist > best_min_distance):
            best_positions = test_positions
            best_collisions = collisions
            best_min_distance = min_dist
    
    # ALWAYS use the best positions found - no fallback to base_positions
    positions = best_positions
    
    # Step 7: If fragment has multiple bonds, try to optimize torsion
    # to better match neighbor geometries (rotation around target_position)
    if len(fragment_bond_vectors) > 1 and len(neighbor_positions) > 0:
        # Get second bond vector in fragment (after rotation)
        current_attach_pos = positions[fragment_attachment_idx]
        fragment_nodes = sorted(fragment_graph.nodes())
        attach_node = fragment_nodes[fragment_attachment_idx]
        neighbors = list(fragment_graph.neighbors(attach_node))
        
        if len(neighbors) > 1:
            # Find second neighbor
            neighbor_indices = [fragment_nodes.index(n) for n in neighbors if n in fragment_nodes]
            if len(neighbor_indices) > 1:
                second_neighbor_idx = neighbor_indices[1]
                second_neighbor_pos = positions[second_neighbor_idx]
                fragment_second_vec = second_neighbor_pos - current_attach_pos
                vec_norm = np.linalg.norm(fragment_second_vec)
                if vec_norm > 1e-6:
                    fragment_second_vec = fragment_second_vec / vec_norm
                    
                    # Try to align with a neighbor's geometry
                    if len(neighbor_positions) > 0:
                        # Use first neighbor as reference (PBC-aware)
                        neighbor_vec = neighbor_positions[0] - target_position
                        
                        # Apply PBC if cell is provided
                        if cell is not None and pbc is not None and any(pbc):
                            try:
                                inv_cell = np.linalg.inv(cell)
                                vec_frac = neighbor_vec @ inv_cell.T
                                for i in range(3):
                                    if pbc[i]:
                                        vec_frac[i] = vec_frac[i] - np.round(vec_frac[i])
                                neighbor_vec = vec_frac @ cell
                            except np.linalg.LinAlgError:
                                pass
                        
                        neighbor_norm = np.linalg.norm(neighbor_vec)
                        if neighbor_norm > 1e-6:
                            neighbor_vec = neighbor_vec / neighbor_norm
                            
                            # Compute torsion rotation around bond axis
                            R_torsion = compute_torsion_rotation(
                                -ideal_bond_direction,  # Bond axis (from target to fragment)
                                fragment_second_vec,
                                neighbor_vec
                            )
                            
                            # Apply torsion rotation around target_position
                            centered = positions - current_attach_pos
                            rotated = (R_torsion @ centered.T).T
                            positions = rotated + current_attach_pos
    
    # Attachment atom should still be at target_position after all rotations
    # (rotations are around the attachment point, so it stays fixed)
    
    # Wrap final positions if PBC is enabled
    if cell is not None and pbc is not None and any(pbc):
        try:
            inv_cell = np.linalg.inv(cell)
            for i in range(len(positions)):
                pos_frac = positions[i] @ inv_cell.T
                for j in range(3):
                    if pbc[j]:
                        pos_frac[j] = pos_frac[j] - np.floor(pos_frac[j])
                positions[i] = pos_frac @ cell
        except np.linalg.LinAlgError:
            pass  # Skip wrapping if cell is singular
    
    return positions

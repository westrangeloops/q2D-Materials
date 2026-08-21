"""Cavity deformation analysis for cuboctahedral and square antiprism structures.

This module provides functions to calculate deformation metrics (Eta, Kappa, Nu)
for both full cuboctahedral cavities (A-site) and square antiprism half-cage structures
(DJ/RP spacers).

Uses graph-based topology for face identification and geometry for measurements.
Face identification uses direct B-X connectivity and atom properties (is_equatorial, is_terminal).

Deformation Metrics:
- Eta: 90° - mean deviation of square face angles
- Kappa: 60° (equilateral) or 90-45-45 (isosceles) - triangle face angle deviations
- Nu: Distance from A-site cation/molecule center to X-cage geometric center
- Omega: Mean X-X edge length deviation (calculated from face geometry)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Set
from scipy.spatial import ConvexHull
import networkx as nx


def calculate_ideal_cuboctahedron_properties(bx_distance: float) -> Dict[str, float]:
    """Calculate all ideal cuboctahedron properties analytically.
    
    For a cuboctahedron where each X atom is at distance BX from center B atom:
    - Edge length (X-X): a = BX × √2
    - Volume: V = (20/3) × d³
    - Surface area: S = 2(6 + 3√3) × a²
    
    Parameters
    ----------
    bx_distance : float
        B-X bond distance in Angstroms
        
    Returns
    -------
    dict
        {
            'edge_length': float,      # a = d × √2
            'volume': float,           # (20/3) × d³
            'surface_area': float,     # 2(6 + 3√3) × a²
            'square_angle': 90.0,
            'triangle_angle': 60.0,
        }
    """
    edge_length = bx_distance * np.sqrt(2)
    volume = (20.0 / 3.0) * (bx_distance ** 3)
    surface_area = 2 * (6 + 3 * np.sqrt(3)) * (edge_length ** 2)
    
    return {
        'edge_length': edge_length,
        'volume': volume,
        'surface_area': surface_area,
        'square_angle': 90.0,
        'triangle_angle': 60.0,
    }


def calculate_ideal_antiprism_properties(bx_distance: float) -> Dict[str, float]:
    """Calculate ideal square antiprism (half-cuboctahedron) properties analytically.

    A square antiprism is formed by cutting a cuboctahedron parallel to square faces.
    Geometry:
    - Edge length (X-X): a = BX × √2
    - Square side length: √2·a = 2·BX
    - Volume: V_half = (10/3) × d³ (half of cuboctahedron)
    - Surface area: 2 squares + 4 equilateral triangles + 4 isosceles triangles

    The structure has:
    - 2 square faces (top and bottom) with 90° angles
    - 4 equilateral triangle faces with 60° angles
    - 4 right-isosceles triangle faces with 90°-45°-45° angles

    Parameters
    ----------
    bx_distance : float
        B-X bond distance in Angstroms

    Returns
    -------
    dict
        {
            'edge_length': float,              # a = d × √2
            'volume': float,                   # (10/3) × d³
            'surface_area': float,             # Total surface area
            'square_angle': 90.0,
            'equilateral_angle': 60.0,
            'isosceles_angles': [90.0, 45.0, 45.0],
        }
    """
    edge_length = bx_distance * np.sqrt(2)
    square_side = edge_length * np.sqrt(2)  # = 2·BX

    # Volume: exactly half of cuboctahedron
    volume = (10.0 / 3.0) * (bx_distance ** 3)

    # Surface area components:
    # - 2 square faces (top and bottom)
    square_area = 2 * (square_side ** 2)

    # - 4 equilateral triangles (60°-60°-60°)
    equilateral_area = 4 * (np.sqrt(3) / 4) * (edge_length ** 2)

    # - 4 right-isosceles triangles (90°-45°-45°)
    # Each has two sides of length 'edge_length' at 90°
    isosceles_area = 4 * (0.5 * edge_length * edge_length)

    surface_area = square_area + equilateral_area + isosceles_area

    return {
        'edge_length': edge_length,
        'volume': volume,
        'surface_area': surface_area,
        'square_angle': 90.0,
        'equilateral_angle': 60.0,
        'isosceles_angles': [90.0, 45.0, 45.0],
    }


def measure_cavity_properties(cavity: 'Cavity', cage_node_id: str = None) -> Dict[str, Any]:
    """Measure actual cavity properties using path-minimization face identification.

    Uses path minimization in the B-X graph to identify faces, then measures
    geometric properties (angles, distances) on the identified faces.

    Parameters
    ----------
    cavity : Cavity
        Cavity object with subgraph containing X-atom positions
    cage_node_id : str, optional
        If provided, only measure properties for atoms connected to this cage.
        Critical for DJ spacers where half-cages can share atoms.

    Returns
    -------
    dict
        {
            'square_angles': list,
            'equilateral_angles': list,
            'isosceles_angles': list,
            'edge_lengths': list,
            'volume': float,
            'surface_area': float,
            'x_center': np.ndarray,
            'terminal_count': int,
            'equilateral_count': int,
            'isosceles_count': int,
            'square_count': int,
        }
    """

    # Extract X positions and node IDs (filtered by cage if specified)
    x_positions, x_node_ids = _extract_x_positions(cavity, cage_node_id=cage_node_id)
    node_to_index = {node_id: i for i, node_id in enumerate(x_node_ids)}
    x_node_ids_set = set(x_node_ids)

    # Calculate convex hull for volume/surface
    hull = ConvexHull(x_positions)

    # Graph-based face identification (cage-aware for DJ spacers)
    square_faces = _identify_square_faces_topology(
        cavity, cage_node_id=cage_node_id
    )
    equilateral_faces = _identify_equilateral_triangles_from_graph(
        cavity, cage_node_id=cage_node_id
    )
    isosceles_faces = _identify_isosceles_triangles_from_graph(
        cavity, cage_node_id=cage_node_id
    )

    # Helper: measure angles for a face given node IDs
    def measure_face_angles(face_nodes):
        """Measure all angles in a face (triangle or square)."""
        angles = []
        indices = [node_to_index[node] for node in face_nodes]
        n = len(indices)

        # For each vertex, calculate angle at that vertex
        for i in range(n):
            prev_idx = indices[(i - 1) % n]
            curr_idx = indices[i]
            next_idx = indices[(i + 1) % n]

            angle = _calculate_angle(
                x_positions[prev_idx],
                x_positions[curr_idx],
                x_positions[next_idx]
            )
            angles.append(angle)

        return angles

    # Measure angles for each face type
    square_angles = []
    for square in square_faces:
        angles = measure_face_angles(square)
        # Filter out NaN values (from coincident atoms)
        valid_angles = [a for a in angles if not np.isnan(a)]
        square_angles.extend(valid_angles)

    equilateral_angles = []
    for triangle in equilateral_faces:
        angles = measure_face_angles(triangle)
        # Filter out NaN values (from coincident atoms)
        valid_angles = [a for a in angles if not np.isnan(a)]
        equilateral_angles.extend(valid_angles)

    isosceles_angles = []
    for triangle in isosceles_faces:
        angles = measure_face_angles(triangle)
        # Filter out NaN values (from coincident atoms)
        valid_angles = [a for a in angles if not np.isnan(a)]
        isosceles_angles.extend(valid_angles)

    # Calculate X-X edge lengths from face geometry
    # Extract edge lengths from identified faces (squares and triangles)
    edge_lengths = []
    edge_set = set()  # Track edges to avoid duplicates
    
    # Helper: add edges from a face
    def add_face_edges(face_nodes):
        """Add edge lengths from a face, avoiding duplicates."""
        n = len(face_nodes)
        for i in range(n):
            node_i = face_nodes[i]
            node_j = face_nodes[(i + 1) % n]
            
            # Create canonical edge identifier (sorted tuple)
            edge_key = tuple(sorted([node_i, node_j]))
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                if node_i in node_to_index and node_j in node_to_index:
                    idx_i = node_to_index[node_i]
                    idx_j = node_to_index[node_j]
                    dist = np.linalg.norm(x_positions[idx_i] - x_positions[idx_j])
                    edge_lengths.append(dist)
    
    # Extract edges from all identified faces
    for square in square_faces:
        add_face_edges(square)
    for triangle in equilateral_faces:
        add_face_edges(triangle)
    for triangle in isosceles_faces:
        add_face_edges(triangle)

    # Count terminal atoms (only in this cage if cage filtering is active)
    if cage_node_id is not None:
        terminal_count = sum(
            1 for node in x_node_ids
            if cavity.subgraph.nodes[node].get('is_terminal', False)
        )
    else:
        terminal_count = sum(
            1 for node in cavity.subgraph.nodes()
            if (cavity.subgraph.nodes[node].get('is_terminal', False) and
                cavity.subgraph.nodes[node].get('is_X', False))
        )

    return {
        'square_angles': square_angles,
        'equilateral_angles': equilateral_angles,
        'isosceles_angles': isosceles_angles,
        'edge_lengths': edge_lengths,
        'volume': hull.volume,
        'surface_area': hull.area,
        'x_center': np.mean(x_positions, axis=0),
        'terminal_count': terminal_count,
        'equilateral_count': len(equilateral_faces),
        'isosceles_count': len(isosceles_faces),
        'square_count': len(square_faces),
    }


def _extract_b_species(cavity: 'Cavity') -> List[str]:
    """Extract unique B-site species from cavity subgraph.

    B-sites are identified by the is_B property.

    Parameters
    ----------
    cavity : Cavity
        Cavity object with subgraph containing atom nodes

    Returns
    -------
    list of str
        Unique B-site element symbols (sorted)

    Raises
    ------
    ValueError
        If no B-site atoms found in cavity
    """
    b_species = set()

    # B-atoms are identified by is_B property
    # Use nodes(data=True) to get node data in a single pass O(n)
    for node, node_data in cavity.subgraph.nodes(data=True):
        if node_data.get('is_B', False):
            symbol = node_data.get('symbol', '')
            if symbol:
                b_species.add(symbol)

    if not b_species:
        raise ValueError(f"No B-site atoms found in cavity {cavity.id}")

    return sorted(list(b_species))


def _extract_x_species(cavity: 'Cavity') -> List[str]:
    """Extract unique X-site species from cavity subgraph.

    X-sites are identified using the is_X property.

    Parameters
    ----------
    cavity : Cavity
        Cavity object with subgraph containing atom nodes

    Returns
    -------
    list of str
        Unique X-site element symbols (sorted)

    Raises
    ------
    ValueError
        If no X-site atoms found in cavity
    """
    x_species = set()

    # Use nodes(data=True) to get node data in a single pass O(n)
    for node, node_data in cavity.subgraph.nodes(data=True):
        if node_data.get('node_type') == 'atom' and node_data.get('is_X', False):
            symbol = node_data.get('symbol', '')
            if symbol:
                x_species.add(symbol)

    if not x_species:
        raise ValueError(f"No X-site atoms found in cavity {cavity.id}")

    return sorted(list(x_species))


def _calculate_mean_bx_distance(
    b_species: Union[str, List[str]],
    x_species: Union[str, List[str]]
) -> float:
    """Calculate mean BX distance for potentially mixed compositions.
    
    For double perovskites or mixed halides, calculates all B-X combinations
    using the ionic radii database and returns their mean.
    
    Uses the existing calculate_BX_distance function which already handles
    lists of species via vectorized operations.
    
    Parameters
    ----------
    b_species : str or list of str
        B-site cation symbol(s)
    x_species : str or list of str
        X-site anion symbol(s)
        
    Returns
    -------
    float
        Mean B-X bond distance in Angstroms
        
    Examples
    --------
    Pure perovskite:
        _calculate_mean_bx_distance('Pb', 'Br')  # Returns: 3.15 Å
    
    Double perovskite:
        _calculate_mean_bx_distance(['Pb', 'Sn'], 'Br')
        # Returns: mean(BX(Pb,Br), BX(Sn,Br))
    
    Mixed halide:
        _calculate_mean_bx_distance('Pb', ['Br', 'I'])
        # Returns: mean(BX(Pb,Br), BX(Pb,I))
    
    Double perovskite + mixed halide:
        _calculate_mean_bx_distance(['Pb', 'Sn'], ['Br', 'I'])
        # Returns: mean(BX(Pb,Br), BX(Pb,I), BX(Sn,Br), BX(Sn,I))
    """
    from ...utils.sites.A_sites import calculate_BX_distance
    
    # Use the existing calculate_BX_distance which handles lists with mode='mean'
    # This already performs all pairwise combinations and returns their mean
    return calculate_BX_distance(b_species, x_species, mode='mean')


def generate_ideal_cuboctahedron(bx_distance: float) -> np.ndarray:
    """Generate 12 vertices of an ideal cuboctahedron with given BX distance.
    
    A cuboctahedron has 12 vertices at permutations of (±1, ±1, 0) when
    normalized. The vertices are scaled so that the distance from the center
    to each vertex equals bx_distance.
    
    Parameters
    ----------
    bx_distance : float
        Distance from center to vertex (B-X bond length in Angstroms)
        
    Returns
    -------
    np.ndarray
        Shape (12, 3) array of vertex positions centered at origin
    """
    # Cuboctahedron vertices: all permutations of (±1, ±1, 0)
    # 4 vertices in XY plane: (±1, ±1, 0)
    # 4 vertices in XZ plane: (±1, 0, ±1)
    # 4 vertices in YZ plane: (0, ±1, ±1)
    
    vertices = np.array([
        # XY plane (z=0)
        [1, 1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [-1, -1, 0],
        # XZ plane (y=0)
        [1, 0, 1],
        [1, 0, -1],
        [-1, 0, 1],
        [-1, 0, -1],
        # YZ plane (x=0)
        [0, 1, 1],
        [0, 1, -1],
        [0, -1, 1],
        [0, -1, -1],
    ], dtype=float)
    
    # Normalize: distance from origin to vertex in unit cuboctahedron is sqrt(2)
    # Scale to achieve desired bx_distance
    scale_factor = bx_distance / np.sqrt(2.0)
    vertices *= scale_factor
    
    return vertices


def generate_ideal_antiprism(bx_distance: float) -> np.ndarray:
    """Generate 8 vertices of an ideal square antiprism (half-cuboctahedron).

    A square antiprism represents a half-cage structure for DJ/RP spacers.
    It is formed by cutting a cuboctahedron parallel to its square faces.
    The resulting geometry has:
    - 4 vertices forming a square top (side length √2·BX)
    - 4 vertices forming a square bottom (side length √2·BX)
    - 4 equilateral triangular faces (from original cuboctahedron)
    - 4 right-isosceles triangular faces (from bisecting original square faces)

    Parameters
    ----------
    bx_distance : float
        Distance from center to vertex (B-X bond length in Angstroms)

    Returns
    -------
    np.ndarray
        Shape (8, 3) array of vertex positions centered at origin
    """
    # Start with a full cuboctahedron (12 vertices)
    full_cubo = generate_ideal_cuboctahedron(bx_distance)

    # Select 8 vertices by excluding the XY plane (z=0)
    # This keeps XZ plane (y=0) and YZ plane (x=0) vertices
    # These form the square antiprism geometry
    z_coords = full_cubo[:, 2]
    antiprism_vertices = full_cubo[np.abs(z_coords) > 1e-6]

    # Verify we have exactly 8 vertices
    if len(antiprism_vertices) != 8:
        raise ValueError(f"Expected 8 antiprism vertices, got {len(antiprism_vertices)}")

    return antiprism_vertices


def _get_b_atoms_for_cage(cavity: 'Cavity', cage_node_id: str = None) -> List[str]:
    """Get B atoms filtered by cage node.
    
    Parameters
    ----------
    cavity : Cavity
        Cavity object with subgraph containing atom nodes
    cage_node_id : str, optional
        If provided, only get B atoms connected to this cage node.
        This is crucial for DJ spacers where half-cages can share atoms.
        
    Returns
    -------
    list of str
        List of B atom node IDs
    """
    b_atoms = []
    
    # If cage filtering is requested, get B atoms connected to that cage
    if cage_node_id is not None:
        if cage_node_id not in cavity.subgraph.nodes():
            raise ValueError(f"Cage node '{cage_node_id}' not found in cavity subgraph")
        
        # Get all B-atoms connected to this cage
        for neighbor in cavity.subgraph.neighbors(cage_node_id):
            node_data = cavity.subgraph.nodes[neighbor]
            if (node_data.get('node_type') == 'atom' and 
                node_data.get('is_B', False)):
                edge_data = cavity.subgraph.get_edge_data(cage_node_id, neighbor)
                if edge_data and edge_data.get('edge_type') == 'contains':
                    b_atoms.append(neighbor)
    else:
        # No filtering, extract all B-atoms
        # Use nodes(data=True) to get node data in a single pass O(n)
        for node, node_data in cavity.subgraph.nodes(data=True):
            if (node_data.get('node_type') == 'atom' and 
                node_data.get('is_B', False)):
                b_atoms.append(node)
    
    return b_atoms


def _identify_octagonal_rings(
    cavity: 'Cavity',
    cage_node_id: str = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Identify octagonal rings formed by B and X atoms for shearing analysis.

    For antiprisms (half-cages):
    - Ring 1: 4 B atoms + 4 equatorial X atoms
    - Ring 2: 4 B atoms + 4 terminal X atoms

    For cuboctahedra (A-sites):
    - Ring 1: 4 B atoms + 4 equatorial X atoms (lower z)
    - Ring 2: 4 B atoms + 4 equatorial X atoms (upper z)

    Parameters
    ----------
    cavity : Cavity
        Cavity object with subgraph
    cage_node_id : str, optional
        If provided, only consider atoms connected to this cage node

    Returns
    -------
    tuple
        (octagon1_positions, octagon2_positions) where each is (8, 3) array
        or (None, None) if octagons cannot be identified
    """
    # Get B atoms for this cage
    b_atoms = _get_b_atoms_for_cage(cavity, cage_node_id=cage_node_id)

    if len(b_atoms) < 4:
        return None, None

    # Extract B atom positions
    b_positions = []
    for b_node in b_atoms[:4]:  # Take first 4 B atoms
        node_data = cavity.subgraph.nodes[b_node]
        pbc_pos = node_data.get('pbc_position')
        if pbc_pos is not None:
            b_positions.append(np.array(pbc_pos))

    if len(b_positions) != 4:
        return None, None

    b_positions = np.array(b_positions)

    # Get equatorial and terminal X atoms
    # IMPORTANT: Preserve graph connectivity by iterating through B atoms in order
    # This ensures B[i] is paired with X_terminal[i] and X_equatorial[i]
    equatorial_x_positions = []
    terminal_x_positions = []

    if cage_node_id is not None:
        # For each B atom (in order), find its bonded X atoms
        for b_node in b_atoms[:4]:  # Use first 4 B atoms in order
            # Find equatorial and terminal X bonded to this specific B
            b_equatorial_x = None
            b_terminal_x = None
            
            for neighbor in cavity.subgraph.neighbors(b_node):
                neighbor_data = cavity.subgraph.nodes[neighbor]
                if (neighbor_data.get('node_type') == 'atom' and
                    neighbor_data.get('is_X', False)):
                    edge_data = cavity.subgraph.get_edge_data(b_node, neighbor)
                    if edge_data and edge_data.get('edge_type') == 'bonded_to':
                        pbc_pos = neighbor_data.get('pbc_position')
                        if pbc_pos is None:
                            continue
                        
                        if neighbor_data.get('is_equatorial', False):
                            b_equatorial_x = np.array(pbc_pos)
                        elif neighbor_data.get('is_terminal', False):
                            b_terminal_x = np.array(pbc_pos)
            
            # Add X atoms in the same order as B atoms (preserving connectivity)
            if b_equatorial_x is not None:
                equatorial_x_positions.append(b_equatorial_x)
            if b_terminal_x is not None:
                terminal_x_positions.append(b_terminal_x)
    else:
        # No cage_node_id filtering - still preserve B-X connectivity
        # For each B atom (in order), find its bonded X atoms
        for b_node in b_atoms[:4]:  # Use first 4 B atoms in order
            # Find equatorial and terminal X bonded to this specific B
            b_equatorial_x = None
            b_terminal_x = None
            
            for neighbor in cavity.subgraph.neighbors(b_node):
                neighbor_data = cavity.subgraph.nodes[neighbor]
                if (neighbor_data.get('node_type') == 'atom' and
                    neighbor_data.get('is_X', False)):
                    edge_data = cavity.subgraph.get_edge_data(b_node, neighbor)
                    if edge_data and edge_data.get('edge_type') == 'bonded_to':
                        pbc_pos = neighbor_data.get('pbc_position')
                        if pbc_pos is None:
                            continue
                        
                        if neighbor_data.get('is_equatorial', False):
                            b_equatorial_x = np.array(pbc_pos)
                        elif neighbor_data.get('is_terminal', False):
                            b_terminal_x = np.array(pbc_pos)
            
            # Add X atoms in the same order as B atoms (preserving connectivity)
            if b_equatorial_x is not None:
                equatorial_x_positions.append(b_equatorial_x)
            if b_terminal_x is not None:
                terminal_x_positions.append(b_terminal_x)

    # For antiprisms (DJ/RP spacers), we expect 4 equatorial and 4 terminal X atoms
    if len(equatorial_x_positions) >= 4 and len(terminal_x_positions) >= 4:
        equatorial_x_positions = np.array(equatorial_x_positions[:4])
        terminal_x_positions = np.array(terminal_x_positions[:4])

        # Octagon 1: 4 B atoms + 4 equatorial X atoms
        octagon1 = np.vstack([b_positions, equatorial_x_positions])

        # Octagon 2: 4 B atoms + 4 terminal X atoms
        octagon2 = np.vstack([b_positions, terminal_x_positions])

        return octagon1, octagon2

    # For A-sites (cuboctahedra), we expect 8 equatorial X atoms split by z-coordinate
    if len(equatorial_x_positions) >= 8:
        equatorial_x_positions = np.array(equatorial_x_positions)

        # Sort equatorial X atoms by z-coordinate
        z_coords = equatorial_x_positions[:, 2]
        sorted_indices = np.argsort(z_coords)

        # Split into lower 4 and upper 4
        lower_indices = sorted_indices[:4]
        upper_indices = sorted_indices[4:8]

        equatorial_x_lower = equatorial_x_positions[lower_indices]
        equatorial_x_upper = equatorial_x_positions[upper_indices]

        # Octagon 1: 4 B atoms + 4 lower equatorial X atoms
        octagon1 = np.vstack([b_positions, equatorial_x_lower])

        # Octagon 2: 4 B atoms + 4 upper equatorial X atoms
        octagon2 = np.vstack([b_positions, equatorial_x_upper])

        return octagon1, octagon2

    return None, None


def calculate_shearing_deformation(
    cavity: 'Cavity',
    cage_node_id: str = None
) -> Optional[Dict[str, Any]]:
    """Calculate shearing deformation between octagonal rings.

    Measures the displacement (magnitude and direction) between centroids
    of two octagonal rings formed by B and X atoms. This captures asymmetric
    distortion and shearing in the cage structure.

    Parameters
    ----------
    cavity : Cavity
        Cavity object with subgraph
    cage_node_id : str, optional
        If provided, only consider atoms connected to this cage node

    Returns
    -------
    dict or None
        {
            'shear_magnitude': float,         # Distance between centroids (Angstroms)
            'shear_vector': np.ndarray,       # 3D displacement vector
            'shear_direction': np.ndarray,    # Normalized direction vector
            'octagon1_centroid': np.ndarray,  # Centroid of ring 1
            'octagon2_centroid': np.ndarray,  # Centroid of ring 2
        }
        Returns None if octagonal rings cannot be identified
    """
    octagon1, octagon2 = _identify_octagonal_rings(cavity, cage_node_id=cage_node_id)

    if octagon1 is None or octagon2 is None:
        return None

    # Calculate centroids
    centroid1 = np.mean(octagon1, axis=0)
    centroid2 = np.mean(octagon2, axis=0)

    # Calculate displacement vector
    shear_vector = centroid2 - centroid1
    shear_magnitude = np.linalg.norm(shear_vector)

    # Normalized direction (handle zero magnitude)
    if shear_magnitude > 1e-10:
        shear_direction = shear_vector / shear_magnitude
    else:
        shear_direction = np.array([0.0, 0.0, 0.0])

    return {
        'shear_magnitude': float(shear_magnitude),
        'shear_vector': shear_vector.tolist(),
        'shear_direction': shear_direction.tolist(),
        'octagon1_centroid': centroid1.tolist(),
        'octagon2_centroid': centroid2.tolist(),
    }


def _extract_x_positions(cavity: 'Cavity', cage_node_id: str = None) -> Tuple[np.ndarray, List[str]]:
    """Extract only X-site atom positions from cavity subgraph.

    Parameters
    ----------
    cavity : Cavity
        Cavity object with subgraph containing atom nodes
    cage_node_id : str, optional
        If provided, only extract X-atoms connected to this cage node.
        This is crucial for DJ spacers where half-cages can share atoms.

    Returns
    -------
    tuple
        (x_positions, x_node_ids) where x_positions is (N, 3) array and
        x_node_ids is list of node identifiers
    """
    x_positions = []
    x_node_ids = []

    # If cage filtering is requested, get atoms connected to that cage
    if cage_node_id is not None:
        if cage_node_id not in cavity.subgraph.nodes():
            raise ValueError(f"Cage node '{cage_node_id}' not found in cavity subgraph")

        # Get all X-atoms via B atoms connected to this cage
        # X atoms are not directly connected to half_cage nodes, they connect via B atoms
        b_atoms_in_cage = _get_b_atoms_for_cage(cavity, cage_node_id=cage_node_id)
        
        # Get all X atoms connected to these B atoms via bonded_to edges
        x_atoms_set = set()  # Use set to avoid duplicates
        for b_node in b_atoms_in_cage:
            for neighbor in cavity.subgraph.neighbors(b_node):
                neighbor_data = cavity.subgraph.nodes[neighbor]
                if (neighbor_data.get('node_type') == 'atom' and 
                    neighbor_data.get('is_X', False)):
                    edge_data = cavity.subgraph.get_edge_data(b_node, neighbor)
                    if edge_data and edge_data.get('edge_type') == 'bonded_to':
                        x_atoms_set.add(neighbor)
        
        # Extract positions for all X atoms
        for x_node in x_atoms_set:
            node_data = cavity.subgraph.nodes[x_node]
            pbc_pos = node_data.get('pbc_position')
            if pbc_pos is not None:
                x_positions.append(np.array(pbc_pos))
                x_node_ids.append(x_node)
    else:
        # No filtering, extract all X-atoms
        # Use nodes(data=True) to get node data in a single pass O(n)
        for node, node_data in cavity.subgraph.nodes(data=True):
            if node_data.get('node_type') == 'atom' and node_data.get('is_X', False):
                pbc_pos = node_data.get('pbc_position')
                if pbc_pos is not None:
                    x_positions.append(np.array(pbc_pos))
                    x_node_ids.append(node)

    if not x_positions:
        raise ValueError(f"No X-site atoms found in cavity {cavity.id}" +
                        (f" for cage '{cage_node_id}'" if cage_node_id else ""))

    return np.array(x_positions), x_node_ids


def _identify_equilateral_triangles_from_graph(
    cavity: 'Cavity',
    cage_node_id: str = None
) -> List[Tuple[str, ...]]:
    """Identifies equilateral triangles by finding 3 X atoms bonded to each B atom.
    
    For antiprisms (half-cages), each B atom is bonded to 3 X atoms that form
    an equilateral triangle face. This method directly uses B-X connectivity
    via bonded_to edges.
    
    Parameters
    ----------
    cavity : Cavity
        Cavity object with subgraph containing B-X edges
    cage_node_id : str, optional
        If provided, only consider B atoms connected to this cage node.
        Critical for DJ spacers where half-cages can share atoms.
        
    Returns
    -------
    List[Tuple[str, ...]]
        List of equilateral triangle faces, each as (node_id1, node_id2, node_id3)
    """
    triangles = []
    
    # Get B atoms filtered by cage if specified
    b_atoms = _get_b_atoms_for_cage(cavity, cage_node_id=cage_node_id)
    
    # For each B atom, get its 3 X neighbors via bonded_to edges
    for b_node in b_atoms:
        x_neighbors = []
        
        # Get all X atoms connected to this B atom via bonded_to edges
        for neighbor in cavity.subgraph.neighbors(b_node):
            neighbor_data = cavity.subgraph.nodes[neighbor]
            if (neighbor_data.get('node_type') == 'atom' and 
                neighbor_data.get('is_X', False)):
                edge_data = cavity.subgraph.get_edge_data(b_node, neighbor)
                if edge_data and edge_data.get('edge_type') == 'bonded_to':
                    x_neighbors.append(neighbor)
        
        # Form triangle from the X atoms (expect 3, but take what we have)
        if len(x_neighbors) >= 3:
            # Take first 3 X atoms to form the triangle
            triangle = tuple(sorted(x_neighbors[:3]))
            triangles.append(triangle)
        elif len(x_neighbors) > 0:
            # If we have fewer than 3, still form a triangle (may be incomplete)
            triangle = tuple(sorted(x_neighbors))
            triangles.append(triangle)
    
    return triangles


def _identify_isosceles_triangles_from_graph(
    cavity: 'Cavity',
    cage_node_id: str = None
) -> List[Tuple[str, str, str]]:
    """Identifies isosceles triangle faces using B-atom pairs and equatorial/terminal properties.

    For antiprisms, isosceles triangles are formed by:
    - 1 shared equatorial X-atom (bridging two B-atoms, is_equatorial=True)
    - 1 terminal X-atom bonded to first B atom (is_terminal=True)
    - 1 terminal X-atom bonded to second B atom (is_terminal=True)

    Parameters
    ----------
    cavity : Cavity
        Cavity object with subgraph containing B-X edges
    cage_node_id : str, optional
        If provided, only consider B atoms connected to this cage node.
        Critical for DJ spacers where half-cages can share atoms.

    Returns
    -------
    List[Tuple[str, str, str]]
        List of isosceles triangle faces, each as (node_id1, node_id2, node_id3)
    """
    triangles = []
    
    # Get B atoms filtered by cage if specified
    b_atoms = _get_b_atoms_for_cage(cavity, cage_node_id=cage_node_id)
    
    # Make all pairs of B atoms (order doesn't matter)
    for i in range(len(b_atoms)):
        for j in range(i + 1, len(b_atoms)):
            b1, b2 = b_atoms[i], b_atoms[j]
            
            # Find shared equatorial X atoms (bonded to both B1 and B2)
            def get_x_atoms_for_b(b_node):
                """Get X atoms bonded to a B atom via bonded_to edges."""
                x_atoms = []
                for neighbor in cavity.subgraph.neighbors(b_node):
                    neighbor_data = cavity.subgraph.nodes[neighbor]
                    if (neighbor_data.get('node_type') == 'atom' and 
                        neighbor_data.get('is_X', False)):
                        edge_data = cavity.subgraph.get_edge_data(b_node, neighbor)
                        if edge_data and edge_data.get('edge_type') == 'bonded_to':
                            x_atoms.append(neighbor)
                return set(x_atoms)
            
            x_atoms_b1 = get_x_atoms_for_b(b1)
            x_atoms_b2 = get_x_atoms_for_b(b2)
            shared_x_atoms = x_atoms_b1 & x_atoms_b2
            
            # Filter to only equatorial X atoms
            shared_equatorial_x = [
                x for x in shared_x_atoms
                if cavity.subgraph.nodes[x].get('is_equatorial', False)
            ]
            
            # For each shared equatorial X atom, form triangles
            for shared_x in shared_equatorial_x:
                # Get terminal X atoms bonded to B1
                terminal_x_b1 = [
                    x for x in x_atoms_b1
                    if (cavity.subgraph.nodes[x].get('is_terminal', False) and
                        x != shared_x)
                ]
                
                # Get terminal X atoms bonded to B2
                terminal_x_b2 = [
                    x for x in x_atoms_b2
                    if (cavity.subgraph.nodes[x].get('is_terminal', False) and
                        x != shared_x)
                ]
                
                # Form triangles: (shared_equatorial_X, terminal_X_from_B1, terminal_X_from_B2)
                # for all combinations
                for t1 in terminal_x_b1:
                    for t2 in terminal_x_b2:
                        triangle = tuple(sorted([shared_x, t1, t2]))
                        triangles.append(triangle)
    
    return triangles


def _identify_square_faces_topology(cavity: 'Cavity', cage_node_id: str = None) -> List[Tuple[str, ...]]:
    """Identifies square faces for antiprisms (half-cages).
    
    For square antiprisms, there are two square faces:
    1. First square: all X atoms where is_equatorial=True and is_X=True
    2. Second square: all X atoms where is_X=True and is_terminal=True
    
    Parameters
    ----------
    cavity : Cavity
        Cavity object with subgraph containing X atoms
    cage_node_id : str, optional
        If provided, only consider X atoms connected to this cage node.
        Critical for DJ spacers where half-cages can share atoms.
        
    Returns
    -------
    List[Tuple[str, ...]]
        List of square faces, each as (node_id1, node_id2, node_id3, node_id4)
    """
    squares = []
    
    # Get X atoms filtered by cage if specified
    if cage_node_id is not None:
        if cage_node_id not in cavity.subgraph.nodes():
            raise ValueError(f"Cage node '{cage_node_id}' not found in cavity subgraph")
        
        # Get all X-atoms via B atoms connected to this cage
        # X atoms are not directly connected to half_cage nodes, they connect via B atoms
        x_atoms_equatorial = []
        x_atoms_terminal = []
        
        # First, get all B atoms connected to this cage
        b_atoms_in_cage = _get_b_atoms_for_cage(cavity, cage_node_id=cage_node_id)
        
        # Then, get all X atoms connected to these B atoms via bonded_to edges
        x_atoms_set = set()  # Use set to avoid duplicates
        for b_node in b_atoms_in_cage:
            for neighbor in cavity.subgraph.neighbors(b_node):
                neighbor_data = cavity.subgraph.nodes[neighbor]
                if (neighbor_data.get('node_type') == 'atom' and 
                    neighbor_data.get('is_X', False)):
                    edge_data = cavity.subgraph.get_edge_data(b_node, neighbor)
                    if edge_data and edge_data.get('edge_type') == 'bonded_to':
                        x_atoms_set.add(neighbor)
        
        # Now filter by equatorial/terminal properties
        for x_node in x_atoms_set:
            node_data = cavity.subgraph.nodes[x_node]
            if node_data.get('is_equatorial', False):
                x_atoms_equatorial.append(x_node)
            if node_data.get('is_terminal', False):
                x_atoms_terminal.append(x_node)
    else:
        # No filtering, extract all X-atoms
        # Use nodes(data=True) to get node data in a single pass O(n)
        x_atoms_equatorial = []
        x_atoms_terminal = []
        
        for node, node_data in cavity.subgraph.nodes(data=True):
            if (node_data.get('node_type') == 'atom' and 
                node_data.get('is_X', False)):
                if node_data.get('is_equatorial', False):
                    x_atoms_equatorial.append(node)
                if node_data.get('is_terminal', False):
                    x_atoms_terminal.append(node)
    
    # First square: all equatorial X atoms (is_equatorial=True and is_X=True)
    if len(x_atoms_equatorial) > 0:
        square1 = tuple(sorted(x_atoms_equatorial))
        squares.append(square1)
    
    # Second square: all terminal X atoms (is_X=True and is_terminal=True)
    if len(x_atoms_terminal) > 0:
        square2 = tuple(sorted(x_atoms_terminal))
        squares.append(square2)
    
    return squares


def _calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate angle at p2 formed by p1-p2-p3 in degrees.

    Parameters
    ----------
    p1, p2, p3 : np.ndarray
        3D positions

    Returns
    -------
    float
        Angle in degrees (or NaN if atoms are coincident)
    """
    vec1 = p1 - p2
    vec2 = p3 - p2

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Handle degenerate cases (coincident atoms)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return np.nan

    cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    return np.degrees(angle)


def _calculate_isosceles_deviation(angles: List[float], ideal_angles: List[float]) -> float:
    """Calculate deviation of isosceles triangle angles from ideal pattern.

    For right-isosceles triangles (90°-45°-45°), calculates how much the
    measured angles deviate from the expected pattern.

    Parameters
    ----------
    angles : List[float]
        List of all measured angles from isosceles triangles
    ideal_angles : List[float]
        Expected angle pattern, e.g., [90.0, 45.0, 45.0]

    Returns
    -------
    float
        Mean absolute deviation from ideal pattern
    """
    if not angles:
        return 0.0

    # Sort measured angles into groups similar to ideal pattern
    # For 90-45-45: expect 1/3 of angles near 90°, 2/3 near 45°
    sorted_angles = sorted(angles, reverse=True)

    # Split into groups based on expected pattern
    deviations = []

    # For right-isosceles (90-45-45), we expect:
    # - 1/3 of angles around 90°
    # - 2/3 of angles around 45°
    n_total = len(sorted_angles)
    n_90 = n_total // 3
    n_45 = n_total - n_90

    # Calculate deviations for 90° angles (largest)
    for i in range(n_90):
        deviations.append(abs(sorted_angles[i] - 90.0))

    # Calculate deviations for 45° angles (smaller)
    for i in range(n_90, n_total):
        deviations.append(abs(sorted_angles[i] - 45.0))

    return float(np.mean(deviations)) if deviations else 0.0


def calculate_cuboctahedron_deformation(
    cavity: 'Cavity',
    bx_distance: float,
    mode: str = 'delta',
    full: bool = False
) -> Dict[str, Any]:
    """Calculate deformation metrics for a full cuboctahedral cavity using analytical approach.
    
    Uses analytical formulas for ideal geometry instead of geometric alignment.
    
    Parameters
    ----------
    cavity : Cavity
        Cavity object with subgraph containing X-atom positions
    bx_distance : float
        Ideal B-X bond distance in Angstroms
    mode : str, optional
        Output mode: 'delta' (default) or 'absolute'
    full : bool, optional
        If True, return detailed per-face/edge data
        
    Returns
    -------
    dict
        Deformation metrics with mode='delta' or mode='absolute'
    """
    # Get ideal properties analytically
    ideal_props = calculate_ideal_cuboctahedron_properties(bx_distance)
    
    # Measure actual cavity properties
    actual_props = measure_cavity_properties(cavity)
    
    if mode == 'delta':
        # Delta mode: deviations from ideal

        # Calculate Eta: 90° - mean(square angle deviation)
        if not actual_props['square_angles']:
            import sys
            print(
                f"WARNING: No square faces found in cavity {cavity.id}. "
                f"Expected 6 square faces for cuboctahedron. "
                f"Returning NaN for eta.",
                file=sys.stderr
            )
            eta = np.nan
        else:
            eta_deviations = [abs(a - 90.0) for a in actual_props['square_angles']]
            eta = 90.0 - np.mean(eta_deviations)

        # Calculate Kappa: 60° - mean(equilateral triangle angle deviation)
        # Cuboctahedra have only equilateral triangles, no isosceles
        if not actual_props['equilateral_angles']:
            import sys
            print(
                f"WARNING: No equilateral triangle faces found in cavity {cavity.id}. "
                f"Expected 8 triangular faces for cuboctahedron. "
                f"Returning NaN for kappa.",
                file=sys.stderr
            )
            kappa = np.nan
        else:
            kappa_deviations = [abs(a - 60.0) for a in actual_props['equilateral_angles']]
            kappa = 60.0 - np.mean(kappa_deviations)

        # Nu: displacement of A-site from X-cage center
        nu = np.linalg.norm(cavity.center_position - actual_props['x_center'])

        # Omega: mean X-X edge length deviation
        if not actual_props['edge_lengths']:
            import sys
            print(
                f"WARNING: No X-X edge lengths calculated for cavity {cavity.id}. "
                f"Returning NaN for omega.",
                file=sys.stderr
            )
            omega = np.nan
        else:
            omega_deviations = [abs(e - ideal_props['edge_length']) for e in actual_props['edge_lengths']]
            omega = np.mean(omega_deviations)
        
        # Delta Volume: volume deviation
        delta_volume = actual_props['volume'] - ideal_props['volume']

        # Delta Surface Area: surface area deviation
        delta_surface_area = actual_props['surface_area'] - ideal_props['surface_area']

        # Calculate shearing deformation (displacement between octagonal ring centroids)
        # For A-sites, this may return None if octagonal rings cannot be identified
        shearing = calculate_shearing_deformation(cavity, cage_node_id=None)

        result = {
            'eta': float(eta),
            'kappa': float(kappa),
            'nu': float(nu),
            'omega': float(omega),
            'delta_volume': float(delta_volume),
            'delta_surface_area': float(delta_surface_area),
            'shear_magnitude': shearing['shear_magnitude'] if shearing else np.nan,
            'shear_vector': shearing['shear_vector'] if shearing else [np.nan, np.nan, np.nan],
            'shear_direction': shearing['shear_direction'] if shearing else [np.nan, np.nan, np.nan],
        }
        
        if full:
            result.update({
                'square_angles_detail': [
                    {'angle': float(a), 'deviation': float(abs(a - 90.0))}
                    for a in actual_props['square_angles']
                ],
                'equilateral_angles_detail': [
                    {'angle': float(a), 'deviation': float(abs(a - 60.0))}
                    for a in actual_props['equilateral_angles']
                ],
                'edge_lengths_detail': [
                    {'length': float(e), 'ideal': float(ideal_props['edge_length']), 'deviation': float(abs(e - ideal_props['edge_length']))}
                    for e in actual_props['edge_lengths']
                ],
                'cavity_type': cavity.cavity_type,
            })
        
        return result
    
    elif mode == 'absolute':
        # Absolute mode: raw measurements without comparison
        result = {
            'square_angles': {
                'mean': float(np.mean(actual_props['square_angles'])) if actual_props['square_angles'] else np.nan,
                'std': float(np.std(actual_props['square_angles'])) if actual_props['square_angles'] else np.nan,
                'values': [float(a) for a in actual_props['square_angles']],
            },
            'equilateral_angles': {
                'mean': float(np.mean(actual_props['equilateral_angles'])) if actual_props['equilateral_angles'] else np.nan,
                'std': float(np.std(actual_props['equilateral_angles'])) if actual_props['equilateral_angles'] else np.nan,
                'values': [float(a) for a in actual_props['equilateral_angles']],
            },
            'nu': float(np.linalg.norm(cavity.center_position - actual_props['x_center'])),
            'edge_lengths': {
                'mean': float(np.mean(actual_props['edge_lengths'])) if actual_props['edge_lengths'] else np.nan,
                'std': float(np.std(actual_props['edge_lengths'])) if actual_props['edge_lengths'] else np.nan,
                'values': [float(e) for e in actual_props['edge_lengths']],
                'ideal': float(ideal_props['edge_length']),
            },
            'volume': float(actual_props['volume']),
            'surface_area': float(actual_props['surface_area']),
            'ideal_volume': float(ideal_props['volume']),
            'ideal_surface_area': float(ideal_props['surface_area']),
            'cavity_type': cavity.cavity_type,
        }

        return result
    
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'delta' or 'absolute'.")



def calculate_antiprism_deformation(
    cavity: 'Cavity',
    bx_distance: float,
    mode: str = 'delta',
    full: bool = False
) -> Dict[str, Any]:
    """Calculate deformation metrics for a square antiprism using graph-based analysis.

    For DJ/RP spacers with 8 X-atoms forming a square antiprism.
    Uses graph topology for face identification and geometry for measurements.

    **IMPORTANT**: For DJ spacers, half-cages can share atoms. This function
    analyzes each half-cage separately using cage node filtering to avoid
    mixing geometry from different cages.

    Parameters
    ----------
    cavity : Cavity
        Cavity object with subgraph containing X-atom positions
    bx_distance : float
        Ideal B-X bond distance in Angstroms
    mode : str, optional
        Output mode: 'delta' (default) or 'absolute'
    full : bool, optional
        If True, return detailed per-face/edge data

    Returns
    -------
    dict
        Deformation metrics with mode='delta' or mode='absolute'
        For DJ spacers, returns per-cage results in 'cages' list plus averaged metrics.
    """
    # Get ideal properties analytically
    ideal_props = calculate_ideal_antiprism_properties(bx_distance)

    # Check if this is a DJ spacer (has multiple cage nodes)
    # Note: cage nodes can be 'cage' (complete) or 'half_cage' (incomplete/antiprism)
    cage_nodes = [
        node for node in cavity.subgraph.nodes()
        if cavity.subgraph.nodes[node].get('node_type') in ('cage', 'half_cage')
    ]

    # For DJ spacers, analyze each half-cage separately
    if cavity.cavity_type == 'spacer_dj' and len(cage_nodes) >= 2:
        # Analyze each cage separately
        cage_results = []
        cage_centroids = []  # Store centroids for distance calculation
        for cage_node in cage_nodes:
            actual_props = measure_cavity_properties(cavity, cage_node_id=cage_node)
            cage_result = _compute_antiprism_metrics(
                actual_props, ideal_props, cavity, mode, full, cage_node_id=cage_node
            )
            cage_result['cage_node_id'] = cage_node
            cage_results.append(cage_result)
            # Store the X-atom centroid for this half-cage
            cage_centroids.append(actual_props['x_center'])

        # Calculate distance between half-cage centroids
        centroid_vector = cage_centroids[1] - cage_centroids[0]
        centroid_distance = np.linalg.norm(centroid_vector)
        
        # Calculate shear magnitude: projection of centroid distance onto XY plane
        centroid_vector_xy = centroid_vector.copy()
        centroid_vector_xy[2] = 0.0  # Project onto XY plane (set z=0)
        shear_magnitude = np.linalg.norm(centroid_vector_xy)
        
        # Calculate shear angle relative to cell XY basis vectors
        # Try to get cell from graph attributes, otherwise use identity (orthogonal cell)
        cell = None
        if hasattr(cavity.subgraph, 'graph') and 'cell' in cavity.subgraph.graph:
            cell = np.array(cavity.subgraph.graph['cell'])
        elif hasattr(cavity, 'subgraph') and hasattr(cavity.subgraph, 'graph'):
            # Try parent graph if subgraph has reference
            parent_graph = getattr(cavity.subgraph, '_parent_graph', None)
            if parent_graph is not None and hasattr(parent_graph, 'graph') and 'cell' in parent_graph.graph:
                cell = np.array(parent_graph.graph['cell'])
        
        if cell is not None and cell.shape == (3, 3):
            # Get XY basis vectors (first two cell vectors projected to XY plane)
            cell_x = cell[0, :2]  # X vector in XY plane
            cell_y = cell[1, :2]  # Y vector in XY plane
            
            # Project centroid_vector_xy onto cell XY basis
            # Convert to fractional coordinates in XY plane
            # Solve: centroid_vector_xy[:2] = fractional[0] * cell_x + fractional[1] * cell_y
            xy_basis = np.array([cell_x, cell_y]).T  # Shape (2, 2): columns are cell_x and cell_y
            if abs(np.linalg.det(xy_basis)) > 1e-10:  # Check if basis is valid
                # Solve for fractional coordinates
                fractional = np.linalg.solve(xy_basis, centroid_vector_xy[:2])
                # Calculate angle from X-axis in cell coordinates
                shear_angle = np.degrees(np.arctan2(fractional[1], fractional[0]))
            else:
                # Fallback: use global XY coordinates
                shear_angle = np.degrees(np.arctan2(centroid_vector_xy[1], centroid_vector_xy[0]))
        else:
            # No cell available, use global XY coordinates
            if shear_magnitude > 1e-10:
                shear_angle = np.degrees(np.arctan2(centroid_vector_xy[1], centroid_vector_xy[0]))
            else:
                shear_angle = 0.0  # Undefined for zero magnitude

        # Compute averaged metrics across both cages
        if mode == 'delta':
            # Helper to compute mean with proper NaN handling
            def safe_mean(values):
                valid_values = [v for v in values if not np.isnan(v)]
                return np.mean(valid_values) if valid_values else np.nan
            
            # Compute combined kappa (mean of equilateral and isosceles) for compatibility
            kappa_eq_avg = safe_mean([c['kappa_equilateral'] for c in cage_results])
            kappa_iso_avg = safe_mean([c['kappa_isosceles'] for c in cage_results])
            kappa_combined = np.nan
            if not np.isnan(kappa_eq_avg) and not np.isnan(kappa_iso_avg):
                kappa_combined = (kappa_eq_avg + kappa_iso_avg) / 2.0
            elif not np.isnan(kappa_eq_avg):
                kappa_combined = kappa_eq_avg
            elif not np.isnan(kappa_iso_avg):
                kappa_combined = kappa_iso_avg
            
            avg_result = {
                'eta': safe_mean([c['eta'] for c in cage_results]),
                'kappa': float(kappa_combined),  # Combined kappa for compatibility
                'kappa_equilateral': float(kappa_eq_avg),
                'kappa_isosceles': float(kappa_iso_avg),
                'nu': safe_mean([c['nu'] for c in cage_results]),
                'omega': safe_mean([c['omega'] for c in cage_results]),
                'delta_volume': safe_mean([c['delta_volume'] for c in cage_results]),
                'volume': safe_mean([c.get('volume', np.nan) for c in cage_results]),  # Actual volume
                'delta_surface_area': safe_mean([c['delta_surface_area'] for c in cage_results]),
                'terminal_count': sum(c['terminal_count'] for c in cage_results),
                'centroid_distance': float(centroid_distance),  # Distance between half-cage centroids
                'shear_magnitude': float(shear_magnitude),  # XY-plane projection of centroid distance
                'shear_angle': float(shear_angle),  # Angle of shear vector relative to cell X-axis (degrees)
                'cages': cage_results  # Include per-cage details
            }
            return avg_result
        else:  # mode == 'absolute'
            # Aggregate absolute mode results across cages
            def safe_mean(values):
                valid_values = [v for v in values if not np.isnan(v)]
                return np.mean(valid_values) if valid_values else np.nan
            
            def safe_std(values):
                valid_values = [v for v in values if not np.isnan(v)]
                return np.std(valid_values) if len(valid_values) > 1 else (0.0 if valid_values else np.nan)
            
            # Collect all values from cages
            all_square_angles = []
            all_equilateral_angles = []
            all_isosceles_angles = []
            all_edge_lengths = []
            all_volumes = []
            all_nu = []
            
            for cage_result in cage_results:
                if 'square_angles' in cage_result and 'values' in cage_result['square_angles']:
                    all_square_angles.extend(cage_result['square_angles']['values'])
                if 'equilateral_angles' in cage_result and 'values' in cage_result['equilateral_angles']:
                    all_equilateral_angles.extend(cage_result['equilateral_angles']['values'])
                if 'isosceles_angles' in cage_result and 'values' in cage_result['isosceles_angles']:
                    all_isosceles_angles.extend(cage_result['isosceles_angles']['values'])
                if 'edge_lengths' in cage_result and 'values' in cage_result['edge_lengths']:
                    all_edge_lengths.extend(cage_result['edge_lengths']['values'])
                if 'volume' in cage_result:
                    all_volumes.append(cage_result['volume'])
                if 'nu' in cage_result:
                    all_nu.append(cage_result['nu'])
            
            avg_result = {
                'square_angles': {
                    'mean': safe_mean(all_square_angles),
                    'std': safe_std(all_square_angles),
                    'values': all_square_angles,
                },
                'equilateral_angles': {
                    'mean': safe_mean(all_equilateral_angles),
                    'std': safe_std(all_equilateral_angles),
                    'values': all_equilateral_angles,
                },
                'isosceles_angles': {
                    'mean': safe_mean(all_isosceles_angles),
                    'std': safe_std(all_isosceles_angles),
                    'values': all_isosceles_angles,
                },
                'nu': safe_mean(all_nu),
                'edge_lengths': {
                    'mean': safe_mean(all_edge_lengths),
                    'std': safe_std(all_edge_lengths),
                    'values': all_edge_lengths,
                    'ideal': ideal_props['edge_length'] if cage_results else np.nan,
                },
                'volume': safe_mean(all_volumes),
                'ideal_volume': ideal_props['volume'],
                'centroid_distance': float(centroid_distance),  # Distance between half-cage centroids
                'shear_magnitude': float(shear_magnitude),  # XY-plane projection of centroid distance
                'shear_angle': float(shear_angle),  # Angle of shear vector relative to cell X-axis (degrees)
                'cages': cage_results,
                'cavity_type': cavity.cavity_type
            }
            return avg_result

    # Single cage analysis (RP spacer or old DJ without cage nodes)
    actual_props = measure_cavity_properties(cavity)
    return _compute_antiprism_metrics(actual_props, ideal_props, cavity, mode, full)


def _compute_antiprism_metrics(
    actual_props: Dict[str, Any],
    ideal_props: Dict[str, Any],
    cavity: 'Cavity',
    mode: str,
    full: bool,
    cage_node_id: str = None
) -> Dict[str, Any]:
    """Helper function to compute antiprism deformation metrics from properties.

    Parameters
    ----------
    actual_props : dict
        Measured cavity properties
    ideal_props : dict
        Ideal antiprism properties
    cavity : Cavity
        Cavity object
    mode : str
        'delta' or 'absolute'
    full : bool
        Include detailed data
    cage_node_id : str, optional
        Cage node ID for identification

    Returns
    -------
    dict
        Deformation metrics
    """

    if mode == 'delta':
        # Delta mode: deviations from ideal
        import sys

        # Prepare cage identifier for warnings
        cage_str = f" (cage '{cage_node_id}')" if cage_node_id else ""

        # Calculate Eta: 90° - mean(square angle deviation)
        if not actual_props['square_angles']:
            print(
                f"WARNING: No square faces found in cavity {cavity.id}{cage_str}. "
                f"Expected 2 square faces for antiprism ({cavity.cavity_type}). "
                f"Returning NaN for eta.",
                file=sys.stderr
            )
            eta = np.nan
        else:
            eta_deviations = [abs(a - 90.0) for a in actual_props['square_angles']]
            eta = 90.0 - np.mean(eta_deviations)

        # Calculate Kappa (equilateral triangles): 60° - mean deviation
        if not actual_props['equilateral_angles']:
            print(
                f"WARNING: No equilateral triangle faces found in cavity {cavity.id}{cage_str}. "
                f"Expected 4 equilateral triangular faces for antiprism ({cavity.cavity_type}). "
                f"Returning NaN for kappa_equilateral.",
                file=sys.stderr
            )
            kappa_equilateral = np.nan
        else:
            kappa_equi_dev = [abs(a - 60.0) for a in actual_props['equilateral_angles']]
            kappa_equilateral = 60.0 - np.mean(kappa_equi_dev)

        # Calculate Kappa (isosceles triangles): deviation from 90-45-45 pattern
        # For right-isosceles, expect one 90° angle and two 45° angles
        # FIXED: Return "ideal - deviation" to match kappa_equilateral pattern
        if not actual_props['isosceles_angles']:
            print(
                f"WARNING: No isosceles triangle faces found in cavity {cavity.id}{cage_str}. "
                f"Expected 4 isosceles triangular faces for antiprism ({cavity.cavity_type}). "
                f"Returning NaN for kappa_isosceles.",
                file=sys.stderr
            )
            kappa_isosceles = np.nan
        else:
            isosceles_deviation = _calculate_isosceles_deviation(
                actual_props['isosceles_angles'],
                ideal_angles=[90.0, 45.0, 45.0]
            )
            # Compute mean ideal angle for isosceles: (90 + 45 + 45) / 3 = 60°
            kappa_isosceles = 60.0 - isosceles_deviation

        # Nu: displacement of spacer molecule from X-cage center
        nu = np.linalg.norm(cavity.center_position - actual_props['x_center'])

        # Omega: mean X-X edge length deviation
        if not actual_props['edge_lengths']:
            print(
                f"WARNING: No X-X edge lengths calculated for cavity {cavity.id}{cage_str}. "
                f"Returning NaN for omega.",
                file=sys.stderr
            )
            omega = np.nan
        else:
            omega_deviations = [abs(e - ideal_props['edge_length']) for e in actual_props['edge_lengths']]
            omega = np.mean(omega_deviations)

        # Delta Volume: volume deviation
        delta_volume = actual_props['volume'] - ideal_props['volume']

        # Delta Surface Area: surface area deviation
        delta_surface_area = actual_props['surface_area'] - ideal_props['surface_area']

        # Terminal atom validation
        # For per-cage analysis (DJ spacers), expect 4 terminal atoms per half-cage
        if cage_node_id is not None:
            expected_terminal = 4  # Half-cage
        else:
            expected_terminal = 4 if cavity.cavity_type == 'spacer_rp' else 8
        terminal_deviation = abs(actual_props['terminal_count'] - expected_terminal)

        # Calculate shearing deformation (displacement between octagonal ring centroids)
        shearing = calculate_shearing_deformation(cavity, cage_node_id=cage_node_id)

        # Store actual volume for compatibility
        actual_volume = actual_props['volume']

        # Compute combined kappa (mean of equilateral and isosceles) for compatibility
        kappa_combined = np.nan
        if not np.isnan(kappa_equilateral) and not np.isnan(kappa_isosceles):
            kappa_combined = (kappa_equilateral + kappa_isosceles) / 2.0
        elif not np.isnan(kappa_equilateral):
            kappa_combined = kappa_equilateral
        elif not np.isnan(kappa_isosceles):
            kappa_combined = kappa_isosceles
        
        result = {
            'eta': float(eta),
            'kappa': float(kappa_combined),  # Combined kappa for compatibility
            'kappa_equilateral': float(kappa_equilateral),
            'kappa_isosceles': float(kappa_isosceles),
            'nu': float(nu),
            'omega': float(omega),
            'delta_volume': float(delta_volume),
            'volume': float(actual_volume),  # Actual volume for compatibility
            'delta_surface_area': float(delta_surface_area),
            'terminal_count': int(actual_props['terminal_count']),
            'terminal_deviation': int(terminal_deviation),
            'equilateral_count': int(actual_props['equilateral_count']),
            'isosceles_count': int(actual_props['isosceles_count']),
            'square_count': int(actual_props['square_count']),
            'shear_magnitude': shearing['shear_magnitude'] if shearing else np.nan,
            'shear_vector': shearing['shear_vector'] if shearing else [np.nan, np.nan, np.nan],
            'shear_direction': shearing['shear_direction'] if shearing else [np.nan, np.nan, np.nan],
        }

        if full:
            result.update({
                'square_angles_detail': [
                    {'angle': float(a), 'deviation': float(abs(a - 90.0))}
                    for a in actual_props['square_angles']
                ],
                'equilateral_angles_detail': [
                    {'angle': float(a), 'deviation': float(abs(a - 60.0))}
                    for a in actual_props['equilateral_angles']
                ],
                'isosceles_angles_detail': [
                    {'angle': float(a)}
                    for a in actual_props['isosceles_angles']
                ],
                'edge_lengths_detail': [
                    {'length': float(e), 'ideal': float(ideal_props['edge_length']), 'deviation': float(abs(e - ideal_props['edge_length']))}
                    for e in actual_props['edge_lengths']
                ],
                'cavity_type': cavity.cavity_type,
            })

        return result

    elif mode == 'absolute':
        # Absolute mode: raw measurements without comparison
        result = {
            'square_angles': {
                'mean': float(np.mean(actual_props['square_angles'])) if actual_props['square_angles'] else np.nan,
                'std': float(np.std(actual_props['square_angles'])) if actual_props['square_angles'] else np.nan,
                'values': [float(a) for a in actual_props['square_angles']],
            },
            'equilateral_angles': {
                'mean': float(np.mean(actual_props['equilateral_angles'])) if actual_props['equilateral_angles'] else np.nan,
                'std': float(np.std(actual_props['equilateral_angles'])) if actual_props['equilateral_angles'] else np.nan,
                'values': [float(a) for a in actual_props['equilateral_angles']],
            },
            'isosceles_angles': {
                'mean': float(np.mean(actual_props['isosceles_angles'])) if actual_props['isosceles_angles'] else np.nan,
                'std': float(np.std(actual_props['isosceles_angles'])) if actual_props['isosceles_angles'] else np.nan,
                'values': [float(a) for a in actual_props['isosceles_angles']],
            },
            'nu': float(np.linalg.norm(cavity.center_position - actual_props['x_center'])),
            'edge_lengths': {
                'mean': float(np.mean(actual_props['edge_lengths'])) if actual_props['edge_lengths'] else np.nan,
                'std': float(np.std(actual_props['edge_lengths'])) if actual_props['edge_lengths'] else np.nan,
                'values': [float(e) for e in actual_props['edge_lengths']],
                'ideal': float(ideal_props['edge_length']),
            },
            'volume': float(actual_props['volume']),
            'surface_area': float(actual_props['surface_area']),
            'ideal_volume': float(ideal_props['volume']),
            'ideal_surface_area': float(ideal_props['surface_area']),
            'terminal_count': int(actual_props['terminal_count']),
            'equilateral_count': int(actual_props['equilateral_count']),
            'isosceles_count': int(actual_props['isosceles_count']),
            'square_count': int(actual_props['square_count']),
            'cavity_type': cavity.cavity_type,
        }

        return result

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'delta' or 'absolute'.")



def calculate_cavity_deformation(
    cavity: 'Cavity',
    bx_distance: float,
    mode: str = 'delta',
    full: bool = False
) -> Dict[str, Any]:
    """Calculate deformation metrics for any cavity type.
    
    Automatically dispatches to appropriate function based on cavity type.
    Uses analytical formulas for ideal geometry properties.
    
    Parameters
    ----------
    cavity : Cavity
        Cavity object
    bx_distance : float
        Ideal B-X bond distance in Angstroms
    mode : str, optional
        Output mode: 'delta' (default) or 'absolute'
    full : bool, optional
        If True, return detailed data
        
    Returns
    -------
    dict
        Deformation metrics
    """
    if cavity.cavity_type == 'a_site':
        return calculate_cuboctahedron_deformation(cavity, bx_distance, mode=mode, full=full)
    elif cavity.cavity_type in ('spacer_dj', 'spacer_rp'):
        return calculate_antiprism_deformation(cavity, bx_distance, mode=mode, full=full)
    else:
        raise ValueError(f"Unknown cavity type: {cavity.cavity_type}")


def _normalize_xy_projection(
    nh3_xy: np.ndarray,
    b_xy: np.ndarray,
    x_terminal_xy: np.ndarray,
    x_equatorial_xy: np.ndarray,
    octagon_centroid_xy: np.ndarray,
    atoms_connectivity: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, Tuple[int, int], int]:
    """Normalize XY projection by centering octagon and aligning shortest B-B pair midpoint to X-axis.

    Uses atoms_connectivity dict to find B-B pairs connected by X_equatorial atoms.
    Each X_equatorial has 'connected_to' containing exactly 2 B atom IDs.

    Normalization steps:
    1. Calculate octagon centroid (4B + 4X_eq) and translate all points so centroid is at origin
    2. Extract B-B pairs from atoms_connectivity (via X_equatorial 'connected_to')
    3. Calculate distance for each B-B pair and find the shortest one
    4. Calculate midpoint of the shortest B-B pair
    5. Rotate all points around origin so midpoint has Y=0 (lies on X-axis)

    Parameters
    ----------
    nh3_xy, b_xy, x_terminal_xy, x_equatorial_xy, octagon_centroid_xy : np.ndarray
        XY coordinates to normalize
    atoms_connectivity : Dict[str, Any]
        Dictionary with keys like 'B0', 'B1', 'X_equatorial0', 'X_terminal0', etc.
        Each X_equatorial entry has 'connected_to': ['B0', 'B2'] listing connected B atoms.
        Each entry also has 'role': 'terminal' or 'equatorial' for X atoms.

    Returns
    -------
    tuple
        (nh3_xy_norm, b_xy_norm, x_terminal_xy_norm, x_equatorial_xy_norm,
         octagon_centroid_xy_norm, rotation_angle_deg, translation_vector,
         pair_indices, bridging_x_eq_idx)
        where pair_indices is the shortest B-B pair indices (e.g., (0, 1))
        and bridging_x_eq_idx is the index of the X_equatorial connecting them
    """
    # Step 1: Calculate octagon centroid (4B + 4X_eq) and translate all points to origin
    all_points = np.vstack([b_xy, x_equatorial_xy])
    centroid = np.mean(all_points, axis=0)

    # Translate all points so centroid is at origin
    nh3_xy_trans = nh3_xy - centroid if not np.any(np.isnan(nh3_xy)) else nh3_xy
    b_xy_trans = b_xy - centroid
    x_terminal_xy_trans = x_terminal_xy - centroid
    x_equatorial_xy_trans = x_equatorial_xy - centroid
    octagon_centroid_xy_trans = octagon_centroid_xy - centroid

    # Step 2: Extract B-B pairs from atoms_connectivity via X_equatorial 'connected_to'
    # Each X_equatorial connects exactly 2 B atoms - these form valid B-B pairs
    # Topology: B-X-B-X-B-X-B-X octagon with 4 B and 4 X_equatorial connected by bonded_to edges
    b_pairs = []  # List of ((b_idx_1, b_idx_2), x_eq_idx)

    for key, data in atoms_connectivity.items():
        if not key.startswith('X_equatorial'):
            continue

        # Get X_equatorial array index from key (e.g., 'X_equatorial0' -> 0)
        x_eq_idx = int(key.replace('X_equatorial', ''))

        # Get connected B atoms from 'connected_to'
        connected_to = data.get('connected_to', [])

        # Extract B indices from connected_to (e.g., ['B0', 'B2'] -> [0, 2])
        b_indices = []
        for b_id in connected_to:
            if b_id.startswith('B'):
                try:
                    b_idx = int(b_id[1:])
                    b_indices.append(b_idx)
                except ValueError:
                    continue

        # Valid X_equatorial connects exactly 2 B atoms
        if len(b_indices) == 2:
            b_pair = tuple(sorted(b_indices))
            b_pairs.append((b_pair, x_eq_idx))

    if not b_pairs:
        raise RuntimeError(
            "No valid B-B pairs found via X_equatorial connectivity. "
            "Expected octagonal topology B-X-B-X-B-X-B-X with each X_equatorial "
            "connecting exactly 2 B atoms. Graph may be malformed."
        )

    # Step 3: Calculate distance for each B-B pair and find the shortest one
    pair_distances = []
    for (b_i, b_j), x_eq_idx in b_pairs:
        dist = np.linalg.norm(b_xy_trans[b_i] - b_xy_trans[b_j])
        pair_distances.append(dist)

    shortest_idx = np.argmin(pair_distances)
    shortest_pair, bridging_x_eq_idx = b_pairs[shortest_idx]

    # Step 4: Calculate midpoint of the shortest B-B pair
    midpoint = (b_xy_trans[shortest_pair[0]] + b_xy_trans[shortest_pair[1]]) / 2.0

    # Step 5: Calculate rotation angle to make midpoint have Y=0 (lies on X-axis)
    # Rotate around origin so midpoint aligns with X-axis
    theta = -np.arctan2(midpoint[1], midpoint[0])  # Negative to align with X-axis

    # Rotation matrix (rotate by theta around origin)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])

    # Apply rotation to all points
    nh3_xy_norm = rotation_matrix @ nh3_xy_trans if not np.any(np.isnan(nh3_xy_trans)) else nh3_xy_trans
    b_xy_norm = (rotation_matrix @ b_xy_trans.T).T
    x_terminal_xy_norm = (rotation_matrix @ x_terminal_xy_trans.T).T
    x_equatorial_xy_norm = (rotation_matrix @ x_equatorial_xy_trans.T).T
    octagon_centroid_xy_norm = rotation_matrix @ octagon_centroid_xy_trans

    rotation_angle_deg = float(np.degrees(theta))
    translation_vector = centroid  # The centroid is the translation vector

    return (nh3_xy_norm, b_xy_norm, x_terminal_xy_norm, x_equatorial_xy_norm,
            octagon_centroid_xy_norm, rotation_angle_deg, translation_vector,
            shortest_pair, bridging_x_eq_idx)


def _build_octagon_edges_from_connectivity(
    atoms_connectivity: Dict[str, Any],
    b_xy: np.ndarray,
    x_equatorial_xy: np.ndarray
) -> List[Dict[str, Any]]:
    """Build edge connectivity for octagon using atoms_connectivity dict lookup.

    Uses direct dict lookup to extract B-X_equatorial edges from connectivity.
    Expected topology: B-X-B-X-B-X-B-X octagon with each X_equatorial connecting 2 B atoms.

    Parameters
    ----------
    atoms_connectivity : dict
        Dictionary with keys like 'B0', 'B1', 'X_equatorial0', 'X_terminal0', etc.
        Each entry has 'connected_to': list of connected atom IDs.
        X atoms have 'role': 'terminal' or 'equatorial'.
    b_xy : np.ndarray
        Shape (4, 2) - B atom XY positions (kept for API compatibility)
    x_equatorial_xy : np.ndarray
        Shape (4, 2) - X_equatorial atom XY positions (kept for API compatibility)

    Returns
    -------
    list of dict
        Edge list with 'from' and 'to' keys containing {'type': str, 'index': int}
    """
    edges = []

    # Build edges from X_equatorial 'connected_to' (each connects 2 B atoms)
    for key, data in atoms_connectivity.items():
        if not key.startswith('X_equatorial'):
            continue

        x_eq_idx = int(key.replace('X_equatorial', ''))
        connected_to = data.get('connected_to', [])

        # Extract B indices from connected_to
        b_indices = []
        for b_id in connected_to:
            if b_id.startswith('B'):
                try:
                    b_idx = int(b_id[1:])
                    b_indices.append(b_idx)
                except ValueError:
                    continue

        # Each X_equatorial should connect exactly 2 B atoms
        if len(b_indices) == 2:
            b_i, b_j = b_indices
            # Add bidirectional edges: B[i] <-> X_eq <-> B[j]
            edges.append({
                'from': {'type': 'B', 'index': b_i},
                'to': {'type': 'X_eq', 'index': x_eq_idx}
            })
            edges.append({
                'from': {'type': 'X_eq', 'index': x_eq_idx},
                'to': {'type': 'B', 'index': b_j}
            })
            edges.append({
                'from': {'type': 'B', 'index': b_j},
                'to': {'type': 'X_eq', 'index': x_eq_idx}
            })
            edges.append({
                'from': {'type': 'X_eq', 'index': x_eq_idx},
                'to': {'type': 'B', 'index': b_i}
            })

    return edges




def project_halfcage_to_xy(cavity: 'Cavity', cage_node_id: str = None, normalize: bool = False) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Project half-cage octagon (4B + 4X) and NH3 centroid onto XY plane.
    
    For spacer cavities, extracts the half-cage geometry (4B atoms + 4 terminal X + 4 equatorial X)
    and NH3 position, projects them to the XY plane for 2D analysis.
    
    Parameters
    ----------
    cavity : Cavity
        Cavity object (must be spacer_rp or spacer_dj type)
    cage_node_id : str, optional
        If provided, only project this specific half-cage (for DJ spacers)
        If None and cavity is DJ, returns list of projections for all half-cages
    normalize : bool, optional
        If True, normalize the projection by centering the octagon at origin and
        aligning the shortest B-B pair midpoint to the X-axis. Default: False
        
    Returns
    -------
    dict or list of dict
        For RP spacer: Single dict with keys:
            - nh3_xy: (2,) NH3 position in XY
            - nh3_3d: (3,) NH3 position in 3D
            - b_xy: (4, 2) B atoms in XY
            - b_3d: (4, 3) B atoms in 3D
            - x_terminal_xy: (4, 2) Terminal X atoms in XY
            - x_terminal_3d: (4, 3) Terminal X atoms in 3D
            - x_equatorial_xy: (4, 2) Equatorial X atoms in XY
            - x_equatorial_3d: (4, 3) Equatorial X atoms in 3D
            - octagon_centroid_xy: (2,) Centroid of 4B + 4 terminal X
            - displacement_xy: (2,) NH3 - octagon_centroid
            - displacement_magnitude: float Distance from NH3 to centroid
            - cage_node_id: str Cage identifier
            
            If normalize=True, additional keys:
            - normalized: True
            - rotation_angle: float (degrees)
            - translation_vector: [float, float]
            - reference_pair_indices: [int, int]
            - edges: list of edge dicts
        
        For DJ spacer: List of 2 dicts (one per half-cage) with same structure
    """
    import sys
    
    # Validate cavity type
    if cavity.cavity_type not in ('spacer_rp', 'spacer_dj'):
        raise ValueError(f"project_halfcage_to_xy only supports spacer cavities. Got: {cavity.cavity_type}")
    
    # Helper function to extract NH3 position for a given cage
    def get_nh3_position_for_cage(cavity: 'Cavity', cage_node_id: str = None) -> Optional[np.ndarray]:
        """Extract NH3 (N atom) position from anchor nodes in subgraph."""
        anchor_nodes = []
        
        # Find all anchor nodes
        for node in cavity.subgraph.nodes():
            if cavity.subgraph.nodes[node].get('node_type') == 'anchor':
                anchor_nodes.append(node)
        
        if not anchor_nodes:
            print(
                f"WARNING: No anchor nodes found in cavity {cavity.id}",
                file=sys.stderr
            )
            return None
        
        # If cage_node_id specified, filter to anchors connected to that cage
        if cage_node_id is not None:
            cage_anchors = []
            for anchor in anchor_nodes:
                # Check if this anchor is connected to the cage
                for neighbor in cavity.subgraph.neighbors(anchor):
                    if neighbor == cage_node_id:
                        cage_anchors.append(anchor)
                        break
            if cage_anchors:
                anchor_nodes = cage_anchors
        
        # Get N atom position from first anchor
        if anchor_nodes:
            anchor_node = anchor_nodes[0]
            # Find N atom connected to this anchor via 'contains' edge
            for neighbor in cavity.subgraph.neighbors(anchor_node):
                neighbor_data = cavity.subgraph.nodes[neighbor]
                if neighbor_data.get('symbol') == 'N':
                    pbc_pos = neighbor_data.get('pbc_position')
                    if pbc_pos is not None:
                        return np.array(pbc_pos)
        
        return None
    
    # Helper function to project and build result for a single cage
    def project_single_cage(cavity: 'Cavity', cage_node_id: str = None, normalize: bool = False) -> Dict[str, Any]:
        """Project a single half-cage to XY."""
        # Get B atoms for this cage
        b_atoms = _get_b_atoms_for_cage(cavity, cage_node_id=cage_node_id)
        
        if len(b_atoms) < 4:
            raise RuntimeError(
                f"Failed to get 4 B atoms for cavity {cavity.id}"
                + (f" (cage '{cage_node_id}')" if cage_node_id else "")
            )
        
        # Extract B atom positions and build connectivity map
        # IMPORTANT: Order X atoms by their connection to B atoms so indices match arrays
        b_nodes = []
        b_positions_3d = []
        b_positions_xy = []
        
        # Map to store X atoms by node_id (before ordering)
        x_atom_data = {}  # X node_id -> {'role': 'terminal'|'equatorial', 'b_connected': [b_indices], 'xyz', 'xy'}
        
        # First pass: collect all B and X atoms with their connectivity
        for b_idx, b_node in enumerate(b_atoms[:4]):
            node_data = cavity.subgraph.nodes[b_node]
            pbc_pos = node_data.get('pbc_position')
            if pbc_pos is None:
                continue
            
            b_nodes.append(b_node)
            b_positions_3d.append(np.array(pbc_pos))
            b_positions_xy.append(np.array(pbc_pos[:2]))
            
            # Find X atoms bonded to this B
            for neighbor in cavity.subgraph.neighbors(b_node):
                neighbor_data = cavity.subgraph.nodes[neighbor]
                if (neighbor_data.get('node_type') == 'atom' and
                    neighbor_data.get('is_X', False)):
                    edge_data = cavity.subgraph.get_edge_data(b_node, neighbor)
                    if edge_data and edge_data.get('edge_type') == 'bonded_to':
                        if neighbor not in x_atom_data:
                            # Check both node and edge properties for terminal/equatorial status
                            # (matches logic in cavity_tracing._get_antiprism_x_atoms)
                            is_terminal_node = neighbor_data.get('is_terminal', False)
                            is_terminal_edge = edge_data.get('geometry') == 'terminal'
                            is_terminal = is_terminal_node or is_terminal_edge

                            is_equatorial_node = neighbor_data.get('is_equatorial', False)
                            is_equatorial_edge = edge_data.get('geometry') == 'equatorial'
                            is_equatorial = is_equatorial_node or is_equatorial_edge

                            role = 'terminal' if is_terminal else ('equatorial' if is_equatorial else 'unknown')

                            x_pbc_pos = neighbor_data.get('pbc_position')
                            if x_pbc_pos is None:
                                continue

                            x_atom_data[neighbor] = {
                                'role': role,
                                'b_connected': [],
                                'xyz': np.array(x_pbc_pos),
                                'xy': np.array(x_pbc_pos[:2])
                            }

                        # Record bidirectional connectivity
                        x_atom_data[neighbor]['b_connected'].append(b_idx)
        
        if len(b_positions_3d) != 4:
            raise RuntimeError(
                f"Failed to get 4 B atom positions for cavity {cavity.id}"
                + (f" (cage '{cage_node_id}')" if cage_node_id else "")
            )
        
        # Build arrays in order: B[0..3], then X_terminal[0..3] (each connected to B[i]), then X_equatorial[0..3]
        b_3d = np.array(b_positions_3d)
        b_xy = np.array(b_positions_xy)
        
        # Collect ALL unique X atoms first (not per-B iteration)
        # Terminal X: each connects to exactly 1 B
        # Equatorial X: each connects to exactly 2 B atoms
        x_terminal_3d = []
        x_terminal_xy = []
        x_equatorial_3d = []
        x_equatorial_xy = []
        x_terminal_node_ids = []
        x_equatorial_node_ids = []
        
        # Collect all unique X atoms
        for x_node_id, x_data in x_atom_data.items():
            if x_data['role'] == 'terminal':
                x_terminal_3d.append(x_data['xyz'])
                x_terminal_xy.append(x_data['xy'])
                x_terminal_node_ids.append(x_node_id)
            elif x_data['role'] == 'equatorial':
                x_equatorial_3d.append(x_data['xyz'])
                x_equatorial_xy.append(x_data['xy'])
                x_equatorial_node_ids.append(x_node_id)
        
        # Convert to numpy arrays (expect 4 terminal and 4 equatorial for antiprism)
        x_terminal_3d = np.array(x_terminal_3d) if len(x_terminal_3d) == 4 else np.empty((0, 3))
        x_terminal_xy = np.array(x_terminal_xy) if len(x_terminal_xy) == 4 else np.empty((0, 2))
        x_equatorial_3d = np.array(x_equatorial_3d) if len(x_equatorial_3d) == 4 else np.empty((0, 3))
        x_equatorial_xy = np.array(x_equatorial_xy) if len(x_equatorial_xy) == 4 else np.empty((0, 2))
        
        # Get NH3 position
        nh3_3d = get_nh3_position_for_cage(cavity, cage_node_id=cage_node_id)
        if nh3_3d is None:
            print(
                f"WARNING: Could not extract NH3 position for cavity {cavity.id}"
                + (f" (cage '{cage_node_id}')" if cage_node_id else ""),
                file=sys.stderr
            )
            nh3_xy = np.array([np.nan, np.nan])
        else:
            nh3_xy = nh3_3d[:2]
        
        # Compute octagon centroid (4B + 4 terminal X)
        octagon_points_xy = np.vstack([b_xy, x_terminal_xy])
        octagon_centroid_xy = np.mean(octagon_points_xy, axis=0)
        
        # Compute displacement
        displacement_xy = nh3_xy - octagon_centroid_xy
        displacement_magnitude = np.linalg.norm(displacement_xy)
        
        # Build connectivity information for JSON output
        # Read connectivity DIRECTLY from the subgraph edges (built by cavity_tracing.py)
        # Do NOT create or infer any new connectivity here
        
        atoms_connectivity = {}
        
        # Create mappings: node_id -> array index
        b_node_to_idx = {b_nodes[i]: i for i in range(len(b_nodes))}
        terminal_x_node_to_idx = {x_terminal_node_ids[i]: i for i in range(len(x_terminal_node_ids))}
        equatorial_x_node_to_idx = {x_equatorial_node_ids[i]: i for i in range(len(x_equatorial_node_ids))}
        
        # Add B atoms with their connectivity (read from subgraph edges)
        for b_idx, b_node in enumerate(b_nodes[:4]):
            connected_x = []
            
            # Get all X neighbors from subgraph
            for neighbor in cavity.subgraph.neighbors(b_node):
                neighbor_data = cavity.subgraph.nodes[neighbor]
                if not neighbor_data.get('is_X', False):
                    continue
                
                # Check if this is a bonded_to edge
                edge_data = cavity.subgraph.get_edge_data(b_node, neighbor)
                if not edge_data or edge_data.get('edge_type') != 'bonded_to':
                    continue
                
                # Determine if terminal or equatorial
                is_terminal = neighbor_data.get('is_terminal', False)
                is_equatorial = neighbor_data.get('is_equatorial', False)
                
                # Map to array index and add to connected list
                if is_terminal and neighbor in terminal_x_node_to_idx:
                    x_idx = terminal_x_node_to_idx[neighbor]
                    connected_x.append(f'X_terminal{x_idx}')
                elif is_equatorial and neighbor in equatorial_x_node_to_idx:
                    x_idx = equatorial_x_node_to_idx[neighbor]
                    connected_x.append(f'X_equatorial{x_idx}')
            
            atoms_connectivity[f'B{b_idx}'] = {
                'xyz': b_3d[b_idx].tolist(),
                'xy': b_xy[b_idx].tolist(),
                'connected_to': connected_x
            }
        
        # Add X_terminal atoms with their connectivity (read from subgraph edges)
        for x_idx, x_node_id in enumerate(x_terminal_node_ids):
            connected_b = []
            
            # Get all B neighbors from subgraph
            for neighbor in cavity.subgraph.neighbors(x_node_id):
                neighbor_data = cavity.subgraph.nodes[neighbor]
                if not neighbor_data.get('is_B', False):
                    continue
                
                # Check if this is a bonded_to edge
                edge_data = cavity.subgraph.get_edge_data(x_node_id, neighbor)
                if not edge_data or edge_data.get('edge_type') != 'bonded_to':
                    continue
                
                # Map to array index
                if neighbor in b_node_to_idx:
                    b_idx = b_node_to_idx[neighbor]
                    connected_b.append(f'B{b_idx}')
            
            atoms_connectivity[f'X_terminal{x_idx}'] = {
                'xyz': x_terminal_3d[x_idx].tolist() if x_idx < len(x_terminal_3d) else [],
                'xy': x_terminal_xy[x_idx].tolist() if x_idx < len(x_terminal_xy) else [],
                'connected_to': connected_b,
                'role': 'terminal'
            }
        
        # Add X_equatorial atoms with their connectivity (read from subgraph edges)
        for x_idx, x_node_id in enumerate(x_equatorial_node_ids):
            connected_b = []
            
            # Get all B neighbors from subgraph
            for neighbor in cavity.subgraph.neighbors(x_node_id):
                neighbor_data = cavity.subgraph.nodes[neighbor]
                if not neighbor_data.get('is_B', False):
                    continue
                
                # Check if this is a bonded_to edge
                edge_data = cavity.subgraph.get_edge_data(x_node_id, neighbor)
                if not edge_data or edge_data.get('edge_type') != 'bonded_to':
                    continue
                
                # Map to array index
                if neighbor in b_node_to_idx:
                    b_idx = b_node_to_idx[neighbor]
                    connected_b.append(f'B{b_idx}')
            
            atoms_connectivity[f'X_equatorial{x_idx}'] = {
                'xyz': x_equatorial_3d[x_idx].tolist() if x_idx < len(x_equatorial_3d) else [],
                'xy': x_equatorial_xy[x_idx].tolist() if x_idx < len(x_equatorial_xy) else [],
                'connected_to': connected_b,
                'role': 'equatorial'
            }
        
        # Build result dictionary
        result = {
            'nh3_xy': nh3_xy,
            'nh3_3d': nh3_3d,
            'b_xy': b_xy,
            'b_3d': b_3d,
            'x_terminal_xy': x_terminal_xy,
            'x_terminal_3d': x_terminal_3d,
            'x_equatorial_xy': x_equatorial_xy,
            'x_equatorial_3d': x_equatorial_3d,
            'octagon_centroid_xy': octagon_centroid_xy,
            'displacement_xy': displacement_xy,
            'displacement_magnitude': float(displacement_magnitude),
            'cage_node_id': str(cage_node_id) if cage_node_id else 'cage_0',
            'atoms_connectivity': atoms_connectivity  # Graph connectivity preserved
        }
        
        # Apply normalization if requested
        if normalize:
            (nh3_xy_norm, b_xy_norm, x_terminal_xy_norm, x_equatorial_xy_norm,
             octagon_centroid_xy_norm, rotation_angle_deg, translation_vector,
             pair_indices, bridging_x_eq_idx) = \
                _normalize_xy_projection(
                    nh3_xy, b_xy, x_terminal_xy, x_equatorial_xy, octagon_centroid_xy,
                    atoms_connectivity
                )

            # Update result with normalized coordinates
            result['nh3_xy'] = nh3_xy_norm
            result['b_xy'] = b_xy_norm
            result['x_terminal_xy'] = x_terminal_xy_norm
            result['x_equatorial_xy'] = x_equatorial_xy_norm
            result['octagon_centroid_xy'] = octagon_centroid_xy_norm

            # Update connectivity coordinates to normalized values
            for b_idx in range(4):
                result['atoms_connectivity'][f'B{b_idx}']['xy'] = b_xy_norm[b_idx].tolist()
            for x_idx in range(len(x_terminal_xy_norm)):
                if x_idx < len(x_terminal_xy_norm):
                    result['atoms_connectivity'][f'X_terminal{x_idx}']['xy'] = x_terminal_xy_norm[x_idx].tolist()
            for x_idx in range(len(x_equatorial_xy_norm)):
                if x_idx < len(x_equatorial_xy_norm):
                    result['atoms_connectivity'][f'X_equatorial{x_idx}']['xy'] = x_equatorial_xy_norm[x_idx].tolist()

            # Recalculate displacement with normalized coordinates
            result['displacement_xy'] = nh3_xy_norm - octagon_centroid_xy_norm
            result['displacement_magnitude'] = float(np.linalg.norm(result['displacement_xy']))

            # Add normalization metadata
            result['normalized'] = True
            result['rotation_angle'] = rotation_angle_deg
            result['translation_vector'] = translation_vector.tolist()
            result['reference_pair_indices'] = list(pair_indices)
            result['reference_x_equatorial_idx'] = bridging_x_eq_idx

            # Build octagon edges from graph connectivity
            result['edges'] = _build_octagon_edges_from_connectivity(
                atoms_connectivity, b_xy_norm, x_equatorial_xy_norm
            )
        else:
            result['normalized'] = False
            # Build edges from graph connectivity even when not normalized
            result['edges'] = _build_octagon_edges_from_connectivity(
                atoms_connectivity, b_xy, x_equatorial_xy
            )
        
        return result
    
    # Process based on cavity type
    if cavity.cavity_type == 'spacer_rp':
        # Single half-cage
        return project_single_cage(cavity, cage_node_id=cage_node_id, normalize=normalize)
    
    elif cavity.cavity_type == 'spacer_dj':
        # Multiple half-cages
        if cage_node_id is not None:
            # Project specific cage
            return project_single_cage(cavity, cage_node_id=cage_node_id, normalize=normalize)
        else:
            # Find all cage nodes and project each; sort by node id for consistent order
            # (e.g. half_cage_0, half_cage_1) so projection list order matches across cavities
            cage_nodes = sorted(
                [
                    node for node in cavity.subgraph.nodes()
                    if cavity.subgraph.nodes[node].get('node_type') in ('cage', 'half_cage')
                ]
            )
            
            if not cage_nodes:
                print(
                    f"WARNING: No cage nodes found in DJ cavity {cavity.id}. Falling back to single projection.",
                    file=sys.stderr
                )
                return project_single_cage(cavity, cage_node_id=None, normalize=normalize)
            
            results = []
            for cage_node in cage_nodes:
                result = project_single_cage(cavity, cage_node_id=cage_node, normalize=normalize)
                results.append(result)
            
            return results


def calculate_weighted_centroid_displacement(
    cavity: 'Cavity',
    cage_node_id: str = None,
    epsilon: float = 0.01
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Calculate distance-weighted displacement from NH3 to terminal X atoms.
    
    Computes the weighted centroid of 4 terminal X atoms where weights are
    inversely proportional to distance from NH3. This gives more influence
    to closer terminals, representing local interaction geometry.
    
    Parameters
    ----------
    cavity : Cavity
        Cavity object (must be spacer_rp or spacer_dj type)
    cage_node_id : str, optional
        If provided, only calculate for this specific half-cage (for DJ spacers)
        If None and cavity is DJ, returns list of results for all half-cages
    epsilon : float, optional
        Small constant to avoid division by zero. Default: 0.01
        
    Returns
    -------
    dict or list of dict
        For RP spacer: Single dict with keys:
            - weighted_centroid: (3,) Weighted centroid position in 3D
            - displacement: float Distance from NH3 to weighted centroid
            - displacement_vector: (3,) NH3 - weighted_centroid
            - weights: (4,) Weight for each terminal X
            - distances: (4,) Distance from NH3 to each terminal X
            - nh3_position: (3,) NH3 position (reference)
            - terminal_x_positions: (4, 3) Terminal X positions (reference)
            - cage_node_id: str Cage identifier
        
        For DJ spacer: List of 2 dicts (one per half-cage) with same structure
    """
    import sys
    
    # Validate cavity type
    if cavity.cavity_type not in ('spacer_rp', 'spacer_dj'):
        raise ValueError(f"calculate_weighted_centroid_displacement only supports spacer cavities. Got: {cavity.cavity_type}")
    
    # Helper function to extract NH3 position
    def get_nh3_position_for_cage(cavity: 'Cavity', cage_node_id: str = None) -> Optional[np.ndarray]:
        """Extract NH3 (N atom) position from anchor nodes in subgraph."""
        anchor_nodes = []
        
        # Find all anchor nodes
        for node in cavity.subgraph.nodes():
            if cavity.subgraph.nodes[node].get('node_type') == 'anchor':
                anchor_nodes.append(node)
        
        if not anchor_nodes:
            return None
        
        # If cage_node_id specified, filter to anchors connected to that cage
        if cage_node_id is not None:
            cage_anchors = []
            for anchor in anchor_nodes:
                # Check if this anchor is connected to the cage
                for neighbor in cavity.subgraph.neighbors(anchor):
                    if neighbor == cage_node_id:
                        cage_anchors.append(anchor)
                        break
            if cage_anchors:
                anchor_nodes = cage_anchors
        
        # Get N atom position from first anchor
        if anchor_nodes:
            anchor_node = anchor_nodes[0]
            # Find N atom connected to this anchor via 'contains' edge
            for neighbor in cavity.subgraph.neighbors(anchor_node):
                neighbor_data = cavity.subgraph.nodes[neighbor]
                if neighbor_data.get('symbol') == 'N':
                    pbc_pos = neighbor_data.get('pbc_position')
                    if pbc_pos is not None:
                        return np.array(pbc_pos)
        
        return None
    
    # Helper function to compute weighted centroid for single cage
    def compute_weighted_centroid_single_cage(
        cavity: 'Cavity',
        cage_node_id: str = None,
        epsilon: float = 0.01
    ) -> Dict[str, Any]:
        """Compute weighted centroid displacement for a single half-cage."""
        # Get octagonal rings
        octagon1, octagon2 = _identify_octagonal_rings(cavity, cage_node_id=cage_node_id)
        
        if octagon1 is None or octagon2 is None:
            raise RuntimeError(
                f"Failed to identify octagonal rings for cavity {cavity.id}"
                + (f" (cage '{cage_node_id}')" if cage_node_id else "")
            )
        
        # Terminal X atoms are last 4 in octagon2
        terminal_x_positions = octagon2[4:, :]  # Shape (4, 3)
        
        # Get NH3 position
        nh3_position = get_nh3_position_for_cage(cavity, cage_node_id=cage_node_id)
        if nh3_position is None:
            print(
                f"WARNING: Could not extract NH3 position for cavity {cavity.id}"
                + (f" (cage '{cage_node_id}')" if cage_node_id else ""),
                file=sys.stderr
            )
            # Return NaN results
            return {
                'weighted_centroid': np.array([np.nan, np.nan, np.nan]),
                'displacement': np.nan,
                'displacement_vector': np.array([np.nan, np.nan, np.nan]),
                'weights': np.array([np.nan] * 4),
                'distances': np.array([np.nan] * 4),
                'nh3_position': np.array([np.nan, np.nan, np.nan]),
                'terminal_x_positions': terminal_x_positions,
                'cage_node_id': str(cage_node_id) if cage_node_id else 'cage_0'
            }
        
        # Calculate distances from NH3 to each terminal X
        distances = np.array([
            np.linalg.norm(nh3_position - x_pos)
            for x_pos in terminal_x_positions
        ])
        
        # Calculate weights: w_i = 1 / (d_i + epsilon)
        weights = 1.0 / (distances + epsilon)
        
        # Normalize weights
        weights_sum = np.sum(weights)
        normalized_weights = weights / weights_sum
        
        # Compute weighted centroid
        weighted_centroid = np.sum(
            normalized_weights[:, np.newaxis] * terminal_x_positions,
            axis=0
        )
        
        # Compute displacement
        displacement_vector = nh3_position - weighted_centroid
        displacement = np.linalg.norm(displacement_vector)
        
        return {
            'weighted_centroid': weighted_centroid,
            'displacement': float(displacement),
            'displacement_vector': displacement_vector,
            'weights': normalized_weights,
            'distances': distances,
            'nh3_position': nh3_position,
            'terminal_x_positions': terminal_x_positions,
            'cage_node_id': str(cage_node_id) if cage_node_id else 'cage_0'
        }
    
    # Process based on cavity type
    if cavity.cavity_type == 'spacer_rp':
        # Single half-cage
        return compute_weighted_centroid_single_cage(cavity, cage_node_id=cage_node_id, epsilon=epsilon)
    
    elif cavity.cavity_type == 'spacer_dj':
        # Multiple half-cages
        if cage_node_id is not None:
            # Compute for specific cage
            return compute_weighted_centroid_single_cage(cavity, cage_node_id=cage_node_id, epsilon=epsilon)
        else:
            # Find all cage nodes and compute for each
            cage_nodes = [
                node for node in cavity.subgraph.nodes()
                if cavity.subgraph.nodes[node].get('node_type') in ('cage', 'half_cage')
            ]
            
            if not cage_nodes:
                print(
                    f"WARNING: No cage nodes found in DJ cavity {cavity.id}. Falling back to single computation.",
                    file=sys.stderr
                )
                return compute_weighted_centroid_single_cage(cavity, cage_node_id=None, epsilon=epsilon)
            
            results = []
            for cage_node in cage_nodes:
                result = compute_weighted_centroid_single_cage(
                    cavity,
                    cage_node_id=cage_node,
                    epsilon=epsilon
                )
                results.append(result)
            
            return results

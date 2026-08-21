"""
Twisted bilayer generator for monolayer structures.

Based on the Quadratic Twisted Bilayer Generator by Gabriel Xavier Pereira:
Institute of Physics, University of São Paulo, São Paulo, SP, Brazil
Email: gxpereira@usp.br

This module creates commensurate Moiré patterns from perovskite monolayers.
"""

import numpy as np
from ase import Atoms
from math import gcd


def _strip_vacuum(coords):
    """
    Remove vacuum from z-coordinates by shifting so minimum z is at 0.
    
    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of coordinates
        
    Returns
    -------
    np.ndarray
        Coordinates shifted so min(z) = 0
    float
        The original minimum z value (vacuum below)
    float
        The structure thickness (max_z - min_z)
    """
    z_min = np.min(coords[:, 2])
    z_max = np.max(coords[:, 2])
    shifted = coords.copy()
    shifted[:, 2] -= z_min
    return shifted, z_min, z_max - z_min


def _compute_twist_angle(m, n):
    """
    Compute the twist angle from commensurate indices (m, n).
    
    The twist angle θ is given by:
        θ = arctan(2mn / (m² - n²))
    
    Parameters
    ----------
    m, n : int
        Commensurate twist indices with m > n > 0
        
    Returns
    -------
    float
        Twist angle in radians
    """
    return np.arctan2(2 * m * n, m**2 - n**2)


def _get_rotation_matrix(m, n):
    """
    Get the 3x3 rotation matrix for twist angle from (m, n) indices.
    
    Parameters
    ----------
    m, n : int
        Commensurate twist indices
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix (rotates in XY plane, z unchanged)
    """
    denom = m**2 + n**2
    cos_theta = (m**2 - n**2) / denom
    sin_theta = (2 * m * n) / denom
    return np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ], dtype=np.float64)


def _lcm(a, b):
    """Compute least common multiple of two integers."""
    return abs(a * b) // gcd(a, b) if a and b else 0


def create_twisted_bilayer(mono1, mono2, m, n, interlayer_distance=11.0, vacuum=12.0, **kwargs):
    """
    Create a twisted bilayer structure from two monolayers.

    This function generates a twisted bilayer structure from two given monolayer
    crystals. It constructs two stacked layers that are related by a well-defined
    twist angle determined by the integers (m, n) according to the standard
    commensurate twist formula.
    
    The twist angle θ = arctan(2mn / (m² - n²)) creates a commensurate Moiré 
    superlattice with a supercell size factor of (m² + n²).

    Parameters
    ----------
    mono1 : q2DStructure
        First monolayer structure (bottom layer, not rotated)
    mono2 : q2DStructure
        Second monolayer structure (top layer, rotated by twist angle)
    m : int
        First integer parameter for twist angle calculation (m > n)
    n : int
        Second integer parameter for twist angle calculation (n > 0)
    interlayer_distance : float, optional
        Vertical distance between the top of bottom layer and bottom of top layer
        in Angstroms (default: 11.0)
    vacuum : float, optional
        Total vacuum space to add (split evenly above and below) in Angstroms 
        (default: 12.0)
    **kwargs : dict
        Additional parameters (currently unused, reserved for future use)

    Returns
    -------
    q2DStructure
        New q2DStructure with the twisted bilayer

    Raises
    ------
    ValueError
        If structure_type is not 'monolayer'
    ImportError
        If pymatgen is not available
    """
    for i, mono in enumerate([mono1, mono2], 1):
        if mono.structure_type != 'monolayer':
            raise ValueError(
                f"twist() method is only available for monolayer structures. "
                f"Structure {i} has structure_type: {mono.structure_type}"
            )
    
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.ase import AseAtomsAdaptor
    from q2D_Materials.core.structure import q2DStructure
    
    adapter = AseAtomsAdaptor()
    
    # Convert to pymatgen structures
    mono1_pmg = adapter.get_structure(mono1)
    mono2_pmg = adapter.get_structure(mono2)
    
    # Handle orthorhombic cells by squaring them
    def ensure_square_cell(pmg_structure):
        a, b = pmg_structure.lattice.lengths[:2]
        if abs(a - b) / max(a, b) > 0.01:  # >1% difference
            avg_a = (a + b) / 2.0
            old_matrix = pmg_structure.lattice.matrix
            new_matrix = np.array([
                [avg_a, 0, 0],
                [0, avg_a, 0],
                old_matrix[2]
            ])
            return Structure(
                Lattice(new_matrix),
                [site.species for site in pmg_structure],
                pmg_structure.frac_coords
            )
        return pmg_structure
    
    mono1_pmg = ensure_square_cell(mono1_pmg)
    mono2_pmg = ensure_square_cell(mono2_pmg)
    
    # Create Moiré supercells using the commensurate transformation
    # Bottom layer: transformation matrix [[m, n, 0], [-n, m, 0], [0, 0, 1]]
    slab_bottom = mono1_pmg.copy()
    slab_bottom.make_supercell([[m, n, 0], [-n, m, 0], [0, 0, 1]])
    
    # Top layer: transformation matrix [[m, -n, 0], [n, m, 0], [0, 0, 1]]
    slab_top = mono2_pmg.copy()
    slab_top.make_supercell([[m, -n, 0], [n, m, 0], [0, 0, 1]])
    
    # Extract coordinates
    bottom_coords = np.array([site.coords for site in slab_bottom], dtype=np.float64)
    top_coords = np.array([site.coords for site in slab_top], dtype=np.float64)
    
    # Strip vacuum from both layers (shift so each layer starts at z=0)
    bottom_coords, _, bottom_thickness = _strip_vacuum(bottom_coords)
    top_coords, _, top_thickness = _strip_vacuum(top_coords)
    
    # Apply rotation to top layer (the twist)
    rot_matrix = _get_rotation_matrix(m, n)
    top_coords_rotated = top_coords @ rot_matrix.T
    
    # Position layers:
    # Bottom layer: starts at z = vacuum/2
    # Top layer: starts at bottom_layer_top + interlayer_distance
    vacuum_below = vacuum / 2.0
    bottom_coords[:, 2] += vacuum_below
    
    top_layer_start = vacuum_below + bottom_thickness + interlayer_distance
    top_coords_rotated[:, 2] += top_layer_start
    
    # Combine coordinates and species
    all_coords = np.vstack([bottom_coords, top_coords_rotated])
    all_species = [site.species for site in slab_bottom] + [site.species for site in slab_top]
    
    # Calculate cell z-dimension: vacuum/2 + bottom + gap + top + vacuum/2
    total_z = vacuum + bottom_thickness + interlayer_distance + top_thickness
    
    # Create new lattice with proper z-dimension
    base_matrix = slab_bottom.lattice.matrix.copy()
    base_matrix[2] = [0, 0, total_z]
    new_lattice = Lattice(base_matrix)
    
    # Create bilayer structure
    bilayer = Structure(new_lattice, all_species, all_coords, coords_are_cartesian=True)
    
    # Convert back to ASE
    bilayer_atoms = adapter.get_atoms(bilayer)
    
    # Calculate twist angle for metadata
    twist_angle_rad = _compute_twist_angle(m, n)
    twist_angle_deg = np.degrees(twist_angle_rad)
    
    # Prepare metadata
    new_metadata = mono1._metadata.copy() if hasattr(mono1, '_metadata') else {}
    new_metadata['twist_params'] = (m, n)
    new_metadata['twist_angle_deg'] = twist_angle_deg
    new_metadata['interlayer_distance'] = interlayer_distance
    new_metadata['vacuum'] = vacuum
    new_metadata['supercell_factor'] = m**2 + n**2
    new_metadata['layer1_composition'] = {
        'A_ions': mono1.A_ions,
        'B_ions': mono1.B_ions,
        'X_ions': mono1.X_ions,
        'spacer': getattr(mono1, 'spacer', None)
    }
    new_metadata['layer2_composition'] = {
        'A_ions': mono2.A_ions,
        'B_ions': mono2.B_ions,
        'X_ions': mono2.X_ions,
        'spacer': getattr(mono2, 'spacer', None)
    }

    return q2DStructure(
        bilayer_atoms,
        structure_type='twister',
        BX_dist=mono1.BX_dist,
        A_ions=mono1.A_ions,
        B_ions=mono1.B_ions,
        X_ions=mono1.X_ions,
        xy_expansion=mono1.xy_expansion,
        spacer=getattr(mono1, 'spacer', None),
        spacer_molecule=getattr(mono1, 'spacer_molecule', None),
        **new_metadata
    )


def create_twisted_multilayer(monolayers, twist_angles, interlayer_distances, vacuum=12.0):
    """
    Create a twisted multilayer structure from multiple monolayers.
    
    This function stacks multiple monolayers with specified twist angles and
    interlayer distances. The first layer is the reference (no rotation),
    and subsequent layers are rotated relative to the first.

    Parameters
    ----------
    monolayers : list of q2DStructure
        List of monolayer structures to stack. Minimum 2 required.
    twist_angles : list of tuple (m, n) or None
        List of (m, n) tuples for twist angles. One per layer after the first.
        Use None for no rotation on a layer.
        Length must be len(monolayers) - 1.
    interlayer_distances : list of float
        Vertical distances between consecutive layers in Angstroms.
        Length must be len(monolayers) - 1.
    vacuum : float, optional
        Total vacuum space (split evenly above/below) in Angstroms (default: 12.0)

    Returns
    -------
    q2DStructure
        New q2DStructure with the twisted multilayer stack
    """
    if len(monolayers) < 2:
        raise ValueError(f"Need at least 2 monolayers, got {len(monolayers)}")
    
    if len(twist_angles) != len(monolayers) - 1:
        raise ValueError(
            f"twist_angles must have length {len(monolayers) - 1}, "
            f"got {len(twist_angles)}"
        )
    
    if len(interlayer_distances) != len(monolayers) - 1:
        raise ValueError(
            f"interlayer_distances must have length {len(monolayers) - 1}, "
            f"got {len(interlayer_distances)}"
        )
    
    for i, mono in enumerate(monolayers):
        if mono.structure_type != 'monolayer':
            raise ValueError(
                f"All structures must have structure_type='monolayer'. "
                f"Structure {i} has structure_type='{mono.structure_type}'"
            )
    
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.ase import AseAtomsAdaptor
    from q2D_Materials.core.structure import q2DStructure
    
    adapter = AseAtomsAdaptor()
    
    # Find common supercell from xy_expansions
    xy_expansions = [
        mono.xy_expansion if mono.xy_expansion else (1, 1) 
        for mono in monolayers
    ]
    
    common_nx = xy_expansions[0][0]
    common_ny = xy_expansions[0][1]
    for exp in xy_expansions[1:]:
        common_nx = _lcm(common_nx, exp[0])
        common_ny = _lcm(common_ny, exp[1])
    
    # Get the largest supercell expansion factors from twist angles
    max_supercell = 1
    for angle in twist_angles:
        if angle is not None:
            m, n = angle
            max_supercell = max(max_supercell, m**2 + n**2)
    
    # Prepare all layers
    all_coords = []
    all_species = []
    layer_thicknesses = []
    base_lattice = None
    
    for i, mono in enumerate(monolayers):
        mono_pmg = adapter.get_structure(mono)
        
        # Scale to common supercell
        exp = xy_expansions[i]
        scale_x = common_nx // exp[0]
        scale_y = common_ny // exp[1]
        
        if scale_x > 1 or scale_y > 1:
            mono_pmg.make_supercell([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
        
        # Handle orthorhombic cells
        a, b = mono_pmg.lattice.lengths[:2]
        if abs(a - b) / max(a, b) > 0.01:
            avg_a = (a + b) / 2.0
            old_matrix = mono_pmg.lattice.matrix
            new_matrix = np.array([[avg_a, 0, 0], [0, avg_a, 0], old_matrix[2]])
            mono_pmg = Structure(
                Lattice(new_matrix),
                [site.species for site in mono_pmg],
                mono_pmg.frac_coords
            )
        
        # Apply Moiré supercell transformation if twist angle specified for this layer
        if i > 0 and twist_angles[i - 1] is not None:
            m, n = twist_angles[i - 1]
            # Use the conjugate transformation for consistency
            mono_pmg.make_supercell([[m, -n, 0], [n, m, 0], [0, 0, 1]])
        elif i == 0:
            # First layer gets the standard transformation (from first twist angle if exists)
            if twist_angles[0] is not None:
                m, n = twist_angles[0]
                mono_pmg.make_supercell([[m, n, 0], [-n, m, 0], [0, 0, 1]])
        
        if base_lattice is None:
            base_lattice = mono_pmg.lattice
        
        coords = np.array([site.coords for site in mono_pmg], dtype=np.float64)
        species = [site.species for site in mono_pmg]
        
        # Strip vacuum
        coords, _, thickness = _strip_vacuum(coords)
        layer_thicknesses.append(thickness)
        
        # Apply rotation if specified
        if i > 0 and twist_angles[i - 1] is not None:
            m, n = twist_angles[i - 1]
            rot_matrix = _get_rotation_matrix(m, n)
            coords = coords @ rot_matrix.T
        
        all_coords.append(coords)
        all_species.extend(species)
    
    # Stack layers with proper spacing
    vacuum_below = vacuum / 2.0
    current_z = vacuum_below
    
    stacked_coords = []
    for i, coords in enumerate(all_coords):
        shifted = coords.copy()
        shifted[:, 2] += current_z
        stacked_coords.append(shifted)
        
        if i < len(monolayers) - 1:
            current_z += layer_thicknesses[i] + interlayer_distances[i]
    
    final_coords = np.vstack(stacked_coords)
    
    # Calculate total z
    total_z = vacuum + sum(layer_thicknesses) + sum(interlayer_distances)
    
    # Create new lattice
    new_matrix = base_lattice.matrix.copy()
    new_matrix[2] = [0, 0, total_z]
    new_lattice = Lattice(new_matrix)
    
    # Build structure
    multilayer = Structure(new_lattice, all_species, final_coords, coords_are_cartesian=True)
    atoms = adapter.get_atoms(multilayer)
    
    # Metadata
    first_mono = monolayers[0]
    metadata = first_mono._metadata.copy() if hasattr(first_mono, '_metadata') else {}
    metadata['twist_params'] = twist_angles
    metadata['interlayer_distances'] = interlayer_distances
    metadata['vacuum'] = vacuum
    metadata['n_layers'] = len(monolayers)
    metadata['common_supercell'] = (common_nx, common_ny)
    
    return q2DStructure(
        atoms,
        structure_type='twister',
        BX_dist=first_mono.BX_dist,
        A_ions=first_mono.A_ions,
        B_ions=first_mono.B_ions,
        X_ions=first_mono.X_ions,
        xy_expansion=(common_nx, common_ny),
        spacer=getattr(first_mono, 'spacer', None),
        spacer_molecule=getattr(first_mono, 'spacer_molecule', None),
        **metadata
    )

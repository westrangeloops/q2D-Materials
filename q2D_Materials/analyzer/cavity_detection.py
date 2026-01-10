"""A-site cation identification and cavity analysis.

This module identifies A-site cations and distinguishes them from spacer molecules
using cavity geometry and interlayer region analysis.
"""

import numpy as np
import networkx as nx
from collections import defaultdict

from ..utils.geometry import _calculate_distances
from .perovskite_constants import MOLECULAR_A_SITE_PATTERNS


def _find_attachment_nitrogens(
    atom_positions: np.ndarray,
    atom_symbols: list,
    cell: np.ndarray,
    graph: nx.Graph,
    atoms_in_octahedra: set,
) -> list:
    """Find nitrogen atoms that serve as attachment points for spacers.

    In 2D perovskites, spacers attach to the inorganic framework via NH3+ groups
    that hydrogen-bond to halide anions.

    Parameters
    ----------
    atom_positions : np.ndarray
        Array of atom positions
    atom_symbols : list
        List of atomic symbols
    cell : np.ndarray
        Unit cell matrix
    graph : nx.Graph
        Connectivity graph
    atoms_in_octahedra : set
        Indices of atoms in octahedra

    Returns
    -------
    list of dict
        Attachment nitrogen information
    """
    halide_elements = {'F', 'Cl', 'Br', 'I'}
    attachment_nitrogens = []
    
    for i, symbol in enumerate(atom_symbols):
        if symbol != 'N':
            continue
        if i in atoms_in_octahedra:
            continue
            
        atom_node = f'atom_{i}'
        if not graph.has_node(atom_node):
            continue
        
        # Check connectivity
        neighbors = list(graph.neighbors(atom_node))
        
        h_neighbors = []
        c_neighbors = []
        halide_neighbors = []

        for neighbor in neighbors:
            if neighbor.startswith('atom_'):
                neighbor_idx = int(neighbor.replace('atom_', ''))
                neighbor_symbol = atom_symbols[neighbor_idx]

                if neighbor_symbol == 'H':
                    h_neighbors.append(neighbor_idx)
                elif neighbor_symbol == 'C':
                    c_neighbors.append(neighbor_idx)
                elif neighbor_symbol in halide_elements:
                    halide_neighbors.append(neighbor_idx)

        is_attachment = False
        attachment_type = None

        if len(h_neighbors) >= 2:
            has_halide_contact = len(halide_neighbors) > 0

            if not has_halide_contact:
                for h_idx in h_neighbors:
                    h_node = f'atom_{h_idx}'
                    for h_neighbor in graph.neighbors(h_node):
                        if h_neighbor.startswith('atom_'):
                            h_neighbor_idx = int(h_neighbor.replace('atom_', ''))
                            if atom_symbols[h_neighbor_idx] in halide_elements:
                                has_halide_contact = True
                                break
                    if has_halide_contact:
                        break

            if has_halide_contact:
                is_attachment = True
                if len(c_neighbors) > 0:
                    attachment_type = 'NH3_organic'
                else:
                    attachment_type = 'NH4'
        
        if is_attachment:
            attachment_nitrogens.append({
                'index': i,
                'position': atom_positions[i].copy(),
                'type': attachment_type,
                'h_neighbors': h_neighbors,
                'c_neighbors': c_neighbors,
                'halide_neighbors': halide_neighbors,
            })
    
    return attachment_nitrogens

def _classify_molecular_type(mol, octahedra_info: list, atom_positions: np.ndarray) -> str:
    """Classify a molecular component as spacer or A-site cation.

    - DJ spacer: bifunctional, connects two slab faces (2 attachment points)
    - RP spacer: monofunctional, connects to one slab face (1 attachment point)
    - A-site: small cation enclosed in octahedral cavity

    Parameters
    ----------
    mol : ase.Atoms
        Molecular component
    octahedra_info : list
        List of octahedra information
    atom_positions : np.ndarray
        Full structure atom positions

    Returns
    -------
    str
        Classification: 'dj_spacer', 'rp_spacer', 'a_site', or 'unknown'
    """
    n_attachments = mol.info.get('n_attachments', 0)
    formula = mol.get_chemical_formula(mode='hill')
    n_atoms = len(mol)

    if formula in MOLECULAR_A_SITE_PATTERNS:
        if n_attachments <= 1 and n_atoms <= 8:
            return 'a_site'

    if n_attachments >= 2:
        return 'dj_spacer'
    elif n_attachments == 1:
        return 'rp_spacer'
    elif n_atoms <= 8:
        return 'a_site'
    else:
        return 'unknown'

def _has_path_to_multiple_layers(
    mol_indices: list,
    graph: nx.Graph,
    layer_info: dict,
    octahedra_info: list,
) -> tuple:
    """Check if a molecular component has connections to multiple layers.

    Spacers bridge between layers, while A-sites are enclosed within one layer's cavities.

    Parameters
    ----------
    mol_indices : list
        Indices of atoms in the molecular component
    graph : nx.Graph
        Connectivity graph
    layer_info : dict
        Layer information from _identify_layers
    octahedra_info : list
        List of octahedra dictionaries

    Returns
    -------
    tuple
        (has_multi_layer_path, connected_layer_ids)
    """
    oct_to_layer = {}
    for layer_id, layer_data in layer_info.items():
        for oct_idx in layer_data.get('octahedra', []):
            oct_to_layer[oct_idx] = layer_id

    atom_to_oct = {}
    for oct_idx, oct_data in enumerate(octahedra_info):
        central_idx = oct_data.get('central_atom_index')
        if central_idx is not None:
            atom_to_oct.setdefault(central_idx, []).append(oct_idx)

        for neighbor_list in ['terminal_atoms', 'interlayer_atoms', 'intralayer_atoms']:
            for atom_idx in oct_data.get(neighbor_list, []):
                atom_to_oct.setdefault(atom_idx, []).append(oct_idx)

    connected_layers = set()

    for atom_idx in mol_indices:
        atom_node = f'atom_{atom_idx}'
        if not graph.has_node(atom_node):
            continue

        for neighbor in graph.neighbors(atom_node):
            if neighbor.startswith('atom_'):
                neighbor_idx = int(neighbor.replace('atom_', ''))
                if neighbor_idx in atom_to_oct:
                    for oct_idx in atom_to_oct[neighbor_idx]:
                        if oct_idx in oct_to_layer:
                            connected_layers.add(oct_to_layer[oct_idx])

    has_multi_layer = len(connected_layers) > 1
    return has_multi_layer, list(connected_layers)

def _analyze_slab_structure(
    layers: dict,
    octahedra_info: list,
    shared_atoms: dict,
    atom_positions: np.ndarray = None,
) -> dict:
    """
    Analyze slab structure to determine thickness and terminal octahedra.
    
    This is crucial for distinguishing A-sites from spacers:
    - Terminal octahedra have axial connections in only one direction
    - For thickness=1: NO A-sites possible (all octahedra are terminal)
    - A-sites exist only in cavities WITHIN a slab (not between slabs)
    
    Axial connections are identified by:
    1. Corner-sharing (1 shared atom) between octahedra
    2. At different z-levels (different layers in the stacking direction)
    
    Parameters
    ----------
    layers : dict
        Layer information from _identify_layers
    octahedra_info : list
        List of octahedra dictionaries
    shared_atoms : dict
        Dictionary mapping octahedra pairs to shared atom indices
    atom_positions : np.ndarray, optional
        Atom positions for z-level analysis
        
    Returns
    -------
    dict
        Slab analysis with:
        - 'thickness': Number of octahedra layers in the slab
        - 'terminal_octahedra': Set of octahedra indices on slab surfaces
        - 'internal_octahedra': Set of octahedra indices inside slab
        - 'has_a_site_cavities': Boolean indicating if A-site cavities exist
    """
    n_octahedra = len(octahedra_info)
    
    if n_octahedra == 0:
        return {
            'thickness': 0,
            'terminal_octahedra': set(),
            'internal_octahedra': set(),
            'has_a_site_cavities': False,
        }

    oct_z_coords = {}
    for oct_idx, oct_data in enumerate(octahedra_info):
        center_idx = oct_data.get('central_atom_index')
        if center_idx is not None and atom_positions is not None:
            oct_z_coords[oct_idx] = atom_positions[center_idx][2]
        else:
            oct_z_coords[oct_idx] = 0.0

    z_tolerance = 1.0
    z_levels = []
    oct_to_level = {}

    for oct_idx in range(n_octahedra):
        z = oct_z_coords[oct_idx]
        found_level = None
        for level_idx, (level_z, level_octs) in enumerate(z_levels):
            if abs(z - level_z) < z_tolerance:
                found_level = level_idx
                break

        if found_level is not None:
            z_levels[found_level][1].append(oct_idx)
            oct_to_level[oct_idx] = found_level
        else:
            z_levels.append((z, [oct_idx]))
            oct_to_level[oct_idx] = len(z_levels) - 1

    z_levels.sort(key=lambda x: x[0])

    axial_connections = defaultdict(set)

    for (oct1, oct2), shared in shared_atoms.items():
        n_shared = len(shared)
        if n_shared == 1:
            level1 = oct_to_level.get(oct1, -1)
            level2 = oct_to_level.get(oct2, -1)
            if level1 != level2:
                axial_connections[oct1].add(oct2)
                axial_connections[oct2].add(oct1)

    terminal_octahedra = set()
    internal_octahedra = set()

    if len(z_levels) == 1:
        terminal_octahedra = set(range(n_octahedra))
    elif len(z_levels) >= 2:
        bottom_level_octs = z_levels[0][1]
        top_level_octs = z_levels[-1][1]
        terminal_octahedra = set(bottom_level_octs + top_level_octs)

        for level_idx in range(1, len(z_levels) - 1):
            internal_octahedra.update(z_levels[level_idx][1])

    thickness = len(z_levels)
    has_a_site_cavities = len(internal_octahedra) > 0 or thickness > 1
    
    return {
        'thickness': thickness,
        'terminal_octahedra': terminal_octahedra,
        'internal_octahedra': internal_octahedra,
        'has_a_site_cavities': has_a_site_cavities,
        'axial_connections': dict(axial_connections),
        'z_levels': [(z, octs) for z, octs in z_levels],
    }

def _is_in_interlayer_region(
    position: np.ndarray,
    octahedra_info: list,
    atom_positions: np.ndarray,
    terminal_octahedra: set,
    cell: np.ndarray,
    z_levels: list = None,
) -> bool:
    """Check if a position is in the interlayer region (between slabs).

    Atoms in the interlayer region are spacers, NOT A-sites.
    For bulk structures (no interlayer gap), this returns False.

    Parameters
    ----------
    position : np.ndarray
        Position to check
    octahedra_info : list
        List of octahedra dictionaries
    atom_positions : np.ndarray
        Array of atom positions
    terminal_octahedra : set
        Indices of terminal octahedra
    cell : np.ndarray
        Unit cell matrix
    z_levels : list, optional
        List of (z_coord, octahedra_indices) tuples

    Returns
    -------
    bool
        True if position is in interlayer region (spacer territory)
    """
    if not terminal_octahedra or not octahedra_info:
        return False

    all_z = []
    for oct in octahedra_info:
        center_idx = oct.get('central_atom_index')
        if center_idx is not None:
            all_z.append(atom_positions[center_idx][2])

    if len(all_z) < 2:
        return False

    min_oct_z = min(all_z)
    max_oct_z = max(all_z)

    inv_cell = np.linalg.inv(cell)
    pos_z = position[2]
    
    # Estimate interlayer gap: typical B-X distance * 2 (space between terminal X atoms)
    # For 2D perovskites, there should be a clear gap (> 3 Å) beyond the octahedra
    gap_threshold = 3.0  # Å - minimum gap to be considered interlayer
    
    # For interlayer detection, we need to identify if the position is in a GAP
    # between slabs, not just in a regular A-site cavity.
    #
    # Key insight derived from B-X geometry:
    # - Regular octahedra layer spacing = 2 × B-X distance (apical X to apical X)
    # - Regular A-site cavity ≈ B-X × √3 (cuboctahedral cavity)
    # - 2D interlayer gap > 2 × regular spacing (due to spacer molecules)
    
    # Calculate the expected layer spacing from actual B-X distances in octahedra
    # Filter to only include valid B-X bonds (< 5 Å for typical perovskites)
    max_bx_distance = 5.0  # Maximum reasonable B-X bond distance
    bx_distances = []
    for oct in octahedra_info:
        center_idx = oct.get('central_atom_index')
        if center_idx is None:
            continue
        b_pos = atom_positions[center_idx]
        for neighbor_list in ['terminal_atoms', 'interlayer_atoms', 'intralayer_atoms']:
            for x_idx in oct.get(neighbor_list, []):
                x_pos = atom_positions[x_idx]
                dist = np.linalg.norm(b_pos - x_pos)
                # Only include actual B-X bonds, not PBC artifacts
                if dist < max_bx_distance:
                    bx_distances.append(dist)
    
    if bx_distances:
        avg_bx = np.mean(bx_distances)
        # Expected layer spacing = 2 × B-X (apical X to apical X across octahedron)
        expected_layer_spacing = 2 * avg_bx
        # A-site cavity radius = B-X × √3
        cavity_radius = avg_bx * np.sqrt(3)
    else:
        # Fallback for unknown structures
        expected_layer_spacing = 7.0
        cavity_radius = 6.0
    
    # For 2D perovskites, check BOTH internal gaps AND cell boundary gaps
    # Even if internal gaps are uniform, there may be an interlayer at cell boundaries
    
    cell_z = cell[2, 2]
    gap_at_bottom = min_oct_z
    gap_at_top = cell_z - max_oct_z
    
    # Calculate boundary gap threshold
    # For bulk structures, octahedra centers are offset from cell origin by ~1 B-X
    # An interlayer gap should be significantly larger than this (at least 1 full layer spacing)
    # to contain spacer molecules
    interlayer_gap_threshold = expected_layer_spacing * 0.8
    
    # Only consider it an interlayer if the gap is larger than 1 B-X distance
    # (normal surface boundary is about 1 B-X, interlayer should be larger)
    has_boundary_interlayer = (gap_at_bottom > interlayer_gap_threshold or 
                                gap_at_top > interlayer_gap_threshold)
    
    if z_levels and len(z_levels) >= 2:
        z_levels_sorted = sorted(z_levels, key=lambda x: x[0])
        
        # Calculate gaps between adjacent z-levels
        gaps = []
        for i in range(len(z_levels_sorted) - 1):
            z1 = z_levels_sorted[i][0]
            z2 = z_levels_sorted[i + 1][0]
            gaps.append(z2 - z1)
        
        if gaps:
            min_gap = min(gaps)
            max_gap = max(gaps)
            
            # If all INTERNAL gaps are similar and there's NO boundary interlayer,
            # this is a bulk structure
            gap_variation = max_gap / min_gap if min_gap > 0 else 1.0
            gaps_are_uniform = gap_variation < 1.3
            gaps_match_expected = all(
                abs(g - expected_layer_spacing) < expected_layer_spacing * 0.3 
                for g in gaps
            )
            
            if gaps_are_uniform and gaps_match_expected and not has_boundary_interlayer:
                # Uniform spacing AND no boundary gap - true bulk perovskite
                return False
            
            # If there's a significantly larger INTERNAL gap,
            # check if position is in it
            interlayer_threshold = expected_layer_spacing * 1.5
            
            for i in range(len(z_levels_sorted) - 1):
                z1 = z_levels_sorted[i][0]
                z2 = z_levels_sorted[i + 1][0]
                gap = z2 - z1
                
                if gap > interlayer_threshold:
                    # This is an interlayer gap - check if position is in it
                    gap_center = (z1 + z2) / 2
                    if abs(pos_z - gap_center) < gap / 2.5:
                        return True
    
    # Check if position is in the interlayer region at cell boundaries
    # For 2D perovskites, there should be a vacuum/spacer region beyond the slabs
    cell_z = cell[2, 2]
    
    # Calculate the expected interlayer gap for 2D perovskites
    # Interlayer region starts immediately beyond the octahedra
    # (where the terminal X-site atoms end, about B-X distance from center)
    surface_distance = avg_bx  # Distance from B-site to terminal X
    
    # Check if there's a significant vacuum/interlayer gap at cell boundaries
    # The gap should be larger than normal A-site cavity distance
    gap_at_bottom = min_oct_z
    gap_at_top = cell_z - max_oct_z
    
    # For 2D perovskites, the interlayer gap is at least 1.5× layer spacing
    # because it contains spacer molecules or atoms
    interlayer_gap_threshold = expected_layer_spacing * 0.5  # Half a layer spacing
    
    # If there's a genuine interlayer gap at the cell boundary
    has_interlayer_at_bottom = gap_at_bottom > interlayer_gap_threshold
    has_interlayer_at_top = gap_at_top > interlayer_gap_threshold
    
    # Check if position is in the interlayer region
    # Position should be beyond the octahedra by at least the B-X distance
    # Use >= to include atoms exactly at the boundary
    if has_interlayer_at_bottom and pos_z <= min_oct_z - surface_distance:
        return True
    elif has_interlayer_at_top and pos_z >= max_oct_z + surface_distance:
        return True
    
    # Position is within the slab structure - not interlayer
    return False

def _calculate_cavity_radius_from_octahedra(
    octahedra_info: list,
    atom_positions: np.ndarray,
    cell: np.ndarray,
) -> float:
    """
    Calculate the expected A-site to B-site cavity radius from actual octahedra geometry.
    
    The cavity radius is derived from measured B-X distances in the detected octahedra,
    not from lattice parameters (which would be wrong for supercells or distorted systems).
    
    Geometric relationship:
    - First, measure actual B-X distances from all detected octahedra
    - Average these to account for distortions
    - A-B distance ≈ mean(B-X) × √3 (cuboctahedral cavity geometry)
    
    Parameters
    ----------
    octahedra_info : list
        List of octahedra dictionaries with central_atom_index and neighbor lists
    atom_positions : np.ndarray
        Array of atom positions
    cell : np.ndarray
        Unit cell matrix for PBC-aware distance calculations
        
    Returns
    -------
    float
        Calculated A-B cavity radius in Angstroms (with 20% tolerance for distortions)
    """
    if not octahedra_info:
        return 6.5  # Default fallback for typical perovskites
    
    # Collect all B-X distances from detected octahedra
    bx_distances = []
    
    # Maximum reasonable B-X bond distance (filters out PBC artifacts)
    max_bx_distance = 5.0
    
    for oct_data in octahedra_info:
        central_idx = oct_data.get('central_atom_index')
        if central_idx is None:
            continue
        
        b_position = atom_positions[central_idx]
        
        # Measure distances to all X-site neighbors in this octahedron
        for neighbor_list in ['terminal_atoms', 'interlayer_atoms', 'intralayer_atoms']:
            for x_idx in oct_data.get(neighbor_list, []):
                x_position = atom_positions[x_idx]
                distance = _calculate_distances(b_position, [x_position], cell)[0]
                # Only include actual B-X bonds, not PBC artifacts
                if distance < max_bx_distance:
                    bx_distances.append(distance)
    
    if not bx_distances:
        return 6.5  # Default fallback
    
    # Average B-X distance (accounts for distortions)
    avg_bx = np.mean(bx_distances)
    
    # A-B distance ≈ B-X × √3 (cuboctahedral geometry)
    # Add 20% tolerance for distorted structures and thermal effects
    cavity_radius = avg_bx * np.sqrt(3) * 1.2
    
    return cavity_radius

def _is_molecule_in_cavity(
    mol_center: np.ndarray,
    octahedra_info: list,
    atom_positions: np.ndarray,
    cell: np.ndarray,
    cavity_radius: float = None,
) -> bool:
    """
    Check if a molecule's center is within an octahedral cavity.
    
    A-site cations sit in the cuboctahedral cavity formed by 8 corner-sharing
    octahedra. The cavity radius is calculated geometrically from measured 
    B-X bond distances (not lattice parameters):
    
        cavity_radius ≈ mean(B-X) × √3 × 1.2
    
    This works correctly for supercells and distorted structures because it
    uses actual measured distances from detected octahedra.
    
    Parameters
    ----------
    mol_center : np.ndarray
        Center of mass of the molecule
    octahedra_info : list
        List of octahedra dictionaries with neighbor information
    atom_positions : np.ndarray
        Full structure atom positions
    cell : np.ndarray
        Unit cell matrix for PBC-aware calculations
    cavity_radius : float, optional
        Maximum distance from octahedral center to be considered "in cavity".
        If None, calculated automatically from octahedra geometry.
        
    Returns
    -------
    bool
        True if molecule is within the calculated cavity radius from B-sites
    """
    if not octahedra_info:
        # If no octahedra info, assume it's in a cavity (e.g., for unknown structures)
        return True
    
    # Get octahedral centers
    oct_centers = []
    for oct_data in octahedra_info:
        central_idx = oct_data.get('central_atom_index')
        if central_idx is not None:
            oct_centers.append(atom_positions[central_idx])
    
    if not oct_centers:
        return True  # No centers to compare against
    
    oct_centers = np.array(oct_centers)
    
    # Calculate cavity radius from actual octahedra geometry if not provided
    if cavity_radius is None:
        cavity_radius = _calculate_cavity_radius_from_octahedra(
            octahedra_info, atom_positions, cell
        )
    
    # Calculate distances to octahedral centers
    distances = _calculate_distances(mol_center, oct_centers, cell)
    min_distance = np.min(distances)
    
    # Minimum distance prevents selecting B-site atoms themselves
    # The B-X distance is typically ~3 Å, so use half of that as minimum
    min_threshold = cavity_radius * 0.3  # ~30% of cavity radius
    
    return min_threshold < min_distance < cavity_radius

def _identify_a_site_cations(
    atom_positions: np.ndarray,
    atom_symbols: list,
    cell: np.ndarray,
    b_site_indices: set,
    atoms_in_octahedra: set,
    spacer_atom_indices: set,
    graph: nx.Graph = None,
    octahedra_info: list = None,
    layers: dict = None,
    shared_atoms: dict = None,
) -> list:
    """
    Identify A-site cations (atoms in octahedral cavities, not B-site).
    
    A-sites are cations located in the cavities formed by corner-sharing
    octahedra WITHIN a slab. They are not part of the octahedra themselves.
    
    Key logic:
    - Thickness=1: NO A-sites possible (all atoms between slabs are spacers)
    - Atoms in interlayer region (between terminal octahedra) are spacers, not A-sites
    - A-sites exist only in cavities within a slab (between corner-sharing octahedra)
    
    Supports both atomic (Cs, Rb, K) and molecular (MA, FA) A-site cations.
    
    Parameters
    ----------
    atom_positions : np.ndarray
        Array of atom positions
    atom_symbols : list
        List of atomic symbols
    cell : np.ndarray
        Unit cell matrix
    b_site_indices : set
        Indices of B-site (octahedral center) atoms
    atoms_in_octahedra : set
        Indices of all atoms in octahedra (B and X)
    spacer_atom_indices : set
        Indices of atoms identified as spacers
    graph : nx.Graph, optional
        Connectivity graph for better molecular detection
    octahedra_info : list, optional
        List of octahedra information for cavity detection
    layers : dict, optional
        Layer information for slab analysis
    shared_atoms : dict, optional
        Shared atoms between octahedra for connectivity analysis
        
    Returns
    -------
    list of dict
        A-site information dicts
    """
    # Analyze slab structure to determine if A-site cavities exist
    slab_info = None
    if octahedra_info and shared_atoms is not None:
        slab_info = _analyze_slab_structure(
            layers or {}, octahedra_info, shared_atoms, atom_positions
        )
    
    # For thickness=1 or no internal octahedra: no A-site cavities exist
    # All non-octahedral atoms between slabs are spacers
    has_a_site_cavities = True
    terminal_octahedra = set()
    z_levels = []
    if slab_info:
        has_a_site_cavities = slab_info.get('has_a_site_cavities', True)
        terminal_octahedra = slab_info.get('terminal_octahedra', set())
        z_levels = slab_info.get('z_levels', [])
    
    # Common A-site cations (atomic)
    atomic_a_sites = {'Cs', 'Rb', 'K', 'Na', 'Li', 'Ba', 'Sr', 'Ca', 'Tl', 'Ag'}
    
    # Elements that could be part of molecular A-sites
    molecular_a_elements = {'C', 'N', 'H', 'O'}
    
    # Known molecular A-site formulas (sorted element symbols -> name)
    molecular_a_patterns = {
        'CHHHHHHN': 'MA',      # CH3NH3+ (methylammonium)
        'CHHHHNN': 'FA',       # HC(NH2)2+ (formamidinium)
        'HHHHN': 'NH4',        # NH4+ (ammonium)
        'CCHHHHHHHN': 'EA',    # C2H5NH3+ (ethylammonium)
        'CCHHHHHHHHN': 'DMA',  # (CH3)2NH2+ (dimethylammonium)
    }
    
    a_sites = []
    processed_indices = set()
    
    # First pass: Find atomic A-sites
    for i, symbol in enumerate(atom_symbols):
        if i in processed_indices:
            continue
        if i in b_site_indices:
            continue
        if i in atoms_in_octahedra:
            continue
        if i in spacer_atom_indices:
            continue
        
        if symbol in atomic_a_sites:
            # Check if this atom is in interlayer region (spacer territory)
            is_interlayer = False
            if terminal_octahedra and octahedra_info:
                is_interlayer = _is_in_interlayer_region(
                    atom_positions[i], octahedra_info, atom_positions,
                    terminal_octahedra, cell, z_levels
                )
            
            # Only classify as A-site if:
            # 1. A-site cavities exist (thickness > 1 or bulk), AND
            # 2. Atom is NOT in interlayer region
            if has_a_site_cavities and not is_interlayer:
                a_sites.append({
                    'atom_index': i,
                    'symbol': symbol,
                    'position': atom_positions[i].tolist(),
                    'is_molecular': False,
                    'molecule_atoms': [i],
                'formula': symbol,
            })
            processed_indices.add(i)
    
    # Second pass: Find molecular A-sites (MA, FA, etc.)
    # These are small organic cations not identified as spacers
    for i, symbol in enumerate(atom_symbols):
        if i in processed_indices:
            continue
        if i in b_site_indices:
            continue
        if i in atoms_in_octahedra:
            continue
        if i in spacer_atom_indices:
            continue
        
        if symbol not in molecular_a_elements:
            continue
        
        # Use graph if available for better connectivity
        if graph is not None:
            # BFS to find connected molecular component
            connected = set()
            queue = deque([i])
            while queue:
                idx = queue.popleft()
                if idx in connected or idx in processed_indices:
                    continue
                if idx in atoms_in_octahedra or idx in spacer_atom_indices:
                    continue
                if atom_symbols[idx] not in molecular_a_elements:
                    continue
                    
                connected.add(idx)
                
                atom_node = f'atom_{idx}'
                if graph.has_node(atom_node):
                    for neighbor in graph.neighbors(atom_node):
                        if neighbor.startswith('atom_'):
                            neighbor_idx = int(neighbor.replace('atom_', ''))
                            if neighbor_idx not in connected:
                                queue.append(neighbor_idx)
            
            connected = list(connected)
        else:
            # Fallback: distance-based connectivity
            connected = [i]
            for j, sym_j in enumerate(atom_symbols):
                if j == i or j in atoms_in_octahedra or j in spacer_atom_indices:
                    continue
                if j in processed_indices:
                    continue
                if sym_j in molecular_a_elements:
                    dist = _calculate_distances(atom_positions[i], [atom_positions[j]], cell)[0]
                    if dist < 2.0:  # Within bonding distance
                        connected.append(j)
        
        if not connected:
            continue
        
        # Check if all atoms in this molecule are already processed
        if all(idx in processed_indices for idx in connected):
            continue
        
        # Get formula for this molecular component
        mol_symbols = sorted([atom_symbols[idx] for idx in connected])
        mol_formula_key = ''.join(mol_symbols)
        
        # Count elements
        c_count = mol_symbols.count('C')
        h_count = mol_symbols.count('H')
        n_count = mol_symbols.count('N')
        total_atoms = len(mol_symbols)
        
        # Check if it matches known A-site patterns
        a_site_name = None
        
        if mol_formula_key in molecular_a_patterns:
            a_site_name = molecular_a_patterns[mol_formula_key]
        elif c_count == 1 and n_count == 1 and 4 <= h_count <= 6:
            a_site_name = 'MA'  # Methylammonium-like
        elif c_count == 1 and n_count == 2 and 4 <= h_count <= 5:
            a_site_name = 'FA'  # Formamidinium-like
        elif n_count == 1 and h_count == 4 and c_count == 0:
            a_site_name = 'NH4'  # Ammonium
        elif n_count >= 1 and c_count <= 2 and total_atoms <= 12:
            # Small molecule with nitrogen - likely molecular A-site
            a_site_name = f'organic_A'
        
        if a_site_name:
            # Verify it's in a cavity (not bridging layers)
            mol_center = np.mean(atom_positions[connected], axis=0)
            is_in_cavity = True
            if octahedra_info:
                # Calculate cavity radius from actual octahedra geometry
                cavity_radius = _calculate_cavity_radius_from_octahedra(
                    octahedra_info, atom_positions, cell
                )
                is_in_cavity = _is_molecule_in_cavity(
                    mol_center, octahedra_info, atom_positions, cell, cavity_radius
                )
            
            if is_in_cavity:
                a_sites.append({
                    'atom_index': connected[0],  # Primary index
                    'symbol': a_site_name,
                    'position': mol_center.tolist(),
                    'is_molecular': True,
                    'molecule_atoms': connected,
                    'formula': ''.join(sorted(mol_symbols)),
                })
                for idx in connected:
                    processed_indices.add(idx)
    
    return a_sites



"""
Template-based structure building.

This module provides geometry-first structure building. All structures
are built as templates based on BX distance (octahedron size) and then
populated with actual elements later.

This separation allows:
- Glazer tilting to work on pure geometry without element identification
- Compositional mixing at any site (A, B, X) in any structure type
- Clean separation of geometry from chemistry
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from ase import Atoms


# -----------------------------------------------------------------------------
# Site role constants
# -----------------------------------------------------------------------------
SITE_ROLE_KEY = "site_role"
SITE_A = "A_site"
SITE_B = "B_site"
SITE_X = "X_site"
SITE_SPACER = "spacer"

# Placeholder symbols for template structures
# Using noble gases that won't conflict with real perovskite elements
TEMPLATE_A = "He"  # A-site placeholder
TEMPLATE_B = "Ne"  # B-site placeholder
TEMPLATE_X = "Ar"  # X-site placeholder


@dataclass
class TemplateInfo:
    """Metadata about a template structure."""
    BX_dist: float
    supercell: Tuple[int, int, int]
    n_A_sites: int
    n_B_sites: int
    n_X_sites: int
    structure_type: str = "bulk"


def build_cubic_template(
    BX_dist: float,
    supercell: Tuple[int, int, int] = (1, 1, 1),
) -> Atoms:
    """
    Build a cubic perovskite template based on BX distance.
    
    The template uses placeholder elements (He, Ne, Ar) for A, B, X sites.
    These are later replaced with actual elements by populate_structure().
    
    Parameters
    ----------
    BX_dist : float
        B-X bond distance in Angstroms. This determines the octahedron size.
        The cubic lattice parameter a0 = 2 * BX_dist.
    supercell : tuple of int
        Supercell dimensions (nx, ny, nz)
        
    Returns
    -------
    Atoms
        Template structure with:
        - Placeholder symbols (He=A, Ne=B, Ar=X)
        - Site roles in atoms.arrays[SITE_ROLE_KEY]
        - Template info in atoms.info['template_info']
        
    Notes
    -----
    The standard cubic perovskite ABX3 has:
      - A at (0, 0, 0) corners
      - B at (0.5, 0.5, 0.5) body center
      - X at (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5) face centers
    """
    a0 = 2 * BX_dist
    
    # Fractional positions for single unit cell
    frac_positions = {
        SITE_A: [(0.0, 0.0, 0.0)],
        SITE_B: [(0.5, 0.5, 0.5)],
        SITE_X: [(0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)],
    }
    
    symbols_map = {
        SITE_A: TEMPLATE_A,
        SITE_B: TEMPLATE_B,
        SITE_X: TEMPLATE_X,
    }
    
    nx, ny, nz = supercell
    
    symbols = []
    positions = []
    roles = []
    
    for role, role_positions in frac_positions.items():
        symbol = symbols_map[role]
        
        for frac in role_positions:
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        x = (frac[0] + ix) * a0
                        y = (frac[1] + iy) * a0
                        z = (frac[2] + iz) * a0
                        symbols.append(symbol)
                        positions.append([x, y, z])
                        roles.append(role)
    
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[nx * a0, ny * a0, nz * a0],
        pbc=True,
    )
    
    # Tag sites with roles
    atoms.arrays[SITE_ROLE_KEY] = np.array(roles, dtype=object)
    
    # Store template info
    atoms.info['template_info'] = TemplateInfo(
        BX_dist=BX_dist,
        supercell=supercell,
        n_A_sites=sum(1 for r in roles if r == SITE_A),
        n_B_sites=sum(1 for r in roles if r == SITE_B),
        n_X_sites=sum(1 for r in roles if r == SITE_X),
        structure_type="bulk",
    )
    atoms.info['BX_dist'] = BX_dist
    
    return atoms


def build_slab_template(
    BX_dist: float,
    n_layers: int,
    supercell_xy: Tuple[int, int] = (1, 1),
    include_surface_A: bool = False,
    reduced: bool = False,
) -> Atoms:
    """
    Build a (100)-oriented slab template with n octahedral layers.
    
    Parameters
    ----------
    BX_dist : float
        B-X bond distance in Angstroms
    n_layers : int
        Number of octahedral layers
    supercell_xy : tuple
        In-plane supercell dimensions (nx, ny)
    include_surface_A : bool
        If False, exclude surface A-sites (for 2D phases where spacers replace A).
        If True, include all A-sites (for bulk-like slabs).
    reduced : bool
        If True, create reduced (2-octahedra) cell by post-processing complete cell.
        If False, create complete (4-octahedra) cell.
        
    Returns
    -------
    Atoms
        Slab template with proper octahedral connectivity
    """
    a0 = 2 * BX_dist
    nx, ny = supercell_xy
    
    symbols = []
    positions = []
    roles = []
    
    # Build B and X sites for each octahedral layer
    for iz in range(n_layers):
        for ix in range(nx):
            for iy in range(ny):
                # B-site at center of octahedron
                x_b = (0.5 + ix) * a0
                y_b = (0.5 + iy) * a0
                z_b = (0.5 + iz) * a0
                symbols.append(TEMPLATE_B)
                positions.append([x_b, y_b, z_b])
                roles.append(SITE_B)
                
                # X-sites at faces (3 per octahedron, shared with neighbors)
                # Bottom axial (z = iz * a0)
                symbols.append(TEMPLATE_X)
                positions.append([(0.5 + ix) * a0, (0.5 + iy) * a0, iz * a0])
                roles.append(SITE_X)
                
                # Equatorial (side faces)
                symbols.append(TEMPLATE_X)
                positions.append([(0.5 + ix) * a0, iy * a0, (0.5 + iz) * a0])
                roles.append(SITE_X)
                
                symbols.append(TEMPLATE_X)
                positions.append([ix * a0, (0.5 + iy) * a0, (0.5 + iz) * a0])
                roles.append(SITE_X)
    
    # Top capping X atoms (axial at top surface)
    z_top = n_layers * a0
    for ix in range(nx):
        for iy in range(ny):
            symbols.append(TEMPLATE_X)
            positions.append([(0.5 + ix) * a0, (0.5 + iy) * a0, z_top])
            roles.append(SITE_X)
    
    # A-sites
    if include_surface_A:
        # All A-sites including surfaces
        for iz in range(n_layers + 1):
            for ix in range(nx):
                for iy in range(ny):
                    symbols.append(TEMPLATE_A)
                    positions.append([ix * a0, iy * a0, iz * a0])
                    roles.append(SITE_A)
    else:
        # Only internal A-sites (between octahedral layers, for n > 1)
        for iz in range(1, n_layers):
            for ix in range(nx):
                for iy in range(ny):
                    symbols.append(TEMPLATE_A)
                    positions.append([ix * a0, iy * a0, iz * a0])
                    roles.append(SITE_A)
    
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[nx * a0, ny * a0, n_layers * a0],
        pbc=True,
    )
    atoms.arrays[SITE_ROLE_KEY] = np.array(roles, dtype=object)
    
    # Store template info
    atoms.info['template_info'] = TemplateInfo(
        BX_dist=BX_dist,
        supercell=(nx, ny, n_layers),
        n_A_sites=sum(1 for r in roles if r == SITE_A),
        n_B_sites=sum(1 for r in roles if r == SITE_B),
        n_X_sites=sum(1 for r in roles if r == SITE_X),
        structure_type="slab",
    )
    atoms.info['BX_dist'] = BX_dist
    atoms.info['n_layers'] = n_layers
    
    # Apply reduction if requested
    if reduced:
        atoms = reduce_slab_cell(atoms, BX_dist)
    
    return atoms


def extract_slab_from_bulk(
    bulk: Atoms,
    n_layers: int,
    bulk_nz: int,
    include_surface_A: bool = False,
    reduced: bool = False,
) -> Atoms:
    """
    Extract a slab from a bulk template (e.g., after Glazer tilting).
    
    This preserves any geometric distortions (like Glazer tilting) while
    extracting the correct number of octahedral layers.
    
    Parameters
    ----------
    bulk : Atoms
        Bulk template (possibly with Glazer tilting applied)
    n_layers : int
        Number of octahedral layers to extract
    bulk_nz : int
        Number of octahedral layers in the bulk (z-direction)
    include_surface_A : bool
        If False, exclude surface A-sites
    reduced : bool
        If True, apply reduction transformation after extraction to create
        2-octahedra cell. The B-X distance will be measured from the structure.
        
    Returns
    -------
    Atoms
        Slab template with top capping X atoms
    """
    slab = bulk.copy()
    roles = slab.arrays.get(SITE_ROLE_KEY)
    positions = slab.get_positions()
    cell = slab.get_cell()
    cell_lengths = np.linalg.norm(cell, axis=1)
    symbols = slab.get_chemical_symbols()
    
    # Calculate layer height
    layer_height = cell_lengths[2] / bulk_nz
    
    # Normalize z to start at 0
    z_coords = positions[:, 2]
    z_min_bulk = z_coords.min()
    z_normalized = z_coords - z_min_bulk
    
    # Include atoms up to n_layers (with tolerance for tilting)
    z_max_slab = n_layers * layer_height
    tolerance = 0.3 * layer_height
    
    mask = z_normalized <= (z_max_slab + tolerance)
    
    # Exclude surface A-sites for 2D structures
    if not include_surface_A and roles is not None:
        a_site_tolerance = 0.2 * layer_height
        for i in range(len(roles)):
            if roles[i] == SITE_A:
                z = z_normalized[i]
                # Bottom surface
                if z < a_site_tolerance:
                    mask[i] = False
                # Top surface
                elif abs(z - z_max_slab) < a_site_tolerance:
                    mask[i] = False
    
    # Extract selected atoms
    new_symbols = [symbols[i] for i in range(len(slab)) if mask[i]]
    new_positions = positions[mask].copy()
    new_roles = list(roles[mask]) if roles is not None else None
    
    # Normalize z
    z_min = new_positions[:, 2].min()
    new_positions[:, 2] -= z_min
    
    # New cell height for slab
    new_cell = cell.copy()
    new_cell[2] = new_cell[2] * n_layers / bulk_nz
    slab_height = n_layers * layer_height
    
    # Add top capping X atoms (missing from bulk due to PBC)
    # Only needed when n_layers == bulk_nz (extracting full z extent)
    # When n_layers < bulk_nz, top caps already exist in the bulk
    if n_layers == bulk_nz:
        # Find B-sites in the top octahedral layer and add X atoms above them
        b_top_z = (n_layers - 0.5) * layer_height
        b_tolerance = 0.3 * layer_height
        top_cap_positions = []
        
        for i, pos in enumerate(new_positions):
            # Check if this is a B-site in the top layer
            is_b_site = new_roles is not None and new_roles[i] == SITE_B
            in_top_layer = abs(pos[2] - b_top_z) < b_tolerance
            
            if is_b_site and in_top_layer:
                # Add top axial X at same (x, y), z = slab_height
                top_cap_positions.append([pos[0], pos[1], slab_height])
        
        # Append top cap atoms
        for cap_pos in top_cap_positions:
            new_symbols.append(TEMPLATE_X)
            new_positions = np.vstack([new_positions, cap_pos])
            if new_roles is not None:
                new_roles.append(SITE_X)
    
    new_slab = Atoms(
        symbols=new_symbols,
        positions=new_positions,
        cell=new_cell,
        pbc=True,
    )
    
    if new_roles is not None:
        new_slab.arrays[SITE_ROLE_KEY] = np.array(new_roles, dtype=object)
    
    # Preserve template info
    if 'BX_dist' in bulk.info:
        new_slab.info['BX_dist'] = bulk.info['BX_dist']
    new_slab.info['n_layers'] = n_layers
    
    # Apply reduction if requested
    if reduced:
        # Measure actual B-X distance from the structure (may have changed due to Glazer)
        BX_dist = new_slab.info.get('BX_dist')
        if BX_dist is None:
            # Estimate from structure
            BX_dist = np.mean(cell_lengths[:2]) / 2.0
        new_slab = reduce_slab_cell(new_slab, BX_dist)
    
    return new_slab


def is_template(atoms: Atoms) -> bool:
    """Check if a structure is a template (uses placeholder symbols)."""
    symbols = set(atoms.get_chemical_symbols())
    template_symbols = {TEMPLATE_A, TEMPLATE_B, TEMPLATE_X}
    return bool(symbols & template_symbols)


def get_site_indices(atoms: Atoms, site_type: str) -> np.ndarray:
    """
    Get indices of atoms with specified site type.
    
    Parameters
    ----------
    atoms : Atoms
        Structure with site_role tags
    site_type : str
        One of SITE_A, SITE_B, SITE_X, SITE_SPACER
        
    Returns
    -------
    np.ndarray
        Indices of matching atoms
    """
    roles = atoms.arrays.get(SITE_ROLE_KEY)
    if roles is None:
        return np.array([], dtype=int)
    return np.where(roles == site_type)[0]


def reduce_slab_cell(
    slab: Atoms,
    BX_dist: Optional[float] = None,
) -> Atoms:
    """
    Convert a complete (4-octahedra) slab to a reduced (2-octahedra) cell.
    
    This function rotates the slab 45° around z-axis, cuts to keep 2 octahedra
    (center + corner), and adjusts the cell so the diagonal equals the B-X distance.
    The B-X distance is measured from the structure (after potential Glazer tilting).
    
    Parameters
    ----------
    slab : Atoms
        Complete (4-octahedra) slab template
    BX_dist : float, optional
        B-X bond distance. If None, will be estimated from the structure.
        After Glazer tilting, this may differ from the input BX_dist.
        
    Returns
    -------
    Atoms
        Reduced (2-octahedra) slab with cell diagonal = BX_dist
    """
    reduced = slab.copy()
    positions = reduced.get_positions()
    cell = reduced.get_cell()
    roles = reduced.arrays.get(SITE_ROLE_KEY)
    
    # Get or estimate BX_dist from structure
    if BX_dist is None:
        if 'BX_dist' in reduced.info:
            BX_dist = reduced.info['BX_dist']
        else:
            # Estimate from cell dimensions (for cubic: a0 = 2*BX_dist)
            cell_lengths = np.linalg.norm(cell, axis=1)
            BX_dist = np.mean(cell_lengths[:2]) / 2.0
    
    # Get cell center for rotation
    cell_lengths = np.linalg.norm(cell, axis=1)
    cell_center = np.array([cell_lengths[0] / 2.0, cell_lengths[1] / 2.0, 0.0])
    
    # Rotate 45° around z-axis at cell center
    angle_rad = np.pi / 4.0  # 45 degrees
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rotation matrix around z-axis
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    # Translate to origin, rotate, translate back
    positions_centered = positions.copy()
    positions_centered[:, :2] -= cell_center[:2]
    positions_rotated = positions_centered.copy()
    positions_rotated[:, :2] = positions_centered[:, :2] @ R[:2, :2].T
    positions_rotated[:, :2] += cell_center[:2]
    
    # Identify B-sites to determine which octahedra to keep
    # In the original cell, we have 4 octahedra at:
    # - (0.5*a0, 0.5*a0) - center
    # - (1.5*a0, 0.5*a0) - right
    # - (0.5*a0, 1.5*a0) - top
    # - (1.5*a0, 1.5*a0) - corner
    # After rotation, we want to keep center and one corner
    a0 = 2 * BX_dist
    original_cell_size = cell_lengths[0]
    
    # Find B-sites and their positions
    if roles is not None:
        b_indices = np.where(roles == SITE_B)[0]
    else:
        # Fallback: find Ne atoms (template B)
        symbols = reduced.get_chemical_symbols()
        b_indices = np.where(np.array(symbols) == TEMPLATE_B)[0]
    
    if len(b_indices) == 0:
        raise ValueError("No B-sites found in slab")
    
    b_positions = positions_rotated[b_indices]
    
    # Determine which octahedra to keep
    # After rotation, the center octahedron should be near the cell center
    # We'll keep octahedra within a certain region
    center_tolerance = 0.3 * original_cell_size
    
    # Keep atoms associated with octahedra near center (0, 0) and one corner
    # The corner after rotation will be at approximately (sqrt(2)*a0/2, sqrt(2)*a0/2)
    keep_mask = np.zeros(len(positions_rotated), dtype=bool)
    
    # Find B-sites in the region we want to keep
    # Center region: near (cell_center_x, cell_center_y)
    # Corner region: one of the rotated corners
    center_b_mask = np.zeros(len(b_indices), dtype=bool)
    for i, b_pos in enumerate(b_positions):
        dist_from_center = np.linalg.norm(b_pos[:2] - cell_center[:2])
        # Keep if near center or near one of the rotated corner positions
        if dist_from_center < center_tolerance:
            center_b_mask[i] = True
        else:
            # Check if this is a corner octahedron (diagonally from center)
            # After rotation, corners are at specific positions
            corner_positions = [
                cell_center[:2] + np.array([a0/np.sqrt(2), a0/np.sqrt(2)]),
                cell_center[:2] + np.array([-a0/np.sqrt(2), a0/np.sqrt(2)]),
                cell_center[:2] + np.array([a0/np.sqrt(2), -a0/np.sqrt(2)]),
                cell_center[:2] + np.array([-a0/np.sqrt(2), -a0/np.sqrt(2)]),
            ]
            for corner_pos in corner_positions:
                if np.linalg.norm(b_pos[:2] - corner_pos) < center_tolerance:
                    center_b_mask[i] = True
                    break
    
    # Keep only one corner octahedron (the first one found)
    corner_found = False
    for i, b_pos in enumerate(b_positions):
        if not center_b_mask[i]:
            dist_from_center = np.linalg.norm(b_pos[:2] - cell_center[:2])
            if dist_from_center > center_tolerance and not corner_found:
                # This is a corner octahedron
                center_b_mask[i] = True
                corner_found = True
    
    # Now find all atoms belonging to the kept octahedra
    # For each kept B-site, find its associated X and A atoms
    kept_b_positions = b_positions[center_b_mask]
    
    # For each atom, check if it's associated with a kept octahedron
    octahedron_radius = BX_dist * 1.5  # Approximate octahedron size
    
    for i in range(len(positions_rotated)):
        pos = positions_rotated[i]
        
        # Check if this atom is near any kept B-site
        for kept_b_pos in kept_b_positions:
            dist = np.linalg.norm(pos - kept_b_pos)
            if dist < octahedron_radius:
                keep_mask[i] = True
                break
    
    # Extract kept atoms
    new_symbols = [reduced.get_chemical_symbols()[i] for i in range(len(reduced)) if keep_mask[i]]
    new_positions = positions_rotated[keep_mask].copy()
    new_roles = list(roles[keep_mask]) if roles is not None else None
    
    # Calculate new cell size: diagonal = BX_dist, so side = BX_dist / sqrt(2)
    new_cell_size = BX_dist / np.sqrt(2.0)
    cell_z = cell_lengths[2]
    
    # Translate positions to fit new cell (center at origin)
    # Find the bounding box of kept positions
    if len(new_positions) > 0:
        pos_min = new_positions[:, :2].min(axis=0)
        pos_max = new_positions[:, :2].max(axis=0)
        pos_center = (pos_min + pos_max) / 2.0
        
        # Center the structure
        new_positions[:, :2] -= pos_center
        new_positions[:, :2] += new_cell_size / 2.0
    
    # Create new cell
    new_cell = np.array([
        [new_cell_size, 0.0, 0.0],
        [0.0, new_cell_size, 0.0],
        [0.0, 0.0, cell_z]
    ])
    
    reduced_atoms = Atoms(
        symbols=new_symbols,
        positions=new_positions,
        cell=new_cell,
        pbc=reduced.pbc
    )
    
    if new_roles is not None:
        reduced_atoms.arrays[SITE_ROLE_KEY] = np.array(new_roles, dtype=object)
    
    # Preserve metadata
    for key, value in reduced.info.items():
        reduced_atoms.info[key] = value
    reduced_atoms.info['reduced_cell'] = True
    reduced_atoms.info['BX_dist'] = BX_dist
    
    return reduced_atoms


def apply_lateral_shift(
    atoms: Atoms,
    shift: Tuple[float, float],
    BX_dist: float,
) -> Atoms:
    """
    Apply a lateral shift in units of the unit cell.
    
    Parameters
    ----------
    atoms : Atoms
        Structure to shift
    shift : tuple
        Fractional shift (e.g., (0.5, 0.5) for RP)
    BX_dist : float
        B-X distance (unit cell = 2 * BX_dist)
        
    Returns
    -------
    Atoms
        Shifted structure
    """
    if shift == (0.0, 0.0):
        return atoms
    
    a0 = 2 * BX_dist
    shifted = atoms.copy()
    
    dx = shift[0] * a0
    dy = shift[1] * a0
    
    positions = shifted.get_positions()
    positions[:, 0] += dx
    positions[:, 1] += dy
    shifted.set_positions(positions)
    
    return shifted


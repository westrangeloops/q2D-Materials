"""
2D slab structure operations.

This module handles 2D perovskite phase construction:
- Monolayer (isolated slab with vacuum)
- DJ (Dion-Jacobson, no lateral shift)
- RP (Ruddlesden-Popper, (0.5, 0.5) shift)

All functions work on templates first, then spacers/population are added.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from ase import Atoms

from .template import (
    SITE_ROLE_KEY, SITE_A, SITE_B, SITE_X,
    build_slab_template, extract_slab_from_bulk, apply_lateral_shift,
)


def build_monolayer(
    slab: Atoms,
    vacuum: float = 15.0,
) -> Atoms:
    """
    Build a monolayer structure with vacuum.
    
    Parameters
    ----------
    slab : Atoms
        Slab template (from build_slab_template or extract_slab_from_bulk)
    vacuum : float
        Vacuum thickness on each side in Angstroms
        
    Returns
    -------
    Atoms
        Monolayer with vacuum
    """
    result = slab.copy()
    positions = result.get_positions()
    cell = result.get_cell()
    
    # Normalize to start at z=0
    z_min = positions[:, 2].min()
    positions[:, 2] -= z_min
    result.set_positions(positions)
    
    # Get slab height
    slab_height = positions[:, 2].max()
    
    # Add vacuum
    positions[:, 2] += vacuum
    result.set_positions(positions)
    
    # Set new cell
    cell_a = np.linalg.norm(cell[0])
    cell_b = np.linalg.norm(cell[1])
    total_height = slab_height + 2 * vacuum
    result.set_cell([cell_a, cell_b, total_height])
    
    result.pbc = True
    result.info['structure_type'] = 'monolayer'
    result.info['vacuum'] = vacuum
    
    return result


def build_dj(
    slab: Atoms,
    interlayer_height: float,
) -> Atoms:
    """
    Build a Dion-Jacobson phase structure.
    
    DJ structure: Single slab per unit cell, no lateral shift.
    The spacer bridges to the periodic image.
    
    Parameters
    ----------
    slab : Atoms
        Slab template
    interlayer_height : float
        Total height of interlayer region (spacer region)
        
    Returns
    -------
    Atoms
        DJ structure
    """
    result = slab.copy()
    positions = result.get_positions()
    cell = result.get_cell()
    
    # Normalize to start at z=0
    z_min = positions[:, 2].min()
    positions[:, 2] -= z_min
    result.set_positions(positions)
    
    # Get slab height
    slab_height = positions[:, 2].max()
    
    # Cell height = slab + interlayer
    cell_a = np.linalg.norm(cell[0])
    cell_b = np.linalg.norm(cell[1])
    total_height = slab_height + interlayer_height
    result.set_cell([cell_a, cell_b, total_height])
    
    result.pbc = True
    result.info['structure_type'] = 'dj'
    result.info['interlayer_height'] = interlayer_height
    
    return result


def build_rp(
    slab: Atoms,
    BX_dist: float,
    spacer_gap: float,
    is_atomic_spacer: bool = True,
) -> Atoms:
    """
    Build a Ruddlesden-Popper phase structure.
    
    RP structure: Two slabs per unit cell, shifted by (0.5, 0.5).
    
    For atomic spacers:
        - Gap between slabs = BX_dist (spacer at A-site positions)
        - Cell z = slab + BX + slab + BX
    
    For molecular spacers:
        - Gap between slabs = spacer_gap (typically 2 Å between molecule tails)
        - Cell z = decorated_slab + gap + decorated_slab + gap
    
    Parameters
    ----------
    slab : Atoms
        Slab template
    BX_dist : float
        B-X distance for calculating shift
    spacer_gap : float
        Gap between slabs. For atomic: BX_dist. For molecular: 2 Å.
    is_atomic_spacer : bool
        Whether using atomic or molecular spacer
        
    Returns
    -------
    Atoms
        RP structure (just the slabs, spacers added later)
    """
    from .molecule_builder import add_atoms
    
    cell = slab.get_cell()
    cell_a = np.linalg.norm(cell[0])
    cell_b = np.linalg.norm(cell[1])
    
    # Create slab 1 (normalized to z=0)
    slab1 = slab.copy()
    positions1 = slab1.get_positions()
    z_min = positions1[:, 2].min()
    positions1[:, 2] -= z_min
    slab1.set_positions(positions1)
    
    slab_height = positions1[:, 2].max()
    
    # Create slab 2 with lateral shift (0.5, 0.5)
    slab2 = apply_lateral_shift(slab1.copy(), (0.5, 0.5), BX_dist)
    
    # Position slab 2 above slab 1
    # Gap = spacer_gap (BX for atomic, 2Å for molecular)
    positions2 = slab2.get_positions()
    z_offset = slab_height + spacer_gap
    positions2[:, 2] += z_offset
    slab2.set_positions(positions2)
    
    # Combine slabs
    result = add_atoms(slab1, slab2)
    
    # Cell height = slab + gap + slab + gap
    total_height = 2 * slab_height + 2 * spacer_gap
    result.set_cell([cell_a, cell_b, total_height])
    
    result.pbc = True
    result.info['structure_type'] = 'rp'
    result.info['spacer_gap'] = spacer_gap
    result.info['slab_height'] = slab_height
    result.info['is_atomic_spacer'] = is_atomic_spacer
    
    return result


def get_surface_z(
    slab: Atoms,
    surface: str = "top",
) -> float:
    """
    Get the z-coordinate of surface X-sites (halogens).
    
    Parameters
    ----------
    slab : Atoms
        Slab structure
    surface : str
        'top' or 'bottom'
        
    Returns
    -------
    float
        Z coordinate of surface halogens
    """
    roles = slab.arrays.get(SITE_ROLE_KEY)
    positions = slab.get_positions()
    
    if roles is not None:
        x_mask = roles == SITE_X
        if x_mask.any():
            x_z = positions[x_mask, 2]
            return x_z.max() if surface == "top" else x_z.min()
    
    # Fallback
    return positions[:, 2].max() if surface == "top" else positions[:, 2].min()


def get_slab_height(slab: Atoms) -> float:
    """Get the height of a slab (max z - min z)."""
    positions = slab.get_positions()
    return positions[:, 2].max() - positions[:, 2].min()


def normalize_slab_z(slab: Atoms) -> Atoms:
    """Shift slab so minimum z is 0."""
    result = slab.copy()
    positions = result.get_positions()
    z_min = positions[:, 2].min()
    positions[:, 2] -= z_min
    result.set_positions(positions)
    return result


# =============================================================================
# Structure-specific spacer placement
# =============================================================================

def add_spacers_to_monolayer(
    structure: Atoms,
    spacer: Atoms,
    BX_dist: float,
    nx: int,
    ny: int,
    penetration: Optional[float] = None,
) -> Atoms:
    """
    Add spacers to a monolayer structure (both surfaces, pointing outward).
    
    Parameters
    ----------
    structure : Atoms
        Monolayer structure from build_monolayer()
    spacer : Atoms
        Spacer molecule or atom
    BX_dist : float
        B-X distance
    nx, ny : int
        Number of unit cells in x and y
    penetration : float, optional
        Distance from surface X to spacer attachment.
        Default: -1 Å for molecules, +ionic_radius for atoms.
        
    Returns
    -------
    Atoms
        Structure with spacers on both surfaces
    """
    from .spacer import prepare_spacer, place_spacers_grid
    
    spacer_len, default_penet, spacer_atoms = prepare_spacer(spacer, BX_dist)
    is_atomic = len(spacer) == 1
    penet = penetration if penetration is not None else default_penet
    
    # Get cell dimensions
    cell = structure.get_cell()
    cell_a = np.linalg.norm(cell[0])
    cell_b = np.linalg.norm(cell[1])
    ax = cell_a / nx
    ay = cell_b / ny
    
    # Get surface positions
    positions = structure.get_positions()
    z_bottom = positions[:, 2].min()
    z_top = positions[:, 2].max()
    
    result = structure.copy()
    
    # Bottom surface: spacer pointing down
    z_attach_bottom = z_bottom - penet
    result = place_spacers_grid(
        result, spacer_atoms, ax, ay, nx, ny, z_attach_bottom, "down", is_atomic
    )
    
    # Top surface: spacer pointing up
    z_attach_top = z_top + penet
    result = place_spacers_grid(
        result, spacer_atoms, ax, ay, nx, ny, z_attach_top, "up", is_atomic
    )
    
    return result


def add_spacers_to_dj(
    structure: Atoms,
    spacer: Atoms,
    BX_dist: float,
    nx: int,
    ny: int,
    penetration: Optional[float] = None,
) -> Atoms:
    """
    Add spacers to a DJ structure (top surface only, bridging to PBC image).
    
    For atomic spacers: centered in the interlayer gap.
    For molecular spacers: N at penetration distance from top surface.
    
    Parameters
    ----------
    structure : Atoms
        DJ structure from build_dj()
    spacer : Atoms
        Spacer molecule or atom
    BX_dist : float
        B-X distance
    nx, ny : int
        Number of unit cells in x and y
    penetration : float, optional
        Distance from surface X to spacer attachment (molecules only).
        Default: -1 Å for molecules. Ignored for atomic spacers.
        
    Returns
    -------
    Atoms
        Structure with spacers in interlayer region
    """
    from .spacer import prepare_spacer, place_spacers_grid
    
    spacer_len, default_penet, spacer_atoms = prepare_spacer(spacer, BX_dist)
    is_atomic = len(spacer) == 1
    
    # Get cell dimensions
    cell = structure.get_cell()
    cell_a = np.linalg.norm(cell[0])
    cell_b = np.linalg.norm(cell[1])
    ax = cell_a / nx
    ay = cell_b / ny
    
    # Get slab top position
    positions = structure.get_positions()
    z_top = positions[:, 2].max()
    
    # Get interlayer height from structure info
    interlayer_height = structure.info.get('interlayer_height', spacer_len)
    
    # Calculate z position based on spacer type
    if is_atomic:
        # Atomic: center in interlayer gap
        z_attach = z_top + interlayer_height / 2.0
    else:
        # Molecular: N at penetration from surface
        penet = penetration if penetration is not None else default_penet
        z_attach = z_top + penet
    
    result = structure.copy()
    
    # Add spacers at top surface (pointing up, bridging to PBC image above)
    result = place_spacers_grid(
        result, spacer_atoms, ax, ay, nx, ny, z_attach, "up", is_atomic
    )
    
    return result


def add_spacers_to_rp(
    structure: Atoms,
    spacer: Atoms,
    BX_dist: float,
    nx: int,
    ny: int,
    penetration: Optional[float] = None,
) -> Atoms:
    """
    Add spacers to an RP structure.
    
    For atomic spacers:
        - Spacer at A-site positions (same z as surface X atoms)
        - 4 positions: bottom/top of slab 1, bottom/top of slab 2
        - Slab 2 positions are shifted by (0.5, 0.5)
    
    For molecular spacers:
        - Molecule attached at penetration depth from surface X
        - 4 positions: bottom/top of slab 1, bottom/top of slab 2
        - Slab 2 positions are shifted by (0.5, 0.5)
    
    Parameters
    ----------
    structure : Atoms
        RP structure from build_rp()
    spacer : Atoms
        Spacer molecule or atom
    BX_dist : float
        B-X distance
    nx, ny : int
        Number of unit cells in x and y
    penetration : float, optional
        Distance from surface X to spacer attachment.
        
    Returns
    -------
    Atoms
        Structure with spacers
    """
    from .spacer import prepare_spacer, place_spacers_grid
    
    spacer_len, default_penet, spacer_atoms = prepare_spacer(spacer, BX_dist)
    is_atomic = len(spacer) == 1
    penet = penetration if penetration is not None else default_penet
    
    # Get cell dimensions
    cell = structure.get_cell()
    cell_a = np.linalg.norm(cell[0])
    cell_b = np.linalg.norm(cell[1])
    ax = cell_a / nx
    ay = cell_b / ny
    
    # Get geometry from structure info
    slab_height = structure.info.get('slab_height')
    spacer_gap = structure.info.get('spacer_gap')
    
    if slab_height is None or spacer_gap is None:
        positions = structure.get_positions()
        z_min = positions[:, 2].min()
        z_max = positions[:, 2].max()
        cell_z = np.linalg.norm(cell[2])
        spacer_gap = spacer_gap or BX_dist
        slab_height = (cell_z - 2 * spacer_gap) / 2
    
    # Slab positions:
    # Slab 1: z = 0 to slab_height
    # Slab 2: z = slab_height + gap to 2*slab_height + gap
    slab1_bottom = 0.0
    slab1_top = slab_height
    slab2_bottom = slab_height + spacer_gap
    slab2_top = 2 * slab_height + spacer_gap
    
    result = structure.copy()
    
    # Fractional shift for slab 2 positions (0.5, 0.5 in unit cell coordinates)
    shift = (0.5, 0.5)
    
    if is_atomic:
        # Atomic spacers at A-site positions (same z as surface X)
        # Slab 1 has A-sites at (0, 0), Slab 2 (shifted) has A-sites at (0.5, 0.5)
        # Slab 1 bottom (z = 0) - no shift
        result = place_spacers_grid(
            result, spacer_atoms, ax, ay, nx, ny, slab1_bottom, "up", is_atomic
        )
        # Slab 1 top (z = slab_height) - no shift
        result = place_spacers_grid(
            result, spacer_atoms, ax, ay, nx, ny, slab1_top, "up", is_atomic
        )
        # Slab 2 bottom (z = slab_height + gap) - shifted
        result = place_spacers_grid(
            result, spacer_atoms, ax, ay, nx, ny, slab2_bottom, "up", is_atomic, shift=shift
        )
        # Slab 2 top (z = 2*slab_height + gap) - shifted
        result = place_spacers_grid(
            result, spacer_atoms, ax, ay, nx, ny, slab2_top, "up", is_atomic, shift=shift
        )
    else:
        # Molecular spacers with penetration
        # Slab 1 bottom (pointing down) - no shift
        result = place_spacers_grid(
            result, spacer_atoms, ax, ay, nx, ny, slab1_bottom - penet, "down", is_atomic
        )
        # Slab 1 top (pointing up) - no shift
        result = place_spacers_grid(
            result, spacer_atoms, ax, ay, nx, ny, slab1_top + penet, "up", is_atomic
        )
        # Slab 2 bottom (pointing down) - shifted
        result = place_spacers_grid(
            result, spacer_atoms, ax, ay, nx, ny, slab2_bottom - penet, "down", is_atomic, shift=shift
        )
        # Slab 2 top (pointing up) - shifted
        result = place_spacers_grid(
            result, spacer_atoms, ax, ay, nx, ny, slab2_top + penet, "up", is_atomic, shift=shift
        )
    
    return result


def add_spacers_to_aci(
    structure: Atoms,
    spacer: Atoms,
    BX_dist: float,
    nx: int,
    ny: int,
    penetration: Optional[float] = None,
) -> Atoms:
    """
    Add spacers to an ACI structure (similar to RP).
    
    ACI has two slabs per unit cell shifted by (0.5, 0). Spacers go on:
    - Bottom of slab 1 (pointing down)
    - Top of slab 2 (pointing up)
    
    Parameters
    ----------
    structure : Atoms
        ACI structure
    spacer : Atoms
        Spacer molecule or atom
    BX_dist : float
        B-X distance
    nx, ny : int
        Number of unit cells in x and y
    penetration : float, optional
        Distance from surface X to spacer attachment.
        
    Returns
    -------
    Atoms
        Structure with spacers on outer surfaces
    """
    # ACI spacer placement is same as RP
    return add_spacers_to_rp(structure, spacer, BX_dist, nx, ny, penetration)

"""
Planar defect modeling for tilted perovskites.

This module implements the mathematical framework from Beanland (2008) for modeling
domain walls and anti-phase boundaries in tilted perovskite systems. Planar defects
necessitate different local octahedral tilting arrangements than the bulk crystal.

Key concepts:
- Anti-phase boundaries (APBs): Phase shifts in octahedral tilting
- Domain walls (twins): Orientation changes in tilting patterns
- Local symmetry: Defects have different space groups than bulk
"""

from typing import Dict, List, Tuple, Optional, NamedTuple
import numpy as np

from .glazer_notation import parse_glazer_notation, GlazerSystem


class DefectSpecification(NamedTuple):
    """Specification for a planar defect."""
    type: str  # "apb" (anti-phase boundary) or "twin" (domain wall)
    plane: str  # Defect plane, e.g., "001", "011", "111"
    position: float  # Position along defect normal (0.0 to 1.0)
    width: float  # Defect width in Angstroms


class DefectRegion(NamedTuple):
    """Defines a region with modified tilt properties."""
    bounds: Tuple[float, float]  # (z_min, z_max) for defect region
    tilt_pattern: List[str]  # Local tilt pattern in this region
    tilt_angles: List[float]  # Local tilt angles in this region


def _get_local_tilt_system(
    bulk_notation: str,
    defect_type: str,
    defect_plane: str
) -> Optional[GlazerSystem]:
    """
    Get the local tilt system for a defect based on Tables 4-5 from Beanland (2008).

    Parameters
    ----------
    bulk_notation : str
        Bulk Glazer notation (e.g., "a'a'a'")
    defect_type : str
        "apb" or "twin"
    defect_plane : str
        Defect plane (e.g., "001", "011")

    Returns
    -------
    GlazerSystem or None
        Local tilt system at the defect, or None if not tabulated
    """
    # Table 4: Domain walls (twins) in a'a'a' system
    twin_table = {
        ("a'a'a'", "011"): "a0b'b'",  # Imma
        ("a'a'a'", "001"): "a0a0c'",  # I4/mcm
        ("a'a'a'", "011_apb"): "a'a0a0",  # I4/mcm
        ("a'a'a'", "001_apb"): "a'a'a0",  # Imma
    }

    # Table 5: Special defects on {001} planes
    apb_table = {
        ("a'a'a'", "001_displaced"): "a0a0c*",  # P4/mbm
    }

    # Look up in tables
    key = (bulk_notation, f"{defect_plane}")
    if defect_type == "twin":
        local_notation = twin_table.get(key)
    elif defect_type == "apb":
        local_notation = apb_table.get(key)
    else:
        return None

    if local_notation:
        try:
            return parse_glazer_notation(local_notation)
        except ValueError:
            return None

    return None


def create_defect_regions(
    bulk_system: GlazerSystem,
    defect: DefectSpecification,
    supercell_length: float,
    transition_width: float = 2.0,
) -> List[DefectRegion]:
    """
    Create defect regions with appropriate local tilt systems.

    Parameters
    ----------
    bulk_system : GlazerSystem
        Bulk tilt system
    defect : DefectSpecification
        Defect specification
    supercell_length : float
        Length of supercell along defect normal
    transition_width : float
        Width of transition region in Angstroms

    Returns
    -------
    List[DefectRegion]
        Regions with their local tilt properties
    """
    regions = []

    # Get local tilt system for the defect
    local_system = _get_local_tilt_system(
        bulk_system.notation, defect.type, defect.plane
    )

    if local_system is None:
        # If no specific local system found, use bulk system
        local_system = bulk_system

    # Calculate defect position in absolute coordinates
    defect_z = defect.position * supercell_length

    # Define defect region bounds
    defect_min = defect_z - defect.width / 2.0
    defect_max = defect_z + defect.width / 2.0

    # Create transition regions
    transition_min = defect_min - transition_width
    transition_max = defect_max + transition_width

    # Region 1: Bulk (z < transition_min)
    regions.append(DefectRegion(
        bounds=(0.0, transition_min),
        tilt_pattern=bulk_system.tilt_pattern,
        tilt_angles=[10.0, 10.0, 10.0],  # Default bulk angles
    ))

    # Region 2: Transition to defect (transition_min <= z < defect_min)
    # Linear interpolation between bulk and defect tilts
    regions.append(DefectRegion(
        bounds=(transition_min, defect_min),
        tilt_pattern=local_system.tilt_pattern,
        tilt_angles=[5.0, 5.0, 5.0],  # Reduced angles for transition
    ))

    # Region 3: Defect core (defect_min <= z <= defect_max)
    regions.append(DefectRegion(
        bounds=(defect_min, defect_max),
        tilt_pattern=local_system.tilt_pattern,
        tilt_angles=[0.0, 0.0, 0.0],  # No tilting at defect core (Pm3m-like)
    ))

    # Region 4: Transition back to bulk (defect_max < z <= transition_max)
    regions.append(DefectRegion(
        bounds=(defect_max, transition_max),
        tilt_pattern=local_system.tilt_pattern,
        tilt_angles=[5.0, 5.0, 5.0],  # Reduced angles for transition
    ))

    # Region 5: Bulk (z > transition_max)
    regions.append(DefectRegion(
        bounds=(transition_max, supercell_length),
        tilt_pattern=bulk_system.tilt_pattern,
        tilt_angles=[10.0, 10.0, 10.0],  # Default bulk angles
    ))

    return regions


def apply_tilt_with_defect(
    position_matrix: Dict[str, List[List[float]]],
    lattice_vectors: Tuple[float, float, float],
    supercell: Tuple[int, int, int],
    bulk_notation: str,
    defect: DefectSpecification,
    base_angle: float = 10.0,
) -> Tuple[Dict[str, List[List[float]]], Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Apply Glazer tilting with a planar defect.

    This implements the framework from Beanland (2008) where planar defects
    (domain walls, APBs) have different local octahedral tilt arrangements.

    Parameters
    ----------
    position_matrix : dict
        Positions with keys 'A','B','X'
    lattice_vectors : (float, float, float)
        Lattice vector lengths (a, b, c)
    supercell : (int, int, int)
        Supercell dimensions (nx, ny, nz)
    bulk_notation : str
        Bulk Glazer notation (e.g., "a'a'a'")
    defect : DefectSpecification
        Defect specification
    base_angle : float
        Base tilt angle in degrees

    Returns
    -------
    (positions, lv_unit, cell_lengths)
        Modified positions and lattice parameters
    """
    from .glazer_tilting import apply_glazer_tilt

    # Parse bulk system
    bulk_system = parse_glazer_notation(bulk_notation)

    # For now, implement simplified defect modeling
    # In a full implementation, we would need to:
    # 1. Divide the supercell into regions along the defect normal
    # 2. Apply different tilt patterns/angles in each region
    # 3. Handle continuity at region boundaries

    # Simplified approach: Apply bulk tilting but modify angles near defect
    # This is a placeholder for the full defect modeling

    supercell_length = lattice_vectors[2] * supercell[2]  # Assume defect normal is z
    regions = create_defect_regions(bulk_system, defect, supercell_length)

    # For now, just apply the bulk tilting
    # Full implementation would require spatially varying tilt application
    angles = [base_angle if phase != "0" else 0.0 for phase in bulk_system.tilt_pattern]

    return apply_glazer_tilt(
        position_matrix=position_matrix,
        lattice_vectors=lattice_vectors,
        supercell=supercell,
        angles=angles,
        tilt_pattern=bulk_system.tilt_pattern,
        adjust_cell=True,
    )

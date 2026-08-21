"""Planar defect modeling for tilted perovskites."""

from typing import Dict, List, Tuple, Optional, NamedTuple
import numpy as np

from .glazer_notation import parse_glazer_notation, GlazerSystem


class DefectSpecification(NamedTuple):
    """Specification for a planar defect."""
    type: str
    plane: str
    position: float
    width: float


class DefectRegion(NamedTuple):
    """Defines a region with modified tilt properties."""
    bounds: Tuple[float, float]
    tilt_pattern: List[str]
    tilt_angles: List[float]


def _get_local_tilt_system(
    bulk_notation: str,
    defect_type: str,
    defect_plane: str
) -> Optional[GlazerSystem]:
    """Get the local tilt system for a defect."""
    twin_table = {
        ("a'a'a'", "011"): "a0b'b'",  # Imma
        ("a'a'a'", "001"): "a0a0c'",  # I4/mcm
        ("a'a'a'", "011_apb"): "a'a0a0",  # I4/mcm
        ("a'a'a'", "001_apb"): "a'a'a0",  # Imma
    }

    apb_table = {
        ("a'a'a'", "001_displaced"): "a0a0c*",  # P4/mbm
    }

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
    """Create defect regions with appropriate local tilt systems."""
    regions = []
    local_system = _get_local_tilt_system(
        bulk_system.notation, defect.type, defect.plane
    )
    if local_system is None:
        local_system = bulk_system

    defect_z = defect.position * supercell_length
    defect_min = defect_z - defect.width / 2.0
    defect_max = defect_z + defect.width / 2.0
    transition_min = defect_min - transition_width
    transition_max = defect_max + transition_width

    regions.append(DefectRegion(
        bounds=(0.0, transition_min),
        tilt_pattern=bulk_system.tilt_pattern,
        tilt_angles=[10.0, 10.0, 10.0],
    ))
    regions.append(DefectRegion(
        bounds=(transition_min, defect_min),
        tilt_pattern=local_system.tilt_pattern,
        tilt_angles=[5.0, 5.0, 5.0],
    ))
    regions.append(DefectRegion(
        bounds=(defect_min, defect_max),
        tilt_pattern=local_system.tilt_pattern,
        tilt_angles=[0.0, 0.0, 0.0],
    ))
    regions.append(DefectRegion(
        bounds=(defect_max, transition_max),
        tilt_pattern=local_system.tilt_pattern,
        tilt_angles=[5.0, 5.0, 5.0],
    ))
    regions.append(DefectRegion(
        bounds=(transition_max, supercell_length),
        tilt_pattern=bulk_system.tilt_pattern,
        tilt_angles=[10.0, 10.0, 10.0],
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
    """Apply Glazer tilting with a planar defect."""
    from .glazer_tilting import apply_glazer_tilt

    bulk_system = parse_glazer_notation(bulk_notation)
    supercell_length = lattice_vectors[2] * supercell[2]
    regions = create_defect_regions(bulk_system, defect, supercell_length)
    angles = [base_angle if phase != "0" else 0.0 for phase in bulk_system.tilt_pattern]

    return apply_glazer_tilt(
        position_matrix=position_matrix,
        lattice_vectors=lattice_vectors,
        supercell=supercell,
        angles=angles,
        tilt_pattern=bulk_system.tilt_pattern,
        adjust_cell=True,
    )

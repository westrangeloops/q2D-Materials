"""
Population module for populating structure matrices with atoms.

This module handles ion assignment, molecular alignment, and spacer attachment
to convert abstract position matrices into complete ASE Atoms objects.
"""

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from typing import Union, List, Tuple, Dict, Optional, Any

from .q_builder import QBuilderOutput
from .templates import FloorSchema, flatten_floor_schema
from .molecule_builder import (
    align_ase_molecule_for_perovskite,
    center_of_mass_correction,
    place_atoms_at_location,
    add_atoms,
    end_to_origin,
    translate_atoms,
    get_molecule_length,
    com_to_origin
)
from ..utils.A_sites import get_ionic_radius, is_molecular_a_cation, get_a_site_object
from .optimizers import find_optimal_spacer_vectors_global
from .spacer import count_nh3_groups, SpacerMolecule, _find_terminal_nitrogens
from .collision import resolve_collisions


def place_spacer_at_location(atoms, r, attachment_end):
    """
    Place a spacer molecule with its NH3+ N atom at the location r.

    Parameters
    ----------
    atoms : Atoms
        Spacer molecule (already aligned with NH3+ at correct end)
    r : array
        Vector for the translation (3,)
    attachment_end : str
        'top' or 'bottom' - which end of molecule contains the NH3+ group

    Returns
    -------
    mod_atoms : Atoms
        The modified atoms object with NH3+ N atom at position r.
    """
    mod_atoms = atoms.copy()
    spacer = SpacerMolecule.from_atoms(mod_atoms)

    nh3_n_idx = spacer.primary_nh3_index

    if nh3_n_idx is not None:
        n_pos = mod_atoms.get_positions()[nh3_n_idx]
        translation = np.array(r) - n_pos
        mod_atoms.positions += translation
    else:
        # Fallback: use COM placement
        mod_atoms = place_atoms_at_location(mod_atoms, r)

    return mod_atoms


def _normalize_ion_list(
    ions: Union[str, Atoms, List[Union[str, Atoms]], None],
    normalize_func,
    copy_atoms: bool = False
) -> Union[str, Atoms, List[Union[str, Atoms]], None]:
    """
    Generic helper to normalize ion lists (single values or lists).

    Handles the common pattern of checking if input is a list, and if so,
    normalizing each element. If not a list, normalizes the single value.

    Parameters
    ----------
    ions : str, Atoms, list, or None
        Input ions (single or list)
    normalize_func : callable
        Function to normalize string inputs (e.g., normalize_a_site or normalize_spacer)
    copy_atoms : bool
        Whether to copy Atoms objects (needed for sharp_spacer)

    Returns
    -------
    str, Atoms, list, or None
        Normalized ions in same structure as input
    """
    if ions is None:
        return None

    if isinstance(ions, list):
        normalized = []
        for ion in ions:
            if isinstance(ion, Atoms):
                normalized.append(ion.copy() if copy_atoms else ion)
            elif isinstance(ion, str):
                normalized.append(normalize_func(ion))
            else:
                normalized.append(normalize_func(ion))
        return normalized
    else:
        # Single value
        if isinstance(ions, Atoms):
            return ions.copy() if copy_atoms else ions
        elif isinstance(ions, str):
            return normalize_func(ions)
        else:
            return normalize_func(ions)


def _get_next_ion_from_list(ions, site_type: str, site_counters: Dict[str, int]):
    """
    Get the next ion from a list (cycling through) or single value, with Atoms copying.

    Handles the common pattern of cycling through ion lists for assignment.
    If the list is empty, returns None.

    Parameters
    ----------
    ions : str, Atoms, or list
        Input ions (single value or list to cycle through)
    site_type : str
        Site type for counter tracking (e.g., 'A', 'B', 'X')
    site_counters : dict
        Dictionary to track counters for each site type

    Returns
    -------
    str or Atoms or None
        Next ion to assign, with Atoms objects copied. Returns None if list is empty.
    """
    if isinstance(ions, list):
        if not ions:
            return None
        if site_type not in site_counters:
            site_counters[site_type] = 0
        ion = ions[site_counters[site_type] % len(ions)]
        site_counters[site_type] += 1
    else:
        ion = ions

    # Always copy Atoms objects to avoid shared references
    if isinstance(ion, Atoms):
        ion = ion.copy()

    return ion


def normalize_a_site(A: Union[str, Atoms]) -> Union[str, Atoms]:
    """
    Normalize A-site input to handle both strings and Atoms objects.
    Converts molecular cation strings (MA, FA, etc.) to Atoms objects.

    Parameters
    ----------
    A : str or Atoms
        A-site cation (string like "MA", "FA", "Cs" or Atoms object)

    Returns
    -------
    str or Atoms
        Atomic cations as strings, molecular cations as Atoms objects
    """
    from q2D_Materials.pipeline.common import _normalize_to_atoms_or_string
    return _normalize_to_atoms_or_string(A, return_atoms_only=False)


def apply_penetration_offsets(
    positions: Dict[str, np.ndarray],
    penetration: float | List[float] = 0.0,
    BX_dist: float | None = None,
) -> Dict[str, np.ndarray]:
    """
    Shift Ap (per-site, cycling) and S# (uniform) z by ± BX_dist * penetration.

    Bottom Ap/S# shift downward; top Ap/S# shift upward. Sites not at min/max z
    are left unchanged.
    """
    if penetration == 0.0:
        return positions
    if BX_dist is None:
        return positions

    if isinstance(penetration, (list, tuple, np.ndarray)):
        pen_list = [float(x) for x in np.asarray(penetration, dtype=float).flatten().tolist()]
        if not pen_list:
            return positions
    else:
        pen_list = [float(penetration)]

    all_z: List[float] = []
    for site, coords in positions.items():
        if site == "Ap" or (site.startswith("S") and len(site) > 1 and site[1:].isdigit()):
            if len(coords) > 0:
                all_z.extend(np.array(coords, dtype=float)[:, 2].tolist())
    if not all_z:
        return positions
    z_min = min(all_z)
    z_max = max(all_z)
    tol = 1e-6

    out: Dict[str, np.ndarray] = {}
    for site, coords in positions.items():
        arr = np.array(coords, dtype=float, copy=True)
        if site == "Ap":
            for i in range(arr.shape[0]):
                mag = pen_list[i % len(pen_list)] * BX_dist
                if arr[i, 2] <= z_min + tol:
                    arr[i, 2] -= mag
                elif arr[i, 2] >= z_max - tol:
                    arr[i, 2] += mag
        elif site.startswith("S") and len(site) > 1 and site[1:].isdigit():
            mag = pen_list[0] * BX_dist
            for i in range(arr.shape[0]):
                if arr[i, 2] <= z_min + tol:
                    arr[i, 2] -= mag
                elif arr[i, 2] >= z_max - tol:
                    arr[i, 2] += mag
        out[site] = arr
    return out


def populate_from_floor_schema(
    schema: FloorSchema,
    A_ions: Union[str, Atoms, List],
    B_ions: Union[str, List],
    X_ions: Union[str, List],
    Ap_ions: Optional[Union[str, Atoms, List]] = None,
    sharp_spacer: Optional[List[Union[str, Atoms]]] = None,
    penetration: float = 0.0,
    BX_dist: float | None = None,
    spacer_orientation: Optional[List[str]] = None,
) -> Atoms:
    """
    Populate directly from a floor schema (numbered floors with cartesian coords).
    """
    positions, site_labels = flatten_floor_schema(schema)
    positions = apply_penetration_offsets(positions, penetration, BX_dist=BX_dist)
    lattice_vec_sizes = np.linalg.norm(schema.cell, axis=1)
    matrix = QBuilderOutput(
        positions=positions,
        lattice_vector_sizes=lattice_vec_sizes,
        cell_vectors=schema.cell,
    )
    return populate_structure(
        matrix=matrix,
        A_ions=A_ions,
        B_ions=B_ions,
        X_ions=X_ions,
        Ap_ions=Ap_ions,
        sharp_spacer=sharp_spacer,
        site_labels=site_labels,
        floors_cart=list(schema.floors.values()),
        BX_dist=BX_dist,
        spacer_orientation=spacer_orientation,
    )


def assign_ions_to_sites(
    position_template: Dict[str, np.ndarray],
    A_ions: Union[str, Atoms, List],
    B_ions: Union[str, List],
    X_ions: Union[str, List],
    Ap_ions: Optional[Union[str, Atoms, List]] = None,
    sharp_spacer: Optional[List[Union[str, Atoms]]] = None,
    site_labels: Optional[Dict[str, List[str]]] = None,
) -> List[Tuple[str, Union[str, Atoms], np.ndarray, str]]:
    """
    Assign ions to positions using explicit patterns with site label metadata.
    
    This function assigns ions to positions based on explicit patterns provided by the user.
    If a single ion is provided, it's used for all positions of that type.
    If a list is provided, ions are assigned sequentially, cycling if the list is shorter.
    Processes positions in layer-by-layer order (sorted by z-coordinate).
    Maintains Ap override: Ap positions take precedence over A positions.
    
    Parameters
    ----------
    position_template : dict
        Dictionary with site types ('A', 'B', 'X', 'Ap', 'S1', 'S2', etc.) and numpy arrays of positions
    A_ions : str/Atoms or list[str/Atoms]
        A-site ion(s). Single value or list pattern.
    B_ions : str or list[str]
        B-site ion(s). Single value or list pattern.
    X_ions : str or list[str]
        X-site ion(s). Single value or list pattern.
    Ap_ions : optional
        Ap-site spacer(s). Single value or list pattern.
    sharp_spacer : optional list
        Spacer molecules for S# sites (double or mono). List cycles through S# labels.
    site_labels : optional dict
        Dictionary mapping site types to lists of labels (e.g., {"S1": ["S1", "S1", ...], "A": ["A", "A", ...]})
        
    Returns
    -------
    list[tuple]
        List of (site_type, ion, position, label) tuples where:
        - site_type is the site type ('A', 'B', 'X', 'Ap', 'S1', etc.)
        - ion is str or Atoms object
        - position is numpy array [x, y, z]
        - label is the site label (e.g., "S1", "S2", "A")
    """
    assignments = []
    
    # Collect all positions with their labels and z-coordinates for layer-by-layer processing
    all_positions_with_labels = []
    
    # Process all site types including S# sites
    for site_type, positions in position_template.items():
        if len(positions) == 0:
            continue
        
        # Get labels for this site type
        labels = site_labels.get(site_type, [site_type] * len(positions)) if site_labels else [site_type] * len(positions)
        
        for i, pos in enumerate(positions):
            label = labels[i] if i < len(labels) else site_type
            all_positions_with_labels.append((site_type, pos, label))
    
    # Separate S# sites from other positions - S# will be processed per floor pair later
    other_positions = []
    
    for site_type, pos, label in all_positions_with_labels:
        if site_type.startswith('S') and len(site_type) > 1 and site_type[1:].isdigit():
            # Skip S# sites here - they will be processed on-the-fly per floor pair
            # Store them with a placeholder to maintain position information
            assignments.append((site_type, None, pos, label))
        else:
            other_positions.append((site_type, pos, label))
    
    # Sort other positions by z-coordinate for layer-by-layer processing
    other_positions.sort(key=lambda x: x[1][2])
    
    # Track counters for each site type for cycling through lists
    site_counters = {}
    
    # Process other positions in layer order
    for site_type, pos, label in other_positions:
        # Skip Ap positions - they will be handled separately after A positions
        if site_type == 'Ap':
            continue
        
        # Handle A sites (but Ap will override later)
        if site_type == 'A':
            ion = _get_next_ion_from_list(A_ions, 'A', site_counters)
            assignments.append((site_type, ion, pos, label))
            continue

        # Handle B sites
        if site_type == 'B':
            ion = _get_next_ion_from_list(B_ions, 'B', site_counters)
            assignments.append((site_type, ion, pos, label))
            continue

        # Handle X sites
        if site_type == 'X':
            ion = _get_next_ion_from_list(X_ions, 'X', site_counters)
            assignments.append((site_type, ion, pos, label))
            continue
    
    # Now handle Ap sites (they override A sites)
    Ap_positions = position_template.get('Ap', np.array([]).reshape(0, 3))
    if Ap_ions is not None and len(Ap_positions) > 0:
        Ap_labels = site_labels.get('Ap', ['Ap'] * len(Ap_positions)) if site_labels else ['Ap'] * len(Ap_positions)
        Ap_positions_with_labels = list(zip(Ap_positions, Ap_labels))
        Ap_positions_with_labels.sort(key=lambda x: x[0][2])  # Sort by z
        
        if isinstance(Ap_ions, list):
            if 'Ap' not in site_counters:
                site_counters['Ap'] = 0
            for pos, label in Ap_positions_with_labels:
                ion = Ap_ions[site_counters['Ap'] % len(Ap_ions)]
                site_counters['Ap'] += 1
                if isinstance(ion, str) and ion.upper() == "HOLE":
                    continue  # Skip this position - creates a hole
                if isinstance(ion, Atoms):
                    ion = ion.copy()
                assignments.append(('Ap', ion, pos, label))
        else:
            # Single spacer value - check if it's "HOLE"
            if isinstance(Ap_ions, str) and Ap_ions.upper() == "HOLE":
                # Skip all positions - creates holes everywhere
                pass
            else:
                for pos, label in Ap_positions_with_labels:
                    ion = Ap_ions
                    if isinstance(ion, Atoms):
                        ion = ion.copy()
                    assignments.append(('Ap', ion, pos, label))
    
    return assignments


def _place_mono_sharp_spacer(molecule: Atoms, position: np.ndarray, attachment_end: str) -> Optional[Atoms]:
    """Align and place a mono spacer (single NH3 or atomic) at a target position."""
    if not isinstance(molecule, Atoms):
        return None
    aligned = align_ase_molecule_for_perovskite(molecule.copy(), attachment_end=attachment_end)
    return place_spacer_at_location(aligned, position, attachment_end)


def _build_floor_slabs(
    resolved_floors: List[List[Tuple[str, Union[str, Atoms], np.ndarray, str]]]
) -> List[Dict[str, Any]]:
    """Create JSON-style slabs per floor containing structured metadata."""
    slabs: List[Dict[str, Any]] = []
    for entries in resolved_floors:
        if not entries:
            continue
        slab: Dict[str, Any] = {
            "z": float(entries[0][2][2]),
            "entries": [],
            "sites": {},
            "s_labels": {},
        }
        for site_type, ion, pos, label in entries:
            record = {
                "site_type": site_type,
                "ion": ion,
                "position": np.array(pos, dtype=float),
                "label": label,
            }
            slab["entries"].append(record)
            slab.setdefault("sites", {}).setdefault(site_type, []).append(record)
            if site_type.startswith('S') and len(site_type) > 1 and site_type[1:].isdigit():
                slab.setdefault("s_labels", {}).setdefault(label, []).append(record)
        slabs.append(slab)
    return slabs


def _add_atomic_site(structure: Atoms, symbol: str, position: np.ndarray) -> Atoms:
    """Add a single atomic ion to the structure."""
    if symbol is None:
        return structure
    atom = Atoms(symbol, positions=[np.array(position, dtype=float)])
    return add_atoms(structure, atom)


def _place_a_molecule(structure: Atoms, molecule: Atoms, position: np.ndarray) -> Atoms:
    """Align and place a molecular A-site at the requested coordinates."""
    mol_aligned = align_ase_molecule_for_perovskite(molecule.copy())
    mol_placed = place_atoms_at_location(mol_aligned, position)
    return add_atoms(structure, mol_placed)


def _place_ap_molecule(structure: Atoms, molecule: Atoms, position: np.ndarray) -> Atoms:
    """Place an Ap spacer respecting whether it sits on ground or sky."""
    cell = structure.cell if structure.cell is not None else np.eye(3)
    try:
        z_extent = float(cell[2][2])
    except Exception:
        z_extent = float(np.linalg.norm(cell[2]))
    z_center = z_extent / 2.0 if z_extent else 0.0
    attachment_end = 'top' if float(position[2]) < z_center else 'bottom'
    mol_aligned = align_ase_molecule_for_perovskite(molecule.copy(), attachment_end=attachment_end)
    mol_placed = place_spacer_at_location(mol_aligned, position, attachment_end)
    return add_atoms(structure, mol_placed)


def populate_a_sites(structure: Atoms, slab: Dict[str, Any]) -> Atoms:
    """Populate A and Ap sites for a ground slab."""
    for entry in slab.get("sites", {}).get('A', []):
        ion = entry["ion"]
        position = entry["position"]
        if ion is None:
            continue
        if isinstance(ion, Atoms):
            structure = _place_a_molecule(structure, ion, position)
            continue
        if isinstance(ion, str):
            try:
                if is_molecular_a_cation(ion):
                    mol = get_a_site_object(ion)
                    structure = _place_a_molecule(structure, mol, position)
                    continue
            except (ImportError, ValueError):
                pass
        structure = _add_atomic_site(structure, ion, position)

    for entry in slab.get("sites", {}).get('Ap', []):
        ion = entry["ion"]
        position = entry["position"]
        if ion is None:
            continue
        if isinstance(ion, Atoms):
            structure = _place_ap_molecule(structure, ion, position)
            continue
        if isinstance(ion, str):
            try:
                if is_molecular_a_cation(ion):
                    mol = get_a_site_object(ion)
                    structure = _place_ap_molecule(structure, mol, position)
                    continue
            except (ImportError, ValueError):
                pass
        structure = _add_atomic_site(structure, ion, position)

    return structure


def populate_b_sites(structure: Atoms, slab: Dict[str, Any]) -> Atoms:
    """Populate B sites for a ground slab."""
    for entry in slab.get("sites", {}).get('B', []):
        ion = entry["ion"]
        position = entry["position"]
        if ion is None:
            continue
        if isinstance(ion, Atoms):
            structure = add_atoms(structure, place_atoms_at_location(ion.copy(), position))
        else:
            structure = _add_atomic_site(structure, ion, position)
    return structure


def populate_x_sites(structure: Atoms, slab: Dict[str, Any]) -> Atoms:
    """Populate X sites for a ground slab."""
    for entry in slab.get("sites", {}).get('X', []):
        ion = entry["ion"]
        position = entry["position"]
        if ion is None:
            continue
        if isinstance(ion, Atoms):
            structure = add_atoms(structure, place_atoms_at_location(ion.copy(), position))
        else:
            structure = _add_atomic_site(structure, ion, position)
    return structure


def _slab_has_spacer_sites(slab: Dict[str, Any]) -> bool:
    """Return True if the slab contains any S# entries."""
    return bool(slab.get("s_labels"))


def _prepare_sharp_template(template: Union[str, Atoms]) -> Optional[Atoms]:
    """Normalize sharp spacer template to an Atoms object if possible."""
    if isinstance(template, Atoms):
        return template.copy()
    if isinstance(template, str):
        try:
            from q2D_Materials.pipeline.common import normalize_spacer
            normalized = normalize_spacer(template)
            if isinstance(normalized, Atoms):
                return normalized
        except Exception:
            normalized = normalize_a_site(template)
            if isinstance(normalized, Atoms):
                return normalized
            if isinstance(normalized, str):
                try:
                    return Atoms(normalized, positions=[[0.0, 0.0, 0.0]])
                except Exception:
                    return None
    return None


def _sort_spacer_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort S-site entries deterministically for pairing ground and sky."""
    def sort_key(entry):
        # Try to sort by S# number if the label starts with 'S' followed by digits
        label = entry.get("label", "")
        if label.startswith("S") and len(label) > 1 and label[1:].isdigit():
            # Sort by the numeric part of S# labels (S1, S2, S3, etc.)
            return (0, int(label[1:]))
        else:
            # Fallback to position-based sorting for other labels
            return (1, round(float(entry["position"][0]), 4),
                       round(float(entry["position"][1]), 4),
                       round(float(entry["position"][2]), 4))

    return sorted(entries, key=sort_key)


def _adjust_positions_for_double_spacer(
    ground_pos: np.ndarray,
    sky_pos: np.ndarray,
    spacer_template: Atoms,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project ground/sky anchors onto the molecule's intrinsic N-N span.
    
    CRITICAL: Preserves X, Y coordinates of original S# sites exactly.
    The kinematic solver will handle aligning the molecule between the two points,
    so we don't need to adjust positions here - just return them as-is.
    """
    # Simply return original positions - the kinematic solver will handle
    # aligning the molecule between P1 and P2 while respecting bond constraints
    return ground_pos, sky_pos


class _SharpSpacerSequence:
    """Simple cursor that cycles through user-provided sharp spacers across floor pairs."""

    def __init__(self, templates: Optional[List[Union[str, Atoms]]]):
        self.templates = templates or []
        self.index = 0

    def has_templates(self) -> bool:
        return len(self.templates) > 0

    def next_template(self) -> Optional[Union[str, Atoms]]:
        if not self.templates:
            return None
        template = self.templates[self.index % len(self.templates)]
        self.index += 1
        return template


class _SpacerOrientationSequence:
    """Simple cursor that cycles through user-provided spacer orientations across floor pairs."""

    def __init__(self, orientations: Optional[List[str]]):
        self.orientations = orientations or []
        self.index = 0

    def has_orientations(self) -> bool:
        return len(self.orientations) > 0

    def next_orientation(self) -> Optional[str]:
        if not self.orientations:
            return None
        orientation = self.orientations[self.index % len(self.orientations)]
        self.index += 1
        return orientation


def _calculate_xy_pbc_distances(reference_pos: np.ndarray, positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    Calculate PBC-aware distances considering only XY periodic images (9 total: center + 8 neighbors).
    Z coordinate is kept fixed (no wrapping in Z direction).
    
    Parameters
    ----------
    reference_pos : np.ndarray
        Reference position [x, y, z]
    positions : np.ndarray
        Array of positions to calculate distances to, shape (n, 3)
    cell : np.ndarray
        Unit cell matrix (3x3)
        
    Returns
    -------
    np.ndarray
        Array of shortest distances considering XY PBC only, shape (n,)
    """
    reference_pos = np.asarray(reference_pos, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    
    # Calculate base difference (no periodic shift)
    diff_base = positions - reference_pos  # Shape: (n, 3)
    
    # Get XY components and Z component separately
    diff_xy_base = diff_base[:, :2]  # Shape: (n, 2)
    diff_z = diff_base[:, 2]  # Shape: (n,) - keep Z as-is
    
    # Convert to fractional coordinates using full cell (but we'll only use XY)
    try:
        inv_cell = np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        # Fallback to simple distance if cell is singular
        return np.linalg.norm(diff_base, axis=1)
    
    # Convert full difference to fractional
    diff_frac_full = diff_base @ inv_cell.T  # Shape: (n, 3)
    diff_frac_xy = diff_frac_full[:, :2]  # Shape: (n, 2) - only XY fractional coords
    
    # Generate all 9 XY periodic images: (-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)
    shifts = np.array([(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]], dtype=np.float64)  # Shape: (9, 2)
    
    # For each position, check all 9 XY images and find the minimum distance
    n_positions = len(positions)
    min_distances = np.full(n_positions, np.inf)
    
    for shift in shifts:
        # Apply shift to fractional XY coordinates only
        diff_frac_xy_shifted = diff_frac_xy - shift  # Shape: (n, 2)
        
        # Convert back to Cartesian using full cell
        # Create full fractional coords with shifted XY and original Z
        diff_frac_full_shifted = np.column_stack([
            diff_frac_xy_shifted,
            diff_frac_full[:, 2]  # Keep Z fractional coordinate unchanged
        ])  # Shape: (n, 3)
        
        # Convert back to Cartesian
        diff_shifted = diff_frac_full_shifted @ cell  # Shape: (n, 3)
        
        # Calculate distances
        distances = np.linalg.norm(diff_shifted, axis=1)  # Shape: (n,)
        
        # Update minimum distances
        min_distances = np.minimum(min_distances, distances)
    
    return min_distances


def populate_sharp(
    structure: Atoms,
    ground_slab: Dict[str, Any],
    sky_slab: Optional[Dict[str, Any]],
    sharp_sequence: Optional[_SharpSpacerSequence],
    optimizer: str = "KS",
    BX_dist: Optional[float] = None,
    spacer_orientation: Optional[List[str]] = None,
    collision_strategy: str = "rotate",
) -> Atoms:
    """Populate sharp (S#) sites between a ground slab and an optional sky slab."""
    if sharp_sequence is None or not sharp_sequence.has_templates():
        return structure
    if sky_slab is None:
        return structure
    if not _slab_has_spacer_sites(ground_slab) or not _slab_has_spacer_sites(sky_slab):
        return structure

    ground_labels = ground_slab.get("s_labels", {})
    sky_labels = sky_slab.get("s_labels", {})
    shared_labels = sorted(set(ground_labels.keys()) & set(sky_labels.keys()))
    if not shared_labels:
        return structure

    from q2D_Materials.builders.spacer import place_double_spacer_between_positions
    from ase.data import covalent_radii  # Import locally to avoid scoping issues

    # Get cell for PBC-aware calculations
    cell = structure.cell if structure.cell is not None else None

    # Create orientation sequence for cycling through orientations
    orientation_sequence = _SpacerOrientationSequence(spacer_orientation)

    for label in shared_labels:
        template = sharp_sequence.next_template()
        spacer_template = _prepare_sharp_template(template)
        if spacer_template is None:
            continue

        # Apply plane alignment if orientation is specified
        if orientation_sequence.has_orientations() and cell is not None:
            orientation = orientation_sequence.next_orientation()
            if orientation in ["A", "B"]:
                from .molecule_builder import align_molecule_plane
                spacer_template = align_molecule_plane(spacer_template, orientation, cell)

        ground_entries = _sort_spacer_entries(ground_labels.get(label, []))
        sky_entries = _sort_spacer_entries(sky_labels.get(label, []))
        if not ground_entries or not sky_entries:
            continue

        nh3_groups = count_nh3_groups(spacer_template)

        if nh3_groups >= 2:
            # Collect all pairs for deterministic vector selection
            all_starts = []
            all_targets = []
            all_pairs_info = []  # Store (ground_entry, sky_entry, ground_pos, sky_pos) for later
            
            if cell is None:
                # Fallback to simple pairing if no cell
                pair_count = min(len(ground_entries), len(sky_entries))
                for pair_idx in range(pair_count):
                    ground_pos = np.array(ground_entries[pair_idx]["position"], dtype=float)
                    sky_pos = np.array(sky_entries[pair_idx]["position"], dtype=float)
                    all_starts.append(ground_pos)
                    all_targets.append(sky_pos)
                    all_pairs_info.append((ground_entries[pair_idx], sky_entries[pair_idx], ground_pos, sky_pos))
            else:
                # For templates like salts, do index-based pairing if groups have equal size
                if len(ground_entries) == len(sky_entries):
                    # Index-based pairing: ground_entries[i] pairs with sky_entries[i]
                    for i in range(len(ground_entries)):
                        ground_pos = np.array(ground_entries[i]["position"], dtype=float)
                        sky_pos = np.array(sky_entries[i]["position"], dtype=float)
                        all_starts.append(ground_pos)
                        all_targets.append(sky_pos)
                        all_pairs_info.append((ground_entries[i], sky_entries[i], ground_pos, sky_pos))
                else:
                    # PBC-aware pairing: match each ground entry to closest sky entry
                    sky_positions = np.array([entry["position"] for entry in sky_entries], dtype=float)
                    used_sky_indices = set()

                    # Pair each ground entry with its closest available sky entry
                    for ground_entry in ground_entries:
                        ground_pos = np.array(ground_entry["position"], dtype=float)

                        # Calculate PBC-aware distances from this ground position to all sky positions
                        distances = _calculate_xy_pbc_distances(ground_pos, sky_positions, cell)

                        # Find the closest unused sky entry
                        best_sky_idx = None
                        best_dist = float('inf')
                        for sky_idx, dist in enumerate(distances):
                            if sky_idx not in used_sky_indices and dist < best_dist:
                                best_dist = dist
                                best_sky_idx = sky_idx

                        if best_sky_idx is None:
                            continue  # No available sky position to pair with

                        # Mark this sky entry as used
                        used_sky_indices.add(best_sky_idx)
                        sky_pos = sky_positions[best_sky_idx]
                        sky_entry = sky_entries[best_sky_idx]

                        all_starts.append(ground_pos)
                        all_targets.append(sky_pos)
                        all_pairs_info.append((ground_entry, sky_entry, ground_pos, sky_pos))
            
            # Deterministic vector selection based on fractional coordinates
            if len(all_starts) > 0 and cell is not None:
                # Use deterministic vector selection for all pairs
                vectors = find_optimal_spacer_vectors_global(
                    all_starts,
                    all_targets,
                    cell
                )
            else:
                # Fallback: calculate vectors individually (no PBC)
                vectors = []
                for i in range(len(all_starts)):
                    vec = all_targets[i] - all_starts[i]
                    vectors.append(vec)
            
            # Place spacers using the assigned vectors
            for i, (ground_entry, sky_entry, ground_pos, sky_pos) in enumerate(all_pairs_info):
                # Debug: Verify Z coordinates are different
                if len(ground_pos) >= 3 and len(sky_pos) >= 3:
                    direction = vectors[i] if i < len(vectors) else (sky_pos - ground_pos)
                    z_diff = abs(direction[2]) if len(direction) >= 3 else abs(ground_pos[2] - sky_pos[2])
                    if z_diff < 1e-6:
                        import warnings
                        warnings.warn(
                            f"Warning: Ground and sky positions have same Z coordinate! "
                            f"Ground pos={ground_pos}, Sky pos={sky_pos}, Z diff={z_diff:.6f}. "
                            f"Ground slab z={ground_slab.get('z', 'unknown')}, Sky slab z={sky_slab.get('z', 'unknown') if sky_slab else 'none'}. "
                            f"This will cause molecules to be placed in plane."
                        )
                
                # CRITICAL FIX: Only adjust Z, preserve X, Y coordinates
                # This ensures molecules are placed at correct S# site positions
                direction = vectors[i] if i < len(vectors) else (sky_pos - ground_pos)
                dist = np.linalg.norm(direction)
                
                if dist > 1e-6:
                    unit = direction / dist
                    # Only adjust Z component to move positions slightly outward
                    # Preserve X, Y to keep molecules at correct S# sites
                    ground_pos_adjusted = ground_pos.copy()
                    sky_pos_adjusted = sky_pos.copy()
                    ground_pos_adjusted[2] = ground_pos[2] - unit[2]  # Only adjust Z
                    sky_pos_adjusted[2] = sky_pos[2] + unit[2]  # Only adjust Z
                else:
                    ground_pos_adjusted = ground_pos.copy()
                    sky_pos_adjusted = sky_pos.copy()

                ground_pos_adjusted, sky_pos_adjusted = _adjust_positions_for_double_spacer(
                    ground_pos_adjusted,
                    sky_pos_adjusted,
                    spacer_template,
                )

                # Update target vector Z component to match adjusted positions
                target_vector = None
                if i < len(vectors):
                    target_vector = vectors[i].copy()
                    target_vector[2] = sky_pos_adjusted[2] - ground_pos_adjusted[2]

                # Place the spacer molecule using the specific vector from global optimization
                placed = place_double_spacer_between_positions(
                    spacer_template.copy(),
                    ground_pos_adjusted,
                    sky_pos_adjusted,
                    optimizer=optimizer,
                    cell=cell,
                    target_vector=target_vector,
                    existing_structure=structure if collision_strategy != "off" else None,
                    collision_strategy=collision_strategy,
                )
                if placed is not None and len(placed) > 0:
                    # Check for and resolve collisions before adding to structure
                    if collision_strategy != "off":
                        placed, collision_resolved = resolve_collisions(
                            placed, structure, cell=cell, strategy=collision_strategy
                        )
                        if not collision_resolved and collision_strategy == "reject":
                            continue  # Skip this spacer placement

                    # Add spacer atoms to structure
                    structure = add_atoms(structure, placed)
        else:
            # Mono spacers: no global optimization needed
            # Note: orientation alignment was already applied above for the template
            for entry in ground_entries:
                placed = _place_mono_sharp_spacer(spacer_template.copy(), entry["position"], 'bottom')
                if placed is not None and len(placed) > 0:
                    # Check for and resolve collisions before adding to structure
                    if collision_strategy != "off":
                        placed, collision_resolved = resolve_collisions(
                            placed, structure, cell=cell, strategy=collision_strategy
                        )
                        if not collision_resolved and collision_strategy == "reject":
                            continue  # Skip this spacer placement

                    structure = add_atoms(structure, placed)
            for entry in sky_entries:
                placed = _place_mono_sharp_spacer(spacer_template.copy(), entry["position"], 'top')
                if placed is not None and len(placed) > 0:
                    # Check for and resolve collisions before adding to structure
                    if collision_strategy != "off":
                        placed, collision_resolved = resolve_collisions(
                            placed, structure, cell=cell, strategy=collision_strategy
                        )
                        if not collision_resolved and collision_strategy == "reject":
                            continue  # Skip this spacer placement

                    structure = add_atoms(structure, placed)

    return structure


def _process_floor_slab(
    structure: Atoms,
    ground_slab: Dict[str, Any],
    sky_slab: Optional[Dict[str, Any]],
    sharp_sequence: Optional[_SharpSpacerSequence],
    optimizer: str = "KS",
    BX_dist: Optional[float] = None,
    spacer_orientation: Optional[List[str]] = None,
    collision_strategy: str = "rotate",
) -> Atoms:
    """Process a single ground slab and optionally populate sharp spacers to the sky."""
    structure = populate_a_sites(structure, ground_slab)
    structure = populate_b_sites(structure, ground_slab)
    structure = populate_x_sites(structure, ground_slab)
    structure = populate_sharp(structure, ground_slab, sky_slab, sharp_sequence, optimizer=optimizer, BX_dist=BX_dist, spacer_orientation=spacer_orientation, collision_strategy=collision_strategy)
    return structure


def populate_structure(
    matrix: QBuilderOutput,
    floors_cart: List[List[List[float]]],
    A_ions: Union[str, Atoms, List],
    B_ions: Union[str, List],
    X_ions: Union[str, List],
    Ap_ions: Optional[Union[str, Atoms, List]] = None,
    sharp_spacer: Optional[List[Union[str, Atoms]]] = None,
    site_labels: Optional[Dict[str, List[str]]] = None,
    optimizer: str = "KS",
    BX_dist: Optional[float] = None,
    spacer_orientation: Optional[List[str]] = None,
    collision_strategy: str = "rotate",
) -> Atoms:
    """
    Populate structure matrix with atoms based on site labels (A, B, X, Ap, S#).

    Floors are processed strictly in the provided cartesian order (floors_cart).
    Handles double spacers (sharp_spacer) for S# sites that connect adjacent layers.

    Parameters
    ----------
    collision_strategy : str, default "rotate"
        Strategy for resolving atomic collisions during placement:
        - "rotate": Rotate molecule around N-N axis to find collision-free orientation
        - "nudge": Apply small XY translations to resolve collisions
        - "optimize": Use geometry optimization to push atoms apart (slowest but most robust)
        - "reject": Raise warning and skip placement if collisions detected
        - "off": Skip collision detection entirely
    """
    # Normalize A-site ions (convert molecular strings to Atoms objects)
    A_ions = _normalize_ion_list(A_ions, normalize_a_site)

    # Normalize Ap-site ions if provided
    Ap_ions = _normalize_ion_list(Ap_ions, normalize_a_site)

    # Normalize sharp_spacer if provided
    if sharp_spacer is not None:
        def normalize_spacer_or_fallback(ds):
            if isinstance(ds, str):
                # Try to normalize as spacer first
                try:
                    from q2D_Materials.pipeline.common import normalize_spacer
                    return normalize_spacer(ds)
                except:
                    # Fallback to A-site normalization
                    return normalize_a_site(ds)
            return ds

        sharp_spacer_normalized = _normalize_ion_list(sharp_spacer, normalize_spacer_or_fallback, copy_atoms=True)
    else:
        sharp_spacer_normalized = None
    
    # Assign ions to positions using patterns with site labels
    assignments = assign_ions_to_sites(
        matrix.positions,
        A_ions, B_ions, X_ions, Ap_ions,
        sharp_spacer=sharp_spacer_normalized,
        site_labels=site_labels,
    )
    
    # Build floors strictly from provided floors_cart (ordered)
    if floors_cart is None:
        raise ValueError("floors_cart is required for populate_structure")

    # Lookup assignments by site/label/coords
    assignment_lookup = {}
    for st, ion, pos, label in assignments:
        key = (st, label, round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4))
        assignment_lookup[key] = (st, ion, pos, label)

    floors: List[Tuple[float, List[Tuple[str, Union[str, Atoms], np.ndarray, str]]]] = []
    for floor_entries in floors_cart:
        resolved: List[Tuple[str, Union[str, Atoms], np.ndarray, str]] = []
        for entry in floor_entries:
            if len(entry) < 4:
                continue
            st, x, y, z = entry[0], float(entry[1]), float(entry[2]), float(entry[3])
            key = (st, st, round(x, 4), round(y, 4), round(z, 4))
            chosen = assignment_lookup.get(key)
            if chosen is None:
                # fallback: ignore label match
                for k, v in assignment_lookup.items():
                    if k[0] == st and k[2] == round(x, 4) and k[3] == round(y, 4) and k[4] == round(z, 4):
                        chosen = v
                        break
            if chosen is None:
                continue
            st_r, ion_r, pos_r, label_r = chosen
            # S# sites will have ion_r=None from assign_ions_to_sites - will be assigned per floor pair
            resolved.append((st_r, ion_r, np.array([x, y, z], dtype=float), label_r))
        if resolved:
            floors.append((resolved[0][2][2], resolved))
    
    # Prepare ordered slabs (ground/sky metadata)
    floors.sort(key=lambda x: x[0])
    resolved_floors = [entries for _, entries in floors]
    slabs = _build_floor_slabs(resolved_floors)

    # Initialize structure and cell
    structure = Atoms()
    structure.set_cell(matrix.cell_vectors)
    structure.pbc = [1, 1, 1]

    sharp_sequence = _SharpSpacerSequence(sharp_spacer_normalized) if sharp_spacer_normalized else None

    # Process slabs ground-by-ground
    for idx, ground_slab in enumerate(slabs):
        sky_slab = slabs[idx + 1] if idx + 1 < len(slabs) else None
        structure = _process_floor_slab(
            structure,
            ground_slab,
            sky_slab,
            sharp_sequence,
            optimizer=optimizer,
            BX_dist=BX_dist,
            spacer_orientation=spacer_orientation,
            collision_strategy=collision_strategy,
        )

    return structure


def get_existing_spacer_molecules(structure: Atoms) -> List[Atoms]:
    """
    Extract individual spacer molecules from structure.

    Groups atoms by connectivity or site_role tag to identify separate molecules.

    Parameters
    ----------
    structure : Atoms
        The full structure containing multiple molecules

    Returns
    -------
    List[Atoms]
        List of individual spacer molecules
    """
    from .spacer import SITE_ROLE_KEY

    spacer_molecules = []

    if len(structure) == 0:
        return spacer_molecules

    # Check if we have site_role information
    if SITE_ROLE_KEY in structure.arrays:
        site_roles = structure.arrays[SITE_ROLE_KEY]

        # Find all unique molecule IDs (groups of atoms with same site_role)
        unique_roles = set()
        for role in site_roles:
            if role == "spacer":
                unique_roles.add(role)

        # For now, treat all spacer atoms as one molecule (simplified approach)
        # In practice, you'd want to group by connectivity or molecule ID
        spacer_indices = [i for i, role in enumerate(site_roles) if role == "spacer"]

        if spacer_indices:
            # Create a molecule from all spacer atoms
            spacer_atoms = structure[spacer_indices]
            spacer_molecules.append(spacer_atoms)

    else:
        # Fallback: no site_role info, try to identify molecules by connectivity
        # This is a simplified approach - in practice you'd use bond detection
        # For now, return empty list (no existing spacers to avoid)
        pass

    return spacer_molecules


def get_effective_spacer_size(spacer: Atoms) -> float:
    """
    Get effective size of spacer for layer separation calculations.
    
    For molecules: returns the molecule length along z-axis.
    For single atoms: returns 2 * ionic_radius to account for sphere diameter.
    
    Parameters
    ----------
    spacer : Atoms
        Spacer molecule or atom (ASE Atoms object)
        
    Returns
    -------
    float
        Effective spacer size in Angstroms
    """
    # Check if it's a single atom
    if len(spacer) == 1:
        # Single atom: use ionic radius
        symbol = spacer.get_chemical_symbols()[0]
        try:
            # Try to get ionic radius from A-site database
            ionic_rad = get_ionic_radius("A", symbol)
            # Return 2 * ionic_radius to account for sphere diameter
            return 2.0 * ionic_rad
        except (ValueError, KeyError):
            # If not found in A-site, try a reasonable default
            # Common atomic spacers: Cs (1.88), Rb (1.72), K (1.64)
            # Use 2.0 as fallback (reasonable for most atomic cations)
            return 2.0 * 1.8  # Default to ~3.6 Angstrom for unknown atoms
    else:
        # Molecule: use molecule length
        return get_molecule_length(spacer)


def setup_cell_for_spacers(structure: Atoms, nx: int, ny: int, lv0: float, lv1: float):
    """Set cell to supercell size (x and y expanded, z unchanged)."""
    current_cell = structure.cell
    if current_cell is not None:
        if hasattr(current_cell, 'lengths'):
            _, _, current_z = current_cell.lengths()
        else:
            current_z = np.linalg.norm(current_cell[2]) if hasattr(current_cell[0], '__len__') else current_cell[2]
        structure.set_cell([nx * lv0, ny * lv1, current_z])
        structure.pbc = [1, 1, 1]


def calculate_z_levels(attachment_end: str, n: int, lv2: float, penet: float) -> Tuple[List, float, float]:
    """Calculate z-levels and orientations for spacer attachment."""
    lv2_n = n * lv2
    penet_z = penet * 0.5 * lv2
    top_z = lv2_n - penet_z
    bottom_z_adjusted = penet_z
    
    if attachment_end in ('bottom', 'bot'):
        return [(bottom_z_adjusted, 'top', False)], bottom_z_adjusted, top_z
    elif attachment_end == 'top':
        return [(top_z, 'bottom', True)], bottom_z_adjusted, top_z
    else:  # 'both'
        return [
            (bottom_z_adjusted, 'top', False),
            (top_z, 'bottom', True)
        ], bottom_z_adjusted, top_z


def generate_attachment_positions(z_levels: List, nx: int, ny: int, lv0: float, lv1: float) -> List[Tuple]:
    """Generate all attachment positions for supercell expansion."""
    base_positions_frac = [[0.25, 0.75], [0.75, 0.25]]
    attachments = []
    
    for z, end_side, rotate_180 in z_levels:
        for base_pos_frac in base_positions_frac:
            for ix in range(nx):
                for iy in range(ny):
                    x = (base_pos_frac[0] + ix) * lv0
                    y = (base_pos_frac[1] + iy) * lv1
                    attachments.append((x, y, z, end_side, rotate_180))
    
    expected = len(z_levels) * len(base_positions_frac) * nx * ny
    if len(attachments) != expected:
        raise ValueError(f"Spacer position generation failed: expected {expected} positions, got {len(attachments)}")
    
    return attachments


def adjust_atomic_spacer_z(z: float, attachment_end: str, n: int, lv2: float, 
                           bottom_z_adjusted: float, top_z: float, symbol: str) -> float:
    """Adjust z position for atomic spacers based on ionic radius."""
    try:
        ionic_rad = get_ionic_radius("A", symbol)
    except (ValueError, KeyError):
        ionic_rad = 1.8
    
    bottom_z = 0.5 * ionic_rad
    top_z_atomic = n * lv2 - 0.5 * ionic_rad
    
    if attachment_end in ('bottom', 'bot'):
        return bottom_z
    elif attachment_end == 'top':
        return top_z_atomic
    else:  # 'both'
        return bottom_z if abs(z - bottom_z_adjusted) < abs(z - top_z) else top_z_atomic


def process_spacer_attachment(spacer_atoms: Atoms, x: float, y: float, z: float, 
                             end_side: str, rotate_180: bool, is_atomic_spacer: bool,
                             attachment_end: str, n: int, lv2: float, 
                             bottom_z_adjusted: float, top_z: float) -> Atoms:
    """Process a single spacer: rotate, align, and position."""
    if is_atomic_spacer:
        symbol = spacer_atoms.get_chemical_symbols()[0]
        z = adjust_atomic_spacer_z(z, attachment_end, n, lv2, bottom_z_adjusted, top_z, symbol)
    
    if rotate_180 and not is_atomic_spacer:
        com = spacer_atoms.get_center_of_mass()
        spacer_atoms.rotate(180, 'x', center=com)
    
    spacer_atoms = end_to_origin(spacer_atoms, end_side)
    return translate_atoms(spacer_atoms, [x, y, z])


def attach_spacers(spacer: Union[Atoms, List[Atoms]], layer: Atoms, n: int, 
                   lattice_vector_sizes: np.ndarray, supercell_size: Tuple[int, int, int],
                   penet: float = 0.3, attachment_end: str = 'both', mol_len: float = None,
                   ap_positions: Optional[List[np.ndarray]] = None) -> Atoms:
    """
    Attach spacer molecules to 2D layer using pattern-based assignment with supercell expansion.
    
    Parameters
    ----------
    spacer : Atoms or list[Atoms]
        Spacer molecule(s). If list, assigned sequentially to positions (pattern-based).
    layer : Atoms
        The 2D inorganic layer
    n : int
        Number of octahedral layers
    lattice_vector_sizes : np.ndarray
        Lattice vector sizes [a, b, c]
    supercell_size : tuple[int, int, int]
        Supercell dimensions (nx, ny, nz) - only nx and ny are used for spacer positions
    penet : float
        Penetration of spacer into layer
    attachment_end : str
        'top', 'bottom', or 'both'
    mol_len : float, optional
        Pre-computed molecule length (for efficiency)
    ap_positions : list[np.ndarray], optional
        Pre-computed spacer (Ap) positions from builder. If provided, used for x/y
        placement; z-levels still determined by attachment_end/penet.
        
    Returns
    -------
    Atoms
        Structure with spacers attached
    """
    if not isinstance(spacer, list):
        spacer = [spacer]
    
    if isinstance(lattice_vector_sizes, (int, float)):
        lattice_vector_sizes = np.array([lattice_vector_sizes] * 3)
    
    structure = layer.copy()
    nx, ny, _ = supercell_size
    lv0, lv1, lv2 = lattice_vector_sizes[0], lattice_vector_sizes[1], lattice_vector_sizes[2]
    
    setup_cell_for_spacers(structure, nx, ny, lv0, lv1)
    
    z_levels, bottom_z_adjusted, top_z = calculate_z_levels(attachment_end, n, lv2, penet)
    if ap_positions is not None and len(ap_positions) > 0:
        attachments = []
        for z, end_side, rotate_180 in z_levels:
            for pos in ap_positions:
                attachments.append((pos[0], pos[1], z, end_side, rotate_180))
    else:
        attachments = generate_attachment_positions(z_levels, nx, ny, lv0, lv1)
    
    for i, (x, y, z, end_side, rotate_180) in enumerate(attachments):
        current_spacer = spacer[i % len(spacer)].copy()
        
        if len(current_spacer) == 0:
            raise ValueError(f"Spacer {i} has no atoms after copy")
        
        is_atomic_spacer = (len(current_spacer) == 1)
        positioned_spacer = process_spacer_attachment(
            current_spacer, x, y, z, end_side, rotate_180, is_atomic_spacer,
            attachment_end, n, lv2, bottom_z_adjusted, top_z
        )
        
        structure = add_atoms(structure, positioned_spacer)
    
    return structure

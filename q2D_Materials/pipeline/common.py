"""
Shared helpers for perovskite pipeline entry points.

This module centralizes reusable steps such as template selection,
Glazer tilting, BX distance resolution, spacer normalization, and population.
"""

from typing import List, Tuple, Optional, Dict

import numpy as np
from ase import Atoms

from q2D_Materials.builders.templates import (
    available_templates,
    build_floor_schema,
    build_structure_matrix,
    flatten_floor_schema,
)
from q2D_Materials.builders.populate import populate_structure
from q2D_Materials.utils.sites.A_sites import (
    calculate_BX_distance,
    get_a_site_object,
)
from q2D_Materials.builders.molecule_builder import smiles_to_ase_atoms


def get_template(template_name: str | Dict) -> str | Dict:
    """
    Return a template name, validating it exists (based on available JSONs),
    or return the template data directly if a dict/JSON string is provided.
    """
    if isinstance(template_name, dict):
        return template_name
        
    # Check if input is a valid JSON string
    if isinstance(template_name, str) and template_name.strip().startswith("{"):
        return template_name
        
    name = template_name.lower()
    valid = available_templates()
    # Case-insensitive comparison: convert valid templates to lowercase for matching
    valid_lower = [v.lower() for v in valid]
    if name not in valid_lower:
        raise ValueError(f"template must be one of {valid} or a valid JSON/dict")
    # Return the original case template name
    valid_map = {v.lower(): v for v in valid}
    return valid_map[name]


def auto_calculate_BX_distance(B: str, X: str) -> float:
    """Calculate BX distance from ionic radii data with a small fallback."""
    try:
        return calculate_BX_distance(B, X)
    except ValueError:
        return 2.0


def _get_first_element(value):
    """
    Get the first element if value is a list/tuple, otherwise return the value itself.

    Parameters
    ----------
    value : any
        Input value that might be a list, tuple, or single value

    Returns
    -------
    any
        First element if list/tuple, otherwise the value itself
    """
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def resolve_BX_distance(B, X, BX_dist: float | None) -> float:
    """
    Return the BX distance, computing it when not provided.
    Accepts either single species or lists/tuples.
    """
    if BX_dist is not None:
        return BX_dist
    B_first = _get_first_element(B)
    X_first = _get_first_element(X)
    return auto_calculate_BX_distance(B_first, X_first)


def _create_atomic_atoms(element_str: str) -> Atoms:
    """
    Create an Atoms object from an atomic element string.

    Handles both pymatgen-based element validation and fallback logic.

    Parameters
    ----------
    element_str : str
        Atomic element symbol (e.g., 'Cs', 'K', 'Na')

    Returns
    -------
    Atoms
        Single-atom Atoms object

    Raises
    ------
    ValueError
        If the string is not a valid atomic element
    """
    from pymatgen.core.periodic_table import Element

    try:
        Element(element_str)
        return Atoms(element_str, positions=[[0, 0, 0]])
    except (ValueError, KeyError):
        pass

    # Fallback without pymatgen
    atomic_spacers = ['Cs', 'K', 'Rb', 'Na', 'Li', 'Ca', 'Sr', 'Ba', 'Mg']
    if element_str in atomic_spacers:
        return Atoms(element_str, positions=[[0, 0, 0]])
    if len(element_str) <= 2 and element_str[0].isupper():
        return Atoms(element_str, positions=[[0, 0, 0]])

    raise ValueError(f"'{element_str}' is not a recognized atomic element")


def _normalize_to_atoms_or_string(input_value, return_atoms_only: bool = False):
    """
    Common base function for normalizing string inputs to Atoms objects or strings.

    Parameters
    ----------
    input_value : str or Atoms
        Input value to normalize
    return_atoms_only : bool
        If True, always return Atoms objects (for spacers)
        If False, return strings for atomic cations, Atoms for molecular (for A-sites)

    Returns
    -------
    str or Atoms
        Normalized value
    """
    if isinstance(input_value, Atoms):
        return input_value.copy()

    if isinstance(input_value, str):
        normalized = get_a_site_object(input_value)
        if isinstance(normalized, Atoms):
            return normalized
        if isinstance(normalized, str):
            if return_atoms_only:
                # For spacers, first try to create Atoms from atomic elements
                # If that fails, try SMILES conversion
                try:
                    return _create_atomic_atoms(normalized)
                except ValueError:
                    # Not an atomic element, try SMILES conversion
                    return smiles_to_ase_atoms(input_value)
            else:
                # For A-sites, return atomic cations as strings
                return normalized

        if return_atoms_only:
            # Try SMILES conversion for spacers (when get_a_site_object returns None)
            return smiles_to_ase_atoms(input_value)
        else:
            # For A-sites, return unrecognized strings as-is
            return input_value

    if return_atoms_only:
        raise ValueError(f"Input must be a string (SMILES or abbreviation) or Atoms object, got {type(input_value)}")
    else:
        return input_value


def normalize_spacer(spacer):
    """
    Normalize spacer input to handle strings (SMILES or abbreviations) and Atoms objects.

    Supports both atomic spacers (e.g., "Cs", "Rb", "K") and molecular spacers (SMILES strings).

    Parameters
    ----------
    spacer : str or Atoms
        Spacer molecule as SMILES string, abbreviation, or Atoms object

    Returns
    -------
    Atoms
        ASE Atoms object of the spacer molecule or atom
    """
    return _normalize_to_atoms_or_string(spacer, return_atoms_only=True)


def calculate_max_sharp_spacer_span(
    sharp_spacer: Optional[List],
    atomic_span: Optional[float] = None,
) -> Optional[float]:
    """
    Return the maximum span required by sharp_spacer molecules.

    For double-NH3 spacers, uses the N–N distance. For mono-NH3 or atomic
    spacers, uses the single-molecule length along its main axis.
    """
    if sharp_spacer is None or len(sharp_spacer) == 0:
        return None

    from q2D_Materials.builders.spacer import calculate_double_spacer_nh3_distances
    from q2D_Materials.pipeline.common import normalize_spacer
    from q2D_Materials.builders.molecule_builder import get_molecule_length, align_ase_molecule_for_perovskite

    max_span = 0.0
    saw_valid = False
    atomic_only = True

    for spacer in sharp_spacer:
        if isinstance(spacer, str):
            try:
                spacer_atoms = normalize_spacer(spacer)
            except Exception:
                continue
        elif isinstance(spacer, Atoms):
            spacer_atoms = spacer
        else:
            continue

        if len(spacer_atoms) == 0:
            continue

        saw_valid = True
        if len(spacer_atoms) > 1:
            atomic_only = False

        from q2D_Materials.builders.spacer import count_nh3_groups
        nh3_count = count_nh3_groups(spacer_atoms)
        if nh3_count >= 2:
            span = calculate_double_spacer_nh3_distances(spacer_atoms)
        else:
            aligned = align_ase_molecule_for_perovskite(spacer_atoms.copy())
            span = get_molecule_length(aligned)

        if span > max_span:
            max_span = span

    if max_span > 0:
        return max_span

    if saw_valid and atomic_only and atomic_span is not None:
        return float(atomic_span)

    return None


def build_cell_positions(
    template_name: str | Dict,
    BX_dist: float,
    jahn_teller_dist: float,
    layer_sequence: Optional[List[str] | str] = None,
    xy_expansion: Tuple[int, int] = (1, 1),
    penetration: float = 0.0,
    spacer_provided: bool = False,
    attachment_end: Optional[str] = None,
    sharp_spacer_span: Optional[float] = None,
    glazer_angles: Optional[List[float]] = None,
    glazer_pattern: Optional[List[str]] = None,
    lattice_multipliers: Optional[List[float]] = None,
    interlayer_distances: Optional[Dict[int, float]] = None,
) -> Dict[str, object]:
    """
    Build floor schema then flatten to site-indexed positions.

    Penetration and spacer_provided are retained for signature compatibility
    but no longer alter the template geometry in this refactor.
    """
    schema = build_floor_schema(
        template_name=template_name,
        BX_dist=BX_dist,
        jahn_teller_dist=jahn_teller_dist,
        layer_sequence=layer_sequence,
        xy_expansion=xy_expansion,
        sharp_spacer_nn_distance=sharp_spacer_span,
        glazer_angles=glazer_angles,
        glazer_pattern=glazer_pattern,
        lattice_multipliers=lattice_multipliers,
        interlayer_distances=interlayer_distances,
    )

    if spacer_provided and attachment_end:
        _apply_attachment_end_ap(schema, attachment_end)

    positions, site_labels = flatten_floor_schema(schema)
    cell_matrix = schema.cell
    lattice_vec_sizes = np.linalg.norm(cell_matrix, axis=1)

    if penetration != 0.0:
        from q2D_Materials.builders.populate import apply_penetration_offsets

        positions = apply_penetration_offsets(positions, penetration, BX_dist=BX_dist)

    return {
        "schema": schema,
        "positions": positions,
        "site_labels": site_labels,
        "unit_cell_matrix": cell_matrix,
        "lattice_vec_sizes": lattice_vec_sizes,
        "floors_cart": list(schema.floors.values()),
    }


def _apply_attachment_end_ap(schema, attachment_end: str) -> None:
    """
    Convert A sites on selected floors to Ap based on attachment_end ('bottom', 'top', 'both').
    """
    if attachment_end is None:
        return

    end = attachment_end.lower()
    keys = list(schema.floors.keys())
    if not keys:
        return

    floors_to_convert: List[str] = []
    if end in ("bottom", "bot"):
        floors_to_convert.append(keys[0])
    elif end == "top":
        floors_to_convert.append(keys[-1])
    elif end == "both":
        floors_to_convert.extend([keys[0], keys[-1]])
    else:
        return

    for fk in floors_to_convert:
        entries = schema.floors.get(fk, [])
        for entry in entries:
            if entry and entry[0] == "A":
                entry[0] = "Ap"


def populate_positions(
    positions: Dict[str, List[List[float]]],
    lattice_vec_sizes: np.ndarray,
    cell_matrix: np.ndarray,
    floors_cart: List[List[List[float]]],
    A,
    B,
    X,
    Ap_ions=None,
    sharp_spacer=None,
    site_labels=None,
    optimizer: str = "KS",
    BX_dist=None,
    spacer_orientation: Optional[List[str]] = None,
    collision_strategy: str = "rotate",
) -> Atoms:
    """Populate ions using the population toolchain."""
    positions_np = {site: np.asarray(coords, dtype=float) for site, coords in positions.items()}
    matrix = build_structure_matrix(positions_np, lattice_vec_sizes, cell_vectors=cell_matrix)
    return populate_structure(
        matrix=matrix,
        floors_cart=floors_cart,
        A_ions=A,
        B_ions=B,
        X_ions=X,
        Ap_ions=Ap_ions,
        sharp_spacer=sharp_spacer,
        site_labels=site_labels,
        optimizer=optimizer,
        BX_dist=BX_dist,
        spacer_orientation=spacer_orientation,
        collision_strategy=collision_strategy,
    )


def _split_layer_sequence_string(seq: str) -> List[str]:
    """Split a hyphen/space-separated layer string into explicit names."""
    raw = seq
    for sep in ("-", ","):
        raw = raw.replace(sep, " ")
    parts = [p for p in raw.split() if p]
    return parts if parts else [seq]


def _parse_layer_sequence_with_distances(seq: str) -> Tuple[List[str], Optional[Dict[int, float]]]:
    """
    Parse a layer sequence string with optional inter-floor distances.

    Examples:
    - "L1-L2-L3" -> (["L1", "L2", "L3"], None)
    - "L1-(1.5)-L2-(2.0)-L3" -> (["L1", "L2", "L3"], {0: 1.5, 1: 2.0})

    Parameters
    ----------
    seq : str
        Layer sequence string, e.g. "L1-(1.5)-L3-M2-(3.2)-M1"

    Returns
    -------
    Tuple[List[str], Optional[Dict[int, float]]]
        Floor labels and optional interlayer distances keyed by gap index.
    """
    import re

    # Pattern to match floor labels and optional distances in parentheses
    # Matches: LABEL or LABEL-(DISTANCE)
    pattern = r'([A-Za-z]\w*)\s*(?:-\s*\((\d+(?:\.\d+)?)\))?'
    matches = re.findall(pattern, seq)

    if not matches:
        # Fallback to old parsing if regex fails
        floor_labels = _split_layer_sequence_string(seq)
        return floor_labels, None

    floor_labels = []
    distances = {}

    for i, (label, dist_str) in enumerate(matches):
        floor_labels.append(label)
        if dist_str:
            try:
                distances[i] = float(dist_str)
            except ValueError:
                # Skip invalid numeric values
                continue

    # Only return distances if any were found
    return floor_labels, distances if distances else None


def default_layer_sequence(layer_sequence: Optional[str | List[str]], thickness: int, structure_type: str = "monolayer") -> Tuple[str | List[str], Optional[Dict[int, float]]]:
    """
    Return the default layer sequence for a given thickness if not provided.

    Returns both the layer sequence and any interlayer distances specified in the sequence.
    """
    if layer_sequence is None:
        if structure_type.lower() == "bulk":
            return "-".join(["L1-L2"] * thickness), None
        else:
            # For monolayers, add L1 at the end
            base_sequence = "-".join(["L1-L2"] * thickness)
            return f"{base_sequence}-L1", None
    elif isinstance(layer_sequence, str) and layer_sequence.upper() == "DJ":
        # DJ keyword: (L1-L2) * thickness + "-M1-M2" for bulk
        if structure_type.lower() == "bulk" and thickness > 1:
            parts = ["L2", "L1"] * thickness
            parts.pop() # remove the last L1
            base_sequence = "-".join(parts)
            return f"{base_sequence}-M1-M1", None
        elif structure_type.lower() == "bulk" and thickness == 1:
            return "L2-M1-M1", None
    elif isinstance(layer_sequence, str) and layer_sequence.upper() == "RP":
        # RP keyword: fixed sequence using RP layers
        return "L2-M1-RP1-RP2-RP1-M1", None
    else:
        if isinstance(layer_sequence, str):
            if any(sep in layer_sequence for sep in ("-", ",", " ")):
                # Try to parse with distances first
                floor_labels, distances = _parse_layer_sequence_with_distances(layer_sequence)
                return floor_labels, distances
            return layer_sequence, None
        return layer_sequence, None

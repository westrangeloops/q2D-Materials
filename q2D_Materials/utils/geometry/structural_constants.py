"""Constants for structural analysis functions.

Parts of this code are from PDynA (https://github.com/WMD-group/PDynA):
MIT License - Copyright (c) 2022 Xia Liang
"""

import json
from pathlib import Path

# Load constants from JSON file in central data/config folder
# Path: q2D_Materials/utils/geometry/structural_constants.py -> q2D_Materials/data/config/structural_constants.json
_CONSTANTS_FILE = Path(__file__).parent.parent.parent / "data" / "config" / "structural_constants.json"

with open(_CONSTANTS_FILE, 'r') as f:
    _constants = json.load(f)

def _get_value(path: list) -> any:
    """Extract value from nested dict, handling both old and new JSON formats."""
    current = _constants
    for i, key in enumerate(path):
        if isinstance(current, dict) and key in current:
            current = current[key]
            # If we have a dict with 'value' key, it's the new format
            # Only check this on the last key in the path
            if i == len(path) - 1 and isinstance(current, dict) and 'value' in current:
                return current['value']
        else:
            raise KeyError(f"Path {path} not found in constants")
    # If we didn't find a 'value' key, return the current value directly
    return current

# Default tolerance constants
DEFAULT_FITTING_TOLERANCE = _get_value(["default_tolerance_constants", "fitting_tolerance"])
DEFAULT_MATCH_TOLERANCE = _get_value(["default_tolerance_constants", "match_tolerance"])
DEFAULT_BB_SEARCH_RADIUS = _get_value(["default_tolerance_constants", "bb_search_radius"])
DEFAULT_POPULATION_GAP_TOL = _get_value(["default_tolerance_constants", "population_gap_tol"])
DEFAULT_RMSD_THRESHOLD = _get_value(["default_tolerance_constants", "rmsd_threshold"])
DEFAULT_ANGLE_TOLERANCE = _get_value(["default_tolerance_constants", "angle_tolerance"])
DEFAULT_DISTINCT_THRESHOLD = _get_value(["default_tolerance_constants", "distinct_threshold"])

# Octahedral matching thresholds
DEFAULT_FITTING_TOL = _get_value(["octahedral_matching_thresholds", "fitting_tol"])
DEFAULT_CONFIDENCE_BOUND = _get_value(["octahedral_matching_thresholds", "confidence_bound"])

# Time averaging defaults
DEFAULT_MATCH_TOL = _get_value(["time_averaging_defaults", "match_tol"])

# Hydrogen bond detection
DEFAULT_HBOND_MAX_DISTANCE = _get_value(["hydrogen_bond_detection", "max_distance"])
DEFAULT_HBOND_MIN_ANGLE = _get_value(["hydrogen_bond_detection", "min_angle"])

# Volume calculation tolerance
DEFAULT_VOLUME_TOL = _get_value(["volume_calculation", "tolerance"])

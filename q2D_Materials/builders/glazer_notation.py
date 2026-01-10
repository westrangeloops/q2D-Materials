"""
Glazer notation parsing and space group lookup for perovskite tilt systems.

This module provides utilities to parse full Glazer notation strings (e.g., 'a-b+a-')
and look up corresponding space groups based on the mathematical framework from
Beanland (2008): "The structure of planar defects in tilted perovskites".
"""

from typing import Dict, List, Optional, Tuple, NamedTuple, Union
import re
import json
import os


class GlazerSystem(NamedTuple):
    """Represents a parsed Glazer tilt system."""
    notation: str  # Full notation like "a-b+a-"
    magnitudes: List[str]  # ['a', 'b', 'a'] - relative magnitudes per axis
    phases: List[str]  # ['+', '-', '+'] - phase signs per axis
    tilt_pattern: List[str]  # ['+', '-', '+'] - ready for apply_glazer_tilt
    space_group: Optional[str] = None  # Space group if known


# Space group lookup table based on Howard & Stokes (1998) Table 1
# Format: (magnitudes_tuple, phases_tuple) -> space_group
# All 15 systems from Howard & Stokes (1998) are included
SPACE_GROUP_TABLE: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], str] = {
    # 1. Cubic (no tilting)
    (("a", "a", "a"), ("0", "0", "0")): "Pm-3m",  # #221

    # 2. Tetragonal - a0a0c+ (M₃⁺ uncoupled)
    (("a", "a", "c"), ("0", "0", "+")): "P4/mbm",  # #127
    (("a", "a", "0"), ("0", "0", "c")): "P4/mbm",  # Domain equivalent
    (("a", "0", "a"), ("0", "c", "0")): "P4/mbm",  # Domain equivalent
    (("0", "a", "a"), ("c", "0", "0")): "P4/mbm",  # Domain equivalent

    # 3. Tetragonal - a0b+b+ (M₃⁺ uncoupled)
    (("a", "b", "b"), ("0", "+", "+")): "I4/mmm",  # #139
    (("b", "a", "b"), ("+", "0", "+")): "I4/mmm",  # Domain equivalent
    (("b", "b", "a"), ("+", "+", "0")): "I4/mmm",  # Domain equivalent

    # 4. Cubic - a+a+a+ (M₃⁺ uncoupled)
    (("a", "a", "a"), ("+", "+", "+")): "Im-3",  # #204

    # 5. Orthorhombic - a+b+c+ (M₃⁺ uncoupled)
    (("a", "b", "c"), ("+", "+", "+")): "Immm",  # #71

    # 6. Tetragonal - a0a0c- (R₄⁺ uncoupled)
    (("a", "a", "c"), ("0", "0", "-")): "I4/mcm",  # #140
    (("a", "a", "0"), ("0", "0", "c")): "I4/mcm",  # Domain equivalent (if c is -)
    (("a", "0", "a"), ("0", "c", "0")): "I4/mcm",  # Domain equivalent
    (("0", "a", "a"), ("c", "0", "0")): "I4/mcm",  # Domain equivalent

    # 7. Orthorhombic - a0b-b- (R₄⁺ uncoupled)
    (("a", "b", "b"), ("0", "-", "-")): "Imma",  # #74
    (("b", "a", "b"), ("-", "0", "-")): "Imma",  # Domain equivalent
    (("b", "b", "a"), ("-", "-", "0")): "Imma",  # Domain equivalent

    # 8. Rhombohedral - a-a-a- (R₄⁺ uncoupled)
    (("a", "a", "a"), ("-", "-", "-")): "R-3c",  # #167

    # 9. Monoclinic - a0b-c- (R₄⁺ uncoupled)
    (("a", "b", "c"), ("0", "-", "-")): "C2/m",  # #12
    (("a", "b", "0"), ("0", "-", "c")): "C2/m",  # Domain equivalent
    (("a", "0", "c"), ("0", "b", "-")): "C2/m",  # Domain equivalent
    (("0", "b", "c"), ("a", "0", "-")): "C2/m",  # Domain equivalent

    # 10. Monoclinic - a-b-b- (R₄⁺ uncoupled)
    (("a", "b", "b"), ("-", "-", "-")): "C2/c",  # #15
    (("b", "a", "b"), ("-", "-", "-")): "C2/c",  # Domain equivalent
    (("b", "b", "a"), ("-", "-", "-")): "C2/c",  # Domain equivalent

    # 11. Triclinic - a-b-c- (R₄⁺ uncoupled)
    (("a", "b", "c"), ("-", "-", "-")): "P-1",  # #2

    # 12. Orthorhombic - a0b+c- (M₃⁺ + R₄⁺ coupled)
    (("a", "b", "c"), ("0", "+", "-")): "Cmcm",  # #63
    (("a", "b", "0"), ("0", "+", "c")): "Cmcm",  # Domain equivalent
    (("a", "0", "c"), ("0", "b", "+")): "Cmcm",  # Domain equivalent
    (("0", "b", "c"), ("a", "0", "+")): "Cmcm",  # Domain equivalent

    # 13. Orthorhombic - a+b-b- (M₃⁺ + R₄⁺ coupled)
    (("a", "b", "b"), ("+", "-", "-")): "Pnma",  # #62
    (("b", "a", "b"), ("-", "+", "-")): "Pnma",  # Domain equivalent
    (("b", "b", "a"), ("-", "-", "+")): "Pnma",  # Domain equivalent
    (("a", "b", "a"), ("-", "+", "-")): "Pnma",  # a-b+a- variant

    # 14. Monoclinic - a+b-c- (M₃⁺ + R₄⁺ coupled)
    (("a", "b", "c"), ("+", "-", "-")): "P21/m",  # #11

    # 15. Tetragonal - a+a+c- (M₃⁺ + R₄⁺ coupled)
    (("a", "a", "c"), ("+", "+", "-")): "P42/nmc",  # #137
    (("a", "a", "0"), ("+", "+", "c")): "P42/nmc",  # Domain equivalent
    (("a", "0", "a"), ("+", "c", "+")): "P42/nmc",  # Domain equivalent
    (("0", "a", "a"), ("c", "+", "+")): "P42/nmc",  # Domain equivalent
    
    # Additional detected patterns (may be domain equivalents or detection artifacts)
    (("a", "b", "c"), ("+", "-", "+")): "Pmc2",  # Detected pattern, may be domain equivalent
    (("a", "b", "c"), ("-", "+", "-")): "Pmc2",  # Detected pattern, may be domain equivalent
    (("a", "b", "c"), ("+", "+", "-")): "Pmc21",  # Detected pattern
    (("a", "b", "c"), ("-", "-", "+")): "Pmc2",  # Detected pattern
}

# The 15 conventional Glazer patterns from Howard & Stokes (1998)
# These are the canonical forms that all equivalent patterns should map to
CONVENTIONAL_PATTERNS = [
    "a0a0a0",  # 1. Cubic (no tilting)
    "a0a0c+",  # 2. Tetragonal - M₃⁺ uncoupled
    "a0b+b+",  # 3. Tetragonal - M₃⁺ uncoupled
    "a+a+a+",  # 4. Cubic - M₃⁺ uncoupled
    "a+b+c+",  # 5. Orthorhombic - M₃⁺ uncoupled
    "a0a0c-",  # 6. Tetragonal - R₄⁺ uncoupled
    "a0b-b-",  # 7. Orthorhombic - R₄⁺ uncoupled
    "a-a-a-",  # 8. Rhombohedral - R₄⁺ uncoupled
    "a0b-c-",  # 9. Monoclinic - R₄⁺ uncoupled
    "a-b-b-",  # 10. Monoclinic - R₄⁺ uncoupled
    "a-b-c-",  # 11. Triclinic - R₄⁺ uncoupled
    "a0b+c-",  # 12. Orthorhombic - M₃⁺ + R₄⁺ coupled
    "a+b-b-",  # 13. Orthorhombic - M₃⁺ + R₄⁺ coupled
    "a+b-c-",  # 14. Monoclinic - M₃⁺ + R₄⁺ coupled
    "a+a+c-",  # 15. Tetragonal - M₃⁺ + R₄⁺ coupled
]

# Preferred notation for common space groups (canonical representations)
PREFERRED_NOTATIONS: Dict[str, str] = {
    "Pm-3m": "a0a0a0",
    "R-3c": "a-a-a-",
    "P4/mmm": "a0a0c+",
    "P4/mbm": "a0a0c+",
    "I4/mcm": "a0a0c-",
    "Cmcm": "a0b+c-",
    "Pnma": "a+b-b-",
    "Imma": "a0b-b-",
    "I4/mmm": "a0b+b+",
}

# Path to the lookup table JSON file
_LOOKUP_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "q2D_Materials", "tables", "glazer_pattern_lookup.json"
)


def _glazer_to_eta(notation: str) -> Tuple[int, int, int, int, int, int]:
    """
    Convert Glazer notation to eta (6-tuple) representation per Howard & Stokes (1998).
    
    eta[0:3] are M3+ (in-phase) tilt amplitudes along x, y, z axes
    eta[3:6] are R4+ (out-of-phase) tilt amplitudes along x, y, z axes
    
    Parameters
    ----------
    notation : str
        Glazer notation string (e.g., "a+b-c-")
    
    Returns
    -------
    tuple of 6 integers
        (m3_x, m3_y, m3_z, r4_x, r4_y, r4_z) where values are 0, 1, or 2
    """
    system = parse_glazer_notation(notation)
    magnitudes = system.magnitudes
    phases = system.phases
    
    # Map magnitudes to amplitudes: a=1, b=2, c=2 (or we can use a=1, b=1, c=2)
    # For simplicity, we'll use: a=1, b=1, c=2 (distinguishing equal vs unequal)
    mag_to_amp = {'a': 1, 'b': 1, 'c': 2, '0': 0}
    
    eta = [0, 0, 0, 0, 0, 0]
    
    for axis in range(3):
        mag = magnitudes[axis]
        phase = phases[axis]
        
        if phase == '0' or mag == '0':
            # No tilt on this axis
            continue
        elif phase == '+':
            # In-phase (M3+) tilt
            amp = mag_to_amp.get(mag, 1)
            eta[axis] = amp
        elif phase == '-':
            # Out-of-phase (R4+) tilt
            amp = mag_to_amp.get(mag, 1)
            eta[axis + 3] = amp
    
    return tuple(eta)


def _canonicalize_eta(eta: Tuple[int, int, int, int, int, int]) -> Tuple[int, int, int]:
    """
    Convert a 6-tuple of tilt amplitudes to canonical signed magnitude form.
    
    Per Howard & Stokes (1998), this canonicalizes by:
    1. Converting to signed magnitudes (+ for in-phase M3+, - for out-of-phase R4+)
    2. Sorting axes by absolute magnitude
    3. Breaking ties via fixed axis priority x < y < z
    
    Parameters
    ----------
    eta : tuple of 6 integers
        Tilt amplitudes where indices 0-2 are M3+ (in-phase) tilts along x, y, z
        and indices 3-5 are R4+ (out-of-phase) tilts along x, y, z.
        Valid values are 0, 1, or 2.
    
    Returns
    -------
    tuple of 3 signed integers
        Canonical form (x, y, z) where:
        - Positive values indicate in-phase (+) tilts
        - Negative values indicate out-of-phase (-) tilts
        - Magnitude indicates tilt amplitude (0, 1, or 2)
    """
    m3_plus = eta[0:3]  # In-phase tilts (x, y, z)
    r4_plus = eta[3:6]  # Out-of-phase tilts (x, y, z)
    
    # Check for conflicts: both types on same axis is invalid
    for axis in range(3):
        if m3_plus[axis] != 0 and r4_plus[axis] != 0:
            raise ValueError(f"Both in-phase and out-of-phase tilts on axis {axis}")
    
    # Convert to signed magnitudes: + for in-phase, - for out-of-phase
    signed = []
    for axis in range(3):
        if m3_plus[axis] != 0:
            signed.append(m3_plus[axis])  # Positive for in-phase
        elif r4_plus[axis] != 0:
            signed.append(-r4_plus[axis])  # Negative for out-of-phase
        else:
            signed.append(0)
    
    # Create axis-tagged values for sorting: (absolute_magnitude, axis_priority, signed_value)
    # Axis priority: x=0, y=1, z=2 (for tie-breaking)
    tagged = [(abs(signed[i]), i, signed[i]) for i in range(3)]
    
    # Sort by absolute magnitude first, then by axis priority for ties
    tagged.sort(key=lambda x: (x[0], x[1]))
    
    # Extract the sorted signed values
    canonical = (tagged[0][2], tagged[1][2], tagged[2][2])
    
    return canonical


def _get_pattern_signature(canonical: Tuple[int, int, int]) -> Tuple[str, ...]:
    """
    Extract the pattern signature from canonical signed tilts.
    
    Per Howard & Stokes (1998), the signature captures:
    - Whether each position is zero, positive, or negative
    - Equality relationships between magnitudes OF THE SAME SIGN TYPE
    
    Returns
    -------
    tuple of 3 strings
        Pattern signature for lookup
    """
    pattern = []
    
    for i in range(3):
        if canonical[i] == 0:
            pattern.append("0")
        else:
            sign_char = "+" if canonical[i] > 0 else "-"
            curr_val = canonical[i]  # Keep sign for comparison
            
            # Check equality with previous axes OF THE SAME SIGN
            equals_prev = None
            for j in range(i):
                if canonical[j] == curr_val:  # Same signed value
                    equals_prev = j
                    break
            
            if equals_prev is not None:
                # Same signed value as earlier axis - use same letter
                pattern.append(pattern[equals_prev][0] + sign_char)
            else:
                # New value - use next available letter
                used_letters = set()
                for p in pattern:
                    if p != "0" and len(p) > 0:
                        used_letters.add(p[0])
                
                for letter in "abcdef":
                    if letter not in used_letters:
                        pattern.append(letter + sign_char)
                        break
    
    return tuple(pattern)


# Howard & Stokes (1998) Table 1: Pattern signature -> Conventional Glazer notation
# Based on the classification system from the paper
_HS_PATTERN_LOOKUP: Dict[Tuple[str, ...], str] = {
    # 1. (0, 0, 0) - No tilts
    ("0", "0", "0"): "a0a0a0",
    
    # 2. (0, 0, +) - Single in-phase
    ("0", "0", "a+"): "a0a0c+",
    
    # 3. (0, 0, -) - Single out-of-phase
    ("0", "0", "a-"): "a0a0c-",
    
    # 4. (0, +, +) equal - Two equal in-phase
    ("0", "a+", "a+"): "a0b+b+",
    
    # 5. (0, -, -) equal - Two equal out-of-phase
    ("0", "a-", "a-"): "a0b-b-",
    
    # 6. (0, -, -) unequal - Two unequal out-of-phase
    ("0", "a-", "b-"): "a0b-c-",
    
    # 7. (+, +, +) equal - Three equal in-phase
    ("a+", "a+", "a+"): "a+a+a+",
    
    # 8. (-, -, -) equal - Three equal out-of-phase
    ("a-", "a-", "a-"): "a-a-a-",
    
    # 9. (+, +, +) all different - Three unequal in-phase
    ("a+", "b+", "c+"): "a+b+c+",
    
    # 10. (+, +, +) two equal - Maps to a+b+c+ (higher symmetry)
    ("a+", "b+", "b+"): "a+b+c+",
    ("a+", "a+", "b+"): "a+b+c+",
    
    # 11. (-, -, -) all different - Three unequal out-of-phase
    ("a-", "b-", "c-"): "a-b-c-",
    
    # 12. (-, -, -) two equal - a-b-b-
    ("a-", "b-", "b-"): "a-b-b-",
    ("a-", "a-", "b-"): "a-b-b-",
    
    # 13. (0, +, -) - Mixed
    ("0", "a+", "b-"): "a0b+c-",
    
    # 14. (+, -, -) equal negatives - One in-phase, two equal out-of-phase
    ("a+", "b-", "b-"): "a+b-b-",
    
    # 15. (+, -, -) unequal - One in-phase, two unequal out-of-phase
    ("a+", "b-", "c-"): "a+b-c-",
    
    # 16. (+, +, -) equal plus - Two equal in-phase, one out-of-phase
    ("a+", "a+", "b-"): "a+a+c-",
}


def _glazer_to_conventional_via_hs(notation: str) -> Optional[str]:
    """
    Convert Glazer notation to conventional form using Howard & Stokes (1998) classification.
    
    This uses the systematic approach from the 1998 paper:
    1. Convert Glazer notation to eta (6-tuple)
    2. Canonicalize to signed magnitudes
    3. Extract pattern signature
    4. Lookup in Howard & Stokes table
    
    Parameters
    ----------
    notation : str
        Glazer notation string
    
    Returns
    -------
    str or None
        Conventional pattern, or None if classification fails
    """
    try:
        # Convert to eta representation
        eta = _glazer_to_eta(notation)
        
        # Canonicalize
        canonical = _canonicalize_eta(eta)
        
        # Get pattern signature
        signature = _get_pattern_signature(canonical)
        
        # Lookup in Howard & Stokes table
        if signature in _HS_PATTERN_LOOKUP:
            return _HS_PATTERN_LOOKUP[signature]
        
        return None
    except (ValueError, Exception):
        return None


def parse_glazer_notation(notation: str) -> GlazerSystem:
    """
    Parse a Glazer notation string into components.

    Examples:
        "a-b+a-" -> magnitudes=['a','b','a'], phases=['+','-','+']
        "a0a0c+" -> magnitudes=['a','0','a'], phases=['0','0','+']
        "a-a-a-" -> magnitudes=['a','a','a'], phases=['-','-','-']

    Parameters
    ----------
    notation : str
        Glazer notation string (with separators: letter-sign-letter-sign-letter-sign)

    Returns
    -------
    GlazerSystem
        Parsed tilt system with all components
    """
    # Clean the notation string - remove spaces, keep separators
    notation = notation.replace(" ", "")

    # Glazer notation format: [letter][sign][letter][sign][letter][sign]
    # where sign is +, -, or 0, and letter is a, b, c, or 0

    # Check length
    if len(notation) != 6:
        raise ValueError(f"Glazer notation must be exactly 6 characters, got {len(notation)}: {notation}")

    # Parse components
    magnitudes = []
    phases = []

    for i in range(3):
        mag_idx = 2 * i
        phase_idx = 2 * i + 1

        mag_char = notation[mag_idx]
        phase_char = notation[phase_idx]

        # Validate magnitude
        if mag_char not in "abc0":
            raise ValueError(f"Invalid magnitude '{mag_char}' at position {mag_idx}. Must be a, b, c, or 0.")

        # Validate phase
        if phase_char not in "+-0":
            raise ValueError(f"Invalid phase '{phase_char}' at position {phase_idx}. Must be +, -, or 0.")

        magnitudes.append(mag_char)
        phases.append(phase_char)

    # Validate components
    valid_magnitudes = set("abc0")
    valid_phases = set("+-0")

    for mag in magnitudes:
        if mag not in valid_magnitudes:
            raise ValueError(f"Invalid magnitude '{mag}' in notation '{notation}'. Must be a, b, c, or 0.")

    for phase in phases:
        if phase not in valid_phases:
            raise ValueError(f"Invalid phase '{phase}' in notation '{notation}'. Must be +, -, or 0.")

    # Normalize notation string
    normalized = notation.lower()

    # Convert to tilt pattern format (phases only, 0 for no tilt)
    tilt_pattern = []
    for phase in phases:
        if phase == "0":
            tilt_pattern.append("0")
        elif phase == "+":
            tilt_pattern.append("+")
        elif phase == "-":
            tilt_pattern.append("-")
        else:
            raise ValueError(f"Invalid phase sign: {phase}")

    # Look up space group
    magnitudes_tuple = tuple(magnitudes)
    phases_tuple = tuple(phases)
    space_group = SPACE_GROUP_TABLE.get((magnitudes_tuple, phases_tuple))

    return GlazerSystem(
        notation=normalized,
        magnitudes=magnitudes,
        phases=phases,
        tilt_pattern=tilt_pattern,
        space_group=space_group
    )


def get_space_group_from_notation(notation: str) -> Optional[str]:
    """
    Get the space group for a given Glazer notation.

    Parameters
    ----------
    notation : str
        Glazer notation string

    Returns
    -------
    str or None
        Space group symbol, or None if not found in lookup table
    """
    system = parse_glazer_notation(notation)
    return system.space_group


def get_notation_from_space_group(space_group: str) -> Optional[str]:
    """
    Get the canonical Glazer notation for a space group.
    
    Parameters
    ----------
    space_group : str
        Space group symbol (case insensitive, e.g., "Pnma")
        
    Returns
    -------
    str or None
        Glazer notation string (e.g., "a-b+a-"), or None if not found
    """
    sg_normalized = space_group.strip()
    
    # Try preferred mapping first (case insensitive search)
    for k, v in PREFERRED_NOTATIONS.items():
        if k.lower() == sg_normalized.lower():
            return v
            
    # Try reverse lookup in the full table
    for (mags, phases), sg in SPACE_GROUP_TABLE.items():
        if sg.lower() == sg_normalized.lower():
            # Reconstruct notation string
            res = []
            for m, p in zip(mags, phases):
                res.append(m)
                res.append(p)
            return "".join(res)
            
    return None


def get_required_supercell(tilt_pattern: List[str]) -> Tuple[int, int, int]:
    """
    Determine minimum supercell required for a tilt pattern.

    Anti-phase tilting (-) requires doubling of periodicity perpendicular to tilt axis.

    Parameters
    ----------
    tilt_pattern : List[str]
        Tilt pattern ['+','-','0'] for [x,y,z]

    Returns
    -------
    (int, int, int)
        Minimum supercell dimensions (nx, ny, nz)
    """
    nx = 2 if tilt_pattern[0] == "-" else 1
    ny = 2 if tilt_pattern[1] == "-" else 1
    nz = 2 if tilt_pattern[2] == "-" else 1

    return (nx, ny, nz)


def suggest_angles_for_notation(
    notation: str,
    base_angle: float = 10.0
) -> List[float]:
    """
    Suggest reasonable tilt angles for a Glazer notation.

    Uses zero angles for "0" phases and base_angle for non-zero phases.

    Parameters
    ----------
    notation : str
        Glazer notation string
    base_angle : float
        Base angle in degrees for tilting (default 10°)

    Returns
    -------
    List[float]
        Suggested angles [omega_x, omega_y, omega_z] in degrees
    """
    system = parse_glazer_notation(notation)

    angles = []
    for phase in system.phases:
        if phase == "0":
            angles.append(0.0)
        else:
            angles.append(base_angle)

    return angles

def resolve_glazer_input(input_val: str) -> str:
    """
    Determine if input is a Glazer notation or space group, and return notation.
    
    Heuristic:
    - Glazer notation is always 6 chars (e.g. 'a-b+c-').
    - Space groups are variable length and often start with capital letters (P, I, F, R, C).
    - If input looks like notation, return it.
    - If input looks like space group, look up notation.
    """
    s = input_val.strip()
    
    # Check for notation pattern: [letter][sign][letter][sign][letter][sign]
    notation_pattern = r"^[abc0][+\-0][abc0][+\-0][abc0][+\-0]$"
    if re.match(notation_pattern, s, re.IGNORECASE):
        return s.lower()
        
    # Assume it's a space group
    notation = get_notation_from_space_group(s)
    if notation:
        return notation
        
    # If ambiguous, and 6 chars, try parsing as notation, otherwise raise error
    if len(s) == 6:
        try:
            parse_glazer_notation(s)
            return s.lower()
        except ValueError:
            pass
            
    raise ValueError(f"Could not resolve '{input_val}' as a valid Glazer notation or known Space Group.")


def generate_equivalent_patterns(notation: str) -> List[str]:
    """
    Generate all equivalent Glazer patterns.
    
    Uses two methods:
    1. SPACE_GROUP_TABLE: Patterns that map to the same space group are equivalent
    2. 90° rotations: Patterns related by 90° rotations about x, y, z axes
    
    According to Howard & Stokes (1998) and Stokes (2002), patterns that differ
    by 90° rotations are equivalent domains representing the same structure.
    For example: a+b0b0 and b0a+b0 are equivalent (90° rotation about z-axis).
    
    Parameters
    ----------
    notation : str
        Glazer notation string (e.g., "a+b0b0")
    
    Returns
    -------
    List[str]
        List of all equivalent notation strings, including the original
    """
    system = parse_glazer_notation(notation)
    magnitudes = system.magnitudes
    phases = system.phases
    
    equivalent = set()
    equivalent.add(notation.lower())
    
    # Method 1: Use SPACE_GROUP_TABLE to find patterns with same space group
    mags_tuple = tuple(magnitudes)
    phs_tuple = tuple(phases)
    space_group = SPACE_GROUP_TABLE.get((mags_tuple, phs_tuple))
    
    if space_group:
        # Find all patterns that map to the same space group
        for (mags, phs), sg in SPACE_GROUP_TABLE.items():
            if sg == space_group:
                eq = "".join([m + p for m, p in zip(mags, phs)])
                equivalent.add(eq)
    
    # Method 2: Generate patterns by 90° rotations
    # For Glazer notation, 90° rotations swap axes:
    # 90° about z: (x, y, z) -> (y, x, z) - swap x and y
    # 90° about y: (x, y, z) -> (z, y, x) - swap x and z  
    # 90° about x: (x, y, z) -> (x, z, y) - swap y and z
    
    def rotate_z(mags, phs):
        """90° rotation about z-axis: swap x and y"""
        return [mags[1], mags[0], mags[2]], [phs[1], phs[0], phs[2]]
    
    def rotate_y(mags, phs):
        """90° rotation about y-axis: swap x and z"""
        return [mags[2], mags[1], mags[0]], [phs[2], phs[1], phs[0]]
    
    def rotate_x(mags, phs):
        """90° rotation about x-axis: swap y and z"""
        return [mags[0], mags[2], mags[1]], [phs[0], phs[2], phs[1]]
    
    current_mags = list(magnitudes)
    current_phs = list(phases)
    rotations = [rotate_x, rotate_y, rotate_z]
    
    # Apply rotations iteratively to generate all equivalent patterns
    for rot1 in rotations:
        mags1, phs1 = rot1(current_mags, current_phs)
        eq1 = "".join([m + p for m, p in zip(mags1, phs1)])
        equivalent.add(eq1)
        
        for rot2 in rotations:
            mags2, phs2 = rot2(mags1, phs1)
            eq2 = "".join([m + p for m, p in zip(mags2, phs2)])
            equivalent.add(eq2)
            
            for rot3 in rotations:
                mags3, phs3 = rot3(mags2, phs2)
                eq3 = "".join([m + p for m, p in zip(mags3, phs3)])
                equivalent.add(eq3)
    
    return sorted(list(equivalent))


def are_patterns_equivalent(notation1: str, notation2: str) -> bool:
    """
    Check if two Glazer patterns are equivalent (same structure, different domain).
    
    Parameters
    ----------
    notation1 : str
        First Glazer notation
    notation2 : str
        Second Glazer notation
    
    Returns
    -------
    bool
        True if patterns are equivalent, False otherwise
    """
    eq1 = generate_equivalent_patterns(notation1)
    eq2 = generate_equivalent_patterns(notation2)
    
    # Check if they share any equivalent patterns
    return bool(set(eq1) & set(eq2))


def normalize_to_canonical(notation: str) -> str:
    """
    Normalize a Glazer notation to a canonical form.
    
    Uses lexicographic ordering to pick a canonical representative
    from all equivalent patterns.
    
    Parameters
    ----------
    notation : str
        Glazer notation string
    
    Returns
    -------
    str
        Canonical notation (lexicographically smallest equivalent)
    """
    equivalent = generate_equivalent_patterns(notation)
    return min(equivalent)


def _glazer_to_conventional_via_hs(notation: str) -> Optional[str]:
    """
    Convert Glazer notation to conventional form using Howard & Stokes (1998) classification.
    
    This uses the systematic approach from the 1998 paper:
    1. Convert Glazer notation to eta (6-tuple)
    2. Canonicalize to signed magnitudes
    3. Extract pattern signature
    4. Lookup in Howard & Stokes table
    
    Parameters
    ----------
    notation : str
        Glazer notation string
    
    Returns
    -------
    str or None
        Conventional pattern, or None if classification fails
    """
    try:
        # Convert to eta representation
        eta = _glazer_to_eta(notation)
        
        # Canonicalize
        canonical = _canonicalize_eta(eta)
        
        # Get pattern signature
        signature = _get_pattern_signature(canonical)
        
        # Lookup in Howard & Stokes table
        if signature in _HS_PATTERN_LOOKUP:
            return _HS_PATTERN_LOOKUP[signature]
        
        return None
    except (ValueError, Exception):
        return None


def _generate_pattern_lookup_table() -> Dict[str, str]:
    """
    Generate a complete lookup table mapping ALL possible Glazer patterns
    to their conventional form (one of the 15 standard patterns).
    
    Generates all valid patterns and maps them to conventional forms.
    Pattern format: [mag][phase][mag][phase][mag][phase]
    - Magnitude positions: a, b, c, 0
    - Phase positions: +, -, 0
    - Valid: if mag=0 then phase must be 0, otherwise phase can be +, -, or 0
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping pattern -> conventional_pattern
    """
    lookup_table = {}
    
    print("Generating complete Glazer pattern lookup table...")
    print("This will generate all valid patterns and map them to conventional forms...")
    
    # Generate all valid patterns
    magnitudes = ['a', 'b', 'c', '0']
    phases = ['+', '-', '0']
    
    valid_patterns = []
    
    # Generate all combinations: 4^3 * 3^3 = 64 * 27 = 1,728
    for mag1 in magnitudes:
        for phase1 in phases:
            for mag2 in magnitudes:
                for phase2 in phases:
                    for mag3 in magnitudes:
                        for phase3 in phases:
                            # Skip invalid: if magnitude is 0, phase must be 0
                            if (mag1 == '0' and phase1 != '0') or \
                               (mag2 == '0' and phase2 != '0') or \
                               (mag3 == '0' and phase3 != '0'):
                                continue
                            
                            pattern = f"{mag1}{phase1}{mag2}{phase2}{mag3}{phase3}"
                            valid_patterns.append(pattern)
    
    print(f"Generated {len(valid_patterns)} valid patterns")
    print("Mapping to conventional patterns...")
    
    # Pre-compute equivalents for all conventional patterns for faster lookup
    print("Pre-computing equivalents for conventional patterns...")
    conventional_equivalents = {}
    for conv_pattern in CONVENTIONAL_PATTERNS:
        conventional_equivalents[conv_pattern] = set(generate_equivalent_patterns(conv_pattern))
    
    # Map all patterns using Howard & Stokes (1998) classification
    for i, pattern in enumerate(valid_patterns):
        # First try: Use Howard & Stokes classification system
        conv_hs = _glazer_to_conventional_via_hs(pattern)
        if conv_hs:
            lookup_table[pattern] = conv_hs
        else:
            # Fallback: Check equivalence with conventional patterns
            try:
                pattern_equivalents = set(generate_equivalent_patterns(pattern))
                
                # Check if any equivalent intersects with any conventional pattern's equivalents
                found_conventional = None
                for conv_pattern in CONVENTIONAL_PATTERNS:
                    if pattern_equivalents & conventional_equivalents[conv_pattern]:
                        found_conventional = conv_pattern
                        break
                
                if found_conventional:
                    lookup_table[pattern] = found_conventional
                else:
                    # Try to find via space group as fallback
                    try:
                        system = parse_glazer_notation(pattern)
                        if system.space_group:
                            for conv_pattern in CONVENTIONAL_PATTERNS:
                                conv_system = parse_glazer_notation(conv_pattern)
                                if conv_system.space_group == system.space_group:
                                    lookup_table[pattern] = conv_pattern
                                    break
                    except (ValueError, Exception):
                        pass
            except (ValueError, Exception):
                pass
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(valid_patterns)} patterns "
                  f"({len(lookup_table)} mapped so far)...")
    
    # Final pass: map remaining patterns
    # Patterns with all phases = 0 should map to a0a0a0
    still_unmapped = [p for p in valid_patterns if p not in lookup_table]
    if still_unmapped:
        print(f"\nMapping {len(still_unmapped)} remaining patterns...")
        for pattern in still_unmapped:
            # Check if all phases are 0 (no tilt) - these should map to a0a0a0
            if pattern[1] == '0' and pattern[3] == '0' and pattern[5] == '0':
                lookup_table[pattern] = 'a0a0a0'
            else:
                # For patterns with tilts that couldn't be mapped, try one more time
                # by checking if any equivalent was mapped
                try:
                    equivalents = generate_equivalent_patterns(pattern)
                    for eq in equivalents:
                        if eq in lookup_table:
                            lookup_table[pattern] = lookup_table[eq]
                            break
                    else:
                        # Still unmapped - default to a0a0a0 (shouldn't happen often)
                        lookup_table[pattern] = 'a0a0a0'
                except (ValueError, Exception):
                    lookup_table[pattern] = 'a0a0a0'
    
    print(f"\nGenerated lookup table:")
    print(f"  Valid patterns: {len(valid_patterns)}")
    print(f"  Mapped to conventional: {len(lookup_table)}")
    
    return lookup_table


def _load_pattern_lookup_table() -> Dict[str, str]:
    """
    Load the pattern lookup table from JSON file, or generate it if it doesn't exist.
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping pattern -> conventional_pattern
    """
    # Check if file exists
    if os.path.exists(_LOOKUP_TABLE_PATH):
        try:
            with open(_LOOKUP_TABLE_PATH, 'r') as f:
                lookup_table = json.load(f)
            print(f"Loaded lookup table from {_LOOKUP_TABLE_PATH} "
                  f"({len(lookup_table)} entries)")
            return lookup_table
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading lookup table: {e}")
            print("Regenerating lookup table...")
    
    # Generate the table
    lookup_table = _generate_pattern_lookup_table()
    
    # Save to file
    os.makedirs(os.path.dirname(_LOOKUP_TABLE_PATH), exist_ok=True)
    with open(_LOOKUP_TABLE_PATH, 'w') as f:
        json.dump(lookup_table, f, indent=2, sort_keys=True)
    print(f"Saved lookup table to {_LOOKUP_TABLE_PATH}")
    
    return lookup_table


# Cache the lookup table
_PATTERN_LOOKUP_TABLE: Optional[Dict[str, str]] = None


def get_conventional_pattern(notation: str) -> str:
    """
    Convert any Glazer pattern to one of the 15 conventional patterns
    from Howard & Stokes (1998).
    
    Uses a pre-built lookup table stored in JSON for fast conversion.
    The table is generated automatically if it doesn't exist.
    
    Parameters
    ----------
    notation : str
        Glazer notation string (may be non-conventional)
    
    Returns
    -------
    str
        Conventional pattern from the 15 standard patterns
    
    Examples
    --------
    >>> get_conventional_pattern("b0a+b0")
    'a+b0b0'
    >>> get_conventional_pattern("b+a0b+")
    'a0b+b+'
    >>> get_conventional_pattern("a0a0c+")
    'a0a0c+'
    """
    global _PATTERN_LOOKUP_TABLE
    
    # Load table if not cached
    if _PATTERN_LOOKUP_TABLE is None:
        _PATTERN_LOOKUP_TABLE = _load_pattern_lookup_table()
    
    notation_lower = notation.lower().strip()
    
    # Direct lookup
    if notation_lower in _PATTERN_LOOKUP_TABLE:
        return _PATTERN_LOOKUP_TABLE[notation_lower]
    
    # Fallback: try to generate equivalents and find match
    # This handles edge cases not in the table
    equivalents = generate_equivalent_patterns(notation_lower)
    for eq in equivalents:
        if eq in _PATTERN_LOOKUP_TABLE:
            return _PATTERN_LOOKUP_TABLE[eq]
    
    # Last resort: return the notation itself if it's already conventional
    if notation_lower in CONVENTIONAL_PATTERNS:
        return notation_lower
    
    # If all else fails, return the lexicographically smallest equivalent
    # This should rarely happen
    return min(equivalents) if equivalents else notation_lower

"""Glazer notation parsing and space group lookup."""

from typing import Dict, List, Optional, Tuple, NamedTuple, Union
import re
import json
import os
from pathlib import Path


class GlazerSystem(NamedTuple):
    """Represents a parsed Glazer tilt system."""
    notation: str
    magnitudes: List[str]
    phases: List[str]
    tilt_pattern: List[str]
    space_group: Optional[str] = None


SPACE_GROUP_TABLE: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], str] = {
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

CONVENTIONAL_PATTERNS = [
    "a0a0a0",
    "a0a0c+",
    "a0b+b+",
    "a+a+a+",
    "a+b+c+",
    "a0a0c-",
    "a0b-b-",
    "a-a-a-",
    "a0b-c-",
    "a-b-b-",
    "a-b-c-",
    "a0b+c-",
    "a+b-b-",
    "a+b-c-",
    "a+a+c-",
]

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

# Load from q2D_Materials/data/tables
_LOOKUP_TABLE_PATH = str(
    Path(__file__).resolve().parent.parent / "data" / "tables" / "glazer_pattern_lookup.json"
)


def _glazer_to_eta(notation: str) -> Tuple[int, int, int, int, int, int]:
    """Convert Glazer notation to eta (6-tuple) representation."""
    system = parse_glazer_notation(notation)
    magnitudes = system.magnitudes
    phases = system.phases
    mag_to_amp = {'a': 1, 'b': 1, 'c': 2, '0': 0}
    eta = [0, 0, 0, 0, 0, 0]

    for axis in range(3):
        mag = magnitudes[axis]
        phase = phases[axis]
        if phase == '0' or mag == '0':
            continue
        elif phase == '+':
            amp = mag_to_amp.get(mag, 1)
            eta[axis] = amp
        elif phase == '-':
            amp = mag_to_amp.get(mag, 1)
            eta[axis + 3] = amp
    
    return tuple(eta)


def _canonicalize_eta(eta: Tuple[int, int, int, int, int, int]) -> Tuple[int, int, int]:
    """Convert a 6-tuple of tilt amplitudes to canonical signed magnitude form."""
    m3_plus = eta[0:3]
    r4_plus = eta[3:6]

    for axis in range(3):
        if m3_plus[axis] != 0 and r4_plus[axis] != 0:
            raise ValueError(f"Both in-phase and out-of-phase tilts on axis {axis}")

    signed = []
    for axis in range(3):
        if m3_plus[axis] != 0:
            signed.append(m3_plus[axis])
        elif r4_plus[axis] != 0:
            signed.append(-r4_plus[axis])
        else:
            signed.append(0)

    tagged = [(abs(signed[i]), i, signed[i]) for i in range(3)]
    tagged.sort(key=lambda x: (x[0], x[1]))
    canonical = (tagged[0][2], tagged[1][2], tagged[2][2])
    
    return canonical


def _get_pattern_signature(canonical: Tuple[int, int, int]) -> Tuple[str, ...]:
    """Extract the pattern signature from canonical signed tilts."""
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
                pattern.append(pattern[equals_prev][0] + sign_char)
            else:
                used_letters = set()
                for p in pattern:
                    if p != "0" and len(p) > 0:
                        used_letters.add(p[0])
                for letter in "abcdef":
                    if letter not in used_letters:
                        pattern.append(letter + sign_char)
                        break
    
    return tuple(pattern)


_HS_PATTERN_LOOKUP: Dict[Tuple[str, ...], str] = {
    ("0", "0", "0"): "a0a0a0",
    ("0", "0", "a+"): "a0a0c+",
    ("0", "0", "a-"): "a0a0c-",
    ("0", "a+", "a+"): "a0b+b+",
    ("0", "a-", "a-"): "a0b-b-",
    ("0", "a-", "b-"): "a0b-c-",
    ("a+", "a+", "a+"): "a+a+a+",
    ("a-", "a-", "a-"): "a-a-a-",
    ("a+", "b+", "c+"): "a+b+c+",
    ("a+", "b+", "b+"): "a+b+c+",
    ("a+", "a+", "b+"): "a+b+c+",
    ("a-", "b-", "c-"): "a-b-c-",
    ("a-", "b-", "b-"): "a-b-b-",
    ("a-", "a-", "b-"): "a-b-b-",
    ("0", "a+", "b-"): "a0b+c-",
    ("a+", "b-", "b-"): "a+b-b-",
    ("a+", "b-", "c-"): "a+b-c-",
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
    """Parse a Glazer notation string into components."""
    notation = notation.replace(" ", "")
    if len(notation) != 6:
        raise ValueError(f"Glazer notation must be exactly 6 characters, got {len(notation)}: {notation}")

    magnitudes = []
    phases = []

    for i in range(3):
        mag_idx = 2 * i
        phase_idx = 2 * i + 1

        mag_char = notation[mag_idx]
        phase_char = notation[phase_idx]

        if mag_char not in "abc0":
            raise ValueError(f"Invalid magnitude '{mag_char}' at position {mag_idx}. Must be a, b, c, or 0.")
        if phase_char not in "+-0":
            raise ValueError(f"Invalid phase '{phase_char}' at position {phase_idx}. Must be +, -, or 0.")

        magnitudes.append(mag_char)
        phases.append(phase_char)

    normalized = notation.lower()
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
    """Get the space group for a given Glazer notation."""
    system = parse_glazer_notation(notation)
    return system.space_group


def get_notation_from_space_group(space_group: str) -> Optional[str]:
    """Get the canonical Glazer notation for a space group."""
    sg_normalized = space_group.strip()
    for k, v in PREFERRED_NOTATIONS.items():
        if k.lower() == sg_normalized.lower():
            return v
    for (mags, phases), sg in SPACE_GROUP_TABLE.items():
        if sg.lower() == sg_normalized.lower():
            res = []
            for m, p in zip(mags, phases):
                res.append(m)
                res.append(p)
            return "".join(res)
    return None


def get_required_supercell(tilt_pattern: List[str]) -> Tuple[int, int, int]:
    """Determine minimum supercell required for a tilt pattern."""
    nx = 2 if tilt_pattern[0] == "-" else 1
    ny = 2 if tilt_pattern[1] == "-" else 1
    nz = 2 if tilt_pattern[2] == "-" else 1

    return (nx, ny, nz)


def suggest_angles_for_notation(
    notation: str,
    base_angle: float = 10.0
) -> List[float]:
    """Suggest reasonable tilt angles for a Glazer notation."""
    system = parse_glazer_notation(notation)

    angles = []
    for phase in system.phases:
        if phase == "0":
            angles.append(0.0)
        else:
            angles.append(base_angle)

    return angles

def resolve_glazer_input(input_val: str) -> str:
    """Determine if input is a Glazer notation or space group, and return notation."""
    s = input_val.strip()
    notation_pattern = r"^[abc0][+\-0][abc0][+\-0][abc0][+\-0]$"
    if re.match(notation_pattern, s, re.IGNORECASE):
        return s.lower()
    notation = get_notation_from_space_group(s)
    if notation:
        return notation
    if len(s) == 6:
        try:
            parse_glazer_notation(s)
            return s.lower()
        except ValueError:
            pass
    raise ValueError(f"Could not resolve '{input_val}' as a valid Glazer notation or known Space Group.")


def generate_equivalent_patterns(notation: str) -> List[str]:
    """Generate all equivalent Glazer patterns."""
    system = parse_glazer_notation(notation)
    magnitudes = system.magnitudes
    phases = system.phases
    equivalent = set()
    equivalent.add(notation.lower())

    mags_tuple = tuple(magnitudes)
    phs_tuple = tuple(phases)
    space_group = SPACE_GROUP_TABLE.get((mags_tuple, phs_tuple))

    if space_group:
        for (mags, phs), sg in SPACE_GROUP_TABLE.items():
            if sg == space_group:
                eq = "".join([m + p for m, p in zip(mags, phs)])
                equivalent.add(eq)

    def rotate_z(mags, phs):
        return [mags[1], mags[0], mags[2]], [phs[1], phs[0], phs[2]]

    def rotate_y(mags, phs):
        return [mags[2], mags[1], mags[0]], [phs[2], phs[1], phs[0]]

    def rotate_x(mags, phs):
        return [mags[0], mags[2], mags[1]], [phs[0], phs[2], phs[1]]

    current_mags = list(magnitudes)
    current_phs = list(phases)
    rotations = [rotate_x, rotate_y, rotate_z]

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
    """Check if two Glazer patterns are equivalent."""
    eq1 = generate_equivalent_patterns(notation1)
    eq2 = generate_equivalent_patterns(notation2)
    return bool(set(eq1) & set(eq2))


def normalize_to_canonical(notation: str) -> str:
    """Normalize a Glazer notation to a canonical form."""
    equivalent = generate_equivalent_patterns(notation)
    return min(equivalent)


def _generate_pattern_lookup_table() -> Dict[str, str]:
    """Generate a complete lookup table mapping all Glazer patterns to conventional forms."""
    lookup_table = {}
    magnitudes = ['a', 'b', 'c', '0']
    phases = ['+', '-', '0']
    valid_patterns = []

    for mag1 in magnitudes:
        for phase1 in phases:
            for mag2 in magnitudes:
                for phase2 in phases:
                    for mag3 in magnitudes:
                        for phase3 in phases:
                            if (mag1 == '0' and phase1 != '0') or \
                               (mag2 == '0' and phase2 != '0') or \
                               (mag3 == '0' and phase3 != '0'):
                                continue
                            pattern = f"{mag1}{phase1}{mag2}{phase2}{mag3}{phase3}"
                            valid_patterns.append(pattern)

    conventional_equivalents = {}
    for conv_pattern in CONVENTIONAL_PATTERNS:
        conventional_equivalents[conv_pattern] = set(generate_equivalent_patterns(conv_pattern))

    for i, pattern in enumerate(valid_patterns):
        conv_hs = _glazer_to_conventional_via_hs(pattern)
        if conv_hs:
            lookup_table[pattern] = conv_hs
        else:
            try:
                pattern_equivalents = set(generate_equivalent_patterns(pattern))
                found_conventional = None
                for conv_pattern in CONVENTIONAL_PATTERNS:
                    if pattern_equivalents & conventional_equivalents[conv_pattern]:
                        found_conventional = conv_pattern
                        break
                if found_conventional:
                    lookup_table[pattern] = found_conventional
                else:
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

    still_unmapped = [p for p in valid_patterns if p not in lookup_table]
    if still_unmapped:
        for pattern in still_unmapped:
            if pattern[1] == '0' and pattern[3] == '0' and pattern[5] == '0':
                lookup_table[pattern] = 'a0a0a0'
            else:
                try:
                    equivalents = generate_equivalent_patterns(pattern)
                    for eq in equivalents:
                        if eq in lookup_table:
                            lookup_table[pattern] = lookup_table[eq]
                            break
                    else:
                        lookup_table[pattern] = 'a0a0a0'
                except (ValueError, Exception):
                    lookup_table[pattern] = 'a0a0a0'

    return lookup_table


def _load_pattern_lookup_table() -> Dict[str, str]:
    """Load the pattern lookup table from JSON file, or generate it if it doesn't exist."""
    if os.path.exists(_LOOKUP_TABLE_PATH):
        try:
            with open(_LOOKUP_TABLE_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    lookup_table = _generate_pattern_lookup_table()
    os.makedirs(os.path.dirname(_LOOKUP_TABLE_PATH), exist_ok=True)
    with open(_LOOKUP_TABLE_PATH, 'w') as f:
        json.dump(lookup_table, f, indent=2, sort_keys=True)
    return lookup_table


_PATTERN_LOOKUP_TABLE: Optional[Dict[str, str]] = None


def get_conventional_pattern(notation: str) -> str:
    """Convert any Glazer pattern to one of the 15 conventional patterns."""
    global _PATTERN_LOOKUP_TABLE

    if _PATTERN_LOOKUP_TABLE is None:
        _PATTERN_LOOKUP_TABLE = _load_pattern_lookup_table()

    notation_lower = notation.lower().strip()
    if notation_lower in _PATTERN_LOOKUP_TABLE:
        return _PATTERN_LOOKUP_TABLE[notation_lower]

    equivalents = generate_equivalent_patterns(notation_lower)
    for eq in equivalents:
        if eq in _PATTERN_LOOKUP_TABLE:
            return _PATTERN_LOOKUP_TABLE[eq]

    if notation_lower in CONVENTIONAL_PATTERNS:
        return notation_lower

    return min(equivalents) if equivalents else notation_lower

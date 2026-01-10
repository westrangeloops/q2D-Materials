"""
Element-specific bond radii and constants for perovskite analysis.

This module contains empirical bond distance data tuned for common perovskite
materials, based on VESTA cutoffs and literature values.

Constants
---------
PEROVSKITE_BOND_RADII : dict
    Element pair-specific maximum bond distances (Angstroms)
COVALENT_RADII : dict
    Standard covalent radii for elements (Angstroms)
AMMONIUM_NITROGEN_ELEMENTS : set
    Elements that can form NH3+ attachment points
MOLECULAR_A_SITE_PATTERNS : dict
    Common molecular A-site cation formulas

Functions
---------
get_bond_cutoff(symbol1, symbol2, multiplier=1.1)
    Get bond cutoff distance for two elements
"""

# =============================================================================
# ELEMENT-SPECIFIC BOND RADII FOR PEROVSKITES
# =============================================================================
# Based on VESTA cutoffs, tuned for common perovskite elements
# Format: (Element1, Element2) -> max_bond_distance in Angstroms

PEROVSKITE_BOND_RADII = {
    # Pb-X bonds (lead halide perovskites)
    ('Pb', 'I'): 3.70, ('I', 'Pb'): 3.70,
    ('Pb', 'Br'): 3.40, ('Br', 'Pb'): 3.40,
    ('Pb', 'Cl'): 3.20, ('Cl', 'Pb'): 3.20,
    ('Pb', 'O'): 3.04, ('O', 'Pb'): 3.04,

    # Sn-X bonds (tin halide perovskites)
    ('Sn', 'I'): 3.52, ('I', 'Sn'): 3.52,
    ('Sn', 'Br'): 3.22, ('Br', 'Sn'): 3.22,
    ('Sn', 'Cl'): 3.13, ('Cl', 'Sn'): 3.13,

    # A-site to X-site (ionic/coordination)
    ('Cs', 'I'): 4.41, ('I', 'Cs'): 4.41,
    ('Cs', 'Br'): 4.07, ('Br', 'Cs'): 4.07,
    ('Cs', 'Cl'): 3.91, ('Cl', 'Cs'): 3.91,
    ('Rb', 'I'): 4.33, ('I', 'Rb'): 4.33,
    ('Rb', 'Br'): 4.05, ('Br', 'Rb'): 4.05,
    ('K', 'I'): 4.12, ('I', 'K'): 4.12,
    ('K', 'Br'): 3.85, ('Br', 'K'): 3.85,

    # Organic bonds (covalent)
    ('C', 'C'): 1.89, ('C', 'N'): 1.79, ('N', 'C'): 1.79,
    ('C', 'H'): 1.20, ('H', 'C'): 1.20,
    ('N', 'H'): 1.20, ('H', 'N'): 1.20,
    ('C', 'O'): 1.97, ('O', 'C'): 1.97,
    ('N', 'N'): 1.88, ('O', 'O'): 1.70,
    ('C', 'S'): 2.15, ('S', 'C'): 2.15,
    ('C', 'P'): 1.94, ('P', 'C'): 1.94,
    ('O', 'H'): 1.20, ('H', 'O'): 1.20,

    # Hydrogen bonds (N-H...X)
    ('N', 'I'): 3.80, ('I', 'N'): 3.80,
    ('N', 'Br'): 3.60, ('Br', 'N'): 3.60,
    ('N', 'Cl'): 3.48, ('Cl', 'N'): 3.48,
    ('H', 'I'): 3.20, ('I', 'H'): 3.20,
    ('H', 'Br'): 3.00, ('Br', 'H'): 3.00,
    ('H', 'Cl'): 2.80, ('Cl', 'H'): 2.80,
}

# Covalent radii for fallback bond calculation
COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'S': 1.05, 'Cl': 0.99, 'Br': 1.20, 'I': 1.39, 'P': 1.07,
    'Pb': 1.46, 'Sn': 1.39, 'Bi': 1.48, 'Sb': 1.39,
    'Cs': 2.44, 'Rb': 2.20, 'K': 2.03, 'Na': 1.66, 'Li': 1.28,
    'Ba': 2.15, 'Sr': 1.95, 'Ca': 1.76,
}

# Elements that can form NH3+/ammonium attachment points (S# sites)
AMMONIUM_NITROGEN_ELEMENTS = {'N'}

# Common A-site molecular cations (will be detected as distinct from spacers)
MOLECULAR_A_SITE_PATTERNS = {
    'CH6N': 'MA',   # Methylammonium CH3NH3+
    'CH5N2': 'FA',  # Formamidinium HC(NH2)2+
    'H4N': 'NH4',   # Ammonium NH4+
}


def get_bond_cutoff(symbol1: str, symbol2: str, multiplier: float = 1.1) -> float:
    """
    Get bond cutoff distance for two elements.

    Uses PEROVSKITE_BOND_RADII if available, otherwise falls back to
    sum of covalent radii with a multiplier.

    Parameters
    ----------
    symbol1 : str
        First element symbol
    symbol2 : str
        Second element symbol
    multiplier : float, optional
        Multiplier for covalent radii sum (default 1.1)

    Returns
    -------
    float
        Maximum bond distance in Angstroms

    Examples
    --------
    >>> get_bond_cutoff('Pb', 'I')
    3.7
    >>> get_bond_cutoff('C', 'H')
    1.2
    >>> get_bond_cutoff('Ti', 'O')  # Falls back to covalent radii
    2.42
    """
    pair = (symbol1, symbol2)
    if pair in PEROVSKITE_BOND_RADII:
        return PEROVSKITE_BOND_RADII[pair]

    # Fallback to covalent radii
    r1 = COVALENT_RADII.get(symbol1, 1.5)
    r2 = COVALENT_RADII.get(symbol2, 1.5)
    return (r1 + r2) * multiplier

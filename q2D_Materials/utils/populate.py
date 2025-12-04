"""
Element population for template structures.

This module handles the chemistry: populating template structures with
actual elements, including support for compositional mixing at any site.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from ase import Atoms

from .template import (
    SITE_ROLE_KEY, SITE_A, SITE_B, SITE_X, SITE_SPACER,
    TEMPLATE_A, TEMPLATE_B, TEMPLATE_X,
    get_site_indices, is_template,
)
from .common_a_sites import is_molecular_a_cation


# Representative atoms for molecular A-site cations
# Most organic cations have NH3+ groups, so N is a good representative
MOLECULAR_A_REPRESENTATIVE = {
    'MA': 'N',   # Methylammonium - CH3NH3+
    'FA': 'N',   # Formamidinium - CH(NH2)2+
    'EA': 'N',   # Ethylammonium - CH3CH2NH3+
    'DMA': 'N',  # Dimethylammonium - (CH3)2NH2+
    'GA': 'N',   # Guanidinium - C(NH2)3+
    'NH4': 'N',  # Ammonium - NH4+
}


def _is_element_symbol(s: str) -> bool:
    """
    Check if string is a valid element symbol.
    
    Element symbols are:
    - Single uppercase: H, C, N, O, K, I, etc.
    - Uppercase + lowercase: Ca, Rb, Ti, Pb, etc.
    
    Database abbreviations are ALL UPPERCASE: MA, FA, PEA, HDA, etc.
    """
    if len(s) == 1:
        return s.isupper()  # H, C, N, O, K, I, etc.
    elif len(s) == 2:
        # Element: first upper, second lower (Ca, Rb, Ti)
        # Abbreviation: both upper (MA, FA)
        return s[0].isupper() and s[1].islower()
    else:
        # Longer strings are not element symbols
        return False


def _get_representative_symbol(ion: str) -> str:
    """
    Get the representative atomic symbol for an ion.
    
    For molecular A-site cations (like MA, FA), returns a representative
    atom (usually N) since ASE can't handle molecular symbols.
    For atomic ions, returns the ion as-is.
    
    Priority:
    1. Element symbols (Cs, Rb, La - upper+lower) -> return as-is
    2. Database abbreviations (MA, FA - all uppercase) -> return N
    
    Parameters
    ----------
    ion : str
        Ion name (e.g., 'Cs', 'MA', 'Pb')
        
    Returns
    -------
    str
        Valid atomic symbol for ASE
    """
    # Priority 1: Check if it's a valid element symbol
    if _is_element_symbol(ion):
        return ion
    
    # Priority 2: Check if it's a known molecular cation (all uppercase)
    if ion in MOLECULAR_A_REPRESENTATIVE:
        return MOLECULAR_A_REPRESENTATIVE[ion]
    
    # Priority 3: Check database for other molecular cations
    if is_molecular_a_cation(ion):
        # Default to N for nitrogen-based organic cations
        return 'N'
    
    # Fallback: return as-is (will likely cause ASE error if invalid)
    return ion


def populate_structure(
    template: Atoms,
    A_ions: Union[str, List[str]],
    B_ions: Union[str, List[str]],
    X_ions: Union[str, List[str]],
    ratios: Optional[Dict[str, List[float]]] = None,
    seed: Optional[int] = None,
) -> Atoms:
    """
    Populate a template structure with actual elements.
    
    This replaces placeholder symbols (He, Ne, Ar) with real elements,
    supporting compositional mixing at any site.
    
    Parameters
    ----------
    template : Atoms
        Template structure with placeholder symbols and site_role tags
    A_ions : str or list of str
        A-site cation(s). Can be single element or list for mixing.
        Examples: 'Cs', ['Cs', 'MA'], ['Cs', 'FA', 'MA']
    B_ions : str or list of str
        B-site cation(s)
        Examples: 'Pb', ['Pb', 'Sn']
    X_ions : str or list of str
        X-site anion(s)
        Examples: 'I', ['I', 'Br'], ['I', 'Br', 'Cl']
    ratios : dict, optional
        Mixing ratios for each site type. Keys are 'A', 'B', 'X'.
        Values are lists of floats that must sum to 1.0.
        If not provided, equal mixing is assumed.
        Example: {'A': [0.5, 0.5], 'B': [0.8, 0.2], 'X': [0.6, 0.3, 0.1]}
    seed : int, optional
        Random seed for reproducible mixing
        
    Returns
    -------
    Atoms
        Structure with real elements
        
    Examples
    --------
    >>> # Simple case: single elements
    >>> populated = populate_structure(template, 'Cs', 'Pb', 'I')
    
    >>> # Mixed halides
    >>> populated = populate_structure(template, 'Cs', 'Pb', ['I', 'Br'],
    ...                                ratios={'X': [0.7, 0.3]})
    
    >>> # Triple cation
    >>> populated = populate_structure(template, 
    ...     ['Cs', 'FA', 'MA'], 'Pb', ['I', 'Br'],
    ...     ratios={'A': [0.05, 0.79, 0.16], 'X': [0.83, 0.17]})
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Normalize inputs to lists
    A_list = [A_ions] if isinstance(A_ions, str) else list(A_ions)
    B_list = [B_ions] if isinstance(B_ions, str) else list(B_ions)
    X_list = [X_ions] if isinstance(X_ions, str) else list(X_ions)
    
    # Default ratios: equal distribution
    if ratios is None:
        ratios = {}
    
    A_ratios = ratios.get('A', [1.0 / len(A_list)] * len(A_list))
    B_ratios = ratios.get('B', [1.0 / len(B_list)] * len(B_list))
    X_ratios = ratios.get('X', [1.0 / len(X_list)] * len(X_list))
    
    # Validate ratios sum to 1
    for name, r in [('A', A_ratios), ('B', B_ratios), ('X', X_ratios)]:
        if abs(sum(r) - 1.0) > 1e-6:
            raise ValueError(f"{name} ratios must sum to 1.0, got {sum(r)}")
    
    # Create populated structure
    atoms = template.copy()
    symbols = list(atoms.get_chemical_symbols())
    roles = atoms.arrays.get(SITE_ROLE_KEY)
    
    if roles is None:
        raise ValueError("Template has no site_role tags")
    
    # Populate each site type
    for site_type, ions, site_ratios, placeholder in [
        (SITE_A, A_list, A_ratios, TEMPLATE_A),
        (SITE_B, B_list, B_ratios, TEMPLATE_B),
        (SITE_X, X_list, X_ratios, TEMPLATE_X),
    ]:
        indices = get_site_indices(atoms, site_type)
        
        if len(indices) == 0:
            continue
        
        if len(ions) == 1:
            # Simple case: single element
            # For molecular A-site cations, use representative atom
            symbol = _get_representative_symbol(ions[0]) if site_type == SITE_A else ions[0]
            for idx in indices:
                symbols[idx] = symbol
        else:
            # Mixing case: randomly assign based on ratios
            assigned = np.random.choice(
                ions, 
                size=len(indices), 
                p=site_ratios
            )
            for idx, ion in zip(indices, assigned):
                # For molecular A-site cations, use representative atom
                symbol = _get_representative_symbol(ion) if site_type == SITE_A else ion
                symbols[idx] = symbol
    
    atoms.set_chemical_symbols(symbols)
    
    # Store composition info
    atoms.info['A_ions'] = A_ions
    atoms.info['B_ions'] = B_ions
    atoms.info['X_ions'] = X_ions
    if ratios:
        atoms.info['mixing_ratios'] = ratios
    
    return atoms


def get_composition(atoms: Atoms) -> Dict[str, Dict[str, int]]:
    """
    Get composition breakdown by site type.
    
    Parameters
    ----------
    atoms : Atoms
        Structure with site_role tags
        
    Returns
    -------
    dict
        Composition by site type.
        Example: {'A_site': {'Cs': 4, 'MA': 4}, 'B_site': {'Pb': 8}, ...}
    """
    roles = atoms.arrays.get(SITE_ROLE_KEY)
    symbols = atoms.get_chemical_symbols()
    
    if roles is None:
        return {}
    
    composition = {SITE_A: {}, SITE_B: {}, SITE_X: {}, SITE_SPACER: {}}
    
    for i, (role, symbol) in enumerate(zip(roles, symbols)):
        if role in composition:
            composition[role][symbol] = composition[role].get(symbol, 0) + 1
    
    # Remove empty site types
    return {k: v for k, v in composition.items() if v}


def substitute_at_site(
    atoms: Atoms,
    site_type: str,
    old_symbol: str,
    new_symbol: str,
    fraction: float = 1.0,
    seed: Optional[int] = None,
) -> Atoms:
    """
    Substitute elements at a specific site type.
    
    Parameters
    ----------
    atoms : Atoms
        Input structure with site_role tags
    site_type : str
        Site type to modify (SITE_A, SITE_B, SITE_X)
    old_symbol : str
        Element to replace
    new_symbol : str
        Element to substitute
    fraction : float
        Fraction of matching sites to substitute (0 to 1)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Atoms
        Modified structure
    """
    if seed is not None:
        np.random.seed(seed)
    
    result = atoms.copy()
    symbols = list(result.get_chemical_symbols())
    roles = result.arrays.get(SITE_ROLE_KEY)
    
    if roles is None:
        raise ValueError("Structure has no site_role tags")
    
    # Find matching atoms
    matching = []
    for i, (role, symbol) in enumerate(zip(roles, symbols)):
        if role == site_type and symbol == old_symbol:
            matching.append(i)
    
    if not matching:
        return result
    
    # Select subset to substitute
    n_substitute = int(len(matching) * fraction)
    if n_substitute == 0:
        return result
    
    selected = np.random.choice(matching, n_substitute, replace=False)
    
    for idx in selected:
        symbols[idx] = new_symbol
    
    result.set_chemical_symbols(symbols)
    return result


def revert_to_template(atoms: Atoms) -> Atoms:
    """
    Convert a populated structure back to a template.
    
    This replaces all site elements with placeholder symbols.
    
    Parameters
    ----------
    atoms : Atoms
        Structure with real elements and site_role tags
        
    Returns
    -------
    Atoms
        Template with placeholder symbols
    """
    result = atoms.copy()
    symbols = list(result.get_chemical_symbols())
    roles = result.arrays.get(SITE_ROLE_KEY)
    
    if roles is None:
        raise ValueError("Structure has no site_role tags")
    
    symbol_map = {
        SITE_A: TEMPLATE_A,
        SITE_B: TEMPLATE_B,
        SITE_X: TEMPLATE_X,
    }
    
    for i, role in enumerate(roles):
        if role in symbol_map:
            symbols[i] = symbol_map[role]
    
    result.set_chemical_symbols(symbols)
    return result


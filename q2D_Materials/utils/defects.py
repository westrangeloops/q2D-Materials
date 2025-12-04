"""
Defect operations for perovskite structures.

This module handles:
- Vacancy creation
- Elemental substitution
- Site-specific modifications
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase import Atoms

from .template import SITE_ROLE_KEY, SITE_A, SITE_B, SITE_X, get_site_indices


def add_vacancies(
    structure: Atoms,
    site_type: str,
    fraction: float = 0.1,
    seed: Optional[int] = None,
) -> Atoms:
    """
    Create vacancies at specified site type.
    
    Parameters
    ----------
    structure : Atoms
        Input structure with site_role tags
    site_type : str
        Site type to create vacancies in (SITE_A, SITE_B, or SITE_X)
    fraction : float
        Fraction of sites to remove (0 to 1)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Atoms
        Structure with vacancies
    """
    if seed is not None:
        np.random.seed(seed)
    
    atoms = structure.copy()
    roles = atoms.arrays.get(SITE_ROLE_KEY)
    
    if roles is None:
        raise ValueError("Structure has no site_role tags")
    
    # Find target sites
    target_indices = get_site_indices(atoms, site_type)
    n_remove = int(len(target_indices) * fraction)
    
    if n_remove == 0:
        return atoms
    
    # Randomly select sites to remove
    remove_indices = np.random.choice(target_indices, n_remove, replace=False)
    
    # Create mask for atoms to keep
    keep_mask = np.ones(len(atoms), dtype=bool)
    keep_mask[remove_indices] = False
    
    result = atoms[keep_mask]
    
    # Store vacancy info
    result.info['vacancies'] = {
        'site_type': site_type,
        'fraction': fraction,
        'n_removed': n_remove,
    }
    
    return result


def substitute_sites(
    structure: Atoms,
    old_symbol: str,
    new_symbol: str,
    site_type: Optional[str] = None,
    fraction: float = 1.0,
    seed: Optional[int] = None,
) -> Atoms:
    """
    Substitute atoms at specified sites.
    
    Parameters
    ----------
    structure : Atoms
        Input structure
    old_symbol : str
        Element symbol to replace
    new_symbol : str
        Element symbol to substitute
    site_type : str, optional
        Limit substitution to this site type
    fraction : float
        Fraction of matching sites to substitute (0 to 1)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Atoms
        Structure with substitutions
    """
    if seed is not None:
        np.random.seed(seed)
    
    atoms = structure.copy()
    symbols = list(atoms.get_chemical_symbols())
    roles = atoms.arrays.get(SITE_ROLE_KEY)
    
    # Find matching atoms
    match_mask = np.array([s == old_symbol for s in symbols])
    
    if site_type is not None and roles is not None:
        site_mask = roles == site_type
        match_mask &= site_mask
    
    match_indices = np.where(match_mask)[0]
    n_substitute = int(len(match_indices) * fraction)
    
    if n_substitute == 0:
        return atoms
    
    # Randomly select sites to substitute
    sub_indices = np.random.choice(match_indices, n_substitute, replace=False)
    
    # Perform substitution
    for idx in sub_indices:
        symbols[idx] = new_symbol
    
    atoms.set_chemical_symbols(symbols)
    
    # Store substitution info
    if 'substitutions' not in atoms.info:
        atoms.info['substitutions'] = []
    atoms.info['substitutions'].append({
        'old': old_symbol,
        'new': new_symbol,
        'site_type': site_type,
        'fraction': fraction,
        'n_substituted': n_substitute,
    })
    
    return atoms


def get_vacancy_concentration(
    original: Atoms,
    defected: Atoms,
    site_type: str,
) -> float:
    """
    Calculate vacancy concentration for a site type.
    
    Parameters
    ----------
    original : Atoms
        Original structure without vacancies
    defected : Atoms
        Structure with vacancies
    site_type : str
        Site type to check
        
    Returns
    -------
    float
        Vacancy concentration (0 to 1)
    """
    original_count = len(get_site_indices(original, site_type))
    defected_count = len(get_site_indices(defected, site_type))
    
    if original_count == 0:
        return 0.0
    
    return 1.0 - (defected_count / original_count)


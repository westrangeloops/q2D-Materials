"""
Perovskite Ion Recommender System

This module provides functionality to recommend compatible ions for perovskite structures
based on occurrence frequency in the perovskite database.
"""

import os
import pandas as pd
import re

# Cache for database
_perovskite_database = None


def _load_perovskite_database():
    """
    Load the perovskite ions database from CSV file and cache it.
    
    Returns
    -------
    pandas.DataFrame
        The loaded database with all ion information
    """
    global _perovskite_database
    
    if _perovskite_database is not None:
        return _perovskite_database
    
    # Get the path to the CSV file
    # Path: q2D_Materials/utils/other/recomender.py -> q2D_Materials/data/tables/Perovskite_ions_data.csv
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Load from central data/tables directory
    csv_path = os.path.join(current_dir, '..', '..', 'data', 'tables', 'Perovskite_ions_data.csv')
    
    try:
        _perovskite_database = pd.read_csv(csv_path)
        # Fill NaN values in occurrence column with 0
        _perovskite_database['number_of_occurances_in_the_perovskite_database'] = \
            _perovskite_database['number_of_occurances_in_the_perovskite_database'].fillna(0)
    except Exception as e:
        raise RuntimeError(f"Could not load perovskite database from CSV: {e}")
    
    return _perovskite_database


def _count_atoms_in_formula(formula):
    """
    Count total number of atoms in a molecular formula.
    
    Handles formats like "C8H12N+", "CH6N+", "C2H10N2+2", etc.
    Ignores charge indicators (+ and -) and counts all elements.
    
    Parameters
    ----------
    formula : str
        Molecular formula string (e.g., "C8H12N+", "CH6N3+")
        
    Returns
    -------
    int
        Total number of atoms (0 if formula is empty or invalid)
    """
    if pd.isna(formula) or not formula or str(formula).strip() == '':
        return 0
    
    formula = str(formula).strip()
    
    # Remove charge indicators (+ and -) and numbers after them
    formula = re.sub(r'[+-]\d*$', '', formula)
    formula = re.sub(r'[+-]$', '', formula)
    
    # Pattern to match element symbols followed by optional numbers
    # Matches: C, C8, H12, N, etc.
    pattern = r'([A-Z][a-z]?)(\d*)'
    
    total_atoms = 0
    for match in re.finditer(pattern, formula):
        element = match.group(1)
        count_str = match.group(2)
        count = int(count_str) if count_str else 1
        total_atoms += count
    
    return total_atoms


def _is_spacer(ion_data):
    """
    Determine if an A-site ion is a spacer based on atom count.
    
    Criteria: If molecular formula has >10 atoms, it's a spacer only.
    If ≤10 atoms, it can be used as both A-site and spacer.
    
    Parameters
    ----------
    ion_data : pandas.Series or dict
        Ion data row from database with 'molecular_formula' field
        
    Returns
    -------
    bool
        True if the ion is a spacer (>10 atoms), False if it can be both
    """
    formula = ion_data.get('molecular_formula', '')
    atom_count = _count_atoms_in_formula(formula)
    return atom_count > 10


def _get_ion_by_abbreviation(abbrev, ion_type=None):
    """
    Lookup ion by abbreviation or alternative abbreviation.
    
    Parameters
    ----------
    abbrev : str
        Ion abbreviation or alternative abbreviation
    ion_type : str, optional
        Filter by ion type ('A', 'B', or 'X'). If None, searches all types.
        
    Returns
    -------
    pandas.Series or None
        Ion data row if found, None otherwise
    """
    db = _load_perovskite_database()
    abbrev = str(abbrev).strip()
    
    # Filter by ion type if specified
    if ion_type:
        db = db[db['perovskite_ion_type'] == ion_type.upper()]
    
    # Try exact match on abbreviation
    match = db[db['abbreviation'] == abbrev]
    if not match.empty:
        return match.iloc[0]
    
    # Try case-insensitive match
    match = db[db['abbreviation'].str.upper() == abbrev.upper()]
    if not match.empty:
        return match.iloc[0]
    
    # Try alternative abbreviations
    for _, row in db.iterrows():
        alt_abbrevs = str(row.get('alternative_abbreviations', '')).strip()
        if alt_abbrevs and alt_abbrevs.lower() != 'nan':
            # Split by comma and check each
            for alt in alt_abbrevs.split(','):
                alt = alt.strip()
                if alt.upper() == abbrev.upper():
                    return row
    
    return None


def _rank_by_occurrence(ions, top_n=10, min_occurrences=1):
    """
    Sort ions by occurrence count and return top N.
    
    Parameters
    ----------
    ions : pandas.DataFrame
        DataFrame of ions to rank
    top_n : int
        Number of top recommendations to return
    min_occurrences : float
        Minimum occurrence count threshold
        
    Returns
    -------
    list
        List of dictionaries with ion information, sorted by occurrence
    """
    if ions.empty:
        return []
    
    # Filter by minimum occurrences
    ions = ions[ions['number_of_occurances_in_the_perovskite_database'] >= min_occurrences]
    
    # Sort by occurrence count (descending)
    ions = ions.sort_values('number_of_occurances_in_the_perovskite_database', ascending=False)
    
    # Convert to list of dictionaries
    results = []
    for _, row in ions.head(top_n).iterrows():
        result = {
            'abbreviation': str(row.get('abbreviation', '')),
            'common_name': str(row.get('common_name', '')),
            'occurrences': float(row.get('number_of_occurances_in_the_perovskite_database', 0)),
            'molecular_formula': str(row.get('molecular_formula', '')),
            'smile': str(row.get('smile', '')) if pd.notna(row.get('smile')) else '',
            'ion_type': str(row.get('perovskite_ion_type', ''))
        }
        results.append(result)
    
    return results


def get_recommendations(X=None, B=None, spacer=None, top_n=10, min_occurrences=1):
    """
    Get recommendations for perovskite ions based on user input.
    
    Parameters
    ----------
    X : str, optional
        X-site anion abbreviation
    B : str, optional
        B-site cation abbreviation
    spacer : str, optional
        Spacer molecule abbreviation
    top_n : int, default=10
        Number of recommendations per category
    min_occurrences : float, default=1
        Minimum occurrence count in database
        
    Returns
    -------
    dict
        Dictionary with keys 'A', 'B', 'X', 'spacer' containing lists of recommendations
    """
    if not any([X, B, spacer]):
        raise ValueError("At least one of X, B, or spacer must be provided")
    
    db = _load_perovskite_database()
    
    # Initialize result structure
    recommendations = {
        'A': [],
        'B': [],
        'X': [],
        'spacer': []
    }
    
    # Get all ions by type
    a_ions = db[db['perovskite_ion_type'] == 'A'].copy()
    b_ions = db[db['perovskite_ion_type'] == 'B'].copy()
    x_ions = db[db['perovskite_ion_type'] == 'X'].copy()
    
    # If X provided: recommend B, A, and spacers
    if X:
        # Verify X exists
        x_ion = _get_ion_by_abbreviation(X, 'X')
        if x_ion is None:
            raise ValueError(f"X-site ion '{X}' not found in database")
        
        recommendations['B'] = _rank_by_occurrence(b_ions, top_n, min_occurrences)
        recommendations['A'] = _rank_by_occurrence(a_ions, top_n, min_occurrences)
        
        # Separate spacers from A-site (spacers are A-site ions with >10 atoms)
        spacer_ions = a_ions[a_ions.apply(_is_spacer, axis=1)]
        recommendations['spacer'] = _rank_by_occurrence(spacer_ions, top_n, min_occurrences)
        recommendations['X'] = [{
            'abbreviation': str(x_ion.get('abbreviation', '')),
            'common_name': str(x_ion.get('common_name', '')),
            'occurrences': float(x_ion.get('number_of_occurances_in_the_perovskite_database', 0)),
            'molecular_formula': str(x_ion.get('molecular_formula', '')),
            'smile': str(x_ion.get('smile', '')) if pd.notna(x_ion.get('smile')) else '',
            'ion_type': 'X'
        }]
    
    # If B provided: recommend X, A, and spacers
    elif B:
        # Verify B exists
        b_ion = _get_ion_by_abbreviation(B, 'B')
        if b_ion is None:
            raise ValueError(f"B-site ion '{B}' not found in database")
        
        recommendations['X'] = _rank_by_occurrence(x_ions, top_n, min_occurrences)
        recommendations['A'] = _rank_by_occurrence(a_ions, top_n, min_occurrences)
        
        # Separate spacers from A-site
        spacer_ions = a_ions[a_ions.apply(_is_spacer, axis=1)]
        recommendations['spacer'] = _rank_by_occurrence(spacer_ions, top_n, min_occurrences)
        recommendations['B'] = [{
            'abbreviation': str(b_ion.get('abbreviation', '')),
            'common_name': str(b_ion.get('common_name', '')),
            'occurrences': float(b_ion.get('number_of_occurances_in_the_perovskite_database', 0)),
            'molecular_formula': str(b_ion.get('molecular_formula', '')),
            'smile': str(b_ion.get('smile', '')) if pd.notna(b_ion.get('smile')) else '',
            'ion_type': 'B'
        }]
    
    # If spacer provided: recommend A, B, X
    elif spacer:
        # Verify spacer exists (check in A-site ions)
        spacer_ion = _get_ion_by_abbreviation(spacer, 'A')
        if spacer_ion is None:
            raise ValueError(f"Spacer '{spacer}' not found in database")
        
        recommendations['A'] = _rank_by_occurrence(a_ions, top_n, min_occurrences)
        recommendations['B'] = _rank_by_occurrence(b_ions, top_n, min_occurrences)
        recommendations['X'] = _rank_by_occurrence(x_ions, top_n, min_occurrences)
        recommendations['spacer'] = [{
            'abbreviation': str(spacer_ion.get('abbreviation', '')),
            'common_name': str(spacer_ion.get('common_name', '')),
            'occurrences': float(spacer_ion.get('number_of_occurances_in_the_perovskite_database', 0)),
            'molecular_formula': str(spacer_ion.get('molecular_formula', '')),
            'smile': str(spacer_ion.get('smile', '')) if pd.notna(spacer_ion.get('smile')) else '',
            'ion_type': 'A'
        }]
    
    return recommendations


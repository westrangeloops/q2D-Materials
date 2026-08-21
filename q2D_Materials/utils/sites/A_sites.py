"""
Common A-site Cations for Perovskite Structures

This module provides ionic radii data for common A-site cations and their
aliases, along with utilities to create molecular A-site cations using SMILES.

References:
[1] https://doi.org/10.1021/acs.chemrev.8b00539
[2] https://doi.org/10.1021/acs.chemmater.9b05273  
[3] https://doi.org/10.1107/S0567739476001551
"""

import tempfile
import os
import numpy as np
import pandas as pd
from ase.io import read

# Import pymatgen for periodic table functionality
from pymatgen.core.periodic_table import Element

# Import RDKit
from rdkit import Chem

# Import SMILES conversion function from molecules package
from q2D_Materials.utils.molecules.molecule_builder import smiles_to_ase_atoms

# Cache for CSV data
_a_ion_database = None
_a_ion_lookup = None

# Ionic radii data for A-site cations
ionic_radii = {
    "A":{
        # This is completely arbitrary choice of Cs for the perovskites
        # that don't contain an A-site cation (e.g. n = 1 2DPKs).
        "NA"   : 1.88, # See Note above

        "NH4"  : 1.46, # [1]
        "MA"   : 2.17, # [1]
        "FA"   : 2.53, # [1]
        "HZA"  : 2.17, # [1]
        "AZ"   : 2.50, # [1]
        "HXA"  : 2.16, # [1]
        "IMA"  : 2.58, # [1]
        "EA"   : 2.74, # [1]
        "DMA"  : 2.72, # [1]
        "GA"   : 2.78, # [1]
        "TMA"  : 2.92, # [1]
        "TA"   : 3.20, # [1]
        "3-PYR": 2.72, # [1]
        "TPY"  : 3.33, # [1]
        "K"    : 1.64, # [1]
        "Rb"   : 1.72, # [1]
        "Cs"   : 1.88, # [1]
        "MHy"  : 2.64, # [2]
    },
    "B":{
        "Pb"   : 1.19, # [1]
        "Sn"   : 1.10, # [1]
        "Ge"   : 0.73, # [1]
        "Mg"   : 0.72, # [1]
        "Ca"   : 1.00, # [1]
        "Sr"   : 1.18, # [1]
        "Ba"   : 1.35, # [1]
        "Cu"   : 0.73, # [1]
        "Fe"   : 0.78, # [1]
        "Pd"   : 0.86, # [1]
        "Eu"   : 1.17, # [1]
        "Bi"   : 1.03, # [1]
        "Sb3+" : 0.76, # [1]
        "Co"   : 0.79, # [3] 
        "Hg"   : 1.16, # [3]
        "Zn"   : 0.88, # [3]
        "Cd"   : 1.09, # [3]
    },
    "X":{
        "F"    : 1.29, # [1]
        "Cl"   : 1.81, # [1]
        "Br"   : 1.96, # [1]
        "I"    : 2.20, # [1]
    },
}

A_cation_aliases = {
    "NH4"  : ["ammonium", "ammonium cation"],
    "MA"   : ["methylammonium", "[CH3NH3]+", "CH3NH3"],
    "FA"   : ["formamidinium", "[CH(NH2)2]+", "CH(NH2)2"],
    "HZA"  : ["hydrazinium", "[NH3NH2]+", "NH3NH2"],
    "AZ"   : ["azetidinium", "[(CH2)3NH2]+", "(CH2)3NH2"],
    "HXA"  : ["hydroxylammonium", "[NH3OH]+", "NH3OH"],
    "IMA"  : ["imidazolium", "[C3N2H5]+", "[C3N2H5]"],
    "EA"   : ["ethylammonium", "[(CH3CH2)NH3]+", "(CH3CH2)NH3"],
    "DMA"  : ["dimethylammonium", "[(CH3)2NH2]+", "(CH3)2NH2"],
    "GA"   : ["guanidinium", "[(NH2)3C]+", "(NH2)3C"],
    "TMA"  : ["tetramethylammonium", "[(CH3)4N]+", "(CH3)4N"],
    "TA"   : ["thiazolium", "[C3H4NS]+", "C3H4NS"],
    "3-PYR": ["3-pyrrolinium", "[NC4H8]+", "NC4H8"],
    "TPY"  : ["tropylium", "[C7H7]+", "C7H7"],
    "MHy"  : ["methylhydrazinium", "[CH7N2]+", "CH7N2"],
    "K"    : ["K+", "potassium"],
    "Cs"   : ["Cs+", "cesium"],
    "Rb"   : ["Rb+", "rubidium"],
}

# SMILES strings for common A-site molecular cations (fallback if CSV not available)
A_cation_smiles = {
    "MA": "C[NH3+]",  # Methylammonium
    "FA": "C(=[NH2+])[NH3+]",  # Formamidinium
    "EA": "CC[NH3+]",  # Ethylammonium
    "DMA": "C[NH2+]C",  # Dimethylammonium
    "GA": "C(=[NH2+])([NH3+])[NH3+]",  # Guanidinium
    "NH4": "[NH4+]",  # Ammonium
}


def _load_a_ion_database():
    """
    Load the A-ion database from CSV file and create lookup dictionaries.
    
    Returns
    -------
    tuple
        (dataframe, lookup_dict) where lookup_dict maps abbreviations to SMILES
    """
    global _a_ion_database, _a_ion_lookup
    
    if _a_ion_database is not None:
        return _a_ion_database, _a_ion_lookup
    
    # Get the path to the CSV file from central data/tables directory
    # Path: q2D_Materials/utils/sites/A_sites.py -> q2D_Materials/data/tables/A-ion_data.csv
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', '..', 'data', 'tables', 'A-ion_data.csv')
    
    try:
        # Load CSV file
        _a_ion_database = pd.read_csv(csv_path)
        
        # Create lookup dictionary: abbreviation -> SMILES
        _a_ion_lookup = {}
        
        for _, row in _a_ion_database.iterrows():
            # Get abbreviation
            abbrev_val = row['Abbreviation']
            abbrev = str(abbrev_val).strip() if pd.notna(abbrev_val) and str(abbrev_val).strip() != '' else None
            
            # Get alternative abbreviations
            alt_abbrevs_val = row['Alternative_abbreviations']
            alt_abbrevs = str(alt_abbrevs_val).strip() if pd.notna(alt_abbrevs_val) and str(alt_abbrevs_val).strip() != '' else None
            
            # Get SMILES
            smiles_val = row['SMILE']
            smiles = str(smiles_val).strip() if pd.notna(smiles_val) and str(smiles_val).strip() != '' and str(smiles_val).strip().lower() != 'nan' else None
            
            # Add main abbreviation
            if abbrev and smiles:
                _a_ion_lookup[abbrev.upper()] = smiles
                _a_ion_lookup[abbrev] = smiles  # Also store original case
            
            # Add alternative abbreviations
            if alt_abbrevs and smiles:
                # Alternative_abbreviations can be comma-separated
                for alt in alt_abbrevs.split(','):
                    alt = alt.strip()
                    if alt and alt.lower() != 'nan':
                        _a_ion_lookup[alt.upper()] = smiles
                        _a_ion_lookup[alt] = smiles  # Also store original case
        
    except Exception as e:
        # If CSV loading fails, use empty database
        print(f"Warning: Could not load A-ion database from CSV: {e}")
        print("Falling back to hardcoded SMILES data.")
        _a_ion_database = pd.DataFrame()
        _a_ion_lookup = {}
    
    return _a_ion_database, _a_ion_lookup


def _lookup_smiles_from_database(a_cation):
    """
    Look up SMILES string for an A-site cation from the CSV database.
    
    Case-sensitive lookup: 'Ca' (Calcium) and 'CA' (azetidinium) are different.
    
    Parameters
    ----------
    a_cation : str
        A-site cation abbreviation or alternative abbreviation
        
    Returns
    -------
    str or None
        SMILES string if found, None otherwise
    """
    _, lookup = _load_a_ion_database()
    
    # Try exact case match first (case-sensitive)
    if a_cation in lookup:
        return lookup[a_cation]
    
    # Only try uppercase if exact match fails
    # This allows 'CA' to match but prevents 'Ca' from matching 'CA'
    a_cation_upper = a_cation.upper()
    if a_cation_upper in lookup and a_cation_upper != a_cation:
        # Only use uppercase match if it's different from the original
        # This prevents 'Ca' from matching 'CA' in the database
        return lookup[a_cation_upper]
    
    return None


def get_ionic_radius(site, name):
    """
    Get the ionic radius for a specific ion at a specific site.
    
    Uses hardcoded database first, then falls back to pymatgen if available.
    
    Parameters
    ----------
    site : str
        The crystallographic site ('A', 'B', or 'X')
    name : str
        The name or symbol of the ion
        
    Returns
    -------
    float
        Ionic radius in Angstroms
        
    Raises
    ------
    ValueError
        If the ion is not found in the database and pymatgen fallback fails
    """
    site_dict = ionic_radii.get(site)
    if site_dict is None:
        raise ValueError(f"Invalid site '{site}'. Must be 'A', 'B', or 'X'")
    
    val = site_dict.get(name, None)
    if val is not None:
        return val
    
    # Fallback to pymatgen (provides ionic radii - correct for perovskites)
    try:
        # Get ionic radius from pymatgen
        # Note: pymatgen's ionic radii depend on coordination number and oxidation state
        # We'll use a reasonable default coordination number based on the site
        element = Element(name)
        
        # Try to get ionic radius with typical coordination numbers
        # For perovskites: A-site typically 12-coordinated, B-site 6-coordinated, X-site 6-coordinated
        coord_map = {"A": 12, "B": 6, "X": 6}
        coordination = coord_map.get(site, 6)
        
        # Try common oxidation states for each site
        # Note: pymatgen's ionic_radii[oxi_state] returns a FloatWithUnit, not a dict
        if site == "A":
            # A-site: typically +1 (alkali metals) or +2 (alkaline earth)
            for oxi_state in [1, 2]:
                try:
                    radius_obj = element.ionic_radii.get(oxi_state)
                    if radius_obj is not None:
                        # Convert FloatWithUnit to float (already in Angstroms)
                        return float(radius_obj)
                except (KeyError, AttributeError, TypeError, ValueError):
                    continue
        elif site == "B":
            # B-site: typically +2, +3, +4, or +5
            for oxi_state in [2, 3, 4, 5]:
                try:
                    radius_obj = element.ionic_radii.get(oxi_state)
                    if radius_obj is not None:
                        # Convert FloatWithUnit to float (already in Angstroms)
                        return float(radius_obj)
                except (KeyError, AttributeError, TypeError, ValueError):
                    continue
        elif site == "X":
            # X-site: typically -1 (halides: F-, Cl-, Br-, I-) or -2 (oxides: O^2-)
            for oxi_state in [-1, -2]:
                try:
                    radius_obj = element.ionic_radii.get(oxi_state)
                    if radius_obj is not None:
                        # Convert FloatWithUnit to float (already in Angstroms)
                        return float(radius_obj)
                except (KeyError, AttributeError, TypeError, ValueError):
                    continue
        
        # If no ionic radius found, try to get atomic radius as fallback
        try:
            atomic_radius = element.atomic_radius
            if atomic_radius is not None:
                # Convert from pm to Angstroms (divide by 100)
                return atomic_radius / 100.0
        except (AttributeError, TypeError):
            pass
                
    except (ValueError, KeyError, AttributeError):
        # Element not found or no radius data available
        pass
    
    # If all fallbacks fail, raise error
    raise ValueError(f"Ion '{name}' not found for site '{site}'")


def create_a_site_molecule(a_cation):
    """
    Create an ASE Atoms object for an A-site molecular cation using SMILES.
    
    First tries to look up the SMILES from the CSV database (A-ion_data.csv),
    then falls back to hardcoded SMILES if not found.
    
    Parameters
    ----------
    a_cation : str
        A-site cation symbol (e.g., 'MA', 'FA', 'EA', 'PEA', etc.)
        
    Returns
    -------
    ase.Atoms
        ASE Atoms object of the A-site molecular cation
        
    Raises
    ------
    ValueError
        If the A-site cation is not supported or RDKit is not available
    """
    
    # First, try to look up SMILES from CSV database
    smiles = _lookup_smiles_from_database(a_cation)
    
    # If not found in database, try hardcoded SMILES
    if smiles is None:
        if a_cation in A_cation_smiles:
            smiles = A_cation_smiles[a_cation]
        else:
            raise ValueError(
                f"A-site cation '{a_cation}' not found in database or hardcoded list. "
                f"Available in hardcoded: {list(A_cation_smiles.keys())}"
            )
    
    # Convert SMILES directly to ASE Atoms
    if smiles_to_ase_atoms is None:
        raise ValueError("smiles_to_ase_atoms function is not available.")
    
    try:
        molecule = smiles_to_ase_atoms(smiles)
        return molecule
    except Exception as e:
        raise ValueError(f"Failed to create molecule from SMILES '{smiles}' for cation '{a_cation}': {e}")


def is_molecular_a_cation(a_cation):
    """
    Check if an A-site cation is molecular (requires SMILES) or atomic.
    
    Case-sensitive: 'Ca' (Calcium, atomic) and 'CA' (azetidinium, molecular) are different.
    Atomic elements are never treated as molecular, regardless of database entries.
    
    Parameters
    ----------
    a_cation : str
        A-site cation symbol
        
    Returns
    -------
    bool
        True if molecular (has SMILES and is not a simple atomic ion), False if atomic
    """
    # First, check if it's a valid atomic element symbol (case-sensitive)
    # This prevents atomic elements from being matched as molecular cations
    try:
        # Try to create an Element object - this will raise ValueError if not a valid element
        # Case-sensitive: 'Ca' is valid, 'CA' is not
        Element(a_cation)
        # If we get here, it's a valid element symbol - treat as atomic
        return False
    except (ValueError, KeyError):
        # Not a valid element symbol, continue to check if it's molecular
        pass
    
    # Check if it's in the database
    smiles = _lookup_smiles_from_database(a_cation)
    if smiles:
        # Check if it's a simple atomic ion (SMILES like [Cs+], [K+], etc.)
        # These are typically single atoms in brackets
        if smiles.strip().startswith('[') and smiles.strip().endswith(']'):
            # Check if it's just a single element with charge
            inner = smiles.strip()[1:-1].strip()
            # Simple atomic ions: [Cs+], [K+], [Rb+], [Na+], [Li+], etc.
            if len(inner) <= 3 and ('+' in inner or '-' in inner):
                return False
        return True
    
    # Fall back to hardcoded check
    return a_cation in A_cation_smiles


def get_a_site_object(a_cation):
    """
    Get the appropriate A-site object (string for atomic, Atoms for molecular).
    
    First tries to look up the cation in the CSV database. If found with SMILES,
    creates an Atoms object. If it's an atomic cation, returns the string.
    
    Parameters
    ----------
    a_cation : str
        A-site cation symbol or abbreviation
        
    Returns
    -------
    str or ase.Atoms
        String for atomic cations, ASE Atoms object for molecular cations
    """
    # Check if it's molecular (has SMILES in database or hardcoded)
    if is_molecular_a_cation(a_cation):
        try:
            return create_a_site_molecule(a_cation)
        except ValueError:
            # If creation fails, might be atomic - return as string
            return a_cation
    else:
        # Return as string for atomic cations (K, Cs, Rb, etc.)
        return a_cation


def calculate_BX_distance(B_cation, X_anion, mode: str = "mean"):
    """
    Calculate B–X bond distance(s) with optional aggregation modes.

    Accepts single species or iterables for B and X. When lists/arrays are
    provided, all pairwise combinations are evaluated using vectorized numpy
    operations, then reduced with the requested mode.

    Parameters
    ----------
    B_cation : str or iterable[str]
        B-site cation symbol(s)
    X_anion : str or iterable[str]
        X-site anion symbol(s)
    mode : str, optional
        One of 'mean', 'min', 'max', or 'single'. Defaults to 'mean'.

    Returns
    -------
    float
        Aggregated B–X distance in Å (r_B + r_X + 0.20 Å buffer)
    """
    def _radius_array(site, species):
        if isinstance(species, (str, bytes)):
            names = [species]
        else:
            try:
                names = list(species)
            except TypeError as exc:
                raise ValueError(f"{site}-site species must be a string or iterable.") from exc
        if not names:
            raise ValueError(f"{site}-site species list cannot be empty.")
        return np.asarray([get_ionic_radius(site, name) for name in names], dtype=float)

    radii_B = _radius_array("B", B_cation)
    radii_X = _radius_array("X", X_anion)

    distances = radii_B[:, None] + radii_X[None, :] + 0.20  # broadcast all pairs

    mode_normalized = "mean" if not isinstance(mode, str) else mode.lower()
    if distances.size == 1 or mode_normalized == "single":
        return float(distances.ravel()[0])
    if mode_normalized == "mean":
        return float(distances.mean())
    if mode_normalized == "min":
        return float(distances.min())
    if mode_normalized == "max":
        return float(distances.max())

    raise ValueError(f"Unknown BX distance mode '{mode}'. Use 'mean', 'min', 'max', or 'single'.")


def calculate_tolerance_factor(A_cation, B_cation, X_anion):
    """
    Calculate the Goldschmidt tolerance factor for a perovskite composition.
    
    The tolerance factor t = (r_A + r_X) / (√2 * (r_B + r_X))
    where r_A, r_B, r_X are the ionic radii.
    
    Parameters
    ----------
    A_cation : str
        A-site cation symbol
    B_cation : str
        B-site cation symbol  
    X_anion : str
        X-site anion symbol
        
    Returns
    -------
    float
        Goldschmidt tolerance factor
    """
    try:
        r_A = get_ionic_radius("A", A_cation)
        r_B = get_ionic_radius("B", B_cation)
        r_X = get_ionic_radius("X", X_anion)
        
        tolerance_factor = (r_A + r_X) / (2**0.5 * (r_B + r_X))
        return tolerance_factor
    except ValueError as e:
        print(f"Error calculating tolerance factor: {e}")
        raise


def print_available_a_cations():
    """
    Print all available A-site cations and their types.
    """
    print("Available A-site Cations:")
    print("=" * 40)
    print("Atomic Cations:")
    for cation in ionic_radii["A"].keys():
        if cation not in A_cation_smiles:
            aliases = A_cation_aliases.get(cation, [])
            print(f"  {cation}: {aliases}")
    
    print("\nMolecular Cations (from SMILES):")
    for cation in A_cation_smiles.keys():
        aliases = A_cation_aliases.get(cation, [])
        smiles = A_cation_smiles[cation]
        print(f"  {cation}: {aliases} (SMILES: {smiles})")
    print("=" * 40)
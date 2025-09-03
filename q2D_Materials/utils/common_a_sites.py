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
from ase.io import read
from .smiles_to_xyz import smiles_to_xyz, RDKIT_AVAILABLE

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

# SMILES strings for common A-site molecular cations
A_cation_smiles = {
    "MA": "C[NH3+]",  # Methylammonium
    "FA": "C(=[NH2+])[NH3+]",  # Formamidinium
    "EA": "CC[NH3+]",  # Ethylammonium
    "DMA": "C[NH2+]C",  # Dimethylammonium
    "GA": "C(=[NH2+])([NH3+])[NH3+]",  # Guanidinium
    "NH4": "[NH4+]",  # Ammonium
}


def get_ionic_radius(site, name):
    """
    Get the ionic radius for a specific ion at a specific site.
    
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
        If the ion is not found in the database
    """
    site_dict = ionic_radii.get(site)
    if site_dict is None:
        raise ValueError(f"Invalid site '{site}'. Must be 'A', 'B', or 'X'")
    
    val = site_dict.get(name, None)
    if val is not None:
        return val
    else:
        raise ValueError(f"Ion '{name}' not found for site '{site}'")


def create_a_site_molecule(a_cation):
    """
    Create an ASE Atoms object for an A-site molecular cation using SMILES.
    
    Parameters
    ----------
    a_cation : str
        A-site cation symbol (e.g., 'MA', 'FA', 'EA')
        
    Returns
    -------
    ase.Atoms
        ASE Atoms object of the A-site molecular cation
        
    Raises
    ------
    ValueError
        If the A-site cation is not supported or RDKit is not available
    """
    if not RDKIT_AVAILABLE:
        raise ValueError("RDKit is not available. Cannot create molecular A-site cations from SMILES.")
    
    if a_cation not in A_cation_smiles:
        raise ValueError(f"A-site cation '{a_cation}' not supported. Available: {list(A_cation_smiles.keys())}")
    
    smiles = A_cation_smiles[a_cation]
    
    # Create temporary XYZ file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.xyz') as tmp_file:
        try:
            smiles_to_xyz(smiles, tmp_file.name)
            # Read the XYZ file as ASE Atoms object
            molecule = read(tmp_file.name)
            return molecule
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)


def is_molecular_a_cation(a_cation):
    """
    Check if an A-site cation is molecular (requires SMILES) or atomic.
    
    Parameters
    ----------
    a_cation : str
        A-site cation symbol
        
    Returns
    -------
    bool
        True if molecular, False if atomic
    """
    return a_cation in A_cation_smiles


def get_a_site_object(a_cation):
    """
    Get the appropriate A-site object (string for atomic, Atoms for molecular).
    
    Parameters
    ----------
    a_cation : str
        A-site cation symbol
        
    Returns
    -------
    str or ase.Atoms
        String for atomic cations, ASE Atoms object for molecular cations
    """
    if is_molecular_a_cation(a_cation):
        return create_a_site_molecule(a_cation)
    else:
        # Return as string for atomic cations (K, Cs, Rb, etc.)
        return a_cation


def calculate_BX_distance(B_cation, X_anion):
    """
    Calculate the B-X bond distance based on ionic radii.
    
    Parameters
    ----------
    B_cation : str
        B-site cation symbol
    X_anion : str
        X-site anion symbol
        
    Returns
    -------
    float
        B-X bond distance in Angstroms
    """
    try:
        r_B = get_ionic_radius("B", B_cation)
        r_X = get_ionic_radius("X", X_anion)
        return r_B + r_X
    except ValueError as e:
        print(f"Error calculating B-X distance: {e}")
        raise


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

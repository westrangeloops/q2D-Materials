"""
SMILES to XYZ conversion utilities

This module provides functionality to convert SMILES strings to XYZ coordinates
using RDKit for molecular structure generation.
"""

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def smiles_to_xyz(smiles, output_file):
    """
    Convert a SMILES string to XYZ format.
    
    Parameters
    ----------
    smiles : str
        SMILES string representing the molecule
    output_file : str
        Path to output XYZ file
        
    Raises
    ------
    ImportError
        If RDKit is not available
    ValueError
        If SMILES string is invalid
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMILES to XYZ conversion. Please install rdkit-pypi.")
    
    # Create molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    
    # Get conformer
    conf = mol.GetConformer()
    
    # Write XYZ file
    with open(output_file, 'w') as f:
        num_atoms = mol.GetNumAtoms()
        f.write(f"{num_atoms}\n")
        f.write(f"Generated from SMILES: {smiles}\n")
        
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            symbol = atom.GetSymbol()
            f.write(f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")


def smiles_to_ase_atoms(smiles):
    """
    Convert a SMILES string directly to an ASE Atoms object.
    
    Parameters
    ----------
    smiles : str
        SMILES string representing the molecule
        
    Returns
    -------
    ase.Atoms
        ASE Atoms object
        
    Raises
    ------
    ImportError
        If RDKit is not available
    ValueError
        If SMILES string is invalid
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMILES conversion. Please install rdkit-pypi.")
    
    from ase import Atoms
    import tempfile
    import os
    
    # Create temporary XYZ file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.xyz') as tmp_file:
        try:
            smiles_to_xyz(smiles, tmp_file.name)
            # Read back as ASE Atoms object
            from ase.io import read
            atoms = read(tmp_file.name)
            return atoms
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)

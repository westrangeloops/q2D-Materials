"""
Handles conversion of SMILES strings to 3D molecular structures using RDKit.
This module is designed to be optional. If RDKit is not installed,
functionality will be disabled gracefully.
"""
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

def smiles_to_xyz(smiles: str, filename: str):
    """
    Converts a SMILES string to an XYZ file.

    Args:
        smiles: The SMILES string of the molecule.
        filename: The path to save the output XYZ file.
        
    Raises:
        ImportError: If RDKit is not installed.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is not installed. This functionality is unavailable.")

    # Create a molecule object from the SMILES string
    mol = Chem.MolFromSmiles(smiles)
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.UFFOptimizeMolecule(mol)
    
    # Get atom symbols and coordinates
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = mol.GetConformer().GetPositions()
    
    # Write to XYZ file
    with open(filename, 'w') as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"Molecule created from SMILES: {smiles}\n")
        for symbol, coord in zip(symbols, coords):
            f.write(f"{symbol} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n") 
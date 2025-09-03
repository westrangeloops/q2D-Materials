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

def smiles_to_xyz(smiles: str, filename: str, optimize_geometry=True, max_attempts=5):
    """
    Converts a SMILES string to an XYZ file with robust error handling.
    Enhanced with multiple optimization strategies and validation.

    Args:
        smiles: The SMILES string of the molecule.
        filename: The path to save the output XYZ file.
        optimize_geometry: Whether to optimize molecular geometry.
        max_attempts: Maximum number of embedding attempts.
        
    Raises:
        ImportError: If RDKit is not installed.
        ValueError: If SMILES string is invalid.
        RuntimeError: If 3D coordinate generation fails.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is not installed. This functionality is unavailable.")

    import numpy as np

    # Create a molecule object from the SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Validate molecule size
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        raise ValueError("Molecule has no atoms after hydrogen addition")
    if num_atoms > 1000:
        print(f"Warning: Large molecule with {num_atoms} atoms - this may take time")
    
    # Try multiple embedding methods with different parameters
    embedding_methods = [
        ("ETKDGv3", lambda: AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())),
        ("ETKDGv2", lambda: AllChem.EmbedMolecule(mol, AllChem.ETKDGv2())),
        ("Basic with seed", lambda: AllChem.EmbedMolecule(mol, randomSeed=42)),
        ("Distance Geometry", lambda: AllChem.EmbedMolecule(mol)),
        ("Multiple conformers", lambda: AllChem.EmbedMultipleConfs(mol, numConfs=1, randomSeed=123)),
    ]
    
    success = False
    method_used = None
    
    for attempt in range(max_attempts):
        for method_name, method_func in embedding_methods:
            try:
                result = method_func()
                if result == 0 or (isinstance(result, int) and result >= 0):  # Success
                    # Check for NaN coordinates
                    coords = mol.GetConformer().GetPositions()
                    if not np.any(np.isnan(coords)) and not np.any(np.isinf(coords)):
                        success = True
                        method_used = method_name
                        break
            except Exception as e:
                continue
        if success:
            break
        
        # If first attempt failed, try with different random seed
        for method_name, _ in embedding_methods[:2]:  # Try top methods with new seed
            try:
                if method_name == "ETKDGv3":
                    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(randomSeed=attempt*1000))
                elif method_name == "ETKDGv2":
                    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv2(randomSeed=attempt*1000))
                
                if result == 0:
                    coords = mol.GetConformer().GetPositions()
                    if not np.any(np.isnan(coords)) and not np.any(np.isinf(coords)):
                        success = True
                        method_used = f"{method_name} (attempt {attempt+1})"
                        break
            except Exception:
                continue
        if success:
            break
    
    if not success:
        raise RuntimeError(f"Failed to generate valid 3D coordinates for SMILES: {smiles} after {max_attempts} attempts")
    
    print(f"3D coordinates generated using {method_used}")
    
    # Try to optimize the geometry (optional)
    if optimize_geometry:
        try:
            # Try multiple optimization methods
            optimization_success = False
            
            # Method 1: UFF optimization
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
                coords = mol.GetConformer().GetPositions()
                if not np.any(np.isnan(coords)) and not np.any(np.isinf(coords)):
                    optimization_success = True
                    print("Geometry optimized using UFF")
            except:
                pass
            
            # Method 2: MMFF optimization (fallback)
            if not optimization_success:
                try:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
                    coords = mol.GetConformer().GetPositions()
                    if not np.any(np.isnan(coords)) and not np.any(np.isinf(coords)):
                        optimization_success = True
                        print("Geometry optimized using MMFF")
                except:
                    pass
            
            if not optimization_success:
                print("Warning: Geometry optimization failed, using unoptimized structure")
                
        except Exception as e:
            print(f"Warning: Geometry optimization failed ({e}), using unoptimized structure")
    
    # Final coordinate validation
    coords = mol.GetConformer().GetPositions()
    if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
        raise RuntimeError(f"Generated coordinates contain invalid values for SMILES: {smiles}")
    
    # Validate reasonable coordinate ranges
    coord_range = np.ptp(coords, axis=0)  # Range in each dimension
    if np.any(coord_range > 100):  # More than 100 Å in any dimension seems unreasonable
        print(f"Warning: Large molecular dimensions detected: {coord_range}")
    
    # Get atom symbols and coordinates
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    # Validate symbols
    if len(symbols) != len(coords):
        raise RuntimeError("Mismatch between number of atoms and coordinates")
    
    # Write to XYZ file
    try:
        with open(filename, 'w') as f:
            f.write(f"{len(symbols)}\n")
            f.write(f"Molecule created from SMILES: {smiles} using {method_used}\n")
            for symbol, coord in zip(symbols, coords):
                f.write(f"{symbol} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n")
        
        print(f"Successfully wrote {len(symbols)} atoms to {filename}")
        
    except IOError as e:
        raise RuntimeError(f"Failed to write XYZ file: {e}")


def validate_smiles(smiles: str) -> bool:
    """
    Validate a SMILES string without generating coordinates.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not RDKIT_AVAILABLE:
        return False
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def get_molecular_info(smiles: str) -> dict:
    """
    Get basic molecular information from SMILES.
    
    Args:
        smiles: SMILES string
        
    Returns:
        dict: Molecular information (formula, weight, etc.)
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is not installed")
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    mol_with_h = Chem.AddHs(mol)
    
    return {
        'formula': Chem.rdMolDescriptors.CalcMolFormula(mol_with_h),
        'molecular_weight': Chem.rdMolDescriptors.CalcExactMolWt(mol_with_h),
        'num_atoms': mol_with_h.GetNumAtoms(),
        'num_heavy_atoms': mol.GetNumHeavyAtoms(),
        'smiles_canonical': Chem.MolToSmiles(mol)
    } 
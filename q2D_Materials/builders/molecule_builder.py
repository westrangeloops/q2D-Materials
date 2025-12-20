"""
Molecule Builder - Unified module for SMILES conversion and molecular alignment.

This module combines:
- SMILES string to ASE Atoms conversion
- Molecular alignment for perovskite structures
- Special handling for small molecules like NH3
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from ase import Atoms

# Try to import RDKit for SMILES support
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ============================================================================
# SMILES Conversion Functions
# ============================================================================

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


def smiles_to_ase_atoms(smiles: str):
    """
    Convert a SMILES string directly to an ASE Atoms object.
    
    This function uses the robust smiles_to_xyz() function internally
    to generate 3D coordinates, then converts them to an ASE Atoms object.
    
    Parameters
    ----------
    smiles : str
        SMILES string representing the molecule
        
    Returns
    -------
    ase.Atoms
        ASE Atoms object with 3D coordinates
        
    Raises
    ------
    ImportError
        If RDKit is not available
    ValueError
        If SMILES string is invalid
    RuntimeError
        If 3D coordinate generation fails
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMILES conversion. Please install rdkit-pypi.")
    
    import tempfile
    import os
    import time
    
    # Create temporary XYZ file
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.xyz') as tmp_file:
            tmp_file_path = tmp_file.name
            # Use the robust smiles_to_xyz function
            smiles_to_xyz(smiles, tmp_file_path, optimize_geometry=True)
        
        # Read back as ASE Atoms object after file is closed
        # This ensures the file handle is released on Windows
        from ase.io import read
        atoms = read(tmp_file_path)
        return atoms
    finally:
        # Clean up temporary file with retry logic for Windows
        if tmp_file_path and os.path.exists(tmp_file_path):
            # On Windows, file deletion may fail if still in use
            # Retry a few times with small delays
            for attempt in range(5):
                try:
                    os.remove(tmp_file_path)
                    break
                except (PermissionError, OSError):
                    if attempt < 4:
                        time.sleep(0.1)
                    else:
                        # If still can't delete, just leave it (temp files will be cleaned up eventually)
                        pass


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


# ============================================================================
# Molecular Alignment Functions
# ============================================================================

def _align_nitrogens(molecule_df, tar='Z'):
    """
    Align two nitrogen atoms along a target direction.
    
    Parameters
    ----------
    molecule_df : pd.DataFrame
        Molecule data with 'Element', 'X', 'Y', 'Z' columns
    tar : str or array
        Target direction ('Z' for z-axis) or custom vector
        
    Returns
    -------
    pd.DataFrame
        Aligned molecule data
    """
    nitrogen_atoms = molecule_df[molecule_df['Element'] == 'N']
    if len(nitrogen_atoms) < 2:
        # Not enough nitrogen atoms to align, return original DataFrame
        return molecule_df

    # Find the coordinates of the two Nitrogen atoms
    n1_coords = nitrogen_atoms[['X', 'Y', 'Z']].iloc[0].values
    n2_coords = nitrogen_atoms[['X', 'Y', 'Z']].iloc[1].values

    # Calculate the vector between the two Nitrogen atoms
    v_ref = n1_coords - n2_coords

    if tar == 'Z':
        # Define the target vector (aligned with the z-axis)
        v_tar = np.array([0, 0, np.linalg.norm(v_ref)])
    else:
        v_tar = tar
    
    # Check for zero-length vectors
    if np.linalg.norm(v_ref) < 1e-10 or np.linalg.norm(v_tar) < 1e-10:
        return molecule_df # Cannot align, return original

    # Calculate the rotation matrix that transforms v_ref to v_tar
    cos_theta = np.dot(v_ref, v_tar) / (np.linalg.norm(v_ref) * np.linalg.norm(v_tar))
    
    # Clamp value to avoid numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    if np.isclose(cos_theta, 1.0):
        # Already aligned
        return molecule_df
    elif np.isclose(cos_theta, -1.0):
        # Anti-aligned, find a perpendicular vector for 180-degree rotation
        perp_vec = np.array([1.0, 0.0, 0.0])
        if np.linalg.norm(np.cross(v_ref, perp_vec)) < 1e-10:
            perp_vec = np.array([0.0, 1.0, 0.0])
        axis = np.cross(v_ref, perp_vec)
    else:
        axis = np.cross(v_ref, v_tar)
    
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return molecule_df # Parallel vectors, no rotation needed

    axis /= axis_norm
    sin_theta = np.sqrt(1 - cos_theta**2)
    I = np.eye(3)
    skew_axis = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = cos_theta * I + (1 - cos_theta) * np.outer(axis, axis) + sin_theta * skew_axis

    # Apply the rotation to the molecule coordinates
    coords = molecule_df[['X', 'Y', 'Z']].values
    coords_tar = (R @ coords.T).T
    molecule_df[['X', 'Y', 'Z']] = coords_tar

    return molecule_df


def align_ase_molecule_for_perovskite(ase_atoms, attachment_end='top'):
    """
    Align an ASE Atoms object for perovskite structures.
    
    This function ensures molecules are properly oriented to attach to perovskite
    structures, with special handling for small molecules like NH3.
    
    For 2D structures (DJ/RP/Monolayer), this ensures NH3/NH3+ groups face
    the perovskite layer correctly.
    
    Parameters
    ----------
    ase_atoms : ase.Atoms
        The molecule to align
    attachment_end : str, optional
        'top' or 'bottom' - which end of the molecule will attach to the layer.
        For 2D structures: 'top' means attaching from bottom (N at top end),
        'bottom' means attaching from top (N at bottom end).
        Default: 'top'
        
    Returns
    -------
    ase.Atoms
        Properly aligned molecule with attachment point pointing into perovskite
    """
    # Convert ASE Atoms to DataFrame
    symbols = ase_atoms.get_chemical_symbols()
    positions = ase_atoms.get_positions()
    
    # Store original number of atoms for validation
    num_atoms_original = len(ase_atoms)
    
    df = pd.DataFrame({
        'Element': symbols,
        'X': positions[:, 0],
        'Y': positions[:, 1],
        'Z': positions[:, 2]
    })
    
    # Validate DataFrame has all atoms
    if len(df) != num_atoms_original:
        raise ValueError(f"DataFrame conversion lost atoms: {num_atoms_original} -> {len(df)}")
    
    # Align the molecule with attachment direction awareness
    aligned_df = align_molecule_for_perovskite_2d(df, attachment_end=attachment_end)
    
    # Validate aligned DataFrame has all atoms
    if len(aligned_df) != num_atoms_original:
        raise ValueError(f"Alignment lost atoms: {num_atoms_original} -> {len(aligned_df)}")
    
    # Reconstruct Atoms object from aligned DataFrame to ensure all atoms are preserved
    # This is safer than just updating positions, as it ensures symbols and positions match
    aligned_symbols = aligned_df['Element'].tolist()
    aligned_positions = aligned_df[['X', 'Y', 'Z']].values
    
    # Create new Atoms object with aligned positions
    aligned_atoms = Atoms(aligned_symbols, positions=aligned_positions)
    
    # Preserve any additional properties from original atoms (like tags, momenta, etc.)
    if hasattr(ase_atoms, 'tags') and ase_atoms.tags is not None:
        aligned_atoms.set_tags(ase_atoms.get_tags())
    if hasattr(ase_atoms, 'momenta') and ase_atoms.get_momenta() is not None:
        aligned_atoms.set_momenta(ase_atoms.get_momenta())
    
    return aligned_atoms


def _df_to_atoms_for_nh3_check(df) -> Atoms:
    """Convert DataFrame to Atoms for NH3 detection."""
    return Atoms(
        symbols=df['Element'].tolist(),
        positions=df[['X', 'Y', 'Z']].values
    )


def _find_next_atom_from_nh3(df: pd.DataFrame, nh3_n_idx: int) -> Optional[int]:
    """
    Find the non-hydrogen atom bonded to an NH3+ nitrogen atom.

    Parameters
    ----------
    df : pd.DataFrame
        Molecule data with 'Element', 'X', 'Y', 'Z' columns
    nh3_n_idx : int
        Index of the NH3+ nitrogen atom in the DataFrame

    Returns
    -------
    int or None
        Index of the next non-hydrogen atom, or None if not found
    """
    # Covalent radii (in Angstroms)
    covalent_radii = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
        'S': 1.05, 'Cl': 0.99, 'Br': 1.20, 'I': 1.39, 'P': 1.07,
        'Si': 1.11, 'B': 0.84, 'Al': 1.21, 'Mg': 1.41, 'Ca': 1.76
    }

    # Get NH3+ nitrogen position
    n_pos = df.loc[nh3_n_idx, ['X', 'Y', 'Z']].values
    n_symbol = df.loc[nh3_n_idx, 'Element']

    # Find non-hydrogen atoms within bonding distance
    bonded_atoms = []
    for idx, row in df.iterrows():
        if idx == nh3_n_idx:  # Skip self
            continue

        if row['Element'] == 'H':  # Skip hydrogens
            continue

        atom_pos = row[['X', 'Y', 'Z']].values
        distance = np.linalg.norm(atom_pos - n_pos)

        # Check if within bonding distance
        r1 = covalent_radii.get(n_symbol, 0.71)  # Default to N
        r2 = covalent_radii.get(row['Element'], 0.76)  # Default to C
        bond_cutoff = r1 + r2 + 0.45  # Add tolerance

        if distance <= bond_cutoff:
            bonded_atoms.append((idx, distance))

    # Return closest bonded atom
    if bonded_atoms:
        return min(bonded_atoms, key=lambda x: x[1])[0]

    return None


def align_molecule_for_perovskite_2d(molecule_df, attachment_end='top'):
    """
    Align molecule specifically for 2D perovskite structures (DJ/RP/Monolayer).
    
    Unified function that works for all molecules with NH3+ groups:
    - RP/Monolayer: 1 NH3+ group
    - DJ: 2 NH3+ groups
    
    Uses geometric center-to-NH3+ vector alignment for all cases.
    
    Parameters
    ----------
    molecule_df : pd.DataFrame
        Molecule data with 'Element', 'X', 'Y', 'Z' columns
    attachment_end : str
        'top' or 'bottom' - which end will attach to the perovskite layer.
        - 'top': NH3+ should be at the TOP end (max Z) of the molecule
        - 'bottom': NH3+ should be at the BOTTOM end (min Z) of the molecule
        
    Returns
    -------
    pd.DataFrame
        Properly aligned molecule data with NH3+ at the correct attachment end
    """
    df = molecule_df.copy()
    
    # Store original center for later restoration
    original_center = df[['X', 'Y', 'Z']].values.mean(axis=0)
    
    # Center temporarily for rotation calculations
    center = df[['X', 'Y', 'Z']].values.mean(axis=0)
    df['X'] -= center[0]
    df['Y'] -= center[1]
    df['Z'] -= center[2]
    
    # Identify NH3+ groups
    nitrogen_atoms = df[df['Element'] == 'N']
    nitrogen_row_indices = nitrogen_atoms.index.tolist()  # Row indices in original df
    from .spacer import _find_terminal_nitrogens
    temp_atoms = _df_to_atoms_for_nh3_check(df)
    _, nh3_indices = _find_terminal_nitrogens(temp_atoms)

    if len(nh3_indices) == 0:
        # No NH3+ groups found, return as-is (shouldn't happen for valid spacers)
        df['X'] += original_center[0]
        df['Y'] += original_center[1]
        df['Z'] += original_center[2]
        return df

    # Map nh3_indices (indices into df) to indices into nitrogen_atoms
    nh3_positions_in_nitrogen_list = [nitrogen_row_indices.index(idx) for idx in nh3_indices]

    # Get NH3+ N atom positions
    nh3_n_atoms = nitrogen_atoms.iloc[nh3_positions_in_nitrogen_list]
    nh3_n_positions = nh3_n_atoms[['X', 'Y', 'Z']].values
    
    # Calculate geometric center
    com = df[['X', 'Y', 'Z']].values.mean(axis=0)
    
    # For molecules with multiple NH3+ groups (DJ), select the one at the attachment end
    # For single NH3+ (RP/Monolayer), use that one
    if len(nh3_indices) == 1:
        # Single NH3+ group - use it
        nh3_n_coords = nh3_n_positions[0]
    else:
        # Multiple NH3+ groups (DJ) - find the one at the attachment end
        # Calculate which NH3+ is furthest in the desired direction
        com_to_nh3_vectors = nh3_n_positions - com
        if attachment_end == 'top':
            # Find NH3+ with highest Z (furthest above geometric center)
            z_components = com_to_nh3_vectors[:, 2]
            selected_idx = np.argmax(z_components)
        else:  # 'bottom'
            # Find NH3+ with lowest Z (furthest below geometric center)
            z_components = com_to_nh3_vectors[:, 2]
            selected_idx = np.argmin(z_components)
        nh3_n_coords = nh3_n_positions[selected_idx]
    
    # Calculate vector from geometric center to NH3+ N atom
    com_to_nh3_vector = nh3_n_coords - com
    com_to_nh3_norm = np.linalg.norm(com_to_nh3_vector)
    
    # Align geometric center-to-NH3+ vector to be vertical
    if com_to_nh3_norm > 1e-6:
        # Normalize vector
        com_to_nh3_unit = com_to_nh3_vector / com_to_nh3_norm
        
        # Determine target direction based on attachment_end
        if attachment_end == 'top':
            target = np.array([0, 0, 1])  # NH3+ should be above geometric center (positive Z)
        else:  # 'bottom'
            target = np.array([0, 0, -1])  # NH3+ should be below geometric center (negative Z)
        
        # Calculate rotation to align com_to_nh3_unit with target
        dot_product = np.dot(com_to_nh3_unit, target)
        
        # Check XY component of the unit vector (should be 0 if perfectly vertical)
        xy_component_unit = np.sqrt(com_to_nh3_unit[0]**2 + com_to_nh3_unit[1]**2)
        
        # Always apply rotation if there's ANY XY component (molecule is not vertical)
        # OR if not already perfectly aligned
        # This ensures molecules with any inclination get rotated to vertical
        needs_rotation = (xy_component_unit > 1e-3) or (abs(dot_product) < 0.9999)
        
        if needs_rotation:  # Need to align
            axis = np.cross(com_to_nh3_unit, target)
            axis_norm = np.linalg.norm(axis)
            
            if axis_norm > 1e-6:
                axis = axis / axis_norm
                cos_theta = np.dot(com_to_nh3_unit, target)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                sin_theta = np.sqrt(1 - cos_theta**2)
                
                # Rodrigues' rotation formula
                I = np.eye(3)
                skew_axis = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                rotation_matrix = cos_theta * I + (1 - cos_theta) * np.outer(axis, axis) + sin_theta * skew_axis
                
                # Apply rotation around the geometric center
                coords = df[['X', 'Y', 'Z']].values
                # Translate to origin (geometric center), rotate, then translate back
                coords_centered = coords - com
                coords_rotated = np.dot(coords_centered, rotation_matrix.T)
                coords_final = coords_rotated + com
                
                df['X'] = coords_final[:, 0]
                df['Y'] = coords_final[:, 1]
                df['Z'] = coords_final[:, 2]
            else:
                # Vectors are parallel/anti-parallel, check if we need to flip
                if dot_product < 0:  # Anti-parallel, flip
                    df['Z'] = -df['Z']
        else:
            # Already aligned, but check direction
            if attachment_end == 'top' and com_to_nh3_unit[2] < 0:
                # NH3+ is below geometric center but should be above, flip
                df['Z'] = -df['Z']
            elif attachment_end == 'bottom' and com_to_nh3_unit[2] > 0:
                # NH3+ is above geometric center but should be below, flip
                df['Z'] = -df['Z']

    # Enhanced alignment: ensure NH3+-next atom vector is optimally aligned
    if len(nh3_indices) > 1:
        # DJ molecules (2+ NH3+) - align NH3+-next atom vector parallel to N-N vector
        # Get both NH3+ positions
        nh3_n_positions_all = nh3_n_positions  # All NH3+ positions

        if len(nh3_n_positions_all) >= 2:
            # Calculate N-N vector
            n_n_vector = nh3_n_positions_all[1] - nh3_n_positions_all[0]
            n_n_length = np.linalg.norm(n_n_vector)
            if n_n_length > 1e-6:
                n_n_unit = n_n_vector / n_n_length

                # Find next atom for the attachment-end NH3+ (already selected as nh3_n_coords)
                next_atom_idx = _find_next_atom_from_nh3(df, nh3_indices[selected_idx])
                if next_atom_idx is not None:
                    next_atom_pos = df.loc[next_atom_idx, ['X', 'Y', 'Z']].values
                    nh3_to_next_vector = next_atom_pos - nh3_n_coords
                    nh3_to_next_length = np.linalg.norm(nh3_to_next_vector)

                    if nh3_to_next_length > 1e-6:
                        nh3_to_next_unit = nh3_to_next_vector / nh3_to_next_length

                        # Project NH3-to-next vector onto plane perpendicular to N-N vector
                        proj_parallel = np.dot(nh3_to_next_unit, n_n_unit) * n_n_unit
                        proj_perp = nh3_to_next_unit - proj_parallel
                        proj_perp_length = np.linalg.norm(proj_perp)

                        if proj_perp_length > 1e-3:  # If there's a significant perpendicular component
                            # Rotate around N-N axis to minimize the perpendicular component
                            # This makes NH3-to-next vector as parallel as possible to N-N vector
                            axis = n_n_unit  # Rotation axis is N-N vector

                            # Get XY projections
                            n_n_xy = n_n_unit[:2]  # [X, Y] components
                            nh3_to_next_xy = nh3_to_next_unit[:2]  # [X, Y] components

                            n_n_xy_length = np.linalg.norm(n_n_xy)
                            nh3_to_next_xy_length = np.linalg.norm(nh3_to_next_xy)

                            if n_n_xy_length > 1e-6 and nh3_to_next_xy_length > 1e-6:
                                # Normalize XY projections
                                n_n_xy_unit = n_n_xy / n_n_xy_length
                                nh3_to_next_xy_unit = nh3_to_next_xy / nh3_to_next_xy_length

                                # Calculate angle between XY projections
                                cos_angle_xy = np.dot(n_n_xy_unit, nh3_to_next_xy_unit)
                                cos_angle_xy = np.clip(cos_angle_xy, -1.0, 1.0)

                                # Only rotate if not already well-aligned
                                if abs(cos_angle_xy) < 0.95:  # cos(18°) ≈ 0.95
                                    # Calculate rotation angle
                                    # We want to rotate nh3_to_next_xy_unit to align with n_n_xy_unit
                                    # Find the angle between them
                                    cross_xy = n_n_xy_unit[0] * nh3_to_next_xy_unit[1] - n_n_xy_unit[1] * nh3_to_next_xy_unit[0]
                                    sin_angle_xy = cross_xy  # This gives the sign

                                    # Rotation matrix around Z axis
                                    cos_theta = cos_angle_xy
                                    sin_theta = sin_angle_xy

                                    rotation_matrix_z = np.array([
                                        [cos_theta, -sin_theta, 0],
                                        [sin_theta, cos_theta, 0],
                                        [0, 0, 1]
                                    ])

                                    # Apply rotation around Z axis passing through attachment NH3+
                                    coords = df[['X', 'Y', 'Z']].values
                                    coords_centered = coords - nh3_n_coords  # Center on attachment NH3+
                                    coords_rotated = np.dot(coords_centered, rotation_matrix_z.T)
                                    coords_final = coords_rotated + nh3_n_coords

                                    df['X'] = coords_final[:, 0]
                                    df['Y'] = coords_final[:, 1]
                                    df['Z'] = coords_final[:, 2]
    else:
        # Single NH3+ molecules - align NH3+-next atom vector vertically
        # Find next atom for the NH3+
        next_atom_idx = _find_next_atom_from_nh3(df, nh3_indices[0])
        if next_atom_idx is not None:
            next_atom_pos = df.loc[next_atom_idx, ['X', 'Y', 'Z']].values
            nh3_to_next_vector = next_atom_pos - nh3_n_coords
            nh3_to_next_length = np.linalg.norm(nh3_to_next_vector)

            if nh3_to_next_length > 1e-6:
                nh3_to_next_unit = nh3_to_next_vector / nh3_to_next_length

                # Target direction: vertical (Z axis) in attachment direction
                target_z = 1.0 if attachment_end == 'top' else -1.0
                target = np.array([0, 0, target_z])

                # Project both vectors onto XY plane
                nh3_to_next_xy = nh3_to_next_unit[:2]  # [X, Y] components
                target_xy = target[:2]  # [0, 0] - target is vertical

                nh3_to_next_xy_length = np.linalg.norm(nh3_to_next_xy)

                # If NH3-to-next has XY components, rotate around Z to align with vertical
                if nh3_to_next_xy_length > 1e-3:
                    # We want to rotate so that nh3_to_next_unit aligns with target (vertical)
                    # Calculate the rotation angle in XY plane
                    cos_angle = nh3_to_next_unit[2]  # Z component (alignment with vertical)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)

                    # Calculate the angle in XY plane
                    # Find angle of nh3_to_next_xy vector
                    angle_xy = np.arctan2(nh3_to_next_xy[1], nh3_to_next_xy[0])

                    # Rotation matrix around Z axis to align with vertical
                    # Since target is [0,0,±1], we want to rotate so XY components become zero
                    cos_theta = cos_angle
                    sin_theta = -np.sin(angle_xy)  # Rotate to cancel XY components

                    rotation_matrix_z = np.array([
                        [cos_theta, -sin_theta, 0],
                        [sin_theta, cos_theta, 0],
                        [0, 0, 1]
                    ])

                    # But actually, simpler: rotate by -angle_xy to make XY components zero
                    cos_theta = np.cos(-angle_xy)
                    sin_theta = np.sin(-angle_xy)

                    rotation_matrix_z = np.array([
                        [cos_theta, -sin_theta, 0],
                        [sin_theta, cos_theta, 0],
                        [0, 0, 1]
                    ])

                    # Apply rotation around Z axis passing through NH3+
                    coords = df[['X', 'Y', 'Z']].values
                    coords_centered = coords - nh3_n_coords  # Center on NH3+
                    coords_rotated = np.dot(coords_centered, rotation_matrix_z.T)
                    coords_final = coords_rotated + nh3_n_coords

                    df['X'] = coords_final[:, 0]
                    df['Y'] = coords_final[:, 1]
                    df['Z'] = coords_final[:, 2]

                    # After rotation, check if direction is correct
                    # Recalculate the vector after rotation
                    next_atom_pos_rotated = df.loc[next_atom_idx, ['X', 'Y', 'Z']].values
                    nh3_to_next_vector_rotated = next_atom_pos_rotated - nh3_n_coords
                    nh3_to_next_unit_rotated = nh3_to_next_vector_rotated / np.linalg.norm(nh3_to_next_vector_rotated)

                    # If Z component has wrong sign, flip around Z
                    if (attachment_end == 'top' and nh3_to_next_unit_rotated[2] < 0) or \
                       (attachment_end == 'bottom' and nh3_to_next_unit_rotated[2] > 0):
                        df['Z'] = -df['Z']

    # Final check: ensure NH3+ is at the correct end
    # Re-identify NH3+ after rotation to get final position
    temp_atoms_final = _df_to_atoms_for_nh3_check(df)
    _, nh3_indices_final = _find_terminal_nitrogens(temp_atoms_final)
    if len(nh3_indices_final) > 0:
        nitrogen_atoms_final = df[df['Element'] == 'N']
        nitrogen_row_indices_final = nitrogen_atoms_final.index.tolist()  # Row indices in original df
        nh3_positions_in_nitrogen_list_final = [nitrogen_row_indices_final.index(idx) for idx in nh3_indices_final]
        nh3_n_atoms_final = nitrogen_atoms_final.iloc[nh3_positions_in_nitrogen_list_final]
        nh3_n_positions_final = nh3_n_atoms_final[['X', 'Y', 'Z']].values
        
        # Select the same NH3+ we used for alignment
        if len(nh3_indices_final) == 1:
            nh3_n_coords_final = nh3_n_positions_final[0]
        else:
            # Multiple NH3+ - use same selection logic
            com_final = df[['X', 'Y', 'Z']].values.mean(axis=0)
            com_to_nh3_vectors_final = nh3_n_positions_final - com_final
            if attachment_end == 'top':
                z_components = com_to_nh3_vectors_final[:, 2]
                selected_idx = np.argmax(z_components)
            else:
                z_components = com_to_nh3_vectors_final[:, 2]
                selected_idx = np.argmin(z_components)
            nh3_n_coords_final = nh3_n_positions_final[selected_idx]
        
        z_coords = df['Z'].values
        z_max = np.max(z_coords)
        z_min = np.min(z_coords)
        
        if attachment_end == 'top':
            # NH3+ should be at top (max Z)
            if nh3_n_coords_final[2] < (z_max - 0.1):  # NH3+ is not at the top
                # Find which end NH3+ is closer to
                dist_to_top = abs(nh3_n_coords_final[2] - z_max)
                dist_to_bottom = abs(nh3_n_coords_final[2] - z_min)
                if dist_to_bottom < dist_to_top:
                    # NH3+ is at bottom, flip molecule along Z
                    df['Z'] = -df['Z']
        else:  # 'bottom'
            # NH3+ should be at bottom (min Z)
            if nh3_n_coords_final[2] > (z_min + 0.1):  # NH3+ is not at the bottom
                # Find which end NH3+ is closer to
                dist_to_top = abs(nh3_n_coords_final[2] - z_max)
                dist_to_bottom = abs(nh3_n_coords_final[2] - z_min)
                if dist_to_top < dist_to_bottom:
                    # NH3+ is at top, flip molecule along Z
                    df['Z'] = -df['Z']
    
    # Restore original center position
    df['X'] += original_center[0]
    df['Y'] += original_center[1]
    df['Z'] += original_center[2]
    
    return df


# ============================================================================
# Molecular Placement and Manipulation Functions
# ============================================================================

def center_of_mass_correction(mol, mol_index=0):
    """
    Calculate center of mass correction for molecular placement.
    
    Parameters
    ----------
    mol : Atoms
        The molecule
    mol_index : int
        Index of molecule (default: 0)
        
    Returns
    -------
    np.ndarray
        Correction vector [x, y, z]
    """
    r_mol_index = mol.positions[mol_index]
    r_com = np.around(mol.get_center_of_mass(), decimals=4)
    return r_mol_index - r_com


def com_to_origin(atoms):
    """
    Center the center of mass of some atoms on the origin.
    
    Parameters
    ----------
    atoms : Atoms
        Input atoms

    Returns
    -------
    mod_atoms : Atoms
        Modified atoms object, centered on origin.
    """
    mod_atoms = atoms.copy()
    com = mod_atoms.get_center_of_mass()
    # Vectorized operation: subtract COM from all positions at once
    mod_atoms.positions = np.around(mod_atoms.positions - com, decimals=4)
    return mod_atoms


def end_to_origin(atoms, side):
    """
    Translate the origin of the molecule to the end (either top or bottom w.r.t z-axis).
    
    For the A' cations, it is necessary to be careful about the layer 
    penetration depth. For this reason, it is desired to translate the origin 
    of the molecule to the end (either top or bottom w.r.t z-axis).
    
    Parameters
    ----------
    atoms : Atoms
        The ase Atoms object containing the desired molecule.
    side : str
        'top' and 'bottom' center the molecule around the top or 
        bottom part of the molecule.
        *I.e. if you're attaching to the bottom of a perovskite,
        you want side = 'top'.

    Returns
    -------
    mod_atoms : Atoms
        The ase Atoms object with the bottom/top of the molecule
        (wrt z) centered on origin. bottom/top atom will be at coordinates
        [CoM x, CoM, y, min(z coords in molecule)]
    """
    mod_atoms = atoms.copy()
    com = mod_atoms.get_center_of_mass()
    if side == 'bottom':
        zmin = min(mod_atoms.positions[:, 2])
        translation_vector = np.array([-com[0], -com[1], -zmin])
    elif side == 'top':
        zmax = max(mod_atoms.positions[:, 2])
        translation_vector = np.array([-com[0], -com[1], -zmax])
    mod_atoms = translate_atoms(mod_atoms, translation_vector)
    return mod_atoms


def add_atoms(atoms, new_atoms):
    """
    Add new atoms to existing Atoms object.
    
    Parameters
    ----------
    atoms : Atoms
        Existing atoms object
    new_atoms : Atoms
        New atoms to add
        
    Returns
    -------
    Atoms
        Combined atoms object
    """
    combined = atoms.copy()
    combined.extend(new_atoms)
    return combined


def translate_atoms(atoms, r):
    """
    Translate atoms by vector r.
    
    Parameters
    ----------
    atoms : Atoms
        Atoms to translate
    r : array-like
        Translation vector [x, y, z]
        
    Returns
    -------
    Atoms
        Translated atoms
    """
    mod_atoms = atoms.copy()
    mod_atoms.positions += np.array(r)
    return mod_atoms


def get_molecule_length(atoms, direction='z'):
    """
    Determine the length of the molecule along a desired direction.
    
    Parameters
    ----------
    atoms : Atoms
        The input molecule's Atoms object
    direction : str
        'x', 'y', 'z' direction for the computation.
        
    Returns
    -------
    mol_length : float
        Molecule length along desired direction.
    """
    pos = atoms.positions
    if direction == 'x':
        mol_length = max(pos[:, 0]) - min(pos[:, 0])
    elif direction == 'y':
        mol_length = max(pos[:, 1]) - min(pos[:, 1])
    else:  # 'z'
        mol_length = max(pos[:, 2]) - min(pos[:, 2])
    return mol_length


def place_atoms_at_location(atoms, r):
    """
    Place the desired atoms' CoM at the location r.
    
    Parameters
    ----------
    atoms : Atoms
        Input atoms
    r : array
        Vector for the translation (3,)
        
    Returns
    -------
    mod_atoms : Atoms
        The modified atoms object.
    """
    mod_atoms = atoms.copy()
    mod_atoms = com_to_origin(mod_atoms)
    mod_atoms = translate_atoms(mod_atoms, r)
    return mod_atoms


def get_nearby_atoms_pbc(atoms: Atoms, center: np.ndarray,
                         cutoff: float) -> List[Tuple[int, np.ndarray]]:
    """
    Get atoms within cutoff distance, considering PBC.

    Based on mofun approach for better PBC-aware neighbor finding.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure to search
    center : np.ndarray
        Center point for distance calculation
    cutoff : float
        Maximum distance to include atoms

    Returns
    -------
    List[Tuple[int, np.ndarray]]
        List of (atom_index, position) tuples within cutoff
    """
    if atoms.cell is None:
        # Non-periodic: simple distance check
        positions = atoms.get_positions()
        dists = np.linalg.norm(positions - center, axis=1)
        return [(i, pos) for i, pos in enumerate(positions) if dists[i] < cutoff]

    # PBC-aware: check all 27 unit cell images
    cell = atoms.cell
    positions = atoms.get_positions()

    # Get unit cell neighbor offsets (same as mofun)
    multipliers = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 1, 3)
    uc_offsets = np.array([np.matmul(cell.T, mult[0]) for mult in multipliers])

    nearby = []
    for i, pos in enumerate(positions):
        pos_images = pos + uc_offsets
        dists = np.linalg.norm(pos_images - center, axis=1)
        min_dist = np.min(dists)
        if min_dist < cutoff:
            # Find which image is closest and return that position
            closest_idx = np.argmin(dists)
            nearby_pos = pos + uc_offsets[closest_idx]
            nearby.append((i, nearby_pos))

    return nearby


def find_molecule_patterns(structure: Atoms, pattern: Atoms,
                           atol: float = 0.1) -> List[Tuple[int, ...]]:
    """
    Find instances of pattern molecule in structure.

    Based on mofun's find_pattern_in_structure approach, simplified for ASE Atoms.

    Parameters
    ----------
    structure : ase.Atoms
        The structure to search in
    pattern : ase.Atoms
        The pattern molecule to find
    atol : float
        Absolute tolerance for atom position matching

    Returns
    -------
    List[Tuple[int, ...]]
        List of atom index tuples matching the pattern
    """
    from scipy.spatial.distance import cdist

    if len(pattern) > len(structure):
        return []

    # Pre-calculate pattern distance matrix
    pattern_positions = pattern.get_positions()
    pattern_dists = cdist(pattern_positions, pattern_positions)
    pattern_elements = pattern.get_chemical_symbols()

    matches = []

    # For each possible starting position in structure
    for i in range(len(structure) - len(pattern) + 1):
        # Quick element check: first few atoms must match
        if structure.get_chemical_symbols()[i:i+len(pattern)] != pattern_elements:
            continue

        # Check distance matrix
        struct_positions = structure.positions[i:i+len(pattern)]
        struct_dists = cdist(struct_positions, struct_positions)

        # Check if distance matrices match within tolerance
        if np.allclose(pattern_dists, struct_dists, atol=atol):
            matches.append(tuple(range(i, i+len(pattern))))

    return matches


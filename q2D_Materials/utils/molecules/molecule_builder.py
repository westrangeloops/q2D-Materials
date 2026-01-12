"""
Molecule Builder - Unified module for SMILES conversion and molecular alignment.

This module combines:
- SMILES string to ASE Atoms conversion
- Molecular alignment for perovskite structures
- Special handling for small molecules like NH3
"""

import numpy as np
import pandas as pd
from ase import Atoms

# Import RDKit for SMILES support
from rdkit import Chem
from rdkit.Chem import AllChem


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
    import tempfile
    import os
    
    # Create temporary XYZ file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.xyz') as tmp_file:
        try:
            # Use the robust smiles_to_xyz function
            smiles_to_xyz(smiles, tmp_file.name, optimize_geometry=True)
            # Read back as ASE Atoms object
            from ase.io import read
            atoms = read(tmp_file.name)
            return atoms
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)


def validate_smiles(smiles: str) -> bool:
    """
    Validate a SMILES string without generating coordinates.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
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


def align_molecule_for_perovskite(molecule_df):
    """
    Comprehensive molecular alignment for perovskite structures (bulk).
    
    NOTE: For 2D structures (DJ/RP/Monolayer), use align_molecule_for_perovskite_2d instead.
    This function is kept for bulk structures or as a fallback.
    
    This function ensures that:
    1. For small molecules (like NH3): Nitrogen atom points INTO perovskite (positive Z)
    2. For larger molecules: Largest dimension aligned along Z-axis
    3. For molecules with 2+ N atoms: N-N vector aligned along Z-axis
    4. The molecule is positioned for proper attachment
    
    Parameters
    ----------
    molecule_df : pd.DataFrame
        Molecule data with 'Element', 'X', 'Y', 'Z' columns
        
    Returns
    -------
    pd.DataFrame
        Properly aligned molecule data
    """
    df = molecule_df.copy()
    
    # Step 1: Store original center for later use
    original_center = df[['X', 'Y', 'Z']].values.mean(axis=0)
    
    # Step 2: Center temporarily for rotation calculations
    center = df[['X', 'Y', 'Z']].values.mean(axis=0)
    df['X'] -= center[0]
    df['Y'] -= center[1]
    df['Z'] -= center[2]
    
    # Step 3: Check for nitrogen atoms
    nitrogen_atoms = df[df['Element'] == 'N']
    num_nitrogens = len(nitrogen_atoms)
    
    # Step 4: Handle different molecule types
    coords = df[['X', 'Y', 'Z']].values
    ranges = np.max(coords, axis=0) - np.min(coords, axis=0)
    max_range = np.max(ranges)
    
    # For small molecules (like NH3) with single N atom, ensure N points into perovskite
    if num_nitrogens == 1 and max_range < 5.0:  # Small molecule threshold
        # Get the nitrogen atom position (already centered)
        n_coords = nitrogen_atoms[['X', 'Y', 'Z']].iloc[0].values
        
        # For NH3 and similar small molecules, we want the N atom to point INTO the perovskite
        # This means N should be at the positive Z end of the molecule
        
        # Calculate the center of mass of non-N atoms (H atoms in NH3)
        non_n_atoms = df[df['Element'] != 'N']
        if len(non_n_atoms) > 0:
            # Calculate vector from non-N center to N atom
            non_n_center = non_n_atoms[['X', 'Y', 'Z']].values.mean(axis=0)
            n_vector = n_coords - non_n_center
            
            # Normalize the vector
            n_vector_norm = np.linalg.norm(n_vector)
            if n_vector_norm > 1e-6:
                n_vector_unit = n_vector / n_vector_norm
                
                # We want this vector to point in positive Z direction
                # Calculate rotation to align n_vector_unit with [0, 0, 1]
                target = np.array([0, 0, 1])
                
                # Calculate rotation axis and angle
                axis = np.cross(n_vector_unit, target)
                axis_norm = np.linalg.norm(axis)
                
                if axis_norm > 1e-6:
                    axis = axis / axis_norm
                    cos_theta = np.dot(n_vector_unit, target)
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
                    
                    # Apply rotation
                    coords_rotated = np.dot(coords, rotation_matrix.T)
                    df['X'] = coords_rotated[:, 0]
                    df['Y'] = coords_rotated[:, 1]
                    df['Z'] = coords_rotated[:, 2]
        
        # Final check: ensure N is at positive Z (pointing into perovskite)
        n_coords_final = df[df['Element'] == 'N'][['X', 'Y', 'Z']].iloc[0].values
        if n_coords_final[2] < 0:
            # Flip along Z-axis to ensure N points into perovskite
            df['Z'] = -df['Z']
    
    elif num_nitrogens >= 2:
        # For molecules with 2+ N atoms, align N-N vector along Z
        df = _align_nitrogens(df, tar='Z')
        
        # Ensure the N-N vector points in positive Z direction
        n_atoms = df[df['Element'] == 'N']
        if len(n_atoms) >= 2:
            n1_z = n_atoms['Z'].iloc[0]
            n2_z = n_atoms['Z'].iloc[1]
            if n1_z < n2_z:  # First N is below second N
                # Flip so first N is above (positive Z)
                df['Z'] = -df['Z']
    
    else:
        # For other molecules: align largest dimension to Z-axis
        largest_dim = np.argmax(ranges)
        
        if largest_dim == 0:  # X is largest, rotate to Z
            angle = np.radians(90)
            rotation_matrix = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
        elif largest_dim == 1:  # Y is largest, rotate to Z
            angle = np.radians(-90)
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
        else:  # Z is already largest
            rotation_matrix = np.eye(3)
        
        # Apply rotation
        coords = df[['X', 'Y', 'Z']].values
        coords_rotated = np.dot(coords, rotation_matrix.T)
        df['X'] = coords_rotated[:, 0]
        df['Y'] = coords_rotated[:, 1]
        df['Z'] = coords_rotated[:, 2]
    
    # Step 5: Final orientation check - ensure molecule extends in positive Z
    # This ensures attachment point points INTO the perovskite
    z_coords = df['Z'].values
    z_max = np.max(z_coords)
    z_min = np.min(z_coords)
    
    # If molecule extends more in negative Z, flip it
    if abs(z_min) > abs(z_max):
        df['Z'] = -df['Z']
    
    # Step 6: Restore original center position (don't center at origin)
    df['X'] += original_center[0]
    df['Y'] += original_center[1]
    df['Z'] += original_center[2]
    
    return df


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


def _identify_nh3_groups(molecule_df):
    """
    Identify NH3+ groups in a molecule by finding N atoms with 3 H atoms nearby.
    
    Parameters
    ----------
    molecule_df : pd.DataFrame
        Molecule data with 'Element', 'X', 'Y', 'Z' columns
        
    Returns
    -------
    list
        List of indices (in the nitrogen_atoms DataFrame) of N atoms that are part of NH3+ groups
    """
    nitrogen_atoms = molecule_df[molecule_df['Element'] == 'N']
    hydrogen_atoms = molecule_df[molecule_df['Element'] == 'H']
    
    nh3_indices = []
    
    if len(hydrogen_atoms) == 0:
        return nh3_indices
    
    n_positions = nitrogen_atoms[['X', 'Y', 'Z']].values
    h_positions = hydrogen_atoms[['X', 'Y', 'Z']].values
    
    # Typical N-H bond distance is around 1.0-1.1 Å
    nh_bond_cutoff = 1.2
    
    for i, n_pos in enumerate(n_positions):
        # Calculate distances from this N to all H atoms
        distances = np.linalg.norm(h_positions - n_pos, axis=1)
        nearby_h_count = np.sum(distances < nh_bond_cutoff)
        
        # NH3+ groups have 3 H atoms nearby
        if nearby_h_count == 3:
            nh3_indices.append(i)
    
    return nh3_indices


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
    nh3_indices = _identify_nh3_groups(df)
    
    if len(nh3_indices) == 0:
        # No NH3+ groups found, return as-is (shouldn't happen for valid spacers)
        df['X'] += original_center[0]
        df['Y'] += original_center[1]
        df['Z'] += original_center[2]
        return df
    
    # Get NH3+ N atom positions
    nh3_n_atoms = nitrogen_atoms.iloc[nh3_indices]
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
    
    # Final check: ensure NH3+ is at the correct end
    # Re-identify NH3+ after rotation to get final position
    nh3_indices_final = _identify_nh3_groups(df)
    if len(nh3_indices_final) > 0:
        nitrogen_atoms_final = df[df['Element'] == 'N']
        nh3_n_atoms_final = nitrogen_atoms_final.iloc[nh3_indices_final]
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


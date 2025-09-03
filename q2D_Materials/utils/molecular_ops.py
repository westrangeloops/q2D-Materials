import numpy as np
import pandas as pd

def rot_mol(data, degree):
    """
    Rotate a molecule around the axis defined by two nitrogen atoms.
    Enhanced with better error handling and validation.
    
    Parameters
    ----------
    data : pd.DataFrame
        Molecule data with 'Element', 'X', 'Y', 'Z' columns
    degree : float
        Rotation angle in degrees
        
    Returns
    -------
    pd.DataFrame
        Rotated molecule data
    """
    df = data.copy()
    
    # find the coordinates of the two nitrogen atoms
    nitrogen_atoms = df[df['Element'] == 'N'][['X', 'Y', 'Z']].values
    
    if len(nitrogen_atoms) < 2:
        print("Warning: Less than 2 nitrogen atoms found. Using molecular center for rotation.")
        # Fallback: rotate around molecular center of mass
        center = df[['X', 'Y', 'Z']].values.mean(axis=0)
        axis = np.array([0, 0, 1])  # Default Z-axis rotation
        rotation_center = center
    else:
        # calculate the axis of rotation as the vector between the two nitrogen atoms
        axis = nitrogen_atoms[1] - nitrogen_atoms[0]
        rotation_center = nitrogen_atoms[0]
    
    # normalize the axis vector
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        print("Warning: Nitrogen atoms are too close. Using Z-axis for rotation.")
        axis = np.array([0, 0, 1])
    else:
        axis = axis / axis_norm

    # convert the rotation angle from degrees to radians
    angle = np.radians(degree)

    # create the rotation matrix using Rodrigues' rotation formula
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    
    rotation_matrix = np.array([
        [t*axis[0]**2 + c, t*axis[0]*axis[1] - s*axis[2], t*axis[0]*axis[2] + s*axis[1]],
        [t*axis[0]*axis[1] + s*axis[2], t*axis[1]**2 + c, t*axis[1]*axis[2] - s*axis[0]],
        [t*axis[0]*axis[2] - s*axis[1], t*axis[1]*axis[2] + s*axis[0], t*axis[2]**2 + c]
    ])

    # apply the rotation to all atoms in the dataframe
    atoms = df[['X', 'Y', 'Z']].values
    atoms -= rotation_center
    atoms = np.dot(atoms, rotation_matrix)
    atoms += rotation_center
    df[['X', 'Y', 'Z']] = atoms
    print(f'Spacer rotated by {degree}° around nitrogen axis!')
    return df

def transform_mol(mol, T1, T2, T3):
    molecule_df = mol.molecule_df.copy()

    # Define reference and target points
    A_ref = mol.molecule_D1
    B_ref = mol.molecule_D2
    H_ref = mol.molecule_D3
    A_tar = T1
    B_tar = T2
    H_tar = T3

    # Calculate normal vectors to reference and target planes
    N_ref = np.cross(B_ref - A_ref, H_ref - A_ref)
    N_tar = np.cross(B_tar - A_tar, H_tar - A_tar)

    # Check for degenerate cases (collinear points or zero-length vectors)
    N_ref_norm = np.linalg.norm(N_ref)
    N_tar_norm = np.linalg.norm(N_tar)
    
    if N_ref_norm < 1e-10 or N_tar_norm < 1e-10:
        # Degenerate case: points are collinear, use identity transformation
        print("Warning: Degenerate molecular configuration detected, using identity transformation")
        R = np.eye(3)
    else:
        # Normalize the normal vectors
        N_ref_unit = N_ref / N_ref_norm
        N_tar_unit = N_tar / N_tar_norm
        
        # Calculate rotation matrix that transforms reference plane to target plane
        cos_theta = np.clip(np.dot(N_ref_unit, N_tar_unit), -1.0, 1.0)  # Clamp to avoid numerical errors
        
        if abs(cos_theta - 1.0) < 1e-10:
            # Vectors are already aligned
            R = np.eye(3)
        elif abs(cos_theta + 1.0) < 1e-10:
            # Vectors are anti-aligned, need 180-degree rotation
            # Find a perpendicular vector to rotate around
            if abs(N_ref_unit[0]) < 0.9:
                perp = np.array([1, 0, 0])
            else:
                perp = np.array([0, 1, 0])
            axis = np.cross(N_ref_unit, perp)
            axis = axis / np.linalg.norm(axis)
            R = 2 * np.outer(axis, axis) - np.eye(3)  # 180-degree rotation
        else:
            # General case
            sin_theta = np.sqrt(1 - cos_theta**2)
            axis = np.cross(N_ref_unit, N_tar_unit)
            axis_norm = np.linalg.norm(axis)
            
            if axis_norm < 1e-10:
                # Vectors are parallel, no rotation needed
                R = np.eye(3)
            else:
                axis = axis / axis_norm
                I = np.eye(3)
                skew_axis = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                R = cos_theta * I + (1 - cos_theta) * np.outer(axis, axis) + sin_theta * skew_axis

    # Calculate translation vector that moves reference plane to target plane
    p_ref = (A_ref + B_ref + H_ref) / 3
    p_tar = (A_tar + B_tar + H_tar) / 3
    T = p_tar - R @ p_ref

    # Apply transformation to molecule coordinates
    coords = molecule_df[['X', 'Y', 'Z']].values
    coords_tar = (R @ coords.T + T.reshape(-1, 1)).T
    molecule_df[['X', 'Y', 'Z']] = coords_tar

    return molecule_df

def align_nitrogens(molecule_df, tar='Z'):
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

def inclinate_molecule(df, angle_degrees):
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Rotation matrix around the x-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_radians), -np.sin(angle_radians)],
        [0, np.sin(angle_radians), np.cos(angle_radians)]
    ])

    # Extract the z, y, and x coordinates from the DataFrame
    z_coords = df['Z'].to_numpy()
    y_coords = df['Y'].to_numpy()
    x_coords = df['X'].to_numpy()

    # Stack the coordinates into a single matrix for matrix multiplication
    coords = np.column_stack((x_coords, y_coords, z_coords))

    # Apply the rotation matrix to the coordinates
    rotated_coords = np.dot(coords, rotation_matrix.T)

    # Update the DataFrame with the new rotated coordinates
    df['X'] = rotated_coords[:, 0]
    df['Y'] = rotated_coords[:, 1]
    df['Z'] = rotated_coords[:, 2]

    return df


def align_molecule_for_perovskite(molecule_df):
    """
    Comprehensive molecular alignment for perovskite structures.
    
    This function ensures that:
    1. The molecule's largest dimension is aligned along the Z-axis
    2. Nitrogen atoms are properly positioned for perovskite coordination
    3. The molecule is positioned for proper attachment (not centered at origin)
    
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
    
    # Step 3: Determine the largest dimension
    coords = df[['X', 'Y', 'Z']].values
    ranges = np.max(coords, axis=0) - np.min(coords, axis=0)
    largest_dim = np.argmax(ranges)
    
    # Step 4: Align largest dimension to Z-axis
    if largest_dim == 0:  # X is largest, rotate to Z
        # Rotate around Y-axis by 90 degrees
        angle = np.radians(90)
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif largest_dim == 1:  # Y is largest, rotate to Z
        # Rotate around X-axis by -90 degrees
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
    
    # Step 5: Align nitrogen atoms if present
    nitrogen_atoms = df[df['Element'] == 'N']
    if len(nitrogen_atoms) >= 2:
        # Use the existing align_nitrogens function
        df = align_nitrogens(df, tar='Z')
    
    # Step 6: Ensure molecule is oriented with positive Z for perovskite attachment
    # Check if the molecule extends more in negative Z direction
    z_coords = df['Z'].values
    if np.min(z_coords) < -np.max(z_coords):
        # Flip the molecule along Z-axis
        df['Z'] = -df['Z']
    
    # Step 7: Restore original center position (don't center at origin)
    df['X'] += original_center[0]
    df['Y'] += original_center[1]
    df['Z'] += original_center[2]
    
    return df


def align_ase_molecule_for_perovskite(ase_atoms):
    """
    Align an ASE Atoms object for perovskite structures.
    
    Parameters
    ----------
    ase_atoms : ase.Atoms
        The molecule to align
        
    Returns
    -------
    ase.Atoms
        Properly aligned molecule
    """
    import pandas as pd
    
    # Convert ASE Atoms to DataFrame
    symbols = ase_atoms.get_chemical_symbols()
    positions = ase_atoms.get_positions()
    
    df = pd.DataFrame({
        'Element': symbols,
        'X': positions[:, 0],
        'Y': positions[:, 1],
        'Z': positions[:, 2]
    })
    
    # Align the molecule
    aligned_df = align_molecule_for_perovskite(df)
    
    # Convert back to ASE Atoms
    aligned_atoms = ase_atoms.copy()
    aligned_atoms.set_positions(aligned_df[['X', 'Y', 'Z']].values)
    
    return aligned_atoms 
import numpy as np
import pandas as pd

def rot_mol(data, degree):
    df = data.copy()
    # find the coordinates of the two nitrogen atoms
    nitrogen_atoms = df[df['Element'] == 'N'][['X', 'Y', 'Z']].values

    # calculate the axis of rotation as the vector between the two nitrogen atoms
    axis = nitrogen_atoms[1] - nitrogen_atoms[0]

    # normalize the axis vector
    axis = np.linalg.norm(axis)

    # convert the rotation angle from degrees to radians
    angle = np.radians(degree)

    # create the rotation matrix
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    rotation_matrix = np.array([[t*axis[0]**2 + c, t*axis[0]*axis[1] - s*axis[2], t*axis[0]*axis[2] + s*axis[1]],
                                [t*axis[0]*axis[1] + s*axis[2], t*axis[1]**2 + c, t*axis[1]*axis[2] - s*axis[0]],
                                [t*axis[0]*axis[2] - s*axis[1], t*axis[1]*axis[2] + s*axis[0], t*axis[2]**2 + c]])

    # apply the rotation to all atoms in the dataframe
    atoms = df[['X', 'Y', 'Z']].values
    atoms -= nitrogen_atoms[0]
    atoms = np.dot(atoms, rotation_matrix)
    atoms += nitrogen_atoms[0]
    df[['X', 'Y', 'Z']] = atoms
    print('Spacer rotated!')
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

    # Calculate rotation matrix that transforms reference plane to target plane
    cos_theta = np.dot(N_ref, N_tar) / (np.linalg.norm(N_ref) * np.linalg.norm(N_tar))
    sin_theta = np.sqrt(1 - cos_theta**2)
    axis = np.cross(N_ref, N_tar) / np.linalg.norm(np.cross(N_ref, N_tar))
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
    # Find the coordinates of the two Nitrogen atoms
    n1_coords = molecule_df.loc[molecule_df['Element'] == 'N', ['X', 'Y', 'Z']].iloc[0].values
    n2_coords = molecule_df.loc[molecule_df['Element'] == 'N', ['X', 'Y', 'Z']].iloc[1].values

    # Calculate the vector between the two Nitrogen atoms
    v_ref = n1_coords - n2_coords

    if tar == 'Z':
        # Define the target vector (aligned with the z-axis)
        v_tar = np.array([0, 0, np.linalg.norm(v_ref)])
    else:
        v_tar = tar

    # Calculate the rotation matrix that transforms v_ref to v_tar
    cos_theta = np.dot(v_ref, v_tar) / (np.linalg.norm(v_ref) * np.linalg.norm(v_tar))
    sin_theta = np.sqrt(1 - cos_theta**2)
    axis = np.cross(v_ref, v_tar) / np.linalg.norm(np.cross(v_ref, v_tar))
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
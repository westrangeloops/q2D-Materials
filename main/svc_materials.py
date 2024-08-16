# Imports
import pandas as pd
import numpy as np
from ase.io import read
from ase.visualize import view
from rdkit import Chem
from os import remove
import numpy.linalg as LA

from scipy.spatial import distance
from ase.neighborlist import NeighborList
from ase import Atoms, visualize
from ase.geometry.analysis import Analysis
import math
from scipy.spatial.distance import cdist


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

def mol_load(file):
    # Load the molecule from the file
    if file.endswith('.mol2'): 
        mol = Chem.MolFromMol2File(file, removeHs=False)
    elif file.endswith('.xyz'):
        mol = Chem.MolFromXYZFile(file, removeHs=False)
    else:
        print('WARNING: Not currently supported format')

    # Check if the molecule was loaded successfully
    if mol is None:
        raise ValueError(f"Could not load molecule from file {file}")

    # Get the number of atoms in the molecule
    num_atoms = mol.GetNumAtoms()

    # Get the atomic symbols and 3D coordinates from the molecule
    atoms = []
    for i in range(num_atoms):
        symbol = mol.GetAtomWithIdx(i).GetSymbol()
        pos = mol.GetConformer().GetAtomPosition(i)
        atoms.append([symbol, pos.x, pos.y, pos.z])

    # Create a Pandas DataFrame with the atomic coordinates
    df = pd.DataFrame(atoms, columns=['Element', 'X', 'Y', 'Z'])
    return df


def vasp_load(file_path):
    try:
        lines = []
        f = open(file_path, 'r')
        # Remove empty lines:
        for line in f:
            lines.append(line.split())
        lines = [x for x in lines if x != []]
        #Proc lines
        box = lines[1:5] # box[0] es la escala box[1:4] a, b, c respectivamente, scale = box[0][0], a=box[1][0], b=[2][1] y c=[3][2]
        # box array:
        a = [box[1][0], box[2][0], box[3][0]]
        b = [box[1][1], box[2][1], box[3][1]]
        c = [box[1][2], box[2][2], box[3][2]]
        # Elements quantity
        elements = lines[5:8] # elements[0] = type of element, elements[1] = number of elements
        # Box contain all other information
        # [[[a], [b], [c]], [['H', 'Pb', 'C', 'I', 'N'], ['24', '4', '4', '12', '4'], ['Cartesian']]]
        box = [[a, b, c], elements]
        box[1][2] = ['Cartesian']
    
        atoms = read(file_path)
        atoms.wrap()

        # Extract atomic symbols and coordinates
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()

        # Create DataFrame
        df = pd.DataFrame({
            'Element': symbols,
            'X': positions[:, 0],
            'Y': positions[:, 1],
            'Z': positions[:, 2]
        })
        
        return df, box

    except Exception as e:
        print(f"Error reading VASP file: {e}")
        return None
    

    # USE THIS TO SAVE TO VASP FILE
def save_vasp(dt, box, name='svc', dynamics=False, order=False, B=False):
    a = box[0][0]
    b = box[0][1]
    c = box[0][2]
    elements = box[1]
    MP = dt
    MP.sort_values('Element', inplace=True)

    if order is False:
        pass
    else:
        print('The order is: {}'.format(order))
        atom_order = order
        MP.sort_values(by='Element', key=lambda column: column.map(lambda x: elements[0].index(x)), inplace=True)
        # Define the desired atom order

        # Reorder the dataframe based on the desired atom order
        MP['Element'] = pd.Categorical(MP['Element'], categories=atom_order, ordered=True)


    if dynamics:
        # Calculate length based on largest atom in 'Z' coordinate
        length = np.amax(MP['Z']) / 2.0

        # Find the first B atom below and above the length
        first_pb_below = MP.loc[(MP['Element'] == '{}'.format(B)) & (MP['Z'] < length), 'Z'].max()
        first_pb_above = MP.loc[(MP['Element'] == '{}'.format(B)) & (MP['Z'] > length), 'Z'].min()

        # Boolean array indicating atoms between first_B_below and first_B_above with elements N, C, H
        mask = (MP['Element'].isin(['N', 'C', 'H'])) & (MP['Z'] > first_pb_below) & (MP['Z'] < first_pb_above)

        # Create the 'DYN' column with 'F   F   F' or 'T   T   T'
        MP['DYN'] = np.where(mask, 'T   T   T', 'F   F   F')

    # Elements to print
    elem_idx = MP.groupby('Element', sort=False).count().index
    elem_val = MP.groupby('Element', sort=False).count()['X']

    with open('{}'.format(name), 'w') as vasp_file:
        # Vasp name
        vasp_file.write(''.join(name)+'\n')
        vasp_file.write('1'+'\n') # Scale
        for n in range(0, 3): # Vectors of a, b and c
            vasp_file.write('            ' + str(a[n]) + ' ' + str(b[n]) + ' ' + str(c[n])+'\n')
        # Vasp number of atoms
        vasp_file.write('      ' + '   '.join(elem_idx)+'\n')
        vasp_file.write('      ' + '   '.join([str(x) for x in elem_val])+'\n')
        # Vasp box vectors
        if dynamics:
            vasp_file.write('Selective dynamics'+'\n')
            vasp_file.write(elements[-1][0]+'\n')
            # Vasp Atoms and positions (with 'DYN' column)
            vasp_file.write(MP.loc[:, 'X':'DYN'].to_string(index=False, header=False))
        else:
            vasp_file.write(elements[-1][0]+'\n')
            # Vasp Atoms and positions (original)
            vasp_file.write(MP.loc[:, 'X':'Z'].to_string(index=False, header=False))

    vasp_file.close()
    #print('Your file was saved as {}'.format(name))


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

def pi(argu):
    return np.array(list(map(float,argu.strip().split()))[:3])


def make_svc(DF_MA_1, DF_MA_2):
    dis_1 = DF_MA_1.sort_values(by='Z').iloc[0, 3]
    DF_MA_1['Z'] = DF_MA_1['Z'].apply(lambda x: x - dis_1)
    dis_2 = DF_MA_2.sort_values(by='Z').iloc[0, 3]
    DF_MA_2['Z'] = DF_MA_2['Z'].apply(lambda x: x - dis_2)
    organic_spacers = pd.concat([DF_MA_1, DF_MA_2])
    organic_spacers.reset_index(drop=True, inplace=True)
    return organic_spacers


def pos_finder(mol):
    perov = read(mol.perovskite_file)

def bulk_creator(mol, slab, hn):
    # Create ATOMS class from ase and call the spacer
    spacer = mol.svc
    B = mol.B
    X = mol.X

    if slab % 2 != 0: # Odd
        slab_ss = int(slab/2 + 1)
    else:
        slab_ss = int(slab/2)

    # If we want a slab
    # Supercell that would be cut in half
    super_cell = read(mol.perovskite_file)
    super_cell = super_cell*[1, 1, slab_ss]

    # Extract symbols and positions from Atoms object
    symbols = super_cell.get_chemical_symbols()
    positions = super_cell.positions
    supa_df = pd.DataFrame({'Element': symbols, 'X': positions[:, 0], 'Y': positions[:, 1], 'Z': positions[:, 2]})[['Element', 'X', 'Y', 'Z']]

    # B planes, this would be used to slice in two parts the perovskite
    b_planes = supa_df.query("Element == @B").sort_values(by='Z')
    ub_planes = b_planes.round(1).drop_duplicates(subset='Z')
    # Len of the axial bond of B and X:
    gh = ub_planes.iloc[0, 3]
    bond_len = supa_df.query('Element == @X and Z >= @gh + 1').sort_values(by='Z').iloc[0, 3]
    bond_len = bond_len - b_planes.iloc[0, 3]
    up_df = supa_df.query('Z >= @gh + 1')
    dwn_df = supa_df.query('Z <= @gh + 1')

    # Clean the up_df, ereasing all that is lower than the lowest B plane in the up_df and is not X
    # First copy the terminal X and add it to the dwn_df
    terminal_X = up_df.query("Element == @X").iloc[:, 3].min() # Search for the minimum X plane
    terminal_X = up_df.query("Element == @X and abs(Z - @terminal_X) <= 1") # Copy the entire plane
    # Update the value copied X plane to be over the b_plane in the dwn_df
    terminal_X.loc[:, 'Z'] = terminal_X['Z'].apply(lambda x: b_planes.iloc[:, 3].min() + bond_len) 
    dwn_df = pd.concat([dwn_df, terminal_X], axis=0, ignore_index=True)
    
    # Now we account for the up_df which is different from the upper and the dwn
    if slab == 1:
        list_I = supa_df.query('Element == @X').sort_values(by='Z').iloc[-1, 3]
        up_df = supa_df.query('Z >= @list_I - 1 and Element == @X')
    elif slab % 2 != 0 : # Odd, we copy the highest X plane
        terminal_X = up_df.query("Element == @X").iloc[:, 3].max()
        # Also we erase one perovskite slab, all that is below the second B plane is erased
        sec_B_plane = up_df.query("Element == @B").sort_values(by='Z').loc[:,'Z'].round(2).drop_duplicates().iloc[1]
        up_df = up_df.query("Z >= @sec_B_plane - 1")
        # Copy the plane and align it with bond len
        terminal_X = up_df.query("Element == @X and abs(Z - @terminal_X) <= 1") # Copy the entire plane
        terminal_X['Z'] = terminal_X['Z'].apply(lambda x: sec_B_plane - bond_len) 
        # Now we clean the up_df:
        high_B = up_df.query("Element == @B").iloc[:, 3].min()
        up_df = up_df.query("Z > @high_B - 1")
        # Finally concat the DF
        up_df = pd.concat([up_df, terminal_X], axis=0, ignore_index=True)
    else:
        # Erase all thath is below the plane of lowest B
        low_B = up_df.query("Element == @B").sort_values(by='Z').iloc[0, 3]
        up_df = up_df.query("Z >= @low_B - 1")
        # Copy the terminal_X in the up_df and the dwn_df
        up_df = pd.concat([up_df, terminal_X], axis=0, ignore_index=True)

    
    # ADJUST THE HEIGHT:
    spacer['Z'] = spacer['Z'].apply(lambda x: x - spacer['Z'].min() + b_planes.iloc[:, 3].min() + bond_len - hn)
    up_df.loc[:, 'Z'] = up_df['Z'].apply(lambda x: x - up_df['Z'].min() + spacer['Z'].max() - hn)


    # Concatenate the bulk
    bulk = pd.concat([dwn_df, spacer, up_df], axis=0, ignore_index=True)

    # The box:
    box = mol.box
    # The lowest X plane in the dwn df needs to maintain the bond len:
    correction = bond_len - b_planes.iloc[:, 3].min()

    # Update box
    up_B = bulk.query("Element == @X").sort_values(by='Z', ascending=False).iloc[0].to_list()[3]
    box[0][2][2] = up_B + correction # Highest atom in the Z vector in the box

    return bulk, box

def find_min_max_coordinates(df):
    min_x = df['X'].min()
    max_x = df['X'].max()
    min_y = df['Y'].min()
    max_y = df['Y'].max()
    min_z = df['Z'].min()
    max_z = df['Z'].max()
    return min_x, max_x, min_y, max_y, min_z, max_z

def iso(molecule):
    db = mol_load(molecule.molecule_file)
    min_x, max_x, min_y, max_y, min_z, max_z = find_min_max_coordinates(db)

    db['X'] += 5 + abs(min_x)
    db['Y'] += 5 + abs(min_y)
    db['Z'] += 5 + abs(min_z)

    max_latx = 5 + abs(min_x) + max_x + 5
    max_laty = 5 + abs(min_y) + max_y + 5
    max_latz = 5 + abs(min_z) + max_z + 5

    X = [str(max_latx), '0.0000000000', '0.0000000000']
    Y = ['0.0000000000', str(max_laty), '0.0000000000']
    Z = ['0.0000000000', '0.0000000000', str(max_latz)]

    db.sort_values(by='Element')
    element = [i for i in set(db['Element'].to_list())]
    n_ele = db.groupby(by='Element').count()
    n_ele = n_ele['X'].to_list()
    box = [[X, Y, Z], [element, n_ele, ['Cartesian']]]
    return db, box

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


def direct_to_cartesian(df, lattice_vectors):
    df = df
    # Convert fractional coordinates to Cartesian coordinates
    atomic_positions_direct = df[['x_direct', 'y_direct', 'z_direct']].values
    cartesian_positions = np.dot(atomic_positions_direct, lattice_vectors)

    # Create a new DataFrame with Cartesian coordinates
    df_cartesian = pd.concat([df, pd.DataFrame(cartesian_positions, columns=['x_cartesian', 'y_cartesian', 'z_cartesian'])], axis=1)

    return df_cartesian


# The DJ class would be named as Dj_creator
# We would create a class named Dj_analysis
#  
class q2D_creator:
    def __init__(self, B, X, molecule_xyz, perov_vasp, P1, P2, P3, Q1, Q2, Q3, name, vac=0, n=1):
        self.B = B
        self.X = X
        self.name = name
        self.vac = vac
        self.molecule_file = molecule_xyz
        self.perovskite_file = perov_vasp
        self.P1, self.P2, self.P3, self.Q1, self.Q2, self.Q3 = pi(P1), pi(P2), pi(P3), pi(Q1), pi(Q2), pi(Q3)
        self.molecule_df = align_nitrogens(mol_load(self.molecule_file))
        self.perovskite_df, self.box = vasp_load(perov_vasp)
        self.molecule_D1 = self.molecule_df.loc[self.molecule_df['Element'] == 'N', 'X':'Z'].sort_values(by='Z').values[0]
        self.molecule_D2 = self.molecule_df.loc[self.molecule_df['Element'] == 'C', 'X':'Z'].sort_values(by='Z', ascending=False).values[0]
        self.molecule_D3 = self.molecule_df.loc[self.molecule_df['Element'] == 'N', 'X':'Z'].sort_values(by='Z').values[1]
        self.DF_MA_1 = transform_mol(self, self.P1, self.P2, self.P3)
        self.DF_MA_2 = transform_mol(self, self.Q1, self.Q2, self.Q3)
        self.svc = make_svc(self.DF_MA_1, self.DF_MA_2)
        #self.salt, self.saltbox = make_salt(self)


    def write_svc(self):
        name =  'svc_' + self.name + '.vasp'
        svc = self.svc.copy()
        svc_box = self.box.copy()
        svc_box[0][2][2] = svc['Z'].sort_values(ascending=False).iloc[0] + self.vac  # Z vector in the box
        save_vasp(svc, svc_box, name)


    def rot_spacer(self, degree1, degree2):
        self.DF_MA_1 = rot_mol(self.DF_MA_1, degree1)
        self.DF_MA_2 = rot_mol(self.DF_MA_2, degree2)
        self.svc = make_svc(self.DF_MA_1, self.DF_MA_2)


    def inc_spacer(self, degree):
        self.DF_MA_1 = inclinate_molecule(self.DF_MA_1, degree)
        self.DF_MA_2 = inclinate_molecule(self.DF_MA_2, degree)
        self.svc = make_svc(self.DF_MA_1, self.DF_MA_2)

    def show_svc(self, m=[1, 1, 1]):
        svc_box = self.box.copy()
        svc_box[0][2][2] = self.svc['Z'].sort_values(ascending=False).iloc[0] + self.vac  # Z vector in the box
        # Read the VASP POSCAR file
        save_vasp(self.svc, svc_box, 'temporal_svc.vasp')
        #save_vasp(self.DF_MA_2, svc_box, 'temporal_svc.vasp')
        #save_vasp(self.DF_MA_1, svc_box, 'temporal_svc.vasp')
        # Read in the POSCAR file
        atoms = read('temporal_svc.vasp')
        atoms = atoms*m
        remove('temporal_svc.vasp')
        
        return view(atoms)


    def write_bulk(self, slab=1, m=[1,1,1], hn=0, dynamics=False, order=False):
        if dynamics:
            name = 'bulk_' + self.name + '_SD'+ '.vasp'
        else:
            name = 'bulk_' + self.name + '.vasp'
        bulk, bulk_box = bulk_creator(self, slab, hn)
        save_vasp(bulk, bulk_box, name, dynamics, order, self.B)


    def show_bulk(self, slab=1,m=1, hn=0):
        bulk, bulk_box = bulk_creator(self, slab, hn)
        # Read the VASP POSCAR file
        save_vasp(bulk, bulk_box, 'temporal.vasp')
        # Read in the POSCAR file
        atoms = read('temporal.vasp')
        atoms = atoms*m
        remove('temporal.vasp')
        return view(atoms)

    def write_iso(self):
        db, box = iso(self)
        save_vasp(db, box, name=self.name + '_iso.vasp', dynamics=False, order=False)

    def show_iso(self):
        db, box = iso(self)
        save_vasp(db, box, name='temporal.vasp')
        # Read in the POSCAR file
        atoms = read('temporal.vasp')
        view(atoms)

class q2D_analysis:
        def __init__(self, B, X, crystal):
            self = self
            self.path = '/'.join(crystal.split('/')[0:-1])
            self.name = crystal.split('/')[-1]
            self.B = B
            self.X = X
            self.perovskite_df, self.box = vasp_load(crystal)
        
        def isolate_spacer(self, order=None):
            crystal_df = self.perovskite_df
            B = self.B
            # Find the planes of the perovskite.
            b_planes = crystal_df.query("Element == @B").sort_values(by='Z')
            b_planes['Z'] = b_planes['Z'].apply(lambda x: round(x, 1))
            b_planes.drop_duplicates(subset='Z', inplace=True)
            b_planes.reset_index(inplace=True, drop=True)

            if len(b_planes.values) > 1:
                b_planes['Diff'] = b_planes['Z'].diff()
                id_diff_max = b_planes['Diff'].idxmax()
                b_down_plane = b_planes.iloc[id_diff_max - 1:id_diff_max] 
                b_up_plane = b_planes.iloc[id_diff_max:id_diff_max + 1]
                b_down_plane = b_down_plane['Z'].values[0] + 1
                b_up_plane = b_up_plane['Z'].values[0] - 1
                # Now lets create a df with only the elements that are between that!
                # We call that the salt
                iso_df = crystal_df.query('Z <= @b_up_plane and Z >= @b_down_plane ')
            
            elif len(b_planes.values) == 1:
                b_unique_plane = b_planes['Z'].values[0] + 1
                iso_df = crystal_df.query('Z >= @b_unique_plane')
  
            else:
                print('There was not found {} planes'.format(B))
                return None

            # Update the box to be 10 amstrong in Z
            try:
                iso_df.loc[:, 'Z'] = iso_df['Z'] - iso_df['Z'].min()
                box = self.box
                box[0][2][2] = iso_df['Z'].sort_values(ascending=False).iloc[0] + 10
                # Save the system
                name = self.path + '/' + 'salt_' + self.name
                print('Your isolated salt file was save as: ', name)
                save_vasp(iso_df, box, name, order=order)

            except Exception as e:
                print(f"Error creating the SALT: {e}")

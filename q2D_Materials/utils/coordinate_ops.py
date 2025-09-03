import numpy as np
import pandas as pd
from ase.io import read
from .file_handlers import mol_load

def make_svc(DF_MA_1, DF_MA_2):
    dis_1 = DF_MA_1.sort_values(by='Z').iloc[0, 3]
    DF_MA_1['Z'] = DF_MA_1['Z'].apply(lambda x: x - dis_1)
    dis_2 = DF_MA_2.sort_values(by='Z').iloc[0, 3]
    DF_MA_2['Z'] = DF_MA_2['Z'].apply(lambda x: x - dis_2)
    organic_spacers = pd.concat([DF_MA_1, DF_MA_2])
    organic_spacers.reset_index(drop=True, inplace=True)
    return organic_spacers

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

def direct_to_cartesian(df, lattice_vectors):
    df = df
    # Convert fractional coordinates to Cartesian coordinates
    atomic_positions_direct = df[['x_direct', 'y_direct', 'z_direct']].values
    cartesian_positions = np.dot(atomic_positions_direct, lattice_vectors)

    # Create a new DataFrame with Cartesian coordinates
    df_cartesian = pd.concat([df, pd.DataFrame(cartesian_positions, columns=['x_cartesian', 'y_cartesian', 'z_cartesian'])], axis=1)

    return df_cartesian 

def shift_structure(atoms, direction='a', shift=0.5):
    """
    Shift the structure in the specified direction.
    Enhanced with validation and error handling similar to inspiration.py.
    
    Args:
        atoms: ASE Atoms object
        direction (str): Direction to shift ('a', 'b', or 'c')
        shift (float): Amount to shift in fractional coordinates
        
    Returns:
        ASE Atoms object with shifted positions
    """
    if atoms is None:
        raise ValueError("Atoms object cannot be None")
        
    # Get cell vectors
    cell = atoms.get_cell()
    
    # Validate cell
    if np.any(np.isnan(cell)) or np.any(np.isinf(cell)):
        raise ValueError("Invalid cell parameters detected")
    
    # Convert shift to real space based on direction
    if direction == 'a':
        shift_vector = cell[0] * shift
    elif direction == 'b':
        shift_vector = cell[1] * shift
    elif direction == 'c':
        shift_vector = cell[2] * shift
    else:
        raise ValueError("Direction must be 'a', 'b', or 'c'")
    
    # Create a copy of the atoms object
    shifted_atoms = atoms.copy()
    
    # Shift all positions
    positions = shifted_atoms.get_positions()
    
    # Validate positions
    if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
        raise ValueError("Invalid atomic positions detected")
    
    shifted_atoms.set_positions(positions + shift_vector)
    
    return shifted_atoms


def calculate_center_of_mass(df, mass_weighted=True):
    """
    Calculate center of mass for a molecular DataFrame.
    Similar to inspiration.py's center of mass calculations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Molecule data with 'Element', 'X', 'Y', 'Z' columns
    mass_weighted : bool
        Whether to use atomic masses for weighting
        
    Returns
    -------
    np.array
        Center of mass coordinates [x, y, z]
    """
    if mass_weighted:
        # Simple atomic masses (approximate)
        mass_dict = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
            'F': 18.998, 'Cl': 35.453, 'Br': 79.904, 'I': 126.904,
            'Pb': 207.2, 'Sn': 118.71, 'Ge': 72.64, 'Cs': 132.905,
            'Rb': 85.468, 'K': 39.098, 'Na': 22.990, 'Li': 6.941
        }
        
        masses = df['Element'].map(mass_dict).fillna(1.0)  # Default mass = 1
        total_mass = masses.sum()
        
        if total_mass == 0:
            raise ValueError("Total mass is zero")
        
        weighted_coords = df[['X', 'Y', 'Z']].values * masses.values.reshape(-1, 1)
        center_of_mass = weighted_coords.sum(axis=0) / total_mass
    else:
        # Geometric center
        center_of_mass = df[['X', 'Y', 'Z']].values.mean(axis=0)
    
    return center_of_mass


def validate_molecular_structure(df):
    """
    Validate molecular structure DataFrame.
    Enhanced validation similar to inspiration.py checks.
    
    Parameters
    ----------
    df : pd.DataFrame
        Molecule data to validate
        
    Returns
    -------
    bool
        True if valid, raises ValueError if invalid
    """
    required_columns = ['Element', 'X', 'Y', 'Z']
    
    # Check required columns
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Need: {required_columns}")
    
    # Check for empty dataframe
    if df.empty:
        raise ValueError("Molecule DataFrame is empty")
    
    # Check for NaN values
    if df[['X', 'Y', 'Z']].isnull().any().any():
        raise ValueError("NaN values found in coordinates")
    
    # Check for infinite values
    if np.any(np.isinf(df[['X', 'Y', 'Z']].values)):
        raise ValueError("Infinite values found in coordinates")
    
    # Check for valid element symbols
    valid_elements = {
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'
    }
    
    invalid_elements = set(df['Element']) - valid_elements
    if invalid_elements:
        raise ValueError(f"Invalid element symbols found: {invalid_elements}")
    
    print(f"Molecular structure validation passed: {len(df)} atoms")
    return True 
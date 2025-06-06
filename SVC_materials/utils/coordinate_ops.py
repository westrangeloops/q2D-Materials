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
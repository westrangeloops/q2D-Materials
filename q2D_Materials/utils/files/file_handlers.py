import pandas as pd
import numpy as np
from ase.io import read

def mol_load(file):
    # Load the molecule from the file
    if file.endswith('.xyz'):
        with open(file, 'r') as file:
                lines = file.readlines()
        # Skip the first two lines (atom count and comment)
        atoms = [line.split() for line in lines[2:]]
    else:
        print('WARNING: Not a valid XYZ')

    # Create a Pandas DataFrame with the atomic coordinates
    df = pd.DataFrame(atoms, columns=['Element', 'X', 'Y', 'Z'])
    df['X'] = df['X'].astype('float')
    df['Y'] = df['Y'].astype('float')
    df['Z'] = df['Z'].astype('float')
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
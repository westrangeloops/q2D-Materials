import numpy as np
from ase.io import read
from ase.visualize import view
from os import remove
from ..utils.molecular_ops import rot_mol, transform_mol, align_nitrogens, inclinate_molecule
from ..utils.file_handlers import mol_load, vasp_load, save_vasp
from ..utils.coordinate_ops import make_svc, bulk_creator, iso
from ..utils.smiles_handler import smiles_to_xyz, RDKIT_AVAILABLE
import tempfile

def pi(argu):
    return np.array(list(map(float,argu.strip().split()))[:3])

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

    @classmethod
    def from_smiles(cls, B, X, smiles, perov_vasp, P1, P2, P3, Q1, Q2, Q3, name, vac=0, n=1):
        """
        Creates a q2D structure from a SMILES string for the molecule.
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is not installed. Cannot create from SMILES.")
        
        # Create a temporary XYZ file from the SMILES string
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.xyz') as tmp_file:
            smiles_to_xyz(smiles, tmp_file.name)
            molecule_xyz = tmp_file.name

        instance = cls(B, X, molecule_xyz, perov_vasp, P1, P2, P3, Q1, Q2, Q3, name, vac, n)
        
        # Clean up the temporary file
        remove(molecule_xyz)
        
        return instance 
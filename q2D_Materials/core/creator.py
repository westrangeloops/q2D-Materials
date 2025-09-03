import numpy as np
from ase.io import read, write
from ase.visualize import view
from ase.build import add_adsorbate, molecule, bulk
from ase import Atoms
from os import remove
from ..utils.molecular_ops import rot_mol, transform_mol, align_nitrogens, inclinate_molecule
from ..utils.file_handlers import mol_load, vasp_load, save_vasp
from ..utils.coordinate_ops import make_svc, bulk_creator, iso
from ..utils.smiles_handler import smiles_to_xyz, RDKIT_AVAILABLE
from ..utils.perovskite_builder import (
    make_bulk, make_double, make_2drp, make_dj, make_monolayer, 
    make_2d_double, determine_molecule_orientation, orient_along_z,
    auto_calculate_BX_distance, validate_perovskite_composition
)
from ..utils.common_a_sites import (
    get_ionic_radius, calculate_BX_distance, calculate_tolerance_factor
)
import tempfile
import sys

def pi(argu):
    """Parse coordinate string to numpy array."""
    return np.array(list(map(float,argu.strip().split()))[:3])

class q2D_creator:
    """
    Enhanced q2D perovskite structure creator with integrated ASE-based functionality.
    
    This class provides both the original SVC-Materials workflow and the new
    comprehensive perovskite building capabilities from perovskite_builder.py.
    Similar to the inspiration.py Perovskite class but tailored for 2D materials.
    
    Parameters
    ----------
    B : str
        B-site cation symbol (e.g., 'Pb', 'Sn')
    X : str  
        X-site anion symbol (e.g., 'I', 'Br', 'Cl')
    molecule_xyz : str
        Path to XYZ file containing the organic molecule
    perov_vasp : str
        Path to VASP file containing reference perovskite structure
    P1, P2, P3 : str
        Position coordinates for first orientation (legacy workflow)
    Q1, Q2, Q3 : str
        Position coordinates for second orientation (legacy workflow)
    name : str
        Name identifier for the structure
    vac : float, optional
        Vacuum space in Angstroms (default: 0)
    n : int, optional
        Layer thickness parameter (default: 1)
    Ap : str, optional
        A'-site spacer cation for 2D structures
    A : str, optional
        A-site cation for bulk structures (default: 'Cs')
    Bp : str, optional
        Second B-site cation for double perovskites
    """
    
    def __init__(self, B, X, molecule_xyz, perov_vasp, P1, P2, P3, Q1, Q2, Q3, name, 
                 vac=0, n=1, Ap=None, A=None, Bp=None):
        # Core composition
        self.B = B
        self.X = X
        self.Bp = Bp
        self.A = A if A is not None else 'Cs'  # Default A-site for 2D structures
        self.Ap = Ap  # Spacer cation
        self.name = name
        self.vac = vac
        self.n = n
        
        # File references
        self.molecule_file = molecule_xyz
        self.perovskite_file = perov_vasp
        
        # Load molecule and perovskite data
        self.molecule_df = mol_load(self.molecule_file)
        self.perovskite_df, self.box = vasp_load(perov_vasp)
        
        # Convert molecule to ASE Atoms for enhanced functionality
        self.molecule_atoms = self._df_to_ase_atoms(self.molecule_df)
        
        # Calculate optimal B-X distance from ionic radii
        self.optimal_BX_dist = auto_calculate_BX_distance(self.B, self.X)
        
        # Analyze composition stability
        self.composition_analysis = self._analyze_composition()
        
        # Initialize legacy attributes for original workflow
        # These will only be set if the molecule has the required atoms
        self.P1, self.P2, self.P3, self.Q1, self.Q2, self.Q3 = pi(P1), pi(P2), pi(P3), pi(Q1), pi(Q2), pi(Q3)
        
        # Align molecule if it has nitrogens
        self.molecule_df = align_nitrogens(self.molecule_df)
        
        # Check for required atoms for original workflow
        n_nitrogens = (self.molecule_df['Element'] == 'N').sum()
        n_carbons = (self.molecule_df['Element'] == 'C').sum()
        
        if n_nitrogens >= 2 and n_carbons >= 1:
            try:
                self.molecule_D1 = self.molecule_df.loc[self.molecule_df['Element'] == 'N', 'X':'Z'].sort_values(by='Z').values[0]
                self.molecule_D2 = self.molecule_df.loc[self.molecule_df['Element'] == 'C', 'X':'Z'].sort_values(by='Z', ascending=False).values[0]
                self.molecule_D3 = self.molecule_df.loc[self.molecule_df['Element'] == 'N', 'X':'Z'].sort_values(by='Z').values[1]
                
                self.DF_MA_1 = transform_mol(self, self.P1, self.P2, self.P3)
                self.DF_MA_2 = transform_mol(self, self.Q1, self.Q2, self.Q3)
                self.svc = make_svc(self.DF_MA_1, self.DF_MA_2)
            except IndexError:
                # This can happen if sorting doesn't yield enough values
                self.svc = self.molecule_df.copy() # Fallback
        else:
            # If not, the legacy structures cannot be built.
            # We can initialize svc with the molecule for ASE-based methods.
            self.svc = self.molecule_df.copy()
            
        # Enhanced functionality flags
        self._structures_cache = {}  # Cache for generated structures
        self._analysis_cache = {}    # Cache for analysis results

    def _df_to_ase_atoms(self, df):
        """Convert molecule DataFrame to ASE Atoms object."""
        elements = df['Element'].tolist()
        positions = df[['X', 'Y', 'Z']].values
        return Atoms(symbols=elements, positions=positions)
    
    def _analyze_composition(self):
        """Analyze the perovskite composition using ionic radii data."""
        try:
            return validate_perovskite_composition(self.A, self.B, self.X, show_analysis=False)
        except:
            return None
    
    def get_composition_info(self):
        """
        Get detailed composition analysis including tolerance factor and stability.
        Similar to inspiration.py's compute_goldschmidt_tolerance and related methods.
        
        Returns
        -------
        dict
            Composition analysis results
        """
        if self.composition_analysis is None:
            self.composition_analysis = self._analyze_composition()
        
        if self.composition_analysis:
            print(f"\nComposition Analysis: {self.composition_analysis['composition']}")
            print("=" * 50)
            print(f"Tolerance Factor: {self.composition_analysis['tolerance_factor']:.3f}")
            print(f"B-X Distance: {self.composition_analysis['BX_distance']:.3f} Å")
            print(f"Estimated Lattice Parameter: {self.composition_analysis['estimated_lattice_parameter']:.3f} Å")
            print(f"Stability Assessment: {self.composition_analysis['stability_assessment']}")
            print("=" * 50)
        
        return self.composition_analysis
    
    def get_optimal_BX_distance(self):
        """
        Get the optimal B-X bond distance based on ionic radii.
        Similar to automatic distance calculation in inspiration.py.
        
        Returns
        -------
        float
            Optimal B-X bond distance in Angstroms
        """
        return self.optimal_BX_dist

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
    
    @classmethod
    def from_ase_molecule(cls, B, X, ase_molecule, name, vac=0, n=1, 
                         default_positions=None):
        """
        Creates a q2D structure from an ASE Atoms molecule object.
        This enables direct integration with the advanced perovskite builders.
        
        Parameters
        ----------
        B : str
            B cation symbol
        X : str
            X anion symbol
        ase_molecule : ase.Atoms
            ASE Atoms object containing the molecule
        name : str
            Name for the structure
        vac : float
            Vacuum space
        n : int
            Layer number
        default_positions : array-like, optional
            Default positions for P1, P2, P3, Q1, Q2, Q3
            
        Returns
        -------
        q2D_creator
            Instance with the ASE molecule converted to internal format
        """
        # Convert ASE molecule to temporary XYZ file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.xyz') as tmp_file:
            # Write XYZ format
            symbols = ase_molecule.get_chemical_symbols()
            positions = ase_molecule.get_positions()
            
            tmp_file.write(f"{len(symbols)}\n")
            tmp_file.write(f"Generated from ASE molecule\n")
            for symbol, pos in zip(symbols, positions):
                tmp_file.write(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
            
            molecule_xyz = tmp_file.name
        
        # Create a dummy perovskite file (minimal VASP structure)
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.vasp') as tmp_vasp:
            tmp_vasp.write(f"Dummy perovskite\n")
            tmp_vasp.write("1.0\n")
            tmp_vasp.write("4.0 0.0 0.0\n")
            tmp_vasp.write("0.0 4.0 0.0\n")
            tmp_vasp.write("0.0 0.0 4.0\n")
            tmp_vasp.write(f"{B} {X}\n")
            tmp_vasp.write("1 3\n")
            tmp_vasp.write("Cartesian\n")
            tmp_vasp.write("2.0 2.0 2.0\n")
            tmp_vasp.write("0.0 2.0 2.0\n")
            tmp_vasp.write("2.0 0.0 2.0\n")
            tmp_vasp.write("2.0 2.0 0.0\n")
            
            perov_vasp = tmp_vasp.name
        
        # Default positions if not provided
        if default_positions is None:
            default_positions = [
                "0 0 0", "2 2 0", "0 0 4",  # P1, P2, P3
                "0 0 8", "2 2 8", "0 0 12"  # Q1, Q2, Q3
            ]
        
        P1, P2, P3, Q1, Q2, Q3 = default_positions
        
        try:
            instance = cls(B, X, molecule_xyz, perov_vasp, P1, P2, P3, Q1, Q2, Q3, name, vac, n)
        finally:
            # Clean up temporary files
            remove(molecule_xyz)
            remove(perov_vasp)
        
        return instance

    def create_ase_molecule(self):
        """
        Convert the internal molecule representation to ASE Atoms object.
        This enables compatibility with the advanced perovskite builders.
        
        Returns
        -------
        ase.Atoms
            ASE Atoms object of the molecule
        """
        from ase import Atoms
        
        # Extract coordinates and elements from molecule dataframe
        elements = self.molecule_df['Element'].tolist()
        positions = self.molecule_df[['X', 'Y', 'Z']].values
        
        # Create ASE Atoms object
        ase_mol = Atoms(symbols=elements, positions=positions)
        
        return ase_mol
    
    def create_perovskite_bulk(self, BX_dist=None, cache=True):
        """
        Create bulk perovskite structure using the advanced builder.
        Similar to inspiration.py workflow but with enhanced automation.
        Molecular alignment is handled automatically.
        
        Parameters
        ----------
        BX_dist : float, optional
            The desired B-X bond distance in Angstrom. If not provided,
            will be calculated automatically from ionic radii data.
        cache : bool
            Whether to cache the result for reuse
            
        Returns
        -------
        ase.Atoms
            Bulk perovskite structure
        """
        if cache and 'bulk' in self._structures_cache:
            return self._structures_cache['bulk']
        
        # Use optimal B-X distance if not provided
        if BX_dist is None:
            BX_dist = self.optimal_BX_dist
            print(f"Using optimal B-X distance: {BX_dist:.3f} Å")
        
        # Use molecule as A-site cation and align it properly (handled internally)
        from ..utils.molecular_ops import align_ase_molecule_for_perovskite
        A_molecule = align_ase_molecule_for_perovskite(self.molecule_atoms)
        print(f"Bulk A-site cation molecule aligned for perovskite coordination")
        
        structure = make_bulk(A_molecule, self.B, self.X, BX_dist)
        
        if cache:
            self._structures_cache['bulk'] = structure
            
        return structure
    
    def create_2d_rp(self, n=1, BX_dist=None, penet=0.3, interlayer_penet=0.4, 
                     Ap_Rx=None, Ap_Ry=None, Ap_Rz=None, wrap=False, 
                     output=False, output_type='vasp', file_name=None, cache=True):
        """
        Create 2D Ruddlesden-Popper perovskite structure.
        Enhanced with automatic parameter optimization similar to inspiration.py.
        Molecular alignment is handled automatically.
        
        Parameters
        ----------
        n : int
            Layer thickness of inorganic 2D layers
        BX_dist : float, optional
            Desired B-X bond distance in Angstrom. If not provided,
            will be calculated automatically from ionic radii data.
        penet : float
            Penetration of spacer into inorganic layer (fraction of BX bond)
        interlayer_penet : float
            Interlayer penetration for RP phase (fraction of molecule length)
        Ap_Rx, Ap_Ry, Ap_Rz : float, optional
            Rotation angles in degrees around x, y, z axes
        wrap : bool
            Whether to wrap atoms outside unit cell
        output : bool
            Whether to output structure to file
        output_type : str
            File format for output ('vasp', 'cif', 'xyz')
        file_name : str, optional
            Name for output file
        cache : bool
            Whether to cache the result
            
        Returns
        -------
        ase.Atoms
            2D Ruddlesden-Popper perovskite structure
        """
        cache_key = f'rp_n{n}_p{penet}_ip{interlayer_penet}'
        if cache and cache_key in self._structures_cache:
            return self._structures_cache[cache_key]
        
        # Use optimal B-X distance if not provided
        if BX_dist is None:
            BX_dist = self.optimal_BX_dist
            print(f"Using optimal B-X distance for RP: {BX_dist:.3f} Å")
        
        # Use molecule as A' spacer cation and align it properly (handled internally)
        from ..utils.molecular_ops import align_ase_molecule_for_perovskite
        Ap_molecule = align_ase_molecule_for_perovskite(self.molecule_atoms)
        print(f"RP spacer molecule aligned for perovskite coordination")
        
        # Use the configured A-site cation
        A_cation = self.A
        
        structure = make_2drp(Ap_molecule, A_cation, self.B, self.X, n, BX_dist,
                             penet=penet, interlayer_penet=interlayer_penet,
                             Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz, wrap=wrap,
                             output=output, output_type=output_type, file_name=file_name)
        
        if cache:
            self._structures_cache[cache_key] = structure
            
        return structure
    
    def create_2d_dj(self, n=1, BX_dist=None, penet=0.3, Ap_Rx=None, 
                     Ap_Ry=None, Ap_Rz=None, attachment_end='both', 
                     wrap=False, output=False, output_type='vasp', file_name=None, cache=True):
        """
        Create 2D Dion-Jacobson perovskite structure.
        Enhanced with automatic parameter optimization and caching.
        Molecular alignment is handled automatically.
        
        Parameters
        ----------
        n : int
            Layer thickness of inorganic 2D layers
        BX_dist : float, optional
            Desired B-X bond distance in Angstrom. If not provided,
            will be calculated automatically from ionic radii data.
        penet : float
            Penetration of spacer into inorganic layer (fraction of BX bond)
        Ap_Rx, Ap_Ry, Ap_Rz : float, optional
            Rotation angles in degrees around x, y, z axes
        attachment_end : str
            Where to attach spacer ('top', 'bottom', 'both')
        wrap : bool
            Whether to wrap atoms outside unit cell
        output : bool
            Whether to output structure to file
        output_type : str
            File format for output ('vasp', 'cif', 'xyz')
        file_name : str, optional
            Name for output file
        cache : bool
            Whether to cache the result
            
        Returns
        -------
        ase.Atoms
            2D Dion-Jacobson perovskite structure
        """
        cache_key = f'dj_n{n}_p{penet}_{attachment_end}'
        if cache and cache_key in self._structures_cache:
            return self._structures_cache[cache_key]
        
        # Use optimal B-X distance if not provided
        if BX_dist is None:
            BX_dist = self.optimal_BX_dist
            print(f"Using optimal B-X distance for DJ: {BX_dist:.3f} Å")
        
        # Use molecule as A' spacer cation and align it properly (handled internally)
        from ..utils.molecular_ops import align_ase_molecule_for_perovskite
        Ap_molecule = align_ase_molecule_for_perovskite(self.molecule_atoms)
        print(f"DJ spacer molecule aligned for perovskite coordination")
        
        # Use the configured A-site cation
        # Handle molecular A-site cations (like MA) using common_a_sites
        from q2D_Materials.utils.common_a_sites import get_a_site_object
        A_cation = get_a_site_object(self.A)
        
        structure = make_dj(Ap_molecule, A_cation, self.B, self.X, n, BX_dist,
                           penet=penet, Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz,
                           attachment_end=attachment_end, wrap=wrap, output=output,
                           output_type=output_type, file_name=file_name)
        
        if cache:
            self._structures_cache[cache_key] = structure
            
        return structure
    
    def create_2d_monolayer(self, n=1, BX_dist=None, penet=0.3, vacuum=12,
                           Ap_Rx=None, Ap_Ry=None, Ap_Rz=None, wrap=False,
                           output=False, output_type='vasp', file_name=None):
        """
        Create 2D monolayer perovskite structure.
        Molecular alignment is handled automatically.
        
        Parameters
        ----------
        n : int
            Layer thickness of inorganic 2D layers
        BX_dist : float
            Desired B-X bond distance in Angstrom
        penet : float
            Penetration of spacer into inorganic layer (fraction of BX bond)
        vacuum : float
            Amount of vacuum to add to unit cell in Angstrom
        Ap_Rx, Ap_Ry, Ap_Rz : float, optional
            Rotation angles in degrees around x, y, z axes
        wrap : bool
            Whether to wrap atoms outside unit cell
        output : bool
            Whether to output structure to file
        output_type : str
            File format for output ('vasp', 'cif', 'xyz')
        file_name : str, optional
            Name for output file
            
        Returns
        -------
        ase.Atoms
            2D monolayer perovskite structure
        """
        from ..utils.perovskite_builder import make_monolayer
        
        # Use molecule as A' spacer cation and align it properly (handled internally)
        from ..utils.molecular_ops import align_ase_molecule_for_perovskite
        Ap_molecule = align_ase_molecule_for_perovskite(self.molecule_atoms)
        print(f"Monolayer spacer molecule aligned for perovskite coordination")
        
        # For 2D structures, A can be a simple cation
        A_cation = 'Cs'  # Default, can be modified
        
        return make_monolayer(Ap_molecule, A_cation, self.B, self.X, n, BX_dist,
                             penet=penet, vacuum=vacuum, Ap_Rx=Ap_Rx, 
                             Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz, wrap=wrap, output=output,
                             output_type=output_type, file_name=file_name)
    
    def create_double_perovskite(self, Bp, BX_dist=2.0):
        """
        Create double perovskite structure.
        Molecular alignment is handled automatically.
        
        Parameters
        ----------
        Bp : str
            Second B cation symbol
        BX_dist : float
            Desired B-X bond distance in Angstrom
            
        Returns
        -------
        ase.Atoms
            Double perovskite structure
        """
        from ..utils.perovskite_builder import make_double
        
        # Use molecule as A-site cation and align it properly (handled internally)
        from ..utils.molecular_ops import align_ase_molecule_for_perovskite
        A_molecule = align_ase_molecule_for_perovskite(self.molecule_atoms)
        print(f"Double perovskite A-site cation molecule aligned for perovskite coordination")
        
        return make_double(A_molecule, self.B, Bp, self.X, BX_dist)
    
    def create_2d_double_perovskite(self, Bp, phase='rp', n=1, BX_dist=2.0, 
                                   penet=0.3, interlayer_penet=0.4,
                                   Ap_Rx=None, Ap_Ry=None, Ap_Rz=None, 
                                   wrap=False, output=False, output_type='vasp', 
                                   file_name=None):
        """
        Create 2D double perovskite structure.
        Molecular alignment is handled automatically.
        
        Parameters
        ----------
        Bp : str
            Second B cation symbol
        phase : str
            Phase type ('rp', 'dj', 'monolayer')
        n : int
            Layer thickness of inorganic 2D layers
        BX_dist : float
            Desired B-X bond distance in Angstrom
        penet : float
            Penetration of spacer into inorganic layer (fraction of BX bond)
        interlayer_penet : float
            Interlayer penetration for RP phase (fraction of molecule length)
        Ap_Rx, Ap_Ry, Ap_Rz : float, optional
            Rotation angles in degrees around x, y, z axes
        wrap : bool
            Whether to wrap atoms outside unit cell
        output : bool
            Whether to output structure to file
        output_type : str
            File format for output ('vasp', 'cif', 'xyz')
        file_name : str, optional
            Name for output file
            
        Returns
        -------
        ase.Atoms
            2D double perovskite structure
        """
        from ..utils.perovskite_builder import make_2d_double
        
        # Use molecule as A' spacer cation and align it properly (handled internally)
        from ..utils.molecular_ops import align_ase_molecule_for_perovskite
        Ap_molecule = align_ase_molecule_for_perovskite(self.molecule_atoms)
        print(f"2D double perovskite spacer molecule aligned for perovskite coordination")
        
        # For 2D structures, A can be a simple cation
        A_cation = 'Cs'  # Default, can be modified
        
        return make_2d_double(Ap_molecule, A_cation, self.B, Bp, self.X, n, 
                             BX_dist, phase=phase, penet=penet, 
                             interlayer_penet=interlayer_penet,
                             Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz,
                             wrap=wrap, output=output, output_type=output_type,
                             file_name=file_name)
    
    def analyze_molecule_orientation(self):
        """
        Analyze the orientation of the molecule to aid in structure building.
        
        Returns
        -------
        str
            Best guess for axis of orientation ('x', 'y', or 'z')
        """
        from ..utils.perovskite_builder import determine_molecule_orientation
        
        ase_mol = self.create_ase_molecule()
        return determine_molecule_orientation(ase_mol)
    
    def orient_molecule_along_z(self, theta=90, invert=False):
        """
        Orient the molecule along the Z-axis for better structure building.
        
        Parameters
        ----------
        theta : float
            Rotation angle in degrees
        invert : bool
            Whether to invert the molecule before rotation
            
        Returns
        -------
        ase.Atoms
            Reoriented molecule
        """
        from ..utils.perovskite_builder import orient_along_z
        
        ase_mol = self.create_ase_molecule()
        return orient_along_z(ase_mol, theta=theta, invert=invert)
    
    def view_ase_structure(self, structure):
        """
        Visualize an ASE structure.
        
        Parameters
        ----------
        structure : ase.Atoms
            Structure to visualize
        """
        return view(structure)
    
    def analyze_composition(self, A_cation=None, show_analysis=True):
        """
        Analyze the perovskite composition using ionic radii data.
        
        Parameters
        ----------
        A_cation : str, optional
            A-site cation symbol. If not provided, uses 'Cs' as default.
        show_analysis : bool
            Whether to print detailed analysis
            
        Returns
        -------
        dict
            Analysis results including tolerance factor and stability
        """
        from ..utils.perovskite_builder import validate_perovskite_composition
        
        if A_cation is None:
            A_cation = 'Cs'  # Default for 2D structures
            
        return validate_perovskite_composition(A_cation, self.B, self.X, show_analysis)
    
    def get_optimal_BX_distance(self):
        """
        Get the optimal B-X bond distance based on ionic radii.
        
        Returns
        -------
        float
            Optimal B-X bond distance in Angstroms
        """
        from ..utils.perovskite_builder import auto_calculate_BX_distance
        
        return auto_calculate_BX_distance(self.B, self.X)
    
    def print_available_ions(self, site='all'):
        """
        Print available ions in the ionic radii database.
        
        Parameters
        ----------
        site : str
            Which site to show ('A', 'B', 'X', or 'all')
        """
        from ..utils.ionic_data import print_available_ionic_radii
        
        print_available_ionic_radii(mode=site)
    
    def write_structure(self, structure, filename=None, file_format='vasp'):
        """
        Write a structure to file in various formats.
        Similar to inspiration.py's write_xTB and write_GPAW methods.
        
        Parameters
        ----------
        structure : ase.Atoms
            The structure to write
        filename : str, optional
            Output filename. If not provided, generates based on composition
        file_format : str
            Output format ('vasp', 'cif', 'xyz', 'json')
        """
        if filename is None:
            filename = f"{self.name}_{self.B}{self.X}3"
        
        if not filename.endswith(f'.{file_format}'):
            filename += f'.{file_format}'
        
        write(filename, structure)
        print(f"Structure written to {filename}")
    
    def get_structure_info(self, structure):
        """
        Get basic information about a structure.
        Similar to inspiration.py's analysis methods.
        
        Parameters
        ----------
        structure : ase.Atoms
            The structure to analyze
            
        Returns
        -------
        dict
            Structure information
        """
        info = {
            'formula': structure.get_chemical_formula(),
            'num_atoms': len(structure),
            'cell_volume': structure.get_volume(),
            'cell_parameters': structure.cell.cellpar(),
            'density': len(structure) / structure.get_volume(),  # atoms per Å³
        }
        
        return info
    
    def print_structure_summary(self, structure):
        """
        Print a summary of the structure.
        Similar to inspiration.py's output formatting.
        
        Parameters
        ----------
        structure : ase.Atoms
            The structure to summarize
        """
        info = self.get_structure_info(structure)
        
        print(f"\nStructure Summary: {self.name}")
        print("=" * 40)
        print(f"Formula: {info['formula']}")
        print(f"Number of atoms: {info['num_atoms']}")
        print(f"Cell volume: {info['cell_volume']:.2f} Å³")
        print(f"Density: {info['density']:.4f} atoms/Å³")
        print(f"Cell parameters: a={info['cell_parameters'][0]:.3f}, "
              f"b={info['cell_parameters'][1]:.3f}, c={info['cell_parameters'][2]:.3f} Å")
        print("=" * 40)
    
    def auto_generate_series(self, n_range=[1, 2, 3, 4], phase='dj', 
                           penet=0.3, output_dir='.'):
        """
        Automatically generate a series of structures with different layer thicknesses.
        Similar to high-throughput capabilities in inspiration.py.
        
        Parameters
        ----------
        n_range : list
            Range of layer thicknesses to generate
        phase : str
            Phase type ('dj', 'rp', 'monolayer')
        penet : float
            Penetration parameter
        output_dir : str
            Directory to save structures
            
        Returns
        -------
        dict
            Dictionary of generated structures
        """
        structures = {}
        
        print(f"\nGenerating {phase.upper()} structures for {self.name}")
        print(f"Composition: {self.A}{self.B}{self.X}3 with spacer")
        print(f"B-X distance: {self.optimal_BX_dist:.3f} Å")
        print("=" * 50)
        
        for n in n_range:
            try:
                if phase.lower() == 'dj':
                    structure = self.create_2d_dj(
                        n=n, penet=penet, output=True, 
                        file_name=f"{output_dir}/DJ_{self.name}_n{n}"
                    )
                elif phase.lower() == 'rp':
                    structure = self.create_2d_rp(
                        n=n, penet=penet, output=True,
                        file_name=f"{output_dir}/RP_{self.name}_n{n}"
                    )
                elif phase.lower() == 'monolayer':
                    structure = self.create_2d_monolayer(
                        n=n, penet=penet, output=True,
                        file_name=f"{output_dir}/ML_{self.name}_n{n}"
                    )
                else:
                    raise ValueError(f"Unknown phase: {phase}")
                
                structures[f'n{n}'] = structure
                print(f"✓ n={n}: {len(structure)} atoms")
                
            except Exception as e:
                print(f"✗ n={n}: Failed - {e}")
                
        print("=" * 50)
        print(f"Generated {len(structures)} structures")
        
        return structures
    
    def compare_phases(self, n=2, penet=0.3):
        """
        Compare different phases (DJ, RP, Monolayer) for the same composition.
        Similar to comparative analysis in inspiration.py.
        
        Parameters
        ----------
        n : int
            Layer thickness
        penet : float
            Penetration parameter
            
        Returns
        -------
        dict
            Comparison results
        """
        phases = {}
        
        print(f"\nComparing phases for {self.name} (n={n})")
        print("=" * 50)
        
        try:
            phases['DJ'] = self.create_2d_dj(n=n, penet=penet)
            dj_info = self.get_structure_info(phases['DJ'])
            print(f"DJ: {dj_info['num_atoms']} atoms, {dj_info['cell_volume']:.1f} Å³")
        except Exception as e:
            print(f"DJ: Failed - {e}")
            
        try:
            phases['RP'] = self.create_2d_rp(n=n, penet=penet)
            rp_info = self.get_structure_info(phases['RP'])
            print(f"RP: {rp_info['num_atoms']} atoms, {rp_info['cell_volume']:.1f} Å³")
        except Exception as e:
            print(f"RP: Failed - {e}")
            
        try:
            phases['Monolayer'] = self.create_2d_monolayer(n=n, penet=penet)
            ml_info = self.get_structure_info(phases['Monolayer'])
            print(f"Monolayer: {ml_info['num_atoms']} atoms, {ml_info['cell_volume']:.1f} Å³")
        except Exception as e:
            print(f"Monolayer: Failed - {e}")
            
        print("=" * 50)
        
        return phases
    
    def create_perovskite(self, structure_type="bulk", n=1, BX_dist=None, penet=0.3, 
                         interlayer_penet=0.4, vacuum=12, double=None, 
                         Ap_Rx=None, Ap_Ry=None, Ap_Rz=None, attachment_end='both',
                         wrap=False, output=False, output_type='vasp', file_name=None, cache=True):
        """
        Universal perovskite structure creator that can generate all types of structures.
        This is the main interface that replaces individual create_* methods.
        
        Parameters
        ----------
        structure_type : str
            Type of structure to create:
            - 'bulk': 3D bulk perovskite
            - 'RP' or 'rp': 2D Ruddlesden-Popper phase
            - 'DJ' or 'dj': 2D Dion-Jacobson phase
            - 'monolayer' or 'ML': 2D monolayer phase
        n : int
            Layer thickness for 2D structures (ignored for bulk)
        BX_dist : float, optional
            B-X bond distance. If None, calculated automatically
        penet : float
            Penetration parameter for 2D structures
        interlayer_penet : float
            Interlayer penetration for RP phase
        vacuum : float
            Vacuum space for monolayer structures
        double : str, optional
            Second B-site cation for double perovskites (e.g., 'Sn')
        Ap_Rx, Ap_Ry, Ap_Rz : float, optional
            Molecular rotation angles
        attachment_end : str
            Spacer attachment for DJ structures ('top', 'bottom', 'both')
        wrap : bool
            Whether to wrap atoms
        output : bool
            Whether to save to file
        output_type : str
            Output format ('vasp', 'cif', 'xyz')
        file_name : str, optional
            Output filename
        cache : bool
            Whether to cache results
            
        Returns
        -------
        ase.Atoms
            The created perovskite structure
            
        Examples
        --------
        >>> # Create bulk perovskite
        >>> structure = creator.create_perovskite("bulk")
        
        >>> # Create DJ phase with n=2
        >>> structure = creator.create_perovskite("DJ", n=2, penet=0.2)
        
        >>> # Create double perovskite RP phase
        >>> structure = creator.create_perovskite("RP", n=1, double="Sn")
        
        >>> # Create monolayer with vacuum
        >>> structure = creator.create_perovskite("monolayer", vacuum=15)
        """
        structure_type = structure_type.upper()
        
        if structure_type == "BULK":
            if double:
                return self.create_double_perovskite(Bp=double, BX_dist=BX_dist)
            else:
                return self.create_perovskite_bulk(BX_dist=BX_dist, cache=cache)
                
        elif structure_type in ["RP", "RUDDLESDEN-POPPER"]:
            if double:
                return self.create_2d_double_perovskite(
                    Bp=double, phase='rp', n=n, BX_dist=BX_dist, penet=penet,
                    interlayer_penet=interlayer_penet, Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, 
                    Ap_Rz=Ap_Rz, wrap=wrap, output=output, output_type=output_type,
                    file_name=file_name
                )
            else:
                return self.create_2d_rp(
                    n=n, BX_dist=BX_dist, penet=penet, interlayer_penet=interlayer_penet,
                    Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz, wrap=wrap, output=output,
                    output_type=output_type, file_name=file_name, cache=cache
                )
                
        elif structure_type in ["DJ", "DION-JACOBSON"]:
            if double:
                return self.create_2d_double_perovskite(
                    Bp=double, phase='dj', n=n, BX_dist=BX_dist, penet=penet,
                    Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz, wrap=wrap, output=output,
                    output_type=output_type, file_name=file_name
                )
            else:
                return self.create_2d_dj(
                    n=n, BX_dist=BX_dist, penet=penet, Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry,
                    Ap_Rz=Ap_Rz, attachment_end=attachment_end, wrap=wrap, output=output,
                    output_type=output_type, file_name=file_name, cache=cache
                )
                
        elif structure_type in ["MONOLAYER", "ML"]:
            if double:
                return self.create_2d_double_perovskite(
                    Bp=double, phase='monolayer', n=n, BX_dist=BX_dist, penet=penet,
                    Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz, wrap=wrap, output=output,
                    output_type=output_type, file_name=file_name
                )
            else:
                return self.create_2d_monolayer(
                    n=n, BX_dist=BX_dist, penet=penet, vacuum=vacuum, Ap_Rx=Ap_Rx,
                    Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz, wrap=wrap, output=output,
                    output_type=output_type, file_name=file_name, cache=cache
                )
        else:
            raise ValueError(f"Unknown structure type: {structure_type}. "
                           f"Use 'bulk', 'RP', 'DJ', or 'monolayer'")
    
    def create_comprehensive_series(self, n_range=[1, 2, 3, 4], phases=['DJ', 'RP', 'monolayer'],
                                  double_cations=None, penet=0.3, output_dir='.'):
        """
        Create a comprehensive series of all structure types and variations.
        
        Parameters
        ----------
        n_range : list
            Range of layer thicknesses to generate
        phases : list
            List of phases to generate ('DJ', 'RP', 'monolayer', 'bulk')
        double_cations : list, optional
            List of second B-site cations for double perovskites
        penet : float
            Penetration parameter
        output_dir : str
            Output directory
            
        Returns
        -------
        dict
            Dictionary of all generated structures organized by type
        """
        all_structures = {}
        
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE PEROVSKITE STRUCTURE GENERATION")
        print(f"Composition: {self.name}")
        print(f"B-X distance: {self.optimal_BX_dist:.3f} Å")
        print(f"{'='*60}")
        
        # Generate regular structures
        for phase in phases:
            if phase.upper() == 'BULK':
                print(f"\nGenerating BULK structure...")
                try:
                    structure = self.create_perovskite("bulk", output=True, 
                                                     file_name=f"{output_dir}/BULK_{self.name}")
                    all_structures['bulk'] = structure
                    print(f"✓ BULK: {len(structure)} atoms")
                except Exception as e:
                    print(f"✗ BULK: Failed - {e}")
            else:
                phase_structures = {}
                print(f"\nGenerating {phase.upper()} structures...")
                
                for n in n_range:
                    try:
                        structure = self.create_perovskite(
                            phase, n=n, penet=penet, output=True,
                            file_name=f"{output_dir}/{phase.upper()}_{self.name}_n{n}"
                        )
                        phase_structures[f'n{n}'] = structure
                        print(f"✓ {phase.upper()} n={n}: {len(structure)} atoms")
                    except Exception as e:
                        print(f"✗ {phase.upper()} n={n}: Failed - {e}")
                
                all_structures[phase.lower()] = phase_structures
        
        # Generate double perovskite structures if requested
        if double_cations:
            for bp_cation in double_cations:
                print(f"\nGenerating DOUBLE PEROVSKITE structures with B'={bp_cation}...")
                
                # Double bulk
                try:
                    structure = self.create_perovskite("bulk", double=bp_cation, output=True,
                                                     file_name=f"{output_dir}/DOUBLE_BULK_{self.name}_{bp_cation}")
                    all_structures[f'double_bulk_{bp_cation}'] = structure
                    print(f"✓ DOUBLE BULK {bp_cation}: {len(structure)} atoms")
                except Exception as e:
                    print(f"✗ DOUBLE BULK {bp_cation}: Failed - {e}")
                
                # Double 2D structures
                for phase in phases:
                    if phase.upper() != 'BULK':
                        double_phase_structures = {}
                        
                        for n in n_range:
                            try:
                                structure = self.create_perovskite(
                                    phase, n=n, double=bp_cation, penet=penet, output=True,
                                    file_name=f"{output_dir}/DOUBLE_{phase.upper()}_{self.name}_{bp_cation}_n{n}"
                                )
                                double_phase_structures[f'n{n}'] = structure
                                print(f"✓ DOUBLE {phase.upper()} {bp_cation} n={n}: {len(structure)} atoms")
                            except Exception as e:
                                print(f"✗ DOUBLE {phase.upper()} {bp_cation} n={n}: Failed - {e}")
                        
                        all_structures[f'double_{phase.lower()}_{bp_cation}'] = double_phase_structures
        
        print(f"\n{'='*60}")
        total_structures = sum(len(v) if isinstance(v, dict) else 1 for v in all_structures.values())
        print(f"GENERATION COMPLETE: {total_structures} structures created")
        print(f"{'='*60}")
        
        return all_structures 
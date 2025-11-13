"""
q2DStructure wrapper class for perovskite structures.

This module provides a q2DStructure class that wraps ase.Atoms to preserve
creation metadata while maintaining full ASE compatibility.
"""

from ase import Atoms
import numpy as np


class q2DStructure(Atoms):
    """
    Wrapper class for ASE Atoms that preserves perovskite creation metadata.
    
    This class inherits from ase.Atoms to ensure full compatibility with ASE
    functions like ase.io.write(). It stores metadata about how the structure
    was created (structure type, composition, BX distance, etc.).
    
    Attributes
    ----------
    structure_type : str, optional
        Type of structure: 'bulk', 'rp', 'dj', or 'monolayer'
    BX_dist : float, optional
        B-X bond distance in Angstroms
    A_ions : str, list, or Atoms, optional
        A-site cation(s) used in creation
    B_ions : str or list, optional
        B-site cation(s) used in creation
    X_ions : str or list, optional
        X-site anion(s) used in creation
    supercell_size : tuple, optional
        Supercell dimensions used in creation
    spacer_molecule : str, Atoms, or list, optional
        Spacer molecule(s) used for 2D structures
    """
    
    def __init__(self, atoms, structure_type=None, BX_dist=None,
                 A_ions=None, B_ions=None, X_ions=None, supercell_size=None,
                 spacer_molecule=None, **metadata):
        """
        Initialize q2DStructure with Atoms and metadata.
        
        Parameters
        ----------
        atoms : ase.Atoms
            The ASE Atoms object to wrap
        structure_type : str, optional
            Type of structure: 'bulk', 'rp', 'dj', or 'monolayer'
        BX_dist : float, optional
            B-X bond distance in Angstroms
        A_ions : str, list, or Atoms, optional
            A-site cation(s) used in creation
        B_ions : str or list, optional
            B-site cation(s) used in creation
        X_ions : str or list, optional
            X-site anion(s) used in creation
        supercell_size : tuple, optional
            Supercell dimensions used in creation
        spacer_molecule : str, Atoms, or list, optional
            Spacer molecule(s) used for 2D structures
        **metadata : dict
            Additional metadata to store
        """
        if not isinstance(atoms, Atoms):
            raise TypeError(f"atoms must be an ase.Atoms object, got {type(atoms)}")
        
        # Initialize as Atoms by copying all attributes
        # This makes q2DStructure a proper Atoms object
        super().__init__(
            symbols=atoms.get_chemical_symbols(),
            positions=atoms.get_positions(),
            cell=atoms.cell,
            pbc=atoms.pbc
        )
        
        # Copy any additional arrays and info
        if hasattr(atoms, 'arrays'):
            for key, value in atoms.arrays.items():
                if key not in ['numbers', 'positions']:  # Already set
                    self.arrays[key] = value.copy()
        
        if hasattr(atoms, 'info'):
            self.info.update(atoms.info)
        
        # Store metadata
        self.structure_type = structure_type
        self.BX_dist = BX_dist
        self.A_ions = A_ions
        self.B_ions = B_ions
        self.X_ions = X_ions
        self.supercell_size = supercell_size
        self.spacer_molecule = spacer_molecule
        
        # Store any additional metadata
        self._metadata = metadata
        for key, value in metadata.items():
            setattr(self, key, value)
    
    @property
    def atoms(self):
        """
        Access underlying ASE Atoms object (returns self since we inherit from Atoms).
        
        Returns
        -------
        ase.Atoms
            This q2DStructure object (which is an Atoms object)
        """
        return self
    
    def __getitem__(self, key):
        """
        Slicing returns new q2DStructure with sliced atoms.
        
        Parameters
        ----------
        key : int, slice, or array
            Index or slice to apply to atoms
            
        Returns
        -------
        q2DStructure or Atom
            New q2DStructure instance with sliced atoms (preserving metadata),
            or single Atom if key is an integer
        """
        # Handle boolean array indexing (like the test case)
        if isinstance(key, np.ndarray) and key.dtype == bool:
            # Boolean indexing - create new Atoms with selected atoms
            indices = np.where(key)[0]
            if len(indices) == 0:
                # Empty selection - return empty q2DStructure
                empty_atoms = Atoms()
                return q2DStructure(
                    empty_atoms,
                    structure_type=self.structure_type,
                    BX_dist=self.BX_dist,
                    A_ions=self.A_ions,
                    B_ions=self.B_ions,
                    X_ions=self.X_ions,
                    supercell_size=self.supercell_size,
                    spacer_molecule=self.spacer_molecule,
                    **self._metadata
                )
            # Create new Atoms with selected indices
            sliced_atoms = Atoms(
                symbols=[self.get_chemical_symbols()[i] for i in indices],
                positions=self.get_positions()[indices],
                cell=self.cell,
                pbc=self.pbc
            )
            # Copy arrays and info
            if hasattr(self, 'arrays'):
                for arr_key, arr_value in self.arrays.items():
                    if arr_key not in ['numbers', 'positions']:
                        sliced_atoms.arrays[arr_key] = arr_value[indices]
            if hasattr(self, 'info'):
                sliced_atoms.info.update(self.info)
            
            # Create new q2DStructure with sliced atoms, preserving all metadata
            return q2DStructure(
                sliced_atoms,
                structure_type=self.structure_type,
                BX_dist=self.BX_dist,
                A_ions=self.A_ions,
                B_ions=self.B_ions,
                X_ions=self.X_ions,
                supercell_size=self.supercell_size,
                spacer_molecule=self.spacer_molecule,
                **self._metadata
            )
        
        # For integer or slice indexing, use parent class but wrap result
        result = super().__getitem__(key)
        
        # If result is a single Atom (integer indexing), return as-is
        from ase import Atom
        if isinstance(result, Atom):
            return result
        
        # If result is Atoms (slice indexing), wrap in q2DStructure
        if isinstance(result, Atoms):
            return q2DStructure(
                result,
                structure_type=self.structure_type,
                BX_dist=self.BX_dist,
                A_ions=self.A_ions,
                B_ions=self.B_ions,
                X_ions=self.X_ions,
                supercell_size=self.supercell_size,
                spacer_molecule=self.spacer_molecule,
                **self._metadata
            )
        
        return result
    
    def copy(self):
        """
        Create a copy of this q2DStructure.
        
        Returns
        -------
        q2DStructure
            A new q2DStructure instance with copied atoms and metadata
        """
        # Create a new Atoms object from self
        atoms_copy = Atoms(
            symbols=self.get_chemical_symbols(),
            positions=self.get_positions(),
            cell=self.cell,
            pbc=self.pbc
        )
        
        # Copy arrays and info
        if hasattr(self, 'arrays'):
            for key, value in self.arrays.items():
                if key not in ['numbers', 'positions']:
                    atoms_copy.arrays[key] = value.copy()
        
        if hasattr(self, 'info'):
            atoms_copy.info.update(self.info)
        
        # Create new q2DStructure with metadata
        return q2DStructure(
            atoms_copy,
            structure_type=self.structure_type,
            BX_dist=self.BX_dist,
            A_ions=self.A_ions,
            B_ions=self.B_ions,
            X_ions=self.X_ions,
            supercell_size=self.supercell_size,
            spacer_molecule=self.spacer_molecule,
            **self._metadata
        )
    
    def interface(self, other_structure, vacuum=2.0, **kwargs):
        """
        Create an interface between this structure and another structure.
        
        This is a placeholder method that will be implemented later.
        It returns a new q2DStructure with the interface created.
        
        Parameters
        ----------
        other_structure : q2DStructure or ase.Atoms
            The other structure to create an interface with
        vacuum : float, optional
            Vacuum gap between structures in Angstroms (default: 2.0)
        **kwargs : dict
            Additional parameters for interface creation (to be defined)
            
        Returns
        -------
        q2DStructure
            New q2DStructure with the interface (placeholder implementation)
            
        Raises
        ------
        NotImplementedError
            This method is not yet implemented
        """
        raise NotImplementedError(
            "interface() method is not yet implemented. "
            "This will create an interface between two perovskite structures."
        )
    
    def twist(self, angle=0.0, axis='z', center=None, **kwargs):
        """
        Create a twisted version of this structure.
        
        This is a placeholder method that will be implemented later.
        It returns a new q2DStructure with the twist applied.
        
        Parameters
        ----------
        angle : float, optional
            Twist angle in degrees (default: 0.0)
        axis : str, optional
            Axis to rotate around: 'x', 'y', or 'z' (default: 'z')
        center : array-like, optional
            Center point for rotation (default: None, uses structure center)
        **kwargs : dict
            Additional parameters for twist creation (to be defined)
            
        Returns
        -------
        q2DStructure
            New q2DStructure with the twist applied (placeholder implementation)
            
        Raises
        ------
        NotImplementedError
            This method is not yet implemented
        """
        raise NotImplementedError(
            "twist() method is not yet implemented. "
            "This will create a twisted version of the perovskite structure."
        )

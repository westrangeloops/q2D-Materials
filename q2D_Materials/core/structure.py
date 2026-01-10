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
    xy_expansion : tuple, optional
        XY expansion factors used in creation
    spacer : str, Atoms, or list, optional
        Spacer(s) used for 2D structures (can be molecule or atomic cation)
    """
    
    def __init__(self, atoms, structure_type=None, BX_dist=None,
                 A_ions=None, B_ions=None, X_ions=None, xy_expansion=None,
                 spacer=None, spacer_molecule=None, **metadata):
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
        xy_expansion : tuple, optional
            XY expansion factors used in creation
        spacer : str, Atoms, or list, optional
            Spacer(s) used for 2D structures (can be molecule or atomic cation)
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
        self.xy_expansion = xy_expansion
        # Support both 'spacer' and 'spacer_molecule' for backward compatibility
        self.spacer = spacer if spacer is not None else spacer_molecule
        self.spacer_molecule = self.spacer  # Keep for backward compatibility
        
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
                    xy_expansion=self.xy_expansion,
                    spacer=self.spacer,
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
                xy_expansion=self.xy_expansion,
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
                xy_expansion=self.xy_expansion,
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
            xy_expansion=self.xy_expansion,
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
    
    def twist(self, m, n, interlayer_distance=11.0, vacuum=12.0, other=None, **kwargs):
        """
        Create a twisted bilayer structure from this monolayer.
        
        Based on the Quadratic Twisted Bilayer Generator by Gabriel Xavier Pereira:
        Institute of Physics, University of São Paulo, São Paulo, SP, Brazil
        Email: gxpereira@usp.br
        
        This method generates a twisted bilayer structure. If `other` is not provided,
        creates a self-twisted bilayer (same monolayer twisted against itself).
        The twist angle θ = arctan(2mn / (m² - n²)) creates a commensurate Moiré pattern.
        
        Parameters
        ----------
        m : int
            First integer parameter for twist angle calculation (m > n)
        n : int
            Second integer parameter for twist angle calculation (n > 0)
        interlayer_distance : float, optional
            Vertical distance between the two twisted layers in Angstroms (default: 11.0)
        vacuum : float, optional
            Total vacuum space (split evenly above/below) in Angstroms (default: 12.0)
        other : q2DStructure, optional
            Second monolayer structure. If None, uses self (self-twist).
        **kwargs : dict
            Additional parameters (reserved for future use)
            
        Returns
        -------
        q2DStructure
            New q2DStructure with the twisted bilayer
            
        Raises
        ------
        ValueError
            If structure_type is not 'monolayer'
        ImportError
            If pymatgen is not available
        """
        from q2D_Materials.utils.twist_monolayer import create_twisted_bilayer
        
        # Use self for both layers if other is not specified (self-twist)
        mono2 = other if other is not None else self
        return create_twisted_bilayer(self, mono2, m, n, interlayer_distance, vacuum, **kwargs)

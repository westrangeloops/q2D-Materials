"""q2DStructure wrapper class for perovskite structures."""

from ase import Atoms
import numpy as np
from typing import Optional, List, Union, Dict, Tuple


class q2DStructure(Atoms):
    """
    Wrapper class for ASE Atoms that preserves perovskite creation metadata.

    Inherits from ase.Atoms for full ASE compatibility while storing creation
    metadata (structure type, composition, BX distance, etc.).

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
        Spacer(s) used for 2D structures
    """

    def __init__(
        self,
        atoms: Atoms,
        structure_type: Optional[str] = None,
        BX_dist: Optional[float] = None,
        A_ions: Optional[Union[str, List[str]]] = None,
        B_ions: Optional[Union[str, List[str]]] = None,
        X_ions: Optional[Union[str, List[str]]] = None,
        xy_expansion: Optional[Tuple[int, int]] = None,
        spacer: Optional[Union[str, List[str]]] = None,
        spacer_molecule: Optional[Union[str, List[str]]] = None,
        **metadata: Dict,
    ) -> None:
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
            Spacer(s) used for 2D structures
        spacer_molecule : str, Atoms, or list, optional
            Alias for spacer (backward compatibility)
        **metadata : dict
            Additional metadata to store
        """
        if not isinstance(atoms, Atoms):
            raise TypeError(f"atoms must be an ase.Atoms object, got {type(atoms)}")

        super().__init__(
            symbols=atoms.get_chemical_symbols(),
            positions=atoms.get_positions(),
            cell=atoms.cell,
            pbc=atoms.pbc,
        )

        if hasattr(atoms, "arrays"):
            for key, value in atoms.arrays.items():
                if key not in ["numbers", "positions"]:
                    self.arrays[key] = value.copy()

        if hasattr(atoms, "info"):
            self.info.update(atoms.info)

        self.structure_type = structure_type
        self.BX_dist = BX_dist
        self.A_ions = A_ions
        self.B_ions = B_ions
        self.X_ions = X_ions
        self.xy_expansion = xy_expansion
        self.spacer = spacer if spacer is not None else spacer_molecule
        self.spacer_molecule = self.spacer

        self._metadata = metadata
        for key, value in metadata.items():
            setattr(self, key, value)

    @property
    def is_twister(self) -> bool:
        """True if this structure is a twisted multilayer stack."""
        return self.structure_type == 'twister'

    @property
    def n_layers_stacked(self) -> Optional[int]:
        """Number of stacked monolayer slabs (from twist metadata)."""
        if hasattr(self, '_metadata') and 'n_layers' in self._metadata:
            return self._metadata['n_layers']
        if self.is_twister:
            return 2
        return None

    @property
    def atoms(self) -> Atoms:
        """
        Access underlying ASE Atoms object.

        Returns
        -------
        ase.Atoms
            This q2DStructure object (which is an Atoms object)
        """
        return self

    def __getitem__(self, key: Union[int, slice, np.ndarray]) -> Union["q2DStructure", "Atom"]:
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
        from ase import Atom

        if isinstance(key, np.ndarray) and key.dtype == bool:
            indices = np.where(key)[0]
            if len(indices) == 0:
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
                    **self._metadata,
                )
            sliced_atoms = Atoms(
                symbols=[self.get_chemical_symbols()[i] for i in indices],
                positions=self.get_positions()[indices],
                cell=self.cell,
                pbc=self.pbc,
            )
            if hasattr(self, "arrays"):
                for arr_key, arr_value in self.arrays.items():
                    if arr_key not in ["numbers", "positions"]:
                        sliced_atoms.arrays[arr_key] = arr_value[indices]
            if hasattr(self, "info"):
                sliced_atoms.info.update(self.info)

            return q2DStructure(
                sliced_atoms,
                structure_type=self.structure_type,
                BX_dist=self.BX_dist,
                A_ions=self.A_ions,
                B_ions=self.B_ions,
                X_ions=self.X_ions,
                xy_expansion=self.xy_expansion,
                spacer_molecule=self.spacer_molecule,
                **self._metadata,
            )

        result = super().__getitem__(key)

        if isinstance(result, Atom):
            return result

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
                **self._metadata,
            )

        return result

    def copy(self) -> "q2DStructure":
        """
        Create a copy of this q2DStructure.

        Returns
        -------
        q2DStructure
            A new q2DStructure instance with copied atoms and metadata
        """
        atoms_copy = Atoms(
            symbols=self.get_chemical_symbols(),
            positions=self.get_positions(),
            cell=self.cell,
            pbc=self.pbc,
        )

        if hasattr(self, "arrays"):
            for key, value in self.arrays.items():
                if key not in ["numbers", "positions"]:
                    atoms_copy.arrays[key] = value.copy()

        if hasattr(self, "info"):
            atoms_copy.info.update(self.info)

        return q2DStructure(
            atoms_copy,
            structure_type=self.structure_type,
            BX_dist=self.BX_dist,
            A_ions=self.A_ions,
            B_ions=self.B_ions,
            X_ions=self.X_ions,
            xy_expansion=self.xy_expansion,
            spacer_molecule=self.spacer_molecule,
            **self._metadata,
        )

    def interface(
        self,
        other_structure: Union["q2DStructure", Atoms],
        vacuum: float = 2.0,
        **kwargs: Dict,
    ) -> "q2DStructure":
        """
        Create an interface between this structure and another structure.

        Parameters
        ----------
        other_structure : q2DStructure or ase.Atoms
            The other structure to create an interface with
        vacuum : float, optional
            Vacuum gap between structures in Angstroms (default: 2.0)
        **kwargs : dict
            Additional parameters for interface creation

        Returns
        -------
        q2DStructure
            New q2DStructure with the interface

        Raises
        ------
        NotImplementedError
            This method is not yet implemented
        """
        raise NotImplementedError(
            "interface() method is not yet implemented. "
            "This will create an interface between two perovskite structures."
        )

    def twist(
        self,
        m: int,
        n: int,
        interlayer_distance: float = 11.0,
        vacuum: float = 12.0,
        other: Optional["q2DStructure"] = None,
        **kwargs: Dict,
    ) -> "q2DStructure":
        """
        Create a twisted bilayer structure from this monolayer.

        If `other` is not provided, creates a self-twisted bilayer. The twist angle
        θ = arctan(2mn / (m² - n²)) creates a commensurate Moiré pattern.

        Parameters
        ----------
        m : int
            First integer parameter for twist angle calculation (m > n)
        n : int
            Second integer parameter for twist angle calculation (n > 0)
        interlayer_distance : float, optional
            Vertical distance between layers in Angstroms (default: 11.0)
        vacuum : float, optional
            Total vacuum space split evenly above/below in Angstroms (default: 12.0)
        other : q2DStructure, optional
            Second monolayer structure. If None, uses self (self-twist)
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
        from q2D_Materials.utils.other.twist_monolayer import create_twisted_bilayer

        mono2 = other if other is not None else self
        return create_twisted_bilayer(self, mono2, m, n, interlayer_distance, vacuum, **kwargs)

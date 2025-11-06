from ase.io import read, write
from ase.visualize import view
from ase import Atoms
from ..utils.smiles_to_3D import smiles_to_ase_atoms
from ..utils.perovskite_builder import (
    make_bulk, make_double, make_rp, make_dj, make_monolayer, 
    make_2d_double, make_mixed_bulk, auto_calculate_BX_distance
)

class q2D_creator:
    """
    q2D perovskite structure creator with ASE-based functionality.
    
    This class provides comprehensive perovskite building capabilities for creating
    various 2D and 3D perovskite structures.
    
    Parameters
    ----------
    B : str
        B-site cation symbol (e.g., 'Pb', 'Sn')
    X : str  
        X-site anion symbol (e.g., 'I', 'Br', 'Cl')
    A : str
        A-site cation symbol (e.g., 'Cs', 'MA', 'FA')
    name : str, optional
        Name identifier for the structure (default: 'structure')
        
    Notes
    -----
    For 2D structures (RP, DJ, monolayer), you must provide the organic spacer
    molecule when calling create_perovskite() via the `spacer_molecule` parameter.
    """
    
    def __init__(self, B, X, A, name='structure'):
        # Core composition - essential for all structures
        self.B = B
        self.X = X
        self.A = A
        self.name = name
        
        # Get A-site cation object
        from q2D_Materials.utils.common_a_sites import get_a_site_object
        self.A_cation = get_a_site_object(A)
        
        # Calculate optimal B-X distance from ionic radii
        self.optimal_BX_dist = auto_calculate_BX_distance(self.B, self.X)
    
    def _load_spacer_molecule(self, spacer_input):
        """
        Load spacer molecule from XYZ file, SMILES string, or ASE Atoms object.
        
        Parameters
        ----------
        spacer_input : str or ase.Atoms
            Either:
            - Path to XYZ file containing the organic molecule
            - SMILES string representation (e.g., 'CN' for methylamine)
            - ASE Atoms object directly
        
        Returns
        -------
        ase.Atoms
            Spacer molecule (alignment will be done by the builder functions)
        """
        # If already ASE Atoms, use directly
        if isinstance(spacer_input, Atoms):
            return spacer_input.copy()
        else:
            # Detect if input is SMILES string or file path
            is_smiles = self._is_smiles_string(spacer_input)
            
            if is_smiles:
                # Convert SMILES to ASE Atoms
                return smiles_to_ase_atoms(spacer_input)
            else:
                # Load from XYZ file
                return read(spacer_input)
    
    def _is_smiles_string(self, input_str):
        """
        Detect if input is a SMILES string or a file path.
        
        Simple heuristic: SMILES strings typically:
        - Don't contain '/' or '\' (file path separators)
        - Don't end with file extensions like '.xyz', '.mol', etc.
        - May contain parentheses, brackets, and chemical symbols
        """
        # Check for file path indicators
        if '/' in input_str or '\\' in input_str:
            return False
        
        # Check for common file extensions
        if input_str.endswith(('.xyz', '.mol', '.sdf', '.pdb', '.cif', '.vasp')):
            return False
        
        # Check if it looks like a file path (contains dots that might be extensions)
        # Simple heuristic: if it's a short string without path separators, likely SMILES
        # SMILES are usually < 200 characters and contain chemical symbols
        if len(input_str) > 200:
            return False
        
        # If it contains common SMILES characters and no file extensions, likely SMILES
        smiles_chars = ['(', ')', '[', ']', '=', '#', '@', '+', '-', 'c', 'n', 'o']
        if any(char in input_str for char in smiles_chars):
            return True
        
        # If it's a short alphanumeric string, assume it's SMILES (could be simple like "C" or "CC")
        if len(input_str) < 50 and not '.' in input_str:
            return True
        
        return False
    
    def view_structure(self, structure):
        """Visualize an ASE structure."""
        return view(structure)
    
    def write_structure(self, structure, filename=None, file_format='vasp'):
        """Write structure to file."""
        if filename is None:
            filename = f"{self.name}_{self.B}{self.X}3.{file_format}"
        elif not filename.endswith(f'.{file_format}'):
            filename += f'.{file_format}'
        
        write(filename, structure)
        print(f"Structure written to {filename}")
    
    def create_perovskite(self, structure_type="bulk", **kwargs):
        """
        Create perovskite structures of various types.
        
        Parameters
        ----------
        structure_type : str
            'bulk', 'RP', 'DJ', or 'monolayer'
        **kwargs : dict
            Structure-specific parameters:
            - spacer_molecule (str/Atoms): Required for 2D. XYZ path, SMILES, or Atoms
            - n (int): Layer thickness for 2D (default: 1)
            - Bp (str): Second B cation for double perovskites
            - BX_dist (float): B-X distance (auto-calculated if None)
            - penet (float): Penetration depth (default: 0.3)
            - spacer_distance (float): For RP - vacuum gap between spacers in Å (default: 2.0)
            - vacuum (float): For monolayer only (default: 12)
            - attachment_end (str): For DJ and monolayer - 'top', 'bottom', 'both'
              (DJ default: 'top', monolayer default: 'both')
            - Ap_Rx, Ap_Ry, Ap_Rz (float): Rotation angles in degrees
            - wrap (bool): Wrap atoms to cell (default: False)
            
        Returns
        -------
        ase.Atoms
            The created perovskite structure. Use write_structure() to save to file.
        """
        # Normalize structure type
        structure_type = structure_type.upper()
        
        # Validate structure type
        valid_types = ["BULK", "RP", "RUDDLESDEN-POPPER", "DJ", "DION-JACOBSON", "MONOLAYER", "ML"]
        if structure_type not in valid_types:
            raise ValueError(f"Unknown structure type: {structure_type}. "
                           f"Use 'bulk', 'RP', 'DJ', or 'monolayer'")
        
        # Route to structure-specific creator
        if structure_type == "BULK":
            return self._create_bulk(**kwargs)
        elif structure_type in ["RP", "RUDDLESDEN-POPPER"]:
            return self._create_rp(**kwargs)
        elif structure_type in ["DJ", "DION-JACOBSON"]:
            return self._create_dj(**kwargs)
        elif structure_type in ["MONOLAYER", "ML"]:
            return self._create_monolayer(**kwargs)
    
    def _create_bulk(self, Bp=None, BX_dist=None, **kwargs):
        """Create bulk (3D) perovskite structure."""
        BX_dist = BX_dist or self.optimal_BX_dist
        
        if Bp:
            return make_double(self.A_cation, self.B, Bp, self.X, BX_dist)
        else:
            return make_bulk(self.A_cation, self.B, self.X, BX_dist)
    
    def _create_rp(self, spacer_molecule, n=1, Bp=None, BX_dist=None, penet=0.3, 
                   spacer_distance=2.0, Ap_Rx=None, Ap_Ry=None, Ap_Rz=None, wrap=False, **kwargs):
        """Create Ruddlesden-Popper (RP) 2D perovskite structure."""
        spacer = self._load_spacer_molecule(spacer_molecule)
        BX_dist = BX_dist or self.optimal_BX_dist
        
        if Bp:
            return make_2d_double(spacer, self.A_cation, self.B, Bp, self.X, n, BX_dist, 
                                phase='rp', penet=penet, spacer_distance=spacer_distance,
                                Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz, wrap=wrap)
        else:
            return make_rp(spacer, self.A_cation, self.B, self.X, n, BX_dist, penet=penet,
                           spacer_distance=spacer_distance, Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, 
                           Ap_Rz=Ap_Rz, wrap=wrap)
    
    def _create_dj(self, spacer_molecule, n=1, Bp=None, BX_dist=None, penet=0.3,
                   attachment_end='top', Ap_Rx=None, Ap_Ry=None, Ap_Rz=None,
                   wrap=False, **kwargs):
        """Create Dion-Jacobson (DJ) 2D perovskite structure."""
        spacer = self._load_spacer_molecule(spacer_molecule)
        BX_dist = BX_dist or self.optimal_BX_dist
        
        if Bp:
            return make_2d_double(spacer, self.A_cation, self.B, Bp, self.X, n, BX_dist,
                                phase='dj', penet=penet, Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry,
                                Ap_Rz=Ap_Rz, wrap=wrap)
        else:
            return make_dj(spacer, self.A_cation, self.B, self.X, n, BX_dist,
                         penet=penet, Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz,
                         attachment_end=attachment_end, wrap=wrap)
    
    def _create_monolayer(self, spacer_molecule, n=1, Bp=None, BX_dist=None, penet=0.3,
                         vacuum=12, attachment_end='both', Ap_Rx=None, Ap_Ry=None, 
                         Ap_Rz=None, wrap=False, **kwargs):
        """Create 2D monolayer perovskite structure."""
        spacer = self._load_spacer_molecule(spacer_molecule)
        BX_dist = BX_dist or self.optimal_BX_dist
        
        if Bp:
            return make_2d_double(spacer, self.A_cation, self.B, Bp, self.X, n, BX_dist,
                                phase='monolayer', penet=penet, Ap_Rx=Ap_Rx,
                                Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz, wrap=wrap)
        else:
            return make_monolayer(spacer, self.A_cation, self.B, self.X, n, BX_dist,
                                penet=penet, vacuum=vacuum, attachment_end=attachment_end,
                                Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz, wrap=wrap)
    
    def create_mixed_bulk(self, A_ions=None, A_coefficients=None,
                         B_ions=None, B_coefficients=None,
                         X_ions=None, X_coefficients=None,
                         BX_dist=None, supercell_size=None,
                         distribution='random', seed=None):
        """
        Create a mixed bulk perovskite structure with multiple ions on A, B, or X sites.
        
        Parameters
        ----------
        A_ions : list[str/Atoms], optional
            List of A-site cations. If None, uses self.A as single ion.
        A_coefficients : list[float], optional
            Coefficients for A-site ions (must sum to 1.0). If None and A_ions provided,
            defaults to equal distribution.
        B_ions : list[str], optional
            List of B-site metal cations. If None, uses self.B as single ion.
        B_coefficients : list[float], optional
            Coefficients for B-site ions (must sum to 1.0). If None and B_ions provided,
            defaults to equal distribution.
        X_ions : list[str], optional
            List of X-site anions. If None, uses self.X as single ion.
        X_coefficients : list[float], optional
            Coefficients for X-site ions (must sum to 3.0 for ABX₃). If None and X_ions provided,
            defaults to equal distribution.
        BX_dist : float, optional
            B-X bond distance in Angstrom (auto-calculated if None).
        supercell_size : tuple[int, int, int], optional
            Supercell size (nx, ny, nz). If None, automatically calculated.
        distribution : str
            'random' or 'ordered' - how to distribute mixed ions (default: 'random').
        seed : int, optional
            Random seed for reproducible distributions.
            
        Returns
        -------
        ase.Atoms
            Mixed bulk perovskite structure.
            
        Examples
        --------
        >>> # Triple-cation perovskite (Cs₀.₀₅MA₀.₇₉FA₀.₁₈PbI₃)
        >>> q2d = q2D_creator(B='Pb', X='I', A='Cs')
        >>> perov = q2d.create_mixed_bulk(
        ...     A_ions=["Cs", "MA", "FA"],
        ...     A_coefficients=[0.05, 0.79, 0.18]
        ... )
        
        >>> # Mixed halides (MAPbBr₀.₅I₂.₅)
        >>> q2d = q2D_creator(B='Pb', X='I', A='MA')
        >>> perov = q2d.create_mixed_bulk(
        ...     X_ions=["Br", "I"],
        ...     X_coefficients=[0.5, 2.5]
        ... )
        
        >>> # Mixed B-site (MAPb₀.₅Sn₀.₅I₃)
        >>> perov = q2d.create_mixed_bulk(
        ...     B_ions=["Pb", "Sn"],
        ...     B_coefficients=[0.5, 0.5]
        ... )
        """
        # Import function to convert A-site cations
        from q2D_Materials.utils.common_a_sites import get_a_site_object
        
        # Use defaults from constructor if not specified
        if A_ions is None:
            A_ions = [self.A]
        if A_coefficients is None:
            A_coefficients = [1.0] if len(A_ions) == 1 else [1.0/len(A_ions)] * len(A_ions)
        
        # Validate that ions and coefficients lists have matching lengths
        if len(A_ions) != len(A_coefficients):
            raise ValueError(f"A_ions ({len(A_ions)} ions) and A_coefficients ({len(A_coefficients)} values) must have the same length")
        
        # Convert A-site cations: strings to Atoms objects if molecular
        processed_A_ions = []
        for a_ion in A_ions:
            if isinstance(a_ion, Atoms):
                # Already an Atoms object
                processed_A_ions.append(a_ion)
            elif isinstance(a_ion, str):
                # Convert string to appropriate object (string for atomic, Atoms for molecular)
                processed_A_ions.append(get_a_site_object(a_ion))
            else:
                processed_A_ions.append(a_ion)
        A_ions = processed_A_ions
        
        if B_ions is None:
            B_ions = [self.B]
        if B_coefficients is None:
            B_coefficients = [1.0] if len(B_ions) == 1 else [1.0/len(B_ions)] * len(B_ions)
        
        # Validate B-site lists match
        if len(B_ions) != len(B_coefficients):
            raise ValueError(f"B_ions ({len(B_ions)} ions) and B_coefficients ({len(B_coefficients)} values) must have the same length")
        
        if X_ions is None:
            X_ions = [self.X]
        if X_coefficients is None:
            if len(X_ions) == 1:
                X_coefficients = [3.0]
            else:
                # Distribute 3.0 equally among X ions
                X_coefficients = [3.0/len(X_ions)] * len(X_ions)
        
        # Validate X-site lists match
        if len(X_ions) != len(X_coefficients):
            raise ValueError(f"X_ions ({len(X_ions)} ions) and X_coefficients ({len(X_coefficients)} values) must have the same length")
        
        # Use optimal BX_dist if not provided
        # For mixed compositions, calculate average if multiple B-X pairs exist
        if BX_dist is None:
            if len(B_ions) == 1 and len(X_ions) == 1:
                BX_dist = self.optimal_BX_dist
            else:
                # Calculate average BX distance for mixed compositions
                from q2D_Materials.utils.common_a_sites import calculate_BX_distance
                BX_dists = []
                for B_ion in set(B_ions):
                    for X_ion in set(X_ions):
                        try:
                            dist = calculate_BX_distance(B_ion, X_ion)
                            BX_dists.append(dist)
                        except:
                            pass
                if BX_dists:
                    import numpy as np
                    BX_dist = float(np.mean(BX_dists))  # Ensure it's a Python float
                else:
                    BX_dist = self.optimal_BX_dist
        
        return make_mixed_bulk(
            A_ions=A_ions,
            A_coefficients=A_coefficients,
            B_ions=B_ions,
            B_coefficients=B_coefficients,
            X_ions=X_ions,
            X_coefficients=X_coefficients,
            BX_dist=BX_dist,
            supercell_size=supercell_size,
            distribution=distribution,
            seed=seed
        )
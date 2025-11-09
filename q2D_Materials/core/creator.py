from ase.io import read, write
from ase.visualize import view
from ase import Atoms
from ..utils.smiles_handler import smiles_to_ase_atoms
from ..utils.perovskite_builder import (
    create_bulk_perovskite, create_2d_perovskite, auto_calculate_BX_distance
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
        Load spacer molecule(s) from XYZ file, SMILES string, or ASE Atoms object.
        
        Parameters
        ----------
        spacer_input : str, ase.Atoms, or list
            Either:
            - Path to XYZ file containing the organic molecule
            - SMILES string representation (e.g., 'CN' for methylamine)
            - ASE Atoms object directly
            - List of any of the above for mixed spacers
        
        Returns
        -------
        ase.Atoms or list[ase.Atoms]
            Spacer molecule(s) (alignment will be done by the builder functions)
        """
        # Handle list of spacers
        if isinstance(spacer_input, list):
            return [self._load_spacer_molecule(sp) for sp in spacer_input]
        
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
            
            For mixed spacer compositions (2D structures):
            - Ap_coefficients (list): Coefficients for spacer molecules (must sum to 1.0).
              Mutually exclusive with spacer_pattern.
              Required if spacer_molecule is a list and spacer_pattern is None.
              Used for probability-based (random) selection.
            - spacer_pattern (str): Pattern for spacer assignment ('alternating', 'checkerboard', 'random').
              Mutually exclusive with Ap_coefficients.
              Only used for DJ/RP structures. For monolayer, always uses probability-based.
              If provided, uses deterministic pattern-based selection (equal weights for all spacers).
            - seed (int): Random seed for reproducible mixed spacer distributions
            
            For mixed bulk compositions:
            - A_ions (list): List of A-site cations (if list, requires A_coefficients)
            - A_coefficients (list): Coefficients for A-site ions (must sum to 1.0)
            - B_ions (list): List of B-site cations (if list, requires B_coefficients)
            - B_coefficients (list): Coefficients for B-site ions (must sum to 1.0)
            - X_ions (list): List of X-site anions (if list, requires X_coefficients)
            - X_coefficients (list): Coefficients for X-site ions (must sum to 3.0)
            - supercell_size (tuple): Supercell size for mixed (auto-calculated if None)
            - seed (int): Random seed for mixed compositions
            
        Returns
        -------
        ase.Atoms
            The created perovskite structure. Use write_structure() to save to file.
        """
        # Normalize structure type
        structure_type = structure_type.upper()
        
        # Map to internal structure type names
        structure_type_map = {
            "BULK": "bulk",
            "RP": "rp",
            "RUDDLESDEN-POPPER": "rp",
            "DJ": "dj",
            "DION-JACOBSON": "dj",
            "MONOLAYER": "monolayer",
            "ML": "monolayer"
        }
        
        if structure_type not in structure_type_map:
            raise ValueError(f"Unknown structure type: {structure_type}. "
                           f"Use 'bulk', 'RP', 'DJ', or 'monolayer'")
        
        internal_type = structure_type_map[structure_type]
        return self._create_structure(internal_type, **kwargs)
    
    def _create_structure(self, structure_type, **kwargs):
        """
        Unified method to create perovskite structures.
        
        Parameters
        ----------
        structure_type : str
            Internal structure type: 'bulk', 'rp', 'dj', or 'monolayer'
        **kwargs : dict
            Structure-specific parameters (see create_perovskite docstring)
            
        Returns
        -------
        Atoms
            The created perovskite structure
        """
        BX_dist = kwargs.get('BX_dist') or self.optimal_BX_dist
        Bp = kwargs.get('Bp')
        
        if structure_type == 'bulk':
            # Check if mixed composition is requested
            A_ions = kwargs.get('A_ions')
            B_ions = kwargs.get('B_ions')
            X_ions = kwargs.get('X_ions')
            
            # Detect if mixed (any list provided)
            is_mixed = (A_ions is not None and isinstance(A_ions, list)) or \
                      (B_ions is not None and isinstance(B_ions, list)) or \
                      (X_ions is not None and isinstance(X_ions, list))
            
            if is_mixed:
                # Mixed bulk perovskite
                from q2D_Materials.utils.common_a_sites import get_a_site_object
                
                # Use defaults from constructor if not specified
                if A_ions is None:
                    A_ions = [self.A]
                if B_ions is None:
                    B_ions = [self.B]
                if X_ions is None:
                    X_ions = [self.X]
                
                # Get coefficients (required for lists)
                A_coefficients = kwargs.get('A_coefficients')
                B_coefficients = kwargs.get('B_coefficients')
                X_coefficients = kwargs.get('X_coefficients')
                
                # Set defaults for coefficients if not provided
                if A_coefficients is None:
                    A_coefficients = [1.0] if len(A_ions) == 1 else [1.0/len(A_ions)] * len(A_ions)
                if B_coefficients is None:
                    B_coefficients = [1.0] if len(B_ions) == 1 else [1.0/len(B_ions)] * len(B_ions)
                if X_coefficients is None:
                    if len(X_ions) == 1:
                        X_coefficients = [3.0]
                    else:
                        X_coefficients = [3.0/len(X_ions)] * len(X_ions)
                
                # Convert A-site cations: strings to Atoms objects if molecular
                processed_A_ions = []
                for a_ion in A_ions:
                    if isinstance(a_ion, Atoms):
                        processed_A_ions.append(a_ion)
                    elif isinstance(a_ion, str):
                        processed_A_ions.append(get_a_site_object(a_ion))
                    else:
                        processed_A_ions.append(a_ion)
                A_ions = processed_A_ions
                
                # Calculate BX_dist for mixed if needed
                if BX_dist is None or BX_dist == self.optimal_BX_dist:
                    if len(B_ions) == 1 and len(X_ions) == 1:
                        BX_dist = self.optimal_BX_dist
                    else:
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
                            BX_dist = float(np.mean(BX_dists))
                        else:
                            BX_dist = self.optimal_BX_dist
                
                return create_bulk_perovskite(
                    A_ions, B_ions, X_ions,
                    BX_dist=BX_dist,
                    A_coefficients=A_coefficients,
                    B_coefficients=B_coefficients,
                    X_coefficients=X_coefficients,
                    double=(Bp is not None),
                    Bp=Bp,
                    supercell_size=kwargs.get('supercell_size'),
                    seed=kwargs.get('seed')
                )
            else:
                # Single bulk perovskite
                return create_bulk_perovskite(
                    self.A_cation, self.B, self.X, 
                    BX_dist=BX_dist, double=(Bp is not None), Bp=Bp
                )
        else:
            # 2D structures require spacer molecule
            if 'spacer_molecule' not in kwargs:
                raise ValueError(f"spacer_molecule is required for {structure_type} structures")
            
            spacer = self._load_spacer_molecule(kwargs['spacer_molecule'])
            n = kwargs.get('n', 1)
            penet = kwargs.get('penet', 0.3)
            wrap = kwargs.get('wrap', False)
            Ap_Rx = kwargs.get('Ap_Rx')
            Ap_Ry = kwargs.get('Ap_Ry')
            Ap_Rz = kwargs.get('Ap_Rz')
            
            # Handle double perovskite
            double = (Bp is not None)
            
            # Get mixed spacer parameters
            Ap_coefficients = kwargs.get('Ap_coefficients')
            spacer_pattern = kwargs.get('spacer_pattern')
            seed = kwargs.get('seed')
            
            # Structure-specific parameters
            if structure_type == 'rp':
                spacer_distance = kwargs.get('spacer_distance', 2.0)
                return create_2d_perovskite(
                    spacer, self.A_cation, self.B, self.X, n,
                    structure_type='rp', BX_dist=BX_dist, penet=penet,
                    spacer_distance=spacer_distance, Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry,
                    Ap_Rz=Ap_Rz, wrap=wrap, double=double, Bp=Bp,
                    Ap_coefficients=Ap_coefficients, spacer_pattern=spacer_pattern, seed=seed
                )
            elif structure_type == 'dj':
                attachment_end = kwargs.get('attachment_end', 'top')
                return create_2d_perovskite(
                    spacer, self.A_cation, self.B, self.X, n,
                    structure_type='dj', BX_dist=BX_dist, penet=penet,
                    attachment_end=attachment_end, Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry,
                    Ap_Rz=Ap_Rz, wrap=wrap, double=double, Bp=Bp,
                    Ap_coefficients=Ap_coefficients, spacer_pattern=spacer_pattern, seed=seed
                )
            elif structure_type == 'monolayer':
                vacuum = kwargs.get('vacuum', 12)
                attachment_end = kwargs.get('attachment_end', 'both')
                return create_2d_perovskite(
                    spacer, self.A_cation, self.B, self.X, n,
                    structure_type='monolayer', BX_dist=BX_dist, penet=penet,
                    vacuum=vacuum, attachment_end=attachment_end, Ap_Rx=Ap_Rx,
                    Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz, wrap=wrap, double=double, Bp=Bp,
                    Ap_coefficients=Ap_coefficients, spacer_pattern=spacer_pattern, seed=seed
                )
    
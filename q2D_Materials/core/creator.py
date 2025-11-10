from ase.io import read, write
from ase.visualize import view
from ase import Atoms
from ..utils.molecule_builder import smiles_to_ase_atoms
from ..utils.perovskite_builder import (
    create_perovskite, create_bulk_perovskite, create_2d_perovskite, auto_calculate_BX_distance
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
    
    def create_perovskite(self, structure_type="bulk", **kwargs):
        """
        Create perovskite structures of various types.
        
        Parameters
        ----------
        structure_type : str
            'bulk', 'RP', 'DJ', or 'monolayer'
        **kwargs : dict
            Structure-specific parameters:
            - spacer_molecule (str/Atoms/list): Required for 2D. XYZ path, SMILES, Atoms, or list for patterns
            - supercell (list): Required for 2D. [nx, ny, n_layers] where n_layers is the layer thickness
            - Bp (str): Second B cation for double perovskites
            - BX_dist (float): B-X distance (auto-calculated if None)
            - penet (float): Penetration depth (default: 0.3)
            - spacer_distance (float): For RP - vacuum gap between spacers in Å (default: 2.0)
            - vacuum (float): For monolayer only (default: 12)
            - attachment_end (str): For DJ and monolayer - 'top', 'bottom', 'both'
              (DJ default: 'top', monolayer default: 'both')
            - Ap_Rx, Ap_Ry, Ap_Rz (float): Rotation angles in degrees
            - wrap (bool): Wrap atoms to cell (default: False)
            
            For pattern-based mixed compositions (all structures):
            - A_ions (str/list): A-site cation(s). If list, assigned sequentially to positions.
            - B_ions (str/list): B-site cation(s). If list, assigned sequentially to positions.
            - X_ions (str/list): X-site anion(s). If list, assigned sequentially to positions.
            - supercell_size (tuple): For bulk only. (nx, ny, nz) supercell size.
            
            Note: For pattern-based assignment, provide lists of ions that will be cycled through
            to fill positions. The supercell must be large enough to accommodate your patterns.
            
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
        Unified method to create perovskite structures using the unified pipeline.
        
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
        double = (Bp is not None)
        
        # Prepare common parameters
        create_kwargs = {
            'BX_dist': BX_dist,
            'double': double,
            'Bp': Bp
        }
        
        if structure_type == 'bulk':
            # Check if pattern-based mixed composition is requested
            A_ions = kwargs.get('A_ions')
            B_ions = kwargs.get('B_ions')
            X_ions = kwargs.get('X_ions')
            supercell_size = kwargs.get('supercell_size')
            
            # Detect if mixed (any list provided)
            is_mixed = (A_ions is not None and isinstance(A_ions, list)) or \
                      (B_ions is not None and isinstance(B_ions, list)) or \
                      (X_ions is not None and isinstance(X_ions, list))
            
            if is_mixed:
                # Pattern-based mixed bulk perovskite
                # Require supercell_size for mixed compositions
                if supercell_size is None:
                    raise ValueError(
                        "supercell_size is required for mixed bulk perovskites.\n"
                        "Example: supercell_size=(2, 2, 2) for 2x2x2 supercell.\n"
                        "The supercell must be large enough to accommodate your ion patterns."
                    )
                
                # Use defaults from constructor if not specified
                if A_ions is None:
                    A_ions = self.A
                if B_ions is None:
                    B_ions = self.B
                if X_ions is None:
                    X_ions = self.X
                
                # Calculate BX_dist for mixed if needed
                if BX_dist is None or BX_dist == self.optimal_BX_dist:
                    # Use first ion for BX distance calculation
                    B_first = B_ions[0] if isinstance(B_ions, list) else B_ions
                    X_first = X_ions[0] if isinstance(X_ions, list) else X_ions
                    try:
                        from q2D_Materials.utils.common_a_sites import calculate_BX_distance
                        BX_dist = calculate_BX_distance(B_first, X_first)
                        create_kwargs['BX_dist'] = BX_dist
                    except:
                        pass
                
                create_kwargs.update({
                    'A': A_ions,
                    'B': B_ions,
                    'X': X_ions,
                    'supercell_size': supercell_size
                })
            else:
                # Single bulk perovskite (uniform composition)
                if supercell_size is None:
                    supercell_size = (1, 1, 1)  # Default to single unit cell
                
                create_kwargs.update({
                    'A': self.A_cation,
                    'B': self.B,
                    'X': self.X,
                    'supercell_size': supercell_size
                })
            
            return create_perovskite(structure_type='bulk', **create_kwargs)
        else:
            # 2D structures require spacer molecule
            if 'spacer_molecule' not in kwargs:
                raise ValueError(f"spacer_molecule is required for {structure_type} structures")
            
            spacer = self._load_spacer_molecule(kwargs['spacer_molecule'])
            supercell = kwargs.get('supercell')
            if supercell is None:
                raise ValueError(f"supercell is required for {structure_type} structures (e.g., supercell=[1, 1, 1])")
            
            # Get A/B/X ion patterns (pattern-based only)
            A_ions = kwargs.get('A_ions', self.A_cation)
            B_ions = kwargs.get('B_ions', self.B)
            X_ions = kwargs.get('X_ions', self.X)
            
            # Prepare 2D-specific parameters
            create_kwargs.update({
                'A': A_ions,
                'B': B_ions,
                'X': X_ions,
                'Ap': spacer,
                'supercell_size': supercell,
                'penet': kwargs.get('penet', 0.3),
                'wrap': kwargs.get('wrap', False),
                'Ap_Rx': kwargs.get('Ap_Rx'),
                'Ap_Ry': kwargs.get('Ap_Ry'),
                'Ap_Rz': kwargs.get('Ap_Rz')
            })
            
            # Structure-specific parameters
            if structure_type == 'rp':
                create_kwargs['spacer_distance'] = kwargs.get('spacer_distance', 2.0)
            elif structure_type == 'dj':
                create_kwargs['attachment_end'] = kwargs.get('attachment_end', 'top')
            elif structure_type == 'monolayer':
                create_kwargs['vacuum'] = kwargs.get('vacuum', 12)
                create_kwargs['attachment_end'] = kwargs.get('attachment_end', 'both')
            
            return create_perovskite(structure_type=structure_type, **create_kwargs)
    
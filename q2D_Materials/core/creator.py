from ase.io import read
from ase.visualize import view
from ase import Atoms
from ..utils.molecule_builder import smiles_to_ase_atoms
from ..utils.perovskite_builder import (
    create_perovskite, create_bulk_perovskite, create_2d_perovskite, auto_calculate_BX_distance
)
from ..utils.recomender import get_recommendations
from .structure import q2DStructure

class q2D_creator:
    """
    q2D perovskite structure creator with ASE-based functionality.
    
    This class provides comprehensive perovskite building capabilities for creating
    various 2D and 3D perovskite structures.
        
    Notes
    -----
    All composition parameters (A_ions, B_ions, X_ions) must be provided when calling
    create_perovskite(). For 2D structures (RP, DJ, monolayer), you must also provide
    the spacer via the `spacer` parameter (can be a molecule or atomic cation).
    """
    
    def __init__(self):
        """Initialize an empty q2D_creator instance."""
        pass
    
    def _load_spacer(self, spacer_input):
        """
        Load spacer molecule(s) from XYZ file, SMILES string, atomic cation, or ASE Atoms object.
        
        Parameters
        ----------
        spacer_input : str, ase.Atoms, or list
            Either:
            - Atomic cation string (e.g., 'Cs', 'K', 'Rb') - for RP structures
            - Path to XYZ file containing the organic molecule
            - SMILES string representation (e.g., 'CN' for methylamine)
            - ASE Atoms object directly
            - List of any of the above for mixed spacers
        
        Returns
        -------
        ase.Atoms or list[ase.Atoms]
            Spacer molecule(s) or atomic cation(s) (alignment will be done by the builder functions)
        """
        # Handle list of spacers
        if isinstance(spacer_input, list):
            return [self._load_spacer(sp) for sp in spacer_input]
        
        # If already ASE Atoms, use directly
        if isinstance(spacer_input, Atoms):
            return spacer_input.copy()
        else:
            # Check if it's an atomic cation (single letter or common atomic symbols)
            atomic_cations = ['Cs', 'K', 'Rb', 'Na', 'Li', 'Ca', 'Sr', 'Ba', 'Mg']
            if spacer_input in atomic_cations or (len(spacer_input) <= 2 and spacer_input[0].isupper() and not any(c in spacer_input for c in ['(', ')', '[', ']', '=', '#', '@'])):
                # Atomic cation - create single atom Atoms object
                return Atoms(spacer_input)
            
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
    
    def create_perovskite(self, structure_type="bulk", **kwargs):
        """
        Create perovskite structures of various types.
        
        Parameters
        ----------
        structure_type : str
            'bulk', 'RP', 'DJ', or 'monolayer'
        **kwargs : dict
            Structure-specific parameters:
            - spacer (str/Atoms/list): Required for 2D. Can be:
              - Atomic cation string (e.g., 'Cs', 'K', 'Rb') for all 2D structures
              - Molecule as XYZ path, SMILES, or Atoms object
              - List for pattern-based mixed spacers
            - supercell (list): Required for 2D. [nx, ny, n_layers] where n_layers is the layer thickness
            - Bp (str): Second B cation for double perovskites
            - BX_dist (float): B-X distance (auto-calculated if None)
            - penet (float): Penetration depth (default: 0.3)
            - spacer_distance (float): For RP - vacuum gap between spacers in Å (default: 2.0)
            - interlayer_penet (float): For RP - interlayer penetration as fraction of molecule length (default: 0.0)
            - vacuum (float): For monolayer only (default: 12)
            - attachment_end (str): For monolayer only - 'top', 'bottom', or 'both' (default: 'both')
            - Ap_Rx, Ap_Ry, Ap_Rz (float): Rotation angles in degrees
            
            Required composition parameters (all structures):
            - A_ions (str/list): A-site cation(s). Required. If list, assigned sequentially to positions.
            - B_ions (str/list): B-site cation(s). Required. If list, assigned sequentially to positions.
            - X_ions (str/list): X-site anion(s). Required. If list, assigned sequentially to positions.
            - supercell_size (tuple): For bulk only. (nx, ny, nz) supercell size.
            
            Note: For pattern-based assignment, provide lists of ions that will be cycled through
            to fill positions. The supercell must be large enough to accommodate your patterns.
            
        Returns
        -------
        q2DStructure
            The created perovskite structure as a q2DStructure object. This wraps
            ase.Atoms and preserves creation metadata. Use ASE's write functions to save to file.
            Example: from ase.io import write; write('structure.vasp', structure)
            For VASP with sorting: from ase.io.vasp import write_vasp; write_vasp('POSCAR', structure, sort=True, direct=True)
            Access underlying Atoms: structure.atoms
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
        q2DStructure
            The created perovskite structure as a q2DStructure object
        """
        # Require A_ions, B_ions, X_ions for all structures
        A_ions = kwargs.get('A_ions')
        B_ions = kwargs.get('B_ions')
        X_ions = kwargs.get('X_ions')
        
        if A_ions is None:
            raise ValueError("A_ions is required. Provide as string (e.g., 'MA') or list for pattern-based mixing.")
        if B_ions is None:
            raise ValueError("B_ions is required. Provide as string (e.g., 'Pb') or list for pattern-based mixing.")
        if X_ions is None:
            raise ValueError("X_ions is required. Provide as string (e.g., 'I') or list for pattern-based mixing.")
        
        # Calculate BX_dist on-demand from provided B/X ions
        BX_dist = kwargs.get('BX_dist')
        if BX_dist is None:
            # Get first ions (for pattern-based, use first; for single, use the value)
            B_first = B_ions[0] if isinstance(B_ions, list) else B_ions
            X_first = X_ions[0] if isinstance(X_ions, list) else X_ions
            BX_dist = auto_calculate_BX_distance(B_first, X_first)
        
        Bp = kwargs.get('Bp')
        double = (Bp is not None)
        
        # Prepare common parameters
        create_kwargs = {
            'BX_dist': BX_dist,
            'double': double,
            'Bp': Bp
        }
        
        # Store metadata for q2DStructure
        metadata = {
            'penet': kwargs.get('penet', 0.3),
            'double': double,
            'Bp': Bp
        }
        
        if structure_type == 'bulk':
            supercell_size = kwargs.get('supercell_size')
            
            # Detect if mixed (any list provided)
            is_mixed = isinstance(A_ions, list) or \
                      isinstance(B_ions, list) or \
                      isinstance(X_ions, list)
            
            if is_mixed:
                # Pattern-based mixed bulk perovskite
                # Require supercell_size for mixed compositions
                if supercell_size is None:
                    raise ValueError(
                        "supercell_size is required for mixed bulk perovskites.\n"
                        "Example: supercell_size=(2, 2, 2) for 2x2x2 supercell.\n"
                        "The supercell must be large enough to accommodate your ion patterns."
                    )
            else:
                # Single bulk perovskite (uniform composition)
                if supercell_size is None:
                    supercell_size = (1, 1, 1)  # Default to single unit cell
            
            create_kwargs.update({
                'A': A_ions,
                'B': B_ions,
                'X': X_ions,
                'supercell_size': supercell_size
            })
            
            atoms = create_perovskite(structure_type='bulk', **create_kwargs)
            
            # Wrap in q2DStructure with metadata
            return q2DStructure(
                atoms,
                structure_type='bulk',
                BX_dist=BX_dist,
                A_ions=A_ions,
                B_ions=B_ions,
                X_ions=X_ions,
                supercell_size=supercell_size,
                **metadata
            )
        else:
            # 2D structures require spacer
            # Support both 'spacer' and 'spacer_molecule' for backward compatibility
            if 'spacer' in kwargs:
                spacer_input = kwargs['spacer']
            elif 'spacer_molecule' in kwargs:
                spacer_input = kwargs['spacer_molecule']
            else:
                raise ValueError(f"spacer is required for {structure_type} structures")
            
            spacer = self._load_spacer(spacer_input)
            supercell = kwargs.get('supercell')
            if supercell is None:
                raise ValueError(f"supercell is required for {structure_type} structures (e.g., supercell=[1, 1, 1])")
            
            # Prepare 2D-specific parameters
            # attachment_end and wrap are automatically set based on structure_type
            create_kwargs.update({
                'A': A_ions,
                'B': B_ions,
                'X': X_ions,
                'Ap': spacer,
                'supercell_size': supercell,
                'penet': kwargs.get('penet', 0.3),
                'wrap': True,  # Always wrap for 2D structures
                'Ap_Rx': kwargs.get('Ap_Rx'),
                'Ap_Ry': kwargs.get('Ap_Ry'),
                'Ap_Rz': kwargs.get('Ap_Rz')
            })
            
            # Structure-specific parameters with automatic attachment_end
            if structure_type == 'rp':
                create_kwargs['spacer_distance'] = kwargs.get('spacer_distance', 2.0)
                create_kwargs['interlayer_penet'] = kwargs.get('interlayer_penet', 0.0)
                create_kwargs['attachment_end'] = 'both'  # RP always uses 'both'
                metadata['spacer_distance'] = kwargs.get('spacer_distance', 2.0)
                metadata['interlayer_penet'] = kwargs.get('interlayer_penet', 0.0)
            elif structure_type == 'dj':
                create_kwargs['attachment_end'] = 'top'  # DJ always uses 'top'
            elif structure_type == 'monolayer':
                create_kwargs['vacuum'] = kwargs.get('vacuum', 12)
                metadata['vacuum'] = kwargs.get('vacuum', 12)
                # Monolayer supports flexible attachment - use user-provided or default to 'both'
                if 'attachment_end' in kwargs:
                    create_kwargs['attachment_end'] = kwargs['attachment_end']
                    metadata['attachment_end'] = kwargs['attachment_end']
                else:
                    create_kwargs['attachment_end'] = 'both'  # Default for monolayer
                    metadata['attachment_end'] = 'both'
            
            # Store rotation angles if provided
            if kwargs.get('Ap_Rx') is not None:
                metadata['Ap_Rx'] = kwargs.get('Ap_Rx')
            if kwargs.get('Ap_Ry') is not None:
                metadata['Ap_Ry'] = kwargs.get('Ap_Ry')
            if kwargs.get('Ap_Rz') is not None:
                metadata['Ap_Rz'] = kwargs.get('Ap_Rz')
            
            atoms = create_perovskite(structure_type=structure_type, **create_kwargs)
            
            # Wrap in q2DStructure with metadata
            return q2DStructure(
                atoms,
                structure_type=structure_type,
                BX_dist=BX_dist,
                A_ions=A_ions,
                B_ions=B_ions,
                X_ions=X_ions,
                supercell_size=supercell,
                spacer=spacer_input,  # Store original input, not processed
                **metadata
            )
    
    def recommend(self, X=None, B=None, spacer=None, top_n=10, min_occurrences=1):
        """
        Recommend compatible ions for perovskite structures based on database occurrence.
        
        Given an X-site anion, B-site cation, or spacer molecule, this method recommends
        commonly used combinations based on occurrence frequency in the perovskite database.
        
        Parameters
        ----------
        X : str, optional
            X-site anion abbreviation (e.g., 'Cl', 'I', 'Br')
        B : str, optional
            B-site cation abbreviation (e.g., 'Pb', 'Sn')
        spacer : str, optional
            Spacer molecule abbreviation (e.g., 'PEA', 'BA')
            Note: Spacers are A-site ions with >10 atoms in their molecular formula
        top_n : int, default=10
            Number of recommendations to return per category
        min_occurrences : float, default=1
            Minimum occurrence count in database to include in recommendations
            
        Returns
        -------
        dict
            Dictionary with the following keys:
            - 'A': List of recommended A-site cations (dicts with abbrev, name, occurrences, etc.)
            - 'B': List of recommended B-site cations
            - 'X': List of recommended X-site anions
            - 'spacer': List of recommended spacers (A-site ions with >10 atoms)
            
            Each recommendation entry contains:
            - 'abbreviation': Ion abbreviation
            - 'common_name': Human-readable name
            - 'occurrences': Number of occurrences in database
            - 'molecular_formula': Molecular formula string
            - 'smile': SMILES string (if available)
            - 'ion_type': Ion type ('A', 'B', or 'X')
        
        Examples
        --------
        >>> creator = q2D_creator()
        >>> # Get recommendations based on X = Cl
        >>> recs = creator.recommend(X='Cl', top_n=5)
        >>> print(recs['B'])  # Top 5 B-site cations used with Cl
        >>> print(recs['A'])  # Top 5 A-site cations used with Cl
        >>> print(recs['spacer'])  # Top 5 spacers used with Cl
        
        >>> # Get recommendations based on B = Pb
        >>> recs = creator.recommend(B='Pb', top_n=10)
        
        >>> # Get recommendations based on spacer = PEA
        >>> recs = creator.recommend(spacer='PEA')
        """
        return get_recommendations(X=X, B=B, spacer=spacer, top_n=top_n, min_occurrences=min_occurrences)
    
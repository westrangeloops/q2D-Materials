"""
q2D perovskite structure creator.

This module provides the q2D_creator class which uses a modular pipeline:
  Template → Glazer → Slab → Spacers → Populate → Defects
"""

from ase.io import read
from ase.visualize import view
from ase import Atoms
import numpy as np

# New modular imports
from ..utils.template import (
    build_cubic_template,
    build_slab_template,
    extract_slab_from_bulk,
    SITE_ROLE_KEY,
    TEMPLATE_A,
    TEMPLATE_B,
    TEMPLATE_X,
)
from ..utils.populate import populate_structure
from ..utils.glazer_tilting import apply_glazer_tilting, glazer_notation_to_pattern
from ..utils.slab import (
    build_monolayer, build_dj, build_rp,
    add_spacers_to_monolayer, add_spacers_to_dj, add_spacers_to_rp, add_spacers_to_aci,
)
from ..utils.spacer import prepare_spacer
from ..utils.defects import add_vacancies, substitute_sites

from ..utils.molecule_builder import smiles_to_ase_atoms, add_atoms
from ..utils.common_a_sites import (
    calculate_BX_distance,
    _lookup_smiles_from_database,
    is_molecular_a_cation,
)
from ..utils.jagodzinski import generate_jagodzinski_structure
from ..utils.recomender import get_recommendations
from .structure import q2DStructure


class q2D_creator:
    """
    q2D perovskite structure creator using the modular pipeline.
    
    Pipeline:
      1. Build template (geometry only, placeholder elements)
      2. Apply Glazer tilting (on template - no element identification needed)
      3. Extract slab / build 2D phase
      4. Add spacers
      5. Populate with real elements (supports mixing)
      6. Apply defects
    """
    
    def __init__(self):
        """Initialize an empty q2D_creator instance."""
        pass
    
    def _is_element_symbol(self, s):
        """
        Check if string is a valid element symbol.
        
        Element symbols are:
        - Single uppercase: H, C, N, O, K, I, etc.
        - Uppercase + lowercase: Ca, Rb, Ti, Pb, etc.
        
        Database abbreviations are ALL UPPERCASE: MA, FA, PEA, HDA, etc.
        """
        if len(s) == 1:
            return s.isupper()  # H, C, N, O, K, I, etc.
        elif len(s) == 2:
            # Element: first upper, second lower (Ca, Rb, Ti)
            # Abbreviation: both upper (MA, FA)
            return s[0].isupper() and s[1].islower()
        else:
            # Longer strings are not element symbols
            return False
    
    def _load_spacer(self, spacer_input):
        """
        Load spacer molecule(s) from abbreviation, XYZ file, SMILES string, 
        atomic cation, or ASE Atoms object.
        
        Priority:
        1. ASE Atoms objects (pass through)
        2. Element symbols (Cs, Rb, K - upper+lower or single upper)
        3. Database abbreviations (MA, PEA, HDA - all uppercase)
        4. SMILES strings
        5. File paths
        """
        if isinstance(spacer_input, list):
            return [self._load_spacer(sp) for sp in spacer_input]
        
        if isinstance(spacer_input, Atoms):
            return spacer_input.copy()
        
        # Priority 1: Check if it's a valid element symbol (Cs, Rb, H, K, etc.)
        # Element symbols: single uppercase OR uppercase+lowercase
        if self._is_element_symbol(spacer_input):
            return Atoms(spacer_input)
        
        # Priority 2: Look up in A-ion database (MA, PEA, HDA - all uppercase)
        smiles = _lookup_smiles_from_database(spacer_input)
        if smiles is not None:
            # Check if it's an atomic ion SMILES like [Cs+], [K+]
            if smiles.strip().startswith('[') and smiles.strip().endswith(']'):
                inner = smiles.strip()[1:-1].strip()
                # Simple atomic ions: [Cs+], [K+], [Rb+], etc.
                if len(inner) <= 3 and ('+' in inner or '-' in inner):
                    # Extract element symbol
                    element = ''.join(c for c in inner if c.isalpha())
                    return Atoms(element)
            # It's a molecular spacer, create from SMILES
            return smiles_to_ase_atoms(smiles)
        
        # Priority 3: Try as SMILES string
        if self._is_smiles_string(spacer_input):
            return smiles_to_ase_atoms(spacer_input)
        
        # Priority 4: Try as file path
        return read(spacer_input)
    
    def _is_smiles_string(self, input_str):
        """Detect if input is a SMILES string or a file path."""
        if '/' in input_str or '\\' in input_str:
            return False
        if input_str.endswith(('.xyz', '.mol', '.sdf', '.pdb', '.cif', '.vasp')):
            return False
        if len(input_str) > 200:
            return False
        smiles_chars = ['(', ')', '[', ']', '=', '#', '@', '+', '-', 'c', 'n', 'o']
        if any(char in input_str for char in smiles_chars):
            return True
        if len(input_str) < 50 and '.' not in input_str:
            return True
        return False
    
    def view_structure(self, structure):
        """Visualize an ASE structure."""
        return view(structure)
    
    def create_perovskite(self, structure_type="bulk", **kwargs):
        """
        Create perovskite structures using the modular pipeline.
        
        Parameters
        ----------
        structure_type : str
            'bulk', 'RP', 'DJ', 'ACI', or 'monolayer'
        **kwargs : dict
            Structure-specific parameters:
            
            Required:
            - A_ions (str or list): A-site cation(s)
            - B_ions (str or list): B-site cation(s)
            - X_ions (str or list): X-site anion(s)
            
            Optional:
            - a0 (float): Lattice parameter (auto-calculated if None)
            - supercell_size (tuple): For bulk (nx, ny, nz)
            - supercell (list): For 2D [nx, ny, n_layers]
            - mixing_ratios (dict): {'A': [...], 'B': [...], 'X': [...]}
            
            2D parameters:
            - spacer (str/Atoms): Spacer molecule
            - vacuum (float): Vacuum for monolayer
            - interlayer_gap (float): Gap for RP/ACI
            - penetration (float): Spacer penetration distance
            
            Glazer tilting:
            - glazer_angles (list): [omega_x, omega_y, omega_z]
            - glazer_notation (str): e.g., "a-a-a-"
            
            Jagodzinski:
            - jagodzinski_sequence (str): Stacking sequence
            
            Defects:
            - vacancies (dict): {'site_type': str, 'fraction': float}
            - substitutions (list): [{'old': str, 'new': str, ...}]
        """
        structure_type_map = {
            "BULK": "bulk",
            "RP": "rp",
            "RUDDLESDEN-POPPER": "rp",
            "DJ": "dj",
            "DION-JACOBSON": "dj",
            "ACI": "aci",
            "MONOLAYER": "monolayer",
            "ML": "monolayer",
        }
        
        st_upper = structure_type.upper()
        if st_upper not in structure_type_map:
            raise ValueError(
                f"Unknown structure type: {structure_type}. "
                f"Use 'bulk', 'RP', 'DJ', 'ACI', or 'monolayer'"
            )
        
        internal_type = structure_type_map[st_upper]
        return self._create_structure(internal_type, **kwargs)
    
    def _create_structure(self, structure_type, **kwargs):
        """Build structure using the modular pipeline."""
        A_ions = kwargs.get('A_ions')
        B_ions = kwargs.get('B_ions')
        X_ions = kwargs.get('X_ions')
        
        if A_ions is None:
            raise ValueError("A_ions is required.")
        if B_ions is None:
            raise ValueError("B_ions is required.")
        if X_ions is None:
            raise ValueError("X_ions is required.")
        
        # Get first ion for lattice parameter calculation
        A = A_ions[0] if isinstance(A_ions, list) else A_ions
        B = B_ions[0] if isinstance(B_ions, list) else B_ions
        X = X_ions[0] if isinstance(X_ions, list) else X_ions
        
        # Calculate BX distance and lattice parameter
        a0 = kwargs.get('a0')
        if a0 is None:
            BX_dist = calculate_BX_distance(B, X)
            a0 = 2 * BX_dist
        else:
            BX_dist = a0 / 2.0
        
        # Get mixing ratios
        mixing_ratios = kwargs.get('mixing_ratios')
        seed = kwargs.get('seed')
        
        metadata = {
            'A_ions': A_ions,
            'B_ions': B_ions,
            'X_ions': X_ions,
            'a0': a0,
        }
        
        # Handle Jagodzinski stacking
        jagodzinski_sequence = kwargs.get('jagodzinski_sequence')
        if jagodzinski_sequence:
            elements = {'A': A, 'B': B, 'X': X}
            jag_a_cubic = kwargs.get('jagodzinski_a_cubic', 4.0)
            jag_result = generate_jagodzinski_structure(jagodzinski_sequence, elements, a_cubic=jag_a_cubic)
            atoms = jag_result.atoms
            metadata['jagodzinski_sequence'] = jag_result.layer_sequence
            
            return q2DStructure(
                atoms,
                structure_type='bulk',
                BX_dist=BX_dist,
                **metadata
            )
        
        # Get Glazer tilting parameters
        glazer_angles = kwargs.get('glazer_angles')
        glazer_notation = kwargs.get('glazer_notation')
        glazer_pattern = kwargs.get('glazer_pattern')
        has_glazer = glazer_angles is not None or glazer_notation is not None or glazer_pattern is not None
        
        if glazer_notation is not None:
            glazer_pattern = glazer_notation_to_pattern(glazer_notation)
        
        if structure_type == 'bulk':
            supercell = kwargs.get('supercell_size', (1, 1, 1))
            
            # Validate supercell for Glazer
            if has_glazer and (supercell[0] < 2 or supercell[1] < 2 or supercell[2] < 2):
                raise ValueError(
                    f"Glazer tilting requires at least 2x2x2 supercell. Got {supercell}."
                )
            
            # Build template
            template = build_cubic_template(BX_dist, supercell)
            
            # Apply Glazer tilting to template
            if has_glazer:
                template = apply_glazer_tilting(
                    template,
                    angles=glazer_angles,
                    tilt_pattern=glazer_pattern,
                    BX_dist=BX_dist,
                    default_angle_from_pattern=kwargs.get('glazer_default_angle', 2.0),
                    A_symbol=TEMPLATE_A,
                    B_symbol=TEMPLATE_B,
                    X_symbol=TEMPLATE_X,
                )
                metadata['glazer_angles'] = glazer_angles
                metadata['glazer_pattern'] = glazer_pattern
            
            # Populate with elements
            atoms = populate_structure(template, A_ions, B_ions, X_ions, mixing_ratios, seed)
            metadata['supercell_size'] = supercell
            
            return q2DStructure(
                atoms,
                structure_type='bulk',
                BX_dist=BX_dist,
                **metadata
            )
        
        else:
            # 2D structures: rp, dj, aci, monolayer
            # n_layers: number of octahedral layers per slab
            # supercell: [nx, ny] for XY supercell (z is ignored for 2D)
            n_layers = kwargs.get('n_layers', 1)
            supercell = kwargs.get('supercell', [1, 1])
            nx = supercell[0] if len(supercell) > 0 else 1
            ny = supercell[1] if len(supercell) > 1 else 1
            
            vacuum = kwargs.get('vacuum', 15.0)
            interlayer_gap = kwargs.get('interlayer_gap', 2.0)
            penetration = kwargs.get('penetration')
            
            # Validate supercell for Glazer
            if has_glazer and (nx < 2 or ny < 2):
                raise ValueError(
                    f"Glazer tilting requires at least 2x2 supercell in XY. Got supercell={supercell}."
                )
            
            # Load spacer
            spacer_input = kwargs.get('spacer')
            spacer = None
            if spacer_input is not None:
                spacer = self._load_spacer(spacer_input)
                metadata['spacer'] = spacer_input
            
            # Get reduced cell parameter
            reduced = kwargs.get('reduced', False)
            metadata['reduced_cell'] = reduced
            
            # Build template (with or without Glazer)
            if has_glazer:
                # Build bulk template, apply Glazer, then extract slab
                bulk_nz = max(n_layers, 2)
                bulk_template = build_cubic_template(BX_dist, (nx, ny, bulk_nz))
                
                tilted_bulk = apply_glazer_tilting(
                    bulk_template,
                    angles=glazer_angles,
                    tilt_pattern=glazer_pattern,
                    BX_dist=BX_dist,
                    default_angle_from_pattern=kwargs.get('glazer_default_angle', 2.0),
                    A_symbol=TEMPLATE_A,
                    B_symbol=TEMPLATE_B,
                    X_symbol=TEMPLATE_X,
                )
                
                slab_template = extract_slab_from_bulk(
                    tilted_bulk, n_layers, bulk_nz, include_surface_A=False, reduced=reduced
                )
                metadata['glazer_angles'] = glazer_angles
                metadata['glazer_pattern'] = glazer_pattern
            else:
                slab_template = build_slab_template(
                    BX_dist, n_layers, (nx, ny), include_surface_A=False, reduced=reduced
                )
            
            # Build 2D phase
            # Get spacer length for gap calculations
            spacer_len = 2.0  # default
            if spacer is not None:
                spacer_len = prepare_spacer(spacer, BX_dist)[0]
            
            if structure_type == 'monolayer':
                atoms = build_monolayer(slab_template, vacuum)
            elif structure_type == 'dj':
                atoms = build_dj(slab_template, spacer_len)
            elif structure_type == 'rp':
                is_atomic_spacer = spacer is not None and len(spacer) == 1
                # For atomic spacers: gap = BX_dist (A-site positions)
                # For molecular spacers: gap = spacer_length + interlayer_gap
                #   (molecules fill the space with interlayer_gap between their ends)
                if is_atomic_spacer:
                    rp_gap = BX_dist
                else:
                    rp_gap = spacer_len + interlayer_gap
                atoms = build_rp(slab_template, BX_dist, rp_gap, is_atomic_spacer)
            elif structure_type == 'aci':
                atoms = self._build_aci(slab_template, BX_dist, interlayer_gap)
            else:
                raise ValueError(f"Unknown 2D structure type: {structure_type}")
            
            # Add spacers using structure-specific functions
            if spacer is not None:
                if structure_type == 'monolayer':
                    atoms = add_spacers_to_monolayer(
                        atoms, spacer, BX_dist, nx, ny, penetration
                    )
                elif structure_type == 'dj':
                    atoms = add_spacers_to_dj(
                        atoms, spacer, BX_dist, nx, ny, penetration
                    )
                elif structure_type == 'rp':
                    atoms = add_spacers_to_rp(
                        atoms, spacer, BX_dist, nx, ny, penetration
                    )
                elif structure_type == 'aci':
                    atoms = add_spacers_to_aci(
                        atoms, spacer, BX_dist, nx, ny, penetration
                    )
            
            # Populate with elements
            atoms = populate_structure(atoms, A_ions, B_ions, X_ions, mixing_ratios, seed)
            
            # Apply defects
            vacancies = kwargs.get('vacancies')
            if vacancies:
                atoms = add_vacancies(
                    atoms,
                    vacancies.get('site_type', 'X_site'),
                    vacancies.get('fraction', 0.1),
                    vacancies.get('seed'),
                )
                metadata['vacancies'] = vacancies
            
            substitutions = kwargs.get('substitutions')
            if substitutions:
                for sub in substitutions:
                    atoms = substitute_sites(
                        atoms,
                        sub['old'],
                        sub['new'],
                        sub.get('site_type'),
                        sub.get('fraction', 1.0),
                        sub.get('seed'),
                    )
                metadata['substitutions'] = substitutions
            
            metadata['supercell'] = (nx, ny)
            metadata['n_layers'] = n_layers
            metadata['penetration'] = penetration
            if structure_type in ('rp', 'aci'):
                metadata['interlayer_gap'] = interlayer_gap
            if structure_type == 'monolayer':
                metadata['vacuum'] = vacuum
            
            return q2DStructure(
                atoms,
                structure_type=structure_type,
                BX_dist=BX_dist,
                **metadata
            )
    
    def _build_aci(self, slab_template, BX_dist, interlayer_gap):
        """
        Build ACI phase from slab template.
        
        ACI: Two slabs shifted by (0.5, 0) - alternating cation pattern.
        """
        from ..utils.template import apply_lateral_shift
        
        cell = slab_template.get_cell()
        cell_a = np.linalg.norm(cell[0])
        cell_b = np.linalg.norm(cell[1])
        
        # Normalize slab
        slab = slab_template.copy()
        positions = slab.get_positions()
        z_min = positions[:, 2].min()
        positions[:, 2] -= z_min
        slab.set_positions(positions)
        slab_height = positions[:, 2].max()
        
        # Create slab 2 with (0.5, 0) shift
        slab2 = apply_lateral_shift(slab.copy(), (0.5, 0.0), BX_dist)
        
        # Position slab 2 above slab 1
        positions2 = slab2.get_positions()
        positions2[:, 2] += slab_height + interlayer_gap
        slab2.set_positions(positions2)
        
        # Combine
        atoms = add_atoms(slab, slab2)
        
        # Set cell
        total_height = 2 * slab_height + 2 * interlayer_gap
        atoms.set_cell([cell_a, cell_b, total_height])
        atoms.pbc = True
        atoms.info['structure_type'] = 'aci'
        
        return atoms
    
    def recommend(self, X=None, B=None, spacer=None, top_n=10, min_occurrences=1):
        """
        Recommend compatible ions for perovskite structures.
        """
        return get_recommendations(X=X, B=B, spacer=spacer, top_n=top_n, min_occurrences=min_occurrences)

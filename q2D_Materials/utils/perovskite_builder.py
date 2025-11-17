"""
Module for creating perovskite structures.
"""

from ase.build import add_adsorbate, molecule, bulk
from ase.visualize import view
from ase import Atoms
import numpy as np
import ase.io
from .common_a_sites import (
    get_ionic_radius, ionic_radii, calculate_BX_distance
)


def _normalize_a_site(A):
    """
    Normalize A-site input to handle both strings and Atoms objects.
    Converts molecular cation strings (MA, FA, etc.) to Atoms objects.
    
    Parameters
    ----------
    A : str or Atoms
        A-site cation (string like "MA", "FA", "Cs" or Atoms object)
        
    Returns
    -------
    str or Atoms
        Atomic cations as strings, molecular cations as Atoms objects
    """
    if isinstance(A, Atoms):
        return A
    
    if isinstance(A, str):
        # Check if it's a molecular cation that needs conversion
        try:
            from .common_a_sites import get_a_site_object
            return get_a_site_object(A)
        except (ImportError, ValueError):
            # If conversion fails or not available, return as-is (atomic cation)
            return A
    
    return A


def auto_calculate_BX_distance(B, X):
    """
    Automatically calculate B-X bond distance from ionic radii data.
    
    Parameters
    ----------
    B : str
        B-site cation symbol
    X : str
        X-site anion symbol
        
    Returns
    -------
    float
        B-X bond distance in Angstroms
    """
    try:
        return calculate_BX_distance(B, X)
    except ValueError:
        # Fallback to default values if ions not in database
        return 2.0


def _get_effective_spacer_size(spacer):
    """
    Get effective size of spacer for layer separation calculations.
    
    For molecules: returns the molecule length along z-axis.
    For single atoms: returns 2 * ionic_radius to account for sphere diameter.
    
    Parameters
    ----------
    spacer : Atoms
        Spacer molecule or atom (ASE Atoms object)
        
    Returns
    -------
    float
        Effective spacer size in Angstroms
    """
    from .molecule_builder import get_molecule_length
    
    # Check if it's a single atom
    if len(spacer) == 1:
        # Single atom: use ionic radius
        symbol = spacer.get_chemical_symbols()[0]
        try:
            # Try to get ionic radius from A-site database
            ionic_rad = get_ionic_radius("A", symbol)
            # Return 2 * ionic_radius to account for sphere diameter
            return 2.0 * ionic_rad
        except (ValueError, KeyError):
            # If not found in A-site, try a reasonable default
            # Common atomic spacers: Cs (1.88), Rb (1.72), K (1.64)
            # Use 2.0 as fallback (reasonable for most atomic cations)
            return 2.0 * 1.8  # Default to ~3.6 Angstrom for unknown atoms
    else:
        # Molecule: use molecule length
        return get_molecule_length(spacer)


def create_perovskite(A, B, X, structure_type='bulk', supercell_size=(1, 1, 1), 
                     BX_dist=None, Ap=None, n_layers=1, double=False, Bp=None,
                     penet=0.3, vacuum=12, spacer_distance=2.0,
                     interlayer_penet=0.0, attachment_end=None, Ap_Rx=None, Ap_Ry=None, Ap_Rz=None,
                     wrap=None):
    """
    Unified function to create bulk or 2D perovskite structures.
    
    This is the main entry point for creating perovskite structures. It handles
    both bulk and 2D structures (RP, DJ, monolayer) using a unified pipeline.
    
    Parameters
    ----------
    A : str/Atoms or list[str/Atoms]
        A-site cation(s). Single value or list pattern.
    B : str or list[str]
        B-site cation(s). Single value or list pattern.
    X : str or list[str]
        X-site anion(s). Single value or list pattern.
    structure_type : str
        'bulk', 'rp', 'dj', or 'monolayer' (default: 'bulk').
    supercell_size : tuple[int, int, int]
        For bulk: (nx, ny, nz). For 2D: interpreted as [nx, ny, n_layers] where
        n_layers is the number of octahedral layers.
    BX_dist : float, optional
        B-X bond distance in Angstrom (auto-calculated if None).
    Ap : Atoms or list[Atoms], optional
        Spacer molecule(s) for 2D structures. Required for 2D, ignored for bulk.
    n_layers : int, optional
        Number of octahedral layers for 2D structures (default: 1).
        Overridden by supercell_size[2] if supercell_size is provided as [nx, ny, n_layers].
    double : bool, optional
        Whether to create double perovskite (default: False).
    Bp : str, optional
        Second B-site cation for double perovskite.
    penet : float, optional
        Penetration of spacer into inorganic layer for 2D (default: 0.3).
    vacuum : float, optional
        Vacuum for monolayer structures in Angstrom (default: 12).
    spacer_distance : float, optional
        Vacuum gap between opposing spacers for RP phase (default: 2.0).
    interlayer_penet : float, optional
        Interlayer penetration for RP phase, as a fraction of molecule length (default: 0.0).
        Controls interlocking of Ap cations in RP structures.
    attachment_end : str, optional
        Where to attach spacer for 2D. Automatically set based on structure_type:
        - RP: 'both' (always)
        - DJ: 'top' (always)
        - Monolayer: 'top', 'bottom', or 'both' (default: 'both')
        Users can override for monolayer structures.
    Ap_Rx, Ap_Ry, Ap_Rz : float, optional
        Rotation angles in degrees for spacers (applied as Rx->Ry->Rz).
    wrap : bool, optional
        Whether to wrap atoms to unit cell. Automatically set to True for all 2D structures.
        Users should not need to specify this.
        
    Returns
    -------
    Atoms
        The created perovskite structure.
    """
    # Normalize structure_type
    structure_type = structure_type.lower()
    
    if structure_type == 'bulk':
        # Use bulk creation logic
        return create_bulk_perovskite(A, B, X, supercell_size, BX_dist, double, Bp)
    elif structure_type in ['rp', 'dj', 'monolayer']:
        # Use 2D creation logic
        # Extract n_layers from supercell_size if it's a 3-element tuple/list
        if isinstance(supercell_size, (list, tuple)) and len(supercell_size) == 3:
            nx, ny, n_layers = supercell_size[0], supercell_size[1], supercell_size[2]
            supercell_2d = (nx, ny, n_layers)
        else:
            raise ValueError("For 2D structures, supercell_size must be [nx, ny, n_layers]")
        
        if Ap is None:
            raise ValueError(f"Ap (spacer molecule) is required for {structure_type} structures")
        
        return create_2d_perovskite(
            Ap=Ap, A=A, B=B, X=X, supercell=supercell_2d,
            structure_type=structure_type, BX_dist=BX_dist,
            penet=penet, vacuum=vacuum, spacer_distance=spacer_distance,
            interlayer_penet=interlayer_penet, attachment_end=attachment_end, 
            Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz,
            wrap=wrap, double=double, Bp=Bp
        )
    else:
        raise ValueError(f"Unknown structure_type: {structure_type}. "
                        f"Use 'bulk', 'rp', 'dj', or 'monolayer'")


def create_bulk_perovskite(A, B, X, supercell_size, BX_dist=None, double=False, Bp=None):
    """
    Create bulk perovskite structures using explicit ion patterns.

    Parameters
    ----------
    A : str/Atoms or list[str/Atoms]
        A-site cation(s). Single value for uniform sites, or list pattern for mixed.
        Example: 'Cs' or ['Cs', 'MA', 'FA', 'Cs', ...] for pattern-based assignment.
    B : str or list[str]
        B-site cation(s). Single value or list pattern.
    X : str or list[str]
        X-site anion(s). Single value or list pattern.
    supercell_size : tuple[int, int, int]
        Supercell dimensions (nx, ny, nz). Required for all structures.
    BX_dist : float, optional
        The desired BX bond distance (in Angstrom). If not provided,
        will be calculated automatically from ionic radii data.
    double : bool, optional
        Whether to create double perovskite (default: False).
    Bp : str, optional
        The atomic symbol of the B' cation (required if double=True).

    Returns
    -------
    Atoms
        The bulk perovskite structure as an ASE Atoms object.
        
    Notes
    -----
    Pattern mode: If lists are provided, ions are assigned sequentially to positions.
    The list will cycle if shorter than the number of positions.
    For random patterns, generate the pattern externally and pass as a list.
    """
    # Validate double perovskite parameters
    if double and Bp is None:
        raise ValueError("Bp (second B-site cation) is required for double perovskites")
    
    # Handle double perovskite by modifying B pattern
    if double:
        # For double perovskites, create alternating B/Bp pattern
        if not isinstance(B, list):
            if Bp != B:
                B = [B, Bp]  # Simple alternating pattern
    
    # Calculate BX_dist if not provided
    if BX_dist is None:
        # For patterns, use first ion
        B_first = B[0] if isinstance(B, list) else B
        X_first = X[0] if isinstance(X, list) else X
        BX_dist = auto_calculate_BX_distance(B_first, X_first)
    
    # Calculate lattice vectors
    lattice_vectors = _validate_and_convert_BX_dist(BX_dist)
    
    # Create structure using unified core
    structure = _create_unified_core(
        structure_type='bulk',
        lattice_vectors=lattice_vectors,
        A_ions=A,
        B_ions=B,
        X_ions=X,
        n=1,
        supercell_size=supercell_size
    )
    
    return structure




def _get_bulk_position_template():
    """
    Get position template for bulk perovskite unit cell.
    
    Returns the hardcoded fractional positions that define the perovskite geometry.
    These positions are the physical foundation of the perovskite structure.
        
    Returns
    -------
    dict
        Dictionary with keys 'A', 'B', 'X' containing lists of fractional positions.
        Each position is [x, y, z] in fractional coordinates.
    """
    return {
        'A': [[0.0, 0.0, 0.0]],
        'B': [[0.5, 0.5, 0.5]],
        'X': [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
    }


def _get_2d_layer_position_template(n):
    """
    Get position template for 2D perovskite layer.
    
    Returns the hardcoded fractional positions that define the 2D perovskite geometry.
    These positions preserve the physical structure of the octahedral layers.
    Based on the positions from _make_2d_layer function.
    
    Parameters
    ----------
    n : int
        Number of octahedral layers
        
    Returns
    -------
    dict
        Dictionary with keys 'X', 'B', 'A' containing lists of fractional positions.
        Positions are in fractional coordinates relative to unit cell.
        'X' includes initial X atoms and all X atoms in layers
        'B' includes all B-site positions
        'A' includes A-site positions between layers (if n > 1)
    """
    # Initial X atoms at z=0 (from _make_2d_layer)
    positions = {
        'X': [[0.25, 0.25, 0.0], [0.75, 0.75, 0.0]],
        'B': [],
        'A': []
    }
    
    # Base positions for each layer (8 atoms per layer)
    # Pattern from _make_2d_layer: [X, X, B, X, X, B, X, X]
    # Positions are: [0, 0, .5], [.5, 0, .5], [.25, .25, .5],
    #                [0, .5, .5], [.5, .5, .5], [.75, .75, .5],
    #                [.25, .25, 1], [.75, .75, 1]
    base_positions = [
        [0, 0, 0.5],        # X (index 0)
        [0.5, 0, 0.5],      # X (index 1)
        [0.25, 0.25, 0.5],  # B (index 2)
        [0, 0.5, 0.5],      # X (index 3)
        [0.5, 0.5, 0.5],    # X (index 4)
        [0.75, 0.75, 0.5],  # B (index 5)
        [0.25, 0.25, 1.0],  # X (index 6)
        [0.75, 0.75, 1.0]   # X (index 7)
    ]
    
    # Add positions for each layer
    for i in range(n):
        for j, base_pos in enumerate(base_positions):
            pos = [base_pos[0], base_pos[1], base_pos[2] + i]
            if j in [2, 5]:  # B-site positions (indices 2 and 5)
                positions['B'].append(pos)
            else:  # X-site positions
                positions['X'].append(pos)
    
    # A-site positions between layers (if n > 1)
    # From _make_2d_layer: positions are calculated as:
    # z_pos = lv2 + lv2 * i (in Angstroms) where i ranges from 0 to n-2
    # In fractional coordinates (before scaling): z = 1.0 + i
    # There are 2 A-site positions per unit cell per A-site layer:
    # - [0.25, 0.75, z] for spacers (will be filtered out later, added by _attach_spacer)
    # - [0.75, 0.25, z] for A-site cations
    # CRITICAL: We generate both positions in the template, then during supercell expansion:
    #   - A-site positions [0.75, 0.25] are expanded to ALL unit cells (nx × ny)
    #   - Spacer positions [0.25, 0.75] are filtered out (added separately by _attach_spacer)
    # This ensures each unit cell gets BOTH an A-site AND a spacer at the same (x,y) but different z
    if n > 1:
        for i in range(n - 1):
            z_pos = 1.0 + i  # Fractional z position (will be scaled by lv2)
            # Generate both positions - they will be expanded to all unit cells
            positions['A'].append([0.25, 0.75, z_pos])  # Spacer position (will be filtered during expansion)
            positions['A'].append([0.75, 0.25, z_pos])  # A-site cation position (will be kept)
    
    return positions


def _translate_positions_to_supercell(position_template, supercell_size, lattice_vectors):
    """
    Translate position template to create supercell positions.
    
    Takes base unit cell positions and translates them by (ix, iy, iz) 
    for each unit cell in the supercell.
    
    Parameters
    ----------
    position_template : dict
        Dictionary with site types as keys and lists of fractional positions as values
    supercell_size : tuple[int, int, int]
        Supercell dimensions (nx, ny, nz)
    lattice_vectors : np.ndarray
        Lattice vector sizes [a, b, c]
        
    Returns
    -------
    dict
        Dictionary with same structure as position_template but with all supercell positions.
        Positions are in Angstroms (scaled by lattice vectors).
    """
    nx, ny, nz = supercell_size
    lv0, lv1, lv2 = lattice_vectors[0], lattice_vectors[1], lattice_vectors[2]
    
    supercell_positions = {}
    
    for site_type, positions in position_template.items():
        supercell_positions[site_type] = []
        
        for pos_frac in positions:
            # Translate to each unit cell in supercell
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        # Fractional position in unit cell (ix, iy, iz)
                        # Add integer translation, then convert to Angstroms
                        new_pos_frac = [
                            pos_frac[0] + ix,
                            pos_frac[1] + iy,
                            pos_frac[2] + iz
                        ]
                        # Convert to Angstroms (scale by lattice vectors)
                        new_pos_ang = [
                            new_pos_frac[0] * lv0,
                            new_pos_frac[1] * lv1,
                            new_pos_frac[2] * lv2
                        ]
                        supercell_positions[site_type].append(new_pos_ang)
    
    return supercell_positions


def _assign_ions_to_positions(position_template, A_ions, B_ions, X_ions):
    """
    Assign ions to positions using explicit patterns.
    
    This function assigns ions to positions based on explicit patterns provided by the user.
    If a single ion is provided, it's used for all positions of that type.
    If a list is provided, ions are assigned sequentially, cycling if the list is shorter.
    
    Parameters
    ----------
    position_template : dict
        Dictionary with site types ('A', 'B', 'X') and lists of positions
    A_ions : str/Atoms or list[str/Atoms]
        A-site ion(s). Single value or list pattern.
    B_ions : str or list[str]
        B-site ion(s). Single value or list pattern.
    X_ions : str or list[str]
        X-site ion(s). Single value or list pattern.
        
    Returns
    -------
    list[tuple]
        List of (site_type, ion, position) tuples where ion is str or Atoms object
    """
    assignments = []
    
    # Assign A-site positions
    A_positions = position_template.get('A', [])
    if A_positions:
        if isinstance(A_ions, list):
            # Pattern mode: cycle through list
            for i, pos in enumerate(A_positions):
                ion = A_ions[i % len(A_ions)]
                # Copy Atoms objects to avoid sharing references
                if isinstance(ion, Atoms):
                    ion = ion.copy()
                assignments.append(('A', ion, pos))
        else:
            # Single ion: use for all positions
            for pos in A_positions:
                ion = A_ions
                if isinstance(ion, Atoms):
                    ion = ion.copy()
                assignments.append(('A', ion, pos))
    
    # Assign B-site positions
    B_positions = position_template.get('B', [])
    if B_positions:
        if isinstance(B_ions, list):
            for i, pos in enumerate(B_positions):
                ion = B_ions[i % len(B_ions)]
                if isinstance(ion, Atoms):
                    ion = ion.copy()
                assignments.append(('B', ion, pos))
        else:
            for pos in B_positions:
                ion = B_ions
                if isinstance(ion, Atoms):
                    ion = ion.copy()
                assignments.append(('B', ion, pos))
    
    # Assign X-site positions
    X_positions = position_template.get('X', [])
    if X_positions:
        if isinstance(X_ions, list):
            for i, pos in enumerate(X_positions):
                ion = X_ions[i % len(X_ions)]
                if isinstance(ion, Atoms):
                    ion = ion.copy()
                assignments.append(('X', ion, pos))
        else:
            for pos in X_positions:
                ion = X_ions
                if isinstance(ion, Atoms):
                    ion = ion.copy()
                assignments.append(('X', ion, pos))
    
    return assignments


def _validate_and_convert_BX_dist(BX_dist):
    """
    Validate and convert BX_dist to float, calculating cell size.
    
    Parameters
    ----------
    BX_dist : float or int
        B-X bond distance in Angstrom
        
    Returns
    -------
    np.ndarray
        Cell dimensions [a, b, c] for cubic perovskite (2*BX_dist each)
    """
    try:
        BX_dist_float = float(BX_dist)
        return 2 * np.array([BX_dist_float, BX_dist_float, BX_dist_float])
    except (TypeError, ValueError):
        # Fallback: use default cell size
        return 2 * np.array([3.0, 3.0, 3.0])


def _create_unified_core(structure_type, lattice_vectors, A_ions, B_ions, X_ions, 
                        n=1, supercell_size=(1, 1, 1)):
    """
    Unified core that creates position matrix then populates with ions using patterns.
    
    This is the fundamental building block that all perovskite creation functions
    should use. It creates structures by:
    1. Getting position template for structure_type
    2. Expanding to supercell by translating positions
    3. Assigning ions to each position based on explicit patterns
    4. Creating Atoms object with all positions and symbols
    
    Parameters
    ----------
    structure_type : str
        'bulk' or '2d_layer' - type of structure to create
    lattice_vectors : np.ndarray
        Lattice vector sizes [a, b, c]
    A_ions : str/Atoms or list[str/Atoms]
        A-site cation(s). Single value or explicit pattern.
    B_ions : str or list[str]
        B-site cation(s). Single value or explicit pattern.
    X_ions : str or list[str]
        X-site anion(s). Single value or explicit pattern.
    n : int
        For 2D layers: number of octahedral layers
    supercell_size : tuple[int, int, int]
        Supercell dimensions (nx, ny, nz)
        
    Returns
    -------
    Atoms
        The perovskite structure with all sites populated
    """
    # Normalize A-site ions (convert molecular strings to Atoms objects)
    if isinstance(A_ions, list):
        A_ions_normalized = []
        for A in A_ions:
            if isinstance(A, Atoms):
                A_ions_normalized.append(A)
            elif isinstance(A, str):
                A_ions_normalized.append(_normalize_a_site(A))
            else:
                A_ions_normalized.append(_normalize_a_site(A))
        A_ions = A_ions_normalized
    else:
        # Single value
        if isinstance(A_ions, str):
            A_ions = _normalize_a_site(A_ions)
        # else: already Atoms object
    
    # Get position template for one unit cell
    if structure_type == 'bulk':
        template = _get_bulk_position_template()
    elif structure_type == '2d_layer':
        template = _get_2d_layer_position_template(n)
    else:
        raise ValueError(f"Unknown structure_type: {structure_type}")
    
    # Expand to supercell
    if supercell_size == (1, 1, 1):
        # Single unit cell - convert fractional to Angstroms
        supercell_template = {}
        for site_type, positions in template.items():
            supercell_template[site_type] = []
            for pos_frac in positions:
                pos_ang = [
                    pos_frac[0] * lattice_vectors[0],
                    pos_frac[1] * lattice_vectors[1],
                    pos_frac[2] * lattice_vectors[2]
                ]
                supercell_template[site_type].append(pos_ang)
    else:
        # Supercell - translate positions
        # IMPORTANT: For 2D structures, supercell_size[2] is n_layers, NOT a z-expansion
        # We should only expand in x and y, not in z
        if structure_type == '2d_layer':
            # For 2D: only expand in x and y, keep z as is
            nx, ny, n_layers = supercell_size
            supercell_template = {}
            for site_type, positions in template.items():
                supercell_template[site_type] = []
                for pos_frac in positions:
                    # For A-site positions, we need to track which are A-sites vs spacers
                    # A-sites are at [0.75, 0.25, z], spacers are at [0.25, 0.75, z]
                    # Check if this is an A-site position (before expansion)
                    # Use exact comparison for 0.25 and 0.75 since they're hardcoded
                    x_frac = pos_frac[0]
                    y_frac = pos_frac[1]
                    is_a_site_pos = (abs(x_frac - 0.75) < 0.001) and (abs(y_frac - 0.25) < 0.001)
                    is_spacer_pos = (abs(x_frac - 0.25) < 0.001) and (abs(y_frac - 0.75) < 0.001)
                    
                    # Expand only in x and y
                    # CRITICAL: For EACH unit cell, we need BOTH positions [0.75, 0.25] AND [0.25, 0.75]
                    # So for a 2x2 supercell: 4 unit cells × 2 positions = 8 A-site positions
                    # Translation: new_x = base_x + ix, new_y = base_y + iy
                    for ix in range(nx):
                        for iy in range(ny):
                            # Translate fractional coordinates by unit cell indices
                            # new_pos_frac = [base_x + ix, base_y + iy, z]
                            # Examples for 2x2:
                            #   Unit cell (0,0): [0.75+0, 0.25+0] = [0.75, 0.25] and [0.25+0, 0.75+0] = [0.25, 0.75]
                            #   Unit cell (1,0): [0.75+1, 0.25+0] = [1.75, 0.25] and [0.25+1, 0.75+0] = [1.25, 0.75]
                            #   Unit cell (0,1): [0.75+0, 0.25+1] = [0.75, 1.25] and [0.25+0, 0.75+1] = [0.25, 1.75]
                            #   Unit cell (1,1): [0.75+1, 0.25+1] = [1.75, 1.25] and [0.25+1, 0.75+1] = [1.25, 1.75]
                            new_pos_frac = [
                                pos_frac[0] + ix,  # Translate x by unit cell index
                                pos_frac[1] + iy,  # Translate y by unit cell index
                                pos_frac[2]  # Keep original z coordinate (don't expand in z)
                            ]
                            # Convert to Angstroms by scaling with lattice vectors
                            new_pos_ang = [
                                new_pos_frac[0] * lattice_vectors[0],
                                new_pos_frac[1] * lattice_vectors[1],
                                new_pos_frac[2] * lattice_vectors[2]
                            ]
                            
                            # For A-site positions: add BOTH [0.75, 0.25] AND [0.25, 0.75] as A-sites
                            # This ensures each unit cell has both positions
                            # Spacers will also be added at [0.25, 0.75] by _attach_spacer (at different z-levels)
                            if site_type == 'A':
                                # Add BOTH positions - this gives us 2 positions per unit cell
                                # For 2x2: 4 unit cells × 2 positions = 8 A-site positions
                                supercell_template[site_type].append(new_pos_ang)
                            else:
                                # For X and B sites, add all positions
                                supercell_template[site_type].append(new_pos_ang)
        else:
            # For bulk: expand in all dimensions
            supercell_template = _translate_positions_to_supercell(
                template, supercell_size, lattice_vectors
            )
    
    # Note: For 2D structures, spacer positions [0.25, 0.75, z] are filtered out during
    # supercell expansion above. Only A-site positions [0.75, 0.25, z] are kept.
    # Spacers are added separately by _attach_spacer function.
    
    # Validate A-site positions were generated correctly
    if structure_type == '2d_layer' and supercell_size:
        nx, ny, n_layers = supercell_size
        if 'A' in supercell_template:
            num_A_positions = len(supercell_template['A'])
            if n_layers > 1:
                # Expected A-sites: (n_layers - 1) A-site layers × 2 positions per unit cell × nx × ny unit cells
                # For n=2, 2x2 supercell: (2-1) × 2 × 2 × 2 = 8 A-sites
                # Each unit cell has 2 positions: [0.75, 0.25] and [0.25, 0.75]
                expected_A = (n_layers - 1) * 2 * nx * ny
                if num_A_positions != expected_A:
                    raise ValueError(f"A-site position generation failed: expected {expected_A}, got {num_A_positions}")
    
    # Assign ions to positions using patterns
    
    assignments = _assign_ions_to_positions(
        supercell_template, 
        A_ions, B_ions, X_ions
    )
    
    # Separate atomic and molecular ions
    # IMPORTANT: For A-sites, we need to handle atomic and molecular separately
    # to avoid duplication. Only atomic A-sites go in the main structure,
    # molecular A-sites are added separately with special positioning.
    atomic_symbols = []
    atomic_positions = []
    molecular_atoms = []  # List of (Atoms, position) tuples
    
    for site_type, ion, pos in assignments:
        if isinstance(ion, Atoms):
            # Molecular ion (A-site only) - handle separately
            # Note: ion is already a copy from _assign_ions_to_positions, but we copy again
            # when aligning to be extra safe
            # IMPORTANT: Also copy the position list to avoid reference issues
            pos_copy = [pos[0], pos[1], pos[2]]
            molecular_atoms.append((ion, pos_copy, site_type))
        elif site_type == 'A':
            # A-site: check if it's a molecular cation that failed to convert
            # If it's a string that should be molecular, try to convert again
            if isinstance(ion, str):
                try:
                    from .common_a_sites import is_molecular_a_cation, get_a_site_object
                    if is_molecular_a_cation(ion):
                        # It's a molecular cation - convert it
                        mol_ion = get_a_site_object(ion)
                        molecular_atoms.append((mol_ion, pos, site_type))
                        continue
                except (ImportError, ValueError) as e:
                    # If conversion fails, treat as atomic
                    pass
            # Atomic A-site (or failed molecular conversion)
            # For bulk structures, atomic A-sites need a z-offset to be at the correct position
            # The z-offset is -1.0 * lattice_vectors[2] (double the molecular A-site offset)
            # Molecular A-sites use -0.5 * lattice_vectors[2] + CoM correction
            if structure_type == 'bulk':
                # Apply z-offset (double the molecular offset) to match the correct A-site position
                pos_corrected = pos.copy()
                pos_corrected[2] = pos[2] - 1.0 * lattice_vectors[2]
                atomic_symbols.append(ion)
                atomic_positions.append(pos_corrected)
            else:
                # For 2D structures, use position directly
                atomic_symbols.append(ion)
                atomic_positions.append(pos)
        else:
            # Atomic ion (B or X sites)
            atomic_symbols.append(ion)
            atomic_positions.append(pos)
    
    # Create structure with atomic ions first
    if atomic_symbols:
        structure = Atoms(atomic_symbols, positions=atomic_positions)
    else:
        # No atomic ions (shouldn't happen, but handle gracefully)
        structure = Atoms()
    
    # Set cell dimensions for supercell
    nx, ny, nz = supercell_size
    supercell_vectors = [
        nx * lattice_vectors[0],
        ny * lattice_vectors[1],
        nz * lattice_vectors[2]
    ]
    structure.set_cell(supercell_vectors)
    structure.pbc = [1, 1, 1]
    
    # Add molecular A-sites (if any)
    for i, (mol, pos, site_type) in enumerate(molecular_atoms):
        if site_type == 'A':
            # For molecular A-sites, align molecule first
            from .molecule_builder import align_ase_molecule_for_perovskite
            mol_aligned = align_ase_molecule_for_perovskite(mol.copy())
            
            if structure_type == 'bulk':
                # For bulk: use special z-offset as in original _create_bulk_core
                # Original code: x = 0 + r_corr[0], y = 0 + r_corr[1], z = -0.5 * lattice_vectors[2] + r_corr[2]
                # IMPORTANT: For bulk, add_adsorbate uses (0, 0) as base position, not the template position
                # The template position is only used to determine which unit cell in supercell
                # For each unit cell, we need to calculate the offset from (0, 0, 0) of that unit cell
                from .molecule_builder import center_of_mass_correction
                r_corr = center_of_mass_correction(mol_aligned)
                
                # Calculate which unit cell this position belongs to
                # pos is in Angstroms from supercell template (already translated)
                # We need to find which unit cell (ix, iy, iz) this position came from
                # The position was created as: (template_pos + [ix, iy, iz]) * lattice_vectors
                # So: pos = (template_pos[0] + ix) * lv0, etc.
                # For bulk template: template_pos = [0, 0, 0]
                # So: pos = [ix * lv0, iy * lv1, iz * lv2]
                # Therefore: ix = pos[0] / lv0, etc.
                unit_cell_size = lattice_vectors
                # Use floor division to get the unit cell index
                # Add small epsilon to handle floating point precision issues
                epsilon = 1e-6
                ix = int(np.floor((pos[0] + epsilon) / unit_cell_size[0])) if unit_cell_size[0] > 0 else 0
                iy = int(np.floor((pos[1] + epsilon) / unit_cell_size[1])) if unit_cell_size[1] > 0 else 0
                iz = int(np.floor((pos[2] + epsilon) / unit_cell_size[2])) if unit_cell_size[2] > 0 else 0
                
                # For bulk, each unit cell's A-site is at the template position
                # The template position is [0, 0, 0] in fractional, which is the origin of each unit cell
                # So we should place the molecule's CoM at the template position (pos), not recalculate it
                # The pos is already the correct position in Angstroms from the supercell expansion
                # We just need to add the CoM correction to center the molecule at that position
                x = pos[0] + r_corr[0]
                y = pos[1] + r_corr[1]
                z = pos[2] - 0.5 * unit_cell_size[2] + r_corr[2]
                
                # Use add_adsorbate for bulk (as in original)
                # Note: add_adsorbate position is relative to the structure, not the unit cell
                add_adsorbate(structure, mol_aligned, 
                             position=(x, y), 
                             height=z)
            else:
                # For 2D layers: use place_atoms_at_location (as in original _make_2d_layer)
                # This centers the molecule's CoM at the position
                from .molecule_builder import place_atoms_at_location, add_atoms
                mol_placed = place_atoms_at_location(mol_aligned, pos)
                structure = add_atoms(structure, mol_placed)
    
    return structure


def create_2d_perovskite(Ap, A, B, X, supercell, structure_type='monolayer', BX_dist=None, 
                         penet=0.3, vacuum=12, spacer_distance=2.0, 
                         interlayer_penet=0.0, attachment_end=None, Ap_Rx=None, Ap_Ry=None, Ap_Rz=None, 
                         wrap=None, double=False, Bp=None):
    """
    Create 2D perovskite structures using unified pattern-based assignment (RP, DJ, or monolayer).
    
    This function uses the unified core (_create_unified_core) for consistent structure creation,
    whether using single values or pattern lists for A/B/X ions.
    
    Parameters
    ----------
    Ap : str/Atoms or list[str/Atoms]
        Spacer molecule(s) or atomic cation(s). Can be:
        - Atomic cation string (e.g., 'Cs', 'K', 'Rb') - for all 2D structures
        - Molecules as Atoms objects or SMILES strings
        - List pattern for mixed spacers
        For supercell[0] x supercell[1] = 4 positions, provide list of 4 spacers.
    A : str/Atoms or list[str/Atoms]
        A-site cation(s). Single value or list pattern.
    B : str or list[str]
        B-site cation(s). Single value or list pattern.
    X : str or list[str]
        X-site anion(s). Single value or list pattern.
    supercell : tuple[int, int, int]
        Supercell dimensions [nx, ny, n_layers] where n_layers is the number of octahedral layers.
        Example: [2, 2, 2] creates 2x2 in-plane supercell with 2 layers, needs 4 spacers.
    structure_type : str
        Type of 2D structure: 'rp', 'dj', or 'monolayer' (default: 'monolayer').
    BX_dist : float, optional
        B-X bond distance in Angstrom (auto-calculated if None).
    penet : float
        Penetration of spacer into inorganic layer (fraction of BX bond).
    vacuum : float
        Amount of vacuum to add to unit cell (in Angstrom, for monolayer only).
    spacer_distance : float
        Vacuum gap between opposing spacers for RP phase (in Angstroms, default: 2.0).
    interlayer_penet : float
        Interlayer penetration for RP phase, as a fraction of molecule length (default: 0.0).
        Controls interlocking of Ap cations in RP structures.
    attachment_end : str, optional
        Where to attach spacer. Automatically set based on structure_type:
        - RP: 'both' (always)
        - DJ: 'top' (always)
        - Monolayer: 'top', 'bottom', or 'both' (default: 'both')
        Users can override for monolayer structures.
    Ap_Rx, Ap_Ry, Ap_Rz : float, optional
        Rotation angles in degrees (applied as Rx->Ry->Rz). Applied to all spacers.
    wrap : bool, optional
        Whether to wrap atoms to unit cell. Automatically set to True for all 2D structures.
        Users should not need to specify this.
    double : bool
        Whether to create double perovskite.
    Bp : str, optional
        Second B-site cation for double perovskite.
        
    Returns
    -------
    Atoms
        The 2D perovskite structure.
        
    Notes
    -----
    Pattern mode: Spacer count must match nx * ny of supercell.
    For random patterns, generate the list externally before calling.
    """
    # Extract n_layers from supercell parameter
    if isinstance(supercell, (list, tuple)):
        nx, ny, n_layers = supercell[0], supercell[1], supercell[2]
        # CRITICAL: Pass n_layers in supercell_size so template expansion knows the correct number of layers
        # The expansion logic extracts n_layers from supercell_size[2] for 2D structures
        supercell_size = (nx, ny, n_layers)
    else:
        raise ValueError("supercell must be a list/tuple [nx, ny, n_layers]")
    
    # Validate structure_type
    if structure_type not in ['rp', 'dj', 'monolayer']:
        raise ValueError(f"Invalid structure_type: {structure_type}. Choose 'rp', 'dj', or 'monolayer'.")
    
    # Handle Ap_spacer - can be molecule (Atoms) or atomic cation (str/Atoms with 1 atom)
    # For 2D structures (RP, DJ, monolayer), atomic cations like Cs can be used as spacers
    is_mixed_spacers = isinstance(Ap, list)
    
    # Convert string atomic cations to Atoms objects
    if isinstance(Ap, str):
        # Atomic cation (e.g., 'Cs', 'K', 'Rb') - create single atom
        Ap = Atoms(Ap)
    elif not isinstance(Ap, Atoms):
        raise ValueError("Ap must be a molecule (Atoms object), atomic cation (string), or list of either.")
    
    # Handle spacer molecules/atoms - pattern-based assignment
    # Copy and align spacer molecules (skip alignment for single atoms)
    from .molecule_builder import align_ase_molecule_for_perovskite
    
    if is_mixed_spacers:
        Ap_aligned = []
        for idx, spacer in enumerate(Ap):
            # Convert string atomic cations to Atoms objects
            if isinstance(spacer, str):
                spacer = Atoms(spacer)
            
            if not isinstance(spacer, Atoms):
                raise ValueError(f"Spacer {idx} must be an Atoms object or atomic cation string")
            
            # Validate original spacer
            num_atoms_orig = len(spacer)
            if num_atoms_orig == 0:
                raise ValueError(f"Spacer {idx} has no atoms")
            
            spacer_copy = spacer.copy()
            # Validate after copy
            if len(spacer_copy) != num_atoms_orig:
                raise ValueError(f"Spacer {idx} lost atoms during copy: {num_atoms_orig} -> {len(spacer_copy)}")
            
            # Only align if it's a molecule (more than 1 atom)
            # Single atoms don't need alignment
            if num_atoms_orig > 1:
                spacer_copy = align_ase_molecule_for_perovskite(spacer_copy)
                # Validate after alignment
                if len(spacer_copy) != num_atoms_orig:
                    raise ValueError(f"Spacer {idx} lost atoms during initial alignment: {num_atoms_orig} -> {len(spacer_copy)}")
            
            Ap_aligned.append(spacer_copy)
        Ap = Ap_aligned
    else:
        # Single spacer
        num_atoms_orig = len(Ap)
        if num_atoms_orig == 0:
            raise ValueError("Spacer has no atoms")
        
        Ap = Ap.copy()
        if len(Ap) != num_atoms_orig:
            raise ValueError(f"Spacer lost atoms during copy: {num_atoms_orig} -> {len(Ap)}")
        
        # Only align if it's a molecule (more than 1 atom)
        # Single atoms don't need alignment
        if num_atoms_orig > 1:
            Ap = align_ase_molecule_for_perovskite(Ap)
            if len(Ap) != num_atoms_orig:
                raise ValueError(f"Spacer lost atoms during initial alignment: {num_atoms_orig} -> {len(Ap)}")
    
    # Apply rotations to spacer(s)
    if is_mixed_spacers:
        for spacer in Ap:
            if Ap_Rx:
                spacer.rotate(Ap_Rx, 'x')
            if Ap_Ry:
                spacer.rotate(Ap_Ry, 'y')
            if Ap_Rz:
                spacer.rotate(Ap_Rz, 'z')
    else:
        if Ap_Rx:
            Ap.rotate(Ap_Rx, 'x')
        if Ap_Ry:
            Ap.rotate(Ap_Ry, 'y')
        if Ap_Rz:
            Ap.rotate(Ap_Rz, 'z')
    
    # Handle A/B/X ions - pattern-based
    # Use defaults if not provided
    if A is None:
        A = 'MA'  # Default A-site
    if B is None:
        B = 'Pb'  # Default B-site
    if X is None:
        X = 'I'   # Default X-site
    
    # Handle double perovskite
    if double:
        if Bp is None:
            raise ValueError("Bp (second B-site cation) is required for double perovskites")
        if isinstance(B, list):
            if Bp not in B:
                B = B + [Bp]
        else:
            B = [B, Bp]  # Simple alternating pattern
    
    # Validate and convert BX_dist to lattice vector sizes
    lattice_vector_sizes = _validate_and_convert_BX_dist(BX_dist)
    
    # Apply 2D layer specific transformation (from _make_2d_layer)
    if isinstance(lattice_vector_sizes, (int, float)):
        lattice_vector_sizes = [lattice_vector_sizes, lattice_vector_sizes, lattice_vector_sizes]
    # Pre-compute sqrt calculation once (from _make_2d_layer)
    lv1_sqrt = np.sqrt(lattice_vector_sizes[1]**2 / 2) * 2
    lattice_vector_sizes[0] = lv1_sqrt
    lattice_vector_sizes[1] = lv1_sqrt
    
    # Set structure-specific defaults (automatic based on structure_type)
    # Override any user-provided values - these are fixed for each structure type
    if structure_type == 'rp':
        attachment_end = 'both'  # RP always uses 'both'
        vacuum = spacer_distance
        wrap = True  # Always wrap for 2D structures
    elif structure_type == 'dj':
        attachment_end = 'top'  # DJ always uses 'top'
        wrap = True  # Always wrap for 2D structures
    elif structure_type == 'monolayer':
        # Monolayer supports flexible attachment - use user-provided or default to 'both'
        if attachment_end is None:
            attachment_end = 'both'  # Default for monolayer
        wrap = True  # Always wrap for 2D structures
    
    # Create the base 2D layer using unified core
    try:
        layer = _create_unified_core(
            structure_type='2d_layer',
            lattice_vectors=lattice_vector_sizes,
            A_ions=A,
            B_ions=B,
            X_ions=X,
            n=n_layers,
            supercell_size=supercell_size
        )
    except Exception as e:
        raise
    
    # Compute geometric parameters for cell dimensions BEFORE attaching spacers
    # We need mol_len to calculate proper z positions for RP structures
    # Use effective spacer size (accounts for ionic radius for single atoms)
    from .molecule_builder import add_atoms
    
    # Check if spacers are atomic (single atoms)
    if is_mixed_spacers:
        is_atomic = all(len(sp) == 1 for sp in Ap)
        if is_atomic:
            # For atomic spacers: get ionic radius (not 2*radius)
            ionic_rads = []
            for sp in Ap:
                symbol = sp.get_chemical_symbols()[0]
                try:
                    ionic_rad = get_ionic_radius("A", symbol)
                    ionic_rads.append(ionic_rad)
                except (ValueError, KeyError):
                    ionic_rads.append(1.8)  # Default fallback
            atomic_separation = float(np.mean(ionic_rads))
            mol_len = None  # Not used for atomic spacers
        else:
            # Mixed: average effective spacer size
            mol_lens = [_get_effective_spacer_size(sp) for sp in Ap]
            mol_len = float(np.mean(mol_lens))
            atomic_separation = None
    else:
        is_atomic = (len(Ap) == 1)
        if is_atomic:
            # For atomic spacer: get ionic radius (not 2*radius)
            symbol = Ap.get_chemical_symbols()[0]
            try:
                atomic_separation = get_ionic_radius("A", symbol)
            except (ValueError, KeyError):
                atomic_separation = 1.8  # Default fallback
            mol_len = None  # Not used for atomic spacers
        else:
            mol_len = _get_effective_spacer_size(Ap)
            atomic_separation = None
    
    # Set penetration to 0.0 for atomic spacers by default
    if is_atomic:
        penet = 0.0
    
    # Handle RP structure separately - it requires two shifted layers
    if structure_type == 'rp':
        # Create bottom layer with spacers attached
        bottom_layer = _attach_spacer(
            Ap, layer, n_layers, lattice_vector_sizes, supercell_size,
            attachment_end='both', penet=penet,
            mol_len=mol_len
        )
        
        # Calculate z_length using appropriate formula
        if is_atomic:
            # For atomic spacers: use 2 ionic radii for inter-slab separation
            # Formula: 2 * (layer height) + 2 * ionic_radius
            z_length = 2 * (n_layers * lattice_vector_sizes[2]) + 2 * atomic_separation + 0.5
        else:
            # For molecular spacers: use RP formula with interlayer penetration
            # Formula: n * lattice_vector_sizes[2] * 2 + 2 * (2 * mol_len - 2 * .5 * lattice_vector_sizes[2] * penet - mol_len * interlayer_penet)
            if mol_len is None:
                raise ValueError("mol_len cannot be None for molecular spacers in RP structures")
            z_length = n_layers * lattice_vector_sizes[2] * 2 + 2 * \
                (2 * mol_len - 2 * .5 * lattice_vector_sizes[2] * penet - mol_len * interlayer_penet)
        
        # Calculate the inorganic layer height (without spacers)
        # This is needed to correctly position the top layer
        lv2_n = n_layers * lattice_vector_sizes[2]
        BX_bond_length_z = 0.5 * lattice_vector_sizes[2]
        penet_z = penet * BX_bond_length_z
        
        # Find where the bottom layer's inorganic layer ends (where top spacers attach)
        # Top spacers attach at: lv2_n - penet_z (relative to layer start)
        # But we need the absolute z-position
        layer_z_positions = layer.get_positions()[:, 2]
        layer_bottom_z = np.min(layer_z_positions)  # Bottom of inorganic layer
        layer_top_z = np.max(layer_z_positions)     # Top of inorganic layer
        inorganic_layer_height = layer_top_z - layer_bottom_z
        
        # The top spacers of the bottom layer extend upward from layer_top_z
        # The bottom spacers of the top layer should connect to these
        # So we need to shift the top layer so its bottom (where bottom spacers attach) aligns
        # with the bottom layer's top (where top spacers attach)
        
        # Set cell dimensions
        cell_a = nx * lattice_vector_sizes[0]
        cell_b = ny * lattice_vector_sizes[1]
        bottom_layer.set_cell([cell_a, cell_b, z_length / 2.0, 90, 90, 90])
        bottom_layer.pbc = [1, 1, 1]
        
        # Create top layer by copying the entire bottom layer (with spacers at both ends)
        top_layer = bottom_layer.copy()
        
        # Get the z-center of the original layer (without spacers) for rotation
        # This ensures rotation happens around the correct geometric center
        layer_z_center = (layer_bottom_z + layer_top_z) / 2.0
        
        # Set cell for top_layer before rotation
        # Make cell large enough to contain all atoms after rotation (including molecules that extend beyond)
        top_layer_z_positions = top_layer.get_positions()[:, 2]
        top_layer_z_range = np.max(top_layer_z_positions) - np.min(top_layer_z_positions)
        # Use a larger cell to avoid wrapping issues with molecules
        cell_z_for_rotation = max(z_length / 2.0, top_layer_z_range * 2.0)
        top_layer.set_cell([cell_a, cell_b, cell_z_for_rotation, 90, 90, 90])
        top_layer.pbc = [1, 1, 1]
        
        # Rotate top layer (with spacers) 90 degrees around Z axis
        # Rotate around the cell center in x/y and the z-center of the original layer
        # When we rotate, both the layer AND the spacers rotate together, preserving relative positions
        rotation_center = [cell_a / 2.0, cell_b / 2.0, layer_z_center]
        top_layer.rotate(90, 'z', center=rotation_center)
        
        # CRITICAL: Do NOT wrap after rotation - wrapping breaks molecules by moving atoms individually
        # Instead, we rely on the large cell size to contain all atoms
        # The cell is large enough that atoms shouldn't go outside after rotation
        
        # Calculate the correct shift to position top layer above bottom layer
        # The total structure height should equal z_length
        # We shift the top layer so that: max(top_layer) - min(bottom_layer) = z_length
        
        # After rotation, find the actual z-positions
        top_layer_z_after_rotation = top_layer.get_positions()[:, 2]
        bottom_layer_z_positions = bottom_layer.get_positions()[:, 2]
        
        # Find the current z-range
        bottom_layer_min_z = np.min(bottom_layer_z_positions)
        bottom_layer_max_z = np.max(bottom_layer_z_positions)
        top_layer_min_z = np.min(top_layer_z_after_rotation)
        top_layer_max_z = np.max(top_layer_z_after_rotation)
        
        # Both layers have the same height (they're identical)
        single_layer_height = bottom_layer_max_z - bottom_layer_min_z
        
        # The total structure height should be z_length
        # So: (top_layer_max_z + shift) - bottom_layer_min_z = z_length
        # Therefore: shift = z_length - (top_layer_max_z - bottom_layer_min_z)
        # But we want: (top_layer_max_z + shift) - bottom_layer_min_z = z_length
        # So: shift = z_length - top_layer_max_z + bottom_layer_min_z
        
        # Actually, simpler: shift so that the gap between layers matches the intended spacing
        # The gap should be: z_length - 2 * single_layer_height (if layers don't overlap)
        # But wait, z_length already accounts for both layers and spacing
        
        # Correct approach: shift so total height = z_length
        # Current total if we don't shift: would be 2 * single_layer_height (if layers overlap)
        # We need: (top_layer_max_z + shift) - bottom_layer_min_z = z_length
        shift_amount = z_length - (top_layer_max_z - bottom_layer_min_z)
        top_layer.positions[:, 2] += shift_amount
        
        # Combine both layers
        structure = add_atoms(bottom_layer, top_layer)
        
        # Set final cell dimensions for RP structure
        structure.cell = [cell_a, cell_b, z_length, 90, 90, 90]
        structure.pbc = [1, 1, 1]
        
        # Wrap if requested
        if wrap:
            structure.wrap()
        
        return structure
    
    # Calculate cell dimensions accounting for supercell size
    # IMPORTANT: Multiply x and y by supercell dimensions (nx, ny)
    cell_a = nx * lattice_vector_sizes[0]
    cell_b = ny * lattice_vector_sizes[1]
    
    if structure_type == 'dj' and is_atomic:
        # Special handling for DJ + atomic spacer: create symmetric structure with gaps
        # Structure: lower slab -> gap (2 Å) -> spacer -> gap (2 Å) -> upper slab
        
        # Create lower slab (without spacers)
        lower_slab = layer.copy()
        
        # Get the z-range of the lower slab
        lower_slab_positions = lower_slab.get_positions()
        lower_slab_min_z = np.min(lower_slab_positions[:, 2])
        lower_slab_max_z = np.max(lower_slab_positions[:, 2])
        lower_slab_height = lower_slab_max_z - lower_slab_min_z
        
        # Normalize lower slab to start at z=0
        lower_slab.positions[:, 2] -= lower_slab_min_z
        lower_slab_max_z_normalized = lower_slab_height
        
        # Gap size (2 Å as requested)
        gap_size = 2.0
        
        # Position spacer in the middle
        # Lower slab ends at: lower_slab_height
        # Gap: 2 Å
        # Spacer z-position: lower_slab_height + gap_size
        spacer_z = lower_slab_height + gap_size
        
        # Create spacer atoms at the correct positions
        # Get spacer positions from the original attachment logic
        # But we'll position them manually
        from .molecule_builder import add_atoms
        spacer_structure = Atoms()
        
        # Get spacer symbol
        if is_mixed_spacers:
            spacer_symbol = Ap[0].get_chemical_symbols()[0]
        else:
            spacer_symbol = Ap.get_chemical_symbols()[0]
        
        # Generate spacer positions (same pattern as _generate_attachment_positions)
        # Use the same logic as _generate_attachment_positions
        base_positions_frac = [[0.25, 0.75], [0.75, 0.25]]
        lv0 = lattice_vector_sizes[0]
        lv1 = lattice_vector_sizes[1]
        for base_pos_frac in base_positions_frac:
            for ix in range(nx):
                for iy in range(ny):
                    x = (base_pos_frac[0] + ix) * lv0
                    y = (base_pos_frac[1] + iy) * lv1
                    spacer_structure += Atoms(spacer_symbol, positions=[[x, y, spacer_z]])
        
        # Create upper slab by duplicating and shifting the lower slab
        upper_slab = lower_slab.copy()
        # Upper slab starts after: lower_slab_height + gap_size + spacer_height + gap_size
        # For atomic spacers, spacer_height is essentially 0 (just the atom)
        upper_slab_z_start = lower_slab_height + gap_size + gap_size
        upper_slab.positions[:, 2] += upper_slab_z_start
        
        # Combine: lower slab + spacer + upper slab
        structure = add_atoms(lower_slab, spacer_structure)
        structure = add_atoms(structure, upper_slab)
        
        # Calculate total z_length - use exact positions without extra buffer
        # Get all z positions to find the actual range
        all_positions = structure.get_positions()
        min_z = np.min(all_positions[:, 2])
        max_z = np.max(all_positions[:, 2])
        # Normalize to start at z=0 and set cell to exact size
        structure.positions[:, 2] -= min_z
        # Recalculate max_z after normalization
        all_positions = structure.get_positions()
        max_z_normalized = np.max(all_positions[:, 2])
        # Set z_length to exactly match the structure height (no extra vacuum)
        z_length = max_z_normalized
        
        structure.cell = [cell_a, cell_b, z_length, 90, 90, 90]
    else:
        # For DJ with molecular spacers or monolayer: attach spacers normally
        structure = _attach_spacer(
            Ap, layer, n_layers, lattice_vector_sizes, supercell_size,
            attachment_end=attachment_end, penet=penet,
            mol_len=mol_len if attachment_end == 'both' else None
        )
        
        if structure_type == 'dj':
            # DJ-specific cell calculation for molecular spacers
            z_length = n_layers * lattice_vector_sizes[2] + (mol_len - lattice_vector_sizes[2] * penet)
            structure.cell = [cell_a, cell_b, z_length, 90, 90, 90]
        elif structure_type == 'monolayer':
            # Monolayer: adjust z_length based on attachment type
            if is_atomic:
                # For atomic spacers: use 1 ionic radius for separation
                if attachment_end == 'both':
                    z_length = n_layers * lattice_vector_sizes[2] + 2 * atomic_separation + vacuum
                else:
                    # Only one spacer (top or bottom)
                    z_length = n_layers * lattice_vector_sizes[2] + atomic_separation + vacuum
            else:
                # For molecular spacers: original formulas
                if attachment_end == 'both':
                    # Original calculation for monolayer (with vacuum)
                    z_length = n_layers * lattice_vector_sizes[2] + (2 * mol_len - lattice_vector_sizes[2] * penet) + vacuum
                else:
                    # Only one spacer (top or bottom)
                    z_length = n_layers * lattice_vector_sizes[2] + (mol_len - .5 * lattice_vector_sizes[2] * penet) + vacuum
        
        # Center monolayer structure
        if structure_type == 'monolayer':
            trans_vec = [0, 0, z_length / 2 - structure.get_center_of_mass()[2]]
            from .molecule_builder import translate_atoms
            structure = translate_atoms(structure, trans_vec)
        
        # Set cell first (before checking positions, so positions are relative to cell)
        structure.cell = [cell_a, cell_b, z_length, 90, 90, 90]
        structure.pbc = [1, 1, 1]
        
        # Now check if any atoms are outside the cell bounds
        # Add small buffer if needed
        positions = structure.get_positions()
        if len(positions) > 0:
            min_z = min(pos[2] for pos in positions)
            max_z = max(pos[2] for pos in positions)
            
            # For monolayer, use original logic
            if min_z < -0.1:  # Small tolerance for floating point
                # Translate all atoms up so minimum z is at least 1.0 (buffer)
                trans_z = -min_z + 1.0
                structure.translate([0, 0, trans_z])
                # Recalculate after translation
                positions = structure.get_positions()
                max_z = max(pos[2] for pos in positions) if len(positions) > 0 else max_z
                # Adjust z_length to accommodate the actual range plus buffer
                z_length = max_z - min_z + 2.0  # Add 1.0 buffer on each side
                structure.cell = [cell_a, cell_b, z_length, 90, 90, 90]
            elif max_z > z_length - 0.1:
                # Atoms extend above cell, increase cell size
                z_length = max_z + 1.0  # Add 1.0 buffer at top
                structure.cell = [cell_a, cell_b, z_length, 90, 90, 90]
    
    # For DJ, cell is already set above, but ensure pbc is set
    if structure_type == 'dj':
        structure.pbc = [1, 1, 1]
    
    return structure


def _setup_cell_for_spacers(structure, nx, ny, lv0, lv1):
    """Set cell to supercell size (x and y expanded, z unchanged)."""
    current_cell = structure.cell
    if current_cell is not None:
        if hasattr(current_cell, 'lengths'):
            _, _, current_z = current_cell.lengths()
        else:
            current_z = np.linalg.norm(current_cell[2]) if hasattr(current_cell[0], '__len__') else current_cell[2]
        structure.set_cell([nx * lv0, ny * lv1, current_z])
        structure.pbc = [1, 1, 1]


def _calculate_z_levels(attachment_end, n, lv2, penet):
    """Calculate z-levels and orientations for spacer attachment."""
    lv2_n = n * lv2
    penet_z = penet * 0.5 * lv2
    top_z = lv2_n - penet_z
    bottom_z_adjusted = penet_z
    
    if attachment_end in ('bottom', 'bot'):
        return [(bottom_z_adjusted, 'top', False)], bottom_z_adjusted, top_z
    elif attachment_end == 'top':
        return [(top_z, 'bottom', True)], bottom_z_adjusted, top_z
    else:  # 'both'
        return [
            (bottom_z_adjusted, 'top', False),
            (top_z, 'bottom', True)
        ], bottom_z_adjusted, top_z


def _generate_attachment_positions(z_levels, nx, ny, lv0, lv1):
    """Generate all attachment positions for supercell expansion."""
    base_positions_frac = [[0.25, 0.75], [0.75, 0.25]]
    attachments = []
    
    for z, end_side, rotate_180 in z_levels:
        for base_pos_frac in base_positions_frac:
            for ix in range(nx):
                for iy in range(ny):
                    x = (base_pos_frac[0] + ix) * lv0
                    y = (base_pos_frac[1] + iy) * lv1
                    attachments.append((x, y, z, end_side, rotate_180))
    
    expected = len(z_levels) * len(base_positions_frac) * nx * ny
    if len(attachments) != expected:
        raise ValueError(f"Spacer position generation failed: expected {expected} positions, got {len(attachments)}")
    
    return attachments


def _adjust_atomic_spacer_z(z, attachment_end, n, lv2, bottom_z_adjusted, top_z, symbol):
    """Adjust z position for atomic spacers based on ionic radius."""
    try:
        ionic_rad = get_ionic_radius("A", symbol)
    except (ValueError, KeyError):
        ionic_rad = 1.8
    
    bottom_z = 0.5 * ionic_rad
    top_z_atomic = n * lv2 - 0.5 * ionic_rad
    
    if attachment_end in ('bottom', 'bot'):
        return bottom_z
    elif attachment_end == 'top':
        return top_z_atomic
    else:  # 'both'
        return bottom_z if abs(z - bottom_z_adjusted) < abs(z - top_z) else top_z_atomic


def _process_spacer_attachment(spacer_atoms, x, y, z, end_side, rotate_180, 
                                is_atomic_spacer, attachment_end, n, lv2, 
                                bottom_z_adjusted, top_z):
    """Process a single spacer: rotate, align, and position."""
    from .molecule_builder import end_to_origin, translate_atoms
    
    if is_atomic_spacer:
        symbol = spacer_atoms.get_chemical_symbols()[0]
        z = _adjust_atomic_spacer_z(z, attachment_end, n, lv2, bottom_z_adjusted, top_z, symbol)
    
    if rotate_180 and not is_atomic_spacer:
        com = spacer_atoms.get_center_of_mass()
        spacer_atoms.rotate(180, 'x', center=com)
    
    spacer_atoms = end_to_origin(spacer_atoms, end_side)
    return translate_atoms(spacer_atoms, [x, y, z])


def _attach_spacer(spacer, layer, n, lattice_vector_sizes, supercell_size, 
                   penet=0.3, attachment_end='both', mol_len=None):
    """
    Attach spacer molecules to 2D layer using pattern-based assignment with supercell expansion.
    
    Parameters
    ----------
    spacer : Atoms or list[Atoms]
        Spacer molecule(s). If list, assigned sequentially to positions (pattern-based).
    layer : Atoms
        The 2D inorganic layer
    n : int
        Number of octahedral layers
    lattice_vector_sizes : list[float]
        Lattice vector sizes [a, b, c]
    supercell_size : tuple[int, int, int]
        Supercell dimensions (nx, ny, nz) - only nx and ny are used for spacer positions
    penet : float
        Penetration of spacer into layer
    attachment_end : str
        'top', 'bottom', or 'both'
        
    Returns
    -------
    Atoms
        Structure with spacers attached
    """
    if not isinstance(spacer, list):
        spacer = [spacer]
    
    if isinstance(lattice_vector_sizes, (int, float)):
        lattice_vector_sizes = [lattice_vector_sizes] * 3
    
    structure = layer.copy()
    nx, ny, _ = supercell_size
    lv0, lv1, lv2 = lattice_vector_sizes
    
    _setup_cell_for_spacers(structure, nx, ny, lv0, lv1)
    
    z_levels, bottom_z_adjusted, top_z = _calculate_z_levels(attachment_end, n, lv2, penet)
    attachments = _generate_attachment_positions(z_levels, nx, ny, lv0, lv1)
    
    from .molecule_builder import add_atoms
    
    for i, (x, y, z, end_side, rotate_180) in enumerate(attachments):
        current_spacer = spacer[i % len(spacer)].copy()
        
        if len(current_spacer) == 0:
            raise ValueError(f"Spacer {i} has no atoms after copy")
        
        is_atomic_spacer = (len(current_spacer) == 1)
        positioned_spacer = _process_spacer_attachment(
            current_spacer, x, y, z, end_side, rotate_180, is_atomic_spacer,
            attachment_end, n, lv2, bottom_z_adjusted, top_z
        )
        
        structure = add_atoms(structure, positioned_spacer)
    
    return structure



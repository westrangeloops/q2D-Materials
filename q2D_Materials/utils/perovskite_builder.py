"""
Advanced Perovskite Structure Builder

This module provides comprehensive functionality for creating various types of 
perovskite structures including bulk, 2D variants (Ruddlesden-Popper, Dion-Jacobson, 
monolayer), and double perovskites with sophisticated molecular orientation capabilities.

Integrated from excellent ASE-based perovskite generation code to enhance 
the SVC-Materials package functionality.
"""

from ase.build import add_adsorbate, molecule, bulk
from ase.visualize import view
from ase import Atoms
import numpy as np
import ase.io
from .common_a_sites import (
    get_ionic_radius, ionic_radii, calculate_BX_distance
)

# Default parameters
PENET = 0.3  # Spacer penetration into inorganic layer (fraction of BX bond)
SPACER_DISTANCE = 2.0  # Vacuum gap between opposing spacers in RP structures (Angstroms)


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
        print(f"Warning: Ionic data not available for {B}-{X}, using default distance 2.0 Å")
        return 2.0


def create_bulk_perovskite(A, B, X, BX_dist=None, A_coefficients=None, B_coefficients=None, 
                          X_coefficients=None, double=False, Bp=None, supercell_size=None, 
                          seed=None):
    """
    Unified function to create bulk perovskite structures (single, double, or mixed).
    
    This function consolidates the logic for creating single, double, and mixed bulk perovskites.
    If lists are passed for A, B, or X, the corresponding coefficient lists are required.

    Parameters
    ----------
    A : str/Atoms or list[str/Atoms]
        The A cation(s). If single value: atomic symbol or Atoms object.
        If list: list of A-site cations (requires A_coefficients).
    B : str or list[str]
        The B cation(s). If single value: atomic symbol.
        If list: list of B-site cations (requires B_coefficients).
    X : str or list[str]
        The X anion(s). If single value: atomic symbol.
        If list: list of X-site anions (requires X_coefficients).
    BX_dist : float, optional
        The desired BX bond distance (in Angstrom). If not provided,
        will be calculated automatically from ionic radii data.
    A_coefficients : list[float], optional
        Coefficients for A-site ions (must sum to 1.0). Required if A is a list.
    B_coefficients : list[float], optional
        Coefficients for B-site ions (must sum to 1.0). Required if B is a list.
    X_coefficients : list[float], optional
        Coefficients for X-site ions (must sum to 3.0). Required if X is a list.
    double : bool
        Whether to create double perovskite (default: False).
    Bp : str, optional
        The atomic symbol of the B' cation (required if double=True).
    supercell_size : tuple[int, int, int], optional
        Supercell size for mixed compositions (auto-calculated if None).
    seed : int, optional
        Random seed for reproducible distributions in mixed compositions.

    Returns
    -------
    Atoms
        The bulk perovskite structure as an ASE Atoms object.
    """
    # Detect if mixed composition (any of A, B, X is a list)
    is_mixed = isinstance(A, list) or isinstance(B, list) or isinstance(X, list)
    
    if is_mixed:
        # Mixed composition - validate and convert inputs
        # Convert single values to lists if needed
        A_ions = A if isinstance(A, list) else [A]
        B_ions = B if isinstance(B, list) else [B]
        X_ions = X if isinstance(X, list) else [X]
        
        # Require coefficients for any list that was passed
        if isinstance(A, list):
            if A_coefficients is None:
                raise ValueError("A_coefficients is required when A is a list")
            if len(A_ions) != len(A_coefficients):
                raise ValueError(f"A_ions ({len(A_ions)} ions) and A_coefficients ({len(A_coefficients)} values) must have the same length")
        else:
            A_coefficients = [1.0]
        
        if isinstance(B, list):
            if B_coefficients is None:
                raise ValueError("B_coefficients is required when B is a list")
            if len(B_ions) != len(B_coefficients):
                raise ValueError(f"B_ions ({len(B_ions)} ions) and B_coefficients ({len(B_coefficients)} values) must have the same length")
        else:
            B_coefficients = [1.0]
        
        if isinstance(X, list):
            if X_coefficients is None:
                raise ValueError("X_coefficients is required when X is a list")
            if len(X_ions) != len(X_coefficients):
                raise ValueError(f"X_ions ({len(X_ions)} ions) and X_coefficients ({len(X_coefficients)} values) must have the same length")
        else:
            X_coefficients = [3.0]
        
        # Normalize A-site ions
        A_ions = [_normalize_a_site(A_ion) for A_ion in A_ions]
        
        # Normalize coefficients to handle floating point precision issues
        A_coefficients = list(np.array(A_coefficients) / sum(A_coefficients))
        B_coefficients = list(np.array(B_coefficients) / sum(B_coefficients))
        X_coefficients = list(np.array(X_coefficients) / sum(X_coefficients) * 3.0)  # Scale to 3.0 for X
        
        # Validate coefficients sum (with tolerance for floating point)
        if abs(sum(A_coefficients) - 1.0) > 0.01:
            raise ValueError(f"A_coefficients must sum to 1.0, got {sum(A_coefficients)}")
        if abs(sum(B_coefficients) - 1.0) > 0.01:
            raise ValueError(f"B_coefficients must sum to 1.0, got {sum(B_coefficients)}")
        if abs(sum(X_coefficients) - 3.0) > 0.01:
            raise ValueError(f"X_coefficients must sum to 3.0, got {sum(X_coefficients)}")
        
        # Calculate BX_dist if not provided (use average for mixed)
        if BX_dist is None:
            from .common_a_sites import calculate_BX_distance
            BX_dists = []
            for B_ion in set(B_ions):
                for X_ion in set(X_ions):
                    try:
                        dist = calculate_BX_distance(B_ion, X_ion)
                        BX_dists.append(dist)
                    except:
                        pass
            if BX_dists:
                BX_dist = float(np.mean(BX_dists))
            else:
                # Fallback: use first B and X
                BX_dist = auto_calculate_BX_distance(B_ions[0], X_ions[0])
            print(f"Using calculated B-X distance: {BX_dist:.3f} Å")
        
        # Calculate supercell size if not provided
        if supercell_size is None:
            supercell_size = _calculate_minimal_supercell([A_coefficients, B_coefficients, X_coefficients])
        
        print(f"Creating mixed bulk perovskite with supercell size {supercell_size}")
        print(f"  A-site: {len(A_ions)} ions with coefficients {A_coefficients}")
        print(f"  B-site: {B_ions} with coefficients {B_coefficients}")
        print(f"  X-site: {X_ions} with coefficients {X_coefficients}")
        
        # Calculate lattice vectors
        lattice_vectors = _validate_and_convert_BX_dist(BX_dist)
        
        # Create supercell using the unified core function
        nx, ny, nz = supercell_size
        total_cells = nx * ny * nz
        print(f"  Using unified core: creating {total_cells} independent unit cells")
        
        # Setup RNG
        rng = np.random.default_rng(seed)
        
        # Create all unit cells
        all_cells = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    # Create unit cell with independent random selection
                    cell_seed = None if seed is None else rng.integers(0, 2**31)
                    cell = _create_perovskite_core(
                        A_ions, A_coefficients,
                        B_ions, B_coefficients,
                        X_ions, X_coefficients,
                        structure_type='bulk',
                        lattice_vectors=lattice_vectors,
                        seed=cell_seed
                    )
                    
                    # Translate cell to its position in supercell
                    translation = np.array([ix * lattice_vectors[0],
                                           iy * lattice_vectors[1],
                                           iz * lattice_vectors[2]])
                    cell = _translate(cell, translation)
                    all_cells.append(cell)
        
        # Combine all cells into supercell
        mixed = all_cells[0]
        for cell in all_cells[1:]:
            mixed = _add_atoms(mixed, cell)
        
        # Set supercell dimensions
        supercell_vectors = [nx * lattice_vectors[0],
                            ny * lattice_vectors[1],
                            nz * lattice_vectors[2]]
        mixed.set_cell(supercell_vectors)
        mixed.pbc = [1, 1, 1]
        
        print(f"  ✓ Created {len(mixed)} atoms with proper mixing")
        
        return mixed
    
    # Single or double perovskite (not mixed)
    # Auto-calculate BX distance if not provided
    if BX_dist is None:
        BX_dist = auto_calculate_BX_distance(B, X)
        print(f"Using calculated B-X distance: {BX_dist:.3f} Å")
    
    # Validate double perovskite parameters
    if double and Bp is None:
        raise ValueError("Bp (second B-site cation) is required for double perovskites")
    
    lattice_vectors = _validate_and_convert_BX_dist(BX_dist)
    
    if double:
        # Create 2x2x2 supercell using unified core
        all_cells = []
        for ix in range(2):
            for iy in range(2):
                for iz in range(2):
                    cell = _create_perovskite_core(
                        A_ions=[A], A_probs=[1.0],
                        B_ions=[B], B_probs=[1.0],  # Start with all B
                        X_ions=[X], X_probs=[3.0],
                        structure_type='bulk',
                        lattice_vectors=lattice_vectors,
                        seed=None
                    )
                    # Translate to position in supercell
                    translation = np.array([ix * lattice_vectors[0],
                                           iy * lattice_vectors[1],
                                           iz * lattice_vectors[2]])
                    cell = _translate(cell, translation)
                    all_cells.append(cell)
        
        # Combine cells
        double_uc = all_cells[0]
        for cell in all_cells[1:]:
            double_uc = _add_atoms(double_uc, cell)
        
        # Apply rock-salt ordering: replace B with B' at specific positions
        # Pattern: [0, 3, 5, 6] corresponds to alternating B/B' in 2x2x2
        B_idxs = [i for i, atom in enumerate(double_uc) if atom.symbol == B]
        assert len(B_idxs) == 8, f"Expected 8 B-cations, found {len(B_idxs)}"
        
        # Rock-salt pattern indices in 2x2x2 supercell
        for i in [0, 3, 5, 6]:
            double_uc[B_idxs[i]].symbol = Bp
        
        # Set supercell dimensions
        supercell_vectors = [2 * lattice_vectors[0],
                            2 * lattice_vectors[1],
                            2 * lattice_vectors[2]]
        double_uc.set_cell(supercell_vectors)
        double_uc.pbc = [1, 1, 1]
        
        return double_uc
    else:
        # Single bulk perovskite
        return _create_perovskite_core(
            A_ions=[A], A_probs=[1.0],
            B_ions=[B], B_probs=[1.0],
            X_ions=[X], X_probs=[3.0],
            structure_type='bulk',
            lattice_vectors=lattice_vectors,
            seed=None
        )




def _calculate_minimal_supercell(coefficients_list):
    """
    Calculate minimal supercell size needed to represent fractional coefficients.
    
    Parameters
    ----------
    coefficients_list : list[list[float]]
        List of coefficient lists (e.g., [[0.05, 0.79, 0.18], [1.0], [3.0]])
        
    Returns
    -------
    tuple[int, int, int]
        Supercell size (nx, ny, nz)
    """
    from fractions import Fraction
    import math
    
    # Helper function to calculate LCM
    def _lcm(a, b):
        """Calculate LCM of two numbers."""
        return abs(a * b) // math.gcd(a, b)
    
    # Find LCM of denominators for all coefficient lists
    denominators = []
    for coeffs in coefficients_list:
        if len(coeffs) > 1:  # Only for mixed compositions
            for c in coeffs:
                if c > 0:
                    # Convert to fraction and get denominator
                    frac = Fraction(c).limit_denominator(1000)
                    denominators.append(frac.denominator)
    
    if not denominators:
        # No mixed compositions, use minimal 2x2x2
        return (2, 2, 2)
    
    # Find LCM of all denominators
    lcm_val = denominators[0]
    for d in denominators[1:]:
        lcm_val = _lcm(lcm_val, d)
    
    # Calculate supercell size (aim for ~lcm_val sites)
    # For cubic: n³ ≈ lcm_val, so n ≈ lcm_val^(1/3)
    n = max(2, int(np.ceil(lcm_val ** (1/3))))
    
    # Ensure we have enough sites (at least lcm_val)
    while n * n * n < lcm_val:
        n += 1
    
    return (n, n, n)


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
        print(f"Warning: BX_dist has unexpected type {type(BX_dist)}, using default cell size")
        return 2 * np.array([3.0, 3.0, 3.0])


def _select_ion_by_probability(ion_list, probabilities, rng=None):
    """
    Select an ion from a list based on probabilities.
    
    Parameters
    ----------
    ion_list : list[str/Atoms]
        List of ions to choose from
    probabilities : list[float]
        Probability for each ion (must sum to 1.0)
    rng : np.random.Generator, optional
        Random number generator for reproducibility
        
    Returns
    -------
    str or Atoms
        Selected ion
    """
    if len(ion_list) == 1:
        return ion_list[0]
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Normalize probabilities
    probs = np.array(probabilities) / sum(probabilities)
    
    # Select based on probabilities
    idx = rng.choice(len(ion_list), p=probs)
    return ion_list[idx]


def _generate_spacer_pattern(num_positions, num_spacers, pattern='alternating', rng=None):
    """
    Generate a pattern-based assignment of spacers to positions.
    
    Parameters
    ----------
    num_positions : int
        Number of spacer positions (2 for DJ 'top', 4 for RP 'both')
    num_spacers : int
        Number of different spacer types
    pattern : str
        Pattern type: 'alternating', 'checkerboard', or 'random'
    rng : np.random.Generator, optional
        Random number generator for 'random' pattern
        
    Returns
    -------
    list[int]
        List of spacer indices for each position
    """
    if num_spacers == 1:
        return [0] * num_positions
    
    if pattern == 'alternating':
        # Simple alternating: [0, 1, 0, 1, ...]
        return [i % num_spacers for i in range(num_positions)]
    
    elif pattern == 'checkerboard':
        # Checkerboard pattern (for 4 positions with 2 spacers: [0, 1, 1, 0])
        if num_positions == 4 and num_spacers == 2:
            return [0, 1, 1, 0]  # bottom-left=0, bottom-right=1, top-left=1, top-right=0
        elif num_positions == 2 and num_spacers == 2:
            return [0, 1]  # top-left=0, top-right=1
        else:
            # Fallback to alternating for other cases
            return [i % num_spacers for i in range(num_positions)]
    
    elif pattern == 'random':
        if rng is None:
            rng = np.random.default_rng()
        # Random assignment with equal probability
        return [rng.integers(0, num_spacers) for _ in range(num_positions)]
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}. Use 'alternating', 'checkerboard', or 'random'")


def _create_perovskite_core(A_ions, A_probs, B_ions, B_probs, X_ions, X_probs,
                            structure_type='bulk', n=1, lattice_vectors=None, 
                            seed=None, double_pattern=None):
    """
    Core unified function for creating perovskite structures with mixed sites.
    
    This is the fundamental building block that all perovskite creation functions
    should use. It handles probability-based site placement during structure creation.
    
    Parameters
    ----------
    A_ions : list[str/Atoms]
        List of A-site cations
    A_probs : list[float]
        Probabilities for each A-site ion (must sum to 1.0)
    B_ions : list[str]
        List of B-site cations
    B_probs : list[float]
        Probabilities for each B-site ion (must sum to 1.0)
    X_ions : list[str]
        List of X-site anions
    X_probs : list[float]
        Probabilities for each X-site ion (must sum to 3.0 for ABX₃)
    structure_type : str
        'bulk' or '2d_layer' - type of structure to create
    n : int
        For 2D layers: number of octahedral layers
    lattice_vectors : array
        Lattice vector sizes [a, b, c]
    seed : int, optional
        Random seed for reproducibility
    double_pattern : list[int], optional
        For double perovskites: indices where to place B' instead of B
        E.g., [0, 3, 5, 6] for rock-salt ordering
        
    Returns
    -------
    Atoms
        The perovskite structure with mixed sites
        
    Notes
    -----
    This function creates structures atom-by-atom with probability-based selection,
    eliminating the need for dummy atoms and post-processing replacement.
    """
    # Setup random number generator
    rng = np.random.default_rng(seed)
    
    # Normalize all A-site ions
    A_ions = [_normalize_a_site(A) for A in A_ions]
    
    # Normalize probabilities
    A_probs = np.array(A_probs) / sum(A_probs)
    B_probs = np.array(B_probs) / sum(B_probs)
    X_probs = np.array(X_probs) / sum(X_probs) * 3.0  # Scale X to 3.0 total
    
    if structure_type == 'bulk':
        return _create_bulk_core(A_ions, A_probs, B_ions, B_probs, X_ions, X_probs,
                                lattice_vectors, rng, double_pattern)
    elif structure_type == '2d_layer':
        return _create_2d_layer_core(A_ions, A_probs, B_ions, B_probs, X_ions, X_probs,
                                     n, lattice_vectors, rng, double_pattern)
    else:
        raise ValueError(f"Unknown structure_type: {structure_type}")


def _create_bulk_core(A_ions, A_probs, B_ions, B_probs, X_ions, X_probs,
                     lattice_vectors, rng, double_pattern=None):
    """
    Create a single bulk unit cell with probability-based ion placement.
    
    Internal function used by _create_perovskite_core.
    """
    # Check if we have molecular A-site cations
    has_molecular_A = any(isinstance(A, Atoms) for A in A_ions)
    
    if has_molecular_A:
        # For molecular A-sites, use aligned molecule
        from .molecular_ops import align_ase_molecule_for_perovskite
        A_aligned = []
        for A in A_ions:
            if isinstance(A, Atoms):
                A_aligned.append(align_ase_molecule_for_perovskite(A.copy()))
            else:
                A_aligned.append(A)
        A_ions = A_aligned
        
        # Select A-site ion for this unit cell
        A_selected = _select_ion_by_probability(A_ions, A_probs, rng)
        
        # Select B-site ion
        B_selected = _select_ion_by_probability(B_ions, B_probs, rng)
        
        # Select X-site ions (3 of them)
        X_selected = [_select_ion_by_probability(X_ions, X_probs / 3.0, rng) for _ in range(3)]
        
        # Create structure without A-site first
        perov = Atoms([B_selected] + X_selected,
                     positions=[[0.5, 0.5, 0.5],
                               [0.5, 0.5, 0.0],
                               [0.5, 0.0, 0.5],
                               [0.0, 0.5, 0.5]])
        
        perov.set_cell(lattice_vectors, scale_atoms=True)
        
        # Add molecular A-site
        if isinstance(A_selected, Atoms):
            r_corr = _center_of_mass_correction(A_selected)
            x = 0 + r_corr[0]
            y = 0 + r_corr[1]
            z = -0.5 * lattice_vectors[2] + r_corr[2]
            add_adsorbate(perov, A_selected, position=(x, y), height=z)
        else:
            # Atomic A-site
            A_atom = Atoms([A_selected], positions=[[0.0, 0.0, 0.0]])
            A_atom.set_cell(lattice_vectors, scale_atoms=True)
            perov = _add_atoms(perov, A_atom)
        
        perov.pbc = [1, 1, 1]
        return perov
    else:
        # All atomic sites - simpler case
        A_selected = _select_ion_by_probability(A_ions, A_probs, rng)
        B_selected = _select_ion_by_probability(B_ions, B_probs, rng)
        X_selected = [_select_ion_by_probability(X_ions, X_probs / 3.0, rng) for _ in range(3)]
        
        perov = Atoms([A_selected, B_selected] + X_selected,
                     positions=[[0.0, 0.0, 0.0],
                               [0.5, 0.5, 0.5],
                               [0.5, 0.5, 0.0],
                               [0.5, 0.0, 0.5],
                               [0.0, 0.5, 0.5]])
        
        perov.set_cell(lattice_vectors, scale_atoms=True)
        perov.pbc = [1, 1, 1]
        return perov


def _create_2d_layer_core(A_ions, A_probs, B_ions, B_probs, X_ions, X_probs,
                         n, lattice_vectors, rng, double_pattern=None):
    """
    Create a 2D layer with probability-based ion placement.
    
    Internal function used by _create_perovskite_core.
    This will be implemented to replace _make_2d_layer with mixing support.
    """
    # TODO: Implement unified 2D layer creation with mixing
    # For now, fall back to existing implementation
    raise NotImplementedError("2D layer core with mixing not yet implemented. Use existing _make_2d_layer.")


def create_2d_perovskite(Ap, A, B, X, n, structure_type='monolayer', BX_dist=None, 
                         penet=PENET, vacuum=12, spacer_distance=SPACER_DISTANCE, 
                         attachment_end='both', Ap_Rx=None, Ap_Ry=None, Ap_Rz=None, 
                         wrap=False, double=False, Bp=None, Ap_coefficients=None, 
                         spacer_pattern=None, seed=None):
    """
    Unified function to create 2D perovskite structures (RP, DJ, or monolayer).
    
    This function consolidates the logic for creating Ruddlesden-Popper (RP),
    Dion-Jacobson (DJ), and monolayer 2D perovskite structures.
    
    Parameters
    ----------
    Ap : Atoms or list[Atoms]
        The Ap spacer molecule(s) as an Atoms object or list of Atoms objects.
        If list: requires Ap_coefficients for mixed spacer compositions.
    A : str/Atoms
        The A cation symbol or Atoms object.
    B : str
        The B-site cation symbol.
    X : str
        The X-site anion symbol.
    n : int
        Layer thickness of the inorganic 2D layers.
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
    attachment_end : str
        Where to attach spacer: 'top', 'bottom', or 'both' (default: 'both').
        For RP: always 'both'. For DJ: default 'top'.
    Ap_Rx, Ap_Ry, Ap_Rz : float, optional
        Rotation angles in degrees (applied as Rx->Ry->Rz). Applied to all spacers.
    wrap : bool
        Whether to wrap atoms to unit cell.
    double : bool
        Whether to create double perovskite.
    Bp : str, optional
        Second B-site cation for double perovskite.
    Ap_coefficients : list[float], optional
        Coefficients for spacer molecules (must sum to 1.0). 
        Mutually exclusive with spacer_pattern.
        Required if Ap is a list and spacer_pattern is None.
        Used as probabilities for random selection (probability-based).
    spacer_pattern : str, optional
        Pattern for spacer assignment ('alternating', 'checkerboard', 'random').
        Mutually exclusive with Ap_coefficients.
        Only used for DJ/RP structures. For monolayer, always uses probability-based.
        If provided, uses deterministic pattern-based selection (equal weights for all spacers).
    seed : int, optional
        Random seed for reproducible mixed spacer distributions (probability-based or 'random' pattern).
        
    Returns
    -------
    Atoms
        The 2D perovskite structure.
    """
    # Validate structure_type
    if structure_type not in ['rp', 'dj', 'monolayer']:
        raise ValueError(f"Invalid structure_type: {structure_type}. Choose 'rp', 'dj', or 'monolayer'.")
    
    # Validate Ap_spacer
    if isinstance(Ap, str):
        raise ValueError("Ap must be a molecule in the form of an Atoms object, not a single atom as a string.")
    
    # Detect if mixed spacers
    is_mixed_spacers = isinstance(Ap, list)
    use_pattern = False  # Initialize pattern usage flag
    
    if is_mixed_spacers:
        # Mixed spacers - validate and normalize
        # Ap_coefficients and spacer_pattern are mutually exclusive
        if spacer_pattern is not None and Ap_coefficients is not None:
            print("ERROR: Ap_coefficients and spacer_pattern are mutually exclusive!")
            print("  - Ap_coefficients: Used for probability-based (random) selection")
            print("  - spacer_pattern: Used for deterministic pattern-based selection")
            print("  Please choose one approach:")
            print(f"    Option 1 (pattern-based): Remove Ap_coefficients, keep spacer_pattern='{spacer_pattern}'")
            print(f"    Option 2 (probability-based): Remove spacer_pattern, keep Ap_coefficients={Ap_coefficients}")
            raise ValueError("Ap_coefficients and spacer_pattern are mutually exclusive. "
                           "Use Ap_coefficients for probability-based (random) selection, "
                           "or spacer_pattern for deterministic pattern-based selection.")
        
        # Check if Ap_coefficients is accidentally a string (pattern) instead of a list
        if isinstance(Ap_coefficients, str):
            # User probably meant to use spacer_pattern
            if spacer_pattern is None:
                spacer_pattern = Ap_coefficients
                Ap_coefficients = None  # Will be auto-generated for pattern-based
                print(f"Note: Ap_coefficients was a string ('{spacer_pattern}'), treating it as spacer_pattern")
            else:
                raise ValueError(f"Ap_coefficients must be a list of numbers, not a string. "
                               f"Use spacer_pattern='{Ap_coefficients}' instead, or remove spacer_pattern.")
        
        # Handle pattern-based vs probability-based
        if spacer_pattern is not None:
            # Pattern-based: don't need coefficients, use equal weights
            Ap_coefficients = [1.0 / len(Ap)] * len(Ap)
            print(f"Using pattern-based assignment: '{spacer_pattern}' (equal weights for all spacers)")
        else:
            # Probability-based: require coefficients
            if Ap_coefficients is None:
                raise ValueError("Ap_coefficients is required when Ap is a list and spacer_pattern is not provided")
            
            if not isinstance(Ap_coefficients, (list, np.ndarray)):
                raise ValueError(f"Ap_coefficients must be a list or array of numbers, got {type(Ap_coefficients)}")
            
            if len(Ap) != len(Ap_coefficients):
                raise ValueError(f"Ap ({len(Ap)} spacers) and Ap_coefficients ({len(Ap_coefficients)} values) must have the same length")
            
            # Normalize coefficients
            Ap_coefficients = list(np.array(Ap_coefficients) / sum(Ap_coefficients))
            
            # Validate coefficients sum
            if abs(sum(Ap_coefficients) - 1.0) > 0.01:
                raise ValueError(f"Ap_coefficients must sum to 1.0, got {sum(Ap_coefficients)}")
        
        # Copy and align all spacer molecules
        from .molecular_ops import align_ase_molecule_for_perovskite
        Ap_aligned = []
        for spacer in Ap:
            if isinstance(spacer, str):
                raise ValueError("Ap list must contain Atoms objects, not strings")
            spacer_copy = spacer.copy()
            spacer_copy = align_ase_molecule_for_perovskite(spacer_copy)
            Ap_aligned.append(spacer_copy)
        Ap = Ap_aligned
        
        # Determine mixing strategy based on structure type
        # Monolayer: always probability-based
        # DJ/RP: pattern-based (if specified) or probability-based (fallback)
        if structure_type in ['dj', 'rp']:
            if spacer_pattern is not None:
                use_pattern = True
            else:
                # Auto-detect 50/50 case for 2 spacers -> suggest alternating
                if len(Ap) == 2 and len(Ap_coefficients) == 2 and abs(Ap_coefficients[0] - 0.5) < 0.01:
                    print(f"Note: 50/50 coefficients detected for {structure_type.upper()}. "
                          f"Consider using spacer_pattern='alternating' for deterministic pattern.")
                print(f"Using probability-based assignment for {structure_type.upper()} (consider using spacer_pattern for better physical meaning)")
        
        print(f"Mixed spacers: {len(Ap)} spacer types with coefficients {Ap_coefficients}")
    else:
        # Single spacer - copy and align
        Ap = Ap.copy()
        from .molecular_ops import align_ase_molecule_for_perovskite
        Ap = align_ase_molecule_for_perovskite(Ap)
        print(f"Spacer molecule aligned for perovskite coordination")
    
    # Handle A-site cation
    if not isinstance(A, str):
        A = A.copy()
        if hasattr(A, 'positions'):
            A = align_ase_molecule_for_perovskite(A)
            print(f"A-site cation molecule aligned for perovskite coordination")
    
    # Auto-calculate BX distance if not provided
    if BX_dist is None:
        BX_dist = auto_calculate_BX_distance(B, X)
        print(f"Using calculated B-X distance for 2D {structure_type.upper()}: {BX_dist:.3f} Å")
    
    # Validate and convert BX_dist to lattice vector sizes
    lattice_vector_sizes = _validate_and_convert_BX_dist(BX_dist)
    
    # Set structure-specific defaults
    if structure_type == 'rp':
        attachment_end = 'both'
        vacuum = spacer_distance
        print(f"Creating RP structure (n={n}) with {spacer_distance:.1f} Å vacuum gap between spacers")
    elif structure_type == 'dj':
        if attachment_end == 'both':  # Only override if not explicitly set
            attachment_end = 'top'
        if n == 1:
            print(f"DJ n=1: Using only spacer molecules (no A-site cations between layers)")
        else:
            print(f"DJ n={n}: Using A-site cations between layers + spacer molecules")
    elif structure_type == 'monolayer':
        print(f"Creating monolayer structure (n={n})")
    
    # Make the base 2D layer
    layer = _make_2d_layer(A, B, X, n, lattice_vector_sizes, double=double, Bp=Bp)
    
    # Apply rotations to spacer(s)
    if is_mixed_spacers:
        # Apply rotations to all spacers in the list
        for spacer in Ap:
            if Ap_Rx:
                spacer.rotate(Ap_Rx, 'x')
            if Ap_Ry:
                spacer.rotate(Ap_Ry, 'y')
            if Ap_Rz:
                spacer.rotate(Ap_Rz, 'z')
    else:
        # Single spacer
        if Ap_Rx:
            Ap.rotate(Ap_Rx, 'x')
        if Ap_Ry:
            Ap.rotate(Ap_Ry, 'y')
        if Ap_Rz:
            Ap.rotate(Ap_Rz, 'z')
    
    # Setup RNG for mixed spacers (for probability-based or random pattern)
    rng = np.random.default_rng(seed) if (is_mixed_spacers and not use_pattern) or (use_pattern and spacer_pattern == 'random') else None
    
    # Attach spacers to the layer
    structure = _attach_spacer(Ap, layer, n, lattice_vector_sizes, 
                              attachment_end=attachment_end, penet=penet,
                              spacer_coefficients=Ap_coefficients if is_mixed_spacers else None,
                              spacer_pattern=spacer_pattern if use_pattern else None,
                              rng=rng)
    
    # Compute geometric parameters for cell dimensions
    if is_mixed_spacers:
        # Average molecule length for mixed spacers
        mol_lens = [_get_molecule_length(sp) for sp in Ap]
        mol_len = float(np.mean(mol_lens))
    else:
        mol_len = _get_molecule_length(Ap)
    
    if structure_type == 'dj':
        # DJ-specific cell calculation
        z_length = n * lattice_vector_sizes[2] + (mol_len - lattice_vector_sizes[2] * penet)
        structure.cell = [lattice_vector_sizes[0], lattice_vector_sizes[1], z_length, 90, 90, 90]
    else:
        # RP and monolayer: adjust z_length based on attachment type
        if attachment_end == 'both':
            z_length = n * lattice_vector_sizes[2] + (2 * mol_len - lattice_vector_sizes[2] * penet) + vacuum
        else:
            # Only one spacer (top or bottom)
            z_length = n * lattice_vector_sizes[2] + (mol_len - .5 * lattice_vector_sizes[2] * penet) + vacuum
        
        # Center monolayer structure
        if structure_type == 'monolayer':
            trans_vec = [0, 0, z_length / 2 - structure.get_center_of_mass()[2]]
            structure = _translate(structure, trans_vec)
        
        structure.cell = [lattice_vector_sizes[0], lattice_vector_sizes[1], z_length, 90, 90, 90]
    
    structure.pbc = [1, 1, 1]
    
    if wrap:
        structure.wrap()
    
    return structure


def determine_molecule_orientation(atoms, cartesian=True):
    """
    Aims to determine the cartesian axis of orientation of the input molecule.
    This is to aid users of the code in adding their own spacer molecules in 
    conjunction with the orient_along_z function.

    Parameters
    ----------
    atoms : ase.Atoms object
        The molecule to be analyzed.
    cartesian : bool
        If True, the function will return the cartesian axis of orientation.

    Returns
    -------
    axis : str
        The best guess for axis of orientation of the molecule. (x, y, or z)
    """
    positions = atoms.positions

    at_num = len(positions)
    if at_num < 2:
        return "Input atoms object is not a molecule."

    i = 0
    j = 0
    dir_vec = np.zeros((3,))
    print("Analyzing Molecule for orientation")
    while i < at_num:
        j = 0
        while i > j:
            dir_vec[0] += abs(positions[i, 0] - positions[j, 0])
            dir_vec[1] += abs(positions[i, 1] - positions[j, 1])
            dir_vec[2] += abs(positions[i, 2] - positions[j, 2])
            j += 1
        i += 1

    if cartesian:
        if dir_vec[0] > dir_vec[1] and dir_vec[0] > dir_vec[2]:
            print("Best guess is current orientation along X-direction")
            cart_dir = 'x'
        elif dir_vec[1] > dir_vec[0] and dir_vec[1] > dir_vec[2]:
            print("Best guess is current orientation along Y-direction")
            cart_dir = 'y'
        elif dir_vec[2] > dir_vec[0] and dir_vec[2] > dir_vec[1]:
            print("Best guess is current orientation along Z-direction")
            cart_dir = 'z'
        else:
            # Default case for symmetrical molecules or ambiguity
            print("Ambiguous orientation; defaulting to Z-direction")
            cart_dir = 'z'
        return cart_dir
    else:
        return ("Not yet implemented, please orient the molecule along one of "
                "the cartesian axes prior to input. Note pubchem_atoms_search will "
                "generally return molecules oriented along x-axis.")


def orient_along_z(atoms, theta=90, invert=False):
    """
    Aims to determine the cartesian axis of orientation of the input molecule,
    and then reorient it along the Z axis for usage with the rest of the code
    base. This is a very primitive function, and may or may not be easier than
    just opening and reorienting the molecule in a visualizer by hand.

    Parameters
    ----------
    atoms : Atoms
        The molecule to be analyzed.
    theta : float
        The angle of rotation to be applied to the molecule (in degrees).
    invert : bool
        If True, the molecule will be inverted prior to rotation.

    Returns
    -------
    mod_atoms : Atoms
        The rotated Atoms object.
    """
    mod_atoms = atoms.copy()

    # Flip the molecule.
    if invert:
        theta += 180
    
    # Shift COM to the origin.
    mod_atoms = _com_to_origin(mod_atoms)

    direction = determine_molecule_orientation(mod_atoms)
    print(direction)
    if direction == 'x':
        mod_atoms.rotate(theta, 'y')
    elif direction == 'y':
        mod_atoms.rotate(theta, 'x')
    else:
        if invert:
            mod_atoms.rotate(180, 'x')
        else:
            print("Already oriented on z-axis, and no inversion requested.")
    return mod_atoms


# Internal helper functions
def _center_of_mass_correction(mol, mol_index=0):
    """
    Intended for internal use only.

    Purpose:
        In ase, add_adsorbate treats the coordinate of the molecule as the
        coordinate of mol_index, this function computes 
        r_corr = r_mol_index - r_com. 
        Then r_mol_index - r_corr = r_com. Useful when the desired placement is
        determined by the CoM of the molecule.

    Parameters
    ----------
    mol : Atoms
        Atoms object containing the molecule
    mol_index : int
        Index of the atom for which the molecules 'position' is determined,
        default 0 in ASE.
    
    Returns
    -------
    r_corr : array
        Correction vector [rx, ry, rz] required to shift position to CoM.
    """
    r_mol_index = mol.positions[mol_index]
    r_com = np.around(mol.get_center_of_mass(), decimals=4)
    return r_mol_index - r_com


def _make_2d_layer(A, B, X, n, lattice_vector_sizes, double=False, Bp=None):
    """
    For internal use only. Generate the 2D layer without A' spacer molecule.

    Parameters
    ----------
    A : Atoms or str
        A-site cation molecule or symbol.
    B : str
        B-site cation symbol.
    X : str
        X-site anion symbol.
    n : int
        Number of octahedral layers.
    lattice_vector_sizes : float or array
        Lattice vector sizes (cubic if float).
    double : bool
        Whether to create double perovskite.
    Bp : str, optional
        Second B-site cation for double perovskite.

    Returns
    -------
    Atoms
        The 2D layer without organic spacers.
    """
    if type(lattice_vector_sizes) == int or type(lattice_vector_sizes) == float:
        lattice_vector_sizes = [lattice_vector_sizes,
                                lattice_vector_sizes, lattice_vector_sizes]
    
    # Normalize A-site (convert molecular cation strings to Atoms objects)
    A = _normalize_a_site(A)
    
    # If still a string (atomic cation), create single atom
    if type(A) == str:
        A = Atoms(A, positions=[[0, 0, 0]])

    # Experimental cell construction
    lattice_vector_sizes[0] = np.sqrt(lattice_vector_sizes[1]**2 / 2) * 2
    lattice_vector_sizes[1] = lattice_vector_sizes[0]
    atomList = [X, X]
    positionList = [[.25, .25, 0], [.75, .75, 0]]
    
    for i in range(n):
        if double:
            if i % 2 == 0:
                this_layer_atoms = [X, X, B, X, X, Bp, X, X]
            if i % 2 == 1:
                this_layer_atoms = [X, X, Bp, X, X, B, X, X]
        else:
            this_layer_atoms = [X, X, B, X, X, B, X, X]

        this_layer_positions = [
            [0, 0, .5 + i], [.5, 0, .5 + i], [.25, .25, .5 + i],
            [0, .5, .5 + i], [.5, .5, .5 + i], [.75, .75, .5 + i],
            [.25, .25, 1 + i], [.75, .75, 1 + i]
        ]
        atomList.extend(this_layer_atoms)
        positionList.extend(this_layer_positions)
    
    layer_2d = Atoms(atomList, positions=positionList)
    layer_2d.positions[:] *= lattice_vector_sizes

    # Place A-site cations between layers
    for i in range(n - 1):
        r1 = [.25 * lattice_vector_sizes[0], .75 * lattice_vector_sizes[1],
              lattice_vector_sizes[2] + lattice_vector_sizes[2] * i]
        r2 = [.75 * lattice_vector_sizes[0], .25 * lattice_vector_sizes[1],
              lattice_vector_sizes[2] + lattice_vector_sizes[2] * i]

        A1 = _place_atoms_at_location(A.copy(), r1)
        layer_2d = _add_atoms(layer_2d, A1)
        A2 = _place_atoms_at_location(A.copy(), r2)
        layer_2d = _add_atoms(layer_2d, A2)

    return layer_2d


def _com_to_origin(atoms):
    """
    For internal use only.
    
    Purpose: Center the center of mass of some atoms on the origin
    
    Parameters
    ----------
    atoms : Atoms
        Input atoms

    Returns
    -------
    mod_atoms : Atoms
        Modified atoms object, centered on origin.
    """
    mod_atoms = atoms.copy()
    com = mod_atoms.get_center_of_mass()
    curr_positions = mod_atoms.get_positions()
    new_positions = np.zeros(curr_positions.shape)
    for i in range(len(curr_positions)):
        new_positions[i] = np.around(curr_positions[i] - com, decimals=4)
    mod_atoms.set_positions(new_positions)
    return mod_atoms


def _end_to_origin(atoms, side):
    """
    For internal use only.

    Purpose: 
        For the A' cations, it is necessary to be careful about the layer 
        penetration depth. For this reason, it desired to translate the origin 
        of the molecule to the end (either top or bottom w.r.t z-axis).

    Parameters
    ----------
    atoms : Atoms
        The ase Atoms object containing the desired molecule.
    side : str
        'top' and 'bottom' center the molecule around the top or 
        bottom part of the molecule.
        *I.e. if you're attaching to the bottom of a perovskite,
        you want side = 'top'.

    Returns
    -------
    mod_atoms : Atoms
        The ase Atoms object with the bottom/top of the molecule
        (wrt z) centered on origin. bottom/top atom will be at coordinates
        [CoM x, CoM, y, min(z coords in molecule)]
    """
    mod_atoms = atoms.copy()
    com = mod_atoms.get_center_of_mass()
    if side == 'bottom':
        zmin = min(mod_atoms.positions[:, 2])
        translation_vector = np.array([-com[0], -com[1], -zmin])
    elif side == 'top':
        zmax = max(mod_atoms.positions[:, 2])
        translation_vector = np.array([-com[0], -com[1], -zmax])
    mod_atoms = _translate(mod_atoms, translation_vector)
    return mod_atoms


def _add_atoms(atoms, new_atoms):
    """
    For internal use only.

    Purpose: 
        Combining ase Atoms objects
        
    Parameters
    ----------
    atoms : Atoms
        First set of atoms *This atoms' cell parameters are used
    new_atoms : Atoms
        Second set of atoms

    Returns
    -------
    combined_atoms : Atoms
        New Atoms object containing both input objects,
        in the cell of the first.

    Note:
        This returns a new Atoms object, not a modified version of the inputs.
    """
    at_syms = atoms.get_chemical_symbols().copy()
    new_at_syms = new_atoms.get_chemical_symbols().copy()
    at_syms.extend(new_at_syms)

    combined_atoms = Atoms(at_syms, cell=atoms.cell)

    num_of_atoms = len(at_syms)
    combined_atoms.set_positions(np.append(
        atoms.get_positions(),
        new_atoms.get_positions()).reshape((num_of_atoms, 3)))
    return combined_atoms


def _translate(atoms, r):
    """
    For internal use only.

    Purpose:
        Apply spatial translations to Atoms objects.

    Parameters
    ----------
    atoms : Atoms
        Input atoms
    r : np.array
        Vector for the translation (3,)

    Returns
    -------
    atoms : Atoms
        Translated atoms (Also modified Atoms object.)
    """
    try:
        r = np.array(r)
    except:
        pass

    curr_positions = atoms.get_positions()
    atoms.set_positions(
        curr_positions + np.broadcast_to(r, curr_positions.shape))
    return atoms


def _place_atoms_at_location(atoms, r):
    """
    For internal use only.

    Purpose:
        Place the desired atoms' CoM at the location r.
        
    Parameters
    ----------
    atoms : Atoms
        Input atoms
    r : array
        Vector for the translation (3,)
        
    Returns
    -------
    mod_atoms : Atoms
        The modified atoms object.
    """
    mod_atoms = atoms.copy()
    mod_atoms = _com_to_origin(mod_atoms)
    mod_atoms = _translate(mod_atoms, r)
    return mod_atoms


def _attach_spacer(spacer, layer, n, lattice_vector_sizes, penet=PENET, 
                   attachment_end='both', spacer_coefficients=None, 
                   spacer_pattern=None, rng=None):
    """
    For internal use only. Attach organic spacers to inorganic layers.

    Parameters
    ----------
    spacer : Atoms or list[Atoms]
        Organic spacer molecule(s). If list, uses pattern-based or probability-based selection.
    layer : Atoms
        2D perovskite layer without organic spacers.
    n : int
        Layer thickness (number of octahedra in out-of-plane direction).
    lattice_vector_sizes : array or float
        Lattice vector sizes.
    penet : float
        Penetration depth of spacer (in units of BX bond length).
    attachment_end : str
        'both', 'top', or 'bottom' - where to attach spacers.
    spacer_coefficients : list[float], optional
        Coefficients for mixed spacers (required if spacer is a list).
    spacer_pattern : str, optional
        Pattern for spacer assignment ('alternating', 'checkerboard', 'random').
        If provided, uses pattern-based selection instead of probability-based.
    rng : np.random.Generator, optional
        Random number generator for probability-based or 'random' pattern selection.

    Returns
    -------
    Atoms
        Layer with attached spacers.
    """
    if type(lattice_vector_sizes) == float or type(lattice_vector_sizes) == int:
        lattice_vector_sizes = [lattice_vector_sizes,
                                lattice_vector_sizes, lattice_vector_sizes]

    BX_bond_length_z = .5 * lattice_vector_sizes[2]
    
    # Determine number of positions
    if attachment_end == 'both':
        num_positions = 4
    else:  # 'top' or 'bottom'
        num_positions = 2
    
    # Generate pattern if using pattern-based selection
    pattern_indices = None
    if isinstance(spacer, list) and spacer_pattern is not None:
        pattern_indices = _generate_spacer_pattern(num_positions, len(spacer), spacer_pattern, rng)
    
    # Helper function to select spacer (single, pattern-based, or probability-based)
    position_idx = [0]  # Use list to track position index across calls
    
    def _select_spacer():
        if isinstance(spacer, list):
            if spacer_pattern is not None:
                # Pattern-based selection
                idx = pattern_indices[position_idx[0]]
                position_idx[0] += 1
                return spacer[idx].copy()
            else:
                # Probability-based selection
                return _select_ion_by_probability(spacer, spacer_coefficients, rng).copy()
        else:
            return spacer.copy()
    
    # Reset position index for each attachment section
    # Add to the bottom
    if attachment_end == 'bottom' or attachment_end == 'bot':
        position_idx[0] = 0  # Reset for bottom section
        # First spacer (bottom-left)
        spacer1 = _select_spacer()
        spacer1 = _end_to_origin(spacer1, 'top')
        spacer1 = _translate(spacer1, [lattice_vector_sizes[0] * .25,
                                     lattice_vector_sizes[1] * .75,
                                     penet * BX_bond_length_z])
        layer = _add_atoms(layer, spacer1)

        # Second spacer (bottom-right)
        spacer2 = _select_spacer()
        spacer2 = _end_to_origin(spacer2, 'top')
        spacer2 = _translate(spacer2, [lattice_vector_sizes[0] * .75,
                                     lattice_vector_sizes[1] * .25,
                                     penet * BX_bond_length_z])
        layer = _add_atoms(layer, spacer2)

    elif attachment_end == 'top':
        position_idx[0] = 0  # Reset for top section
        # First spacer (top-left)
        spacer1 = _select_spacer()
        spacer1.rotate(180, 'x')
        spacer1 = _end_to_origin(spacer1, 'bottom')
        spacer1 = _translate(spacer1, [lattice_vector_sizes[0] * .25,
                                     lattice_vector_sizes[1] * .75,
                                     n * lattice_vector_sizes[2] -
                                     penet * BX_bond_length_z])
        layer = _add_atoms(layer, spacer1)

        # Second spacer (top-right)
        spacer2 = _select_spacer()
        spacer2.rotate(180, 'x')
        spacer2 = _end_to_origin(spacer2, 'bottom')
        spacer2 = _translate(spacer2, [lattice_vector_sizes[0] * .75,
                                     lattice_vector_sizes[1] * .25,
                                     n * lattice_vector_sizes[2] -
                                     penet * BX_bond_length_z])
        layer = _add_atoms(layer, spacer2)

    elif attachment_end == 'both':
        position_idx[0] = 0  # Reset for both section (4 positions)
        # First spacer (bottom-left)
        spacer1 = _select_spacer()
        spacer1 = _end_to_origin(spacer1, 'top')
        spacer1 = _translate(spacer1, [lattice_vector_sizes[0] * .25,
                                      lattice_vector_sizes[1] * .75,
                                      penet * BX_bond_length_z])
        layer = _add_atoms(layer, spacer1)

        # Second spacer (bottom-right)
        spacer2 = _select_spacer()
        spacer2 = _end_to_origin(spacer2, 'top')
        spacer2 = _translate(spacer2, [lattice_vector_sizes[0] * .75,
                                      lattice_vector_sizes[1] * .25,
                                      penet * BX_bond_length_z])
        layer = _add_atoms(layer, spacer2)

        # Third spacer (top-left)
        spacer3 = _select_spacer()
        spacer3.rotate(180, 'x')
        spacer3 = _end_to_origin(spacer3, 'bottom')
        spacer3 = _translate(spacer3, [lattice_vector_sizes[0] * .25,
                                      lattice_vector_sizes[1] * .75,
                                      n * lattice_vector_sizes[2] -
                                      penet * BX_bond_length_z])
        layer = _add_atoms(layer, spacer3)

        # Fourth spacer (top-right)
        spacer4 = _select_spacer()
        spacer4.rotate(180, 'x')
        spacer4 = _end_to_origin(spacer4, 'bottom')
        spacer4 = _translate(spacer4, [lattice_vector_sizes[0] * .75,
                                      lattice_vector_sizes[1] * .25,
                                      n * lattice_vector_sizes[2] -
                                      penet * BX_bond_length_z])
        layer = _add_atoms(layer, spacer4)

    return layer


def _get_molecule_length(atoms, direction='z'):
    """
    For internal use only.

    Purpose:
        Determine the length of the molecule along a desired direction, as
        defined simply by max-min of the positions along the desired axis.
        This is needed for the translation of 2D layers in out-of-plane 
        direction for stacking in the DJ, RP phases.
        
    Parameters
    ----------
    atoms : Atoms
        The input molecule's Atoms object
    direction : str
        'x', 'y', 'z' direction for the computation.
        
    Returns
    -------
    mol_length : float
        Molecule length along desired direction.
    """
    pos = atoms.positions

    if direction == 'x':
        idx = 0
    elif direction == 'y':
        idx = 1
    elif direction == 'z':
        idx = 2

    return max(pos[:, idx]) - min(pos[:, idx])

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


def make_bulk(A, B, X, BX_dist=None):
    """
    Makes a bulk perovskite structure.
    
    This function now uses the unified _create_perovskite_core for consistency.

    Parameters
    ----------
    A : str/Atoms
        The A cation. If str, the atomic symbol of the A cation. If Atoms, the
        Atoms object of the A cation.
    B : str
        The atomic symbol of the B cation.
    X : str
        The atomic symbol of the X anion.
    BX_dist : float, optional
        The desired BX bond distance (in Angstrom). If not provided,
        will be calculated automatically from ionic radii data.

    Returns
    -------
    perov : Atoms
        The bulk perovskite structure as an ASE Atoms object.
    """
    # Auto-calculate BX distance if not provided
    if BX_dist is None:
        BX_dist = auto_calculate_BX_distance(B, X)
        print(f"Using calculated B-X distance: {BX_dist:.3f} Å")
    
    # Use unified core with single ions (probabilities = 1.0)
    lattice_vectors = _validate_and_convert_BX_dist(BX_dist)
    
    return _create_perovskite_core(
        A_ions=[A], A_probs=[1.0],
        B_ions=[B], B_probs=[1.0],
        X_ions=[X], X_probs=[3.0],
        structure_type='bulk',
        lattice_vectors=lattice_vectors,
        seed=None
    )


def make_double(A, B, Bp, X, BX_dist):
    """
    Makes a double perovskite structure with rock-salt B-site ordering.
    
    Uses the unified core to create a 2x2x2 supercell, then applies B/B' ordering.

    Parameters
    ----------
    A : str/Atoms
        The A cation. If str, the atomic symbol of the A cation. If Atoms, the
        Atoms object of the organic A cation.
    B : str
        The atomic symbol of the B cation.
    Bp : str
        The atomic symbol of the B' cation.
    X : str
        The atomic symbol of the X anion.
    BX_dist : float
        The desired BX bond distance (in Angstrom).

    Returns
    -------
    double_perov : Atoms
        The double perovskite structure as an ASE Atoms object.
    """
    # Create 2x2x2 supercell using unified core
    lattice_vectors = _validate_and_convert_BX_dist(BX_dist)
    
    # Create all 8 unit cells
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


def make_mixed_bulk(A_ions, A_coefficients, B_ions, B_coefficients, 
                    X_ions, X_coefficients, BX_dist=None, supercell_size=None,
                    distribution='random', seed=None):
    """
    Create a mixed bulk perovskite structure with multiple ions on A, B, or X sites.
    
    This function uses the "dummy atom approach": creates a supercell with dummy "He" 
    atoms at A-sites, then replaces them with actual mixed ions according to stoichiometric 
    coefficients. This approach is simple, reliable, and could be extended to 2D structures.
    
    TODO: Extend this pattern to make_mixed_rp(), make_mixed_dj(), make_mixed_monolayer()
    
    Parameters
    ----------
    A_ions : list[str/Atoms]
        List of A-site cations (e.g., ["Cs", "MA", "FA"])
    A_coefficients : list[float]
        Coefficients for A-site ions (must sum to 1.0)
    B_ions : list[str]
        List of B-site metal cations (e.g., ["Pb"])
    B_coefficients : list[float]
        Coefficients for B-site ions (must sum to 1.0)
    X_ions : list[str]
        List of X-site anions (e.g., ["I"] or ["Br", "I"])
    X_coefficients : list[float]
        Coefficients for X-site ions (must sum to 3.0 for ABX₃)
    BX_dist : float, optional
        B-X bond distance in Angstrom (auto-calculated if None)
    supercell_size : tuple[int, int, int], optional
        Supercell size (nx, ny, nz). If None, automatically calculated.
    distribution : str
        'random' or 'ordered' - how to distribute mixed ions (default: 'random')
    seed : int, optional
        Random seed for reproducible distributions
        
    Returns
    -------
    Atoms
        Mixed bulk perovskite structure
        
    Examples
    --------
    >>> # Triple-cation perovskite (Cs₀.₀₅MA₀.₇₉FA₀.₁₈PbI₃)
    >>> mixed = make_mixed_bulk(
    ...     A_ions=["Cs", "MA", "FA"],
    ...     A_coefficients=[0.05, 0.79, 0.18],
    ...     B_ions=["Pb"], B_coefficients=[1.0],
    ...     X_ions=["I"], X_coefficients=[3.0]
    ... )
    
    >>> # Mixed halides (MAPbBr₀.₅I₂.₅)
    >>> mixed = make_mixed_bulk(
    ...     A_ions=["MA"], A_coefficients=[1.0],
    ...     B_ions=["Pb"], B_coefficients=[1.0],
    ...     X_ions=["Br", "I"], X_coefficients=[0.5, 2.5]
    ... )
    """
    # Validate that ions and coefficients lists have matching lengths
    if len(A_ions) != len(A_coefficients):
        raise ValueError(f"A_ions ({len(A_ions)} ions) and A_coefficients ({len(A_coefficients)} values) must have the same length")
    if len(B_ions) != len(B_coefficients):
        raise ValueError(f"B_ions ({len(B_ions)} ions) and B_coefficients ({len(B_coefficients)} values) must have the same length")
    if len(X_ions) != len(X_coefficients):
        raise ValueError(f"X_ions ({len(X_ions)} ions) and X_coefficients ({len(X_coefficients)} values) must have the same length")
    
    # Normalize A-site ions
    A_ions = [_normalize_a_site(A) for A in A_ions]
    
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
    # Each unit cell is created independently with probability-based ion selection
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


def make_2d_double(Ap, A, B, Bp, X, n, BX_dist, phase='rp', penet=PENET,
                   spacer_distance=SPACER_DISTANCE, Ap_Rx=None, Ap_Ry=None, Ap_Rz=None, wrap=False):
    """
    Makes a 2D double perovskite structure (RP, DJ, or monolayer phase).

    Parameters
    ----------
    Ap : Atoms
        The Ap spacer molecule as an Atoms object.
    A : str/Atoms
        The A cation symbol or Atoms object.
    B : str
        The B-site cation symbol.
    Bp : str
        The B'-site cation symbol (second B cation).
    X : str
        The X-site anion symbol.
    n : int
        Layer thickness of the inorganic 2D layers.
    BX_dist : float
        B-X bond distance in Angstrom.
    phase : str
        Phase type: 'rp', 'dj', or 'monolayer'.
    penet : float
        Penetration of spacer into inorganic layer (fraction of BX bond).
    spacer_distance : float
        Vacuum gap between opposing spacers for RP phase (in Angstroms, default: 2.0).
    Ap_Rx, Ap_Ry, Ap_Rz : float, optional
        Rotation angles in degrees (applied as Rx->Ry->Rz).
    wrap : bool
        Whether to wrap atoms to unit cell.

    Returns
    -------
    Atoms
        The 2D double perovskite structure.
    """
    if phase == 'rp':
        return make_rp(Ap, A, B, X, n, BX_dist, penet=penet,
                        spacer_distance=spacer_distance, Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, 
                        Ap_Rz=Ap_Rz, wrap=wrap, Bp=Bp, double=True)
    elif phase == 'dj':
        return make_dj(Ap, A, B, X, n, BX_dist, penet=penet,
                      Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz,
                      wrap=wrap, Bp=Bp, double=True)
    elif phase == 'monolayer':
        return make_monolayer(Ap, A, B, X, n, BX_dist, penet=penet,
                            vacuum=12, Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz,
                            wrap=wrap, Bp=Bp, double=True)
    else:
        raise ValueError("Invalid phase. Choose 'rp', 'dj', or 'monolayer'.")


def make_rp(Ap, A, B, X, n, BX_dist=None, penet=PENET, spacer_distance=SPACER_DISTANCE,
              Ap_Rx=None, Ap_Ry=None, Ap_Rz=None, wrap=False, double=False, Bp=None):
    """
    Makes a 2D Ruddlesden-Popper perovskite structure.
    
    RP is a special case of monolayer with spacers attached to both sides and a 
    minimal vacuum gap between opposing spacers (default: 2 Å).

    Parameters
    ----------
    Ap : Atoms
        The Ap spacer molecule as an Atoms object.
    A : str/Atoms
        The A cation symbol or Atoms object.
    B : str
        The B-site cation symbol.
    X : str
        The X-site anion symbol.
    n : int
        Layer thickness of the inorganic layer (number of octahedra).
    BX_dist : float, optional
        B-X bond distance in Angstrom (auto-calculated if None).
    penet : float
        Penetration of spacer into inorganic layer (fraction of BX bond).
    spacer_distance : float
        Vacuum gap between opposing spacers (in Angstroms, default: 2.0).
    Ap_Rx, Ap_Ry, Ap_Rz : float, optional
        Rotation angles in degrees (applied as Rx->Ry->Rz).
    wrap : bool
        Whether to wrap atoms to unit cell.
    double : bool
        Whether to create double perovskite.
    Bp : str, optional
        Second B-site cation for double perovskite.

    Returns
    -------
    Atoms
        The 2D RP perovskite structure.
        
    Notes
    -----
    RP structures are created using the monolayer function with attachment_end='both'
    and vacuum=spacer_distance (minimal gap between spacers).
    """
    print(f"Creating RP structure (n={n}) with {spacer_distance:.1f} Å vacuum gap between spacers")
    
    # RP is simply a monolayer with both-sided attachment and minimal vacuum
    return make_monolayer(Ap, A, B, X, n, BX_dist=BX_dist, penet=penet, 
                         vacuum=spacer_distance, Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, 
                         Ap_Rz=Ap_Rz, wrap=wrap, attachment_end='both',
                         double=double, Bp=Bp)


def make_dj(Ap_spacer, A_site_cation, B_site_cation, X_site_anion, n, BX_dist=None, 
            penet=PENET, Ap_Rx=None, Ap_Ry=None, Ap_Rz=None, wrap=False, 
            attachment_end='top', double=False, Bp=None):
    """
    Makes a 2D Dion-Jacobson perovskite structure.

    Parameters
    ----------
    Ap_spacer : Atoms
        The A'-site spacer molecule as an Atoms object.
    A_site_cation : str/Atoms
        The A-site cation symbol or Atoms object.
        - n=1: A-site cation is ignored (only spacer used)
        - n>1: A-site cations are placed between inorganic layers
    B_site_cation : str
        The B-site cation symbol.
    X_site_anion : str
        The X-site anion symbol.
    n : int
        Layer thickness of the inorganic 2D layers.
    BX_dist : float, optional
        B-X bond distance in Angstrom (auto-calculated if None).
    penet : float
        Penetration of spacer into inorganic layer (fraction of BX bond).
    Ap_Rx, Ap_Ry, Ap_Rz : float, optional
        Rotation angles in degrees (applied as Rx->Ry->Rz).
    wrap : bool
        Whether to wrap atoms to unit cell.
    attachment_end : str
        Where to attach spacer: 'top', 'bottom', or 'both'.
    double : bool
        Whether to create double perovskite.
    Bp : str, optional
        Second B-site cation for double perovskite.
        
    Returns
    -------
    Atoms
        The DJ perovskite structure.
    """
    if type(Ap_spacer) == str:
        raise ValueError("Ap_spacer must be a molecule in the form of an Atoms "
                "object, not a single atom as a string.")
    else:
        Ap_spacer = Ap_spacer.copy()
        # Align the spacer molecule for proper perovskite coordination
        from .molecular_ops import align_ase_molecule_for_perovskite
        Ap_spacer = align_ase_molecule_for_perovskite(Ap_spacer)
        print(f"Spacer molecule aligned for perovskite coordination")

    if type(A_site_cation) != str:
        A_site_cation = A_site_cation.copy()
        # Align the A-site cation molecule if it's an Atoms object
        if hasattr(A_site_cation, 'positions'):
            A_site_cation = align_ase_molecule_for_perovskite(A_site_cation)
            print(f"A-site cation molecule aligned for perovskite coordination")

    # Auto-calculate BX distance if not provided
    if BX_dist is None:
        BX_dist = auto_calculate_BX_distance(B_site_cation, X_site_anion)
        print(f"Using calculated B-X distance for 2D DJ: {BX_dist:.3f} Å")

    # Validate and convert BX_dist to lattice vector sizes
    lattice_vector_sizes = _validate_and_convert_BX_dist(BX_dist)

    # DJ Structure Logic - handle A-site placement based on n:
    # For n=1: No interlayer A-sites needed (only inorganic framework + spacer)
    # For n>1: A-sites needed between inorganic layers
    if n == 1:
        print(f"DJ n=1: Using only spacer molecules (no A-site cations between layers)")
    else:
        print(f"DJ n={n}: Using A-site cations between layers + spacer molecules")
    
    layer = _make_2d_layer(A_site_cation, B_site_cation, X_site_anion, n, 
                          lattice_vector_sizes, double=double, Bp=Bp)

    if Ap_Rx:
        Ap_spacer.rotate(Ap_Rx, 'x')
    if Ap_Ry:
        Ap_spacer.rotate(Ap_Ry, 'y')
    if Ap_Rz:
        Ap_spacer.rotate(Ap_Rz, 'z')
    
    # Attach spacers to the layer
    dj_phase = _attach_spacer(Ap_spacer, layer, n, lattice_vector_sizes,
                             attachment_end=attachment_end, penet=penet)

    # Compute geometric parameters for where along z the spacer will be added
    mol_len = _get_molecule_length(Ap_spacer)
    z_length = n * lattice_vector_sizes[2] + \
        (mol_len - lattice_vector_sizes[2] * penet)

    dj_phase.cell = [lattice_vector_sizes[0],
                     lattice_vector_sizes[1], z_length, 90, 90, 90]
    dj_phase.pbc = [1, 1, 1]

    if wrap:
        dj_phase.wrap()

    return dj_phase


def make_monolayer(Ap, A, B, X, n, BX_dist=None, penet=PENET, vacuum=12, 
                   Ap_Rx=None, Ap_Ry=None, Ap_Rz=None, wrap=False, 
                   attachment_end='both', double=False, Bp=None):
    """
    Makes a monolayer 2D perovskite structure.

    Parameters
    ----------
    Ap : Atoms
        The Ap spacer molecule as an Atoms object.
    A : str/Atoms
        The A cation symbol or Atoms object.
    B : str
        The B-site cation symbol.
    X : str
        The X-site anion symbol.
    n : int
        Layer thickness of the inorganic 2D layers.
    BX_dist : float, optional
        B-X bond distance in Angstrom (auto-calculated if None).
    penet : float
        Penetration of spacer into inorganic layer (fraction of BX bond).
    vacuum : float
        Amount of vacuum to add to unit cell (in Angstrom).
    Ap_Rx, Ap_Ry, Ap_Rz : float, optional
        Rotation angles in degrees (applied as Rx->Ry->Rz).
    wrap : bool
        Whether to wrap atoms to unit cell.
    attachment_end : str
        Where to attach spacer: 'both', 'top', or 'bottom' (default: 'both').
    double : bool
        Whether to create double perovskite.
    Bp : str, optional
        Second B-site cation for double perovskite.

    Returns
    -------
    Atoms
        The monolayer 2D perovskite structure.
    """
    Ap = Ap.copy()
    # Align the spacer molecule for proper perovskite coordination
    from .molecular_ops import align_ase_molecule_for_perovskite
    Ap = align_ase_molecule_for_perovskite(Ap)
    print(f"Monolayer spacer molecule aligned for perovskite coordination")
    
    if type(A) != str:
        A = A.copy()
        # Align the A-site cation molecule if it's an Atoms object
        if hasattr(A, 'positions'):
            A = align_ase_molecule_for_perovskite(A)
            print(f"Monolayer A-site cation molecule aligned for perovskite coordination")

    # Auto-calculate BX distance if not provided
    if BX_dist is None:
        BX_dist = auto_calculate_BX_distance(B, X)
        print(f"Using calculated B-X distance for monolayer: {BX_dist:.3f} Å")

    # Validate and convert BX_dist to lattice vector sizes
    lattice_vector_sizes = _validate_and_convert_BX_dist(BX_dist)

    # Make the base 2d layer with no spacers attached
    layer = _make_2d_layer(A, B, X, n, lattice_vector_sizes, double=double, Bp=Bp)

    if Ap_Rx:
        Ap.rotate(Ap_Rx, 'x')
    if Ap_Ry:
        Ap.rotate(Ap_Ry, 'y')
    if Ap_Rz:
        Ap.rotate(Ap_Rz, 'z')
    
    # Attach spacers to the layer
    ml_phase = _attach_spacer(Ap, layer, n, lattice_vector_sizes, 
                             attachment_end=attachment_end, penet=penet)

    # Compute geometric parameters for where along z the spacer will be added
    mol_len = _get_molecule_length(Ap)
    
    # Adjust z_length based on attachment type
    if attachment_end == 'both':
        z_length = n * lattice_vector_sizes[2] + \
            (2 * mol_len - lattice_vector_sizes[2] * penet) + vacuum
    else:
        # Only one spacer (top or bottom)
        z_length = n * lattice_vector_sizes[2] + \
            (mol_len - .5 * lattice_vector_sizes[2] * penet) + vacuum

    trans_vec = [0, 0, z_length / 2 - ml_phase.get_center_of_mass()[2]]
    ml_phase = _translate(ml_phase, trans_vec)

    ml_phase.cell = [lattice_vector_sizes[0], lattice_vector_sizes[1], z_length, 90, 90, 90]
    ml_phase.pbc = [1, 1, 1]

    if wrap:
        ml_phase.wrap()

    return ml_phase


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
                   attachment_end='both'):
    """
    For internal use only. Attach organic spacers to inorganic layers.

    Parameters
    ----------
    spacer : Atoms
        Organic spacer molecule.
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

    Returns
    -------
    Atoms
        Layer with attached spacers.
    """
    if type(lattice_vector_sizes) == float or type(lattice_vector_sizes) == int:
        lattice_vector_sizes = [lattice_vector_sizes,
                                lattice_vector_sizes, lattice_vector_sizes]

    BX_bond_length_z = .5 * lattice_vector_sizes[2]
    
    # Add to the bottom
    if attachment_end == 'bottom' or attachment_end == 'bot':
        # First spacer (bottom-left)
        spacer1 = spacer.copy()
        spacer1 = _end_to_origin(spacer1, 'top')
        spacer1 = _translate(spacer1, [lattice_vector_sizes[0] * .25,
                                     lattice_vector_sizes[1] * .75,
                                     penet * BX_bond_length_z])
        layer = _add_atoms(layer, spacer1)

        # Second spacer (bottom-right)
        spacer2 = spacer.copy()
        spacer2 = _end_to_origin(spacer2, 'top')
        spacer2 = _translate(spacer2, [lattice_vector_sizes[0] * .75,
                                     lattice_vector_sizes[1] * .25,
                                     penet * BX_bond_length_z])
        layer = _add_atoms(layer, spacer2)

    elif attachment_end == 'top':
        # First spacer (top-left)
        spacer1 = spacer.copy()
        spacer1.rotate(180, 'x')
        spacer1 = _end_to_origin(spacer1, 'bottom')
        spacer1 = _translate(spacer1, [lattice_vector_sizes[0] * .25,
                                     lattice_vector_sizes[1] * .75,
                                     n * lattice_vector_sizes[2] -
                                     penet * BX_bond_length_z])
        layer = _add_atoms(layer, spacer1)

        # Second spacer (top-right)
        spacer2 = spacer.copy()
        spacer2.rotate(180, 'x')
        spacer2 = _end_to_origin(spacer2, 'bottom')
        spacer2 = _translate(spacer2, [lattice_vector_sizes[0] * .75,
                                     lattice_vector_sizes[1] * .25,
                                     n * lattice_vector_sizes[2] -
                                     penet * BX_bond_length_z])
        layer = _add_atoms(layer, spacer2)

    elif attachment_end == 'both':
        # First spacer (bottom-left)
        spacer1 = spacer.copy()
        spacer1 = _end_to_origin(spacer1, 'top')
        spacer1 = _translate(spacer1, [lattice_vector_sizes[0] * .25,
                                      lattice_vector_sizes[1] * .75,
                                      penet * BX_bond_length_z])
        layer = _add_atoms(layer, spacer1)

        # Second spacer (bottom-right)
        spacer2 = spacer.copy()
        spacer2 = _end_to_origin(spacer2, 'top')
        spacer2 = _translate(spacer2, [lattice_vector_sizes[0] * .75,
                                      lattice_vector_sizes[1] * .25,
                                      penet * BX_bond_length_z])
        layer = _add_atoms(layer, spacer2)

        # Third spacer (top-left)
        spacer3 = spacer.copy()
        spacer3.rotate(180, 'x')
        spacer3 = _end_to_origin(spacer3, 'bottom')
        spacer3 = _translate(spacer3, [lattice_vector_sizes[0] * .25,
                                      lattice_vector_sizes[1] * .75,
                                      n * lattice_vector_sizes[2] -
                                      penet * BX_bond_length_z])
        layer = _add_atoms(layer, spacer3)

        # Fourth spacer (top-right)
        spacer4 = spacer.copy()
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

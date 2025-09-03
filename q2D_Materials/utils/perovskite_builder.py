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
import sys
from .common_a_sites import (
    get_ionic_radius, ionic_radii, calculate_BX_distance
)

# Default penetration parameters
PENET = 0.3
INTERLAYER_PENET = 0.4


def spacer_info(spacerdir):
    """
    Load spacer molecule information from a data file.
    
    Parameters
    ----------
    spacerdir : str
        Directory containing spacers.dat file
        
    Returns
    -------
    list
        List of spacer information
    """
    spacers = []
    with open(spacerdir + '/spacers.dat') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if idx > 4:
                spacers.append(line.split())
    return spacers


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


def analyze_perovskite_stability(A, B, X):
    """
    Analyze perovskite stability using ionic radii and tolerance factors.
    Similar to inspiration.py's compute_goldschmidt_tolerance and compute_octahedral_tolerance.
    
    Parameters
    ----------
    A : str
        A-site cation symbol
    B : str
        B-site cation symbol
    X : str
        X-site anion symbol
        
    Returns
    -------
    dict
        Complete stability analysis
    """
    from .common_a_sites import get_ionic_radius, calculate_tolerance_factor, calculate_BX_distance
    
    try:
        # Get ionic radii
        r_A = get_ionic_radius("A", A)
        r_B = get_ionic_radius("B", B)
        r_X = get_ionic_radius("X", X)
        
        # Calculate tolerance factor (Goldschmidt)
        tolerance_factor = calculate_tolerance_factor(A, B, X)
        
        # Calculate octahedral tolerance factor (similar to inspiration.py)
        octahedral_factor = r_B / r_X
        
        # Calculate B-X bond distance
        bx_distance = calculate_BX_distance(B, X)
        
        # Estimate lattice parameter (similar to inspiration.py approach)
        estimated_lattice_parameter = 2 * bx_distance
        
        # Stability assessment based on tolerance factor
        if 0.8 <= tolerance_factor <= 1.1:
            stability = "Highly Stable (Cubic/Tetragonal)"
        elif 0.7 <= tolerance_factor < 0.8:
            stability = "Moderately Stable (Orthorhombic)"
        elif 0.6 <= tolerance_factor < 0.7:
            stability = "Marginally Stable (May form with distortions)"
        elif tolerance_factor > 1.1:
            stability = "Likely Unstable (Too large A-site)"
        else:
            stability = "Likely Unstable (Too small A-site)"
        
        # Additional stability factors
        size_factor_classification = "Ideal" if 0.85 <= tolerance_factor <= 1.0 else "Non-ideal"
        
        return {
            'composition': f"{A}{B}{X}3",
            'tolerance_factor': tolerance_factor,
            'octahedral_factor': octahedral_factor,
            'BX_distance': bx_distance,
            'estimated_lattice_parameter': estimated_lattice_parameter,
            'stability_assessment': stability,
            'size_factor_classification': size_factor_classification,
            'ionic_radii': {
                'A': r_A,
                'B': r_B, 
                'X': r_X
            }
        }
        
    except ValueError as e:
        raise ValueError(f"Cannot analyze {A}{B}{X}3: {e}")


def validate_perovskite_composition(A, B, X, show_analysis=True):
    """
    Validate and analyze a perovskite composition using ionic radii data.
    
    Parameters
    ----------
    A : str
        A-site cation symbol
    B : str
        B-site cation symbol
    X : str
        X-site anion symbol
    show_analysis : bool
        Whether to print detailed analysis
        
    Returns
    -------
    dict
        Analysis results including tolerance factor and stability
    """
    try:
        analysis = analyze_perovskite_stability(A, B, X)
        
        if show_analysis:
            print(f"\nPerovskite Composition Analysis: {analysis['composition']}")
            print("=" * 50)
            print(f"Tolerance Factor: {analysis['tolerance_factor']:.3f}")
            print(f"Octahedral Factor: {analysis['octahedral_factor']:.3f}")
            print(f"B-X Distance: {analysis['BX_distance']:.3f} Å")
            print(f"Estimated Lattice Parameter: {analysis['estimated_lattice_parameter']:.3f} Å")
            print(f"Stability Assessment: {analysis['stability_assessment']}")
            print(f"Size Factor Classification: {analysis['size_factor_classification']}")
            print("=" * 50)
        
        return analysis
    except ValueError as e:
        if show_analysis:
            print(f"Cannot analyze composition: {e}")
        return None


def make_bulk(A, B, X, BX_dist=None):
    """
    Makes a bulk perovskite structure.

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
    
    if type(BX_dist) == int or type(BX_dist) == float:
        cell = 2*np.array([BX_dist, BX_dist, BX_dist])

    if type(A) == Atoms:
        # Align the A-site cation molecule for proper perovskite coordination
        from .molecular_ops import align_ase_molecule_for_perovskite
        A = align_ase_molecule_for_perovskite(A)
        print(f"Bulk A-site cation molecule aligned for perovskite coordination")
        
        perov = Atoms([B, X, X, X], positions=[[0.5, 0.5, 0.5],
                                               [0.5, 0.5, 0.0],
                                               [0.5, 0.0, 0.5],
                                               [0.0, 0.5, 0.5]])

        perov.set_cell(cell, scale_atoms=True)

        r_corr = _center_of_mass_correction(A)
        x = 0 + r_corr[0]
        y = 0 + r_corr[1]
        z = -.5 * cell[2] + r_corr[2]

        add_adsorbate(perov, A, position=(x, y), height=z)

        perov.pbc = [1, 1, 1]
        return perov

    elif type(A) == str:
        perov = Atoms([A, B, X, X, X],
                      positions=[[0.0, 0.0, 0.0],
                                 [0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.0],
                                 [0.5, 0.0, 0.5],
                                 [0.0, 0.5, 0.5]])

        perov.set_cell(cell, scale_atoms=True)
        perov.pbc = [1, 1, 1]
        return perov


def make_double(A, B, Bp, X, BX_dist):
    """
    Makes a double perovskite structure.

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
    bulk_uc = make_bulk(A, B, X, BX_dist)
    double_uc = bulk_uc.repeat((2, 2, 2))
    B_idxs = []
    for atom in double_uc:
        if atom.symbol == B:
            B_idxs.append(atom.index)
    assert len(B_idxs) == 8, "Too many/Too few B-cations"
    
    for i in [0, 3, 5, 6]:
        double_uc[B_idxs[i]].symbol = Bp

    return double_uc


def make_2d_double(Ap, A, B, Bp, X, n, BX_dist, phase='rp', penet=PENET,
                   interlayer_penet=INTERLAYER_PENET, Ap_Rx=None,
                   Ap_Ry=None, Ap_Rz=None, wrap=False, output=False,
                   output_type='cif', file_name=None):
    """
    Makes a 2d double perovskite structure. Available for the rp, dj, and
    monolayer phases.

    Parameters
    ----------
    Ap : str/Atoms
        The Ap cation. If str, the atomic symbol of the Ap cation. If Atoms,
        the Atoms object of the organic Ap cation.
    A : str/Atoms
        The A cation. If str, the atomic symbol of the A cation. If Atoms, the
        Atoms object of the organic A cation.
    B : str
        The atomic symbol of the B cation.
    Bp : str
        The atomic symbol of the B' cation.
    X : str
        The atomic symbol of the X anion.
    n : int
        The layer thickness of the inorganic 2D layers.
    BX_dist : float
        The desired BX bond distance (in Angstrom).
    phase : str
        The phase of the 2D double perovskite. Available options are 'rp', 'dj',
        and 'monolayer'.
    penet : float
        The penetration of the Ap cation into the inorganic layer (as a fraction
        of the BX bond length).
    interlayer_penet : float
        The penetration associated with interlocking of Ap cations in the rp
        phase (as a fraction of molecule length.).
    Ap_Rx : float
        The desired x-rotation of the Ap cation (in degrees).
    Ap_Ry : float
        The desired y-rotation of the Ap cation (in degrees).
    Ap_Rz : float
        The desired z-rotation of the Ap cation (in degrees).
        *** Note these are applied in succession, e.g. Rx->Ry->Rz ***
    wrap : bool
        Whether or not to wrap atoms which land outside of the unit cell.
    output : bool
        Whether or not to output the structure to a file.
    output_type : str
        The file type to output the structure to. Example options are 'cif',
        'xyz', and 'vasp'.
    file_name : str
        The name of the file to output the structure to.

    Returns
    -------
    double_2d : Atoms
        The 2d double perovskite structure as an ASE Atoms object.
    """
    if phase == 'rp':
        double_2d = make_2drp(Ap, A, B, X, n, BX_dist, penet=penet,
                              exp=True, interlayer_penet=interlayer_penet,
                              Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz,
                              wrap=wrap, output=False, Bp=Bp,
                              double=True)

    elif phase == 'dj':
        double_2d = make_dj(Ap, A, B, X, n, BX_dist, penet=penet, exp=True,
                            Ap_Rx=Ap_Rx, Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz,
                            wrap=wrap, output=False, Bp=Bp, double=True)

    elif phase == 'monolayer':
        double_2d = make_monolayer(Ap, A, B, X, n, BX_dist, penet=penet, 
                                   exp=True, vacuum=12, Ap_Rx=Ap_Rx,
                                   Ap_Ry=Ap_Ry, Ap_Rz=Ap_Rz, wrap=wrap,
                                   output=False, Bp=Bp, double=True)
    else:
        print("Not implemented, choose RP DJ or monolayer.")
        print("phase='rp' or phase='dj' or phase='monolayer'.")
        sys.exit(1)

    return double_2d


def make_2drp(Ap, A, B, X, n, BX_dist=None, penet=PENET, exp=True, 
              interlayer_penet=INTERLAYER_PENET, Ap_Rx=None, Ap_Ry=None, 
              Ap_Rz=None, wrap=False, output=False, output_type='cif',
              file_name=None, double=False, Bp=None):
    """
    Makes a 2D Ruddlesden-Popper perovskite structure.

    Parameters
    ----------
    Ap : str/Atoms
        The Ap cation. If str, the atomic symbol of the Ap cation. If Atoms,
        the Atoms object of the organic Ap cation.
    A : str/Atoms
        The A cation. If str, the atomic symbol of the A cation. If Atoms, the
        Atoms object of the organic A cation.
    B : str
        The atomic symbol of the B cation.
    X : str
        The atomic symbol of the X anion.
    n : int
        The layer thickness of the inorganic 2D layers.
    BX_dist : float
        The desired BX bond distance (in Angstrom).
    penet : float
        The penetration of the Ap cation into the inorganic layer (as a fraction
        of the BX bond length).
    exp : bool
        Flag for the type of unit cell to construct. ! THIS FEATURE IS HIGHLY
        EXPERIMENTAL, USE WITH CARE IF SET TO FALSE !
    interlayer_penet : float
        The penetration associated with interlocking of Ap cations in the rp
        phase (as a fraction of molecule length.).
    Ap_Rx : float
        The desired x-rotation of the Ap cation (in degrees).
    Ap_Ry : float
        The desired y-rotation of the Ap cation (in degrees).
    Ap_Rz : float
        The desired z-rotation of the Ap cation (in degrees).
        *** Note these are applied in succession, e.g. Rx->Ry->Rz ***
    wrap : bool
        Whether or not to wrap atoms which land outside of the unit cell.
    output : bool
        Whether or not to output the structure to a file.
    output_type : str
        The file type to output the structure to. Example options are 'cif',
        'xyz', and 'vasp'.
    file_name : str
        The name of the file to output the structure to.

    Returns
    -------
    rp_phase : Atoms
        The 2drp perovskite structure as an ASE Atoms object.
    """
    if type(Ap) == str:
        raise ValueError("Ap must be a molecule in the form of an Atoms "
                "object, not a single atom as a string.")
    else:
        Ap = Ap.copy()
        # Align the spacer molecule for proper perovskite coordination
        from .molecular_ops import align_ase_molecule_for_perovskite
        Ap = align_ase_molecule_for_perovskite(Ap)
        print(f"RP spacer molecule aligned for perovskite coordination")

    if type(A) != str:
        A = A.copy()
        # Align the A-site cation molecule if it's an Atoms object
        if hasattr(A, 'positions'):
            A = align_ase_molecule_for_perovskite(A)
            print(f"RP A-site cation molecule aligned for perovskite coordination")

    # Auto-calculate BX distance if not provided
    if BX_dist is None:
        BX_dist = auto_calculate_BX_distance(B, X)
        print(f"Using calculated B-X distance for 2D RP: {BX_dist:.3f} Å")

    if type(BX_dist) == float or type(BX_dist) == int:
        lattice_vector_sizes = 2*np.array([BX_dist, BX_dist, BX_dist])

    # Make the base 2d layer with no spacers attached.
    layer = _make_2d_layer(A, B, X, n, lattice_vector_sizes, exp=exp,
                           double=double, Bp=Bp)

    if Ap_Rx:
        Ap.rotate(Ap_Rx, 'x')
    if Ap_Ry:
        Ap.rotate(Ap_Ry, 'y')
    if Ap_Rz:
        Ap.rotate(Ap_Rz, 'z')
    
    # Attach both spacers to the bottom layer
    bottom_layer = _attach_spacer(
        Ap, layer, n, lattice_vector_sizes, exp=exp, attachment_end='both',
        penet=penet)

    # Compute geometric parameters for where along z the spacer will be added
    mol_len = _get_molecule_length(Ap)
    z_length = n * lattice_vector_sizes[2] * 2 + 2 * \
        (2 * mol_len - 2 * .5 *
         lattice_vector_sizes[2] * penet - mol_len * interlayer_penet)
    top_layer = bottom_layer.copy()

    # Important note here worth explaining in detail
    # The 2DRP phase is defined as a .5 .5 shift along the in plane directions
    # between layers, however this is w.r.t ABX3 crystal coords, not exp. cells
    # Exp cell is ~2x larger and rotated 45 deg, so (the important part)
    # the code here selects the 0 direction to do the shift for RP phase.
    if exp:
        top_layer.positions[:, 2] += z_length / 2.0
        top_layer.positions[:, 0] += .5 * lattice_vector_sizes[0]
    else:
        top_layer.positions[:, 2] += z_length / 2.0
        top_layer.positions[:, 0] += .5 * lattice_vector_sizes[0]
        top_layer.positions[:, 1] += .5 * lattice_vector_sizes[1]

    rp_phase = _add_atoms(bottom_layer, top_layer)
    rp_phase.cell = [lattice_vector_sizes[0],
                     lattice_vector_sizes[1], z_length, 90, 90, 90]
    rp_phase.pbc = [1, 1, 1]

    if wrap:
        rp_phase.wrap()

    if output:
        if file_name:
            rp_phase.write(file_name + "." + output_type)
        else:
            fname = "2drp_" + \
                str(np.random.randint(0, 10000)) + "." + output_type
            print("No file name given, written to", fname)
            rp_phase.write(fname)
    return rp_phase


def make_dj(Ap_spacer, A_site_cation, B_site_cation, X_site_anion, n, BX_dist=None, 
            penet=PENET, exp=True, Ap_Rx=None, Ap_Ry=None, Ap_Rz=None, wrap=False, 
            output=False, output_type='cif', file_name=None, attachment_end='top',
            double=False, Bp=None):
    """
    Makes a 2D Dion-Jacobson perovskite structure.

    Parameters
    ----------
    Ap_spacer : str/Atoms
        The A'-site spacer cation. If str, the atomic symbol of the Ap cation. 
        If Atoms, the Atoms object of the organic A'-site spacer molecule that 
        caps the surfaces of the inorganic layers.
    A_site_cation : str/Atoms
        The A-site cation. If str, the atomic symbol of the A cation. If Atoms, 
        the Atoms object of the organic A-site cation. For DJ structures:
        - n=1: A-site cation is ignored (only spacer used)
        - n>1: A-site cations are placed between inorganic layers
    B_site_cation : str
        The atomic symbol of the B-site cation (e.g., 'Pb', 'Sn').
    X_site_anion : str
        The atomic symbol of the X-site anion (e.g., 'I', 'Br', 'Cl').
    n : int
        The layer thickness of the inorganic 2D layers.
    BX_dist : float
        The desired B-X bond distance (in Angstrom).
    penet : float
        The penetration of the Ap spacer into the inorganic layer (as a fraction
        of the BX bond length).
    exp : bool
        Flag for the type of unit cell to construct. ! THIS FEATURE IS HIGHLY
        EXPERIMENTAL, USE WITH CARE IF SET TO FALSE !
    Ap_Rx : float
        The desired x-rotation of the Ap spacer (in degrees).
    Ap_Ry : float
        The desired y-rotation of the Ap spacer (in degrees).
    Ap_Rz : float
        The desired z-rotation of the Ap spacer (in degrees).
        *** Note these are applied in succession, e.g. Rx->Ry->Rz ***
    wrap : bool
        Whether or not to wrap atoms which land outside of the unit cell.
    output : bool
        Whether or not to output the structure to a file.
    output_type : str
        The file type to output the structure to. Example options are 'cif',
        'xyz', and 'vasp'.
    file_name : str
        The name of the file to output the structure to.
        
    Returns
    -------
    dj_phase : Atoms
        The DJ perovskite structure as an ASE Atoms object.
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

    # acceptable only for cubic lattices (of the bulk phase)
    if type(BX_dist) == float or type(BX_dist) == int:
        lattice_vector_sizes = np.array([2*BX_dist, 2*BX_dist, 2*BX_dist])

    # DJ Structure Logic - handle A-site placement based on n:
    # For n=1: No interlayer A-sites needed (only inorganic framework + spacer)
    # For n>1: A-sites needed between inorganic layers
    if n == 1:
        print(f"DJ n=1: Using only spacer molecules (no A-site cations between layers)")
        # Create a dummy structure that won't place any A-sites (since n-1 = 0 iterations)
        layer = _make_2d_layer(A_site_cation, B_site_cation, X_site_anion, n, lattice_vector_sizes, exp=exp,
                               double=double, Bp=Bp)
    else:
        print(f"DJ n={n}: Using A-site cations between layers + spacer molecules")
        # For n>1, the _make_2d_layer will place A-sites for n-1 iterations
        layer = _make_2d_layer(A_site_cation, B_site_cation, X_site_anion, n, lattice_vector_sizes, exp=exp,
                               double=double, Bp=Bp)

    if Ap_Rx:
        Ap_spacer.rotate(Ap_Rx, 'x')
    if Ap_Ry:
        Ap_spacer.rotate(Ap_Ry, 'y')
    if Ap_Rz:
        Ap_spacer.rotate(Ap_Rz, 'z')
    
    # Attach spacers to the layer
    dj_phase = _attach_spacer(
        Ap_spacer, layer, n, lattice_vector_sizes, exp=exp,
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
        print(f"Atoms wrapped to unit cell")

    # Debug: Check final structure
    print(f"Final DJ structure: {len(dj_phase)} atoms")
    if len(dj_phase) > 0:
        positions = dj_phase.get_positions()
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        print(f"Position range: X[{min_pos[0]:.2f}, {max_pos[0]:.2f}], Y[{min_pos[1]:.2f}, {max_pos[1]:.2f}], Z[{min_pos[2]:.2f}, {max_pos[2]:.2f}]")

    if output:
        if file_name:
            dj_phase.write(file_name + "." + output_type)
        else:
            fname = "dj_" + \
                str(np.random.randint(0, 10000)) + "." + output_type
            print("No file name given, written to", fname)
            dj_phase.write(fname)

    return dj_phase


def make_monolayer(Ap, A, B, X, n, BX_dist=None, penet=PENET, exp=True,
                   vacuum=12, Ap_Rx=None, Ap_Ry=None, Ap_Rz=None,
                   wrap=False, output=False, output_type='cif',
                   file_name=None, double=False, Bp=None):
    """
    Makes a monolayer 2D perovskite structure.

    Parameters
    ----------
    Ap : str/Atoms
        The Ap cation. If str, the atomic symbol of the Ap cation. If Atoms,
        the Atoms object of the organic Ap cation.
    A : str/Atoms
        The A cation. If str, the atomic symbol of the A cation. If Atoms, the
        Atoms object of the organic A cation.
    B : str
        The atomic symbol of the B cation.
    X : str
        The atomic symbol of the X anion.
    n : int
        The layer thickness of the inorganic 2D layers.
    BX_dist : float
        The desired BX bond distance (in Angstrom).
    penet : float
        The penetration of the Ap cation into the inorganic layer (as a fraction
        of the BX bond length).
    exp : bool
        Flag for the type of unit cell to construct. ! THIS FEATURE IS HIGHLY
        EXPERIMENTAL, USE WITH CARE IF SET TO FALSE !
    vacuum : float
        The amount of vacuum to add to the unit cell (in Angstrom).
    Ap_Rx : float
        The desired x-rotation of the Ap cation (in degrees).
    Ap_Ry : float
        The desired y-rotation of the Ap cation (in degrees).
    Ap_Rz : float
        The desired z-rotation of the Ap cation (in degrees).
        *** Note these are applied in succession, e.g. Rx->Ry->Rz ***
    wrap : bool
        Whether or not to wrap atoms which land outside of the unit cell.
    output : bool
        Whether or not to output the structure to a file.
    output_type : str
        The file type to output the structure to. Example options are 'cif',
        'xyz', and 'vasp'.
    file_name : str
        The name of the file to output the structure to.

    Returns
    -------
    ml_phase : Atoms
        The monolayer 2D perovskite structure as an ASE Atoms object.
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

    if type(BX_dist) == float or type(BX_dist) == int:
        lattice_vector_sizes = 2*np.array([BX_dist, BX_dist, BX_dist])

    # Make the base 2d layer with no spacers attached.
    layer = _make_2d_layer(A, B, X, n, lattice_vector_sizes, exp=exp,
                           double=double, Bp=Bp)

    if Ap_Rx:
        Ap.rotate(Ap_Rx, 'x')
    if Ap_Ry:
        Ap.rotate(Ap_Ry, 'y')
    if Ap_Rz:
        Ap.rotate(Ap_Rz, 'z')
    
    # Attach both spacers to the layer
    ml_phase = _attach_spacer(
        Ap, layer, n, lattice_vector_sizes, exp=exp, attachment_end='both',
        penet=penet)

    # Compute geometric parameters for where along z the spacer will be added
    mol_len = _get_molecule_length(Ap)
    z_length = n * lattice_vector_sizes[2] + \
        (2 * mol_len - lattice_vector_sizes[2] * penet) + vacuum

    trans_vec = [0, 0, z_length / 2 - ml_phase.get_center_of_mass()[2]]
    ml_phase = _translate(ml_phase, trans_vec)

    ml_phase.cell = [lattice_vector_sizes[0],
                     lattice_vector_sizes[1], z_length, 90, 90, 90]
    ml_phase.pbc = [1, 1, 1]

    if wrap:
        ml_phase.wrap()

    if output:
        if file_name:
            ml_phase.write(file_name + "." + output_type)
        else:
            fname = "ml_" + str(np.random.randint(0, 10000)) + "." + output_type
            print("No file name given, written to", fname)
            ml_phase.write(fname)

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


def _supercell_trafo_matrix(nx, ny, nz):
    """
    Intended for internal use only.

    Purpose:
        Generate the transformation matrix for the supercell.

    Parameters
    ----------
    nx, ny, nz : int
        Number of unit cells in each direction.

    Returns
    -------
    P : (3,3) array
        Transformation matrix for the supercell.
    """
    P = np.zeros((3, 3))
    P[0, 0] = nx
    P[1, 1] = ny
    P[2, 2] = nz
    return P


def _make_2d_layer(A, B, X, n, lattice_vector_sizes, exp=True, double=False,
                   Bp=None):
    """
    For internal use only.

    Purpose:
        Generate the 2D layer, without the A' molecule attached.

    Parameters
    ----------
    A : Atoms or string
        A molecule/atom
    B : string
        B atom in standard perovskite formula
    X : string
        X atom in standard perovskite formula
    n : int
        number of 'layers' in 2d perovskite context. (I.e. #octahedra
        per 2d layer)
    lattice_vector_sizes : float or array
        If float: Assumes bulk analog is cubic unit cell, and scales all 3
        cartesian directions accordingly (to put this more concisely, the
        BX distances don't change with with cartesian direction.)
        If array: Scales all 3 cartesian directions by corresponding element 
    exp : bool
        Flag for experimental cell construction
    double : bool
        Whether to create a double perovskite structure
    Bp : str
        Second B cation for double perovskite

    Returns
    -------
    layer_2d : Atoms
        Atoms object which contains a single layer of the 2dpk
        *without* organic spacers.
    """
    if type(lattice_vector_sizes) == int or type(lattice_vector_sizes) == float:
        lattice_vector_sizes = [lattice_vector_sizes,
                                lattice_vector_sizes, lattice_vector_sizes]
    
    if type(A) == str:
        A = Atoms(A, positions=[[0, 0, 0]])

    # Base layer
    if exp:
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

        # Place the building components for the given inorganic 'layer'
        for i in range(n - 1):
            # Inorganic
            r1 = [.25 * lattice_vector_sizes[0], .75 * lattice_vector_sizes[1],
                  lattice_vector_sizes[2] + lattice_vector_sizes[2] * i]
            r2 = [.75 * lattice_vector_sizes[0], .25 * lattice_vector_sizes[1],
                  lattice_vector_sizes[2] + lattice_vector_sizes[2] * i]

            # Organic (or inorganic A-site) - create fresh copies to avoid modification
            A1 = _place_atoms_at_location(A.copy(), r1)
            layer_2d = _add_atoms(layer_2d, A1)
            A2 = _place_atoms_at_location(A.copy(), r2)
            layer_2d = _add_atoms(layer_2d, A2)

    # Non-experimental cell construction
    else:
        atomList = [X]
        positionList = [[.5, .5, 0]]
        for i in range(n):
            this_layer_atoms = [X, X, B, X]
            this_layer_positions = [
                [0, .5, .5 + i], [.5, 0, .5 + i], 
                [.5, .5, .5 + i], [.5, .5, 1 + i]
            ]
            atomList.extend(this_layer_atoms)
            positionList.extend(this_layer_positions)
        layer_2d = Atoms(atomList, positions=positionList)
        layer_2d.positions[:] *= lattice_vector_sizes
        for i in range(n - 1):
            r = [0, 0, lattice_vector_sizes[2] + lattice_vector_sizes[2] * i]
            A_copy = _place_atoms_at_location(A.copy(), r)
            layer_2d = _add_atoms(layer_2d, A_copy)

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


def _attach_spacer(spacer, layer, n, lattice_vector_sizes, exp=True,
                   penet=PENET, attachment_end='both'):
    """
    For internal use only.

    Purpose:
        Attach the large organic spacers to inorganic layers. Assumes z 
        orientation, with +ve side facing up.

    Parameters
    ----------
    spacer : Atoms
        Organic spacer.
    layer : Atoms
        2D perovskite layer without organic spacers capping the surfaces.
    n : int
        Layer thickness (i.e. #octahedra in out of plane direction).
    lattice_vector_sizes : array or float
        If cubic, fine to input as a float, but more properly this is the size 
        of the a,b,c cell dimensions.
    exp : bool
        Flag for experimental cell construction
    penet : float
        The penetration depth of the organic spacer in units of the BX bond length.
    attachment_end : str
        'both', 'top', 'bot' - Determines if the spacer is to be added on the 
        bottom, top, or both sides of the 2D layer.
        E.g. For 2DRP phase, use both. For DJ use one side only.

    Returns
    -------
    layer : Atoms
        Layer with attached spacers
    """
    if type(lattice_vector_sizes) == float or type(lattice_vector_sizes) == int:
        lattice_vector_sizes = [lattice_vector_sizes,
                                lattice_vector_sizes, lattice_vector_sizes]

    BX_bond_length_z = .5 * lattice_vector_sizes[2]
    
    # Add to the bottom
    if attachment_end == 'bottom' or attachment_end == 'bot':
        if exp:
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
        else:
            spacer1 = spacer.copy()
            spacer1 = _end_to_origin(spacer1, 'top')
            spacer1 = _translate(spacer1, [0, 0, penet * BX_bond_length_z])
            layer = _add_atoms(layer, spacer1)

    elif attachment_end == 'top':
        if exp:
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
        else:
            spacer1 = spacer.copy()
            spacer1.rotate(180, 'x')
            spacer1 = _end_to_origin(spacer1, 'bottom')
            spacer1 = _translate(
                spacer1, [0, 0, n * lattice_vector_sizes[2] -
                         penet * BX_bond_length_z])
            layer = _add_atoms(layer, spacer1)

    elif attachment_end == 'both':
        if exp:
            # First spacer (bottom-left)
            spacer1 = spacer.copy()
            spacer1 = _end_to_origin(spacer1, 'top')
            spacer1 = _translate(spacer1,
                               [lattice_vector_sizes[0] * .25,
                                lattice_vector_sizes[1] * .75,
                                penet * BX_bond_length_z])
            layer = _add_atoms(layer, spacer1)

            # Second spacer (bottom-right)
            spacer2 = spacer.copy()
            spacer2 = _end_to_origin(spacer2, 'top')
            spacer2 = _translate(spacer2,
                               [lattice_vector_sizes[0] * .75,
                                lattice_vector_sizes[1] * .25,
                                penet * BX_bond_length_z])
            layer = _add_atoms(layer, spacer2)

            # Third spacer (top-left)
            spacer3 = spacer.copy()
            spacer3.rotate(180, 'x')
            spacer3 = _end_to_origin(spacer3, 'bottom')
            spacer3 = _translate(
                spacer3, [lattice_vector_sizes[0] * .25,
                         lattice_vector_sizes[1] * .75,
                         n * lattice_vector_sizes[2] -
                         penet * BX_bond_length_z])
            layer = _add_atoms(layer, spacer3)

            # Fourth spacer (top-right)
            spacer4 = spacer.copy()
            spacer4.rotate(180, 'x')
            spacer4 = _end_to_origin(spacer4, 'bottom')
            spacer4 = _translate(
                spacer4, [lattice_vector_sizes[0] * .75,
                         lattice_vector_sizes[1] * .25,
                         n * lattice_vector_sizes[2] -
                         penet * BX_bond_length_z])
            layer = _add_atoms(layer, spacer4)
        else:
            # First spacer (bottom)
            spacer1 = spacer.copy()
            spacer1 = _end_to_origin(spacer1, 'top')
            spacer1 = _translate(spacer1, [0, 0, penet * BX_bond_length_z])
            layer = _add_atoms(layer, spacer1)

            # Second spacer (top)
            spacer2 = spacer.copy()
            spacer2.rotate(180, 'x')
            spacer2 = _end_to_origin(spacer2, 'bottom')
            spacer2 = _translate(
                spacer2, [0, 0, n * lattice_vector_sizes[2] -
                         penet * BX_bond_length_z])
            layer = _add_atoms(layer, spacer2)

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

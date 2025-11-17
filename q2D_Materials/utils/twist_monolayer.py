"""
Twisted bilayer generator for monolayer structures.

Based on the Quadratic Twisted Bilayer Generator by Gabriel Xavier Pereira:
Institute of Physics, University of São Paulo, São Paulo, SP, Brazil
Email: gxpereira@usp.br
"""

import numpy as np
from ase import Atoms


def create_twisted_bilayer(monolayer, m, n, interlayer_distance=11.0, vacuum=12.0, **kwargs):
    """
    Create a twisted bilayer structure from a monolayer.
    
    Based on the Quadratic Twisted Bilayer Generator by Gabriel Xavier Pereira:
    Institute of Physics, University of São Paulo, São Paulo, SP, Brazil
    Email: gxpereira@usp.br
    
    This function generates a twisted bilayer structure from a given quadratic monolayer
    crystal. It constructs two stacked layers — a bottom and a top — that are related
    by a well-defined twist angle determined by the integers (m, n) according to the
    standard commensurate twist formula.
    
    Parameters
    ----------
    monolayer : q2DStructure
        The monolayer structure to twist (must have structure_type='monolayer')
    m : int
        First integer parameter for twist angle calculation
    n : int
        Second integer parameter for twist angle calculation
    interlayer_distance : float, optional
        Vertical distance between the two twisted layers (interlayer) in Angstroms (default: 11.0)
    vacuum : float, optional
        Vacuum space outside the sandwich structure (outerlayer) in Angstroms (default: 12.0)
        This adds vacuum above the top layer and below the bottom layer
    **kwargs : dict
        Additional parameters (currently unused, reserved for future use)
        
    Returns
    -------
    q2DStructure
        New q2DStructure with the twisted bilayer (preserves metadata)
        
    Raises
    ------
    ValueError
        If structure_type is not 'monolayer'
    ImportError
        If pymatgen is not available
    """
    if monolayer.structure_type != 'monolayer':
        raise ValueError(
            f"twist() method is only available for monolayer structures. "
            f"Current structure_type: {monolayer.structure_type}"
        )
    
    try:
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError:
        raise ImportError(
            "pymatgen is required for twist() method. Please install pymatgen."
        )
    
    # Import q2DStructure here to avoid circular imports
    from q2D_Materials.core.structure import q2DStructure
    
    # OPTIMIZATION 1: Direct conversion without file I/O
    # Convert ASE Atoms to pymatgen Structure directly
    adapter = AseAtomsAdaptor()
    monolayer_pmg = adapter.get_structure(monolayer)
    
    # Calculate rotation matrix components (pre-calculate once)
    cost = (m**2 - n**2) / (m**2 + n**2)
    sint = (2 * m * n) / (m**2 + n**2)
    rot_matrix = np.array([[cost, -sint, 0],
                           [sint, cost, 0],
                           [0, 0, 1]], dtype=np.float64)
    
    # OPTIMIZATION 2: Create supercells more efficiently
    # === Bottom layer ===
    # Based on Gabriel Xavier Pereira's implementation
    slab_bottom = monolayer_pmg.copy()
    slab_bottom.make_supercell([[m, n, 0],
                                [-n, m, 0],
                                [0, 0, 1]])
    
    # === Top layer ===
    slab_top = monolayer_pmg.copy()
    slab_top.make_supercell([[m, -n, 0],
                            [n, m, 0],
                            [0, 0, 1]])
    
    # OPTIMIZATION 3: Vectorized operations for combining layers
    # Get all coordinates as numpy arrays (much faster than list comprehensions)
    bottom_coords = np.array([site.coords for site in slab_bottom], dtype=np.float64)
    top_coords = np.array([site.coords for site in slab_top], dtype=np.float64)
    
    # Apply rotation and translation to top layer (vectorized)
    top_coords_rotated = top_coords @ rot_matrix.T
    top_coords_rotated[:, 2] -= interlayer_distance  # Vectorized z-shift
    
    # Combine coordinates (vectorized)
    all_coords = np.vstack([bottom_coords, top_coords_rotated])
    
    # Get species (still need list for pymatgen Structure constructor)
    species = [site.species for site in slab_bottom] + [site.species for site in slab_top]
    
    # Create bilayer structure
    new_lattice = slab_bottom.lattice
    bilayer = Structure(new_lattice, species, all_coords, coords_are_cartesian=True)
    
    # OPTIMIZATION 4: Direct conversion back to ASE without file I/O
    bilayer_atoms = adapter.get_atoms(bilayer)
    
    # OPTIMIZATION 5: Optimize vacuum application (single position calculation)
    positions = bilayer_atoms.positions  # Direct access, no recalculation
    min_z = np.min(positions[:, 2])
    max_z = np.max(positions[:, 2])
    
    # Shift structure so bottom is at z = vacuum/2 (vectorized)
    z_shift = vacuum / 2.0 - min_z
    positions[:, 2] += z_shift  # In-place operation
    
    # Calculate new z-length (no need to recalculate positions)
    new_z_length = max_z - min_z + vacuum
    
    # Update cell z-dimension
    current_cell = bilayer_atoms.cell
    bilayer_atoms.cell = [
        current_cell[0],
        current_cell[1],
        [0, 0, new_z_length]
    ]
    
    # Prepare metadata, avoiding conflicts with existing vacuum
    new_metadata = monolayer._metadata.copy()
    new_metadata['twist_params'] = (m, n)
    new_metadata['interlayer_distance'] = interlayer_distance
    new_metadata['vacuum'] = vacuum  # Override original vacuum with twist vacuum
    
    # Return as q2DStructure with updated metadata
    return q2DStructure(
        bilayer_atoms,
        structure_type='monolayer',  # Keep as monolayer (it's a twisted bilayer of monolayers)
        BX_dist=monolayer.BX_dist,
        A_ions=monolayer.A_ions,
        B_ions=monolayer.B_ions,
        X_ions=monolayer.X_ions,
        supercell_size=monolayer.supercell_size,
        spacer=monolayer.spacer,
        spacer_molecule=monolayer.spacer_molecule,
        **new_metadata
    )


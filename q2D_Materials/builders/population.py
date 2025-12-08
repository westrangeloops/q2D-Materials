"""
Population module for populating structure matrices with atoms.

This module handles ion assignment, molecular alignment, and spacer attachment
to convert abstract position matrices into complete ASE Atoms objects.
"""

import numpy as np
from ase import Atoms
from typing import Union, List, Tuple, Dict, Optional

from .q_builder import QBuilderOutput
from ..utils.molecule_builder import (
    align_ase_molecule_for_perovskite,
    center_of_mass_correction,
    place_atoms_at_location,
    add_atoms,
    end_to_origin,
    translate_atoms,
    get_molecule_length
)
from ..utils.A_sites import get_ionic_radius, is_molecular_a_cation, get_a_site_object


def normalize_a_site(A: Union[str, Atoms]) -> Union[str, Atoms]:
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
            return get_a_site_object(A)
        except (ImportError, ValueError):
            # If conversion fails or not available, return as-is (atomic cation)
            return A
    
    return A


def assign_ions_to_sites(
    position_template: Dict[str, np.ndarray],
    A_ions: Union[str, Atoms, List],
    B_ions: Union[str, List],
    X_ions: Union[str, List],
    Ap_ions: Optional[Union[str, Atoms, List]] = None,
) -> List[Tuple[str, Union[str, Atoms], np.ndarray]]:
    """
    Assign ions to positions using explicit patterns.
    
    This function assigns ions to positions based on explicit patterns provided by the user.
    If a single ion is provided, it's used for all positions of that type.
    If a list is provided, ions are assigned sequentially, cycling if the list is shorter.
    
    Parameters
    ----------
    position_template : dict
        Dictionary with site types ('A', 'B', 'X') and numpy arrays of positions
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
    
    # Convert numpy arrays to lists for iteration
    A_positions = position_template.get('A', np.array([]).reshape(0, 3))
    Ap_positions = position_template.get('Ap', np.array([]).reshape(0, 3))
    B_positions = position_template.get('B', np.array([]).reshape(0, 3))
    X_positions = position_template.get('X', np.array([]).reshape(0, 3))
    
    # Assign A-site positions
    if len(A_positions) > 0:
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

    # Assign Ap-site positions (optional spacers)
    if Ap_ions is not None and len(Ap_positions) > 0:
        if isinstance(Ap_ions, list):
            for i, pos in enumerate(Ap_positions):
                ion = Ap_ions[i % len(Ap_ions)]
                if isinstance(ion, Atoms):
                    ion = ion.copy()
                assignments.append(('Ap', ion, pos))
        else:
            for pos in Ap_positions:
                ion = Ap_ions
                if isinstance(ion, Atoms):
                    ion = ion.copy()
                assignments.append(('Ap', ion, pos))
    
    # Assign B-site positions
    if len(B_positions) > 0:
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
    if len(X_positions) > 0:
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


def populate_structure(
    matrix: QBuilderOutput,
    A_ions: Union[str, Atoms, List],
    B_ions: Union[str, List],
    X_ions: Union[str, List],
    Ap_ions: Optional[Union[str, Atoms, List]] = None,
) -> Atoms:
    """
    Populate structure matrix with atoms based purely on site labels (A, B, X, Ap).
    """
    # Normalize A-site ions (convert molecular strings to Atoms objects)
    if isinstance(A_ions, list):
        A_ions_normalized = []
        for A in A_ions:
            if isinstance(A, Atoms):
                A_ions_normalized.append(A)
            elif isinstance(A, str):
                A_ions_normalized.append(normalize_a_site(A))
            else:
                A_ions_normalized.append(normalize_a_site(A))
        A_ions = A_ions_normalized
    else:
        # Single value
        if isinstance(A_ions, str):
            A_ions = normalize_a_site(A_ions)
        # else: already Atoms object

    # Normalize Ap-site ions if provided
    if Ap_ions is not None:
        if isinstance(Ap_ions, list):
            Ap_normalized = []
            for Ap in Ap_ions:
                if isinstance(Ap, Atoms):
                    Ap_normalized.append(Ap)
                elif isinstance(Ap, str):
                    Ap_normalized.append(normalize_a_site(Ap))
                else:
                    Ap_normalized.append(normalize_a_site(Ap))
            Ap_ions = Ap_normalized
        else:
            if isinstance(Ap_ions, str):
                Ap_ions = normalize_a_site(Ap_ions)
            # else: already Atoms object
    
    # Assign ions to positions using patterns
    assignments = assign_ions_to_sites(
        matrix.positions,
        A_ions, B_ions, X_ions, Ap_ions
    )
    
    # Separate atomic and molecular ions
    # IMPORTANT: For A-sites, we need to handle atomic and molecular separately
    # to avoid duplication. Only atomic A-sites go in the main structure,
    # molecular A-sites are added separately with special positioning.
    atomic_symbols = []
    atomic_positions = []
    molecular_atoms = []  # List of (Atoms, position) tuples
    
    lattice_vectors = matrix.lattice_vector_sizes
    
    for site_type, ion, pos in assignments:
        if isinstance(ion, Atoms):
            # Molecular ion (A-site only) - handle separately
            # Note: ion is already a copy from assign_ions_to_sites, but we copy again
            # when aligning to be extra safe
            # IMPORTANT: Also copy the position array to avoid reference issues
            pos_copy = np.array([pos[0], pos[1], pos[2]])
            molecular_atoms.append((ion, pos_copy, site_type))
        elif site_type in ['A', 'Ap']:
            # A/Ap site: check if it's a molecular cation that failed to convert
            # If it's a string that should be molecular, try to convert again
            if isinstance(ion, str):
                try:
                    if is_molecular_a_cation(ion):
                        # It's a molecular cation - convert it
                        mol_ion = get_a_site_object(ion)
                        molecular_atoms.append((mol_ion, pos, site_type))
                        continue
                except (ImportError, ValueError):
                    # If conversion fails, treat as atomic
                    pass
            # Atomic A/Ap-site (or failed molecular conversion)
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
    
    # Set cell dimensions from matrix
    structure.set_cell(matrix.cell_vectors)
    structure.pbc = [1, 1, 1]
    
    # Add molecular A/Ap-sites (if any)
    for i, (mol, pos, site_type) in enumerate(molecular_atoms):
        if site_type in ['A', 'Ap']:
            # Align and place molecule at the provided position
            mol_aligned = align_ase_molecule_for_perovskite(mol.copy())
            mol_placed = place_atoms_at_location(mol_aligned, pos)
            structure = add_atoms(structure, mol_placed)
    
    return structure


def get_effective_spacer_size(spacer: Atoms) -> float:
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


def setup_cell_for_spacers(structure: Atoms, nx: int, ny: int, lv0: float, lv1: float):
    """Set cell to supercell size (x and y expanded, z unchanged)."""
    current_cell = structure.cell
    if current_cell is not None:
        if hasattr(current_cell, 'lengths'):
            _, _, current_z = current_cell.lengths()
        else:
            current_z = np.linalg.norm(current_cell[2]) if hasattr(current_cell[0], '__len__') else current_cell[2]
        structure.set_cell([nx * lv0, ny * lv1, current_z])
        structure.pbc = [1, 1, 1]


def calculate_z_levels(attachment_end: str, n: int, lv2: float, penet: float) -> Tuple[List, float, float]:
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


def generate_attachment_positions(z_levels: List, nx: int, ny: int, lv0: float, lv1: float) -> List[Tuple]:
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


def adjust_atomic_spacer_z(z: float, attachment_end: str, n: int, lv2: float, 
                           bottom_z_adjusted: float, top_z: float, symbol: str) -> float:
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


def process_spacer_attachment(spacer_atoms: Atoms, x: float, y: float, z: float, 
                             end_side: str, rotate_180: bool, is_atomic_spacer: bool,
                             attachment_end: str, n: int, lv2: float, 
                             bottom_z_adjusted: float, top_z: float) -> Atoms:
    """Process a single spacer: rotate, align, and position."""
    if is_atomic_spacer:
        symbol = spacer_atoms.get_chemical_symbols()[0]
        z = adjust_atomic_spacer_z(z, attachment_end, n, lv2, bottom_z_adjusted, top_z, symbol)
    
    if rotate_180 and not is_atomic_spacer:
        com = spacer_atoms.get_center_of_mass()
        spacer_atoms.rotate(180, 'x', center=com)
    
    spacer_atoms = end_to_origin(spacer_atoms, end_side)
    return translate_atoms(spacer_atoms, [x, y, z])


def attach_spacers(spacer: Union[Atoms, List[Atoms]], layer: Atoms, n: int, 
                   lattice_vector_sizes: np.ndarray, supercell_size: Tuple[int, int, int],
                   penet: float = 0.3, attachment_end: str = 'both', mol_len: float = None,
                   ap_positions: Optional[List[np.ndarray]] = None) -> Atoms:
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
    lattice_vector_sizes : np.ndarray
        Lattice vector sizes [a, b, c]
    supercell_size : tuple[int, int, int]
        Supercell dimensions (nx, ny, nz) - only nx and ny are used for spacer positions
    penet : float
        Penetration of spacer into layer
    attachment_end : str
        'top', 'bottom', or 'both'
    mol_len : float, optional
        Pre-computed molecule length (for efficiency)
    ap_positions : list[np.ndarray], optional
        Pre-computed spacer (Ap) positions from builder. If provided, used for x/y
        placement; z-levels still determined by attachment_end/penet.
        
    Returns
    -------
    Atoms
        Structure with spacers attached
    """
    if not isinstance(spacer, list):
        spacer = [spacer]
    
    if isinstance(lattice_vector_sizes, (int, float)):
        lattice_vector_sizes = np.array([lattice_vector_sizes] * 3)
    
    structure = layer.copy()
    nx, ny, _ = supercell_size
    lv0, lv1, lv2 = lattice_vector_sizes[0], lattice_vector_sizes[1], lattice_vector_sizes[2]
    
    setup_cell_for_spacers(structure, nx, ny, lv0, lv1)
    
    z_levels, bottom_z_adjusted, top_z = calculate_z_levels(attachment_end, n, lv2, penet)
    if ap_positions is not None and len(ap_positions) > 0:
        attachments = []
        for z, end_side, rotate_180 in z_levels:
            for pos in ap_positions:
                attachments.append((pos[0], pos[1], z, end_side, rotate_180))
    else:
        attachments = generate_attachment_positions(z_levels, nx, ny, lv0, lv1)
    
    for i, (x, y, z, end_side, rotate_180) in enumerate(attachments):
        current_spacer = spacer[i % len(spacer)].copy()
        
        if len(current_spacer) == 0:
            raise ValueError(f"Spacer {i} has no atoms after copy")
        
        is_atomic_spacer = (len(current_spacer) == 1)
        positioned_spacer = process_spacer_attachment(
            current_spacer, x, y, z, end_side, rotate_180, is_atomic_spacer,
            attachment_end, n, lv2, bottom_z_adjusted, top_z
        )
        
        structure = add_atoms(structure, positioned_spacer)
    
    return structure


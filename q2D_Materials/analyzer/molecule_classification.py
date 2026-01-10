"""
Molecular component identification and classification.

This module identifies organic and atomic molecular components in the structure
and classifies them based on their position relative to slabs.

Uses pymatgen's CovalentBondNN for proper covalent bond detection, which
prevents incorrectly merging molecules that are spatially close but not
covalently bonded.

Functions
---------
_classify_molecules_by_continuity
    Classify molecules as spacers or A-sites based on slab continuity
_find_molecular_components
    Find all molecular components not part of the inorganic framework
_extract_spacer_molecules
    Extract spacer molecules from non-framework atoms
_match_spacers_to_templates
    Match identified spacers to known templates
"""

import numpy as np
import networkx as nx
from collections import deque
from ase import Atoms
from .perovskite_constants import get_bond_cutoff, COVALENT_RADII
from .pymatgen_utils import extract_molecular_components, build_molecular_graph
from ..utils.geometry import _calculate_distances


def _classify_molecules_by_continuity(
    molecules: list,
    slab_info: dict,
    atom_positions: np.ndarray,
    cell: np.ndarray,
) -> tuple:
    """
    Classify molecules as spacers or A-sites based on their position 
    relative to slab continuity regions.
    
    - Molecules in discontinuity regions (between slabs) = spacers
    - Molecules in continuity regions (within slab z-range) = A-sites
    
    Parameters
    ----------
    molecules : list
        List of ASE Atoms objects representing molecules
    slab_info : dict
        Output from _identify_slabs_by_continuity
    atom_positions : np.ndarray
        Array of all atom positions
    cell : np.ndarray
        Unit cell matrix
        
    Returns
    -------
    tuple
        (spacers, a_sites) - lists of classified molecules
    """
    spacers = []
    a_sites = []
    
    slab_z_ranges = slab_info.get('slab_z_ranges', {})
    discontinuity_regions = slab_info.get('discontinuity_regions', [])
    expected_layer_spacing = slab_info.get('expected_layer_spacing', 7.0)
    
    # Check if the structure is PBC-connected (bulk)
    # For bulk structures, boundary gaps connect through PBC
    cell_z = cell[2, 2]
    is_pbc_connected = False
    if slab_z_ranges:
        all_z_min = min(z[0] for z in slab_z_ranges.values())
        all_z_max = max(z[1] for z in slab_z_ranges.values())
        gap_at_bottom = all_z_min
        gap_at_top = cell_z - all_z_max
        total_boundary_gap = gap_at_bottom + gap_at_top
        # If total gap ≈ expected layer spacing, it's PBC-connected
        is_pbc_connected = abs(total_boundary_gap - expected_layer_spacing) < expected_layer_spacing * 0.3
    
    # For PBC-connected (bulk) structures, ALL molecules are A-sites
    if is_pbc_connected:
        for mol in molecules:
            mol.info['classification'] = 'a_site'
            mol.info['region'] = 'pbc_connected'
            a_sites.append(mol)
        return spacers, a_sites
    
    # Calculate tolerance for being "within" a slab
    # A molecule is in a slab if its center is within the slab's z-range
    # plus/minus a tolerance (about 0.5 × B-X distance)
    z_tolerance = expected_layer_spacing * 0.25
    
    for mol in molecules:
        original_indices = mol.info.get('original_indices', [])
        if not original_indices:
            continue
        
        # Calculate molecule center z-coordinate
        mol_z_coords = [atom_positions[idx][2] for idx in original_indices]
        mol_center_z = np.mean(mol_z_coords)
        
        # Check if molecule is in a discontinuity region (spacer)
        is_in_discontinuity = False
        for z_start, z_end in discontinuity_regions:
            if z_start - z_tolerance <= mol_center_z <= z_end + z_tolerance:
                is_in_discontinuity = True
                break
        
        # Check if molecule is within any slab's z-range (A-site)
        is_in_slab = False
        for slab_id, (z_min, z_max) in slab_z_ranges.items():
            if z_min - z_tolerance <= mol_center_z <= z_max + z_tolerance:
                is_in_slab = True
                break
        
        # Also check for discontinuity at cell boundaries (PBC)
        # If there are slabs and gaps at cell boundaries
        if slab_z_ranges and not is_in_discontinuity and not is_in_slab:
            all_z_min = min(z[0] for z in slab_z_ranges.values())
            all_z_max = max(z[1] for z in slab_z_ranges.values())
            cell_z = cell[2, 2]
            
            # Check if molecule is in vacuum/gap region at cell boundaries
            if mol_center_z < all_z_min - z_tolerance or mol_center_z > all_z_max + z_tolerance:
                # Check if this gap is large enough to be interlayer
                gap_at_bottom = all_z_min
                gap_at_top = cell_z - all_z_max
                
                if gap_at_bottom > expected_layer_spacing * 0.5:
                    if mol_center_z < all_z_min:
                        is_in_discontinuity = True
                if gap_at_top > expected_layer_spacing * 0.5:
                    if mol_center_z > all_z_max:
                        is_in_discontinuity = True
        
        # Classify the molecule
        if is_in_discontinuity:
            mol.info['classification'] = 'spacer'
            mol.info['region'] = 'discontinuity'
            spacers.append(mol)
        elif is_in_slab:
            mol.info['classification'] = 'a_site'
            mol.info['region'] = 'continuity'
            a_sites.append(mol)
        else:
            # Default: if unclear, check if it's a known A-site element/molecule
            formula = mol.get_chemical_formula(mode='hill')
            known_a_sites = {'Cs', 'Rb', 'K', 'Na', 'CH6N', 'CH5N2', 'H4N'}
            if formula in known_a_sites or len(mol) <= 8:
                mol.info['classification'] = 'a_site'
                mol.info['region'] = 'unknown'
                a_sites.append(mol)
            else:
                mol.info['classification'] = 'spacer'
                mol.info['region'] = 'unknown'
                spacers.append(mol)
    
    return spacers, a_sites



def _find_molecular_components(
    atom_positions: np.ndarray,
    atom_symbols: list,
    cell: np.ndarray,
    graph: nx.Graph,
    atoms_in_octahedra: set,
    full_atoms: Atoms = None,
) -> list:
    """
    Find all molecular components that are not part of the inorganic framework.

    Uses pymatgen's CovalentBondNN for proper covalent bond detection, which
    prevents incorrectly merging molecules that are spatially close but not
    covalently bonded (e.g., via hydrogen bonds).

    This includes:
    - Organic molecules (C, N, H, O, S, P based)
    - Potential atomic A-sites/spacers (Cs, Rb, K, etc.)

    Parameters
    ----------
    atom_positions : np.ndarray
        Array of atom positions
    atom_symbols : list
        List of atomic symbols
    cell : np.ndarray
        Unit cell matrix
    graph : nx.Graph
        Structural connectivity graph (used for attachment detection)
    atoms_in_octahedra : set
        Indices of atoms in octahedra (framework)
    full_atoms : ase.Atoms, optional
        Full ASE Atoms object (if not provided, will be constructed)

    Returns
    -------
    list
        List of ASE Atoms objects representing molecules
    """
    molecules = []
    processed = set()

    # Elements that form organic molecules
    organic_elements = {'C', 'N', 'H', 'O', 'S', 'P'}

    # Potential A-site/spacer elements
    ionic_elements = {'Cs', 'Rb', 'K', 'Na', 'Li', 'Ba', 'Sr', 'Ca'}

    # Build full Atoms object if not provided
    if full_atoms is None:
        full_atoms = Atoms(
            symbols=atom_symbols,
            positions=atom_positions,
            cell=cell,
            pbc=True
        )

    # Use pymatgen's CovalentBondNN to find molecular components
    # This properly identifies covalent bonds without merging H-bonded molecules
    molecular_components = extract_molecular_components(
        full_atoms,
        exclude_indices=atoms_in_octahedra,
        organic_elements=organic_elements,
    )

    for component_indices, mol in molecular_components:
        processed.update(component_indices)
        indices = list(component_indices)

        mol.info['mol_type'] = 'organic'

        # Determine attachment count (N atoms connected to framework via H-bonds)
        # We check the original graph for hydrogen bond connections
        n_attachments = 0
        for idx in indices:
            if atom_symbols[idx] == 'N':
                atom_node = f'atom_{idx}'
                if graph.has_node(atom_node):
                    for neighbor in graph.neighbors(atom_node):
                        if neighbor.startswith('atom_'):
                            neighbor_idx = int(neighbor.replace('atom_', ''))
                            if neighbor_idx in atoms_in_octahedra:
                                n_attachments += 1
                                break

        mol.info['n_attachments'] = n_attachments
        if n_attachments >= 2:
            mol.info['spacer_type'] = 'dj'
        elif n_attachments == 1:
            mol.info['spacer_type'] = 'rp'
        else:
            mol.info['spacer_type'] = None

        molecules.append(mol)

    # Find ionic (atomic) components - these don't form covalent bonds
    for i, symbol in enumerate(atom_symbols):
        if i in processed or i in atoms_in_octahedra:
            continue

        if symbol in ionic_elements:
            processed.add(i)
            mol = Atoms(
                symbols=[symbol],
                positions=[atom_positions[i]],
            )
            mol.info['original_indices'] = [i]
            mol.info['mol_type'] = 'atomic'
            mol.info['n_attachments'] = 0
            mol.info['spacer_type'] = None
            molecules.append(mol)

    return molecules

def _extract_spacer_molecules(
    atom_positions: np.ndarray,
    atom_symbols: list,
    cell: np.ndarray,
    atoms_in_octahedra: set,
    graph: nx.Graph,
    layers: dict = None,
    octahedra_info: list = None,
    shared_atoms: dict = None,
) -> list:
    """
    Extract spacer molecules from atoms not in octahedra.

    Uses pymatgen's CovalentBondNN for proper covalent bond detection,
    which prevents incorrectly merging molecules that are spatially close.
    Also detects atomic spacers (like Cs in interlayer regions).

    Parameters
    ----------
    atom_positions : np.ndarray
        Array of atom positions
    atom_symbols : list
        List of atomic symbols
    atoms_in_octahedra : set
        Set of atom indices that are part of octahedra
    graph : nx.Graph
        The structural connectivity graph (used for attachment detection)
    layers : dict, optional
        Layer information for slab analysis
    octahedra_info : list, optional
        List of octahedra information
    shared_atoms : dict, optional
        Shared atoms between octahedra

    Returns
    -------
    list of ase.Atoms
        List of spacer molecules, each as an Atoms object
    """
    # Core organic elements (atoms that MUST be present to form a spacer)
    core_organic_elements = {'C', 'N', 'H', 'O', 'S', 'P'}

    # Initialize lists for organic spacers/A-sites
    spacer_molecules = []
    a_site_molecules = []

    # Build full Atoms object for pymatgen
    full_atoms = Atoms(
        symbols=atom_symbols,
        positions=atom_positions,
        cell=cell,
        pbc=True
    )

    # Use pymatgen's CovalentBondNN to build molecular graph
    # This properly identifies covalent bonds without merging H-bonded molecules
    mol_graph = build_molecular_graph(full_atoms, atoms_in_octahedra)

    # Filter to only organic atoms
    organic_indices = set()
    for i, symbol in enumerate(atom_symbols):
        if i not in atoms_in_octahedra and symbol in core_organic_elements:
            organic_indices.add(i)

    if organic_indices:
        # Get subgraph of organic atoms
        organic_subgraph = mol_graph.subgraph(organic_indices).copy()

        # Find connected components (separate molecules)
        for component in nx.connected_components(organic_subgraph):
            component_indices = list(component)
            
            # Create Atoms object for this molecule
            mol_positions = atom_positions[component_indices]
            mol_symbols = [atom_symbols[i] for i in component_indices]
            
            mol = Atoms(symbols=mol_symbols, positions=mol_positions)
            mol.info['original_indices'] = component_indices
            
            # Find attachment atoms (atoms bonded to octahedra atoms)
            attachment_atoms = []
            attachment_nitrogens = []
            halide_elements = {'F', 'Cl', 'Br', 'I'}
            
            for idx in component_indices:
                atom_node = f'atom_{idx}'
                if not graph.has_node(atom_node):
                    continue
                    
                for neighbor in graph.neighbors(atom_node):
                    if neighbor.startswith('atom_'):
                        neighbor_idx = int(neighbor.replace('atom_', ''))
                        if neighbor_idx in atoms_in_octahedra:
                            attachment_atoms.append(idx)
                            # Check if this is a nitrogen (NH3+ attachment)
                            if atom_symbols[idx] == 'N':
                                attachment_nitrogens.append(idx)
                            break
            
            mol.info['attachment_atoms'] = attachment_atoms
            mol.info['attachment_nitrogens'] = attachment_nitrogens
            mol.info['n_attachments'] = len(attachment_nitrogens)
            mol.info['template_match'] = None
            
            # Classify molecule type based on formula and attachment pattern
            formula = mol.get_chemical_formula(mode='hill')
            n_atoms = len(mol)
            n_attach = len(attachment_nitrogens)
            
            # Determine if this is a spacer or A-site cation
            is_a_site = False
            
            # Check against known A-site patterns
            if formula in MOLECULAR_A_SITE_PATTERNS:
                mol.info['template_match'] = MOLECULAR_A_SITE_PATTERNS[formula]
                # Small molecular cations without bridging attachments are A-sites
                if n_attach <= 1 and n_atoms <= 10:
                    is_a_site = True
            
            # DJ spacer: 2 attachment nitrogens (bifunctional diamine)
            # RP spacer: 1 attachment nitrogen (monofunctional amine)
            if n_attach >= 2:
                mol.info['spacer_type'] = 'dj'  # Dion-Jacobson type (bifunctional)
            elif n_attach == 1:
                mol.info['spacer_type'] = 'rp'  # Ruddlesden-Popper type (monofunctional)
            else:
                mol.info['spacer_type'] = None  # Unknown or A-site
            
            if is_a_site:
                mol.info['molecule_type'] = 'a_site'
                a_site_molecules.append(mol)
            else:
                mol.info['molecule_type'] = 'spacer'
                spacer_molecules.append(mol)
        
        # Try to match spacers against known templates
        spacer_molecules = _match_spacers_to_templates(spacer_molecules)
        
        # Store A-site molecules in a module-level cache for later retrieval
        # (or add them to graph metadata)
        if a_site_molecules:
            for mol in a_site_molecules:
                mol.info['molecule_type'] = 'a_site'
    
    # === ATOMIC SPACER DETECTION ===
    # Detect atomic spacers (single atoms in interlayer region, like Cs in DJ)
    # These are atoms NOT in octahedra and NOT in organic molecules
    if layers and octahedra_info and shared_atoms is not None:
        # Analyze slab structure to find terminal octahedra
        slab_info = _analyze_slab_structure(
            layers, octahedra_info, shared_atoms, atom_positions
        )
        terminal_octahedra = slab_info.get('terminal_octahedra', set())
        z_levels = slab_info.get('z_levels', [])
        
        # Get atoms already accounted for
        accounted_atoms = set(atoms_in_octahedra)
        for mol in spacer_molecules + a_site_molecules:
            accounted_atoms.update(mol.info.get('original_indices', []))
        
        # Known A-site elements - these are NOT spacers if they're in a cavity
        a_site_elements = {'Cs', 'Rb', 'K', 'Na', 'Li', 'Ba', 'Sr', 'Ca'}
        
        # Check remaining atoms for interlayer position
        for i, symbol in enumerate(atom_symbols):
            if i in accounted_atoms:
                continue
            
            # Only consider potential spacer elements (alkali, alkaline earth)
            if symbol not in a_site_elements:
                continue
            
            # Check if this atom is in the interlayer region
            is_interlayer = _is_in_interlayer_region(
                atom_positions[i], octahedra_info, atom_positions,
                terminal_octahedra, cell, z_levels
            )
            
            if is_interlayer:
                # This is an atomic spacer (in interlayer region)
                atomic_spacer = Atoms(symbols=[symbol], positions=[atom_positions[i]])
                atomic_spacer.info['original_indices'] = [i]
                atomic_spacer.info['molecule_type'] = 'atomic_spacer'
                atomic_spacer.info['spacer_type'] = 'dj'  # Atomic spacers are typically DJ-type
                atomic_spacer.info['n_attachments'] = 0
                atomic_spacer.info['template_match'] = symbol
                spacer_molecules.append(atomic_spacer)
    
    # Return only true spacers (not A-sites)
    return spacer_molecules



def _match_spacers_to_templates(spacers: list) -> list:
    """
    Attempt to match spacer molecules against known templates.
    
    Parameters
    ----------
    spacers : list of ase.Atoms
        List of spacer molecules
        
    Returns
    -------
    list of ase.Atoms
        Same spacers with 'template_match' info populated if matched
    """
    # Known spacer patterns (formula -> template name)
    # These are common spacers in perovskite literature
    known_templates = {
        'CH6N': 'MA',  # Methylammonium
        'CH6N2': 'FA',  # Formamidinium (approximate)
        'C2H8N': 'EA',  # Ethylammonium
        'C3H10N': 'PA',  # Propylammonium
        'C4H12N': 'BA',  # Butylammonium
        'C6H16N': 'HA',  # Hexylammonium
        'C8H20N': 'OA',  # Octylammonium
        'C4H12N2': 'BDA',  # 1,4-butanediammonium
        'C6H8N': 'PEA',  # Phenethylammonium (approximate)
        'H4N': 'NH4',  # Ammonium
        'H3N': 'NH3',  # Ammonia
    }
    
    for spacer in spacers:
        formula = spacer.get_chemical_formula(mode='hill')
        
        # Direct match
        if formula in known_templates:
            spacer.info['template_match'] = known_templates[formula]
        else:
            # Try to identify by N count and C chain
            symbols = spacer.get_chemical_symbols()
            n_count = symbols.count('N')
            c_count = symbols.count('C')
            
            if n_count == 1 and c_count > 0:
                spacer.info['template_match'] = f'C{c_count}_amine'
            elif n_count == 2 and c_count > 0:
                spacer.info['template_match'] = f'C{c_count}_diamine'
            elif n_count > 0:
                spacer.info['template_match'] = 'amine_unknown'
    
    return spacers



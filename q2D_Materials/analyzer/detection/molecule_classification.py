"""
Molecular component identification and classification.

This module provides TWO complementary molecular extraction approaches:

1. **extract_molecular_components()** (analyzer/utils/pymatgen_utils.py):
   - **Purpose**: Generic graph-based molecular extraction
   - **Use case**: Extract molecules from any structure type
   - **Method**: Pure connectivity analysis using covalent bonds
   - **Key features**:
     * Works for bulk, surfaces, clusters, any structure
     * No assumptions about layer structure
     * Uses unified bond detection from atomic_properties
   - **When to use**: General-purpose molecular extraction, unknown structure types

2. **_find_molecular_components() + _classify_molecules_by_continuity()** (this file):
   - **Purpose**: Layer-aware extraction for 2D perovskites
   - **Use case**: Classify molecules in DJ, RP, monolayer structures
   - **Method**: Distinguishes spacers vs A-sites by layer position
   - **Key features**:
     * Structure-type specific (knows about slabs/layers)
     * Classifies based on z-position relative to inorganic layers
     * Handles PBC-connected molecules across boundaries
   - **When to use**: Classified 2D perovskites (after structure_classification)

Usage Guidelines
----------------

Use **extract_molecular_components()** when:
- Working with unknown/unclassified structures
- Need simple molecular extraction without classification
- Building molecular graphs for generic analysis
- Don't care about spacer vs A-site distinction

Use **_classify_molecules_by_continuity()** when:
- Structure is already classified as DJ/RP/monolayer
- Need to distinguish spacers from A-site cations
- Building layer-specific analysis
- Working with identified slab information

Relationship
------------
Both methods use the same underlying bond detection (unified in atomic_properties),
but differ in how they interpret and classify the resulting molecular components.

Example
-------
>>> from q2D_Materials.analyzer.utils.pymatgen_utils import extract_molecular_components
>>> 
>>> # Generic extraction (any structure)
>>> exclude_indices = set(octahedral_atoms)  # Exclude inorganic framework
>>> molecules = extract_molecular_components(atoms, exclude_indices)
>>> 
>>> # For classified 2D perovskites, use the layer-aware classifier
>>> from q2D_Materials.analyzer.detection.molecule_classification import _classify_molecules_by_continuity
>>> spacers, a_sites = _classify_molecules_by_continuity(molecules, slab_info, positions, cell)
"""

import numpy as np
import networkx as nx
from collections import deque

from ase import Atoms

from ...utils.geometry.geometry import _calculate_distances
from ..utils.pymatgen_utils import build_molecular_graph, extract_molecular_components
from .cavity_tracing import is_point_in_cavity, get_octahedra_data


def _classify_molecules_by_continuity(
    molecules: list,
    slab_info: dict,
    atom_positions: np.ndarray,
    cell: np.ndarray,
    cavities: list = None,
    graph: nx.Graph = None,
) -> tuple:
    """Classify molecules as spacers or A-sites based on slab continuity.

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
    cavities : list, optional
        List of cavity data dictionaries (for geometric containment check)
    graph : nx.Graph, optional
        Structural graph (needed for octahedra data if cavities provided)

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

    cell_z = cell[2, 2]
    is_pbc_connected = False
    if slab_z_ranges:
        all_z_min = min(z[0] for z in slab_z_ranges.values())
        all_z_max = max(z[1] for z in slab_z_ranges.values())
        gap_at_bottom = all_z_min
        gap_at_top = cell_z - all_z_max
        total_boundary_gap = gap_at_bottom + gap_at_top
        is_pbc_connected = abs(total_boundary_gap - expected_layer_spacing) < expected_layer_spacing * 0.3

    if is_pbc_connected:
        for mol in molecules:
            mol.info['classification'] = 'a_site'
            mol.info['region'] = 'pbc_connected'
            a_sites.append(mol)
        return spacers, a_sites

    z_tolerance = expected_layer_spacing * 0.25

    octahedra_data = None
    if cavities and graph:
        octahedra_data = get_octahedra_data(graph)

    for mol in molecules:
        original_indices = mol.info.get('original_indices', [])
        if not original_indices:
            continue

        mol_z_coords = [atom_positions[idx][2] for idx in original_indices]
        mol_center_z = np.mean(mol_z_coords)
        
        # Check cavity containment if available (strongest evidence for A-site)
        is_in_cavity = False
        if cavities and octahedra_data:
            mol_center = np.mean(atom_positions[original_indices], axis=0)
            for cavity in cavities:
                if is_point_in_cavity(mol_center, cavity, octahedra_data, atom_positions, cell):
                    is_in_cavity = True
                    break
        
        if is_in_cavity:
            mol.info['classification'] = 'a_site'
            mol.info['region'] = 'cavity'
            a_sites.append(mol)
            continue

        is_in_discontinuity = False
        for z_start, z_end in discontinuity_regions:
            if z_start - z_tolerance <= mol_center_z <= z_end + z_tolerance:
                is_in_discontinuity = True
                break

        is_in_slab = False
        for slab_id, (z_min, z_max) in slab_z_ranges.items():
            if z_min - z_tolerance <= mol_center_z <= z_max + z_tolerance:
                is_in_slab = True
                break

        if slab_z_ranges and not is_in_discontinuity and not is_in_slab:
            all_z_min = min(z[0] for z in slab_z_ranges.values())
            all_z_max = max(z[1] for z in slab_z_ranges.values())
            cell_z = cell[2, 2]

            if mol_center_z < all_z_min - z_tolerance or mol_center_z > all_z_max + z_tolerance:
                gap_at_bottom = all_z_min
                gap_at_top = cell_z - all_z_max

                if gap_at_bottom > expected_layer_spacing * 0.5:
                    if mol_center_z < all_z_min:
                        is_in_discontinuity = True
                if gap_at_top > expected_layer_spacing * 0.5:
                    if mol_center_z > all_z_max:
                        is_in_discontinuity = True

        if is_in_discontinuity:
            mol.info['classification'] = 'spacer'
            mol.info['region'] = 'discontinuity'
            spacers.append(mol)
        elif is_in_slab:
            mol.info['classification'] = 'a_site'
            mol.info['region'] = 'continuity'
            a_sites.append(mol)
        else:
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
    """Find all molecular components not part of the inorganic framework.

    Uses pymatgen's CovalentBondNN for proper covalent bond detection.
    Includes organic molecules and potential atomic A-sites/spacers.

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

    organic_elements = {'C', 'N', 'H', 'O', 'S', 'P'}
    ionic_elements = {'Cs', 'Rb', 'K', 'Na', 'Li', 'Ba', 'Sr', 'Ca'}

    if full_atoms is None:
        full_atoms = Atoms(
            symbols=atom_symbols,
            positions=atom_positions,
            cell=cell,
            pbc=True
        )

    molecular_components = extract_molecular_components(
        full_atoms,
        exclude_indices=atoms_in_octahedra,
        organic_elements=organic_elements,
    )

    for component_indices, mol in molecular_components:
        processed.update(component_indices)
        indices = list(component_indices)

        mol.info['mol_type'] = 'organic'

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

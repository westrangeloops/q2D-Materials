"""
Structure type classification for perovskites.

This module infers the structure type (bulk, DJ, RP, monolayer) based on
slab analysis, spacer detection, and graph patterns.

Functions
---------
_infer_structure_type_from_graph
    Infer structure type using graph-based slab analysis (primary method)
_infer_structure_type
    Infer structure type using legacy layer-based analysis
"""

import numpy as np
import networkx as nx


def _infer_structure_type_from_graph(
    slab_info: dict,
    spacers: list,
    bx_graph: nx.Graph,
    cell: np.ndarray,
) -> str:
    """
    Infer structure type from the B-X network graph and slab information.
    
    - bulk: Single continuous slab with no discontinuities, PBC-connected
    - dj: Multiple slabs with spacers in discontinuity regions (bifunctional)
    - rp: Multiple slabs with spacers in discontinuity regions (monofunctional)
    - monolayer: Single slab with vacuum (no PBC connectivity in z)
    
    Parameters
    ----------
    slab_info : dict
        Output from _identify_slabs_by_continuity
    spacers : list
        List of classified spacer molecules
    bx_graph : nx.Graph
        B-X network graph
    cell : np.ndarray
        Unit cell matrix
        
    Returns
    -------
    str
        One of: 'bulk', 'dj', 'rp', 'monolayer', 'unknown'
    """
    slabs = slab_info.get('slabs', {})
    slab_z_ranges = slab_info.get('slab_z_ranges', {})
    discontinuity_regions = slab_info.get('discontinuity_regions', [])
    expected_layer_spacing = slab_info.get('expected_layer_spacing', 7.0)
    
    n_slabs = len(slabs)
    n_spacers = len(spacers)
    
    if n_slabs == 0:
        return 'unknown'
    
    # Check for vacuum at cell boundaries
    cell_z = cell[2, 2]
    if slab_z_ranges:
        all_z_min = min(z[0] for z in slab_z_ranges.values())
        all_z_max = max(z[1] for z in slab_z_ranges.values())
        gap_at_bottom = all_z_min
        gap_at_top = cell_z - all_z_max
        
        # For bulk structures with PBC, the gap at bottom + gap at top
        # should equal approximately one layer spacing (they connect)
        # If total_boundary_gap ≈ expected_layer_spacing, it's PBC-connected (bulk)
        total_boundary_gap = gap_at_bottom + gap_at_top
        is_pbc_connected = abs(total_boundary_gap - expected_layer_spacing) < expected_layer_spacing * 0.3
        
        # Large gap at boundaries indicates 2D structure (not bulk)
        # But only if it's NOT a PBC-connected gap
        if is_pbc_connected:
            # PBC-connected: this is bulk, not 2D
            has_vacuum = False
        else:
            has_vacuum = (gap_at_bottom > expected_layer_spacing * 0.5 or 
                          gap_at_top > expected_layer_spacing * 0.5)
    else:
        has_vacuum = False
        is_pbc_connected = False
    
    # Check for monolayer: single slab with significant vacuum
    # Monolayer has vacuum > 2x expected layer spacing on at least one side
    is_monolayer = False
    if n_slabs == 1 and has_vacuum:
        if slab_z_ranges:
            all_z_min = min(z[0] for z in slab_z_ranges.values())
            all_z_max = max(z[1] for z in slab_z_ranges.values())
            gap_at_bottom = all_z_min
            gap_at_top = cell_z - all_z_max
            slab_thickness = all_z_max - all_z_min
            
            # If vacuum (total gap) is much larger than slab thickness, it's monolayer
            # Monolayer typically has vacuum >> slab, not just gaps between layers
            total_gap = gap_at_bottom + gap_at_top
            if total_gap > expected_layer_spacing * 1.5:
                is_monolayer = True
    
    if is_monolayer:
        return 'monolayer'
    
    # Classification logic
    if n_slabs == 1 and n_spacers == 0:
        if has_vacuum:
            return 'monolayer'
        else:
            return 'bulk'
    
    elif n_spacers > 0:
        # Check for organic spacers vs atomic (passivator) "spacers"
        # If all spacers are atomic (single atoms) and vacuum is present, likely monolayer
        organic_spacers = [s for s in spacers if s.info.get('mol_type') != 'atomic']
        atomic_spacers = [s for s in spacers if s.info.get('mol_type') == 'atomic']
        
        # If only atomic "spacers" and significant vacuum, it's likely monolayer
        # (the "spacers" are actually passivators or uncoordinated atoms)
        if len(organic_spacers) == 0 and has_vacuum and n_slabs == 1:
            return 'monolayer'
        
        # Has real organic spacers - determine DJ vs RP based on spacer type
        # DJ: bifunctional spacers (2 attachment points)
        # RP: monofunctional spacers (1 attachment point)
        
        dj_count = 0
        rp_count = 0
        
        for spacer in spacers:
            n_attach = spacer.info.get('n_attachments', 0)
            spacer_type = spacer.info.get('spacer_type', '')
            
            if spacer_type == 'dj' or n_attach >= 2:
                dj_count += 1
            elif spacer_type == 'rp' or n_attach == 1:
                rp_count += 1
            else:
                # Atomic spacers need different treatment
                if spacer.info.get('mol_type') == 'atomic':
                    # Atomic in interlayer = DJ-like
                    dj_count += 1
                else:
                    rp_count += 1
        
        if dj_count > rp_count:
            return 'dj'
        elif rp_count > dj_count:
            return 'rp'
        else:
            # Default to DJ if equal or unknown
            return 'dj' if dj_count > 0 else 'rp'
    
    elif n_slabs > 1:
        # Multiple slabs but no spacers detected
        # This might be a DJ/RP with atomic spacers not yet identified
        if has_vacuum or len(discontinuity_regions) > 0:
            return 'dj'  # Default to DJ for multi-slab structures
        else:
            return 'bulk'
    
    return 'unknown'

def _infer_structure_type(
    graph: nx.Graph,
    layers: dict,
    spacers: list,
    octahedra: list,
    cell: np.ndarray,
    a_sites: list = None,
) -> str:
    """
    Infer structure type from graph patterns and spacer attachment analysis.
    
    Structure types:
    - 'bulk': 3D continuous, no spacers, corner-sharing in all directions
    - 'dj' (Dion-Jacobson): bifunctional spacers (2 NH3+ groups bridging slabs)
    - 'rp' (Ruddlesden-Popper): monofunctional spacers (1 NH3+ group each, paired)
    - 'monolayer': 1-2 slabs with vacuum, passivators on surface
    
    Key distinction:
    - DJ spacers: [NH3+]-R-[NH3+] → single molecule connects both slab faces
    - RP spacers: [NH3+]-R → two molecules face each other between slabs
    
    Parameters
    ----------
    graph : nx.Graph
        The structural connectivity graph
    layers : dict
        Layer information
    spacers : list
        List of spacer molecules
    octahedra : list
        List of octahedra information
    cell : np.ndarray
        Unit cell matrix
    a_sites : list, optional
        List of A-site cations (for atomic spacer detection)
        
    Returns
    -------
    str
        One of: 'bulk', 'dj', 'rp', 'monolayer', 'unknown'
    """
    n_layers = len(layers)
    n_spacers = len(spacers)
    n_octahedra = len(octahedra)
    
    if n_octahedra == 0:
        return 'unknown'
    
    # Check for vacuum (large c-axis compared to actual content)
    c_length = np.linalg.norm(cell[2])
    
    # Estimate structure extent in z-direction
    z_coords = []
    for oct in octahedra:
        if oct.get('central_atom_index') is not None:
            # Get z-coordinate from graph
            atom_node = f"atom_{oct['central_atom_index']}"
            if atom_node in graph.nodes:
                node_data = graph.nodes[atom_node]
                coords = node_data.get('direct_coordinates', [0, 0, 0])
                z_coords.append(coords[2] if len(coords) > 2 else 0)
    
    has_vacuum = False
    if z_coords:
        z_range = max(z_coords) - min(z_coords)
        # If z-range is much less than cell height, there's vacuum
        if c_length > 15 and z_range < c_length * 0.6:
            has_vacuum = True
    
    # Count surface vs central layers
    n_surface = sum(1 for l in layers.values() if l.get('position') == 'surface')
    n_central = sum(1 for l in layers.values() if l.get('position') == 'central')
    
    # No spacers case
    if n_spacers == 0:
        # Check for atomic spacers in interlayer positions
        # These might be Cs or other atoms used as spacers
        has_atomic_spacer = False
        if a_sites:
            # Count A-sites that are NOT in-plane with octahedra (interlayer)
            atomic_a_sites = [a for a in a_sites if not a.get('is_molecular', False)]
            if len(atomic_a_sites) > 0:
                # Check if any atomic A-site is at interlayer position
                # For now, assume they are in the cavity (not spacers)
                pass
        
        if has_vacuum:
            return 'monolayer'
        else:
            return 'bulk'
    
    # Has organic spacers -> analyze attachment patterns
    # Count spacer types based on nitrogen attachments
    dj_type_spacers = 0  # Bifunctional: 2+ attachment nitrogens
    rp_type_spacers = 0  # Monofunctional: 1 attachment nitrogen
    unknown_spacers = 0
    
    for spacer in spacers:
        spacer_type = spacer.info.get('spacer_type')
        n_attach = spacer.info.get('n_attachments', 0)
        attachment_n = spacer.info.get('attachment_nitrogens', [])
        
        # Use explicit spacer_type if available
        if spacer_type == 'dj':
            dj_type_spacers += 1
        elif spacer_type == 'rp':
            rp_type_spacers += 1
        # Fallback to counting attachments
        elif n_attach >= 2 or len(attachment_n) >= 2:
            dj_type_spacers += 1
        elif n_attach == 1 or len(attachment_n) == 1:
            rp_type_spacers += 1
        else:
            # Check for diamine pattern in formula
            symbols = spacer.get_chemical_symbols()
            n_count = symbols.count('N')
            if n_count >= 2:
                dj_type_spacers += 1  # Assume DJ for diamines
            elif n_count == 1:
                rp_type_spacers += 1
            else:
                unknown_spacers += 1
    
    # Determine structure type based on spacer patterns
    total_classified = dj_type_spacers + rp_type_spacers
    
    if total_classified == 0:
        # Couldn't classify spacers - use fallback logic
        if has_vacuum:
            return 'monolayer'
        else:
            return 'bulk'
    
    # DJ vs RP determination
    # DJ: Predominantly bifunctional spacers (diamines like BDA, EDA)
    # RP: Predominantly monofunctional spacers (monoamines like BA, PEA)
    
    dj_fraction = dj_type_spacers / total_classified
    rp_fraction = rp_type_spacers / total_classified
    
    if dj_fraction > 0.7:
        return 'dj'
    elif rp_fraction > 0.7:
        return 'rp'
    elif dj_fraction > rp_fraction:
        # Mixed but more DJ
        return 'dj'
    else:
        # Mixed but more RP, or equal
        return 'rp'

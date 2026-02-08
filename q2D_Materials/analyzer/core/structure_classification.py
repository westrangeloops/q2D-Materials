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
    cell: np.ndarray,
    graph: nx.Graph = None,
    octahedra: list = None,
) -> str:
    """
    Infer structure type from slab information.
    
    Classification priority:
    1. Check for octahedra - bulk needs at least one
    2. Check for terminal X atoms - monolayer should have no terminal X
    3. Classify DJ/RP based on NH3 count in spacers (2+ NH3 = DJ, 1 NH3 = RP)
    4. Default to bulk if not others
    
    Parameters
    ----------
    slab_info : dict
        Output from _identify_slabs_by_continuity
    spacers : list
        List of classified spacer molecules
    cell : np.ndarray
        Unit cell matrix
    graph : nx.Graph, optional
        Structural graph (needed for terminal X detection)
    octahedra : list, optional
        List of octahedra information (needed for bulk check)
        
    Returns
    -------
    str
        One of: 'bulk', 'dj', 'rp', 'monolayer', 'unknown'
    """
    # Step 1: Check for octahedra - bulk needs at least one
    n_octahedra = 0
    if octahedra:
        n_octahedra = len(octahedra)
    elif graph:
        # Count octahedra from graph
        for node in graph.nodes():
            if graph.nodes[node].get('node_type') == 'octahedron':
                n_octahedra += 1
    
    # If no octahedra, cannot be bulk
    if n_octahedra == 0:
        return 'unknown'
    
    # Step 2: Check for terminal X atoms
    # Monolayer structures have terminal X atoms (passivators on surface)
    # Bulk structures typically have no terminal X (all X atoms are shared between octahedra)
    has_terminal_x = False
    if graph:
        from .graph_construction import get_terminal_atoms
        terminal_atoms = get_terminal_atoms(graph)
        has_terminal_x = len(terminal_atoms) > 0
    
    # If no terminal X atoms and has octahedra, it's likely bulk
    # (all X atoms are shared between octahedra in 3D network)
    if not has_terminal_x and n_octahedra > 0:
        # No terminal X and has octahedra = bulk (will be confirmed by other checks)
        pass  # Continue with other checks
    
    slabs = slab_info.get('slabs', {})
    slab_z_ranges = slab_info.get('slab_z_ranges', {})
    discontinuity_regions = slab_info.get('discontinuity_regions', [])
    expected_layer_spacing = slab_info.get('expected_layer_spacing', 7.0)
    
    n_slabs = len(slabs)
    n_spacers = len(spacers)
    
    if n_slabs == 0:
        # No slabs but has octahedra - default to bulk
        return 'bulk' if n_octahedra > 0 else 'unknown'
    
    # Check for vacuum at cell boundaries and PBC connectivity
    cell_z = cell[2, 2]
    is_pbc_connected = False
    has_vacuum = False
    
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
    
    # PRIORITY: If PBC-connected and has octahedra, it's bulk (regardless of other factors)
    if is_pbc_connected and n_octahedra > 0:
        return 'bulk'
    
    # Check for monolayer: single slab with significant vacuum AND terminal X atoms
    # Monolayer has vacuum > 2x expected layer spacing AND terminal X atoms (passivators)
    is_monolayer = False
    if n_slabs == 1 and has_vacuum and has_terminal_x:
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
        # Step 3: Classify DJ/RP based on NH3 count in spacers
        # DJ: 2+ NH3 groups (bifunctional)
        # RP: 1 NH3 group (monofunctional)
        
        dj_count = 0
        rp_count = 0
        
        for spacer in spacers:
            # Check for NH3 count in spacer info or graph
            nh3_count = 0
            
            # Try to get NH3 count from spacer info
            if hasattr(spacer, 'info'):
                nh3_count = spacer.info.get('nh3_count', 0)
                # Also check n_attachments as fallback
                if nh3_count == 0:
                    n_attach = spacer.info.get('n_attachments', 0)
                    if n_attach >= 2:
                        nh3_count = 2  # Assume DJ
                    elif n_attach == 1:
                        nh3_count = 1  # Assume RP
            
            # If still no count, try to count from graph
            if nh3_count == 0 and graph:
                # Find spacer node in graph
                spacer_symbols = spacer.get_chemical_symbols()
                n_count = spacer_symbols.count('N')
                # Rough estimate: if 2+ N atoms, likely DJ; 1 N atom, likely RP
                if n_count >= 2:
                    nh3_count = 2
                elif n_count == 1:
                    nh3_count = 1
            
            # Classify based on NH3 count
            if nh3_count >= 2:
                dj_count += 1
            elif nh3_count == 1:
                rp_count += 1
            else:
                # Unknown - check if atomic spacer
                if hasattr(spacer, 'info') and spacer.info.get('mol_type') == 'atomic':
                    # Atomic spacers default to DJ-like
                    dj_count += 1
                else:
                    # Default to RP for unknown
                    rp_count += 1
        
        if dj_count > rp_count:
            return 'dj'
        elif rp_count > dj_count:
            return 'rp'
        else:
            # Default to DJ if equal or unknown
            return 'dj' if dj_count > 0 else 'bulk'
    
    elif n_slabs > 1:
        # Multiple slabs but no spacers detected
        # If PBC-connected, it's bulk (already checked above)
        # Otherwise, check for vacuum/discontinuities
        if has_vacuum or (len(discontinuity_regions) > 0 and not is_pbc_connected):
            # Has vacuum or discontinuities and NOT PBC-connected = likely DJ/RP
            # But if no spacers, default to bulk if has octahedra
            return 'bulk' if n_octahedra > 0 else 'dj'
        else:
            # Multiple slabs but PBC-connected = bulk
            return 'bulk'
    
    # Step 4: Default to bulk if has octahedra and no other classification
    if n_octahedra > 0:
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

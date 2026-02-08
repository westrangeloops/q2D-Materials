"""Comprehensive graph ontology construction for 2D materials.

This module builds the complete structural connectivity graph including
octahedra, layers, atoms, molecules, and their relationships.

Graph Philosophy:
- Nodes store intrinsic properties only (what the entity IS)
- Edges express relationships (how entities connect)
- Frequently-used structural properties (is_terminal, is_equatorial, is_interlayer) 
  are computed once and stored on X-site atoms for performance
- No index lists in nodes - that's what edges are for

Coordinate System:
- Atom positions are stored as Cartesian coordinates (x, y, z) in graph nodes
- Cell matrix (3x3) is stored in graph metadata as G.graph['cell_matrix']
- For non-orthogonal cells, use get_cell_matrix(G) to access the cell matrix
- All distance calculations should use PBC-aware functions from utils.geometry
"""

import numpy as np
import networkx as nx
from ase import Atoms
import sys

from .layer_identification import _get_octahedron_layer, _identify_layers
from ..octahedral_processing.octahedral_detection import _count_octahedra, find_shared_atoms
from ..utils.pymatgen_utils import get_molecular_connections
from ..utils.clifford_embedding import embed_to_6d, clifford_distance, get_cell_lengths


def _compute_atom_structural_properties(G, atom_symbols):
    """Compute and store structural properties for all atoms.

    This function computes classification and structural properties for atoms
    based on their connectivity to octahedra and molecules in the graph.

    Properties computed:
    - is_X: True if atom is an X-site (ligand atom bonded to B-site)
    - is_terminal: True if X-atom belongs to exactly 1 octahedron (terminal/surface atom)
                   Note: Terminal X atoms form X-B-X angles (within octahedron) but NOT B-X-B angles
                   Only stored when True
    - is_equatorial: True if X-atom is in equatorial position of any octahedron
                     Note: Equatorial X atoms form intra-layer B-X-B angles
                     Only stored when True
    - is_axial: True if X-atom is in axial position of any octahedron
                Only stored when True
    - is_interlayer: True if X-atom is shared between 2 octahedra (axial bridging atom)
                     Note: Interlayer X atoms form inter-layer B-X-B angles
                     Only stored when True
    
    Important: Only equatorial and interlayer X atoms can form B-X-B angles (shared between 2 octahedra).
    Terminal X atoms only participate in X-B-X angles (within a single octahedron).
    
    Note: is_B is no longer stored - B atoms are identified by being connected to octahedra
    with role='center' via CONTAINS edges.

    Parameters
    ----------
    G : nx.Graph
        The structural graph with octahedra and atom nodes
    atom_symbols : list
        List of atomic symbols for all atoms
    """
    # First pass: Identify X-site atoms (ligands bonded to B-sites)
    x_site_atoms = set()
    # Find B atoms (connected to octahedra with role='center')
    b_atoms = set()
    for node in G.nodes():
        if node.startswith('octahedron_'):
            # Find the center atom (B-site) of this octahedron
            for neighbor in G.neighbors(node):
                edge_data = G.get_edge_data(node, neighbor)
                if (edge_data and
                    edge_data.get('edge_type') == 'contains' and
                    edge_data.get('role') == 'center'):
                    b_atoms.add(neighbor)
    
    # Find X atoms bonded to B atoms
    for b_atom in b_atoms:
        for neighbor in G.neighbors(b_atom):
            edge_data = G.get_edge_data(b_atom, neighbor)
            if (edge_data and
                edge_data.get('edge_type') == 'bonded_to' and
                edge_data.get('role') == 'ligand'):
                x_site_atoms.add(neighbor)

    # Second pass: Compute properties for each atom
    for node in G.nodes():
        if not node.startswith('atom_'):
            continue

        # Classify atom type from graph structure
        is_x = node in x_site_atoms

        # Store classification properties
        # Note: is_B is no longer stored - derive from octahedron connections if needed
        # Note: is_A and is_S removed - use A_Site/Spacer nodes instead
        G.nodes[node]['is_X'] = is_x

        # For X-site atoms, compute additional structural properties
        if is_x:
            # Find all octahedra containing this atom via B atoms
            # Path: X atom → B atom → Octahedron
            containing_octahedra = []
            is_equatorial = False
            is_axial = False

            # Find B atoms bonded to this X atom
            for neighbor in G.neighbors(node):
                edge_data = G.get_edge_data(node, neighbor)
                if (edge_data and
                    edge_data.get('edge_type') == 'bonded_to' and
                    edge_data.get('role') == 'ligand'):
                    # Check if this atom is equatorial or axial
                    geometry = edge_data.get('geometry')
                    if geometry == 'equatorial':
                        is_equatorial = True
                    elif geometry == 'axial':
                        is_axial = True
                    
                    # Find octahedra containing this B atom
                    for oct_neighbor in G.neighbors(neighbor):
                        if oct_neighbor.startswith('octahedron_'):
                            oct_edge_data = G.get_edge_data(oct_neighbor, neighbor)
                            if (oct_edge_data and
                                oct_edge_data.get('edge_type') == 'contains' and
                                oct_edge_data.get('role') == 'center'):
                                if oct_neighbor not in containing_octahedra:
                                    containing_octahedra.append(oct_neighbor)

            # Terminal: belongs to exactly 1 octahedron (forms X-B-X angles only, NOT B-X-B)
            is_terminal = (len(containing_octahedra) == 1)

            # Interlayer: shared between 2 octahedra (axial bridging, forms inter-layer B-X-B)
            is_interlayer = (len(containing_octahedra) == 2)

            # Store X-site specific properties only when True
            if is_terminal:
                G.nodes[node]['is_terminal'] = True  # X-B-X angles only
            if is_equatorial:
                G.nodes[node]['is_equatorial'] = True  # Intra-layer B-X-B angles
            if is_axial:
                G.nodes[node]['is_axial'] = True  # Axial position
            if is_interlayer:
                G.nodes[node]['is_interlayer'] = True  # Inter-layer B-X-B angles


def _graph_inorganic_ontology(
    atom_positions,
    atom_symbols,
    cell,
    cutoff_distance=4.0,
    min_tolerance=0.2,
    step=0.1,
    max_steps=20,
    non_metal_symbols=None,
    octahedra_edges=None,
    octahedra_centers=None,
    valid_halogen=None,
    valid_molecule=None,
):
    """Build comprehensive graph-based inorganic ontology for 2D materials.

    Graph structure hierarchy:
    1. Layer Nodes: Represent 2D slabs
    2. Octahedra Nodes: B-X6 coordination units
    3. Atom Nodes: Individual atoms with intrinsic properties
    4. Molecule Nodes: Organic molecules (A-site or spacer)

    Relationships are expressed through edges:
    - CONTAINS: Layer→Octahedron, Octahedron→Atom, Molecule→Atom
    - BONDED_TO: Atom→Atom (molecular bonds)

    Parameters
    ----------
    atom_positions : list
        List of [x, y, z] coordinates
    atom_symbols : list
        List of atomic symbols
    cell : np.ndarray
        3x3 array of unit cell vectors
    cutoff_distance : float
        Maximum distance for octahedral neighbors
    min_tolerance : float
        Starting bond length tolerance for octahedra detection
    step : float
        Increment step for tolerance optimization
    max_steps : int
        Maximum steps for tolerance optimization
    non_metal_symbols : list, optional
        List of non-metal symbols for filtering central atoms
    octahedra_edges : list, optional
        Valid edge counts (e.g., [4, 6])
    octahedra_centers : list, optional
        Valid B-site atoms (e.g., ['Pb', 'Sn'])
    valid_halogen : list, optional
        Valid halogen symbols
    valid_molecule : list, optional
        Valid molecule element symbols

    Returns
    -------
    networkx.Graph
        Complete structural graph with clean node/edge separation
    """
    if non_metal_symbols is None:
        non_metal_symbols = ['C', 'O', 'N', 'H', 'F', 'Cl', 'Br', 'I']

    G = nx.Graph()
    n_atoms_total = len(atom_positions)
    
    # Extract cell lengths for Clifford embedding (used for PBC-aware distances)
    cell_lengths = get_cell_lengths(cell)
    
    # Embed all atoms to 6D Clifford space (done once, for internal calculations)
    atom_positions_array = np.array(atom_positions)
    atom_6d_coords = embed_to_6d(atom_positions_array, cell_lengths)
    
    # Count octahedra using Clifford embedding
    octahedra_count, centers_positions, center_symbols, neighbor_indices, center_atom_indices, octahedra_geometries = _count_octahedra(
        atom_positions, atom_symbols, cutoff_distance, min_tolerance, step, max_steps, cell, non_metal_symbols,
        octahedra_edges=octahedra_edges,
        octahedra_centers=octahedra_centers,
        valid_halogen=valid_halogen,
        valid_molecule=valid_molecule,
    )
    
    shared_atoms = find_shared_atoms(neighbor_indices)
    
    layers, x_atom_classifications = _identify_layers(
        neighbor_indices,
        shared_atoms,
        atom_positions=atom_positions_array,
        center_atom_indices=center_atom_indices,
        cell=cell,
        octahedra_geometries=octahedra_geometries,
    )

    # Add Layer nodes with properties
    for layer_id, layer_info in layers.items():
        # Determine terminal atom count for this layer
        terminal_atoms_count = layer_info.get('terminal_atoms_count', 0)
        
        G.add_node(
            f'layer_{layer_id}',
            node_type='layer',
            z_coord=layer_info.get('z_coord'),
            position=layer_info.get('position', 'unknown'),
            terminal_atoms_count=terminal_atoms_count,
        )

    # Add Atom nodes with separate x, y, z coordinates
    for i, (pos, symbol) in enumerate(zip(atom_positions, atom_symbols)):
        pos_list = pos.tolist() if hasattr(pos, 'tolist') else list(pos)
        node_data = {
            'node_type': 'atom',
            'vasp_index': i,
            'symbol': symbol,
            'x': float(pos_list[0]),
            'y': float(pos_list[1]),
            'z': float(pos_list[2]),
        }
        G.add_node(f'atom_{i}', **node_data)

    # Add Octahedra nodes and their edges
    if octahedra_count > 0 and len(center_symbols) > 0 and len(neighbor_indices) > 0 and len(center_atom_indices) > 0:
        # Check if this is a 3D bulk structure (no terminal atoms)
        all_terminal_count = sum(
            1 for geom_dict in octahedra_geometries 
            for label in geom_dict.values() 
            if 'terminal' in label
        )
        is_bulk_3d = (all_terminal_count == 0)

        for i, (center_pos, center_sym, neighbors, center_idx) in enumerate(zip(
            centers_positions, center_symbols, neighbor_indices, center_atom_indices
        )):
            geometry_info = octahedra_geometries[i] if i < len(octahedra_geometries) else None
            
            if geometry_info is None:
                raise ValueError(
                    f"Missing geometry classification for octahedron {i} with center atom {center_idx} ({center_sym})"
                )
            
            # For 3D bulk structures, reclassify geometrically based on c-axis alignment.
            # Use minimum-image positions (PBC) so we get 6 distinct positions; wrapped positions
            # can duplicate the same atom and misclassify axial vs equatorial.
            if is_bulk_3d and len(neighbors) == 6:
                b_pos = np.array(atom_positions[center_idx])
                c_vec = np.array(cell[2], dtype=np.float64)
                c_norm = np.linalg.norm(c_vec)
                c_axis = c_vec / c_norm if c_norm >= 1e-10 else np.array([0.0, 0.0, 1.0])
                from ...utils.geometry.pbc_distances import find_nearest_image_positions
                halogens = valid_halogen if valid_halogen is not None else ['Cl', 'Br', 'I']
                all_x_indices = np.array([j for j, s in enumerate(atom_symbols) if s in halogens])
                all_x_positions = np.array(atom_positions)[all_x_indices]
                nearest_x_indices, nearest_x_positions, _, _ = find_nearest_image_positions(
                    reference_position=b_pos,
                    candidate_positions=all_x_positions,
                    candidate_indices=all_x_indices,
                    cell=cell,
                    n_neighbors=6,
                    pbc=True,
                )
                alignments = []
                for k in range(len(nearest_x_positions)):
                    vec = nearest_x_positions[k] - b_pos
                    norm = np.linalg.norm(vec) + 1e-9
                    alignment = np.abs(np.dot(vec / norm, c_axis))
                    alignments.append((k, nearest_x_indices[k], alignment))
                alignments.sort(key=lambda x: x[2], reverse=True)
                for k, nbr_idx, _ in alignments:
                    geometry_info[nbr_idx] = 'equatorial'
                for idx in range(min(2, len(alignments))):
                    geometry_info[alignments[idx][1]] = 'axial_interlayer'
            
            # Validate geometry counts
            counts = {
                'axial': sum(1 for v in geometry_info.values() if v in ['axial_terminal', 'axial_interlayer']),
                'equatorial': sum(1 for v in geometry_info.values() if v == 'equatorial'),
                'unknown': sum(1 for v in geometry_info.values() if v == 'unknown')
            }
            
            if counts['axial'] != 2 or counts['equatorial'] != 4:
                pass  # non-standard octahedron (e.g. edge-sharing, partial)

            layer_id = _get_octahedron_layer(i, layers)
            
            # Octahedron node - minimal, no index lists
            G.add_node(
                f'octahedron_{i}',
                node_type='octahedron',
            )
            
            # Layer CONTAINS Octahedron
            G.add_edge(f'layer_{layer_id}', f'octahedron_{i}', edge_type='contains')
            
            # Octahedron CONTAINS center atom (role='center')
            G.add_edge(
                f'octahedron_{i}',
                f'atom_{center_idx}',
                edge_type='contains',
                role='center'
            )
            
            # B atom BONDED_TO X atoms (role='ligand' with geometry)
            # Calculate distances and create B-X bonds
            b_atom_node = f'atom_{center_idx}'
            b_pos = np.array(atom_positions[center_idx])
            
            for neighbor_idx in neighbors:
                geometry_label = geometry_info.get(neighbor_idx, 'unknown')
                
                # Simplify geometry: axial_terminal/axial_interlayer → 'axial'
                if geometry_label in ['axial_terminal', 'axial_interlayer']:
                    simplified_geometry = 'axial'
                elif geometry_label == 'equatorial':
                    simplified_geometry = 'equatorial'
                else:
                    simplified_geometry = 'unknown'
                
                x_atom_node = f'atom_{neighbor_idx}'
                x_pos = np.array(atom_positions[neighbor_idx])
                
                # Calculate B-X distance with PBC
                from ...utils.geometry.pbc_distances import calculate_pbc_distances
                distance = float(calculate_pbc_distances(
                    b_pos[np.newaxis, :],
                    x_pos[np.newaxis, :],
                    cell,
                    pbc=True,
                    mode='auto'
                )[0])
                
                # Create bond type (canonical alphabetical order)
                b_symbol = center_sym
                x_symbol = atom_symbols[neighbor_idx]
                bond_type = f'{b_symbol}-{x_symbol}' if b_symbol <= x_symbol else f'{x_symbol}-{b_symbol}'
                
                # B atom BONDED_TO X atom
                G.add_edge(
                    b_atom_node,
                    x_atom_node,
                    edge_type='bonded_to',
                    role='ligand',
                    geometry=simplified_geometry,
                    bond_type=bond_type,
                    distance=distance
                )

    # Create molecular connections and molecule nodes
    _create_molecular_connections(G, atom_positions, atom_symbols, neighbor_indices, cell, valid_molecule=valid_molecule)

    # Create A_Site and Spacer nodes and classify them
    _create_a_site_spacer_nodes(G, atom_positions_array, atom_symbols, cell, neighbor_indices, valid_molecule=valid_molecule)

    # Compute and store atom classification properties (is_X, is_terminal, is_equatorial, is_interlayer)
    # Must be called AFTER A_Site/Spacer nodes are created
    _compute_atom_structural_properties(G, atom_symbols)
    
    # Recalculate terminal atoms count for each layer using graph-based method
    # This ensures consistency with get_terminal_atoms() function
    # Find terminal atoms: X atoms belonging to exactly 1 octahedron
    atom_octahedra_count = {}
    adjacency = dict(G.adjacency())
    
    # Count octahedra per X atom
    for node, data in G.nodes(data=True):
        if data.get('node_type') == 'atom':
            # Check if this is an X atom (bonded to B atoms with role='ligand')
            for neighbor, edge_data in adjacency.get(node, {}).items():
                if (edge_data.get('edge_type') == 'bonded_to' and
                    edge_data.get('role') == 'ligand'):
                    # This is an X atom, count octahedra via B atoms
                    b_atom = neighbor
                    for oct_neighbor, oct_edge_data in adjacency.get(b_atom, {}).items():
                        if (oct_neighbor.startswith('octahedron_') and
                            oct_edge_data.get('edge_type') == 'contains' and
                            oct_edge_data.get('role') == 'center'):
                            atom_octahedra_count[node] = atom_octahedra_count.get(node, 0) + 1
                    break  # Only count once per X atom
    
    # Terminal atoms are those belonging to exactly 1 octahedron
    terminal_atom_node_ids = [atom_id for atom_id, count in atom_octahedra_count.items() if count == 1]
    
    # Build mapping: terminal atom -> layers via octahedra
    terminal_atom_to_layers = {}
    
    for terminal_atom_id in terminal_atom_node_ids:
        terminal_atom_to_layers[terminal_atom_id] = set()
        # Find octahedra containing this terminal atom via B atoms
        for neighbor, edge_data in adjacency.get(terminal_atom_id, {}).items():
            if (edge_data.get('edge_type') == 'bonded_to' and
                edge_data.get('role') == 'ligand'):
                b_atom = neighbor
                # Find octahedra containing this B atom
                for oct_neighbor, oct_edge_data in adjacency.get(b_atom, {}).items():
                    if (oct_neighbor.startswith('octahedron_') and
                        oct_edge_data.get('edge_type') == 'contains' and
                        oct_edge_data.get('role') == 'center'):
                        # Find which layer contains this octahedron
                        for layer_node, layer_neighbors in adjacency.items():
                            if (layer_node.startswith('layer_') and 
                                oct_neighbor in layer_neighbors and
                                layer_neighbors[oct_neighbor].get('edge_type') == 'contains'):
                                terminal_atom_to_layers[terminal_atom_id].add(layer_node)
                break
    
    # Count terminal atoms per layer and update layer nodes
    for layer_node in G.nodes():
        if layer_node.startswith('layer_'):
            terminal_count = sum(
                1 for atom_id, layers in terminal_atom_to_layers.items()
                if layer_node in layers
            )
            # Update the layer node with correct terminal count
            G.nodes[layer_node]['terminal_atoms_count'] = terminal_count
    
    # Store neighbor_indices and cell matrix in graph for later use
    G.graph['neighbor_indices'] = neighbor_indices
    G.graph['cell_matrix'] = np.array(cell, dtype=np.float64).copy()
    
    # Verify all atoms are in graph
    atom_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'atom']
    octahedra_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'octahedron']
    total_nodes = len(G.nodes())
    is_critical = (total_nodes == n_atoms_total and n_atoms_total > 0)
    
    if is_critical or octahedra_count == 0:
        layers_count_safe = len(layers) if layers is not None and len(layers) > 0 else 0
        msg = f"WARNING: Graph integrity issue - {n_atoms_total} atoms, {total_nodes} nodes, {octahedra_count} octahedra detected, {layers_count_safe} layers"
        print(msg, file=sys.stderr)
    
    if total_nodes == n_atoms_total and n_atoms_total > 0 and octahedra_count == 0:
        error_msg = (
            f"Graph construction failed: Structure has {n_atoms_total} atoms but graph only has {total_nodes} nodes. "
            f"No structural nodes (octahedra/layers) were added. Octahedra detection returned 0. "
            f"This indicates octahedra detection failed for this structure."
        )
        print(f"ERROR: {error_msg}", file=sys.stderr)
    
    # Ensure all atoms are in the graph
    if len(atom_nodes) < n_atoms_total:
        for i in range(n_atoms_total):
            if f'atom_{i}' not in G:
                pos = atom_positions[i]
                pos_list = pos.tolist() if hasattr(pos, 'tolist') else list(pos)
                node_data = {
                    'node_type': 'atom',
                    'vasp_index': i,
                    'symbol': atom_symbols[i] if atom_symbols else 'Unknown',
                    'x': float(pos_list[0]),
                    'y': float(pos_list[1]),
                    'z': float(pos_list[2]),
                }
                G.add_node(f'atom_{i}', **node_data)
    
    # Create Structure root node with comprehensive metadata
    _create_structure_node(G, atom_symbols, cell, layers)
    
    return G


def _create_structure_node(G, atom_symbols, cell, layers):
    """Create the Structure root node with comprehensive metadata.
    
    The Structure node serves as the single root of the graph hierarchy,
    containing global properties and connecting to top-level structural components.
    
    Parameters
    ----------
    G : nx.Graph
        The structural graph to add the Structure node to
    atom_symbols : list
        All atom symbols (for formula calculation)
    cell : np.ndarray
        Unit cell matrix (3x3)
    layers : dict
        Layer information dictionary
    """
    from collections import Counter
    
    # Calculate chemical formula (Hill notation)
    symbol_counts = Counter(atom_symbols)
    formula_parts = []
    # Hill notation: C first, then H, then alphabetical
    if 'C' in symbol_counts:
        formula_parts.append(f"C{symbol_counts['C']}" if symbol_counts['C'] > 1 else 'C')
        del symbol_counts['C']
    if 'H' in symbol_counts:
        formula_parts.append(f"H{symbol_counts['H']}" if symbol_counts['H'] > 1 else 'H')
        del symbol_counts['H']
    for symbol in sorted(symbol_counts.keys()):
        count = symbol_counts[symbol]
        formula_parts.append(f"{symbol}{count}" if count > 1 else symbol)
    formula = ''.join(formula_parts)
    
    # Calculate cell parameters from cell matrix
    cell_array = np.array(cell)
    a_vec = cell_array[0]
    b_vec = cell_array[1]
    c_vec = cell_array[2]
    
    a = float(np.linalg.norm(a_vec))
    b = float(np.linalg.norm(b_vec))
    c = float(np.linalg.norm(c_vec))
    
    # Calculate angles (in degrees)
    def angle_between(v1, v2):
        """Calculate angle between two vectors in degrees."""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Clamp to [-1, 1] to handle numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))
    
    alpha = angle_between(b_vec, c_vec)  # Angle between b and c
    beta = angle_between(a_vec, c_vec)   # Angle between a and c
    gamma = angle_between(a_vec, b_vec)  # Angle between a and b
    
    # Count nodes by type
    layer_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'layer']
    octahedra_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'octahedron']
    a_site_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'a_site']
    spacer_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'spacer']
    atom_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'atom']
    
    # Calculate thickness (number of layers)
    thickness = len(layer_nodes)
    
    # Create Structure node
    structure_id = 'structure_0'
    G.add_node(
        structure_id,
        node_type='structure',
        formula=formula,
        thickness=thickness,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        layer_count=len(layer_nodes),
        octahedra_count=len(octahedra_nodes),
        a_site_count=len(a_site_nodes),
        spacer_count=len(spacer_nodes),
        atom_count=len(atom_nodes),
    )
    
    # Add CONTAINS edges from Structure to all Layer nodes
    for layer_node in layer_nodes:
        G.add_edge(structure_id, layer_node, edge_type='contains')
    
    # Add CONTAINS edges from Structure to all A_Site nodes
    for a_site_node in a_site_nodes:
        G.add_edge(structure_id, a_site_node, edge_type='contains')
    
    # Add CONTAINS edges from Structure to all Spacer nodes
    for spacer_node in spacer_nodes:
        G.add_edge(structure_id, spacer_node, edge_type='contains')
    
    # Store Structure node ID in graph metadata for easy access
    G.graph['structure_node'] = structure_id


def _create_a_site_spacer_nodes(G, atom_positions, atom_symbols, cell, neighbor_indices, valid_molecule=None):
    """Create A_Site and Spacer nodes and classify them.
    
    Uses 12 nearest X atoms around molecule center to determine classification:
    - No terminal X atoms → A-site (closed cuboctahedra)
    - Has terminal X atoms → Spacer (unclosed, interacts with membrane)
    
    X-site atoms are derived from graph structure (atoms with role='ligand' in octahedra),
    not from hardcoded element lists.
    
    Also handles isolated atoms in cavities by creating A_Site nodes for them.
    
    Parameters
    ----------
    G : nx.Graph
        The structural graph (with octahedra and bonded_to edges already added)
    atom_positions : np.ndarray
        All atom positions
    atom_symbols : list
        All atom symbols
    cell : np.ndarray
        Unit cell matrix
    neighbor_indices : list
        Neighbor indices for each octahedron (used for compatibility, but X-site is derived from graph)
    valid_molecule : list, optional
        List of valid organic molecule elements
    """
    from q2D_Materials.utils.geometry.pbc_distances import find_nearest_image_positions
    from ..utils.clifford_embedding import unwrap_relative_coordinate, get_cell_lengths
    
    # First, identify B and X atoms to exclude from molecule detection
    b_atoms = set()
    x_atoms = set()
    
    # Find B atoms (connected to octahedra with role='center')
    for node in G.nodes():
        if node.startswith('octahedron_'):
            for neighbor in G.neighbors(node):
                edge_data = G.get_edge_data(node, neighbor)
                if (edge_data and
                    edge_data.get('edge_type') == 'contains' and
                    edge_data.get('role') == 'center'):
                    b_atoms.add(neighbor)
    
    # Find X atoms (bonded to B atoms with role='ligand')
    for b_atom in b_atoms:
        for neighbor in G.neighbors(b_atom):
            edge_data = G.get_edge_data(b_atom, neighbor)
            if (edge_data and
                edge_data.get('edge_type') == 'bonded_to' and
                edge_data.get('role') == 'ligand'):
                x_atoms.add(neighbor)
    
    # Convert node IDs to atom indices
    b_atom_indices = set()
    x_atom_indices = set()
    for b_node in b_atoms:
        node_data = G.nodes.get(b_node, {})
        atom_idx = node_data.get('vasp_index')
        if atom_idx is not None:
            b_atom_indices.add(atom_idx)
    for x_node in x_atoms:
        node_data = G.nodes.get(x_node, {})
        atom_idx = node_data.get('vasp_index')
        if atom_idx is not None:
            x_atom_indices.add(atom_idx)
    
    # Find all discrete molecules via connected components on bonded_to edges
    # Exclude B and X atoms (they belong to octahedra, not molecules)
    molecules = []  # List of [atom_idx, atom_idx, ...]
    visited_atoms = set()
    
    for node, data in G.nodes(data=True):
        if node.startswith('atom_') and data.get('node_type') == 'atom':
            atom_idx = data.get('vasp_index')
            if atom_idx is None or atom_idx in visited_atoms:
                continue
            
            # Skip B and X atoms - they belong to octahedra, not molecules
            if atom_idx in b_atom_indices or atom_idx in x_atom_indices:
                continue
            
            # Check if this atom is part of a molecule (has bonded_to edges)
            # But exclude edges with role='ligand' (B-X bonds)
            has_bond = False
            for neighbor in G.neighbors(node):
                edge_data = G.get_edge_data(node, neighbor)
                if edge_data and edge_data.get('edge_type') == 'bonded_to':
                    # Skip B-X bonds (role='ligand')
                    if edge_data.get('role') != 'ligand':
                        has_bond = True
                        break
            
            if not has_bond:
                continue
            
            # BFS to find all atoms in this molecule
            molecule = []
            to_visit = [atom_idx]
            mol_visited = set()
            
            while to_visit:
                curr_idx = to_visit.pop(0)
                if curr_idx in mol_visited:
                    continue
                # Skip B and X atoms during traversal
                if curr_idx in b_atom_indices or curr_idx in x_atom_indices:
                    continue
                mol_visited.add(curr_idx)
                visited_atoms.add(curr_idx)
                molecule.append(curr_idx)
                
                # Find neighbors via bonded_to edges (excluding B-X bonds)
                curr_node = f'atom_{curr_idx}'
                if curr_node in G:
                    for neighbor in G.neighbors(curr_node):
                        edge_data = G.get_edge_data(curr_node, neighbor)
                        if edge_data and edge_data.get('edge_type') == 'bonded_to':
                            # Skip B-X bonds (role='ligand')
                            if edge_data.get('role') == 'ligand':
                                continue
                            neighbor_idx = int(neighbor.replace('atom_', ''))
                            # Skip B and X atoms
                            if neighbor_idx in b_atom_indices or neighbor_idx in x_atom_indices:
                                continue
                            if neighbor_idx not in mol_visited:
                                to_visit.append(neighbor_idx)
            
            if molecule:
                molecules.append(molecule)
    
    if not molecules:
        return
    
    # Get cell lengths for PBC unwrapping
    cell_lengths = get_cell_lengths(cell)
    
    # Derive X-site atoms from graph structure (atoms bonded to B atoms with role='ligand')
    x_site_node_ids = set()
    terminal_x_node_ids = set()
    
    # Build mapping of X atom node IDs to octahedra for terminal detection
    atom_to_octahedra = {}
    
    # Find all B atoms (connected to octahedra with role='center')
    b_atoms = set()
    for node in G.nodes():
        if node.startswith('octahedron_'):
            for neighbor in G.neighbors(node):
                edge_data = G.get_edge_data(node, neighbor)
                if (edge_data and
                    edge_data.get('edge_type') == 'contains' and
                    edge_data.get('role') == 'center'):
                    b_atoms.add(neighbor)
    
    # Find X atoms bonded to B atoms
    for b_atom in b_atoms:
        # Find octahedron containing this B atom
        octahedron_node = None
        for neighbor in G.neighbors(b_atom):
            if neighbor.startswith('octahedron_'):
                edge_data = G.get_edge_data(neighbor, b_atom)
                if (edge_data and
                    edge_data.get('edge_type') == 'contains' and
                    edge_data.get('role') == 'center'):
                    octahedron_node = neighbor
                    break
        
        # Find X atoms bonded to this B atom
        for neighbor in G.neighbors(b_atom):
            edge_data = G.get_edge_data(b_atom, neighbor)
            if (edge_data and
                edge_data.get('edge_type') == 'bonded_to' and
                edge_data.get('role') == 'ligand'):
                x_site_node_ids.add(neighbor)
                # Track which octahedra contain this X atom (via B atom)
                if neighbor not in atom_to_octahedra:
                    atom_to_octahedra[neighbor] = []
                if octahedron_node and octahedron_node not in atom_to_octahedra[neighbor]:
                    atom_to_octahedra[neighbor].append(octahedron_node)
    
    # Identify terminal X atoms (belonging to exactly 1 octahedron)
    for x_node_id, oct_list in atom_to_octahedra.items():
        if len(oct_list) == 1:
            terminal_x_node_ids.add(x_node_id)
    
    # Convert X-site node IDs to atom indices and collect positions
    x_positions = []
    x_indices = []
    terminal_x_atoms = set()
    
    for x_node_id in x_site_node_ids:
        node_data = G.nodes.get(x_node_id, {})
        atom_idx = node_data.get('vasp_index')
        if atom_idx is not None:
            x_positions.append(atom_positions[atom_idx])
            x_indices.append(atom_idx)
            if x_node_id in terminal_x_node_ids:
                terminal_x_atoms.add(atom_idx)
    
    if len(x_positions) == 0:
        for node_id, mol_indices in enumerate(molecules):
            _add_a_site_spacer_node(G, node_id, mol_indices, 'a_site', atom_symbols)
        # Handle isolated atoms
        _handle_isolated_atoms(G, b_atom_indices, x_atom_indices, visited_atoms, atom_positions, atom_symbols, cell)
        return
    
    x_positions = np.array(x_positions)
    x_indices = np.array(x_indices)
    
    # Classify each molecule and create A_Site or Spacer nodes
    a_site_count = 0
    spacer_count = 0
    
    for node_id, mol_indices in enumerate(molecules):
        # Calculate molecule center using PBC-aware unwrapping
        ref_pos = atom_positions[mol_indices[0]]
        mol_positions = [ref_pos]
        for mol_idx in mol_indices[1:]:
            unwrapped = unwrap_relative_coordinate(ref_pos, atom_positions[mol_idx], cell_lengths)
            mol_positions.append(unwrapped)
        
        mol_center = np.mean(mol_positions, axis=0)
        
        # Find 12 nearest X atoms
        try:
            nearest_x_indices, nearest_x_positions, nearest_x_distances, nearest_x_labels = \
                find_nearest_image_positions(
                    mol_center, x_positions, x_indices, cell, n_neighbors=12,
                    exclude_indices=np.array(mol_indices)
                )
        except Exception:
            nearest_x_indices = np.array([])
        
        # Check if any of the 12 nearest X atoms are terminals
        has_terminal = any(x_idx in terminal_x_atoms for x_idx in nearest_x_indices)
        
        # Classification: If has terminal X atoms → Spacer, else → A-site
        if has_terminal:
            node_type = 'spacer'
            spacer_count += 1
        else:
            node_type = 'a_site'
            a_site_count += 1
        
        # Create A_Site or Spacer node and CONTAINS edges to atoms
        _add_a_site_spacer_node(G, node_id, mol_indices, node_type, atom_symbols)
    
    # Handle isolated atoms in cavities (not part of molecules, not B/X atoms)
    isolated_count = _handle_isolated_atoms(G, b_atom_indices, x_atom_indices, visited_atoms, atom_positions, atom_symbols, cell)
    


def _handle_isolated_atoms(G, b_atom_indices, x_atom_indices, visited_atoms, atom_positions, atom_symbols, cell):
    """Handle isolated atoms in cavities by creating A_Site nodes for them.
    
    Parameters
    ----------
    G : nx.Graph
        The structural graph
    b_atom_indices : set
        Set of B atom indices (to exclude)
    x_atom_indices : set
        Set of X atom indices (to exclude)
    visited_atoms : set
        Set of atom indices already in molecules
    atom_positions : np.ndarray
        All atom positions
    atom_symbols : list
        All atom symbols
    cell : np.ndarray
        Unit cell matrix
    
    Returns
    -------
    int
        Number of isolated atoms processed
    """
    isolated_count = 0
    all_atom_indices = set(range(len(atom_symbols)))
    
    # Find isolated atoms (not B, not X, not in molecules)
    isolated_atoms = all_atom_indices - b_atom_indices - x_atom_indices - visited_atoms
    
    # Create an A_Site node for each isolated atom
    for atom_idx in isolated_atoms:
        # Get the highest existing node ID to continue numbering
        existing_a_site_nodes = [n for n in G.nodes() if n.startswith('a_site_')]
        if existing_a_site_nodes:
            max_id = max(int(n.split('_')[-1]) for n in existing_a_site_nodes)
            node_id = max_id + 1 + isolated_count
        else:
            node_id = isolated_count
        
        # Create A_Site node for this isolated atom
        _add_a_site_spacer_node(G, node_id, [atom_idx], 'a_site', atom_symbols)
        isolated_count += 1
    
    return isolated_count


def _add_a_site_spacer_node(G, node_id, atom_indices, node_type, atom_symbols):
    """Add an A_Site or Spacer node and its CONTAINS edges to atoms.
    
    Parameters
    ----------
    G : nx.Graph
        The graph to add to
    node_id : int
        Unique node identifier
    atom_indices : list
        List of atom indices in this node
    node_type : str
        'a_site' or 'spacer'
    atom_symbols : list
        All atom symbols (for formula calculation)
    """
    # Calculate formula from atom symbols
    symbol_counts = {}
    for idx in atom_indices:
        symbol = atom_symbols[idx]
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    
    # Hill notation: C first, then H, then alphabetical
    formula_parts = []
    if 'C' in symbol_counts:
        formula_parts.append(f"C{symbol_counts['C']}" if symbol_counts['C'] > 1 else 'C')
        del symbol_counts['C']
    if 'H' in symbol_counts:
        formula_parts.append(f"H{symbol_counts['H']}" if symbol_counts['H'] > 1 else 'H')
        del symbol_counts['H']
    for symbol in sorted(symbol_counts.keys()):
        count = symbol_counts[symbol]
        formula_parts.append(f"{symbol}{count}" if count > 1 else symbol)
    
    formula = ''.join(formula_parts)
    
    # Create node ID based on type
    node_id_str = f'{node_type}_{node_id}'
    
    # Add A_Site or Spacer node
    G.add_node(
        node_id_str,
        node_type=node_type,
        formula=formula,
        nh3_count=0  # Will be set by identify_backbone for spacer molecules
    )
    
    # Add CONTAINS edges from node to each Atom
    for atom_idx in atom_indices:
        G.add_edge(
            node_id_str,
            f'atom_{atom_idx}',
            edge_type='contains'
        )
    
    # Perform backbone identification for spacer nodes
    if node_type == 'spacer':
        from ..molecular_processing.molecule_graph import identify_backbone
        try:
            nh3_count = identify_backbone(G, node_id_str)
            if nh3_count > 0:
                G.nodes[node_id_str]['nh3_count'] = nh3_count
        except Exception:
            pass


def _create_molecular_connections(G, atom_positions, atom_symbols, neighbor_indices, cell, valid_molecule=None):
    """Create BONDED_TO edges within organic molecules.

    Detects connectivity within isolated organic molecules using covalent radii
    distance heuristics. Adds hybridization property to carbon atoms.

    Parameters
    ----------
    G : networkx.Graph
        Graph to add connections to
    atom_positions : list
        List of atom coordinates (3D)
    atom_symbols : list
        List of atomic symbols
    neighbor_indices : list
        List of neighbor indices for each octahedron
    cell : np.ndarray
        Unit cell matrix (3x3)
    valid_molecule : list, optional
        Valid molecule element symbols. Defaults to ['C', 'H', 'O', 'N', 'S']
    """
    from ...utils.properties.atomic_properties import determine_hybridization

    atoms_in_octahedra = set()
    for oct_neighbors in neighbor_indices:
        atoms_in_octahedra.update(oct_neighbors)

    isolated_atoms = [i for i, _ in enumerate(atom_symbols) if i not in atoms_in_octahedra]

    if not isolated_atoms:
        return

    full_atoms = Atoms(
        symbols=atom_symbols,
        positions=atom_positions,
        cell=cell,
        pbc=True
    )

    # Use API parameter instead of hardcoded set
    if valid_molecule is None:
        valid_molecule = ['C', 'H', 'O', 'N', 'S']
    molecule_elements = set(valid_molecule)
    molecule_indices = {i for i, sym in enumerate(atom_symbols) if sym in molecule_elements}

    molecular_connections = get_molecular_connections(full_atoms, atom_indices=molecule_indices)

    # Build connectivity count for hybridization calculation
    connectivity_count = {}
    for i, j, _ in molecular_connections:
        connectivity_count[i] = connectivity_count.get(i, 0) + 1
        connectivity_count[j] = connectivity_count.get(j, 0) + 1

    # Add BONDED_TO edges
    for i, j, bond_length in molecular_connections:
        symbol_i = atom_symbols[i]
        symbol_j = atom_symbols[j]

        # Skip H-H
        if symbol_i == 'H' and symbol_j == 'H':
            continue

        # Canonical bond type ordering (alphabetical)
        bond_type = f'{symbol_i}-{symbol_j}' if symbol_i <= symbol_j else f'{symbol_j}-{symbol_i}'

        G.add_edge(
            f'atom_{i}',
            f'atom_{j}',
            edge_type='bonded_to',
            bond_type=bond_type,
            distance=float(bond_length)
        )

    # Add hybridization to carbon atoms
    for atom_idx in molecule_indices:
        if atom_idx in atoms_in_octahedra:
            continue

        symbol = atom_symbols[atom_idx]
        num_neighbors = connectivity_count.get(atom_idx, 0)

        if symbol == 'C' and num_neighbors > 0:
            try:
                hybridization, _ = determine_hybridization(num_neighbors)
                if f'atom_{atom_idx}' in G.nodes:
                    G.nodes[f'atom_{atom_idx}']['hybridization'] = hybridization
            except ValueError:
                pass


# =============================================================================
# Query Helper Functions
# =============================================================================

def get_structure_node(G):
    """Get the Structure root node and its properties.
    
    The Structure node is the single root of the graph hierarchy containing
    global metadata like formula, cell parameters, and counts.
    
    Parameters
    ----------
    G : nx.Graph
        The structural graph
    
    Returns
    -------
    dict or None
        Structure node properties dictionary, or None if not found.
        Contains: node_type, formula, thickness, a, b, c, alpha, beta, gamma,
        layer_count, octahedra_count, a_site_count, spacer_count, atom_count
    
    Examples
    --------
    >>> structure = get_structure_node(graph)
    >>> print(f"Formula: {structure['formula']}")
    >>> print(f"Cell: a={structure['a']:.3f}, b={structure['b']:.3f}, c={structure['c']:.3f}")
    """
    # Fast path: use stored structure node ID
    structure_id = G.graph.get('structure_node')
    if structure_id and structure_id in G:
        return dict(G.nodes[structure_id])
    
    # Fallback: search for structure node
    for node, data in G.nodes(data=True):
        if data.get('node_type') == 'structure':
            return dict(data)
    
    return None


def get_cell_matrix(G):
    """Get the unit cell matrix from the graph.
    
    The cell matrix is stored in graph metadata during construction and
    contains the cell vectors as rows.
    
    Parameters
    ----------
    G : nx.Graph
        The structural graph
    
    Returns
    -------
    np.ndarray or None
        Cell matrix (3x3) with cell vectors as rows, or None if not found.
        cell[0] = a-vector, cell[1] = b-vector, cell[2] = c-vector
    
    Examples
    --------
    >>> cell = get_cell_matrix(graph)
    >>> if cell is not None:
    ...     a_vec = cell[0]
    ...     c_length = np.linalg.norm(cell[2])
    """
    return G.graph.get('cell_matrix', None)


def get_terminal_atoms(G):
    """Find terminal X atoms (atoms belonging to exactly 1 octahedron).
    
    Traverses: X atom → B atoms → Octahedra
    
    Returns
    -------
    list
        List of atom node IDs that are terminal
    """
    atom_octahedra_count = {}
    
    # Find all X atoms (bonded to B atoms with role='ligand')
    for node, data in G.nodes(data=True):
        if data.get('node_type') == 'atom':
            # Check if this is an X atom
            for neighbor in G.neighbors(node):
                edge_data = G.get_edge_data(node, neighbor)
                if (edge_data and
                    edge_data.get('edge_type') == 'bonded_to' and
                    edge_data.get('role') == 'ligand'):
                    # This is an X atom, count octahedra via B atoms
                    b_atom = neighbor
                    for oct_neighbor in G.neighbors(b_atom):
                        if oct_neighbor.startswith('octahedron_'):
                            oct_edge_data = G.get_edge_data(oct_neighbor, b_atom)
                            if (oct_edge_data and
                                oct_edge_data.get('edge_type') == 'contains' and
                                oct_edge_data.get('role') == 'center'):
                                atom_octahedra_count[node] = atom_octahedra_count.get(node, 0) + 1
                    break  # Only count once per X atom
    
    return [atom_id for atom_id, count in atom_octahedra_count.items() if count == 1]


def get_interlayer_atoms(G):
    """Find interlayer X atoms (atoms shared between octahedra in different layers).
    
    Traverses: X atom → B atoms → Octahedra → Layers
    
    Returns
    -------
    list
        List of atom node IDs that bridge layers
    """
    # Build octahedron -> layer mapping
    oct_to_layer = {}
    for node, data in G.nodes(data=True):
        if data.get('node_type') == 'layer':
            for neighbor in G.neighbors(node):
                edge_data = G.get_edge_data(node, neighbor)
                if edge_data and edge_data.get('edge_type') == 'contains':
                    neighbor_data = G.nodes.get(neighbor, {})
                    if neighbor_data.get('node_type') == 'octahedron':
                        oct_to_layer[neighbor] = node
    
    # Find X atoms connected to octahedra in different layers via B atoms
    atom_layers = {}
    for node, data in G.nodes(data=True):
        if data.get('node_type') == 'atom':
            # Check if this is an X atom (bonded to B atoms with role='ligand')
            for neighbor in G.neighbors(node):
                edge_data = G.get_edge_data(node, neighbor)
                if (edge_data and
                    edge_data.get('edge_type') == 'bonded_to' and
                    edge_data.get('role') == 'ligand'):
                    # This is an X atom, find octahedra via B atom
                    b_atom = neighbor
                    for oct_neighbor in G.neighbors(b_atom):
                        if oct_neighbor.startswith('octahedron_'):
                            oct_edge_data = G.get_edge_data(oct_neighbor, b_atom)
                            if (oct_edge_data and
                                oct_edge_data.get('edge_type') == 'contains' and
                                oct_edge_data.get('role') == 'center'):
                                layer = oct_to_layer.get(oct_neighbor)
                                if node not in atom_layers:
                                    atom_layers[node] = set()
                                if layer:
                                    atom_layers[node].add(layer)
                    break  # Only need to check once per X atom
    
    return [atom_id for atom_id, layers in atom_layers.items() if len(layers) > 1]


def get_center_atom(G, octahedron_id):
    """Get the center (B-site) atom of an octahedron.
    
    Parameters
    ----------
    G : nx.Graph
        The structural graph
    octahedron_id : str
        Octahedron node ID (e.g., 'octahedron_0')
    
    Returns
    -------
    str or None
        Atom node ID of the center atom, or None if not found
    """
    for neighbor in G.neighbors(octahedron_id):
        edge_data = G.get_edge_data(octahedron_id, neighbor)
        if edge_data and edge_data.get('edge_type') == 'contains' and edge_data.get('role') == 'center':
            return neighbor
    return None


def get_ligand_atoms(G, octahedron_id, geometry=None):
    """Get ligand atoms of an octahedron, optionally filtered by geometry.
    
    Traverses: Octahedron → B atom → X atoms
    
    Parameters
    ----------
    G : nx.Graph
        The structural graph
    octahedron_id : str
        Octahedron node ID (e.g., 'octahedron_0')
    geometry : str, optional
        Filter by geometry: 'axial', 'equatorial', or None for all
    
    Returns
    -------
    list
        List of atom node IDs (X atoms)
    """
    # Get B atom from octahedron
    b_atom = get_center_atom(G, octahedron_id)
    if b_atom is None:
        return []
    
    # Get X atoms bonded to B atom
    ligands = []
    for neighbor in G.neighbors(b_atom):
        edge_data = G.get_edge_data(b_atom, neighbor)
        if (edge_data and
            edge_data.get('edge_type') == 'bonded_to' and
            edge_data.get('role') == 'ligand'):
            if geometry is None or edge_data.get('geometry') == geometry:
                ligands.append(neighbor)
    return ligands


def get_molecule_atoms(G, node_id):
    """Get all atoms in an A_Site or Spacer node.
    
    Parameters
    ----------
    G : nx.Graph
        The structural graph
    node_id : str
        A_Site or Spacer node ID (e.g., 'a_site_0', 'spacer_0')
    
    Returns
    -------
    list
        List of atom node IDs
    """
    atoms = []
    for neighbor in G.neighbors(node_id):
        edge_data = G.get_edge_data(node_id, neighbor)
        if edge_data and edge_data.get('edge_type') == 'contains':
            neighbor_data = G.nodes.get(neighbor, {})
            if neighbor_data.get('node_type') == 'atom':
                atoms.append(neighbor)
    return atoms


def get_sharing_octahedra(G, atom_id):
    """Get all octahedra that share a given atom.
    
    For X atoms: Traverses X atom → B atoms → Octahedra
    For B atoms: Traverses B atom → Octahedra directly
    
    Parameters
    ----------
    G : nx.Graph
        The structural graph
    atom_id : str
        Atom node ID (e.g., 'atom_5')
    
    Returns
    -------
    list
        List of octahedron node IDs
    """
    octahedra = []
    
    # Check if this is a B atom (connected to octahedra with role='center')
    for neighbor in G.neighbors(atom_id):
        if neighbor.startswith('octahedron_'):
            edge_data = G.get_edge_data(neighbor, atom_id)
            if (edge_data and
                edge_data.get('edge_type') == 'contains' and
                edge_data.get('role') == 'center'):
                octahedra.append(neighbor)
    
    # If not found as B atom, check if it's an X atom (bonded to B atoms)
    if not octahedra:
        for neighbor in G.neighbors(atom_id):
            edge_data = G.get_edge_data(atom_id, neighbor)
            if (edge_data and
                edge_data.get('edge_type') == 'bonded_to' and
                edge_data.get('role') == 'ligand'):
                # This is an X atom, find octahedra via B atom
                b_atom = neighbor
                for oct_neighbor in G.neighbors(b_atom):
                    if oct_neighbor.startswith('octahedron_'):
                        oct_edge_data = G.get_edge_data(oct_neighbor, b_atom)
                        if (oct_edge_data and
                            oct_edge_data.get('edge_type') == 'contains' and
                            oct_edge_data.get('role') == 'center'):
                            if oct_neighbor not in octahedra:
                                octahedra.append(oct_neighbor)
    
    return octahedra

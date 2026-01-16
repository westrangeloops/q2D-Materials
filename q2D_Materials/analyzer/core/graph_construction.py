"""Comprehensive graph ontology construction for 2D materials.

This module builds the complete structural connectivity graph including
octahedra, layers, atoms, molecules, and their relationships.

Graph Philosophy:
- Nodes store intrinsic properties only (what the entity IS)
- Edges express relationships (how entities connect)
- Frequently-used structural properties (is_terminal, is_equatorial, is_interlayer) 
  are computed once and stored on X-site atoms for performance
- No index lists in nodes - that's what edges are for
"""

import numpy as np
import networkx as nx
from ase import Atoms
import sys

from ..detection.layer_identification import _get_octahedron_layer, _identify_layers
from ..detection.octahedral_detection import _count_octahedra, find_shared_atoms
from ..utils.pymatgen_utils import get_molecular_connections
from ..utils.clifford_embedding import embed_to_6d, clifford_distance, get_cell_lengths


def _compute_atom_structural_properties(G, atom_symbols):
    """Compute and store structural properties for X-site atoms.
    
    This function computes is_terminal and is_equatorial properties for X-site atoms
    based on their connectivity to octahedra in the graph.
    
    Properties computed:
    - is_terminal: True if atom belongs to exactly 1 octahedron (terminal/surface atom)
    - is_equatorial: True if atom is in equatorial position of any octahedron
    - is_interlayer: True if atom is shared between 2 octahedra (axial bridging atom)
    
    Parameters
    ----------
    G : nx.Graph
        The structural graph with octahedra and atom nodes
    atom_symbols : list
        List of atomic symbols for all atoms
    """
    # Define X-site elements (halides and chalcogens typically in perovskites)
    x_site_elements = {'F', 'Cl', 'Br', 'I', 'S', 'Se', 'Te', 'O'}
    
    # For each atom, count octahedra connections and check geometry
    for node in G.nodes():
        if not node.startswith('atom_'):
            continue
        
        node_data = G.nodes[node]
        symbol = node_data.get('symbol', '')
        
        # Only compute for X-site atoms
        if symbol not in x_site_elements:
            continue
        
        # Find all octahedra containing this atom
        containing_octahedra = []
        is_equatorial = False
        
        for neighbor in G.neighbors(node):
            if neighbor.startswith('octahedron_'):
                edge_data = G.get_edge_data(neighbor, node)
                if edge_data and edge_data.get('edge_type') == 'contains':
                    containing_octahedra.append(neighbor)
                    # Check if this atom is equatorial in this octahedron
                    if edge_data.get('geometry') == 'equatorial':
                        is_equatorial = True
        
        # Terminal: belongs to exactly 1 octahedron
        is_terminal = (len(containing_octahedra) == 1)
        
        # Interlayer: shared between 2 octahedra (axial bridging)
        is_interlayer = (len(containing_octahedra) == 2)
        
        # Store properties on the atom node
        G.nodes[node]['is_terminal'] = is_terminal
        G.nodes[node]['is_equatorial'] = is_equatorial
        G.nodes[node]['is_interlayer'] = is_interlayer


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

    # Add Layer nodes (minimal properties - z_coord only)
    for layer_id, layer_info in layers.items():
        G.add_node(
            f'layer_{layer_id}',
            node_type='layer',
            z_coord=layer_info.get('z_coord'),
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
        for i, (center_pos, center_sym, neighbors, center_idx) in enumerate(zip(
            centers_positions, center_symbols, neighbor_indices, center_atom_indices
        )):
            geometry_info = octahedra_geometries[i] if i < len(octahedra_geometries) else None
            
            if geometry_info is None:
                raise ValueError(
                    f"Missing geometry classification for octahedron {i} with center atom {center_idx} ({center_sym})"
                )
            
            # Validate geometry counts (for logging purposes only)
            counts = {
                'axial': sum(1 for v in geometry_info.values() if v in ['axial_terminal', 'axial_interlayer']),
                'equatorial': sum(1 for v in geometry_info.values() if v == 'equatorial'),
                'unknown': sum(1 for v in geometry_info.values() if v == 'unknown')
            }
            
            if counts['axial'] != 2 or counts['equatorial'] != 4:
                if counts['unknown'] > 0 or (counts['axial'] + counts['equatorial'] != len(neighbors)):
                    print(f"  INFO: Non-standard octahedron {i} (center {center_idx}): "
                          f"{counts['axial']} axial, {counts['equatorial']} equatorial, "
                          f"{counts['unknown']} unknown", file=sys.stderr)
            
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
            
            # Octahedron CONTAINS ligand atoms (role='ligand' with geometry)
            for neighbor_idx in neighbors:
                geometry_label = geometry_info.get(neighbor_idx, 'unknown')
                
                # Simplify geometry: axial_terminal/axial_interlayer → 'axial'
                if geometry_label in ['axial_terminal', 'axial_interlayer']:
                    simplified_geometry = 'axial'
                elif geometry_label == 'equatorial':
                    simplified_geometry = 'equatorial'
                else:
                    simplified_geometry = 'unknown'
                
                G.add_edge(
                    f'octahedron_{i}',
                    f'atom_{neighbor_idx}',
                    edge_type='contains',
                    role='ligand',
                    geometry=simplified_geometry
                )
    
    # Compute and store is_terminal and is_equatorial properties for X-site atoms
    _compute_atom_structural_properties(G, atom_symbols)

    # Create molecular connections and molecule nodes
    _create_molecular_connections(G, atom_positions, atom_symbols, neighbor_indices, cell, valid_molecule=valid_molecule)
    
    # Create Molecule nodes and classify as A-site or spacer
    _create_molecule_nodes(G, atom_positions_array, atom_symbols, cell, neighbor_indices, valid_molecule=valid_molecule)
    
    # Store neighbor_indices in graph for later cavity detection
    G.graph['neighbor_indices'] = neighbor_indices
    
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
    molecule_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'molecule']
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
        molecule_count=len(molecule_nodes),
        atom_count=len(atom_nodes),
    )
    
    # Add CONTAINS edges from Structure to all Layer nodes
    for layer_node in layer_nodes:
        G.add_edge(structure_id, layer_node, edge_type='contains')
    
    # Add CONTAINS edges from Structure to all Molecule nodes
    for molecule_node in molecule_nodes:
        G.add_edge(structure_id, molecule_node, edge_type='contains')
    
    # Store Structure node ID in graph metadata for easy access
    G.graph['structure_node'] = structure_id


def _create_molecule_nodes(G, atom_positions, atom_symbols, cell, neighbor_indices, valid_molecule=None):
    """Create Molecule nodes and classify as A-site or spacer.
    
    Uses 12 nearest X atoms around molecule center to determine classification:
    - No terminal X atoms → A-site (closed cuboctahedra)
    - Has terminal X atoms → Spacer (unclosed, interacts with membrane)
    
    Parameters
    ----------
    G : nx.Graph
        The structural graph (with bonded_to edges already added)
    atom_positions : np.ndarray
        All atom positions
    atom_symbols : list
        All atom symbols
    cell : np.ndarray
        Unit cell matrix
    neighbor_indices : list
        Neighbor indices for each octahedron
    valid_molecule : list, optional
        List of valid organic molecule elements
    """
    from q2D_Materials.utils.geometry.pbc_distances import find_nearest_image_positions
    from ..utils.clifford_embedding import unwrap_relative_coordinate, get_cell_lengths
    
    # Find all discrete molecules via connected components on bonded_to edges
    molecules = []  # List of [atom_idx, atom_idx, ...]
    visited_atoms = set()
    
    for node, data in G.nodes(data=True):
        if node.startswith('atom_') and data.get('node_type') == 'atom':
            atom_idx = data.get('vasp_index')
            if atom_idx is None or atom_idx in visited_atoms:
                continue
            
            # Check if this atom is part of a molecule (has bonded_to edges)
            has_bond = False
            for neighbor in G.neighbors(node):
                edge_data = G.get_edge_data(node, neighbor)
                if edge_data and edge_data.get('edge_type') == 'bonded_to':
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
                mol_visited.add(curr_idx)
                visited_atoms.add(curr_idx)
                molecule.append(curr_idx)
                
                # Find neighbors via bonded_to edges
                curr_node = f'atom_{curr_idx}'
                if curr_node in G:
                    for neighbor in G.neighbors(curr_node):
                        edge_data = G.get_edge_data(curr_node, neighbor)
                        if edge_data and edge_data.get('edge_type') == 'bonded_to':
                            neighbor_idx = int(neighbor.replace('atom_', ''))
                            if neighbor_idx not in mol_visited:
                                to_visit.append(neighbor_idx)
            
            if molecule:
                molecules.append(molecule)
    
    if not molecules:
        print(f"  No organic molecules found", file=sys.stderr)
        return
    
    # Get cell lengths for PBC unwrapping
    cell_lengths = get_cell_lengths(cell)
    
    # Build atom_to_octahedra mapping for terminal detection
    atom_to_octahedra = {}
    for oct_idx, oct_neighbors in enumerate(neighbor_indices):
        for neighbor_idx in oct_neighbors:
            if neighbor_idx not in atom_to_octahedra:
                atom_to_octahedra[neighbor_idx] = []
            atom_to_octahedra[neighbor_idx].append(oct_idx)
    
    # Collect terminal X atom indices (atoms belonging to only 1 octahedron)
    terminal_x_atoms = set()
    for atom_idx, oct_list in atom_to_octahedra.items():
        if len(oct_list) == 1:
            symbol = atom_symbols[atom_idx]
            if symbol in ['Cl', 'Br', 'I', 'F', 'O']:
                terminal_x_atoms.add(atom_idx)
    
    # Collect X atom positions and indices
    x_positions = []
    x_indices = []
    
    for atom_idx, symbol in enumerate(atom_symbols):
        if symbol not in ['Cl', 'Br', 'I', 'F', 'O']:
            continue
        x_positions.append(atom_positions[atom_idx])
        x_indices.append(atom_idx)
    
    if len(x_positions) == 0:
        print(f"  No X atoms found in structure: marking all {len(molecules)} molecules as A-sites", file=sys.stderr)
        for mol_id, mol_indices in enumerate(molecules):
            _add_molecule_node(G, mol_id, mol_indices, 'a_site', atom_symbols)
        return
    
    x_positions = np.array(x_positions)
    x_indices = np.array(x_indices)
    
    # Classify each molecule and create Molecule nodes
    a_site_count = 0
    spacer_count = 0
    
    for mol_id, mol_indices in enumerate(molecules):
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
            molecule_type = 'spacer'
            spacer_count += 1
        else:
            molecule_type = 'a_site'
            a_site_count += 1
        
        # Create Molecule node and CONTAINS edges to atoms
        _add_molecule_node(G, mol_id, mol_indices, molecule_type, atom_symbols)
    
    print(f"  Created {a_site_count} A-site molecule(s), {spacer_count} spacer molecule(s)", file=sys.stderr)


def _add_molecule_node(G, mol_id, mol_indices, molecule_type, atom_symbols):
    """Add a Molecule node and its CONTAINS edges to atoms.
    
    Parameters
    ----------
    G : nx.Graph
        The graph to add to
    mol_id : int
        Unique molecule identifier
    mol_indices : list
        List of atom indices in this molecule
    molecule_type : str
        'a_site' or 'spacer'
    atom_symbols : list
        All atom symbols (for formula calculation)
    """
    # Calculate formula from atom symbols
    symbol_counts = {}
    for idx in mol_indices:
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
    
    # Add Molecule node
    G.add_node(
        f'molecule_{mol_id}',
        node_type='molecule',
        molecule_type=molecule_type,
        formula=formula,
    )
    
    # Add CONTAINS edges from Molecule to each Atom
    for atom_idx in mol_indices:
        G.add_edge(
            f'molecule_{mol_id}',
            f'atom_{atom_idx}',
            edge_type='contains'
        )


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
        layer_count, octahedra_count, molecule_count, atom_count
    
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


def get_terminal_atoms(G):
    """Find terminal X atoms (atoms belonging to exactly 1 octahedron).
    
    Returns
    -------
    list
        List of atom node IDs that are terminal
    """
    atom_octahedra_count = {}
    
    for node, data in G.nodes(data=True):
        if data.get('node_type') == 'octahedron':
            for neighbor in G.neighbors(node):
                edge_data = G.get_edge_data(node, neighbor)
                if edge_data and edge_data.get('edge_type') == 'contains' and edge_data.get('role') == 'ligand':
                    atom_octahedra_count[neighbor] = atom_octahedra_count.get(neighbor, 0) + 1
    
    return [atom_id for atom_id, count in atom_octahedra_count.items() if count == 1]


def get_interlayer_atoms(G):
    """Find interlayer X atoms (atoms shared between octahedra in different layers).
    
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
    
    # Find atoms connected to octahedra in different layers
    atom_layers = {}
    for node, data in G.nodes(data=True):
        if data.get('node_type') == 'octahedron':
            layer = oct_to_layer.get(node)
            for neighbor in G.neighbors(node):
                edge_data = G.get_edge_data(node, neighbor)
                if edge_data and edge_data.get('edge_type') == 'contains' and edge_data.get('role') == 'ligand':
                    if neighbor not in atom_layers:
                        atom_layers[neighbor] = set()
                    if layer:
                        atom_layers[neighbor].add(layer)
    
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
        List of atom node IDs
    """
    ligands = []
    for neighbor in G.neighbors(octahedron_id):
        edge_data = G.get_edge_data(octahedron_id, neighbor)
        if edge_data and edge_data.get('edge_type') == 'contains' and edge_data.get('role') == 'ligand':
            if geometry is None or edge_data.get('geometry') == geometry:
                ligands.append(neighbor)
    return ligands


def get_molecule_atoms(G, molecule_id):
    """Get all atoms in a molecule.
    
    Parameters
    ----------
    G : nx.Graph
        The structural graph
    molecule_id : str
        Molecule node ID (e.g., 'molecule_0')
    
    Returns
    -------
    list
        List of atom node IDs
    """
    atoms = []
    for neighbor in G.neighbors(molecule_id):
        edge_data = G.get_edge_data(molecule_id, neighbor)
        if edge_data and edge_data.get('edge_type') == 'contains':
            neighbor_data = G.nodes.get(neighbor, {})
            if neighbor_data.get('node_type') == 'atom':
                atoms.append(neighbor)
    return atoms


def get_sharing_octahedra(G, atom_id):
    """Get all octahedra that share a given atom.
    
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
    for neighbor in G.neighbors(atom_id):
        edge_data = G.get_edge_data(atom_id, neighbor)
        if edge_data and edge_data.get('edge_type') == 'contains':
            neighbor_data = G.nodes.get(neighbor, {})
            if neighbor_data.get('node_type') == 'octahedron':
                octahedra.append(neighbor)
    return octahedra

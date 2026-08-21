"""
Unified Graph Converter - Bidirectional conversion between ASE Atoms, NetworkX graphs, and RDKit Mol.

This module provides the core conversion functions for the unified molecular architecture:
- ASE Atoms ↔ NetworkX Graph ↔ RDKit Mol
- Coordinate mapping and preservation
- Detection of changed atoms for efficient coordinate transfer

Key Features:
- Maintains coordinate mapping between representations
- Detects only changed atoms (unchanged atoms preserve original coordinates)
- Requires at least 2 conserved atoms for reconstruction
- Handles error cases (invalid SMILES, insufficient conserved atoms)

Atomic Properties Data:
- This module uses atomic properties functions from `utils.properties.atomic_properties`,
  which reads data from `q2D_Materials/data/tables/`:
  - covalent_radii.json: Single bond covalent radii for all elements
  - covalent_radii_bond_order.json: Bond-order-specific covalent radii (single, double, triple)
  - atomic_valence.json: Typical valence values for common elements
- Functions used from `atomic_properties.py`:
  - get_covalent_radius(): Get single bond covalent radius
  - get_covalent_radius_by_bond_order(): Get bond-order-specific radius
  - estimate_bond_order(): Estimate bond order from distance
  - are_atoms_bonded(): Check if atoms are bonded based on distance

TODO - Known Issues:
- Charge inference for 5-membered aromatic ring nitrogens (e.g., imidazolium C[N+]1=CSC=C1,
  thiazolium C1=CSC=[NH+]1) currently fails when converting from ASE Atoms objects.
  These nitrogens with 2 neighbors in 5-membered rings should be charged (+1) but are
  incorrectly assigned charge 0. The issue affects molecules like MIC1, ThA, MIC3, MIC2.
  Workaround: Use SMILES strings directly instead of Atoms objects for validation of
  5-membered ring nitrogen-containing molecules.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import networkx as nx
from ase import Atoms

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None

LOGGER = logging.getLogger(__name__)


# ============================================================================
# ASE Atoms ↔ NetworkX Graph
# ============================================================================

def atoms_to_graph(atoms: Atoms, preserve_coords: bool = True, infer_charges: bool = True, compute_partial_charges: bool = False) -> nx.Graph:
    """Convert ASE Atoms to NetworkX graph with coordinate preservation.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object to convert
    preserve_coords : bool, default=True
        If True, store atom positions in graph nodes
    infer_charges : bool, default=True
        If True, use RDKit to infer formal charges from molecular structure
        (required for matching charged patterns like [NH3+])
    compute_partial_charges : bool, default=False
        If True, compute Gasteiger partial charges using RDKit (requires RDKit).
        This performs a graph->rdkit->graph roundtrip internally.

    Returns
    -------
    nx.Graph
        NetworkX graph with:
        - Nodes: atom indices (0, 1, 2, ...)
        - Node attributes: 'symbol', 'position', 'ase_index', 'charge' (formal), 'partial_charge'
        - Edges: covalent bonds (based on distance)
        - Graph attributes: 'ase_to_nx' and 'nx_to_ase' mappings
    """
    G = nx.Graph()

    # Store coordinate mapping
    G.graph['ase_to_nx'] = {}
    G.graph['nx_to_ase'] = {}

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    # Infer charges based on coordination number AND bond lengths + RDKit aromaticity
    charges = None
    if infer_charges:
        try:
            from ...utils.properties.atomic_properties import (
                get_covalent_radius,
                get_covalent_radius_by_bond_order,
                estimate_bond_order,
                are_atoms_bonded
            )

            # Build neighbor lists and analyze bond lengths
            neighbor_lists = [[] for _ in range(len(atoms))]
            bond_orders_inferred = {}  # (i, j) -> inferred bond order

            for i in range(len(atoms)):
                for j in range(i + 1, len(atoms)):
                    dist = np.linalg.norm(positions[i] - positions[j])

                    # First check if atoms are bonded using atomic_properties
                    if are_atoms_bonded(dist, symbols[i], symbols[j], tolerance=0.45):
                        # Use atomic_properties to estimate bond order from distance
                        # This uses bond-order-specific covalent radii for accurate detection
                        bond_order = estimate_bond_order(dist, symbols[i], symbols[j], tolerance=0.15)
                        
                        neighbor_lists[i].append((j, bond_order, dist))
                        neighbor_lists[j].append((i, bond_order, dist))
                        bond_orders_inferred[(min(i,j), max(i,j))] = bond_order

            # Helper function: Check if 4 atoms are coplanar (sp² geometry)
            def is_planar(center_idx, neighbor_indices):
                """Check if center atom and its neighbors are coplanar (sp² hybridization)."""
                if len(neighbor_indices) < 3:
                    return False

                center_pos = positions[center_idx]
                neighbor_pos = [positions[i] for i in neighbor_indices[:3]]

                # Calculate vectors from center to neighbors
                v1 = neighbor_pos[0] - center_pos
                v2 = neighbor_pos[1] - center_pos
                v3 = neighbor_pos[2] - center_pos

                # Normalize vectors
                v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
                v3_norm = v3 / (np.linalg.norm(v3) + 1e-10)

                # Calculate normal to plane defined by v1 and v2
                normal = np.cross(v1_norm, v2_norm)
                normal_norm = np.linalg.norm(normal)

                if normal_norm < 1e-6:
                    # v1 and v2 are collinear, can't define plane
                    return False

                normal = normal / normal_norm

                # Check if v3 is perpendicular to normal (i.e., in the plane)
                # For sp² geometry, angle between v3 and normal should be ~90°
                # So dot product should be ~0
                dot_product = abs(np.dot(v3_norm, normal))

                # Threshold: if dot product < 0.2, atoms are roughly coplanar (sp²)
                # For perfect sp² (120° angles), dot product = 0
                # For sp³ (109.5° tetrahedral), dot product > 0.3
                return dot_product < 0.2

            # Helper function: Check if geometry is tetrahedral (sp³)
            def is_tetrahedral(center_idx, neighbor_indices):
                """Check if center atom has tetrahedral geometry (sp³ hybridization)."""
                if len(neighbor_indices) != 4:
                    return False

                center_pos = positions[center_idx]
                neighbor_pos = [positions[i] for i in neighbor_indices]

                # Calculate all pairwise angles between bonds
                angles = []
                for i in range(len(neighbor_indices)):
                    for j in range(i + 1, len(neighbor_indices)):
                        v1 = neighbor_pos[i] - center_pos
                        v2 = neighbor_pos[j] - center_pos

                        v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
                        v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)

                        cos_angle = np.dot(v1_norm, v2_norm)
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle_deg = np.degrees(np.arccos(cos_angle))
                        angles.append(angle_deg)

                # For perfect tetrahedral: all angles = 109.47°
                # For sp² planar: angles ~ 120° (for 3 bonds) or mixed
                # Check if average angle is close to tetrahedral (105-115°)
                avg_angle = np.mean(angles)
                return 105.0 < avg_angle < 115.0

            # Detect aromatic nitrogen using topological + geometric heuristics
            # Since RDKit requires correct charges to detect aromaticity (chicken-egg problem),
            # we use ring detection + planarity + bond pattern analysis
            aromatic_atoms = set()

            # Build connectivity graph for ring detection
            conn_graph = nx.Graph()
            for i in range(len(atoms)):
                conn_graph.add_node(i)
                for j, _, _ in neighbor_lists[i]:
                    if i < j:
                        conn_graph.add_edge(i, j)

            # Find all simple rings (5-6 membered for aromatic systems)
            try:
                all_cycles = nx.simple_cycles(conn_graph.to_directed())
                rings_5_6 = [cycle for cycle in all_cycles if 5 <= len(cycle) <= 6]

                # Convert back to undirected (remove duplicates)
                unique_rings = []
                seen_rings = set()
                for ring in rings_5_6:
                    ring_set = frozenset(ring)
                    if ring_set not in seen_rings:
                        unique_rings.append(ring)
                        seen_rings.add(ring_set)
            except:
                # Fallback: use cycle_basis
                unique_rings = [cycle for cycle in nx.cycle_basis(conn_graph) if 5 <= len(cycle) <= 6]

            # Check each ring for aromatic character
            for ring in unique_rings:
                # Check if ring has planar geometry
                if len(ring) < 5:
                    continue

                # Calculate average bond length in ring and analyze each bond
                ring_bonds = []
                ring_bond_pairs = []  # Store (element1, element2) for each bond
                for i in range(len(ring)):
                    idx1 = ring[i]
                    idx2 = ring[(i + 1) % len(ring)]
                    dist = np.linalg.norm(positions[idx1] - positions[idx2])
                    ring_bonds.append(dist)
                    ring_bond_pairs.append((symbols[idx1], symbols[idx2]))

                avg_bond_length = np.mean(ring_bonds)
                std_bond_length = np.std(ring_bonds)

                # Calculate expected aromatic characteristics from atomic properties
                # Aromatic bonds are intermediate between single and double bonds
                expected_single = []
                expected_double = []
                for elem1, elem2 in ring_bond_pairs:
                    r_single = get_covalent_radius_by_bond_order(elem1, 1) + \
                              get_covalent_radius_by_bond_order(elem2, 1)
                    r_double = get_covalent_radius_by_bond_order(elem1, 2) + \
                              get_covalent_radius_by_bond_order(elem2, 2)
                    expected_single.append(r_single)
                    expected_double.append(r_double)

                # Expected aromatic bond length: midpoint between single and double
                avg_expected_single = np.mean(expected_single)
                avg_expected_double = np.mean(expected_double)
                expected_aromatic_length = (avg_expected_single + avg_expected_double) / 2.0

                # Tolerance: aromatic bonds can vary within the single-double range
                # Standard deviation threshold: based on bond length differences
                bond_length_range = avg_expected_single - avg_expected_double

                # Aromatic systems have delocalized electrons, so bonds should be similar
                # Allow std dev up to ~30% of the bond length range
                # (accounts for heteroatom variations like C-S vs C-C in thiazole)
                max_std_dev = bond_length_range * 0.30

                # Check if ring has aromatic character:
                # 1. Bond lengths are uniform (low std dev relative to expected variation)
                # 2. Average bond length is near the aromatic midpoint
                is_aromatic_ring = False

                if std_bond_length < max_std_dev:
                    # Allow ±15% around expected aromatic length
                    tolerance = expected_aromatic_length * 0.15
                    if (expected_aromatic_length - tolerance) < avg_bond_length < (expected_aromatic_length + tolerance):
                        is_aromatic_ring = True

                # If ring is aromatic, check for nitrogen atoms
                if is_aromatic_ring:
                    for atom_idx in ring:
                        if symbols[atom_idx] == 'N':
                            # Check if nitrogen is planar (sp²)
                            neighbors = neighbor_lists[atom_idx]
                            neighbor_indices = [j for j, _, _ in neighbors]

                            if len(neighbor_indices) >= 3:
                                if is_planar(atom_idx, neighbor_indices):
                                    # Aromatic nitrogen found!
                                    aromatic_atoms.add(atom_idx)

            # Assign charges based on hybridization and bond orders
            charges = []
            for ase_idx in range(len(atoms)):
                symbol = symbols[ase_idx]
                neighbors = neighbor_lists[ase_idx]
                n_neighbors = len(neighbors)

                # Count bond types
                n_double_bonds = sum(1 for _, order, _ in neighbors if order == 2)
                n_single_bonds = sum(1 for _, order, _ in neighbors if order == 1)
                n_h_neighbors = sum(1 for j, _, _ in neighbors if symbols[j] == 'H')
                neighbor_indices = [j for j, _, _ in neighbors]

                # Check if atom is aromatic (from RDKit)
                is_aromatic = ase_idx in aromatic_atoms

                # Determine formal charge based on bonding pattern
                if symbol == 'N':
                    if n_neighbors == 4:
                        # 4 bonds - check geometry
                        if is_tetrahedral(ase_idx, neighbor_indices):
                            # Tetrahedral sp³ → NH3+ or NH4+ (quaternary ammonium)
                            charges.append(1)
                        else:
                            # Not tetrahedral, might be distorted - check H count
                            if n_h_neighbors >= 3:
                                charges.append(1)  # Likely NH3+ or NH4+
                            else:
                                charges.append(0)  # Unusual geometry, default neutral

                    elif n_neighbors == 3:
                        # 3 bonds - distinguish sp² (planar) vs sp³ (pyramidal)
                        is_sp2 = is_planar(ase_idx, neighbor_indices)

                        # IMPORTANT: Check aromaticity FIRST before double bonds
                        # Aromatic nitrogens have "double bonds" but are actually aromatic
                        if is_aromatic:
                            # Aromatic nitrogen detected → pyridinium, imidazolium, etc.
                            # Aromatic N+ in 5-6 membered rings
                            charges.append(1)
                        elif n_double_bonds >= 1:
                            # Has true double bond (non-aromatic) → imine (=NH), neutral
                            charges.append(0)
                        elif is_sp2:
                            # Planar but not aromatic - could be conjugated system
                            # Default to neutral
                            charges.append(0)
                        elif n_h_neighbors >= 2:
                            # Pyramidal sp³ with 2+ H → NH2 (neutral)
                            charges.append(0)
                        else:
                            # Pyramidal sp³, tertiary amine → neutral
                            charges.append(0)

                    elif n_neighbors == 2:
                        # 2 bonds → could be =N- (imine) or -N≡ (nitrile), neutral
                        charges.append(0)
                    else:
                        charges.append(0)  # Default neutral

                elif symbol == 'O':
                    if n_neighbors == 3:
                        # 3 bonds → oxonium (OH3+, rare)
                        if is_tetrahedral(ase_idx, neighbor_indices) or n_h_neighbors >= 2:
                            charges.append(1)
                        else:
                            charges.append(0)
                    else:
                        charges.append(0)

                elif symbol == 'S':
                    if n_neighbors == 3 and n_single_bonds == 3:
                        # Sulfonium (SR3+) - check pyramidal geometry
                        charges.append(1)
                    else:
                        charges.append(0)
                else:
                    charges.append(0)  # C, H, and other elements default to neutral

        except Exception as e:
            # Charge inference failed, charges remain None
            LOGGER.debug(f"Charge inference failed: {e}")
            charges = None

    # Add atoms with coordinates and charges
    for ase_idx in range(len(atoms)):
        nx_idx = len(G.nodes())  # Sequential NetworkX indices

        node_data = {
            'symbol': symbols[ase_idx],
            'ase_index': ase_idx,
        }

        # Add charge if available
        if charges is not None:
            node_data['charge'] = charges[ase_idx]
        else:
            node_data['charge'] = 0  # Default to neutral

        if preserve_coords:
            node_data['position'] = positions[ase_idx].copy()

        G.add_node(nx_idx, **node_data)
        G.graph['ase_to_nx'][ase_idx] = nx_idx
        G.graph['nx_to_ase'][nx_idx] = ase_idx

    # Add bonds based on covalent distances using atomic_properties
    from ...utils.properties.atomic_properties import are_atoms_bonded

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            dist = np.linalg.norm(positions[i] - positions[j])

            # Use atomic_properties to check if atoms are bonded
            if are_atoms_bonded(dist, symbols[i], symbols[j], tolerance=0.45):
                nx_i = G.graph['ase_to_nx'][i]
                nx_j = G.graph['ase_to_nx'][j]
                G.add_edge(nx_i, nx_j, order=1.0)  # Default to single bond

    # Optionally compute partial charges via RDKit round-trip
    if compute_partial_charges and RDKIT_AVAILABLE:
        try:
            # Create temporary RDKit mol to compute charges
            # We use the graph we just built, which has formal charges inferred
            # preserve_coords=False because Gasteiger charges are topology-based
            mol_tmp = graph_to_rdkit(G, preserve_coords=False)
            
            # Compute charges
            from rdkit.Chem import AllChem
            AllChem.ComputeGasteigerCharges(mol_tmp)
            
            # Map back to graph nodes
            # graph_to_rdkit sorts nodes, and atoms_to_graph creates sequential nodes 0..N-1
            # so the mapping is direct: rdkit_atom_idx == nx_node_idx
            for i, atom in enumerate(mol_tmp.GetAtoms()):
                if atom.HasProp('_GasteigerCharge'):
                    try:
                        charge = float(atom.GetDoubleProp('_GasteigerCharge'))
                        G.nodes[i]['partial_charge'] = charge
                    except:
                        pass
        except Exception as e:
            LOGGER.warning(f"Failed to compute partial charges in atoms_to_graph: {e}")

    return G


def graph_to_atoms(graph: nx.Graph, validate_geometry: bool = False) -> Atoms:
    """Convert NetworkX graph to ASE Atoms.
    
    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph with node attributes: 'symbol', 'position'
    validate_geometry : bool, default=False
        If True, validate bond lengths (not implemented - RDKit structures already validated)
        
    Returns
    -------
    Atoms
        ASE Atoms object with symbols and positions from graph
        
    Raises
    ------
    ValueError
        If coordinates are missing and required
    """
    symbols = []
    positions = []
    
    # Sort nodes to ensure consistent ordering
    sorted_nodes = sorted(graph.nodes())
    
    for nx_idx in sorted_nodes:
        node_data = graph.nodes[nx_idx]
        symbol = node_data.get('symbol')
        if symbol is None:
            raise ValueError(f"Node {nx_idx} missing 'symbol' attribute")
        
        symbols.append(symbol)
        
        position = node_data.get('position')
        if position is None:
            # If coordinates are missing, generate them using RDKit embedding
            # This happens when new atoms are added via RDKit modifications
            try:
                # Convert graph to RDKit, generate coordinates, convert back
                temp_mol = graph_to_rdkit(graph, preserve_coords=False)
                try:
                    from rdkit.Chem import AllChem
                    AllChem.EmbedMolecule(temp_mol, randomSeed=42)
                    try:
                        AllChem.MMFFOptimizeMolecule(temp_mol)
                    except Exception:
                        pass
                except Exception:
                    from rdkit.Chem import rdDistGeom
                    try:
                        rdDistGeom.EmbedMolecule(temp_mol)
                    except Exception:
                        pass
                
                # Get coordinates from RDKit conformer
                if temp_mol.GetNumConformers() > 0:
                    conf = temp_mol.GetConformer(0)
                    # Map NetworkX index to RDKit index
                    nx_to_rdkit = graph.graph.get('nx_to_rdkit', {})
                    rdkit_idx = nx_to_rdkit.get(nx_idx)
                    if rdkit_idx is not None:
                        pos = conf.GetAtomPosition(rdkit_idx)
                        position = np.array([pos.x, pos.y, pos.z])
                
                # If still no position, use default (shouldn't happen)
                if position is None:
                    raise ValueError(
                        f"Node {nx_idx} missing 'position' attribute and could not generate coordinates. "
                        "Coordinates are required for conversion to ASE Atoms."
                    )
            except Exception as e:
                raise ValueError(
                    f"Node {nx_idx} missing 'position' attribute and coordinate generation failed: {e}. "
                    "Coordinates are required for conversion to ASE Atoms."
                )
        
        positions.append(np.array(position))
    
    if len(positions) == 0:
        raise ValueError("Graph has no nodes with positions")
    
    return Atoms(symbols=symbols, positions=np.array(positions))


# ============================================================================
# NetworkX Graph ↔ RDKit Mol
# ============================================================================

def rdkit_to_graph(mol: Chem.Mol, coords: Optional[np.ndarray] = None, compute_partial_charges: bool = False) -> nx.Graph:
    """Convert RDKit Mol to NetworkX graph with coordinate mapping.
    
    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object
    coords : np.ndarray, optional
        Alternative coordinate source (shape: [n_atoms, 3])
    compute_partial_charges : bool, default=False
        If True, compute Gasteiger partial charges using RDKit before conversion.
        Useful if the input mol doesn't have charges pre-calculated.
        
    Returns
    -------
    nx.Graph
        NetworkX graph with:
        - Nodes: sequential indices (0, 1, 2, ...)
        - Node attributes: 'symbol', 'charge' (formal), 'partial_charge', 
          'hcount', 'aromatic', 'degree', 'position', 'rdkit_index'
        - Edges: bonds with 'order' and 'aromatic' attributes
        - Graph attributes: 'rdkit_to_nx' and 'nx_to_rdkit' mappings
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for rdkit_to_graph conversion")
    
    # Optionally compute Gasteiger charges if requested
    if compute_partial_charges:
        try:
            from rdkit.Chem import AllChem
            AllChem.ComputeGasteigerCharges(mol)
        except Exception as e:
            LOGGER.warning(f"Failed to compute Gasteiger charges: {e}")

    G = nx.Graph()
    
    # Store coordinate mapping
    G.graph['rdkit_to_nx'] = {}
    G.graph['nx_to_rdkit'] = {}
    
    # Add atoms with all properties
    for atom in mol.GetAtoms():
        rdkit_idx = atom.GetIdx()
        nx_idx = len(G.nodes())  # Sequential NetworkX indices
        
        # Get coordinates from RDKit if available
        position = None
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer(0)  # Use first conformer
            pos = conf.GetAtomPosition(rdkit_idx)
            position = np.array([pos.x, pos.y, pos.z])
        elif coords is not None and rdkit_idx < len(coords):
            position = coords[rdkit_idx]
        
        # Extract partial charge if available (Gasteiger or Tripos)
        partial_charge = 0.0
        if atom.HasProp('_GasteigerCharge'):
            try:
                partial_charge = float(atom.GetDoubleProp('_GasteigerCharge'))
            except:
                pass
        elif atom.HasProp('_TriposPartialCharge'):
            try:
                partial_charge = float(atom.GetDoubleProp('_TriposPartialCharge'))
            except:
                pass

        G.add_node(nx_idx,
                  symbol=atom.GetSymbol(),
                  charge=atom.GetFormalCharge(),
                  partial_charge=partial_charge,
                  hcount=atom.GetTotalNumHs(),
                  aromatic=atom.GetIsAromatic(),
                  degree=atom.GetDegree(),
                  position=position,
                  rdkit_index=rdkit_idx)
        
        G.graph['rdkit_to_nx'][rdkit_idx] = nx_idx
        G.graph['nx_to_rdkit'][nx_idx] = rdkit_idx
    
    # Add bonds
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        nx_i = G.graph['rdkit_to_nx'][i]
        nx_j = G.graph['rdkit_to_nx'][j]
        
        bond_order = bond.GetBondTypeAsDouble()
        G.add_edge(nx_i, nx_j,
                  order=bond_order,
                  aromatic=bond.GetIsAromatic())
    
    return G


def graph_to_rdkit(graph: nx.Graph, preserve_coords: bool = True) -> Chem.Mol:
    """Convert NetworkX graph to RDKit Mol with coordinate preservation.
    
    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph with node attributes: 'symbol', 'charge', etc.
    preserve_coords : bool, default=True
        If True, preserve coordinates from graph nodes
        
    Returns
    -------
    Chem.Mol
        RDKit molecule object
        
    Raises
    ------
    ImportError
        If RDKit is not available
    ValueError
        If graph nodes are missing required attributes
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for graph_to_rdkit conversion")
    
    mol = Chem.RWMol()
    
    # Mapping from NetworkX to RDKit indices
    nx_to_rdkit = {}
    
    # Sort nodes for consistent ordering
    sorted_nodes = sorted(graph.nodes())
    
    # Add atoms
    for nx_idx in sorted_nodes:
        data = graph.nodes[nx_idx]
        symbol = data.get('symbol')
        if symbol is None:
            raise ValueError(f"Node {nx_idx} missing 'symbol' attribute")
        
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(data.get('charge', 0))
        
        # Restore partial charge if present (useful for round-tripping)
        if 'partial_charge' in data:
            atom.SetDoubleProp('_GasteigerCharge', float(data['partial_charge']))
            
        rdkit_idx = mol.AddAtom(atom)
        nx_to_rdkit[nx_idx] = rdkit_idx
    
    # Add bonds
    for nx_i, nx_j in graph.edges():
        rdkit_i = nx_to_rdkit[nx_i]
        rdkit_j = nx_to_rdkit[nx_j]
        order = int(graph.edges[nx_i, nx_j].get('order', 1))
        mol.AddBond(rdkit_i, rdkit_j, Chem.BondType(order))
    
    mol = mol.GetMol()
    
    # Add coordinates if available
    if preserve_coords:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for nx_idx in sorted_nodes:
            pos = graph.nodes[nx_idx].get('position')
            if pos is not None:
                rdkit_idx = nx_to_rdkit[nx_idx]
                conf.SetAtomPosition(rdkit_idx, Chem.rdGeometry.Point3D(*pos))
        mol.AddConformer(conf)
    
    return mol


# ============================================================================
# Coordinate Mapping Utilities
# ============================================================================

def map_coordinates(source_graph: nx.Graph, target_graph: nx.Graph) -> Dict[int, int]:
    """Map coordinates from source graph to target graph by matching atoms.
    
    Detects which atoms are unchanged between source and target graphs.
    Only atoms that match (same symbol, same connectivity pattern) are included.
    
    Parameters
    ----------
    source_graph : nx.Graph
        Original graph with coordinates
    target_graph : nx.Graph
        Modified graph (may have new/removed atoms)
        
    Returns
    -------
    Dict[int, int]
        Mapping from source graph node indices to target graph node indices
        Only includes atoms that are conserved (unchanged)
        
    Notes
    -----
    Matching is based on:
    - Symbol match
    - Connectivity pattern (neighbors' symbols)
    - This is a simple matching - can be enhanced with graph isomorphism if needed
    """
    mapping = {}
    
    # Build target graph lookup by symbol and connectivity
    target_lookup = {}
    for target_idx in target_graph.nodes():
        target_data = target_graph.nodes[target_idx]
        symbol = target_data.get('symbol')
        if symbol is None:
            continue
        
        # Get neighbor symbols for matching (exclude H for connectivity matching)
        neighbors = list(target_graph.neighbors(target_idx))
        neighbor_symbols = tuple(sorted([
            target_graph.nodes[n].get('symbol', '?')
            for n in neighbors
            if target_graph.nodes[n].get('symbol', '?') != 'H'  # Exclude H from connectivity pattern
        ]))
        
        key = (symbol, neighbor_symbols)
        if key not in target_lookup:
            target_lookup[key] = []
        target_lookup[key].append(target_idx)
    
    # Match source atoms to target atoms
    used_target_indices = set()
    
    for source_idx in source_graph.nodes():
        source_data = source_graph.nodes[source_idx]
        symbol = source_data.get('symbol')
        if symbol is None:
            continue
        
        # Get neighbor symbols (exclude H for connectivity matching)
        neighbors = list(source_graph.neighbors(source_idx))
        neighbor_symbols = tuple(sorted([
            source_graph.nodes[n].get('symbol', '?')
            for n in neighbors
            if source_graph.nodes[n].get('symbol', '?') != 'H'  # Exclude H from connectivity pattern
        ]))
        
        key = (symbol, neighbor_symbols)
        if key in target_lookup:
            # Find unused target atom with matching pattern
            for target_idx in target_lookup[key]:
                if target_idx not in used_target_indices:
                    mapping[source_idx] = target_idx
                    used_target_indices.add(target_idx)
                    break
    
    return mapping


def transfer_coordinates(
    source: nx.Graph,
    target: nx.Graph,
    mapping: Dict[int, int]
) -> nx.Graph:
    """Transfer coordinates from source graph to target graph for conserved atoms.
    
    Parameters
    ----------
    source : nx.Graph
        Source graph with original coordinates
    target : nx.Graph
        Target graph (modified, may have new atoms)
    mapping : Dict[int, int]
        Mapping from source node indices to target node indices
        (from map_coordinates())
        
    Returns
    -------
    nx.Graph
        Target graph with coordinates transferred for conserved atoms
        New atoms (not in mapping) keep their original coordinates if present,
        otherwise will be None (and will need to be calculated)
    """
    # Create a copy to avoid modifying original
    result = target.copy()
    
    # Transfer coordinates for conserved atoms
    for source_idx, target_idx in mapping.items():
        source_pos = source.nodes[source_idx].get('position')
        if source_pos is not None:
            result.nodes[target_idx]['position'] = source_pos.copy()
    
    return result


# ============================================================================
# Helper Functions
# ============================================================================

def validate_conserved_atoms(mapping: Dict[int, int], min_conserved: int = 2) -> None:
    """Validate that enough atoms are conserved for reconstruction.
    
    Parameters
    ----------
    mapping : Dict[int, int]
        Mapping from source to target graph indices
    min_conserved : int, default=2
        Minimum number of conserved atoms required
        
    Raises
    ------
    ValueError
        If less than min_conserved atoms are conserved
    """
    if len(mapping) < min_conserved:
        raise ValueError(
            f"Cannot reconstruct molecule: only {len(mapping)} atoms conserved, "
            f"but at least {min_conserved} are required. "
            "The module supports small changes/additions/replacements only. "
            "At least 2 atoms must remain unchanged for coordinate mapping."
        )


def validate_smiles(smiles: str) -> None:
    """Validate SMILES string using RDKit.
    
    Parameters
    ----------
    smiles : str
        SMILES string to validate
        
    Raises
    ------
    ImportError
        If RDKit is not available
    ValueError
        If SMILES string is invalid (MolFromSmiles returns None)
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMILES validation")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(
            f"Invalid or incompatible SMILES string: '{smiles}'. "
            "RDKit could not parse this SMILES string. "
            "Please check the SMILES syntax and ensure it is valid."
        )

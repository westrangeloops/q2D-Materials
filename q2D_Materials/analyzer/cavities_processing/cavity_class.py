"""Cavity class for representing cuboctahedral cavities in perovskite structures.

This module provides clean object-oriented interfaces for working with cavities
detected in 2D perovskite materials.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Iterator, Union, TYPE_CHECKING
import networkx as nx

if TYPE_CHECKING:
    from ..molecular_processing.spacer_analysis import SpacerAnalysisResult


class Cavity:
    """Represents a single cuboctahedral cavity in a perovskite structure.
    
    A cavity is formed by B-site octahedra surrounding an A-site cation.
    The cavity stores its structure with PBC-unwrapped coordinates so atoms
    can appear multiple times with different coordinates through periodic images.
    
    Attributes
    ----------
    id : str
        Cavity identifier (e.g., 'cavity_0')
    b_atom_indices : list
        Indices of B-site atoms forming the cavity octahedra
    x_atom_indices : list
        Indices of X-site (halide) atoms within the cavity
    a_site_indices : list
        Indices of A-site (cation) atoms contained in the cavity
    pbc_coordinates : dict
        Mapping {atom_idx: unwrapped_3d_position} for PBC-aware positions
    subgraph : nx.Graph
        NetworkX subgraph with unique node IDs for repeated atoms
    center_position : np.ndarray
        3D coordinates of cavity center
    octahedra_info : dict
        Information about octahedra forming the cavity
    contains_a_site : bool
        Whether cavity contains an A-site cation
    is_pbc_wrapped : bool
        Whether cavity spans periodic boundaries
    hull_data : dict, optional
        Cached convex hull data for volume/deformation calculations
    """
    
    def __init__(
        self,
        cavity_id: str,
        b_atom_indices: List[int],
        x_atom_indices: List[int],
        a_site_indices: List[int],
        pbc_coordinates: Dict[int, np.ndarray],
        subgraph: nx.Graph,
        center_position: np.ndarray,
        octahedra_info: Dict,
        contains_a_site: bool = False,
        is_pbc_wrapped: bool = False,
        hull_data: Optional[Dict] = None,
        cavity_type: str = 'a_site',
    ):
        """Initialize a Cavity object.
        
        Parameters
        ----------
        cavity_type : str
            Type of cavity: 'a_site' (8B+12X), 'spacer_rp' (4B+8X), or 'spacer_dj' (4B+8X)
        """
        self.id = cavity_id
        self.b_atom_indices = b_atom_indices
        self.x_atom_indices = x_atom_indices
        self.a_site_indices = a_site_indices
        self.pbc_coordinates = pbc_coordinates
        self.subgraph = subgraph
        self.center_position = center_position
        self.octahedra_info = octahedra_info
        self.contains_a_site = contains_a_site
        self.is_pbc_wrapped = is_pbc_wrapped
        self.hull_data = hull_data
        self.cavity_type = cavity_type
    
    def to_atoms(self) -> 'ase.Atoms':
        """Convert cavity to ASE Atoms object with PBC-unwrapped coordinates.
        
        Returns
        -------
        ase.Atoms
            Atoms object containing all cavity atoms with unwrapped coordinates.
            Each unique atom appearance (from different PBC images) is included.
            No periodic boundary conditions - cavity is isolated.
        """
        from ase import Atoms
        
        symbols = []
        positions = []
        
        # Extract atoms from subgraph nodes (which have unique IDs like atom_1_pbc_0)
        for node in sorted(self.subgraph.nodes()):
            node_data = self.subgraph.nodes[node]
            symbol = node_data.get('symbol', 'X')
            pbc_position = node_data.get('pbc_position')
            
            if pbc_position is not None:
                symbols.append(symbol)
                positions.append(pbc_position)
        
        # Create ASE Atoms object WITHOUT periodic boundary conditions
        # Cavities are isolated point clouds, not periodic structures
        cavity_atoms = Atoms(symbols=symbols, positions=positions)
        return cavity_atoms
    
    def get_volume(self) -> Optional[float]:
        """Get cavity volume from cached hull data.
        
        Returns
        -------
        float or None
            Volume in Ų, or None if not computed
        """
        if self.hull_data is None:
            return None
        return self.hull_data.get('hull_volume')
    
    def get_deformation(
        self, 
        b_cation: Optional[Union[str, List[str]]] = None, 
        x_anion: Optional[Union[str, List[str]]] = None,
        bx_distance: Optional[float] = None,
        mode: str = 'delta',
        full: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get cavity deformation metrics (Eta, Kappa, Nu, Omega, Delta Volume).
        
        Calculates deformation metrics using analytical formulas based on BX distance.
        B and X species are automatically extracted from the cavity subgraph if not provided.
        
        Parameters
        ----------
        b_cation : str or list of str, optional
            B-site cation symbol(s). If not provided, auto-extracted from subgraph.
        x_anion : str or list of str, optional
            X-site anion symbol(s). If not provided, auto-extracted from subgraph.
        bx_distance : float, optional
            Ideal B-X bond distance in Angstroms. If not provided, calculated from
            b_cation and x_anion using ionic radii database.
        mode : str, optional
            Output mode: 'delta' (default) or 'absolute'
            - 'delta': Returns deviations from ideal (η, κ, ν, Ω, ΔV)
            - 'absolute': Returns raw measurements without comparison
        full : bool, optional
            If True, include per-face/edge detailed data
        
        Returns
        -------
        dict or None
            Deformation metrics. Returns None if calculation fails.
            
            Delta mode (default):
                {
                    'eta': float,           # 90 - mean(square_angle_deviations)
                    'kappa': float,         # 60 - mean(triangle_angle_deviations)
                    'nu': float,            # Distance from A-site to X-cage center (Å)
                    'omega': float,         # Mean X-X edge deviation (Å)
                    'delta_volume': float,  # Volume deviation (Ų)
                }
            
            Absolute mode:
                {
                    'square_angles': {'mean': float, 'std': float, 'values': [list]},
                    'triangle_angles': {'mean': float, 'std': float, 'values': [list]},
                    'nu': float,
                    'edge_lengths': {'mean': float, 'std': float, 'values': [list], 'ideal': float},
                    'volume': float,
                    'surface_area': float,
                    'ideal_volume': float,
                    'ideal_surface_area': float,
                    'cavity_type': str,
                }
        
        Examples
        --------
        Fully automatic (delta mode):
            >>> cavity.get_deformation()
            {'eta': 88.5, 'kappa': 59.2, 'nu': 0.15, 'omega': 0.08, 'delta_volume': -2.3}
        
        Absolute measurements:
            >>> cavity.get_deformation(mode='absolute')
            {
                'square_angles': {'mean': 88.5, 'std': 2.1, 'values': [...]},
                'triangle_angles': {'mean': 59.2, 'std': 1.5, 'values': [...]},
                'nu': 0.15,
                'edge_lengths': {'mean': 4.25, 'std': 0.08, 'values': [...], 'ideal': 4.24},
                ...
            }
        
        Custom BX distance:
            >>> cavity.get_deformation(bx_distance=3.0)
            {'eta': 88.5, 'kappa': 59.2, 'nu': 0.15, 'omega': 0.08, 'delta_volume': -2.3}
        
        Full deformation data:
            >>> cavity.get_deformation(full=True)
            {
                'eta': 88.5, 'kappa': 59.2, 'nu': 0.15, 'omega': 0.08, 'delta_volume': -2.3,
                'square_angles_detail': [...],
                'triangle_angles_detail': [...],
                'edge_lengths_detail': [...],
                'cavity_type': 'a_site'
            }
        """
        # Import here to avoid circular dependency
        from .cavity_deformation import (
            calculate_cavity_deformation,
            _extract_b_species,
            _extract_x_species,
            _calculate_mean_bx_distance,
        )
        
        # Determine BX distance
        if bx_distance is None:
            # Auto-extract species if not provided
            if b_cation is None:
                try:
                    b_cation = _extract_b_species(self)
                except ValueError as e:
                    print(f"Warning: {e}")
                    return None
            
            if x_anion is None:
                try:
                    x_anion = _extract_x_species(self)
                except ValueError as e:
                    print(f"Warning: {e}")
                    return None
            
            # Calculate mean BX distance for potentially mixed compositions
            try:
                bx_distance = _calculate_mean_bx_distance(b_cation, x_anion)
            except Exception as e:
                print(f"Warning: BX distance calculation failed for {self.id}: {e}")
                return None
        
        try:
            return calculate_cavity_deformation(self, bx_distance, mode=mode, full=full)
        except Exception as e:
            print(f"Warning: Deformation calculation failed for {self.id}: {e}")
            return None
    
    def is_point_inside(self, point: np.ndarray, tolerance: float = 1e-12) -> bool:
        """Check if a point is inside the cavity.
        
        Parameters
        ----------
        point : np.ndarray
            3D coordinates to test
        tolerance : float
            Tolerance for containment check
            
        Returns
        -------
        bool
            True if point is inside cavity
        """
        if self.hull_data is None or 'hull_equations' not in self.hull_data:
            return False
        
        hull_equations = np.array(self.hull_data['hull_equations'])
        # Check if point satisfies all hull plane equations (normal . point + offset <= 0)
        return all(np.dot(eq[:-1], point) + eq[-1] <= tolerance for eq in hull_equations)
    
    @staticmethod
    def _find_nh3_groups_in_subgraph(subgraph: nx.Graph, anchor_node_id: str = None) -> List[Dict[str, Union[int, List[int]]]]:
        """Find NH3 groups within anchor nodes using graph traversal.
        
        Parameters
        ----------
        subgraph : nx.Graph
            The cavity subgraph
        anchor_node_id : str, optional
            Specific anchor node ID. If None, finds all anchor nodes.
            
        Returns
        -------
        list of dict
            List of NH3 groups: [{'n_index': int, 'h_indices': [int, int, int]}, ...]
        """
        nh3_groups = []
        
        # Find anchor nodes
        if anchor_node_id is None:
            anchor_nodes = [
                node for node in subgraph.nodes()
                if subgraph.nodes[node].get('node_type') == 'anchor'
            ]
        else:
            anchor_nodes = [anchor_node_id] if anchor_node_id in subgraph else []
        
        for anchor_node in anchor_nodes:
            # Get all atoms belonging to the anchor via CONTAINS edges
            anchor_atoms = [
                neighbor for neighbor in subgraph.neighbors(anchor_node)
                if subgraph.get_edge_data(anchor_node, neighbor).get('edge_type') == 'contains'
            ]
            
            # Identify Nitrogen atoms within this anchor
            n_nodes = [
                node for node in anchor_atoms 
                if subgraph.nodes[node].get('symbol') == 'N'
            ]
            
            for n_node in n_nodes:
                h_neighbors = []
                
                # Check bonded neighbors of the Nitrogen atom
                for neighbor in subgraph.neighbors(n_node):
                    edge_data = subgraph.get_edge_data(n_node, neighbor)
                    
                    if edge_data.get('edge_type') == 'bonded_to':
                        if subgraph.nodes[neighbor].get('symbol') == 'H':
                            atom_idx = subgraph.nodes[neighbor].get('vasp_index') or subgraph.nodes[neighbor].get('original_index')
                            if atom_idx is not None:
                                h_neighbors.append(atom_idx)
                
                # Validation: Exactly 3 Hydrogen atoms bonded to 1 Nitrogen
                if len(h_neighbors) == 3:
                    n_idx = subgraph.nodes[n_node].get('vasp_index') or subgraph.nodes[n_node].get('original_index')
                    if n_idx is not None:
                        nh3_groups.append({
                            'n_index': n_idx,
                            'h_indices': h_neighbors
                        })
                
        return nh3_groups
    
    @staticmethod
    def _calculate_penetration_depth_svd(
        membrane_points: np.ndarray, 
        query_point: np.ndarray
    ) -> tuple:
        """Calculate signed distance (penetration) of a query point relative to 
        the best-fit plane of 4 membrane points using SVD.
        
        Parameters
        ----------
        membrane_points : np.ndarray
            4x3 array of terminal atom coordinates
        query_point : np.ndarray
            1x3 array of the NH3 center position
            
        Returns
        -------
        tuple
            (signed_distance, plane_centroid, plane_normal)
        """
        if membrane_points.shape != (4, 3):
            raise ValueError(f"Expected 4x3 array of membrane points, got {membrane_points.shape}")
        
        # Calculate centroid of the 4 points
        centroid = np.mean(membrane_points, axis=0)
        
        # Center the points
        centered_points = membrane_points - centroid
        
        # Compute surface normal using SVD
        u, s, vh = np.linalg.svd(centered_points, full_matrices=True)
        normal = vh[-1, :]  # Last row corresponds to smallest singular value
        
        # Enforce consistent orientation (upward-pointing normal)
        if normal[2] < 0:
            normal = -normal
        
        # Normalize to unit vector
        normal = normal / np.linalg.norm(normal)
        
        # Calculate signed distance
        vec_to_point = query_point - centroid
        signed_distance = np.dot(vec_to_point, normal)
        
        return float(signed_distance), centroid, normal
    
    def calculate_penetration_depth(self) -> Dict[str, Any]:
        """Calculate NH3 penetration depth into terminal atom planes.
        
        Only applicable to spacer cavities (spacer_rp and spacer_dj).
        For RP spacers: calculates penetration of 1 NH3 into 4 terminal atoms.
        For DJ spacers: calculates penetration of 2 NH3 groups into their respective planes.
        
        All required data (atom positions, symbols, terminal atoms) are extracted from
        the cavity's internal subgraph representation.
            
        Returns
        -------
        dict
            For RP spacers:
                {
                    'penetration_depth': float,
                    'nh3_position': list,
                    'plane_centroid': list,
                    'plane_normal': list,
                    'terminal_atom_indices': list,
                    'cavity_type': 'spacer_rp'
                }
            
            For DJ spacers:
                {
                    'top_penetration': float,
                    'bottom_penetration': float,
                    'top_nh3_position': list,
                    'bottom_nh3_position': list,
                    'top_plane_centroid': list,
                    'bottom_plane_centroid': list,
                    'top_plane_normal': list,
                    'bottom_plane_normal': list,
                    'top_terminal_atoms': list,
                    'bottom_terminal_atoms': list,
                    'cavity_type': 'spacer_dj'
                }
            
        Raises
        ------
        ValueError
            If cavity is not a spacer cavity (spacer_rp or spacer_dj)
        """
        # Validate cavity type
        if self.cavity_type not in ('spacer_rp', 'spacer_dj'):
            raise ValueError(
                f"calculate_penetration_depth only works for spacer cavities "
                f"(spacer_rp or spacer_dj), but got cavity type: {self.cavity_type}"
            )
        
        # Extract atom data from subgraph using node IDs (not indices, since PBC images exist)
        # Use node_id as key to handle PBC images correctly
        node_positions = {}  # node_id -> position
        node_symbols = {}    # node_id -> symbol
        
        for node in self.subgraph.nodes():
            node_data = self.subgraph.nodes[node]
            if node_data.get('node_type') == 'atom':
                pbc_pos = node_data.get('pbc_position')
                symbol = node_data.get('symbol')
                if pbc_pos is not None and symbol is not None:
                    node_positions[node] = np.array(pbc_pos)
                    node_symbols[node] = symbol
        
        # Extract TERMINAL X atoms from the cavity subgraph
        # Terminal atoms have is_terminal=True (computed from parent graph)
        # Store as list of (node_id, position) tuples to handle PBC images
        terminal_x_nodes = []  # List of (node_id, position)
        all_x_nodes = []       # List of (node_id, position) for fallback
        
        for node in self.subgraph.nodes():
            node_data = self.subgraph.nodes[node]
            if node_data.get('node_type') == 'atom':
                symbol = node_data.get('symbol', '')
                
                # Check if this is an X atom (halide/chalcogen)
                if symbol in ('F', 'Cl', 'Br', 'I', 'S', 'Se', 'Te', 'O'):
                    pbc_pos = node_data.get('pbc_position')
                    if pbc_pos is not None:
                        pos = np.array(pbc_pos)
                        all_x_nodes.append((node, pos))
                        # Check if terminal
                        if node_data.get('is_terminal', False):
                            terminal_x_nodes.append((node, pos))
        
        # Calculate NH3 centers using graph-based detection
        # Find anchor nodes in subgraph (for spacer cavities)
        anchor_nodes = []
        for node in self.subgraph.nodes():
            node_data = self.subgraph.nodes[node]
            if node_data.get('node_type') == 'anchor':
                anchor_nodes.append(node)
        
        if not anchor_nodes:
            raise ValueError(
                f"Could not find anchor nodes in subgraph. Cannot use graph-based NH3 detection."
            )
        
        # Extract NH3 groups directly from anchor node connections
        # Each anchor node is connected to exactly 1 N atom and 3 H atoms
        nh3_centers = []
        for anchor_node in anchor_nodes:
            # Get all atoms connected to this anchor node via 'contains' edge
            anchor_atoms = [
                neighbor for neighbor in self.subgraph.neighbors(anchor_node)
                if self.subgraph.get_edge_data(anchor_node, neighbor).get('edge_type') == 'contains'
            ]
            
            # Find N and H atoms directly from connected atoms
            n_pos = None
            h_positions = []
            
            for atom_node in anchor_atoms:
                atom_data = self.subgraph.nodes[atom_node]
                symbol = atom_data.get('symbol', '')
                pbc_pos = atom_data.get('pbc_position')
                
                if pbc_pos is None:
                    continue
                
                if symbol == 'N':
                    n_pos = np.array(pbc_pos)
                elif symbol == 'H':
                    h_positions.append(np.array(pbc_pos))
            
            # Calculate NH3 center from N and 3 H atoms directly connected to anchor
            if n_pos is not None and len(h_positions) == 3:
                # Center of mass: (N + 3H) / 4
                center = (n_pos + np.sum(h_positions, axis=0)) / 4.0
                nh3_centers.append(center)
            else:
                # Log warning if anchor doesn't have expected structure
                import sys
                print(
                    f"Warning: Anchor {anchor_node} does not have expected NH3 structure "
                    f"(found N: {n_pos is not None}, H count: {len(h_positions)}). "
                    f"Skipping this anchor.",
                    file=sys.stderr
                )
        
        if not nh3_centers:
            raise ValueError(
                f"Could not calculate NH3 centers from {len(anchor_nodes)} anchor nodes found in spacer cavity"
            )
        
        # Helper function to calculate penetration using positions directly
        def calculate_single_penetration(terminal_positions_array, nh3_pos):
            if len(terminal_positions_array) != 4:
                raise ValueError(f"Expected 4 terminal atom positions, got {len(terminal_positions_array)}")
            
            # Calculate penetration depth using SVD
            signed_dist, plane_centroid, plane_normal = self._calculate_penetration_depth_svd(
                terminal_positions_array, nh3_pos
            )
            
            return signed_dist, plane_centroid, plane_normal
        
        # Handle RP spacers (4 X atoms, 1 NH3)
        if self.cavity_type == 'spacer_rp':
            # Prefer terminal atoms if found, otherwise use all X atoms
            # For RP spacers, all 4 X atoms should be terminal
            membrane_nodes = terminal_x_nodes if len(terminal_x_nodes) >= 4 else all_x_nodes
            
            if len(membrane_nodes) < 4:
                raise ValueError(
                    f"RP spacer cavity should have at least 4 terminal/X atoms, "
                    f"but found {len(membrane_nodes)} (terminal: {len(terminal_x_nodes)}, X: {len(all_x_nodes)})"
                )
            
            # Use the first 4 atoms as the membrane
            membrane_positions = np.array([pos for _, pos in membrane_nodes[:4]])
            membrane_node_ids = [node for node, _ in membrane_nodes[:4]]
            nh3_pos = np.array(nh3_centers[0])
            
            signed_dist, plane_centroid, plane_normal = calculate_single_penetration(
                membrane_positions, nh3_pos
            )
            
            return {
                'penetration_depth': signed_dist,
                'nh3_position': nh3_pos.tolist(),
                'plane_centroid': plane_centroid.tolist(),
                'plane_normal': plane_normal.tolist(),
                'terminal_atom_nodes': membrane_node_ids,
                'cavity_type': 'spacer_rp'
            }
        
        # Handle DJ spacers (8 terminal atoms - 4 top, 4 bottom; 2 NH3)
        elif self.cavity_type == 'spacer_dj':
            # Prefer terminal atoms if found, otherwise use all X atoms
            # For DJ spacers, we need 8 X atoms (4 top terminal, 4 bottom terminal)
            membrane_nodes = terminal_x_nodes if len(terminal_x_nodes) >= 8 else all_x_nodes
            
            if len(membrane_nodes) < 8:
                raise ValueError(
                    f"DJ spacer cavity should have at least 8 terminal/X atoms, "
                    f"but found {len(membrane_nodes)} (terminal: {len(terminal_x_nodes)}, X: {len(all_x_nodes)})"
                )
            
            if len(nh3_centers) < 2:
                raise ValueError(
                    f"DJ spacer cavity should have at least 2 NH3 groups, "
                    f"but found {len(nh3_centers)}"
                )
            
            # Get positions of all membrane atoms
            membrane_positions = np.array([pos for _, pos in membrane_nodes])
            membrane_node_ids = [node for node, _ in membrane_nodes]
            
            # Sort membrane atoms by Z-coordinate
            z_coords = membrane_positions[:, 2]
            sorted_indices = np.argsort(z_coords)
            sorted_z = z_coords[sorted_indices]
            
            # Find the two natural Z-planes by clustering
            # For DJ: expect two groups of 4 atoms with gap between them
            # Take lowest 4 and highest 4 (they form natural planes)
            bottom_indices = sorted_indices[:4]
            top_indices = sorted_indices[-4:]
            
            bottom_positions = membrane_positions[bottom_indices]
            top_positions = membrane_positions[top_indices]
            bottom_node_ids = [membrane_node_ids[i] for i in bottom_indices]
            top_node_ids = [membrane_node_ids[i] for i in top_indices]
            
            # Get bottom and top plane Z-coordinates
            bottom_plane_z = np.mean(sorted_z[:4])
            top_plane_z = np.mean(sorted_z[-4:])
            
            # Get NH3 centers
            nh3_pos_array = np.array(nh3_centers[:2])
            nh3_z_coords = nh3_pos_array[:, 2]
            
            # Match each NH3 to the nearest plane by Z-distance
            nh3_0_dist_to_bottom = abs(nh3_z_coords[0] - bottom_plane_z)
            nh3_0_dist_to_top = abs(nh3_z_coords[0] - top_plane_z)
            
            # Assign each NH3 to its closest plane
            if nh3_0_dist_to_bottom < nh3_0_dist_to_top:
                # NH3 0 is bottom, NH3 1 is top
                top_nh3_pos = nh3_pos_array[1]
                bottom_nh3_pos = nh3_pos_array[0]
            else:
                # NH3 0 is top, NH3 1 is bottom
                top_nh3_pos = nh3_pos_array[0]
                bottom_nh3_pos = nh3_pos_array[1]
            
            # Calculate penetrations
            top_dist, top_centroid, top_normal = calculate_single_penetration(
                top_positions, top_nh3_pos
            )
            bottom_dist, bottom_centroid, bottom_normal = calculate_single_penetration(
                bottom_positions, bottom_nh3_pos
            )
            
            return {
                'top_penetration': top_dist,
                'bottom_penetration': bottom_dist,
                'top_nh3_position': top_nh3_pos.tolist(),
                'bottom_nh3_position': bottom_nh3_pos.tolist(),
                'top_plane_centroid': top_centroid.tolist(),
                'bottom_plane_centroid': bottom_centroid.tolist(),
                'top_plane_normal': top_normal.tolist(),
                'bottom_plane_normal': bottom_normal.tolist(),
                'top_terminal_nodes': top_node_ids,
                'bottom_terminal_nodes': bottom_node_ids,
                'cavity_type': 'spacer_dj'
            }
    
    def get_spacer_analysis(self, spacer_type: Optional[str] = None) -> 'SpacerAnalysisResult':  # type: ignore
        """Get spacer analysis results for this cavity.
        
        Only works for spacer cavities (spacer_rp and spacer_dj).
        Creates a SpacerAnalysis instance using this cavity's PBC-unwrapped
        molecule data and returns the analysis results.
        
        Parameters
        ----------
        spacer_type : str, optional
            Type of spacer ("DJ" or "RP"). If None, inferred from cavity_type.
            
        Returns
        -------
        SpacerAnalysisResult
            Complete spacer analysis with compression, volume, backbone, etc.
            
        Raises
        ------
        ValueError
            If cavity is not a spacer cavity (spacer_rp or spacer_dj)
            
        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.cif")
        >>> analyzer.analyze()
        >>> cavities = analyzer.get_cavities()
        >>> 
        >>> # Get spacer analysis for a DJ spacer cavity
        >>> dj_cavity = cavities.filter(cavity_type='spacer_dj')[0]
        >>> analysis = dj_cavity.get_spacer_analysis()
        >>> print(f"Compression: {analysis.compression:.2f} Å")
        >>> print(f"Backbone length: {analysis.backbone_length}")
        """
        if self.cavity_type not in ('spacer_rp', 'spacer_dj'):
            raise ValueError(
                f"get_spacer_analysis only works for spacer cavities "
                f"(spacer_rp or spacer_dj), but got cavity type: {self.cavity_type}"
            )
        
        # Import here to avoid circular dependency
        from ..molecular_processing.spacer_analysis import SpacerAnalysis
        from ...modifier.molecule_graph import MoleculeGraph
        
        # Determine spacer type
        if spacer_type is None:
            if self.cavity_type == 'spacer_dj':
                spacer_type = 'DJ'
            elif self.cavity_type == 'spacer_rp':
                spacer_type = 'RP'
            else:
                spacer_type = 'DJ'  # Default
        
        # Extract molecule graph from cavity
        # Create a minimal MoleculeGraph wrapper for compatibility
        # The SpacerAnalysis will extract the graph from cavity
        molecule_indices = self.a_site_indices
        
        # Create a dummy MoleculeGraph - SpacerAnalysis will use cavity graph instead
        import networkx as nx
        dummy_graph = nx.Graph()
        dummy_mol_graph = MoleculeGraph(
            graph=dummy_graph,
            original_indices=molecule_indices,
            attachment_points=[],
            molecule_type='spacer',
            molecule_index=0,
            atoms_object=None,
            parent_view=None
        )
        
        # Create SpacerAnalysis with cavity
        analyzer = SpacerAnalysis(
            molecule_graph=dummy_mol_graph,
            analyzer=None,  # Not needed when cavity is provided
            spacer_type=spacer_type,
            cavity=self
        )
        
        return analyzer.compute()
    
    def __repr__(self) -> str:
        """String representation of Cavity."""
        return (
            f"Cavity(id={self.id}, "
            f"type={self.cavity_type}, "
            f"b_atoms={len(self.b_atom_indices)}, "
            f"x_atoms={len(self.x_atom_indices)}, "
            f"a_sites={len(self.a_site_indices)}, "
            f"contains_a_site={self.contains_a_site})"
        )
    
    def __getitem__(self, key: str) -> Any:
        """Dict-like access for backward compatibility.
        
        Example: cavity['b_atom_indices']
        """
        return getattr(self, key)


class CavityCollection:
    """Collection of Cavity objects with convenient methods.
    
    Provides list-like access and bulk operations on multiple cavities.
    
    Attributes
    ----------
    cavities : list of Cavity
        The cavities in this collection
    """
    
    def __init__(self, cavities: List[Cavity]):
        """Initialize CavityCollection.
        
        Parameters
        ----------
        cavities : list of Cavity
            List of Cavity objects
        """
        self._cavities = cavities
    
    def to_atoms(self) -> List['ase.Atoms']:
        """Convert all cavities to ASE Atoms objects.
        
        Returns
        -------
        list of ase.Atoms
            List of Atoms objects, one per cavity
        """
        return [cav.to_atoms() for cav in self._cavities]
    
    def filter(self, **kwargs) -> 'CavityCollection':
        """Filter cavities by attributes.
        
        Parameters
        ----------
        **kwargs
            Attribute filters (e.g., contains_a_site=True, is_pbc_wrapped=False)
            
        Returns
        -------
        CavityCollection
            New collection with filtered cavities
        """
        filtered = [
            cav for cav in self._cavities
            if all(getattr(cav, k, None) == v for k, v in kwargs.items())
        ]
        return CavityCollection(filtered)
    
    def __len__(self) -> int:
        """Number of cavities in collection."""
        return len(self._cavities)
    
    def __getitem__(self, idx: int) -> Cavity:
        """Get cavity by index."""
        return self._cavities[idx]
    
    def __iter__(self) -> Iterator[Cavity]:
        """Iterate over cavities."""
        return iter(self._cavities)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"CavityCollection({len(self._cavities)} cavities)"


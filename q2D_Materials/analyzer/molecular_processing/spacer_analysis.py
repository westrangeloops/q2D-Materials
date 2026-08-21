"""Spacer molecular analysis for DJ and RP perovskites.

This module provides comprehensive analysis of spacer molecules including:
- Penetration depth (N to halogen Z-distance)
- Linker compression (N-N through-space vs path length)
- Backbone identification (longest path)
- Side chain detection (branching points)
"""

from __future__ import annotations

from typing import List, Dict, Optional, Set, Tuple, TYPE_CHECKING, Any, Union
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import networkx as nx

try:
    from scipy.spatial import ConvexHull
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    ConvexHull = None

try:
    from ase.data import vdw_radii, atomic_numbers, atomic_masses
    ASE_DATA_AVAILABLE = True
except ImportError:
    ASE_DATA_AVAILABLE = False
    vdw_radii = None
    atomic_numbers = None
    atomic_masses = None

from ..utils.graph_utils import (
    find_all_paths,
    validate_path_continuity,
)
from ...utils.properties.atomic_properties import get_covalent_radius

if TYPE_CHECKING:
    from ...modifier.molecule_graph import MoleculeGraph
    from ..core.analyzer_class import q2D_analyzer
    from ..cavities_processing.cavity_class import Cavity


class BackboneQuery:
    """Queryable backbone interface for chainable analysis.

    Provides methods to query backbone properties like dihedral angles,
    elements, positions, and bond information.

    Parameters
    ----------
    result : SpacerAnalysisResult
        Parent analysis result containing backbone data
    """

    def __init__(self, result: 'SpacerAnalysisResult'):
        """Initialize backbone query interface."""
        self.result = result
        self.graph = result.graph
        self.backbone_path = result.backbone_path

    def dihedrals(self) -> List[float]:
        """Calculate dihedral angles along backbone.

        Computes dihedral angles for each set of 4 consecutive atoms
        in the backbone path.

        Returns
        -------
        List[float]
            Dihedral angles in degrees. Length is len(backbone) - 3.

        Notes
        -----
        Dihedral angle is defined as the angle between planes formed by
        atoms (i, i+1, i+2) and (i+1, i+2, i+3).
        """
        if len(self.backbone_path) < 4:
            return []

        dihedrals = []

        for i in range(len(self.backbone_path) - 3):
            idx1 = self.backbone_path[i]
            idx2 = self.backbone_path[i + 1]
            idx3 = self.backbone_path[i + 2]
            idx4 = self.backbone_path[i + 3]

            # Get positions
            p1 = self.graph.nodes[idx1]['position']
            p2 = self.graph.nodes[idx2]['position']
            p3 = self.graph.nodes[idx3]['position']
            p4 = self.graph.nodes[idx4]['position']

            # Calculate dihedral angle
            dihedral = self._calculate_dihedral(p1, p2, p3, p4)
            dihedrals.append(dihedral)

        return dihedrals

    def _calculate_dihedral(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        p4: np.ndarray
    ) -> float:
        """Calculate dihedral angle between four points.

        Parameters
        ----------
        p1, p2, p3, p4 : np.ndarray
            Positions of four atoms

        Returns
        -------
        float
            Dihedral angle in degrees
        """
        # Vectors along bonds
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3

        # Normal vectors to planes
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        # Normalize
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)

        if n1_norm < 1e-10 or n2_norm < 1e-10:
            return 0.0

        n1 = n1 / n1_norm
        n2 = n2 / n2_norm

        # Calculate angle
        cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # Determine sign
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
        sign = np.sign(np.dot(m1, n2))

        return float(np.degrees(angle) * sign)

    def elements(self) -> List[str]:
        """Get element symbols along backbone.

        Returns
        -------
        List[str]
            Element symbols in backbone order
        """
        return [
            self.graph.nodes[idx].get('symbol', 'C')
            for idx in self.backbone_path
        ]

    def positions(self) -> np.ndarray:
        """Get atomic positions along backbone.

        Returns
        -------
        np.ndarray
            Array of shape (n_atoms, 3) with xyz coordinates
        """
        positions = [
            self.graph.nodes[idx]['position']
            for idx in self.backbone_path
        ]
        return np.array(positions)

    def indices(self) -> List[int]:
        """Get atom indices along backbone.

        Returns
        -------
        List[int]
            Atom indices in original structure
        """
        return self.backbone_path.copy()

    def bond_lengths(self) -> List[float]:
        """Calculate bond lengths along backbone.

        Returns
        -------
        List[float]
            Bond lengths in Angstroms. Length is len(backbone) - 1.
        """
        if len(self.backbone_path) < 2:
            return []

        bond_lengths = []

        for i in range(len(self.backbone_path) - 1):
            idx1 = self.backbone_path[i]
            idx2 = self.backbone_path[i + 1]

            p1 = self.graph.nodes[idx1]['position']
            p2 = self.graph.nodes[idx2]['position']

            length = np.linalg.norm(p1 - p2)
            bond_lengths.append(float(length))

        return bond_lengths

    def bond_angles(self) -> List[float]:
        """Calculate bond angles along backbone.

        Computes angles for each set of 3 consecutive atoms.

        Returns
        -------
        List[float]
            Bond angles in degrees. Length is len(backbone) - 2.
        """
        if len(self.backbone_path) < 3:
            return []

        bond_angles = []

        for i in range(len(self.backbone_path) - 2):
            idx1 = self.backbone_path[i]
            idx2 = self.backbone_path[i + 1]
            idx3 = self.backbone_path[i + 2]

            p1 = self.graph.nodes[idx1]['position']
            p2 = self.graph.nodes[idx2]['position']
            p3 = self.graph.nodes[idx3]['position']

            # Calculate angle
            v1 = p1 - p2
            v2 = p3 - p2

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)

            bond_angles.append(float(np.degrees(angle)))

        return bond_angles

    def filter(self, element: Optional[str] = None) -> List[int]:
        """Filter backbone atoms by criteria.

        Parameters
        ----------
        element : str, optional
            Filter by element symbol (e.g., 'C', 'N')

        Returns
        -------
        List[int]
            Filtered atom indices
        """
        filtered = []

        for idx in self.backbone_path:
            node_data = self.graph.nodes[idx]

            # Check element filter
            if element is not None:
                if node_data.get('symbol') != element:
                    continue

            filtered.append(idx)

        return filtered

    def to_dict(self) -> Dict[str, Any]:
        """Convert backbone data to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary with all backbone properties
        """
        return {
            'indices': self.indices(),
            'elements': self.elements(),
            'positions': self.positions().tolist(),
            'bond_lengths': self.bond_lengths(),
            'bond_angles': self.bond_angles(),
            'dihedrals': self.dihedrals(),
            'length': len(self.backbone_path)
        }


@dataclass
class SpacerAnalysisResult:
    """Complete analysis results for a DJ spacer molecule.

    Attributes
    ----------
    euclidean_distance : float
        Through-space N-N distance (Å)
    path_length : float
        Sum of bond lengths along backbone (Å)
    ideal_extended_length : float
        Ideal extended end-to-end distance accounting for sp3 geometry (Å)
        Calculated as path_length × 0.85 (typical ratio for fully extended sp3 chain)
    compression : float
        Compression distance: ideal_extended_length - euclidean_distance (Å)
        - 0.0 Å: fully extended (euclidean matches ideal)
        - > 0 Å: compressed by X Angstroms (molecule shorter than ideal)
        - < 0 Å: overstretched by |X| Angstroms (molecule longer than ideal)
    vdw_volume : float
        Van der Waals volume: sum of atomic volumes using vdW radii (Å³)
        V = Σ(4/3 × π × r_vdw³) for all atoms
    convex_hull_volume : float
        Convex hull volume: volume of smallest convex shape containing all atoms (Å³)
        Calculated using scipy.spatial.ConvexHull
    radius_gyration : float
        Radius of gyration: RMS distance of atoms from center of mass (Å)
        R_g = sqrt(Σ(m_i * |r_i - r_com|²) / Σ(m_i))
        Measures molecular compactness (0.0 if ASE unavailable)
    backbone_path : List[int]
        Atom indices in longest path
    backbone_length : int
        Number of atoms in backbone
    backbone_symbols : List[str]
        Element symbols along backbone
    side_chains : List[Dict]
        Side chain data: [{'attachment_idx', 'chain_atoms', 'length'}]
    branching_points : List[int]
        Atoms with degree >= 3
    n_side_chains : int
        Total number of side chains
    molecule_indices : List[int]
        Original structure indices
    terminal_nitrogens : List[int]
        N atom indices (NH3+ groups)
    spacer_type : str
        "DJ" or "RP"
    graph : nx.Graph
        Molecular graph reference
    """
    # Feature 1: Linker Distance & Compression
    euclidean_distance: float = 0.0
    path_length: float = 0.0  # Sum of bond lengths along backbone
    ideal_extended_length: float = 0.0  # Path length × 0.85 (typical for fully extended sp3)
    compression: float = field(default=float('nan'))  # ideal_extended_length - euclidean_distance (Å)
    
    # Feature 1b: Molecular Volume
    vdw_volume: float = 0.0  # Van der Waals volume (sum of atomic volumes) (Å³)
    convex_hull_volume: float = 0.0  # Convex hull volume (Å³)
    radius_gyration: float = 0.0  # Radius of gyration (Å)

    # Feature 2: Backbone Identification
    backbone_path: List[int] = field(default_factory=list)
    backbone_length: int = 0
    backbone_symbols: List[str] = field(default_factory=list)

    # Feature 3: Side Chains
    side_chains: List[Dict] = field(default_factory=list)
    branching_points: List[int] = field(default_factory=list)
    n_side_chains: int = 0

    # Metadata
    molecule_indices: List[int] = field(default_factory=list)
    terminal_nitrogens: List[int] = field(default_factory=list)
    spacer_type: str = "DJ"
    graph: Optional[nx.Graph] = None

    @property
    def is_compressed(self) -> bool:
        """Whether spacer is significantly compressed (compression > 1.0 Å).

        Compression interpretation:
        - 0.0 Å: fully extended (euclidean matches ideal for sp3 chain)
        - > 0 Å: compressed (molecule shorter than ideal extended)
        - < 0 Å: overstretched (molecule longer than ideal)

        Threshold of 1.0 Å indicates significant compression.
        """
        return self.compression > 1.0

    @property
    def compression_factor(self) -> float:
        """Alias for compression (backward compatibility).

        Returns
        -------
        float
            Compression distance in Angstroms (same as compression attribute)
        """
        return self.compression

    @property
    def backbone(self) -> BackboneQuery:
        """Get queryable backbone interface.

        Returns
        -------
        BackboneQuery
            Interface for querying backbone properties

        Examples
        --------
        >>> result = spacer.analyze_spacer()
        >>> dihedrals = result.backbone.dihedrals()
        >>> elements = result.backbone.elements()
        >>> positions = result.backbone.positions()
        >>> bond_lengths = result.backbone.bond_lengths()
        """
        return BackboneQuery(self)

    def __repr__(self) -> str:
        """Readable string representation."""
        return (
            f"SpacerAnalysisResult(\n"
            f"  Compression: euclidean={self.euclidean_distance:.2f} Å, "
            f"ideal_extended={self.ideal_extended_length:.2f} Å, "
            f"compression={self.compression:+.2f} Å\n"
            f"  Volume: vdw={self.vdw_volume:.2f} Å³, "
            f"hull={self.convex_hull_volume:.2f} Å³, "
            f"R_g={self.radius_gyration:.2f} Å\n"
            f"  Backbone: {self.backbone_length} atoms, "
            f"{''.join(self.backbone_symbols[:10])}{'...' if len(self.backbone_symbols) > 10 else ''}\n"
            f"  Side chains: {self.n_side_chains} detected\n"
            f")"
        )


class SpacerAnalysis:
    """Analyzer for DJ spacer molecular properties.

    This class computes structural and geometric properties of DJ spacer molecules
    including compression, volume, backbone, and side chains.

    Parameters
    ----------
    molecule_graph : MoleculeGraph
        Molecule to analyze (from GraphView or MoleculeGraph)
    analyzer : q2D_analyzer, optional
        Full analyzer instance (kept for API compatibility)
    spacer_type : str, default="DJ"
        Type of spacer ("DJ" or "RP")

    Examples
    --------
    >>> from q2D_Materials.analyzer import q2D_analyzer
    >>> from q2D_Materials.modifier import GraphView
    >>>
    >>> analyzer = q2D_analyzer("structure.cif")
    >>> analyzer.analyze()
    >>> view = GraphView(analyzer)
    >>>
    >>> # Analyze specific spacer
    >>> spacer = view.spacers[0]
    >>> analysis = SpacerAnalysis(spacer, analyzer).compute()
    >>> print(analysis.compression_factor)
    """

    def __init__(
        self,
        molecule_graph: 'MoleculeGraph',
        analyzer: Optional['q2D_analyzer'] = None,
        spacer_type: str = "DJ",
        cavity: Optional['Cavity'] = None
    ):
        """Initialize spacer analyzer.
        
        Parameters
        ----------
        molecule_graph : MoleculeGraph
            Molecule to analyze (from GraphView or MoleculeGraph)
        analyzer : q2D_analyzer, optional
            Full analyzer instance (kept for API compatibility)
        spacer_type : str, default="DJ"
            Type of spacer ("DJ" or "RP")
        cavity : Cavity, optional
            Cavity object containing PBC-unwrapped molecule data.
            If provided, uses cavity subgraph for analysis (structure-based mode).
            If None, attempts to find matching cavity from analyzer.
            If no cavity available, uses current structure graph approach (standalone mode).
        """
        self.molecule_graph = molecule_graph
        self.analyzer = analyzer
        self.spacer_type = spacer_type.upper()
        
        # Cavity support
        self._cavity = cavity
        if self._cavity is None:
            # Try to find matching cavity from analyzer
            self._cavity = self._find_matching_cavity()
        
        # Determine which mode to use
        self._use_cavity_mode = self._cavity is not None
        
        # Extract molecule graph from cavity if in cavity mode
        if self._use_cavity_mode:
            self.graph = self._extract_molecule_graph_from_cavity()
        else:
            self.graph = molecule_graph.graph

        # Caches
        self._result_cache: Optional[SpacerAnalysisResult] = None
        self._terminal_N_cache: Optional[List[int]] = None
        self._backbone_cache: Optional[List[int]] = None

    def _find_matching_cavity(self) -> Optional['Cavity']:
        """Find matching cavity from analyzer for this molecule.
        
        Attempts to find a cavity that contains the same molecule atoms
        by matching original_indices with cavity a_site_indices.
        
        Returns
        -------
        Cavity or None
            Matching cavity if found, None otherwise
        """
        if self.analyzer is None:
            return None
        
        try:
            cavities = self.analyzer.get_cavities()
        except Exception:
            return None
        
        if not cavities:
            return None
        
        # Get molecule indices
        molecule_indices = set(self.molecule_graph.original_indices)
        
        # Find cavity with matching spacer type and overlapping indices
        for cavity in cavities:
            # Check if cavity type matches spacer type
            if self.spacer_type == "DJ" and cavity.cavity_type != "spacer_dj":
                continue
            if self.spacer_type == "RP" and cavity.cavity_type != "spacer_rp":
                continue
            
            # Check if cavity contains this molecule
            cavity_indices = set(cavity.a_site_indices)
            if molecule_indices.issubset(cavity_indices) or molecule_indices == cavity_indices:
                return cavity
        
        return None
    
    def _extract_molecule_graph_from_cavity(self) -> nx.Graph:
        """Extract molecule graph from cavity subgraph with PBC-unwrapped positions.
        
        Converts cavity subgraph nodes (atom_{idx}_img_{i}_{j}_{k}) to simple
        integer node IDs (original_indices) and extracts PBC-unwrapped positions.
        Uses the first available image for each atom (typically (0,0,0)).
        
        Returns
        -------
        nx.Graph
            Molecule graph with integer node IDs and PBC-unwrapped positions
        """
        if self._cavity is None:
            raise ValueError("Cannot extract molecule graph: no cavity available")
        
        # Create new graph
        mol_graph = nx.Graph()
        
        # Extract molecule atoms from cavity subgraph
        molecule_indices = set(self.molecule_graph.original_indices)
        
        # Map from original_index to cavity node (use first image encountered)
        index_to_cavity_node = {}  # original_idx -> (cavity_node_id, node_data)
        
        for node_id in self._cavity.subgraph.nodes():
            node_data = self._cavity.subgraph.nodes[node_id]
            
            # Only process atom nodes
            if node_data.get('node_type') != 'atom':
                continue
            
            # Get original index
            original_idx = node_data.get('original_index')
            if original_idx is None:
                original_idx = node_data.get('vasp_index')
            
            if original_idx is None or original_idx not in molecule_indices:
                continue
            
            # Get PBC-unwrapped position
            pbc_pos = node_data.get('pbc_position')
            if pbc_pos is None:
                continue
            
            # Use first image encountered (skip if already have one)
            if original_idx not in index_to_cavity_node:
                index_to_cavity_node[original_idx] = (node_id, node_data)
        
        # Add nodes to graph with integer IDs
        for original_idx, (cavity_node_id, node_data) in index_to_cavity_node.items():
            # Copy node attributes, use pbc_position as position
            new_node_data = {k: v for k, v in node_data.items() 
                           if k not in ('x', 'y', 'z', 'position', 'image_label')}
            new_node_data['position'] = np.array(node_data.get('pbc_position'))
            new_node_data['original_index'] = original_idx
            
            mol_graph.add_node(original_idx, **new_node_data)
        
        # Add edges from cavity subgraph (already has all bonds)
        for original_idx1, (cavity_node_id1, _) in index_to_cavity_node.items():
            for neighbor_cavity_id in self._cavity.subgraph.neighbors(cavity_node_id1):
                neighbor_data = self._cavity.subgraph.nodes[neighbor_cavity_id]
                
                # Check if neighbor is a molecule atom
                neighbor_original_idx = neighbor_data.get('original_index')
                if neighbor_original_idx is None:
                    neighbor_original_idx = neighbor_data.get('vasp_index')
                
                if neighbor_original_idx is None or neighbor_original_idx not in molecule_indices:
                    continue
                
                if neighbor_original_idx not in index_to_cavity_node:
                    continue
                
                # Check if edge represents a bond
                edge_data = self._cavity.subgraph.get_edge_data(cavity_node_id1, neighbor_cavity_id)
                if edge_data and edge_data.get('edge_type') == 'bonded_to':
                    if not mol_graph.has_edge(original_idx1, neighbor_original_idx):
                        mol_graph.add_edge(original_idx1, neighbor_original_idx, **edge_data)
        
        return mol_graph

    def compute(self) -> SpacerAnalysisResult:
        """Compute all features and return results.

        Returns
        -------
        SpacerAnalysisResult
            Complete analysis with all metrics

        Raises
        ------
        ValueError
            If spacer type is not DJ or RP
            If required analyzer context is missing for penetration depth
        """
        if self._result_cache is not None:
            return self._result_cache

        # Initialize result
        result = SpacerAnalysisResult(
            molecule_indices=list(self.molecule_graph.original_indices),
            spacer_type=self.spacer_type,
            graph=self.graph
        )

        # Get terminal nitrogens
        result.terminal_nitrogens = self._get_terminal_nitrogens()

        # Feature 1b: Molecular Volume (for all spacer types)
        try:
            vdw_vol, hull_vol = self._compute_volume()
            result.vdw_volume = vdw_vol
            result.convex_hull_volume = hull_vol
            result.radius_gyration = self._compute_radius_gyration()
        except Exception:
            pass

        # For DJ spacers, compute N-N features
        if self.spacer_type == "DJ" and len(result.terminal_nitrogens) >= 2:
            # Feature 1: Compression
            euclidean, path_len, ideal_extended, compression = self._compute_compression()
            result.euclidean_distance = euclidean
            result.path_length = path_len
            result.ideal_extended_length = ideal_extended
            result.compression = compression

            # Feature 2: Backbone
            if self._backbone_cache is not None:
                result.backbone_path = self._backbone_cache
                result.backbone_length = len(self._backbone_cache)
                result.backbone_symbols = [
                    self.graph.nodes[idx].get('symbol', 'C')
                    for idx in self._backbone_cache
                ]

                # Feature 3: Side chains
                try:
                    side_chains = self._identify_side_chains(self._backbone_cache)
                    result.side_chains = side_chains
                    result.n_side_chains = len(side_chains)
                    result.branching_points = list(set(
                        sc['attachment_idx'] for sc in side_chains
                    ))
                except Exception:
                    pass

        self._result_cache = result
        return result

    def _get_terminal_nitrogens(self) -> List[int]:
        """Get terminal nitrogen atom indices (NH3+ groups).

        Returns
        -------
        List[int]
            Indices of N atoms in terminal groups (using graph node IDs, which are original_indices)
        """
        if self._terminal_N_cache is not None:
            return self._terminal_N_cache

        # Use cavity anchor nodes if in cavity mode
        if self._use_cavity_mode and self._cavity is not None:
            terminal_N = []
            
            # Find anchor nodes in cavity subgraph
            anchor_nodes = [
                node for node in self._cavity.subgraph.nodes()
                if self._cavity.subgraph.nodes[node].get('node_type') == 'anchor'
            ]
            
            for anchor_node in anchor_nodes:
                # Get all atoms connected to this anchor via 'contains' edge
                anchor_atoms = [
                    neighbor for neighbor in self._cavity.subgraph.neighbors(anchor_node)
                    if self._cavity.subgraph.get_edge_data(anchor_node, neighbor).get('edge_type') == 'contains'
                ]
                
                # Find N atom in anchor atoms
                for atom_node in anchor_atoms:
                    atom_data = self._cavity.subgraph.nodes[atom_node]
                    if atom_data.get('symbol') == 'N':
                        # Get original index
                        original_idx = atom_data.get('original_index')
                        if original_idx is None:
                            original_idx = atom_data.get('vasp_index')
                        
                        if original_idx is not None:
                            terminal_N.append(original_idx)
            
            if len(terminal_N) >= 1:  # At least one N found from anchors
                self._terminal_N_cache = terminal_N
                return terminal_N
        
        # Fall back to current method (structure graph approach)
        # The graph from build_molecular_graph uses original_indices as node IDs
        # So node IDs are integers corresponding to atom indices in the original structure
        
        # First, collect all N atoms in the graph
        all_N_atoms = []
        for node, data in self.graph.nodes(data=True):
            # Skip non-atom nodes (like 'molecule_0' if present)
            if isinstance(node, str):
                continue
                
            symbol = data.get('symbol')
            if symbol == 'N':
                neighbors = list(self.graph.neighbors(node))
                all_N_atoms.append((node, len(neighbors), neighbors))

        # If no N atoms found, try checking atoms_object directly as fallback
        if len(all_N_atoms) == 0 and hasattr(self.molecule_graph, 'atoms_object'):
            import warnings
            warnings.warn(
                f"No N atoms found in graph. Graph has {len(self.graph.nodes())} nodes. "
                f"Atoms object has {len(self.molecule_graph.atoms_object)} atoms. "
                f"Graph nodes: {list(self.graph.nodes())[:10]}..."
            )
            # Try to find N atoms in atoms_object and map back to graph
            for i, atom in enumerate(self.molecule_graph.atoms_object):
                if atom.symbol == 'N':
                    # Try to find corresponding node in graph
                    orig_idx = self.molecule_graph.original_indices[i] if i < len(self.molecule_graph.original_indices) else None
                    if orig_idx is not None and orig_idx in self.graph.nodes():
                        neighbors = list(self.graph.neighbors(orig_idx))
                        all_N_atoms.append((orig_idx, len(neighbors), neighbors))
        
        # Additional diagnostic: Check if we have fewer N atoms than expected
        if len(all_N_atoms) < 2 and hasattr(self.molecule_graph, 'atoms_object'):
            # Count N atoms in atoms_object
            n_in_atoms = sum(1 for atom in self.molecule_graph.atoms_object if atom.symbol == 'N')
            if n_in_atoms >= 2:
                import warnings
                warnings.warn(
                    f"Graph has {len(all_N_atoms)} N atoms but atoms_object has {n_in_atoms} N atoms. "
                    f"This suggests the graph is incomplete. "
                    f"Graph nodes: {sorted(list(self.graph.nodes()))[:20]}, "
                    f"Original indices: {self.molecule_graph.original_indices[:20]}"
                )

        # Filter for terminal N atoms (2-4 neighbors typical for NH3+)
        terminal_N = []
        for node, n_neighbors, neighbors in all_N_atoms:
            # Terminal N typically has 3-4 neighbors (3H + 1C or 4 including charge)
            # But we allow 2-4 to be more flexible
            if 2 <= n_neighbors <= 4:
                terminal_N.append(node)

        # If we found fewer than 2 terminal N atoms, use a more lenient approach
        if len(terminal_N) < 2 and len(all_N_atoms) >= 2:
            # Sort by neighbor count (terminal N should have fewer neighbors than backbone N)
            all_N_atoms.sort(key=lambda x: x[1])
            # Take the two with fewest neighbors (likely terminal)
            terminal_N = [node for node, _, _ in all_N_atoms[:2]]
        elif len(terminal_N) < 2 and len(all_N_atoms) == 1:
            # Only one N found - use it but this is unusual for DJ spacers
            terminal_N = [all_N_atoms[0][0]]

        self._terminal_N_cache = terminal_N
        return terminal_N

    def _compute_compression(self) -> Tuple[float, float, float, float]:
        """Compute Euclidean distance, path length, ideal extended length, and compression.

        For a fully extended sp3 carbon chain, the euclidean distance is approximately
        0.85 times the path_length due to tetrahedral bond angles (~109.5°).
        Compression is the difference between ideal and actual distance in Angstroms.

        For PBC structures, reconstructs the molecule starting from one NH3 group to get
        correct unwrapped positions. For standalone molecules (no cell), uses direct positions.

        Returns
        -------
        Tuple[float, float, float, float]
            (euclidean_distance, path_length, ideal_extended_length, compression)
            where:
            - euclidean_distance: straight-line N-N distance (Å)
            - path_length: sum of bond lengths along backbone (Å)
            - ideal_extended_length: path_length × 0.85 (expected distance for fully extended sp3 chain)
            - compression: ideal_extended_length - euclidean_distance (Å)
              - 0.0 Å: fully extended (euclidean matches ideal)
              - > 0 Å: compressed (molecule shorter than ideal by X Angstroms)
              - < 0 Å: overstretched (molecule longer than ideal by |X| Angstroms)
        """
        terminal_N = self._get_terminal_nitrogens()

        if len(terminal_N) < 2:
            raise ValueError(
                f"Compression calculation requires 2 terminal N atoms, found {len(terminal_N)}. "
                f"Terminal N indices: {terminal_N}"
            )

        N1_idx, N2_idx = terminal_N[0], terminal_N[1]
        
        # In cavity mode, positions are already PBC-unwrapped in graph
        if self._use_cavity_mode:
            # Get positions directly from graph (already PBC-unwrapped)
            N1_pos, N2_pos = self._get_n_positions(N1_idx, N2_idx)
            if N1_pos is None or N2_pos is None:
                raise ValueError(
                    f"Failed to get positions for terminal N atoms from cavity. "
                    f"N1_idx: {N1_idx}, N2_idx: {N2_idx}"
                )
            
            # Direct distance (positions already unwrapped)
            euclidean_dist = float(np.linalg.norm(N1_pos - N2_pos))
            reconstructed_positions = None  # Not needed in cavity mode
        else:
            # Check if we have cell/PBC - if so, reconstruct molecule from NH3
            cell, pbc, use_pbc = self._get_cell_info()
            reconstructed_positions = None
            
            if use_pbc and cell is not None:
                # Reconstruct molecule starting from first NH3 to get unwrapped positions
                reconstructed_positions = self._reconstruct_molecule_from_nh3(N1_idx, cell, pbc)
                
                if reconstructed_positions is None:
                    raise RuntimeError(
                        f"Failed to reconstruct molecule from NH3 group. "
                        f"N1_idx: {N1_idx}, N2_idx: {N2_idx}"
                    )
                
                # Get unwrapped positions from reconstruction
                if N1_idx not in reconstructed_positions or N2_idx not in reconstructed_positions:
                    raise ValueError(
                        f"Terminal N atoms not found in reconstructed positions. "
                        f"N1_idx: {N1_idx}, N2_idx: {N2_idx}, "
                        f"Available indices: {list(reconstructed_positions.keys())[:10]}..."
                    )
                
                N1_pos = reconstructed_positions[N1_idx][0]  # (position, image_label)
                N2_pos = reconstructed_positions[N2_idx][0]
                
                # Use direct distance on unwrapped positions (no PBC needed - already unwrapped)
                euclidean_dist = float(np.linalg.norm(N1_pos - N2_pos))
            else:
                # Standalone molecule - use direct positions
                N1_pos, N2_pos = self._get_n_positions(N1_idx, N2_idx)
                if N1_pos is None or N2_pos is None:
                    raise ValueError(
                        f"Failed to get positions for terminal N atoms. "
                        f"N1_idx: {N1_idx}, N2_idx: {N2_idx}"
                    )
                
                # Direct distance for standalone molecules
                euclidean_dist = float(np.linalg.norm(N1_pos - N2_pos))

        # Find longest backbone path
        backbone = self._find_longest_backbone(N1_idx, N2_idx)

        if backbone is None or len(backbone) < 2:
            raise ValueError(
                f"Failed to find backbone path between terminal N atoms. "
                f"N1_idx: {N1_idx}, N2_idx: {N2_idx}, "
                f"Backbone: {backbone}"
            )

        # Calculate path length (sum of bond lengths)
        # In cavity mode, positions are already PBC-unwrapped in graph
        # Otherwise, use reconstructed positions if available
        if self._use_cavity_mode:
            # Positions already unwrapped in graph
            path_length = self._calculate_path_length(backbone)
        elif reconstructed_positions is not None:
            path_length = self._calculate_path_length(backbone, reconstructed_positions)
        else:
            path_length = self._calculate_path_length(backbone)

        # For a fully extended sp3 chain, euclidean ≈ 0.85 × path_length
        # This accounts for tetrahedral bond angles (109.5°) in an all-trans conformation
        IDEAL_EXTENDED_RATIO = 0.85
        ideal_extended_length = path_length * IDEAL_EXTENDED_RATIO

        # Compression in Angstroms: ideal - actual
        # Positive value means compressed (actual < ideal)
        # Negative value means overstretched (actual > ideal)
        # Zero means perfectly extended
        compression = ideal_extended_length - euclidean_dist

        return euclidean_dist, path_length, ideal_extended_length, compression

    def _find_longest_backbone(self, N1_idx: int, N2_idx: int) -> Optional[List[int]]:
        """Find the longest path between terminal nitrogens.

        Parameters
        ----------
        N1_idx : int
            First nitrogen index
        N2_idx : int
            Second nitrogen index

        Returns
        -------
        Optional[List[int]]
            Longest path as list of atom indices, or None if no path
        """
        if self._backbone_cache is not None:
            return self._backbone_cache

        # Find ALL simple paths
        all_paths = find_all_paths(self.graph, N1_idx, N2_idx, max_length=None)

        if not all_paths:
            return None

        # Select LONGEST path (organic chemistry backbone definition)
        longest_path = max(all_paths, key=len)

        # Validate continuity
        if not validate_path_continuity(self.graph, longest_path):
            return None

        self._backbone_cache = longest_path
        return longest_path

    def _get_cell_info(self) -> Tuple[Optional[np.ndarray], Union[bool, List[bool]], bool]:
        """Get cell and PBC information.
        
        Returns
        -------
        Tuple[cell, pbc, use_pbc]
            cell: Unit cell matrix or None
            pbc: PBC flags (bool or list of bool)
            use_pbc: Whether to use PBC (True if cell exists and PBC is enabled)
        """
        cell = None
        pbc = True
        use_pbc = False
        
        if self.analyzer is not None and hasattr(self.analyzer, 'cell'):
            cell_array = self.analyzer.cell.get_cell()
            if cell_array is not None and np.any(cell_array):
                cell = np.array(cell_array)
                pbc_flags = self.analyzer.cell.get_pbc()
                if pbc_flags is not None:
                    pbc = pbc_flags
                    use_pbc = np.any(pbc) if isinstance(pbc, (list, np.ndarray)) else bool(pbc)
        elif hasattr(self.molecule_graph, 'parent_view') and self.molecule_graph.parent_view is not None:
            # Try to get cell from parent view
            if hasattr(self.molecule_graph.parent_view, 'full_structure'):
                cell_array = self.molecule_graph.parent_view.full_structure.get_cell()
                if cell_array is not None and np.any(cell_array):
                    cell = np.array(cell_array)
                    pbc_flags = self.molecule_graph.parent_view.full_structure.get_pbc()
                    if pbc_flags is not None:
                        pbc = pbc_flags
                        use_pbc = np.any(pbc) if isinstance(pbc, (list, np.ndarray)) else bool(pbc)
        
        return cell, pbc, use_pbc
    
    def _get_n_positions(self, N1_idx: int, N2_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get positions of two N atoms.
        
        Always uses graph positions directly to ensure consistency,
        especially in cavity mode where positions are PBC-unwrapped.
        
        Returns
        -------
        Tuple[N1_pos, N2_pos] or (None, None) if positions unavailable
        """
        # Always use graph node positions directly
        try:
            N1_pos = np.array(self.graph.nodes[N1_idx].get('position'))
            N2_pos = np.array(self.graph.nodes[N2_idx].get('position'))
            if N1_pos is not None and N2_pos is not None and not (np.any(np.isnan(N1_pos)) or np.any(np.isnan(N2_pos))):
                return N1_pos, N2_pos
        except (KeyError, AttributeError):
            pass
        
        return None, None
    
    def _reconstruct_molecule_from_nh3(self, nh3_n_idx: int, cell: np.ndarray, pbc: Union[bool, List[bool]]) -> Optional[Dict[int, Tuple[np.ndarray, Tuple[int, int, int]]]]:
        """Reconstruct molecule starting from NH3 group to get unwrapped PBC positions.
        
        Parameters
        ----------
        nh3_n_idx : int
            Index of N atom in first NH3 group (graph node ID, which is original_index)
        cell : np.ndarray
            Unit cell matrix
        pbc : bool or list of bool
            Periodic boundary conditions
            
        Returns
        -------
        dict or None
            Mapping from atom index to (position, image_label) or None if reconstruction fails
        """
        try:
            from ...utils.molecules.pbc_reconstruction import reconstruct_molecule_from_nh3
            
            # Get all atom positions and symbols from the structure
            # nh3_n_idx is already an original_index (graph node ID)
            if self.analyzer is not None:
                full_structure = self.analyzer.cell
                atom_positions = full_structure.get_positions()
                atom_symbols = [atom.symbol for atom in full_structure]
                
                # Get molecule indices in full structure
                molecule_indices = self.molecule_graph.original_indices
            elif hasattr(self.molecule_graph, 'parent_view') and self.molecule_graph.parent_view is not None:
                full_structure = self.molecule_graph.parent_view.full_structure
                atom_positions = full_structure.get_positions()
                atom_symbols = [atom.symbol for atom in full_structure]
                molecule_indices = self.molecule_graph.original_indices
            else:
                return None
            
            # Get NH3 center position directly from full structure (nh3_n_idx is original_index)
            if nh3_n_idx >= len(atom_positions):
                return None
            
            nh3_pos = atom_positions[nh3_n_idx]
            
            # Create a molecular-only graph for reconstruction (prevents leaking to framework atoms)
            # The molecular graph (self.graph) only contains atoms in this molecule
            # We need to convert it to the format expected by reconstruction (with atom_X node IDs)
            molecular_graph = self._create_molecular_graph_for_reconstruction(molecule_indices)
            
            # Verify molecular graph has connectivity
            if len(molecular_graph.nodes()) == 0:
                import warnings
                warnings.warn(
                    f"Molecular graph for reconstruction is empty. "
                    f"Molecule indices: {molecule_indices[:10]}..., "
                    f"Graph nodes: {list(self.graph.nodes())[:10]}...",
                    RuntimeWarning
                )
                return None
            
            # Verify graph has edges (connectivity)
            if len(molecular_graph.edges()) == 0:
                import warnings
                warnings.warn(
                    f"Molecular graph has no edges (no connectivity). "
                    f"Molecule indices: {molecule_indices[:10]}..., "
                    f"Graph edges: {list(self.graph.edges())[:10]}...",
                    RuntimeWarning
                )
                return None
            
            # Reconstruct molecule starting from NH3 using molecular graph only
            reconstructed = reconstruct_molecule_from_nh3(
                molecule_indices=molecule_indices,
                nh3_center=nh3_pos,
                graph=molecular_graph,
                atom_positions=atom_positions,
                atom_symbols=atom_symbols,
                cell=cell,
                pbc=pbc
            )
            
            # Verify reconstruction succeeded and has all atoms
            if reconstructed is None:
                import warnings
                warnings.warn(
                    f"Reconstruction returned None. "
                    f"NH3 N idx: {nh3_n_idx}, molecule indices: {len(molecule_indices)}, "
                    f"Graph nodes: {len(molecular_graph.nodes())}, edges: {len(molecular_graph.edges())}",
                    RuntimeWarning
                )
                return None
            
            # Check if all molecule atoms were reconstructed
            missing_atoms = set(molecule_indices) - set(reconstructed.keys())
            if missing_atoms:
                import warnings
                warnings.warn(
                    f"Reconstruction incomplete: {len(missing_atoms)} atoms missing. "
                    f"Expected {len(molecule_indices)} atoms, got {len(reconstructed)}. "
                    f"Missing: {list(missing_atoms)[:5]}...",
                    RuntimeWarning
                )
            
            return reconstructed
        except Exception as e:
            import warnings
            import traceback
            warnings.warn(
                f"Molecule reconstruction failed: {e}\n{traceback.format_exc()}",
                RuntimeWarning
            )
            return None
    
    def _create_molecular_graph_for_reconstruction(self, molecule_indices: List[int]) -> nx.Graph:
        """Create a graph with only molecular atoms for reconstruction.
        
        Converts the molecular graph (self.graph) to the format expected by
        reconstruction functions, with node IDs like 'atom_{idx}' and BONDED_TO edges.
        This ensures reconstruction only uses connectivity within the molecule,
        not framework atoms.
        
        Parameters
        ----------
        molecule_indices : List[int]
            Original indices of atoms in the molecule
            
        Returns
        -------
        nx.Graph
            Graph with nodes 'atom_{idx}' and BONDED_TO edges, only for molecule atoms
        """
        # Create new graph with structural graph format
        mol_graph = nx.Graph()
        molecule_set = set(molecule_indices)
        
        # Add nodes with atom_X format
        for atom_idx in molecule_indices:
            node_id = f'atom_{atom_idx}'
            if atom_idx in self.graph.nodes():
                node_data = self.graph.nodes[atom_idx].copy()
                # Add vasp_index for compatibility with reconstruction
                node_data['vasp_index'] = atom_idx
                mol_graph.add_node(node_id, **node_data)
        
        # Add edges (bonds) only between atoms in the molecule
        for atom_idx in molecule_indices:
            node_id = f'atom_{atom_idx}'
            if atom_idx in self.graph.nodes():
                # Get neighbors from molecular graph
                for neighbor_idx in self.graph.neighbors(atom_idx):
                    # Only add edge if neighbor is also in the molecule
                    if neighbor_idx in molecule_set:
                        neighbor_node_id = f'atom_{neighbor_idx}'
                        if neighbor_node_id in mol_graph.nodes():
                            # Get edge data from molecular graph
                            edge_data = self.graph.get_edge_data(atom_idx, neighbor_idx, {})
                            edge_data['edge_type'] = 'bonded_to'
                            mol_graph.add_edge(node_id, neighbor_node_id, **edge_data)
        
        return mol_graph
    
    def _calculate_pbc_distance(self, pos1: np.ndarray, pos2: np.ndarray, cell: np.ndarray, pbc: Union[bool, List[bool]]) -> float:
        """Calculate PBC-aware distance between two positions.
        
        Uses minimum image convention to find the shortest distance
        considering periodic boundary conditions (27-image supercell).
        
        Parameters
        ----------
        pos1 : np.ndarray
            First position (3D)
        pos2 : np.ndarray
            Second position (3D)
            
        Returns
        -------
        float
            Minimum image distance in Angstroms
            
        Raises
        ------
        ValueError
            If cell information is not available
        ImportError
            If PBC distance calculation module is not available
        """
        # Get cell from analyzer or molecule_graph
        cell = None
        pbc = True
        
        if self.analyzer is not None and hasattr(self.analyzer, 'cell'):
            cell_array = self.analyzer.cell.get_cell()
            if cell_array is not None and np.any(cell_array):
                cell = np.array(cell_array)
                pbc_flags = self.analyzer.cell.get_pbc()
                if pbc_flags is not None:
                    pbc = pbc_flags
        elif hasattr(self.molecule_graph, 'parent_view') and self.molecule_graph.parent_view is not None:
            # Try to get cell from parent view
            if hasattr(self.molecule_graph.parent_view, 'full_structure'):
                cell_array = self.molecule_graph.parent_view.full_structure.get_cell()
                if cell_array is not None and np.any(cell_array):
                    cell = np.array(cell_array)
                    pbc_flags = self.molecule_graph.parent_view.full_structure.get_pbc()
                    if pbc_flags is not None:
                        pbc = pbc_flags
        
        # Require cell information
        if cell is None or not np.any(cell):
            raise ValueError(
                "Cell information required for PBC-aware distance calculation. "
                "Ensure analyzer or parent_view provides cell data."
            )
        
        # Use PBC-aware distance calculation (minimum image convention)
        from ...utils.geometry.pbc_distances import calculate_pbc_distances
        
        # Calculate PBC distance using minimum image convention
        pos1_array = np.array([pos1])
        pos2_array = np.array([pos2])
        
        distances = calculate_pbc_distances(
            reference_positions=pos1_array,
            target_positions=pos2_array,
            cell=cell,
            pbc=pbc
        )
        
        return float(distances[0])
    
    def _calculate_path_length(self, path: List[int], reconstructed_positions: Optional[Dict[int, Tuple[np.ndarray, Tuple[int, int, int]]]] = None) -> float:
        """Calculate total path length as sum of bond lengths.

        Parameters
        ----------
        path : List[int]
            List of atom indices forming path

        Returns
        -------
        float
            Total path length in Angstroms
        """
        if len(path) < 2:
            return 0.0

        total_length = 0.0

        # Use reconstructed positions if available (PBC-unwrapped)
        if reconstructed_positions is not None:
            for i in range(len(path) - 1):
                atom1_idx = path[i]
                atom2_idx = path[i + 1]
                
                if atom1_idx in reconstructed_positions and atom2_idx in reconstructed_positions:
                    pos1 = reconstructed_positions[atom1_idx][0]  # (position, image_label)
                    pos2 = reconstructed_positions[atom2_idx][0]
                    
                    # Validate positions
                    if pos1 is None or pos2 is None or np.any(np.isnan(pos1)) or np.any(np.isnan(pos2)):
                        continue  # Skip invalid bond
                    
                    # Calculate bond length
                    bond_length = np.linalg.norm(pos1 - pos2)
                    total_length += bond_length
        else:
            # Always use graph node positions directly for consistency
            for i in range(len(path) - 1):
                atom1_idx = path[i]
                atom2_idx = path[i + 1]

                # Get positions from graph
                pos1 = np.array(self.graph.nodes[atom1_idx].get('position'))
                pos2 = np.array(self.graph.nodes[atom2_idx].get('position'))

                # Validate positions
                if pos1 is None or pos2 is None or np.any(np.isnan(pos1)) or np.any(np.isnan(pos2)):
                    continue  # Skip invalid bond

                # Calculate bond length
                bond_length = np.linalg.norm(pos1 - pos2)
                total_length += bond_length

        return float(total_length)

    def _compute_volume(self) -> Tuple[float, float]:
        """Compute molecular volume using two methods.
        
        Returns
        -------
        Tuple[float, float]
            (vdw_volume, convex_hull_volume) in Å³
            - vdw_volume: sum of atomic volumes using van der Waals radii (0.0 if ASE unavailable)
            - convex_hull_volume: volume of convex hull (0.0 if scipy unavailable or <4 atoms)
        """
        # Get all atom positions and symbols
        positions = []
        symbols = []
        
        for node_idx in self.graph.nodes():
            node_data = self.graph.nodes[node_idx]
            pos = node_data.get('position')
            symbol = node_data.get('symbol', 'C')
            
            if pos is not None:
                positions.append(pos)
                symbols.append(symbol)
        
        if not positions:
            return 0.0, 0.0
        
        positions = np.array(positions)
        
        # Method 1: Van der Waals volume (sum of atomic volumes)
        vdw_vol = 0.0
        if not ASE_DATA_AVAILABLE:
            return 0.0, 0.0
        
        for symbol in symbols:
            atomic_num = atomic_numbers.get(symbol)
            if atomic_num is None:
                continue
            
            vdw_radius = vdw_radii[atomic_num]
            
            # Skip if vdw_radius is NaN or invalid
            if np.isnan(vdw_radius) or vdw_radius <= 0:
                continue
            
            # Volume of sphere: V = (4/3) × π × r³
            atom_volume = (4.0 / 3.0) * np.pi * (vdw_radius ** 3)
            vdw_vol += atom_volume
        
        # Method 2: Convex hull volume
        hull_vol = 0.0
        if SCIPY_AVAILABLE and len(positions) >= 4:
            try:
                # Need at least 4 points for 3D convex hull
                hull = ConvexHull(positions)
                hull_vol = float(hull.volume)
            except Exception:
                # Convex hull calculation failed (e.g., coplanar points)
                hull_vol = 0.0
        
        return float(vdw_vol), float(hull_vol)

    def _compute_radius_gyration(self) -> float:
        """Compute radius of gyration (R_g).

        The radius of gyration is the mass-weighted root-mean-square distance
        of all atoms from the molecule's center of mass:

        R_g = sqrt(Σ(m_i * |r_i - r_com|²) / Σ(m_i))

        where r_com is the center of mass: Σ(m_i * r_i) / Σ(m_i)

        This metric characterizes molecular size and compactness.

        Returns
        -------
        float
            Radius of gyration in Angstroms (0.0 if ASE unavailable or <1 atom)
        """
        if not ASE_DATA_AVAILABLE:
            return 0.0

        # Collect atomic positions and masses
        positions = []
        masses = []

        for node_idx in self.graph.nodes():
            node_data = self.graph.nodes[node_idx]
            pos = node_data.get('position')
            symbol = node_data.get('symbol', 'C')

            if pos is not None:
                # Get atomic mass
                atomic_num = atomic_numbers.get(symbol)
                if atomic_num is None:
                    continue

                mass = atomic_masses[atomic_num]

                # Skip if mass is invalid
                if np.isnan(mass) or mass <= 0:
                    continue

                positions.append(pos)
                masses.append(mass)

        if len(positions) < 1:
            return 0.0

        positions = np.array(positions)  # Shape: (n_atoms, 3)
        masses = np.array(masses)         # Shape: (n_atoms,)

        # Compute center of mass
        total_mass = np.sum(masses)
        if total_mass <= 0:
            return 0.0

        center_of_mass = np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass

        # Compute squared distances from center of mass
        displacements = positions - center_of_mass  # Shape: (n_atoms, 3)
        squared_distances = np.sum(displacements ** 2, axis=1)  # Shape: (n_atoms,)

        # Compute mass-weighted RMS distance
        rg_squared = np.sum(masses * squared_distances) / total_mass
        rg = np.sqrt(rg_squared)

        return float(rg)

    def _identify_side_chains(self, backbone_path: List[int]) -> List[Dict]:
        """Identify side chains by branching detection.

        Parameters
        ----------
        backbone_path : List[int]
            Atom indices in backbone

        Returns
        -------
        List[Dict]
            Side chain data: [{'attachment_idx', 'chain_atoms', 'length'}]
        """
        side_chains = []
        backbone_set = set(backbone_path)
        terminal_N = set(self._get_terminal_nitrogens())

        # Find branching points (degree >= 3, not terminal)
        for node in backbone_path:
            neighbors = list(self.graph.neighbors(node))

            # Skip if not branching point or is terminal N
            if len(neighbors) < 3 or node in terminal_N:
                continue

            # Trace each branch
            for neighbor in neighbors:
                if neighbor not in backbone_set:
                    # BFS to find all atoms in this side chain
                    side_chain_atoms = self._trace_side_chain(
                        start=neighbor,
                        backbone=backbone_set
                    )

                    if side_chain_atoms:
                        side_chains.append({
                            'attachment_idx': node,
                            'chain_atoms': side_chain_atoms,
                            'length': len(side_chain_atoms)
                        })

        return side_chains

    def _trace_side_chain(self, start: int, backbone: Set[int]) -> List[int]:
        """BFS to trace side chain atoms.

        Parameters
        ----------
        start : int
            Starting atom index
        backbone : Set[int]
            Set of backbone atom indices to exclude

        Returns
        -------
        List[int]
            Atom indices in side chain
        """
        visited = set()
        queue = deque([start])
        side_chain = []

        while queue:
            node = queue.popleft()

            if node in visited or node in backbone:
                continue

            visited.add(node)
            side_chain.append(node)

            # Add neighbors to queue
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited and neighbor not in backbone:
                    queue.append(neighbor)

        return side_chain

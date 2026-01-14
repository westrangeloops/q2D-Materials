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

from ..utils.graph_utils import (
    find_all_paths,
    validate_path_continuity,
)
from ...utils.properties.atomic_properties import get_covalent_radius

if TYPE_CHECKING:
    from ...modifier.molecule_graph import MoleculeGraph
    from ..core.analyzer_class import q2D_analyzer


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
    penetration_depths : Dict[int, float]
        Z-penetration for each N atom {N_index: z_penetration (Å)}
        Negative values indicate penetration into framework
    avg_penetration : float
        Average penetration across all N atoms (Å)
    min_penetration : float
        Most penetrating N (most negative value) (Å)
    max_penetration : float
        Least penetrating N (most positive value) (Å)
    euclidean_distance : float
        Through-space N-N distance (Å)
    path_length : float
        Fully extended path length (Å)
    compression_factor : float
        path_length / euclidean_distance (>1 = compressed)
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
    # Feature 0: Penetration Depth
    penetration_depths: Dict[int, float] = field(default_factory=dict)
    avg_penetration: float = 0.0
    min_penetration: float = 0.0
    max_penetration: float = 0.0

    # Feature 1: Linker Distance & Compression
    euclidean_distance: float = 0.0
    path_length: float = 0.0
    compression_factor: float = 0.0

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
    def is_penetrating(self) -> bool:
        """Whether spacer penetrates framework (avg penetration < 0)."""
        return self.avg_penetration < 0

    @property
    def is_compressed(self) -> bool:
        """Whether spacer is compressed (compression > 1.1)."""
        return self.compression_factor > 1.1

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
            f"  Penetration: avg={self.avg_penetration:.2f} Å, "
            f"min={self.min_penetration:.2f} Å, max={self.max_penetration:.2f} Å\n"
            f"  Compression: euclidean={self.euclidean_distance:.2f} Å, "
            f"path={self.path_length:.2f} Å, factor={self.compression_factor:.2f}\n"
            f"  Backbone: {self.backbone_length} atoms, "
            f"{''.join(self.backbone_symbols[:10])}{'...' if len(self.backbone_symbols) > 10 else ''}\n"
            f"  Side chains: {self.n_side_chains} detected\n"
            f")"
        )


class SpacerAnalysis:
    """Analyzer for DJ spacer molecular properties.

    This class computes structural and geometric properties of DJ spacer molecules
    including penetration depth, compression, backbone, and side chains.

    Parameters
    ----------
    molecule_graph : MoleculeGraph
        Molecule to analyze (from GraphView or MoleculeGraph)
    analyzer : q2D_analyzer, optional
        Full analyzer instance for accessing halogen positions
        If not provided, penetration depth cannot be computed
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
        spacer_type: str = "DJ"
    ):
        """Initialize spacer analyzer."""
        self.molecule_graph = molecule_graph
        self.graph = molecule_graph.graph
        self.analyzer = analyzer
        self.spacer_type = spacer_type.upper()

        # Caches
        self._result_cache: Optional[SpacerAnalysisResult] = None
        self._terminal_N_cache: Optional[List[int]] = None
        self._terminal_halogens_cache: Optional[List[Dict]] = None
        self._backbone_cache: Optional[List[int]] = None

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

        # Feature 0: Penetration depth (requires analyzer)
        if self.analyzer is not None:
            try:
                penetration_depths = self._compute_penetration_depths()
                result.penetration_depths = penetration_depths

                if penetration_depths:
                    depths = list(penetration_depths.values())
                    result.avg_penetration = float(np.mean(depths))
                    result.min_penetration = float(np.min(depths))
                    result.max_penetration = float(np.max(depths))
            except Exception as e:
                # Penetration calculation failed, continue with other features
                pass

        # For DJ spacers, compute N-N features
        if self.spacer_type == "DJ" and len(result.terminal_nitrogens) >= 2:
            # Feature 1: Compression
            try:
                euclidean, path_len, compression = self._compute_compression()
                result.euclidean_distance = euclidean
                result.path_length = path_len
                result.compression_factor = compression
            except Exception:
                pass

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
            Indices of N atoms in terminal groups
        """
        if self._terminal_N_cache is not None:
            return self._terminal_N_cache

        # Find N atoms with 3-4 neighbors (NH3+ or NH2)
        terminal_N = []
        for node, data in self.graph.nodes(data=True):
            if data.get('symbol') == 'N':
                neighbors = list(self.graph.neighbors(node))
                # Terminal N typically has 3-4 neighbors (3H + 1C or 4 including charge)
                if 2 <= len(neighbors) <= 4:
                    terminal_N.append(node)

        self._terminal_N_cache = terminal_N
        return terminal_N

    def _get_terminal_halogens(self) -> List[Dict]:
        """Get terminal halogen positions from octahedra.

        Returns
        -------
        List[Dict]
            List of terminal halogens: [{'index', 'position', 'symbol', 'z'}]

        Raises
        ------
        ValueError
            If analyzer is not provided
        """
        if self._terminal_halogens_cache is not None:
            return self._terminal_halogens_cache

        if self.analyzer is None:
            raise ValueError("Analyzer required to get terminal halogens")

        terminal_halogens = []
        octahedra = self.analyzer.get_octahedra()
        positions = self.analyzer.cell.get_positions()
        symbols = self.analyzer.cell.get_chemical_symbols()

        for octahedron in octahedra:
            for halogen_idx in octahedron.get('terminal_atoms', []):
                position = positions[halogen_idx]
                terminal_halogens.append({
                    'index': halogen_idx,
                    'position': position,
                    'symbol': symbols[halogen_idx],
                    'z': position[2]
                })

        self._terminal_halogens_cache = terminal_halogens
        return terminal_halogens

    def _compute_penetration_depths(self) -> Dict[int, float]:
        """Compute Z-penetration for each N atom.

        Returns
        -------
        Dict[int, float]
            {N_index: z_penetration} where negative = penetration
        """
        terminal_halogens = self._get_terminal_halogens()
        terminal_N = self._get_terminal_nitrogens()

        if not terminal_halogens:
            return {}

        penetration_depths = {}

        for n_idx in terminal_N:
            n_z = self.graph.nodes[n_idx]['position'][2]

            # Find closest halogen by Z-distance only
            closest_halogen_z = min(h['z'] for h in terminal_halogens)

            # Calculate penetration (negative = penetration into framework)
            penetration = n_z - closest_halogen_z
            penetration_depths[n_idx] = float(penetration)

            # Store as node property
            self.graph.nodes[n_idx]['penetration_depth'] = float(penetration)

        return penetration_depths

    def _compute_compression(self) -> Tuple[float, float, float]:
        """Compute Euclidean distance, path length, compression factor.

        Returns
        -------
        Tuple[float, float, float]
            (euclidean_distance, path_length, compression_factor)
        """
        terminal_N = self._get_terminal_nitrogens()

        if len(terminal_N) < 2:
            return 0.0, 0.0, 0.0

        # Get positions
        N1_idx, N2_idx = terminal_N[0], terminal_N[1]
        N1_pos = self.graph.nodes[N1_idx]['position']
        N2_pos = self.graph.nodes[N2_idx]['position']

        # Euclidean distance (through-space)
        euclidean_dist = float(np.linalg.norm(N1_pos - N2_pos))

        # Find longest backbone path
        backbone = self._find_longest_backbone(N1_idx, N2_idx)

        if backbone is None or len(backbone) < 2:
            return euclidean_dist, 0.0, 0.0

        # Calculate path length (sum of bond lengths)
        path_length = self._calculate_path_length(backbone)

        # Compression factor
        if euclidean_dist > 0:
            compression = path_length / euclidean_dist
        else:
            compression = 0.0

        return euclidean_dist, path_length, compression

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

    def _calculate_path_length(self, path: List[int]) -> float:
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

        for i in range(len(path) - 1):
            atom1_idx = path[i]
            atom2_idx = path[i + 1]

            # Get positions
            pos1 = self.graph.nodes[atom1_idx]['position']
            pos2 = self.graph.nodes[atom2_idx]['position']

            # Calculate bond length
            bond_length = np.linalg.norm(pos1 - pos2)
            total_length += bond_length

        return float(total_length)

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

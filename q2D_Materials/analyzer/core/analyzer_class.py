"""
q2D Analyzer - Graph-based structure analysis for perovskite materials.

This module provides comprehensive analysis of perovskite structures using a
graph-based approach to identify:
- Octahedra (BX6 units)
- Layers/slabs
- Spacer molecules
- A-site cations
- Structure type (bulk, DJ, RP, monolayer)
"""

from typing import Optional, List, Dict, Any, Union, Tuple, Set
import numpy as np
import networkx as nx
from ase.io import read
from ase import Atoms

from .graph_construction import _graph_inorganic_ontology
from ..detection.octahedral_detection import _count_octahedra, find_shared_atoms
from ..detection.layer_identification import _identify_layers
from ..characterization.characterization import (
    _detect_glazer_pattern, 
    _calculate_bxb_angles, 
    _calculate_partial_rdf,
    DEFAULT_TOLERANCE,
    DEFAULT_BOND_SELECTION_THRESHOLD,
    CharacterizationQuery,
)
from ...core.structure import q2DStructure


class q2D_analyzer:
    """
    Graph-based analyzer for 2D quantum materials.
    
    This analyzer decomposes structural files (VASP, CIF, Atoms) back into
    their constituent components using graph-based connectivity analysis.
    
    Attributes
    ----------
    cell : ase.Atoms
        The loaded structure
    graph : networkx.Graph
        The structural connectivity graph after analysis
    octahedra : list
        List of octahedra information
    layers : dict
        Layer/slab information
    spacers : list
        List of identified spacer molecules (as Atoms objects)
    a_sites : list
        List of identified A-site cations
    structure_type : str
        Inferred structure type ('bulk', 'dj', 'rp', 'monolayer')
    """

    def __init__(self, source: Optional[Union[str, Atoms]] = None):
        """
        Initialize the q2D_analyzer.

        Parameters
        ----------
        source : str or ase.Atoms, optional
            Path to structural file (VASP, CIF, etc.) or an ASE Atoms object.
            If provided, the structure is loaded but not analyzed automatically.
        """
        self.file_path = None
        self.experiment_name = None
        self.cell = None

        self._graph: Optional[nx.Graph] = None
        self._structure_type: Optional[str] = None
        self._analyzed: bool = False

        if source is not None:
            self.load(source)
    
    def load(self, source: Union[str, Atoms]) -> "q2D_analyzer":
        """
        Load a structure from file or Atoms object.
        
        Parameters
        ----------
        source : str or ase.Atoms
            Path to structural file or an ASE Atoms object
            
        Returns
        -------
        q2D_analyzer
            Self, for method chaining
        """
        if isinstance(source, str):
            self.file_path = source
            self.experiment_name = source.split('/')[-1].split('.')[0]
            self.cell = read(source)
        elif isinstance(source, Atoms):
            self.cell = source.copy()
            self.experiment_name = "atoms_input"
        else:
            raise TypeError(f"source must be str or Atoms, got {type(source)}")

        self.cell.pbc = True

        self._analyzed = False
        self._graph = None
        self._structure_type = None

        return self
    
    def analyze(
        self,
        cutoff_distance: float = None,
        min_tolerance: float = 0.2,
        step: float = 0.1,
        max_steps: int = 20,
        non_metal_symbols: Optional[List[str]] = None,
        # NEW: Configurable detection parameters
        octahedra_edges: List[int] = None,  # Valid edge counts (default: [6], can add 4)
        octahedra_centers: Optional[List[str]] = None,  # Valid B-site atoms (default: auto-detect)
        valid_halogen: Optional[List[str]] = None,  # Valid halogens (default: ['F','Cl','Br','I','O'])
        valid_molecule: Optional[List[str]] = None,  # Valid molecule atoms (default: ['C','H','O','N','S'])
    ) -> "q2D_analyzer":
        """
        Run full structure analysis using graph-based approach.

        This method builds a connectivity graph with all structural information
        stored as node and edge attributes:
        - Octahedra (BX6 units) as 'octahedron' nodes
        - Layers/slabs as 'layer' nodes
        - Atoms as 'atom' nodes with spacer/A-site classification
        - Structure type inferred from graph patterns

        Parameters
        ----------
        cutoff_distance : float, optional
            Maximum distance for octahedral neighbors. If None (default), uses
            element-specific cutoffs from PEROVSKITE_BOND_RADII for accurate detection.
            Set to explicit value (e.g., 4.0) to override automatic behavior.
        min_tolerance : float, optional
            Starting bond length tolerance (default: 0.2 Å)
        step : float, optional
            Tolerance increment step (default: 0.1 Å)
        max_steps : int, optional
            Maximum tolerance optimization steps (default: 20)
        non_metal_symbols : list of str, optional
            Symbols to exclude from octahedral centers
            (default: ['C', 'O', 'N', 'H', 'F', 'Cl', 'Br', 'I'])
        octahedra_edges : list of int, optional
            Valid edge counts for octahedra (default: [6], can include 4 for 2D structures)
        octahedra_centers : list of str, optional
            Valid B-site atoms (default: ['Pb', 'Sn'])
        valid_halogen : list of str, optional
            Valid halogen/X-site atoms (default: ['Cl', 'Br', 'I'])
        valid_molecule : list of str, optional
            Valid molecule atoms (default: ['C', 'H', 'O', 'N', 'S'])

        Returns
        -------
        q2D_analyzer
            Self, for method chaining
        """
        if self.cell is None:
            raise ValueError("No structure loaded. Call load() first or provide source in constructor.")

        if non_metal_symbols is None:
            non_metal_symbols = ['C', 'O', 'N', 'H', 'F', 'Cl', 'Br', 'I']
        
        # Set defaults for configurable parameters
        if octahedra_edges is None:
            octahedra_edges = [6]  # Default: 6 edges, can add 4 for 2D structures
        if octahedra_centers is None:
            octahedra_centers = ['Pb', 'Sn']  # Default B-site atoms
        if valid_halogen is None:
            valid_halogen = ['Cl', 'Br', 'I']  # Default valid halogens
        if valid_molecule is None:
            valid_molecule = ['C', 'H', 'O', 'N', 'S']  # Default molecule atoms

        atom_positions = self.cell.get_positions()
        atom_symbols = self.cell.get_chemical_symbols()
        cell_matrix = np.array(self.cell.get_cell())

        self._graph = _graph_inorganic_ontology(
            atom_positions,
            atom_symbols,
            cell_matrix,
            cutoff_distance=cutoff_distance,
            min_tolerance=min_tolerance,
            step=step,
            max_steps=max_steps,
            non_metal_symbols=non_metal_symbols,
            octahedra_edges=octahedra_edges,
            octahedra_centers=octahedra_centers,
            valid_halogen=valid_halogen,
            valid_molecule=valid_molecule,
        )
        
        # CRITICAL: Verify graph integrity - ensure all atoms are present
        # and that structural nodes (octahedra, layers) exist
        n_atoms = len(atom_positions)
        atom_nodes = [n for n, d in self._graph.nodes(data=True) if d.get('node_type') == 'atom']
        octahedra_nodes = [n for n, d in self._graph.nodes(data=True) if d.get('node_type') == 'octahedron']
        layer_nodes = [n for n in self._graph.nodes() if 'layer' in str(n).lower()]
        total_nodes = len(self._graph.nodes())
        is_critical = (total_nodes == n_atoms and n_atoms > 0)
        
        # Print to stderr for critical issues
        if is_critical:
            import sys
            print(f"CRITICAL: Graph has {total_nodes} nodes for {n_atoms} atoms - no structural nodes!", file=sys.stderr)
        
        # Ensure all atoms are in the graph
        if len(atom_nodes) < n_atoms:
            # Add missing atoms
            for i in range(n_atoms):
                if f'atom_{i}' not in self._graph:
                    node_data = {
                        'node_type': 'atom',
                        'vasp_index': i,
                        'symbol': atom_symbols[i] if atom_symbols else 'Unknown',
                        'direct_coordinates': atom_positions[i].tolist(),
                    }
                    self._graph.add_node(f'atom_{i}', **node_data)
        
        # CRITICAL: If graph has exactly N nodes for N atoms, structural analysis failed
        # This is a fundamental issue - the graph should have MORE nodes (atoms + octahedra + layers)
        final_total = len(self._graph.nodes())
        final_atom_nodes = len([n for n, d in self._graph.nodes(data=True) if d.get('node_type') == 'atom'])
        final_octahedra = len([n for n, d in self._graph.nodes(data=True) if d.get('node_type') == 'octahedron'])
        final_layers = len([n for n in self._graph.nodes() if 'layer' in str(n).lower()])
        
        # Print to stderr for critical issues
        if final_total == n_atoms and n_atoms > 0:
            import sys
            print(f"ERROR: Graph integrity failure - {n_atoms} atoms but only {final_total} nodes (no structural nodes!)", file=sys.stderr)

        self._classify_molecules_in_graph(cell_matrix)
        self._structure_type = self._infer_structure_type_from_graph()

        self._analyzed = True
        return self
    
    def _classify_molecules_in_graph(self, cell: np.ndarray) -> None:
        """
        Classify molecules as spacers or A-sites and store in graph nodes.

        This method identifies molecules using pymatgen, classifies them based on
        slab continuity, and stores classification as atom node attributes.
        """
        from ..detection.molecule_classification import _find_molecular_components, _classify_molecules_by_continuity
        from ..characterization.network_analysis import _build_bx_network
        from ..detection.layer_identification import _identify_slabs_by_continuity
        from ..detection.octahedral_detection import find_shared_atoms

        atom_positions = self.cell.get_positions()
        atom_symbols = self.cell.get_chemical_symbols()

        atoms_in_octahedra = set()
        octahedra_info = []
        neighbor_indices = []

        for node, data in self._graph.nodes(data=True):
            if data.get('node_type') == 'octahedron':
                central_idx = data.get('central_atom')
                terminal = data.get('terminal_atoms', [])
                interlayer = data.get('interlayer_atoms', [])
                intralayer = data.get('intralayer_atoms', [])

                if central_idx is not None:
                    atoms_in_octahedra.add(central_idx)
                atoms_in_octahedra.update(terminal)
                atoms_in_octahedra.update(interlayer)
                atoms_in_octahedra.update(intralayer)

                octahedra_info.append({
                    'id': node,
                    'central_atom_index': central_idx,
                    'terminal_atoms': terminal,
                    'interlayer_atoms': interlayer,
                    'intralayer_atoms': intralayer,
                })
                neighbor_indices.append(terminal + interlayer + intralayer)

        molecules = _find_molecular_components(
            atom_positions,
            atom_symbols,
            cell,
            self._graph,
            atoms_in_octahedra,
            full_atoms=self.cell,
        )

        shared_atoms = find_shared_atoms(neighbor_indices)
        bx_graph = _build_bx_network(octahedra_info, atom_positions, shared_atoms)
        slab_info = _identify_slabs_by_continuity(bx_graph, octahedra_info, atom_positions)

        spacers, a_sites = _classify_molecules_by_continuity(
            molecules,
            slab_info,
            atom_positions,
            cell,
        )

        for spacer in spacers:
            indices = spacer.info.get('original_indices', [])
            for idx in indices:
                node_id = f'atom_{idx}'
                if self._graph.has_node(node_id):
                    self._graph.nodes[node_id]['is_spacer'] = True
                    self._graph.nodes[node_id]['spacer_formula'] = spacer.get_chemical_formula(mode='hill')
                    self._graph.nodes[node_id]['spacer_indices'] = indices
                    self._graph.nodes[node_id]['spacer_type'] = spacer.info.get('spacer_type')

        for a_site in a_sites:
            indices = a_site.info.get('original_indices', [])
            for idx in indices:
                node_id = f'atom_{idx}'
                if self._graph.has_node(node_id):
                    self._graph.nodes[node_id]['is_a_site'] = True
                    self._graph.nodes[node_id]['a_site_formula'] = a_site.get_chemical_formula(mode='hill')
                    self._graph.nodes[node_id]['a_site_indices'] = indices

    def _infer_structure_type_from_graph(self) -> str:
        """Infer structure type from graph patterns."""
        from .structure_classification import _infer_structure_type_from_graph
        from ..characterization.network_analysis import _build_bx_network
        from ..detection.layer_identification import _identify_slabs_by_continuity
        from ..detection.octahedral_detection import find_shared_atoms

        atom_positions = self.cell.get_positions()

        octahedra_info = []
        neighbor_indices = []
        for node, data in self._graph.nodes(data=True):
            if data.get('node_type') == 'octahedron':
                terminal = data.get('terminal_atoms', [])
                interlayer = data.get('interlayer_atoms', [])
                intralayer = data.get('intralayer_atoms', [])
                octahedra_info.append({'id': node})
                neighbor_indices.append(terminal + interlayer + intralayer)

        shared_atoms = find_shared_atoms(neighbor_indices)
        bx_graph = _build_bx_network(octahedra_info, atom_positions, shared_atoms)
        slab_info = _identify_slabs_by_continuity(bx_graph, octahedra_info, atom_positions)

        spacers = []
        processed_indices = set()
        for node, data in self._graph.nodes(data=True):
            if data.get('node_type') == 'atom' and data.get('is_spacer'):
                atom_idx = data.get('vasp_index')
                if atom_idx in processed_indices:
                    continue
                spacer_indices = data.get('spacer_indices', [atom_idx])
                processed_indices.update(spacer_indices)

                spacer_atoms = Atoms(
                    symbols=[self.cell[i].symbol for i in spacer_indices],
                    positions=[self.cell[i].position for i in spacer_indices],
                )
                spacer_atoms.info['original_indices'] = spacer_indices
                spacer_atoms.info['spacer_type'] = data.get('spacer_type')
                spacers.append(spacer_atoms)

        return _infer_structure_type_from_graph(
            slab_info,
            spacers,
            bx_graph,
            np.array(self.cell.get_cell()),
        )



    def get_graph(self) -> nx.Graph:
        """
        Get the structural connectivity graph.

        Returns
        -------
        networkx.Graph
            The complete structural graph with all information

        Raises
        ------
        RuntimeError
            If analyze() has not been called
        """
        self._ensure_analyzed()
        
        # CRITICAL: Verify graph integrity when accessed
        # If graph has same nodes as atoms, structural analysis failed
        n_atoms = len(self.cell)
        total_nodes = len(self._graph.nodes())
        atom_nodes = [n for n, d in self._graph.nodes(data=True) if d.get('node_type') == 'atom']
        octahedra_nodes = [n for n, d in self._graph.nodes(data=True) if d.get('node_type') == 'octahedron']
        layer_nodes = [n for n in self._graph.nodes() if 'layer' in str(n).lower()]
        other_nodes = [n for n in self._graph.nodes() if n not in atom_nodes and n not in octahedra_nodes and n not in layer_nodes]
        
        
        return self._graph

    def get_octahedra(self) -> List[Dict]:
        """
        Get octahedra information extracted from graph.

        Returns
        -------
        list of dict
            Each dict contains:
            - 'id': octahedron identifier
            - 'central_atom_index': index of B-site atom
            - 'central_atom_symbol': element symbol of B-site
            - 'terminal_atoms': list of terminal (surface) atom indices
            - 'interlayer_atoms': list of inter-layer shared atom indices
            - 'intralayer_atoms': list of intra-layer shared atom indices
        """
        self._ensure_analyzed()
        octahedra = []
        atom_symbols = self.cell.get_chemical_symbols()

        for node, data in self._graph.nodes(data=True):
            if data.get('node_type') == 'octahedron':
                central_idx = data.get('central_atom')
                oct_info = {
                    'id': node,
                    'central_atom_index': central_idx,
                    'central_atom_symbol': atom_symbols[central_idx] if central_idx is not None else None,
                    'terminal_atoms': data.get('terminal_atoms', []),
                    'interlayer_atoms': data.get('interlayer_atoms', []),
                    'intralayer_atoms': data.get('intralayer_atoms', []),
                }
                octahedra.append(oct_info)

        return octahedra

    def get_layers(self) -> Dict:
        """
        Get layer/slab information extracted from graph.

        Returns
        -------
        dict
            Layer ID -> layer info dict containing:
            - 'position': 'surface' or 'central'
            - 'octahedra': list of octahedra IDs in this layer
            - 'octahedra_count': number of octahedra
            - 'z_coord': z-coordinate of layer
        """
        self._ensure_analyzed()
        layers = {}

        for node, data in self._graph.nodes(data=True):
            if data.get('node_type') == 'layer':
                layer_id = node.replace('layer_', '')

                layer_octahedra = []
                for neighbor in self._graph.neighbors(node):
                    if neighbor.startswith('octahedron_'):
                        layer_octahedra.append(neighbor)

                layers[layer_id] = {
                    'position': data.get('position'),
                    'octahedra': layer_octahedra,
                    'octahedra_count': len(layer_octahedra),
                    'z_coord': data.get('z_coord'),
                    'intralayer_x_atoms': data.get('intralayer_x_atoms', []),
                    'interlayer_x_atoms_above': data.get('interlayer_x_atoms_above', []),
                    'interlayer_x_atoms_below': data.get('interlayer_x_atoms_below', []),
                }

        return layers

    def get_spacers(self) -> List[Atoms]:
        """
        Get identified spacer molecules extracted from graph.

        Returns
        -------
        list of ase.Atoms
            Each Atoms object represents a spacer molecule.
            The 'info' dict contains:
            - 'original_indices': atom indices in the original structure
            - 'spacer_type': 'dj' or 'rp'
            - 'spacer_formula': chemical formula
        """
        self._ensure_analyzed()
        spacers = []
        processed_indices = set()

        for node, data in self._graph.nodes(data=True):
            if data.get('node_type') == 'atom' and data.get('is_spacer'):
                atom_idx = data.get('vasp_index')
                if atom_idx in processed_indices:
                    continue

                spacer_indices = data.get('spacer_indices', [atom_idx])
                processed_indices.update(spacer_indices)

                spacer_atoms = Atoms(
                    symbols=[self.cell[i].symbol for i in spacer_indices],
                    positions=[self.cell[i].position for i in spacer_indices],
                )
                spacer_atoms.info['original_indices'] = spacer_indices
                spacer_atoms.info['spacer_type'] = data.get('spacer_type')
                spacer_atoms.info['template_match'] = data.get('spacer_formula')

                spacers.append(spacer_atoms)

        return spacers

    def get_a_sites(self) -> List[Dict]:
        """
        Get A-site cation information extracted from graph.

        Returns
        -------
        list of dict
            Each dict contains:
            - 'atom_index': index in original structure
            - 'symbol': element symbol or formula
            - 'position': cartesian coordinates
            - 'is_molecular': True if part of molecular A-site
            - 'molecule_atoms': indices of all atoms in molecular A-site
            - 'formula': chemical formula
        """
        self._ensure_analyzed()
        a_sites = []
        processed_indices = set()
        atom_positions = self.cell.get_positions()

        for node, data in self._graph.nodes(data=True):
            if data.get('node_type') == 'atom' and data.get('is_a_site'):
                atom_idx = data.get('vasp_index')
                if atom_idx in processed_indices:
                    continue

                a_site_indices = data.get('a_site_indices', [atom_idx])
                processed_indices.update(a_site_indices)

                formula = data.get('a_site_formula', self.cell[atom_idx].symbol)
                a_sites.append({
                    'atom_index': atom_idx,
                    'symbol': formula,
                    'position': atom_positions[atom_idx].tolist(),
                    'is_molecular': len(a_site_indices) > 1,
                    'molecule_atoms': a_site_indices,
                    'formula': formula,
                })

        return a_sites

    def get_glazer_pattern(
        self, 
        tolerance: float = DEFAULT_TOLERANCE,
        tilt_significance_threshold: Optional[float] = None,
        zero_tilt_threshold: Optional[float] = None,
        magnitude_equivalence_threshold: Optional[float] = None,
        bond_selection_threshold: float = DEFAULT_BOND_SELECTION_THRESHOLD,
    ) -> Dict[str, Union[str, List[str], List[float], Optional[str], Dict]]:
        """
        Get Glazer tilt pattern notation from structure.

        Detects the Glazer notation by analyzing octahedral tilting patterns
        and phase relationships across unit cell boundaries.

        Parameters
        ----------
        tolerance : float, default=DEFAULT_TOLERANCE (0.1)
            Base tolerance for angle comparisons (degrees). Used as default for
            all thresholds if specific thresholds are not provided.
        tilt_significance_threshold : float, optional
            Minimum tilt angle (degrees) to consider a tilt significant for
            correlation calculations. Default: tolerance
        zero_tilt_threshold : float, optional
            Maximum tilt angle (degrees) to classify as zero tilt (pattern "0").
            Default: tolerance
        magnitude_equivalence_threshold : float, optional
            Maximum difference (degrees) between tilt angles to consider them
            equivalent (same magnitude letter: a, b, or c).
            Default: tolerance
        bond_selection_threshold : float, default=DEFAULT_BOND_SELECTION_THRESHOLD (0.1)
            Minimum projection value for bond selection along reference axes.
            Ensures consistent sign convention. Units: dimensionless (normalized).

        Returns
        -------
        dict
            Dictionary with keys:
            - 'notation': Full Glazer notation string (e.g., "a-b+a-")
            - 'tilt_pattern': List of tilt phases ['+', '-', '+']
            - 'tilt_angles': List of detected angles [omega_x, omega_y, omega_z] in degrees
            - 'magnitudes': List of magnitude symbols ['a', 'b', 'a']
            - 'space_group': Inferred space group (if available)
            - 'tolerance_info': dict with tolerance values used (for documentation)

        Notes
        -----
        See :func:`_detect_glazer_pattern` for detailed notes on tolerance sensitivity.
        """
        self._ensure_analyzed()
        return _detect_glazer_pattern(
            self, 
            tolerance,
            tilt_significance_threshold,
            zero_tilt_threshold,
            magnitude_equivalence_threshold,
            bond_selection_threshold,
        )

    def get_bxb_angles(
        self,
        bxb_scale: float = 1.4,
        supercell: Tuple[int, int, int] = (1, 1, 1),
        include_bp: bool = False,
    ) -> Dict[str, Union[np.ndarray, float, None]]:
        """
        Get B-X-B bond angles.

        Calculates B-X-B angles using the analyzer's graph structure,
        where X is a shared X-site atom between two octahedra.

        Parameters
        ----------
        bxb_scale : float
            Scale factor for bond cutoff (kept for API compatibility, not used in graph-based approach)
        supercell : Tuple[int, int, int]
            Supercell replication (kept for API compatibility, not used in graph-based approach)
        include_bp : bool
            Whether to calculate B-X-Bp angles (Bp = spacer B-site)

        Returns
        -------
        dict
            Dictionary with keys:
            - 'bxb_angles': numpy array of B-X-B angles in degrees
            - 'bxb_mean': Mean B-X-B angle
            - 'bxb_std': Standard deviation of B-X-B angles
            - 'bxbp_angles': numpy array of B-X-Bp angles (if include_bp=True)
            - 'bxbp_mean': Mean B-X-Bp angle (if include_bp=True)
        """
        self._ensure_analyzed()
        bxb_angles, bxbp_angles = _calculate_bxb_angles(self, bxb_scale, supercell, include_bp)

        result = {
            "bxb_angles": bxb_angles,
            "bxb_mean": float(np.mean(bxb_angles)) if bxb_angles is not None and len(bxb_angles) > 0 else None,
            "bxb_std": float(np.std(bxb_angles)) if bxb_angles is not None and len(bxb_angles) > 0 else None,
        }

        if include_bp:
            result["bxbp_angles"] = bxbp_angles
            result["bxbp_mean"] = (
                float(np.mean(bxbp_angles)) if bxbp_angles is not None and len(bxbp_angles) > 0 else None
            )

        return result

    def get_partial_rdf(
        self,
        element_pairs: List[List[str]],
        max_dist: float = 10.0,
        npoints: Optional[int] = None,
        ss_norm: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Get partial radial distribution functions for element pairs.

        Computes RDFs for specified element pairs using gaussian kernel smoothing
        for smooth, publication-quality results.

        Parameters
        ----------
        element_pairs : List[List[str]]
            List of [element1, element2] pairs to compute RDFs for
            Example: [["Pb", "I"], ["Pb", "Pb"], ["I", "I"]]
        max_dist : float
            Maximum distance in Angstroms for RDF computation (default: 10.0)
        npoints : int, optional
            Number of grid points. If None, automatically determined based on distance range
        ss_norm : bool
            Whether to apply structure-specific normalization (default: False)

        Returns
        -------
        dict
            Dictionary with keys like "PbI_x", "PbI_rdf" for each pair
            - "{pair}_x": x-axis (distance) values
            - "{pair}_rdf": RDF values
        """
        self._ensure_analyzed()
        return _calculate_partial_rdf(self, element_pairs, max_dist, npoints, ss_norm)

    def get_characterization(self) -> CharacterizationQuery:
        """
        Get a query builder for graph-based characterization analysis.

        Provides a unified interface for:
        - Routing standard analyses (Glazer, RDF, B-X-B)
        - Querying the graph structure
        - Method chaining for complex queries

        Returns
        -------
        CharacterizationQuery
            Query builder instance for characterization analysis

        Examples
        --------
        >>> # Standard analysis routing
        >>> result = analyzer.get_characterization().glazer(tolerance=0.1).execute()
        >>> result = analyzer.get_characterization().rdf([["Pb", "I"]]).execute()
        >>> result = analyzer.get_characterization().bxb(include_bp=True).execute()

        >>> # Graph queries with method chaining
        >>> result = (analyzer.get_characterization()
        ...          .octahedra()
        ...          .neighbors(edge_type='shares_atoms')
        ...          .to_list())
        """
        self._ensure_analyzed()
        return CharacterizationQuery(self)

    def compute_delta(
        self,
        octahedra: Optional[List[str]] = None,
        layer: Optional[str] = None,
        layers: Optional[List[str]] = None,
        group_by: Optional[str] = None,
    ) -> Union[float, Dict[str, float]]:
        """
        Compute octahedral bond length distortion parameter (Δ).

        Delta is the mean absolute deviation of B-X bond lengths from
        the average bond length, normalized by the mean bond length:

        Δ = (1/n) Σ |d_i - d_avg| / d_avg

        Parameters
        ----------
        octahedra : list of str, optional
            Specific octahedra IDs to include (e.g., ['octahedron_0', 'octahedron_1'])
        layer : str, optional
            Single layer ID to filter by (e.g., '0')
        layers : list of str, optional
            Multiple layer IDs to filter by (e.g., ['0', '1'])
        group_by : str, optional
            If 'layer', returns dict with per-layer results plus 'global' key

        Returns
        -------
        float or dict
            If group_by is None: Delta distortion parameter (float)
            If group_by == 'layer': Dict with layer IDs as keys, each containing delta value,
                                    plus 'global' key with overall delta
            Returns None if no octahedra found.

        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.cif")
        >>> analyzer.analyze()
        >>> # Global distortion
        >>> delta = analyzer.compute_delta()
        >>> print(f"Bond length distortion: {delta:.4f}")
        >>>
        >>> # Per-layer analysis
        >>> delta_by_layer = analyzer.compute_delta(group_by='layer')
        >>> for layer_id, delta in delta_by_layer.items():
        ...     print(f"Layer {layer_id}: Δ = {delta:.4f}")
        >>>
        >>> # Single layer
        >>> delta_layer0 = analyzer.compute_delta(layer='0')
        >>> print(f"Layer 0 distortion: {delta_layer0:.4f}")
        """
        from ..characterization.distortions import _compute_octahedral_distortions
        self._ensure_analyzed()
        result = _compute_octahedral_distortions(
            self, octahedra=octahedra, layer=layer, layers=layers, group_by=group_by
        )
        if group_by == 'layer':
            return {k: v['delta'] for k, v in result.items()}
        return result['delta']

    def compute_sigma(
        self,
        octahedra: Optional[List[str]] = None,
        layer: Optional[str] = None,
        layers: Optional[List[str]] = None,
        group_by: Optional[str] = None,
    ) -> Union[float, Dict[str, float]]:
        """
        Compute octahedral bond length variance parameter (σ²).

        Sigma squared is the variance of B-X bond lengths, normalized
        by the square of the mean bond length:

        σ² = Var(d_i) / d_avg²

        Parameters
        ----------
        octahedra : list of str, optional
            Specific octahedra IDs to include
        layer : str, optional
            Single layer ID to filter by
        layers : list of str, optional
            Multiple layer IDs to filter by
        group_by : str, optional
            If 'layer', returns dict with per-layer results plus 'global' key

        Returns
        -------
        float or dict
            If group_by is None: Sigma squared distortion parameter (float)
            If group_by == 'layer': Dict with layer IDs as keys
            Returns None if no octahedra found.

        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.cif")
        >>> analyzer.analyze()
        >>> sigma = analyzer.compute_sigma()
        >>> print(f"Bond length variance: {sigma:.6f}")
        >>>
        >>> # Per-layer analysis
        >>> sigma_by_layer = analyzer.compute_sigma(group_by='layer')
        >>> for layer_id, sigma in sigma_by_layer.items():
        ...     print(f"Layer {layer_id}: σ² = {sigma:.6f}")
        """
        from ..characterization.distortions import _compute_octahedral_distortions
        self._ensure_analyzed()
        result = _compute_octahedral_distortions(
            self, octahedra=octahedra, layer=layer, layers=layers, group_by=group_by
        )
        if group_by == 'layer':
            return {k: v['sigma'] for k, v in result.items()}
        return result['sigma']

    def compute_lambda(
        self,
        octahedra: Optional[List[str]] = None,
        layer: Optional[str] = None,
        layers: Optional[List[str]] = None,
        group_by: Optional[str] = None,
    ) -> Union[float, Dict[str, float]]:
        """
        Compute octahedral bond angle distortion parameter (λ²).

        Lambda squared is the variance of X-B-X bond angle deviations
        from ideal angles (90° or 180°):

        λ² = Var(min(|θ_i - 90°|, |θ_i - 180°|))

        Parameters
        ----------
        octahedra : list of str, optional
            Specific octahedra IDs to include
        layer : str, optional
            Single layer ID to filter by
        layers : list of str, optional
            Multiple layer IDs to filter by
        group_by : str, optional
            If 'layer', returns dict with per-layer results plus 'global' key

        Returns
        -------
        float or dict
            If group_by is None: Lambda squared distortion parameter (float)
            If group_by == 'layer': Dict with layer IDs as keys
            Returns None if no octahedra found.

        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.cif")
        >>> analyzer.analyze()
        >>> lambda_param = analyzer.compute_lambda()
        >>> print(f"Bond angle variance: {lambda_param:.4f}")
        >>>
        >>> # Per-layer analysis
        >>> lambda_by_layer = analyzer.compute_lambda(group_by='layer')
        >>> for layer_id, lambda_val in lambda_by_layer.items():
        ...     print(f"Layer {layer_id}: λ² = {lambda_val:.4f}")
        """
        from ..characterization.distortions import _compute_octahedral_distortions
        self._ensure_analyzed()
        result = _compute_octahedral_distortions(
            self, octahedra=octahedra, layer=layer, layers=layers, group_by=group_by
        )
        if group_by == 'layer':
            return {k: v['lambda'] for k, v in result.items()}
        return result['lambda']

    def get_octahedral_distortions(
        self,
        octahedra: Optional[List[str]] = None,
        layer: Optional[str] = None,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Union[float, np.ndarray, str]]]:
        """
        Get detailed per-octahedron distortion metrics with layer information.

        This method returns comprehensive distortion data for each octahedron,
        including delta, sigma, lambda parameters, raw bond lengths and angles,
        and layer assignment. Useful for detailed analysis and custom visualizations.

        Parameters
        ----------
        octahedra : list of str, optional
            Specific octahedra IDs to include (e.g., ['octahedron_0', 'octahedron_1'])
        layer : str, optional
            Single layer ID to filter by (e.g., '0')
        layers : list of str, optional
            Multiple layer IDs to filter by (e.g., ['0', '1'])

        Returns
        -------
        dict
            Octahedron ID -> metrics dict containing:
            - 'delta': Bond length distortion (Δ)
            - 'sigma': Bond length variance (σ²)
            - 'lambda': Bond angle variance (λ²)
            - 'bond_lengths': Array of B-X bond lengths
            - 'bond_angles': Array of X-B-X angles
            - 'mean_bond_length': Mean B-X bond length
            - 'mean_angle': Mean X-B-X angle
            - 'layer': Layer ID this octahedron belongs to
            - 'central_atom_index': Index of B-site atom
            - 'central_atom_symbol': Element symbol of B-site

        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.cif")
        >>> analyzer.analyze()
        >>>
        >>> # Get all octahedra distortions
        >>> oct_data = analyzer.get_octahedral_distortions()
        >>> for oct_id, metrics in oct_data.items():
        ...     print(f"{oct_id} (Layer {metrics['layer']}):")
        ...     print(f"  Δ={metrics['delta']:.4f}, σ²={metrics['sigma']:.6f}")
        >>>
        >>> # Filter by layer
        >>> layer0_oct = analyzer.get_octahedral_distortions(layer='0')
        >>> print(f"Layer 0 has {len(layer0_oct)} octahedra")
        >>>
        >>> # Access specific octahedron
        >>> oct0 = analyzer.get_octahedral_distortions(octahedra=['octahedron_0'])
        >>> print(f"Bond lengths: {oct0['octahedron_0']['bond_lengths']}")
        """
        from ..characterization.distortions import _get_octahedral_distortions_detailed
        self._ensure_analyzed()
        return _get_octahedral_distortions_detailed(
            self, octahedra=octahedra, layer=layer, layers=layers
        )

    @property
    def structure_type(self) -> str:
        """
        Get the inferred structure type.

        Returns
        -------
        str
            One of: 'bulk', 'dj', 'rp', 'monolayer'
        """
        self._ensure_analyzed()
        return self._structure_type

    def analyze_spacer_molecules(self, spacer_type: str = "DJ") -> List:
        """Analyze all spacer molecules in structure.

        Computes penetration depth, compression, backbone, and side chains
        for all spacer molecules detected in the structure.

        Parameters
        ----------
        spacer_type : str, default="DJ"
            Type of spacer to analyze ("DJ" or "RP")

        Returns
        -------
        List[SpacerAnalysisResult]
            List of analysis results, one per spacer molecule

        Raises
        ------
        RuntimeError
            If analyze() has not been called
        ValueError
            If no spacers found in structure

        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.cif")
        >>> analyzer.analyze()
        >>>
        >>> # Analyze all spacers
        >>> results = analyzer.analyze_spacer_molecules()
        >>>
        >>> for i, result in enumerate(results):
        ...     print(f"Spacer {i}: compression={result.compression_factor:.2f}")
        """
        from q2D_Materials.analyzer.characterization.spacer_analysis import SpacerAnalysis
        from q2D_Materials.modifier.graph_view import GraphView

        self._ensure_analyzed()

        # Create GraphView to access molecules
        view = GraphView(self)

        # Get spacers
        spacers = view.spacers.list()

        if not spacers:
            raise ValueError("No spacers found in structure")

        # Analyze each spacer
        results = []
        for spacer in spacers:
            analyzer_inst = SpacerAnalysis(spacer, self, spacer_type=spacer_type)
            results.append(analyzer_inst.compute())

        return results

    def _ensure_analyzed(self) -> None:
        """Ensure analyze() has been called."""
        if not self._analyzed:
            raise RuntimeError(
                "Structure has not been analyzed. Call analyze() first."
            )

    def mol_validate(
        self,
        molecule: Union[Atoms, str],
        spacer_type: str = "DJ",
        initial_pattern: Union[str, List[str]] = None,
        final_pattern: Union[str, List[str]] = None,
        min_chain_length: int = 2,
        allowed_backbone_elements: Optional[Set[str]] = None,
        forbidden_backbone_elements: Optional[Set[str]] = None,
        max_non_carbon_ratio: Optional[float] = None,
        allow_same_carbon: bool = False
    ):
        """
        Validate if molecule is suitable as DJ or RP spacer using pattern matching.

        DJ (Dion-Jacobson) spacers require 2 terminal groups with valid paths between them.
        RP (Ruddlesden-Popper) spacers require at least 1 terminal group.

        Parameters
        ----------
        molecule : Atoms or str
            ASE Atoms object or SMILES string
        spacer_type : str
            Type of spacer: "DJ" or "RP" (default: "DJ")
        initial_pattern : str or List[str], optional
            SMILES pattern(s) for initial terminal (default: 'NH2C')
            Examples: '[NH3+]C', 'NH2C', ['[NH3+]C', 'NH2C']
        final_pattern : str or List[str], optional
            SMILES pattern(s) for final terminal (default: 'NH2C')
            Ignored for RP spacers
        min_chain_length : int
            Minimum atoms between terminal carbons for DJ (default: 2)
            Ignored for RP spacers
        allowed_backbone_elements : Set[str], optional
            Elements allowed in backbone path (default: {'C', 'N', 'O'})
        forbidden_backbone_elements : Set[str], optional
            Elements forbidden in backbone path
        max_non_carbon_ratio : float, optional
            Maximum ratio of non-carbon heavy atoms in backbone (default: 0.3)
        allow_same_carbon : bool
            Whether terminals can share the same carbon (default: False)
            Ignored for RP spacers

        Returns
        -------
        SpacerCandidateResult
            Analysis with validity, terminal groups, and paths (DJ only)

        Examples
        --------
        >>> analyzer = q2D_analyzer()
        >>> # Validate as DJ spacer
        >>> result = analyzer.mol_validate("NCCCCN", spacer_type="DJ")
        >>> print(f"Valid: {result.is_valid}")
        >>> 
        >>> # Validate as RP spacer
        >>> result = analyzer.mol_validate("NCCCCC", spacer_type="RP")
        >>> print(f"Valid: {result.is_valid}")
        """
        from ..characterization.molecule_candidates import analyze_molecule_candidate
        return analyze_molecule_candidate(
            molecule,
            spacer_type=spacer_type.upper(),
            initial_pattern=initial_pattern,
            final_pattern=final_pattern,
            min_chain_length=min_chain_length,
            allowed_backbone_elements=allowed_backbone_elements,
            forbidden_backbone_elements=forbidden_backbone_elements,
            max_non_carbon_ratio=max_non_carbon_ratio,
            allow_same_carbon=allow_same_carbon
        )

    def convert_molecule_nh2_to_nh3(
        self,
        molecule: Atoms,
        n_indices: Optional[List[int]] = None
    ) -> Atoms:
        """
        Convert NH2 groups to NH3 by adding hydrogen atoms.

        Adds H atoms with proper tetrahedral geometry (sp3, 109.5° angles).

        Parameters
        ----------
        molecule : Atoms
            Molecule with NH2 groups
        n_indices : List[int], optional
            Nitrogen indices to convert. If None, converts all NH2.

        Returns
        -------
        Atoms
            Modified molecule with NH3 groups

        Examples
        --------
        >>> result = analyzer.mol_validate("NCCN", spacer_type="DJ")
        >>> # If NH2 groups detected
        >>> modified = analyzer.convert_molecule_nh2_to_nh3(result.original_atoms)
        """
        from ..characterization.molecule_candidates import convert_nh2_to_nh3, clean_molecule

        # If no specific indices, use clean_molecule to convert all NH2 to NH3
        if n_indices is None:
            return clean_molecule(molecule, convert_nh2_to_nh3_flag=True)

        # Convert specific NH2 groups to NH3
        modified = molecule.copy()
        for n_idx in n_indices:
            modified = convert_nh2_to_nh3(modified, n_idx)

        return modified

    def elongate_dj_spacer(
        self,
        molecule: Atoms,
        target_distance: Optional[float] = None,
        max_iterations: int = 100
    ) -> Atoms:
        """
        Elongate DJ spacer molecule using CCD kinematics.

        Uses cyclic coordinate descent (CCD) to extend the molecule
        to reach the target N-N distance.

        Parameters
        ----------
        molecule : Atoms
            DJ spacer molecule to elongate
        target_distance : float, optional
            Target N-N distance in Angstroms. If None, maximizes extension.
        max_iterations : int
            Maximum CCD optimization iterations (default: 100)

        Returns
        -------
        Atoms
            Elongated molecule

        Examples
        --------
        >>> result = analyzer.mol_validate("NCCCCN", spacer_type="DJ")
        >>> elongated = analyzer.elongate_dj_spacer(
        ...     result.original_atoms,
        ...     target_distance=12.0
        ... )
        """
        from ...builders.optimizers import elongate_molecule
        return elongate_molecule(molecule, max_iterations=max_iterations, target_distance=target_distance)

    def to_q2DStructure(self) -> q2DStructure:
        """
        Convert analysis results to a q2DStructure object.
        
        Returns
        -------
        q2DStructure
            Structure with populated metadata from analysis
        """
        self._ensure_analyzed()

        octahedra = self.get_octahedra()
        b_ions = list(set(
            oct['central_atom_symbol']
            for oct in octahedra
            if oct['central_atom_symbol'] is not None
        ))

        atom_symbols = self.cell.get_chemical_symbols()
        x_ions = set()
        for oct in octahedra:
            for idx in oct['terminal_atoms'] + oct['interlayer_atoms'] + oct['intralayer_atoms']:
                x_ions.add(atom_symbols[idx])
        x_ions = list(x_ions)

        a_sites = self.get_a_sites()
        a_ions = list(set(a['symbol'] for a in a_sites))

        spacers = self.get_spacers()
        spacer = spacers if spacers else None
        
        return q2DStructure(
            self.cell,
            structure_type=self._structure_type,
            A_ions=a_ions if a_ions else None,
            B_ions=b_ions if b_ions else None,
            X_ions=x_ions if x_ions else None,
            spacer=spacer,
            analysis_graph=self._graph,
            octahedra=octahedra,
            layers=self.get_layers(),
        )
    
    def export_graph_data(self, output_path: Optional[str] = None) -> str:
        """
        Export graph visualization data as JSON.

        Exports the structural analysis as a JSON file containing:
        - Nodes (octahedra and layers) with metadata
        - Edges (connections between components)
        - Structure metadata (type, counts, formula)
        - Atom details for detailed inspection

        This data can be consumed by the q2D Materials Analyzer web interface
        or other visualization tools.

        Parameters
        ----------
        output_path : str, optional
            Path to save the JSON file. If None, uses '{experiment_name}_graph.json'

        Returns
        -------
        str
            Path to the created JSON file
        """
        self._ensure_analyzed()

        import json
        import os

        if output_path is None:
            output_path = f"{self.experiment_name}_graph.json"

        def to_native(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if obj is None:
                return None
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return [to_native(x) for x in obj.tolist()]
            elif isinstance(obj, list):
                return [to_native(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif hasattr(obj, 'tolist'):
                return to_native(obj.tolist())
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            else:
                return str(obj)

        nodes = []
        node_id_map = {}
        node_counter = 0

        element_colors = {
            'Ti': '#8be9fd', 'Sn': '#50fa7b', 'Pb': '#ff5555', 'Ge': '#ffb86c',
            'Zr': '#bd93f9', 'Hf': '#ff79c6', 'Nb': '#f1fa8c', 'Ta': '#6272a4'
        }
        default_color = '#44475a'

        octahedra = self.get_octahedra()
        for oct in octahedra:
            node_id = oct['id']
            node_id_map[node_id] = node_counter
            b_element = oct.get('central_atom_symbol', 'Unknown')
            color = element_colors.get(b_element, default_color)

            nodes.append({
                'id': int(node_counter),
                'original_id': str(node_id),
                'type': 'octahedron',
                'label': f"Oct-{b_element}",
                'b_element': str(b_element),
                'color': str(color),
                'central_atom_index': to_native(oct.get('central_atom_index')),
                'terminal_atoms': to_native(oct.get('terminal_atoms', [])),
                'interlayer_atoms': to_native(oct.get('interlayer_atoms', [])),
                'intralayer_atoms': to_native(oct.get('intralayer_atoms', [])),
            })
            node_counter += 1

        layers = self.get_layers()
        for layer_id, layer_info in layers.items():
            node_id = f"layer_{layer_id}"
            node_id_map[node_id] = node_counter

            nodes.append({
                'id': int(node_counter),
                'original_id': str(node_id),
                'type': 'layer',
                'label': f"Layer-{layer_id}",
                'layer_id': str(layer_id),
                'position': str(layer_info.get('position', 'unknown')),
                'z_coord': to_native(layer_info.get('z_coord')),
                'octahedra': to_native(layer_info.get('octahedra', [])),
                'octahedra_count': int(layer_info.get('octahedra_count', 0)),
                'intralayer_x_atoms': to_native(layer_info.get('intralayer_x_atoms', [])),
                'interlayer_x_atoms_above': to_native(layer_info.get('interlayer_x_atoms_above', [])),
                'interlayer_x_atoms_below': to_native(layer_info.get('interlayer_x_atoms_below', [])),
                'color': '#ff79c6'
            })
            node_counter += 1

        edges = []
        for source, target in self._graph.edges():
            if source in node_id_map and target in node_id_map:
                edges.append({
                    'from': int(node_id_map[source]),
                    'to': int(node_id_map[target]),
                    'source_id': str(source),
                    'target_id': str(target)
                })

        atom_positions = self.cell.get_positions()
        atom_symbols = self.cell.get_chemical_symbols()
        atoms = []
        for i, (pos, symbol) in enumerate(zip(atom_positions, atom_symbols)):
            atom_data = {
                'index': int(i),
                'symbol': str(symbol),
                'position': to_native(pos),
            }
            atoms.append(atom_data)

        spacers = self.get_spacers()
        spacers_data = []
        for spacer in spacers:
            spacer_info = {
                'formula': str(spacer.get_chemical_formula(mode='hill')),
                'atom_count': int(len(spacer)),
                'original_indices': to_native(spacer.info.get('original_indices', []) if hasattr(spacer, 'info') else [])
            }
            spacers_data.append(spacer_info)

        a_sites = self.get_a_sites()
        a_sites_data = []
        for a_site in a_sites:
            a_sites_data.append({
                'atom_index': to_native(a_site.get('atom_index')),
                'symbol': str(a_site.get('symbol', '')),
                'position': to_native(a_site.get('position')),
                'is_molecular': bool(a_site.get('is_molecular', False)),
                'formula': str(a_site.get('formula', a_site.get('symbol', '')))
            })

        layer_edges = []
        layer_ids = sorted([int(k) for k in layers.keys()])
        for i in range(len(layer_ids) - 1):
            layer_a = str(layer_ids[i])
            layer_b = str(layer_ids[i + 1])
            connecting_x_atoms = layers.get(layer_a, {}).get('interlayer_x_atoms_above', [])
            if connecting_x_atoms:
                layer_edges.append({
                    'from_layer': int(layer_ids[i]),
                    'to_layer': int(layer_ids[i + 1]),
                    'via_x_atoms': to_native(connecting_x_atoms),
                })

        graph_data = {
            'metadata': {
                'experiment_name': str(self.experiment_name),
                'file_path': str(self.file_path) if self.file_path else None,
                'structure_type': str(self._structure_type),
                'total_atoms': int(len(self.cell)),
                'formula': str(self.cell.get_chemical_formula()),
                'octahedra_count': int(len(octahedra)),
                'layers_count': int(len(layers)),
                'spacers_count': int(len(spacers)),
                'a_sites_count': int(len(a_sites)),
                'cell': to_native(self.cell.get_cell()),
            },
            'nodes': nodes,
            'edges': edges,
            'atoms': atoms,
            'spacers': spacers_data,
            'a_sites': a_sites_data,
            'layer_edges': layer_edges,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)

        return os.path.abspath(output_path)

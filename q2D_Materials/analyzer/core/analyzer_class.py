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

from typing import Optional, List, Dict, Any, Union, Tuple, Set, TYPE_CHECKING
import numpy as np
import networkx as nx
from ase.io import read
from ase import Atoms
import os

from .graph_construction import _graph_inorganic_ontology
from ..octahedral_processing.octahedral_detection import _count_octahedra, find_shared_atoms
from .layer_identification import _identify_layers
from ..characterization.characterization import (
    _detect_glazer_pattern, 
    _calculate_bxb_angles, 
    _calculate_partial_rdf,
    DEFAULT_TOLERANCE,
    DEFAULT_BOND_SELECTION_THRESHOLD,
    CharacterizationQuery,
)
from ...core.structure import q2DStructure

if TYPE_CHECKING:
    from .layers_wrapper import Layers
    from .slabs_wrapper import Slabs


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
        self._b_x_atoms_cache: Optional[Dict[str, np.ndarray]] = None
        self._octahedral_distortions_cache: Optional[Dict[tuple, Dict]] = None
        self._layers: Optional["Layers"] = None
        self._stacks_info: Optional[Dict[str, Any]] = None
        self._octahedra_info: Optional[List[Dict]] = None
        self._slabs: Optional["Slabs"] = None

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
            from ...utils.molecules.hydrogen_cleanup import prepare_experimental_structure
            self.cell, self._structure_cleanup = prepare_experimental_structure(
                source, hint=source
            )
            self._n_hydrogens_merged = int(self._structure_cleanup.get("n_h_merged", 0))
        elif isinstance(source, Atoms):
            from ...utils.molecules.hydrogen_cleanup import prepare_experimental_structure
            self.cell, self._structure_cleanup = prepare_experimental_structure(source)
            self._n_hydrogens_merged = int(self._structure_cleanup.get("n_h_merged", 0))
            self.experiment_name = "atoms_input"
        else:
            raise TypeError(f"source must be str or Atoms, got {type(source)}")

        self.cell.pbc = True

        self._analyzed = False
        self._graph = None
        self._structure_type = None
        self._b_x_atoms_cache = None
        self._octahedral_distortions_cache = None
        self._layers = None
        self._stacks_info = None
        self._octahedra_info = None
        self._slabs = None

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
        
        # Store for later use in cavity detection
        self._atom_positions = np.array(atom_positions)
        self._atom_symbols = list(atom_symbols)
        self._cell = cell_matrix
        self._octahedral_distortions_cache = None

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
            # Add missing atoms with x, y, z coordinates
            for i in range(n_atoms):
                if f'atom_{i}' not in self._graph:
                    pos = atom_positions[i]
                    node_data = {
                        'node_type': 'atom',
                        'vasp_index': i,
                        'symbol': atom_symbols[i] if atom_symbols else 'Unknown',
                        'x': float(pos[0]),
                        'y': float(pos[1]),
                        'z': float(pos[2]),
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

        # Structure type inference uses graph patterns (after slab enrichment)
        from ..octahedral_processing.octahedral_detection import build_octahedra_ligand_info
        from ..twister_processing.stack_detection import detect_stacks
        from ..twister_processing.graph_construction import enrich_graph_with_slabs

        atom_symbols = list(atom_symbols)
        octahedra_info, neighbor_indices = build_octahedra_ligand_info(
            self._graph, atom_symbols=atom_symbols
        )
        self._octahedra_info = octahedra_info
        self._stacks_info = detect_stacks(
            self._graph,
            octahedra_info,
            neighbor_indices,
            np.array(atom_positions),
            cell=np.array(self.cell.get_cell()),
        )
        enrich_graph_with_slabs(
            self._graph,
            self._stacks_info,
            atom_positions=np.array(atom_positions),
        )

        # Twister is user/creator-declared only; never inferred from multi-slab geometry.
        declared_type = getattr(self.cell, 'structure_type', None)
        if declared_type == 'twister':
            self._structure_type = 'twister'
        else:
            self._structure_type = self._infer_structure_type_from_graph()

        structure_node = self._graph.graph.get('structure_node', 'structure_0')
        if structure_node in self._graph:
            self._graph.nodes[structure_node]['is_twister'] = (
                self._structure_type == 'twister'
            )

        # Create B/X atom cache during analysis so all functions can access it
        self._compute_b_x_atoms_cache()

        self._analyzed = True
        return self

    def _infer_structure_type_from_graph(self) -> str:
        """Infer structure type from graph patterns.

        Reuses slab/octahedra results already computed in ``analyze()`` so
        ligand walking and slab-continuity are not repeated.
        """
        from .structure_classification import _infer_structure_type_from_graph

        if self._stacks_info is None:
            from .layer_identification import _identify_slabs_by_continuity
            from ..octahedral_processing.octahedral_detection import (
                build_octahedra_ligand_info,
                find_shared_atoms,
            )

            atom_positions = self.cell.get_positions()
            atom_symbols = self.cell.get_chemical_symbols()
            octahedra_info, neighbor_indices = build_octahedra_ligand_info(
                self._graph, atom_symbols=atom_symbols
            )
            shared_atoms = find_shared_atoms(neighbor_indices)
            slab_info = _identify_slabs_by_continuity(
                octahedra_info,
                shared_atoms,
                atom_positions,
                cell=np.array(self.cell.get_cell()),
            )
        else:
            slab_info = self._stacks_info
            octahedra_info = self._octahedra_info
            if not octahedra_info and self._graph is not None:
                octahedra_info = [
                    n for n, d in self._graph.nodes(data=True)
                    if d.get('node_type') == 'octahedron'
                ]

        spacers = self._get_spacers_internal()

        return _infer_structure_type_from_graph(
            slab_info,
            spacers,
            np.array(self.cell.get_cell()),
            graph=self._graph,
            octahedra=octahedra_info,
        )
    
    def _get_spacers_internal(self) -> List[Atoms]:
        """Internal method to get spacers without analyzed check."""
        spacers = []

        # Query Spacer nodes
        for node, data in self._graph.nodes(data=True):
            if data.get('node_type') == 'spacer':
                # Get atoms in this molecule via CONTAINS edges
                atom_indices = []
                for neighbor in self._graph.neighbors(node):
                    edge_data = self._graph.get_edge_data(node, neighbor)
                    if edge_data and edge_data.get('edge_type') == 'contains':
                        neighbor_data = self._graph.nodes.get(neighbor, {})
                        if neighbor_data.get('node_type') == 'atom':
                            atom_idx = neighbor_data.get('vasp_index')
                            if atom_idx is not None:
                                atom_indices.append(atom_idx)
                
                if atom_indices:
                    # Include cell and pbc from parent structure to enable
                    # PBC-aware bond detection in downstream graph construction.
                    # Without this, molecules spanning PBC boundaries will have
                    # broken connectivity when their graph is built.
                    spacer_atoms = Atoms(
                        symbols=[self.cell[i].symbol for i in atom_indices],
                        positions=[self.cell[i].position for i in atom_indices],
                        cell=self.cell.get_cell(),
                        pbc=self.cell.get_pbc(),
                    )
                    spacer_atoms.info['original_indices'] = atom_indices
                    spacer_atoms.info['spacer_formula'] = data.get('formula')
                    spacer_atoms.info['template_match'] = data.get('formula')
                    spacer_atoms.info['nh3_count'] = data.get('nh3_count', 0)
                    spacers.append(spacer_atoms)

        return spacers



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
        
        # Convert to sets for O(1) membership testing in large graphs
        atom_nodes_set = set(atom_nodes)
        octahedra_nodes_set = set(octahedra_nodes)
        layer_nodes_set = set(layer_nodes)
        other_nodes = [n for n in self._graph.nodes() if n not in atom_nodes_set and n not in octahedra_nodes_set and n not in layer_nodes_set]
        
        
        return self._graph

    def get_structure_metadata(self) -> Dict[str, Any]:
        """
        Get structure metadata from the Structure root node.
        
        The Structure node contains global properties like chemical formula,
        cell parameters, and counts of structural components.
        
        Returns
        -------
        dict
            Structure metadata containing:
            - 'formula': Chemical formula (Hill notation)
            - 'thickness': Number of octahedral layers
            - 'a', 'b', 'c': Cell lengths in Angstroms
            - 'alpha', 'beta', 'gamma': Cell angles in degrees
            - 'layer_count': Number of layers
            - 'octahedra_count': Number of octahedra
            - 'a_site_count': Number of A-site nodes
            - 'spacer_count': Number of spacer nodes
            - 'atom_count': Total atoms
            - 'experiment_name': Experiment identifier (if set)
            - 'file_path': Source file path (if loaded from file)
        
        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.vasp")
        >>> analyzer.analyze()
        >>> metadata = analyzer.get_structure_metadata()
        >>> print(f"Formula: {metadata['formula']}")
        >>> print(f"Cell: a={metadata['a']:.3f}, b={metadata['b']:.3f}, c={metadata['c']:.3f}")
        """
        self._ensure_analyzed()
        from .graph_construction import get_structure_node
        
        structure_data = get_structure_node(self._graph)
        if structure_data is None:
            structure_data = {}
        
        # Add analyzer-level metadata
        structure_data['experiment_name'] = self.experiment_name
        structure_data['file_path'] = str(self.file_path) if self.file_path else None
        
        return structure_data

    def _compute_b_x_atoms_cache(self) -> None:
        """
        Compute and cache B-site and X-site atom indices and positions.
        
        This method is called during analyze() to pre-compute the cache.
        B atoms are those that are CONTAINS (role='center') by an octahedron.
        X atoms are those that are BONDED_TO (role='ligand') by a B atom.
        """
        # Use _graph directly since we're called from within analyze()
        # where the graph has already been constructed
        graph = self._graph
        atom_positions = self.cell.get_positions()
        
        b_indices = []
        x_indices_set = set()

        # B-sites: Octahedron --[CONTAINS, role=center]--> B atom
        for node, data in graph.nodes(data=True):
            if data.get('node_type') != 'octahedron':
                continue
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if (edge_data and
                    edge_data.get('edge_type') == 'contains' and
                    edge_data.get('role') == 'center'):
                    vasp_idx = graph.nodes.get(neighbor, {}).get('vasp_index')
                    if vasp_idx is not None:
                        b_indices.append(vasp_idx)
                    break

        # X-sites: B --[BONDED_TO, role=ligand]--> X atom
        for v in b_indices:
            b_node = f'atom_{v}'
            if b_node not in graph:
                continue
            for neighbor in graph.neighbors(b_node):
                edge_data = graph.get_edge_data(b_node, neighbor)
                if (edge_data and
                    edge_data.get('edge_type') == 'bonded_to' and
                    edge_data.get('role') == 'ligand'):
                    vasp_idx = graph.nodes.get(neighbor, {}).get('vasp_index')
                    if vasp_idx is not None:
                        x_indices_set.add(vasp_idx)

        b_indices = np.array(b_indices, dtype=np.int32)
        x_indices = np.array(sorted(x_indices_set), dtype=np.int32)
        b_positions = atom_positions[b_indices]
        x_positions = atom_positions[x_indices]
        
        # Cache the result
        self._b_x_atoms_cache = {
            'b_indices': b_indices,
            'b_positions': b_positions,
            'x_indices': x_indices,
            'x_positions': x_positions,
        }

    def get_b_x_atoms(self) -> Dict[str, np.ndarray]:
        """
        Get cached B-site and X-site atom indices and positions.
        
        The cache is automatically created during analyze(), so this method
        just returns the pre-computed data.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'b_indices': numpy array of B atom indices
            - 'b_positions': numpy array of B atom positions (N, 3)
            - 'x_indices': numpy array of X atom indices
            - 'x_positions': numpy array of X atom positions (M, 3)
        
        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.vasp")
        >>> analyzer.analyze()
        >>> b_x_data = analyzer.get_b_x_atoms()
        >>> print(f"Found {len(b_x_data['b_indices'])} B atoms and {len(b_x_data['x_indices'])} X atoms")
        """
        self._ensure_analyzed()
        
        # Cache should already exist from analyze(), but compute if missing
        if self._b_x_atoms_cache is None:
            self._compute_b_x_atoms_cache()
        
        return self._b_x_atoms_cache

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
            - 'ligand_atoms': list of ligand (X-site) atom indices
        """
        self._ensure_analyzed()
        if self._octahedra_info is not None:
            return self._octahedra_info

        from ..octahedral_processing.octahedral_detection import build_octahedra_ligand_info

        atom_symbols = self.cell.get_chemical_symbols()
        octahedra_info, _ = build_octahedra_ligand_info(self._graph, atom_symbols=atom_symbols)
        self._octahedra_info = octahedra_info
        return octahedra_info

    def get_layers(self) -> Dict:
        """
        Get layer/slab information extracted from graph.

        Returns
        -------
        dict
            Layer ID -> layer info dict containing:
            - 'octahedra': list of octahedra IDs in this layer
            - 'octahedra_count': number of octahedra
            - 'z_coord': z-coordinate of layer
        """
        self._ensure_analyzed()
        layers = {}

        for node, data in self._graph.nodes(data=True):
            if data.get('node_type') == 'layer':
                layer_id = node.replace('layer_', '')

                # Query octahedra via CONTAINS edges
                layer_octahedra = []
                for neighbor in self._graph.neighbors(node):
                    edge_data = self._graph.get_edge_data(node, neighbor)
                    if edge_data and edge_data.get('edge_type') == 'contains':
                        if neighbor.startswith('octahedron_'):
                            layer_octahedra.append(neighbor)

                layers[layer_id] = {
                    'octahedra': layer_octahedra,
                    'octahedra_count': len(layer_octahedra),
                    'z_coord': data.get('z_coord'),
                }

        return layers

    def get_inorganic_n(self) -> Optional[int]:
        """Inorganic perovskite thickness n (octahedral sheets per slab).

        This matches NMSE ``n_layers`` / RP–DJ formula n, and is **not** the
        number of layer or slab graph nodes in the cell (bilayer RP n=1 cells
        typically have two layer nodes).
        """
        self._ensure_analyzed()
        from .inorganic_n import infer_inorganic_n_from_slabs

        return infer_inorganic_n_from_slabs(self.get_slabs(), self._graph, self.cell)

    @property
    def layers(self) -> "Layers":
        """
        Get a Layers wrapper for convenient layer-specific analysis.
        
        Returns
        -------
        Layers
            Wrapper object providing access to layer data and layer-specific
            analysis methods like get_bxb(index=0) for B-X-B angles.
            
        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.vasp")
        >>> analyzer.analyze()
        >>> 
        >>> # Get intra-layer B-X-B angles for layer 0
        >>> bxb_data = analyzer.layers.get_bxb(index=0)
        >>> print(f"Layer 0 B-X-B mean: {bxb_data['bxb_mean']:.2f}°")
        >>>
        >>> # Get inter-layer B-X-B angles between layers 0 and 1
        >>> interlayer_data = analyzer.layers.get_interlayer_bxb('0', '1')
        >>> print(f"Inter-layer mean: {interlayer_data['bxb_mean']:.2f}°")
        >>>
        >>> # Dict-like access to layer data
        >>> layer_0 = analyzer.layers['0']
        >>> print(f"Layer 0 has {layer_0['octahedra_count']} octahedra")
        >>>
        >>> # Iterate over layers
        >>> for layer_id in analyzer.layers:
        ...     print(f"Layer {layer_id}")
        """
        self._ensure_analyzed()
        if self._layers is None:
            from .layers_wrapper import Layers
            self._layers = Layers(self)
        return self._layers

    def get_slabs(self) -> Dict[str, Dict[str, Any]]:
        """Get slab (stack) information from the graph.

        Returns
        -------
        dict
            Slab ID -> info with ``layer_ids``, ``octahedra``, ``octahedra_count``,
            ``z_range``.
        """
        self._ensure_analyzed()
        slabs: Dict[str, Dict[str, Any]] = {}

        for node, data in self._graph.nodes(data=True):
            if data.get('node_type') != 'slab':
                continue
            slab_id = str(node).replace('slab_', '')
            layer_ids = list(data.get('layer_ids', []))
            octahedra: List[str] = []
            for lid in layer_ids:
                layer_node = f'layer_{lid}'
                if layer_node not in self._graph:
                    continue
                for neighbor in self._graph.neighbors(layer_node):
                    edge_data = self._graph.get_edge_data(layer_node, neighbor)
                    if edge_data and edge_data.get('edge_type') == 'contains':
                        if str(neighbor).startswith('octahedron_'):
                            octahedra.append(neighbor)

            slabs[slab_id] = {
                'layer_ids': layer_ids,
                'octahedra': octahedra,
                'octahedra_count': len(octahedra),
                'z_range': data.get('z_range'),
            }

        if not slabs and self._stacks_info:
            layers_dict = self.get_layers()
            oct_to_layer: Dict[str, str] = {}
            for lid, ldata in layers_dict.items():
                for oct_id in ldata.get('octahedra', []):
                    oct_to_layer[oct_id] = str(lid)
            for slab_id, oct_list in self._stacks_info.get('slabs', {}).items():
                key = str(slab_id)
                z_ranges = self._stacks_info.get('slab_z_ranges', {})
                layer_ids = sorted(
                    {oct_to_layer[o] for o in oct_list if o in oct_to_layer},
                    key=lambda x: int(x) if x.isdigit() else 0,
                )
                slabs[key] = {
                    'layer_ids': layer_ids,
                    'octahedra': oct_list,
                    'octahedra_count': len(oct_list),
                    'z_range': z_ranges.get(slab_id),
                }

        return slabs

    @property
    def is_twister(self) -> bool:
        """True only when structure_type was declared as twister by the user/creator.

        Multi-slab geometry (n_slabs >= 2) is common for RP/DJ cells and does
        not imply a twister. Stack detection and stacking-registry APIs remain
        available whenever n_slabs >= 2 regardless of this flag.
        """
        self._ensure_analyzed()
        return self._structure_type == 'twister'

    @property
    def n_slabs(self) -> int:
        """Number of independent perovskite stacks (slabs)."""
        self._ensure_analyzed()
        structure_node = self._graph.graph.get('structure_node', 'structure_0')
        if structure_node in self._graph and 'n_slabs' in self._graph.nodes[structure_node]:
            return int(self._graph.nodes[structure_node]['n_slabs'])
        if self._stacks_info:
            return int(self._stacks_info.get('n_slabs', 1))
        return len(self.get_slabs()) or 1

    @property
    def slabs(self) -> "Slabs":
        """Slabs wrapper for stack-scoped analysis."""
        self._ensure_analyzed()
        if self._slabs is None:
            from .slabs_wrapper import Slabs
            self._slabs = Slabs(self)
        return self._slabs

    def get_stack_interface(self, slab_id1: str = '0', slab_id2: str = '1') -> Dict[str, Any]:
        """Return interface molecules/cations bridging two slabs.

        Parameters
        ----------
        slab_id1, slab_id2 : str
            Slab identifiers (order normalized internally).

        Returns
        -------
        dict
            ``bridging_nodes``, ``formulas``, ``atom_indices`` for interface species.
        """
        self._ensure_analyzed()
        pair = tuple(sorted((str(slab_id1), str(slab_id2)), key=lambda x: int(x) if x.isdigit() else 0))

        bridging_nodes: List[str] = []
        formulas: List[str] = []
        atom_indices: List[List[int]] = []

        for node, data in self._graph.nodes(data=True):
            if data.get('node_type') not in ('a_site', 'spacer'):
                continue
            bridges = data.get('bridges_slabs')
            if not bridges:
                continue
            bridge_set = {str(b) for b in bridges}
            if set(pair) != bridge_set and not set(pair).issubset(bridge_set):
                continue
            bridging_nodes.append(node)
            formulas.append(data.get('formula', ''))
            indices = []
            for neighbor in self._graph.neighbors(node):
                edge_data = self._graph.get_edge_data(node, neighbor)
                if edge_data and edge_data.get('edge_type') == 'contains':
                    nd = self._graph.nodes.get(neighbor, {})
                    if nd.get('node_type') == 'atom':
                        vasp_idx = nd.get('vasp_index')
                        if vasp_idx is not None:
                            indices.append(vasp_idx)
            atom_indices.append(indices)

        return {
            'slab_pair': pair,
            'bridging_nodes': bridging_nodes,
            'formulas': formulas,
            'atom_indices': atom_indices,
        }

    def get_stacking_registry(
        self,
        slab_id1: str = '0',
        slab_id2: str = '1',
        x_frac: float = 0.25,
        cation_frac: float = 0.5,
        x_symbols: Optional[List[str]] = None,
        cation_symbols: Optional[List[str]] = None,
    ):
        """Compute inter-slab stacking registry metrics between two slabs.

        Available for any structure with 2+ independent stacks (slabs),
        including RP, DJ, and user-declared twisters. Requires a
        z-discontinuity between slabs; not gated on ``is_twister``.
        """
        self._ensure_analyzed()
        if self.n_slabs < 2:
            raise ValueError(
                "get_stacking_registry() requires 2+ independent stacks (slabs); "
                "available for any structure_type (bulk/dj/rp/twister) with a "
                "z-discontinuity between slabs."
            )
        from ..twister_processing.stacking_analysis import analyze_stack_interface
        return analyze_stack_interface(
            self,
            slab_id1=slab_id1,
            slab_id2=slab_id2,
            x_frac=x_frac,
            cation_frac=cation_frac,
            x_symbols=x_symbols,
            cation_symbols=cation_symbols,
        )

    def plot_stacking_heatmap(
        self,
        registry=None,
        output_path: Optional[str] = None,
        show_atoms: bool = True,
        grid_pts: int = 300,
        smoothing: Optional[float] = None,
        colorbar_label: Optional[str] = None,
    ) -> None:
        """Plot stacking ratio heatmap for a registry result."""
        self._ensure_analyzed()
        from ..twister_processing.stacking_plots import plot_stacking_heatmap as _plot
        if registry is None:
            registry = self.get_stacking_registry()
        _plot(
            registry,
            output_path=output_path,
            show_atoms=show_atoms,
            grid_pts=grid_pts,
            smoothing=smoothing,
            colorbar_label=colorbar_label,
        )

    def get_spacers(self) -> List[Atoms]:
        """
        Get identified spacer molecules extracted from graph.

        Returns
        -------
        list of ase.Atoms
            Each Atoms object represents a spacer molecule.
            The 'info' dict contains:
            - 'original_indices': atom indices in the original structure
            - 'spacer_formula': chemical formula
        """
        self._ensure_analyzed()
        return self._get_spacers_internal()

    def get_a_sites(self) -> List[Dict]:
        """
        Get A-site cation information extracted from graph.

        Returns
        -------
        list of dict
            Each dict contains:
            - 'atom_index': first atom index in molecule
            - 'symbol': element symbol or formula
            - 'position': cartesian coordinates of first atom
            - 'is_molecular': True if part of molecular A-site
            - 'molecule_atoms': indices of all atoms in molecular A-site
            - 'formula': chemical formula
        """
        self._ensure_analyzed()
        a_sites = []
        atom_positions = self.cell.get_positions()

        # Query A_Site nodes
        for node, data in self._graph.nodes(data=True):
            if data.get('node_type') == 'a_site':
                # Get atoms in this molecule via CONTAINS edges
                atom_indices = []
                for neighbor in self._graph.neighbors(node):
                    edge_data = self._graph.get_edge_data(node, neighbor)
                    if edge_data and edge_data.get('edge_type') == 'contains':
                        neighbor_data = self._graph.nodes.get(neighbor, {})
                        if neighbor_data.get('node_type') == 'atom':
                            atom_idx = neighbor_data.get('vasp_index')
                            if atom_idx is not None:
                                atom_indices.append(atom_idx)
                
                if atom_indices:
                    first_atom_idx = atom_indices[0]
                    formula = data.get('formula', self.cell[first_atom_idx].symbol)
                    a_sites.append({
                        'atom_index': first_atom_idx,
                        'symbol': formula,
                        'position': atom_positions[first_atom_idx].tolist(),
                        'is_molecular': len(atom_indices) > 1,
                        'molecule_atoms': atom_indices,
                        'formula': formula,
                    })

        return a_sites

    def get_cavities(self, a_site_weights=None, spacer_weights=None, max_validation_iterations=10) -> 'CavityCollection':
        """
        Get all detected cavities in the structure as a CavityCollection.

        A cavity is the cuboctahedral cage formed by corner-sharing octahedra
        that typically contains an A-site cation in perovskite structures.

        Parameters
        ----------
        a_site_weights : ASiteWeights, optional
            Weight configuration for A-site X atom selection.
            If None, uses default weights (0.7 for dist_anchor, 0.3 for z_diff).
        spacer_weights : SpacerWeights, optional
            Weight configuration for spacer terminal X atom selection.
            If None, uses default weights (0.4 for dist_anchor, 0.3 for dist_b_center, 0.3 for z_diff).
        max_validation_iterations : int, optional
            Maximum number of iterations for B-X connectivity validation and repair in cavity_tracing.
            Default 10. Increase (e.g. 30) for structures that need more attempts to fix malformed cavities.

        Returns
        -------
        CavityCollection
            Collection of Cavity objects with convenient methods for filtering and bulk operations.

            Each Cavity object has attributes:
            - `id`: cavity identifier (e.g., 'cavity_0')
            - `b_atom_indices`: B-site atom indices forming the cavity
            - `x_atom_indices`: X-site (halide) atom indices
            - `a_site_indices`: A-site (cation) atom indices
            - `center_position`: 3D coordinates of cavity center
            - `contains_a_site`: bool, True if cavity contains A-site
            - `pbc_coordinates`: dict of PBC-unwrapped positions
            - `subgraph`: NetworkX subgraph with detailed cavity structure
            - `hull_data`: Cached convex hull data if computed

            Methods:
            - `to_atoms()`: Convert all cavities to list of ASE Atoms objects
            - `filter(**kwargs)`: Filter cavities by attributes

        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.vasp")
        >>> analyzer.analyze()
        >>> cavities = analyzer.get_cavities()
        >>> print(f"Found {len(cavities)} cavities")
        >>>
        >>> # Use custom weights for cavity detection
        >>> from q2D_Materials.analyzer.cavities_processing import ASiteWeights, SpacerWeights
        >>> a_weights = ASiteWeights(w_dist_anchor=0.8, w_z_diff=0.2)
        >>> s_weights = SpacerWeights(w_dist_anchor=0.6, w_dist_b_center=0.2, w_z_diff=0.2)
        >>> cavities = analyzer.get_cavities(a_site_weights=a_weights, spacer_weights=s_weights)
        >>>
        >>> # Convert all to atoms and save
        >>> atoms_list = cavities.to_atoms()
        >>> for i, cavity_atoms in enumerate(atoms_list):
        ...     from ase.io import write
        ...     write(f'cavity_{i}.vasp', cavity_atoms)
        >>>
        >>> # Filter cavities with A-sites
        >>> with_a_sites = cavities.filter(contains_a_site=True)
        >>> print(f"Cavities with A-sites: {len(with_a_sites)}")
        >>>
        >>> # Work with individual cavities
        >>> for cavity in cavities:
        ...     print(f"{cavity.id}: {len(cavity.b_atom_indices)} B atoms")
        ...     if cavity.hull_data:
        ...         print(f"  Volume: {cavity.get_volume():.2f} Ų")
        """
        from ..cavities_processing.cavity_class import CavityCollection
        
        self._ensure_analyzed()
        
        # Detect cavities on demand using the cavity_tracing module
        from ..cavities_processing.cavity_tracing import detect_all_cavities
        
        try:
            # Detect cavities using the streamlined pipeline
            # Pass analyzer instance to use cached B/X atom data for better performance
            cavities = detect_all_cavities(
                self._graph,
                self._atom_positions,
                self._atom_symbols,
                self._cell,
                analyzer=self,
                a_site_weights=a_site_weights,
                spacer_weights=spacer_weights,
                max_validation_iterations=max_validation_iterations,
            )
            return CavityCollection(cavities)
        except Exception as e:
            # If cavity detection fails, return empty collection
            import sys
            print(f"WARNING: Cavity detection failed: {e}", file=sys.stderr)
            return CavityCollection([])

    def get_cavity_for_a_site(self, atom_idx: int) -> Optional[Dict]:
        """
        Get the cavity containing a specific A-site atom.

        Parameters
        ----------
        atom_idx : int
            Index of the A-site atom in the structure

        Returns
        -------
        dict or None
            Cavity information dict if found, None otherwise.
            See get_cavities() for dict structure.

        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.vasp")
        >>> analyzer.analyze()
        >>> a_sites = analyzer.get_a_sites()
        >>> for a_site in a_sites:
        ...     cavity = analyzer.get_cavity_for_a_site(a_site['atom_index'])
        ...     if cavity:
        ...         print(f"A-site at {a_site['atom_index']} is in {cavity['id']}")
        """
        self._ensure_analyzed()

        # Look for cavity containing this atom via graph edges
        atom_node = f'atom_{atom_idx}'
        if not self._graph.has_node(atom_node):
            return None

        for neighbor in self._graph.neighbors(atom_node):
            if neighbor.startswith('cavity_'):
                edge_data = self._graph[atom_node][neighbor]
                if edge_data.get('edge_type') == 'is_contained_in':
                    # Found the cavity, get its full data
                    data = self._graph.nodes[neighbor]
                    return {
                        'id': neighbor,
                        'upper_octahedra': data.get('upper_octahedra', []),
                        'lower_octahedra': data.get('lower_octahedra', []),
                        'equatorial_x_atoms': data.get('equatorial_x_atoms', []),
                        'axial_x_atoms': data.get('axial_x_atoms', []),
                        'chirality': data.get('chirality'),
                        'center_position': data.get('center_position'),
                        'layer_pair': data.get('layer_pair'),
                        'contains_a_site': data.get('contains_a_site', False),
                        'a_site_indices': data.get('a_site_indices', []),
                        'a_site_type': data.get('a_site_type'),
                        'is_pbc_wrapped': data.get('is_pbc_wrapped', False),
                        'is_single_layer': data.get('is_single_layer', False),
                        'hull_data': data.get('hull_data'),
                    }

        # Fallback: search all cavities for this atom index
        for node, data in self._graph.nodes(data=True):
            if data.get('node_type') == 'cavity':
                if atom_idx in data.get('a_site_indices', []):
                    return {
                        'id': node,
                        'upper_octahedra': data.get('upper_octahedra', []),
                        'lower_octahedra': data.get('lower_octahedra', []),
                        'equatorial_x_atoms': data.get('equatorial_x_atoms', []),
                        'axial_x_atoms': data.get('axial_x_atoms', []),
                        'chirality': data.get('chirality'),
                        'center_position': data.get('center_position'),
                        'layer_pair': data.get('layer_pair'),
                        'contains_a_site': data.get('contains_a_site', False),
                        'a_site_indices': data.get('a_site_indices', []),
                        'a_site_type': data.get('a_site_type'),
                        'is_pbc_wrapped': data.get('is_pbc_wrapped', False),
                        'is_single_layer': data.get('is_single_layer', False),
                        'hull_data': data.get('hull_data'),
                    }

        return None

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
        octahedra: Optional[List[str]] = None,
        layer: Optional[str] = None,
        layers: Optional[List[str]] = None,
        group_by: Optional[str] = None,
    ) -> Dict[str, Union[np.ndarray, float, None, Dict]]:
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
        octahedra : list of str, optional
            Specific octahedra IDs to include (e.g., ['octahedron_0', 'octahedron_1'])
        layer : str, optional
            Single layer ID to filter by (e.g., '0')
        layers : list of str, optional
            Multiple layer IDs to filter by (e.g., ['0', '1'])
        group_by : str, optional
            Group results by 'layer' or None for global

        Returns
        -------
        dict
            If group_by is None:
                Dictionary with keys:
                - 'bxb_angles': numpy array of B-X-B angles in degrees
                - 'bxb_mean': Mean B-X-B angle
                - 'bxb_std': Standard deviation of B-X-B angles
                - 'bxbp_angles': numpy array of B-X-Bp angles (if include_bp=True)
                - 'bxbp_mean': Mean B-X-Bp angle (if include_bp=True)
            
            If group_by == 'layer':
                Dictionary with keys:
                - 'global': dict with global statistics (bxb_angles, bxb_mean, bxb_std, etc.)
                - layer IDs (e.g., '0', '1'): dict with per-layer statistics
                - 'interlayer': dict with interlayer angle statistics (if any)
        """
        self._ensure_analyzed()
        result_tuple = _calculate_bxb_angles(
            self, bxb_scale, supercell, include_bp,
            octahedra=octahedra, layer=layer, layers=layers, group_by=group_by
        )

        # Helper function to calculate deviations from ideal 180°
        def calc_deviation_stats(angles):
            """Calculate mean and std of |180 - angle| deviations."""
            if angles is None or len(angles) == 0:
                return None, None
            deviations = np.abs(180.0 - angles)
            return float(np.mean(deviations)), float(np.std(deviations))

        # Unpack tuple result: (dict/data, bxbp_angles)
        # When group_by='layer', result_tuple is a dict; otherwise it's a tuple
        if isinstance(result_tuple, tuple):
            result_data, _ = result_tuple
        else:
            result_data = result_tuple

        if result_data is None:
            if group_by == 'layer':
                return {
                    'global': {
                        'bxb_angles': None,
                        'bxb_mean': None,
                        'bxb_std': None,
                    }
                }
            return {
                'bxb_angles': None,
                'bxb_equatorial_angles': None,
                'bxb_interlayer_angles': None,
                'bxb_deviation_mean': None,
                'bxb_deviation_std': None,
                'bxb_equatorial_deviation_mean': None,
                'bxb_equatorial_deviation_std': None,
                'bxb_interlayer_deviation_mean': None,
                'bxb_interlayer_deviation_std': None,
            }

        # Handle grouped results
        if group_by == 'layer':
            result = {}
            
            # Process global data
            global_bxb, global_bxbp = result_data.get('global', (None, None))
            result['global'] = {
                "bxb_angles": global_bxb,
                "bxb_mean": float(np.mean(global_bxb)) if global_bxb is not None and len(global_bxb) > 0 else None,
                "bxb_std": float(np.std(global_bxb)) if global_bxb is not None and len(global_bxb) > 0 else None,
            }
            if include_bp:
                result['global']["bxbp_angles"] = global_bxbp
                result['global']["bxbp_mean"] = (
                    float(np.mean(global_bxbp)) if global_bxbp is not None and len(global_bxbp) > 0 else None
                )
            
            # Process per-layer data
            for key, value in result_data.items():
                if key == 'global':
                    continue
                if isinstance(value, dict):
                    layer_bxb = value.get('bxb_angles')
                    layer_bxbp = value.get('bxbp_angles')
                    result[key] = {
                        "bxb_angles": layer_bxb,
                        "bxb_mean": float(np.mean(layer_bxb)) if layer_bxb is not None and len(layer_bxb) > 0 else None,
                        "bxb_std": float(np.std(layer_bxb)) if layer_bxb is not None and len(layer_bxb) > 0 else None,
                    }
                    if include_bp:
                        result[key]["bxbp_angles"] = layer_bxbp
                        result[key]["bxbp_mean"] = (
                            float(np.mean(layer_bxbp)) if layer_bxbp is not None and len(layer_bxbp) > 0 else None
                        )
            
            return result
        else:
            # No grouping, return decomposed format with deviation statistics
            bxb_all = result_data.get('bxb_angles')
            bxb_equatorial = result_data.get('bxb_equatorial_angles')
            bxb_interlayer = result_data.get('bxb_interlayer_angles')
            
            # Calculate deviation statistics for each category
            # Note: Terminal X atoms cannot form B-X-B angles (they belong to only 1 octahedron)
            dev_all_mean, dev_all_std = calc_deviation_stats(bxb_all)
            dev_equatorial_mean, dev_equatorial_std = calc_deviation_stats(bxb_equatorial)
            dev_interlayer_mean, dev_interlayer_std = calc_deviation_stats(bxb_interlayer)
            
            result = {
                # Raw angles (local features)
                "bxb_angles": bxb_all,
                "bxb_mean": (
                    float(np.mean(bxb_all))
                    if bxb_all is not None and len(bxb_all) > 0
                    else None
                ),
                "bxb_std": (
                    float(np.std(bxb_all))
                    if bxb_all is not None and len(bxb_all) > 0
                    else None
                ),
                "bxb_equatorial_angles": bxb_equatorial,
                "bxb_interlayer_angles": bxb_interlayer,
                
                # Global features (deviation from ideal 180°)
                "bxb_deviation_mean": dev_all_mean,
                "bxb_deviation_std": dev_all_std,
                "bxb_equatorial_deviation_mean": dev_equatorial_mean,
                "bxb_equatorial_deviation_std": dev_equatorial_std,
                "bxb_interlayer_deviation_mean": dev_interlayer_mean,
                "bxb_interlayer_deviation_std": dev_interlayer_std,
            }

            # Add B-X-Bp angles if requested (for future use)
            if include_bp:
                result["bxbp_angles"] = result_data.get('bxbp_angles')

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
        ...          .neighbors(edge_type='contains')
        ...          .to_list())
        """
        self._ensure_analyzed()
        return CharacterizationQuery(self)

    def inter_plane_distance(self, debug: bool = False) -> Dict[str, Any]:
        """
        Calculate the distance between terminal atom planes.
        
        Terminal atoms define the boundaries between the spacer molecule region
        and the inorganic slab region. Handles both multi-layer and single-layer cases.
        
        **Single-layer case:** When both top and bottom terminal atoms are in the same layer,
        slab_thickness is calculated as the Z-span of terminal atoms, and interplane_distance
        is derived as cell_z - slab_thickness.
        
        **Multi-layer case:** When terminal atoms are in different layers, fits planes to
        each group and calculates the distance between them.
        
        Parameters
        ----------
        debug : bool, optional
            If True, print detailed debugging information
        
        Returns
        -------
        dict
            {
                'interplane_distance': float (Å),
                'slab_thickness': float (Å) or None,
                'top_centroid': list or None,
                'bottom_centroid': list or None,
                'top_atom_count': int,
                'bottom_atom_count': int,
                'molecule_mean_z': float or None,
                'molecule_centroid': list or None,
                'top_z_range': [min_z, max_z],
                'bottom_z_range': [min_z, max_z],
                'terminal_z_range': [min_z, max_z],
                'error': str or None,
            }
        
        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.vasp")
        >>> analyzer.analyze()
        >>> result = analyzer.inter_plane_distance()
        >>> if result.get('error'):
        ...     print(f"Error: {result['error']}")
        ... else:
        ...     print(f"Interplane distance: {result['interplane_distance']:.3f} Å")
        ...     print(f"Slab thickness: {result.get('slab_thickness', 'N/A'):.3f} Å")
        """
        self._ensure_analyzed()
        from ..characterization.structure_features import calculate_global_interplane_distance
        try:
            return calculate_global_interplane_distance(self._graph, debug=debug)
        except Exception as e:
            # Return error information instead of raising (catch all exceptions)
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            if debug:
                print(f"  ERROR in inter_plane_distance: {error_msg}")
                traceback.print_exc()
            return {
                'interplane_distance': np.nan,
                'slab_thickness': np.nan,
                'top_centroid': None,
                'bottom_centroid': None,
                'top_atom_count': 0,
                'bottom_atom_count': 0,
                'molecule_mean_z': None,
                'molecule_centroid': None,
                'terminal_z_range': [None, None],
                'top_z_range': [None, None],
                'bottom_z_range': [None, None],
                'error': error_msg,
            }

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

    def get_mean_bond_lengths(
        self,
        octahedra: Optional[List[str]] = None,
        layer: Optional[str] = None,
        layers: Optional[List[str]] = None,
        group_by: Optional[str] = None,
    ) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Get mean B-X bond lengths separated by geometry (axial vs equatorial).

        Axial bonds are along the c-axis (to terminal or interlayer X atoms),
        while equatorial bonds are in the ab-plane. These have significantly
        different lengths in layered perovskites due to structural relaxation.

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
        dict
            If group_by is None:
                Dictionary containing:
                - 'mean_bond_length': Mean of all B-X bond lengths (Å)
                - 'mean_bond_length_axial': Mean of axial B-X bond lengths (Å)
                - 'mean_bond_length_equatorial': Mean of equatorial B-X bond lengths (Å)

            If group_by == 'layer':
                Dictionary with layer IDs as keys, each containing the above metrics,
                plus 'global' key with overall metrics

        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.cif")
        >>> analyzer.analyze()
        >>>
        >>> # Global mean bond lengths
        >>> bond_lengths = analyzer.get_mean_bond_lengths()
        >>> print(f"All B-X: {bond_lengths['mean_bond_length']:.3f} Å")
        >>> print(f"Axial B-X: {bond_lengths['mean_bond_length_axial']:.3f} Å")
        >>> print(f"Equatorial B-X: {bond_lengths['mean_bond_length_equatorial']:.3f} Å")
        >>>
        >>> # Per-layer analysis
        >>> by_layer = analyzer.get_mean_bond_lengths(group_by='layer')
        >>> for layer_id, data in by_layer.items():
        ...     print(f"Layer {layer_id}:")
        ...     print(f"  Axial: {data['mean_bond_length_axial']:.3f} Å")
        ...     print(f"  Equatorial: {data['mean_bond_length_equatorial']:.3f} Å")
        """
        from ..characterization.distortions import _compute_octahedral_distortions
        self._ensure_analyzed()
        result = _compute_octahedral_distortions(
            self, octahedra=octahedra, layer=layer, layers=layers, group_by=group_by
        )

        def extract_bond_lengths(data):
            return {
                'mean_bond_length': data.get('mean_bond_length'),
                'mean_bond_length_axial': data.get('mean_bond_length_axial'),
                'mean_bond_length_equatorial': data.get('mean_bond_length_equatorial'),
            }

        if group_by == 'layer':
            return {k: extract_bond_lengths(v) for k, v in result.items()}
        return extract_bond_lengths(result)

    def get_octahedral_tilts(
        self,
        octahedra: Optional[List[str]] = None,
        bond_selection_threshold: float = 0.1,
    ):
        """
        Get octahedral tilt data (Euler angles and rotation matrices).

        Computes Euler angles via Kabsch algorithm + scipy Rotation decomposition.
        Uses XYZ extrinsic convention matching builder: R_total = R_z @ R_y @ R_x.

        Parameters
        ----------
        octahedra : list of str, optional
            Specific octahedra IDs. If None, computes all octahedra.
        bond_selection_threshold : float, default=0.1
            Minimum projection for bond selection (ensures sign consistency)

        Returns
        -------
        OctahedralTiltData
            Container with euler_angles (N_oct, 3), rotation_matrices (N_oct, 3, 3),
            octahedron_ids, b_atom_indices, and reference_axes

        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.vasp")
        >>> analyzer.analyze()
        >>> tilt_data = analyzer.get_octahedral_tilts()
        >>> print(tilt_data.euler_angles)  # (N_oct, 3) array
        """
        from ..octahedral_processing.tilt_calculations import compute_octahedral_tilts
        self._ensure_analyzed()
        return compute_octahedral_tilts(self, octahedra, bond_selection_threshold)

    def compute_mean_tilt_profile(
        self,
        octahedra: Optional[List[str]] = None,
        layer: Optional[str] = None,
        layers: Optional[List[str]] = None,
        group_by: Optional[str] = None,
    ) -> Union[float, Dict[str, float]]:
        """
        Compute mean tilt magnitude profile.

        Formula: μ = (1/N) Σ ||T_i|| where ||T|| = sqrt(α² + β² + γ²)

        Measures the average tilting magnitude, optionally grouped by layer/slab.
        Most useful for 2D layered structures (DJ, RP) to quantify surface
        relaxation effects and depth-dependent distortions.

        Parameters
        ----------
        octahedra : list of str, optional
            Specific octahedra IDs to include (e.g., ['octahedron_0', 'octahedron_1'])
        layer : str, optional
            Single layer ID to filter by (e.g., '0')
        layers : list of str, optional
            Multiple layer IDs to filter by (e.g., ['0', '1'])
        group_by : str, optional
            If 'layer' or 'slab', returns dict with per-layer/slab results plus 'global' key
            If None, returns single global value

        Returns
        -------
        float or dict
            If group_by is None: Mean tilt magnitude (float, degrees)
            If group_by == 'layer': Dict with layer IDs as keys plus 'global' key
            If group_by == 'slab': Dict with slab IDs as keys plus 'global' key

        Examples
        --------
        >>> analyzer = q2D_analyzer("rp_structure.vasp")
        >>> analyzer.analyze()
        >>>
        >>> # Global mean tilt
        >>> mu_global = analyzer.compute_mean_tilt_profile()
        >>> print(f"Global mean tilt: {mu_global:.2f}°")
        >>>
        >>> # Per-layer profile
        >>> profile = analyzer.compute_mean_tilt_profile(group_by='layer')
        >>> for layer_id, mu in profile.items():
        ...     print(f"Layer {layer_id}: μ = {mu:.2f}°")
        >>>
        >>> # Single layer
        >>> mu_layer0 = analyzer.compute_mean_tilt_profile(layer='0')
        >>> print(f"Layer 0 mean tilt: {mu_layer0:.2f}°")
        """
        from ..octahedral_processing.tilting_properties import compute_mean_tilt_profile
        self._ensure_analyzed()
        return compute_mean_tilt_profile(
            self, octahedra=octahedra, layer=layer, layers=layers, group_by=group_by
        )

    def compute_gearing_correlation(
        self,
        octahedra: Optional[List[str]] = None,
        layer: Optional[str] = None,
        layers: Optional[List[str]] = None,
        group_by: Optional[str] = None,
        neighbor_criterion: str = 'shared_x',
    ) -> Union[Dict[Tuple[str, str], float], Dict[str, Dict[Tuple[str, str], float]]]:
        """
        Compute gearing correlation between neighboring octahedra.

        Formula: χ(i,j) = (R_i · R_j) / (||R_i|| ||R_j||)
        where R = rotation vector (axis-angle representation)

        Measures mechanical coordination between corner-sharing octahedra.
        Quantifies cooperative vs. counter-rotating tilting patterns.

        Parameters
        ----------
        octahedra : list of str, optional
            Specific octahedra IDs to include (e.g., ['octahedron_0', 'octahedron_1'])
        layer : str, optional
            Single layer ID to filter by (e.g., '0')
        layers : list of str, optional
            Multiple layer IDs to filter by (e.g., ['0', '1'])
        group_by : str, optional
            If 'layer' or 'slab', returns dict with per-layer/slab results plus 'global' key
            If None, returns single dict of correlations
        neighbor_criterion : str, default='shared_x'
            'shared_x' for corner-sharing neighbors

        Returns
        -------
        dict or dict of dicts
            If group_by is None: {(oct_i, oct_j): χ} where χ ∈ [-1, 1]
            If group_by == 'layer' or 'slab': {layer_id: {(oct_i, oct_j): χ}, 'global': {...}}

            χ interpretation:
            - χ ≈ +1: Cooperative rotation (like meshing gears)
            - χ ≈ -1: Counter-rotation
            - χ ≈ 0: Orthogonal rotations

        Examples
        --------
        >>> analyzer = q2D_analyzer("tilted_bulk.vasp")
        >>> analyzer.analyze()
        >>>
        >>> # Global correlations
        >>> gearing = analyzer.compute_gearing_correlation()
        >>> for (oct_i, oct_j), chi in gearing.items():
        ...     if chi > 0.5:
        ...         print(f"{oct_i} ↔ {oct_j}: Cooperative (χ = {chi:.3f})")
        >>>
        >>> # Per-layer grouped correlations
        >>> gearing_by_layer = analyzer.compute_gearing_correlation(group_by='layer')
        >>> for layer_id, pairs in gearing_by_layer.items():
        ...     if layer_id != 'global':
        ...         avg_chi = np.mean(list(pairs.values()))
        ...         print(f"Layer {layer_id} avg gearing: {avg_chi:.3f}")
        >>>
        >>> # Single layer
        >>> layer0_gearing = analyzer.compute_gearing_correlation(layer='0')
        >>> print(f"Layer 0 has {len(layer0_gearing)} octahedral pairs")
        """
        from ..octahedral_processing.tilting_properties import compute_gearing_correlation
        self._ensure_analyzed()
        return compute_gearing_correlation(
            self,
            octahedra=octahedra,
            layer=layer,
            layers=layers,
            group_by=group_by,
            neighbor_criterion=neighbor_criterion
        )

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
        and layer assignment. Bond lengths are separated into axial (along c-axis)
        and equatorial (in ab-plane) categories, as these have significantly
        different lengths in layered perovskites.

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
            - 'bond_lengths': Array of all B-X bond lengths (6 values)
            - 'bond_lengths_axial': Array of axial B-X bond lengths (2 values)
            - 'bond_lengths_equatorial': Array of equatorial B-X bond lengths (4 values)
            - 'bond_angles': Array of X-B-X angles
            - 'mean_bond_length': Mean of all B-X bond lengths
            - 'mean_bond_length_axial': Mean of axial B-X bond lengths
            - 'mean_bond_length_equatorial': Mean of equatorial B-X bond lengths
            - 'mean_angle': Mean X-B-X angle
            - 'volume': Octahedral volume in ų
            - 'layer': Layer ID this octahedron belongs to
            - 'central_atom_index': Index of B-site atom
            - 'central_atom_symbol': Element symbol of B-site
            - 'geometry': Dict mapping X atom index to geometry type

        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.cif")
        >>> analyzer.analyze()
        >>>
        >>> # Get all octahedra distortions
        >>> oct_data = analyzer.get_octahedral_distortions()
        >>> for oct_id, metrics in oct_data.items():
        ...     print(f"{oct_id} (Layer {metrics['layer']}):")
        ...     print(f"  Axial B-X: {metrics['mean_bond_length_axial']:.3f} Å")
        ...     print(f"  Equatorial B-X: {metrics['mean_bond_length_equatorial']:.3f} Å")
        >>>
        >>> # Filter by layer
        >>> layer0_oct = analyzer.get_octahedral_distortions(layer='0')
        >>> print(f"Layer 0 has {len(layer0_oct)} octahedra")
        >>>
        >>> # Access specific octahedron
        >>> oct0 = analyzer.get_octahedral_distortions(octahedra=['octahedron_0'])
        >>> print(f"Axial bonds: {oct0['octahedron_0']['bond_lengths_axial']}")
        >>> print(f"Equatorial bonds: {oct0['octahedron_0']['bond_lengths_equatorial']}")
        """
        from ..characterization.distortions import _get_octahedral_distortions_detailed
        self._ensure_analyzed()
        return _get_octahedral_distortions_detailed(
            self, octahedra=octahedra, layer=layer, layers=layers
        )
    
    def get_octahedral_volumes(
        self,
        mode: str = 'global',
        octahedra: Optional[List[str]] = None,
    ) -> Union[float, Dict[str, float]]:
        """
        Calculate octahedral volumes using convex hull of X atoms with PBC-aware positioning.
        
        For each octahedron, this method:
        1. Gets the 6 X atoms bonded to the B atom
        2. Uses PBC-aware distance calculations to ensure X atoms are in the minimum image
        3. Calculates the convex hull volume of the 6 X atoms
        
        This ensures that all X atoms are in the closest periodic image relative to the B atom,
        preventing errors from PBC wrapping that could distort the volume calculation.
        
        Parameters
        ----------
        mode : str, default='global'
            Calculation mode:
            - 'global': Return mean volume for all octahedra (single float)
            - 'local': Return volume per octahedron (dict mapping octahedron_id -> volume)
        octahedra : list of str, optional
            Specific octahedra IDs to include (e.g., ['octahedron_0', 'octahedron_1'])
            If None, computes all octahedra.
        
        Returns
        -------
        float or dict
            - If mode='global': Mean octahedral volume in Å³ (float)
            - If mode='local': Dict mapping octahedron_id -> volume in Å³
        
        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.cif")
        >>> analyzer.analyze()
        >>> 
        >>> # Global mode: mean volume
        >>> mean_volume = analyzer.get_octahedral_volumes(mode='global')
        >>> print(f"Mean octahedral volume: {mean_volume:.2f} Å³")
        >>> 
        >>> # Local mode: per-octahedron volumes
        >>> volumes_dict = analyzer.get_octahedral_volumes(mode='local')
        >>> for oct_id, vol in volumes_dict.items():
        ...     print(f"{oct_id}: {vol:.2f} Å³")
        """
        self._ensure_analyzed()
        detailed = self.get_octahedral_distortions(octahedra=octahedra)
        volumes_dict = {
            oct_id: (
                float(metrics.get("volume", np.nan))
                if metrics.get("status") == "valid"
                else np.nan
            )
            for oct_id, metrics in detailed.items()
        }

        if mode == 'global':
            valid_volumes = np.asarray(
                [
                    value
                    for value in volumes_dict.values()
                    if np.isfinite(value) and value > 0
                ],
                dtype=float,
            )
            if len(valid_volumes) == 0:
                return np.nan
            return float(np.mean(valid_volumes))
        elif mode == 'local':
            return volumes_dict
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'global' or 'local'.")

    def get_octahedral_validity(
        self,
        octahedra: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """Return explicit validity states for reconstructed octahedra."""
        detailed = self.get_octahedral_distortions(octahedra=octahedra)
        return {
            oct_id: {
                "status": str(metrics.get("status", "failed")),
                "reason": str(metrics.get("status_reason", "unknown")),
            }
            for oct_id, metrics in detailed.items()
        }
    
    def get_xeq_xeq_and_xax_xax_distances(
        self,
        mode: str = 'global',
    ) -> Union[Tuple[float, float], Tuple[Dict[str, float], Dict[str, float]]]:
        """
        Calculate Xeq-Xeq and Xax-Xax distances for octahedra using graph structure.
        
        For each octahedron:
        1. Gets B atom from octahedron node (via CONTAINS edge)
        2. Gets X atoms from B atom (via BONDED_TO edges)
        3. Classifies X atoms as equatorial/axial from node attributes
        4. Calculates minimum distance between any two equatorial X atoms (Xeq-Xeq)
        5. Calculates distance between the two axial X atoms (Xax-Xax)
        
        Uses PBC-aware distance calculations to ensure accurate measurements.
        
        Parameters
        ----------
        mode : str, default='global'
            Calculation mode:
            - 'global': Return mean distances for all octahedra (tuple of floats)
            - 'local': Return distances per octahedron (tuple of dicts)
        
        Returns
        -------
        tuple
            - If mode='global': (mean_xeq_xeq, mean_xax_xax) in Å (tuple of floats)
            - If mode='local': (xeq_xeq_dict, xax_xax_dict) (tuple of dicts)
        
        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.cif")
        >>> analyzer.analyze()
        >>> 
        >>> # Global mode: mean distances
        >>> mean_xeq_xeq, mean_xax_xax = analyzer.get_xeq_xeq_and_xax_xax_distances(mode='global')
        >>> print(f"Mean Xeq-Xeq: {mean_xeq_xeq:.2f} Å, Mean Xax-Xax: {mean_xax_xax:.2f} Å")
        >>> 
        >>> # Local mode: per-octahedron distances
        >>> xeq_dict, xax_dict = analyzer.get_xeq_xeq_and_xax_xax_distances(mode='local')
        >>> for oct_id in xeq_dict.keys():
        ...     print(f"{oct_id}: Xeq-Xeq={xeq_dict[oct_id]:.2f} Å, Xax-Xax={xax_dict[oct_id]:.2f} Å")
        """
        from ..octahedral_processing.octahedral_analysis import calculate_xeq_xeq_and_xax_xax_distances
        from pymatgen.io.ase import AseAtomsAdaptor
        
        self._ensure_analyzed()
        
        # Convert ASE Atoms to pymatgen Structure
        struct = AseAtomsAdaptor.get_structure(self.cell)
        
        # Get graph
        graph = self.get_graph()
        
        # Calculate distances using simple graph-based approach
        return calculate_xeq_xeq_and_xax_xax_distances(struct, graph, mode=mode)
    
    def get_xbx_angles_by_layer_and_geometry(
        self,
        octahedra: Optional[List[str]] = None,
        layer: Optional[str] = None,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get X-B-X (XBX) angles grouped by layer and geometry (equatorial vs axial/interlayer).
        
        Returns XBX angles (angles between two X atoms and the central B atom within an octahedron)
        separated by layer and by X atom geometry. Ideal angles are 90° (adjacent X) or 180° (opposite X).
        
        Parameters
        ----------
        octahedra : list of str, optional
            Specific octahedra IDs to include
        layer : str, optional
            Single layer ID to filter by
        layers : list of str, optional
            Multiple layer IDs to filter by
        
        Returns
        -------
        dict
            Dictionary with structure:
            {
                'Layer_0': {
                    'equatorial': np.ndarray of XBX angles (both X atoms are equatorial),
                    'axial': np.ndarray of XBX angles (at least one X atom is axial/interlayer)
                },
                'Layer_1': {
                    'equatorial': np.ndarray,
                    'axial': np.ndarray
                },
                ...
            }
        """
        from ...utils.geometry.pbc_distances import find_nearest_image_positions
        from ..characterization.distortions import _get_octahedral_distortions_detailed
        
        self._ensure_analyzed()
        
        # Get all octahedral distortions (contains XBX angles in 'bond_angles')
        all_octahedra_distortions = _get_octahedral_distortions_detailed(
            self, octahedra=octahedra, layer=layer, layers=layers
        )
        
        if not all_octahedra_distortions:
            return {}
        
        # Get graph and positions to determine X atom geometry
        graph = self.get_graph()
        atom_positions = self.cell.get_positions()
        cell = np.array(self.cell.get_cell())
        
        # Get all X atom positions for PBC calculations
        b_x_data = self.get_b_x_atoms()
        all_x_indices = b_x_data['x_indices']
        all_x_positions = b_x_data['x_positions']
        
        # Result structure: layer_id -> {'equatorial': [...], 'axial': [...]}
        result = {}
        
        # Process each octahedron
        for oct_id, oct_data in all_octahedra_distortions.items():
            central_idx = oct_data.get('central_atom_index')
            if central_idx is None:
                continue
            
            # Use already-calculated bond_angles from get_octahedral_distortions()
            bond_angles = oct_data.get('bond_angles')
            if bond_angles is None or len(bond_angles) == 0:
                continue
            
            layer_id = oct_data.get('layer', 'unknown')
            layer_key = f'Layer_{layer_id}'
            
            # Initialize layer dict if needed
            if layer_key not in result:
                result[layer_key] = {'equatorial': [], 'axial': []}
            
            # Reconstruct X atom indices using the same method as distortions.py
            central_pos = atom_positions[central_idx]
            
            # Find 6 nearest X atoms using PBC (same as in distortions.py)
            nearest_x_indices, nearest_x_positions, nearest_distances, image_labels = find_nearest_image_positions(
                reference_position=central_pos,
                candidate_positions=all_x_positions,
                candidate_indices=all_x_indices,
                cell=cell,
                n_neighbors=6,
                pbc=True,
            )
            
            # Check geometry of each X atom from graph
            x_geometry = {}  # x_idx -> 'equatorial' or 'axial' or 'interlayer'
            for x_idx in nearest_x_indices:
                atom_node = f"atom_{x_idx}"
                node_data = graph.nodes.get(atom_node, {})
                
                # Check if interlayer (bridging between layers)
                if node_data.get('is_interlayer', False):
                    x_geometry[x_idx] = 'interlayer'
                # Check if axial
                elif node_data.get('is_axial', False):
                    x_geometry[x_idx] = 'axial'
                # Otherwise equatorial
                else:
                    x_geometry[x_idx] = 'equatorial'
            
            # Classify bond_angles by geometry
            # The order matches distortions.py: for i in range(len(x_positions)):
            #     for j in range(i + 1, len(x_positions)):
            bond_angles_array = np.array(bond_angles) if hasattr(bond_angles, '__iter__') and not isinstance(bond_angles, str) else bond_angles
            
            angle_idx = 0
            for i in range(len(nearest_x_indices)):
                for j in range(i + 1, len(nearest_x_indices)):
                    if angle_idx >= len(bond_angles_array):
                        break
                    
                    x_i_idx = nearest_x_indices[i]
                    x_j_idx = nearest_x_indices[j]
                    angle = bond_angles_array[angle_idx]
                    
                    # Classify angle based on X atom geometry
                    geom_i = x_geometry.get(x_i_idx, 'equatorial')
                    geom_j = x_geometry.get(x_j_idx, 'equatorial')
                    
                    # Equatorial: both X atoms are equatorial
                    # Axial/Interlayer: at least one X atom is axial or interlayer
                    if geom_i == 'equatorial' and geom_j == 'equatorial':
                        result[layer_key]['equatorial'].append(angle)
                    else:
                        # At least one is axial or interlayer
                        result[layer_key]['axial'].append(angle)
                    
                    angle_idx += 1
        
        # Convert lists to numpy arrays
        for layer_key in result:
            result[layer_key]['equatorial'] = np.array(result[layer_key]['equatorial']) if result[layer_key]['equatorial'] else np.array([])
            result[layer_key]['axial'] = np.array(result[layer_key]['axial']) if result[layer_key]['axial'] else np.array([])
        
        return result

    def get_xbx_angles_by_layer_separated(
        self,
        octahedra: Optional[List[str]] = None,
        layer: Optional[str] = None,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get X-B-X (XBX) angles separated by angle type (90° vs 180°) and geometry.
        
        This method separates XBX angles into:
        - 90° type: angles < 135° (ideal = 90°)
        - 180° type: angles ≥ 135° (ideal = 180°)
        
        Further classified by X-atom geometry:
        - eq_eq: both X atoms are equatorial
        - eq_ax: one equatorial, one axial/interlayer
        - ax_ax: both axial/interlayer (only exists for 180° angles)
        
        Parameters
        ----------
        octahedra : list of str, optional
            Specific octahedra IDs to include
        layer : str, optional
            Single layer ID to filter by
        layers : list of str, optional
            Multiple layer IDs to filter by
        
        Returns
        -------
        dict
            Dictionary with structure:
            {
                'Layer_0': {
                    '90_eq_eq': np.ndarray,
                    '90_eq_ax': np.ndarray,
                    '180_eq_eq': np.ndarray,
                    '180_eq_ax': np.ndarray,
                    '180_ax_ax': np.ndarray
                },
                'Layer_1': {...},
                'global': {
                    '90': np.ndarray,  # all 90° angles
                    '180': np.ndarray  # all 180° angles
                }
            }
        """
        from ...utils.geometry.pbc_distances import find_nearest_image_positions
        from ..characterization.distortions import _get_octahedral_distortions_detailed
        
        self._ensure_analyzed()
        
        # Get all octahedral distortions (contains XBX angles in 'bond_angles')
        all_octahedra_distortions = _get_octahedral_distortions_detailed(
            self, octahedra=octahedra, layer=layer, layers=layers
        )
        
        if not all_octahedra_distortions:
            return {}
        
        # Get graph and positions to determine X atom geometry
        graph = self.get_graph()
        atom_positions = self.cell.get_positions()
        cell = np.array(self.cell.get_cell())
        
        # Get all X atom positions for PBC calculations
        b_x_data = self.get_b_x_atoms()
        all_x_indices = b_x_data['x_indices']
        all_x_positions = b_x_data['x_positions']
        
        # Result structure: layer_id -> {angle_type_geometry: [...]}
        result = {}
        
        # Global collectors
        global_90 = []
        global_180 = []
        
        # Process each octahedron
        for oct_id, oct_data in all_octahedra_distortions.items():
            central_idx = oct_data.get('central_atom_index')
            if central_idx is None:
                continue
            
            # Use already-calculated bond_angles from get_octahedral_distortions()
            bond_angles = oct_data.get('bond_angles')
            if bond_angles is None or len(bond_angles) == 0:
                continue
            
            layer_id = oct_data.get('layer', 'unknown')
            layer_key = f'Layer_{layer_id}'
            
            # Initialize layer dict if needed
            if layer_key not in result:
                result[layer_key] = {
                    '90_eq_eq': [],
                    '90_eq_ax': [],
                    '180_eq_eq': [],
                    '180_eq_ax': [],
                    '180_ax_ax': []
                }
            
            # Reconstruct X atom indices and positions (minimum-image)
            central_pos = atom_positions[central_idx]
            nearest_x_indices, nearest_x_positions, nearest_distances, image_labels = find_nearest_image_positions(
                reference_position=central_pos,
                candidate_positions=all_x_positions,
                candidate_indices=all_x_indices,
                cell=cell,
                n_neighbors=6,
                pbc=True,
            )
            nearest_x_positions = np.asarray(nearest_x_positions)

            # Bulk only (3D, no terminals): classify 180° by X-X pair vector alignment. Axial = X-X along Z (c), equatorial = X-X in XY.
            use_geometric_bulk = (getattr(self, '_structure_type', None) == 'bulk')
            c_hat = None
            if use_geometric_bulk:
                c_vec = np.array(cell[2], dtype=np.float64)
                c_norm = np.linalg.norm(c_vec)
                c_hat = c_vec / c_norm if c_norm >= 1e-10 else np.array([0.0, 0.0, 1.0])

            x_geometry = {}
            if not use_geometric_bulk:
                for x_idx in nearest_x_indices:
                    atom_node = f"atom_{x_idx}"
                    node_data = graph.nodes.get(atom_node, {})
                    if 'is_axial' in node_data:
                        x_geometry[x_idx] = 'axial'
                    elif 'is_equatorial' in node_data:
                        x_geometry[x_idx] = 'equatorial'
                    elif 'is_interlayer' in node_data:
                        x_geometry[x_idx] = 'axial'
                    else:
                        x_geometry[x_idx] = 'equatorial'

            bond_angles_array = np.array(bond_angles) if hasattr(bond_angles, '__iter__') and not isinstance(bond_angles, str) else bond_angles
            angle_idx = 0
            for i in range(len(nearest_x_indices)):
                for j in range(i + 1, len(nearest_x_indices)):
                    if angle_idx >= len(bond_angles_array):
                        break
                    angle = bond_angles_array[angle_idx]
                    x_i_idx = nearest_x_indices[i]
                    x_j_idx = nearest_x_indices[j]

                    if angle < 135.0:
                        angle_type = '90'
                        global_90.append(angle)
                        if use_geometric_bulk:
                            geom_class = 'eq_ax'
                        else:
                            geom_i = x_geometry.get(x_i_idx, 'equatorial')
                            geom_j = x_geometry.get(x_j_idx, 'equatorial')
                            if geom_i == 'equatorial' and geom_j == 'equatorial':
                                geom_class = 'eq_eq'
                            elif geom_i == 'axial' and geom_j == 'axial':
                                geom_class = 'ax_ax'
                            else:
                                geom_class = 'eq_ax'
                    else:
                        angle_type = '180'
                        global_180.append(angle)
                        if use_geometric_bulk and c_hat is not None:
                            vec_xx = nearest_x_positions[j] - nearest_x_positions[i]
                            vec_xx_norm = np.linalg.norm(vec_xx) + 1e-9
                            axis_xx = vec_xx / vec_xx_norm
                            if np.abs(np.dot(axis_xx, c_hat)) >= 0.7:
                                geom_class = 'ax_ax'
                            else:
                                geom_class = 'eq_eq'
                        else:
                            geom_i = x_geometry.get(x_i_idx, 'equatorial')
                            geom_j = x_geometry.get(x_j_idx, 'equatorial')
                            if geom_i == 'equatorial' and geom_j == 'equatorial':
                                geom_class = 'eq_eq'
                            elif geom_i == 'axial' and geom_j == 'axial':
                                geom_class = 'ax_ax'
                            else:
                                geom_class = 'eq_ax'

                    key = f'{angle_type}_{geom_class}'
                    if key in result[layer_key]:
                        result[layer_key][key].append(angle)
                    angle_idx += 1
        
        # Convert lists to numpy arrays
        for layer_key in result:
            for angle_key in result[layer_key]:
                result[layer_key][angle_key] = np.array(result[layer_key][angle_key]) if result[layer_key][angle_key] else np.array([])
        
        # Add global statistics
        result['global'] = {
            '90': np.array(global_90) if global_90 else np.array([]),
            '180': np.array(global_180) if global_180 else np.array([])
        }
        
        return result

    def get_xbx_angles_global_separated(
        self,
        octahedra: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get X-B-X (XBX) angles globally separated by equatorial/axial geometry.

        This method extracts ALL XBX angles from all octahedra and classifies them
        by geometry WITHOUT layer separation. This is useful for structures where
        layer-specific classification may fail (e.g., some Br-based structures).

        Classification:
        - For bulk structures (3D): Uses geometric classification based on X-X vector
          alignment with c-axis. If |dot(X-X_vector, c_hat)| >= 0.7, it's axial.
        - For layered structures (DJ/RP): Uses graph node attributes (is_equatorial, is_axial).

        Parameters
        ----------
        octahedra : list of str, optional
            Specific octahedra IDs to include. If None, includes all octahedra.

        Returns
        -------
        dict
            Dictionary with structure:
            {
                '180_equatorial': np.ndarray,  # 180° angles between equatorial X atoms
                '180_axial': np.ndarray,       # 180° angles between axial X atoms
                '90_all': np.ndarray,          # all 90° angles (not separated by geometry)
                '180_all': np.ndarray          # all 180° angles (for validation)
            }

        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.cif")
        >>> analyzer.analyze()
        >>> xbx_global = analyzer.get_xbx_angles_global_separated()
        >>> eq_angles = xbx_global['180_equatorial']
        >>> ax_angles = xbx_global['180_axial']
        >>> print(f"Mean equatorial deviation: {np.mean(np.abs(eq_angles - 180)):.2f}°")
        >>> print(f"Mean axial deviation: {np.mean(np.abs(ax_angles - 180)):.2f}°")
        """
        from ...utils.geometry.pbc_distances import find_nearest_image_positions
        from ..characterization.distortions import _get_octahedral_distortions_detailed

        self._ensure_analyzed()

        # Get all octahedral distortions (contains XBX angles in 'bond_angles')
        all_octahedra_distortions = _get_octahedral_distortions_detailed(
            self, octahedra=octahedra, layer=None, layers=None
        )

        if not all_octahedra_distortions:
            return {
                '180_equatorial': np.array([]),
                '180_axial': np.array([]),
                '90_all': np.array([]),
                '180_all': np.array([])
            }

        # Get graph and positions to determine X atom geometry
        graph = self.get_graph()
        atom_positions = self.cell.get_positions()
        cell = np.array(self.cell.get_cell())

        # Get all X atom positions for PBC calculations
        b_x_data = self.get_b_x_atoms()
        all_x_indices = b_x_data['x_indices']
        all_x_positions = b_x_data['x_positions']

        # Global collectors
        global_180_equatorial = []
        global_180_axial = []
        global_90_all = []
        global_180_all = []

        # Determine classification method
        use_geometric_bulk = (getattr(self, '_structure_type', None) == 'bulk')
        c_hat = None
        if use_geometric_bulk:
            c_vec = np.array(cell[2], dtype=np.float64)
            c_norm = np.linalg.norm(c_vec)
            c_hat = c_vec / c_norm if c_norm >= 1e-10 else np.array([0.0, 0.0, 1.0])

        # Process each octahedron
        for oct_id, oct_data in all_octahedra_distortions.items():
            central_idx = oct_data.get('central_atom_index')
            if central_idx is None:
                continue

            # Use already-calculated bond_angles from get_octahedral_distortions()
            bond_angles = oct_data.get('bond_angles')
            if bond_angles is None or len(bond_angles) == 0:
                continue

            # Reconstruct X atom indices and positions (minimum-image)
            central_pos = atom_positions[central_idx]
            nearest_x_indices, nearest_x_positions, nearest_distances, image_labels = find_nearest_image_positions(
                reference_position=central_pos,
                candidate_positions=all_x_positions,
                candidate_indices=all_x_indices,
                cell=cell,
                n_neighbors=6,
                pbc=True,
            )
            nearest_x_positions = np.asarray(nearest_x_positions)

            # Build X atom geometry classification
            x_geometry = {}
            if not use_geometric_bulk:
                for x_idx in nearest_x_indices:
                    atom_node = f"atom_{x_idx}"
                    node_data = graph.nodes.get(atom_node, {})
                    if 'is_axial' in node_data:
                        x_geometry[x_idx] = 'axial'
                    elif 'is_equatorial' in node_data:
                        x_geometry[x_idx] = 'equatorial'
                    elif 'is_interlayer' in node_data:
                        x_geometry[x_idx] = 'axial'
                    else:
                        x_geometry[x_idx] = 'equatorial'

            # Process all angles
            bond_angles_array = np.array(bond_angles) if hasattr(bond_angles, '__iter__') and not isinstance(bond_angles, str) else bond_angles
            angle_idx = 0
            for i in range(len(nearest_x_indices)):
                for j in range(i + 1, len(nearest_x_indices)):
                    if angle_idx >= len(bond_angles_array):
                        break
                    angle = bond_angles_array[angle_idx]
                    x_i_idx = nearest_x_indices[i]
                    x_j_idx = nearest_x_indices[j]

                    if angle < 135.0:
                        # 90° angle - collect all without geometry separation
                        global_90_all.append(angle)
                    else:
                        # 180° angle - classify by geometry
                        global_180_all.append(angle)

                        if use_geometric_bulk and c_hat is not None:
                            # Geometric classification for bulk structures
                            vec_xx = nearest_x_positions[j] - nearest_x_positions[i]
                            vec_xx_norm = np.linalg.norm(vec_xx) + 1e-9
                            axis_xx = vec_xx / vec_xx_norm
                            if np.abs(np.dot(axis_xx, c_hat)) >= 0.7:
                                global_180_axial.append(angle)
                            else:
                                global_180_equatorial.append(angle)
                        else:
                            # Graph-based classification for layered structures
                            geom_i = x_geometry.get(x_i_idx, 'equatorial')
                            geom_j = x_geometry.get(x_j_idx, 'equatorial')
                            if geom_i == 'equatorial' and geom_j == 'equatorial':
                                global_180_equatorial.append(angle)
                            elif geom_i == 'axial' and geom_j == 'axial':
                                global_180_axial.append(angle)
                            # Note: eq_ax mixed angles are not included in either category

                    angle_idx += 1

        return {
            '180_equatorial': np.array(global_180_equatorial) if global_180_equatorial else np.array([]),
            '180_axial': np.array(global_180_axial) if global_180_axial else np.array([]),
            '90_all': np.array(global_90_all) if global_90_all else np.array([]),
            '180_all': np.array(global_180_all) if global_180_all else np.array([])
        }

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
        from ..molecular_processing.spacer_analysis import SpacerAnalysis
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
        molecule: Union[Atoms, str, nx.Graph],
        spacer_type: Optional[str] = "DJ",
        initial_pattern: Union[str, List[str]] = None,
        final_pattern: Union[str, List[str]] = None,
        min_chain_length: int = 2,
        allowed_backbone_elements: Optional[Set[str]] = None,
        forbidden_backbone_elements: Optional[Set[str]] = None,
        max_non_carbon_ratio: Optional[float] = None,
        allow_same_carbon: bool = False
    ):
        """
        Validate molecule chemistry and optionally DJ/RP spacer patterns.

        Stage 1 (always): chemical completeness (valence, hydrogens, connectivity).
        Stage 2 (when spacer_type is DJ/RP): SMARTS terminal / path checks.

        Parameters
        ----------
        molecule : Atoms, str, or nx.Graph
            ASE Atoms, SMILES string, or molecular/CIF atom subgraph
        spacer_type : str or None
            ``"DJ"``, ``"RP"``, or ``None`` / ``"none"`` for chemistry only
        initial_pattern : str or List[str], optional
            SMILES pattern(s) for initial terminal (default: 'NH2C')
            Examples: '[NH3+]C', 'NH2C', ['[NH3+]C', 'NH2C']
        final_pattern : str or List[str], optional
            SMILES pattern(s) for final terminal (default: 'NH2C')
            Ignored for RP spacers and chemistry-only mode
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
        >>> # Chemistry only (no spacer pattern)
        >>> result = analyzer.mol_validate("C[NH3+]", spacer_type=None)
        >>> print(result.reason)
        """
        from ..molecular_processing.molecule_candidates import analyze_molecule_candidate
        return analyze_molecule_candidate(
            molecule,
            spacer_type=spacer_type,
            initial_pattern=initial_pattern,
            final_pattern=final_pattern,
            min_chain_length=min_chain_length,
            allowed_backbone_elements=allowed_backbone_elements,
            forbidden_backbone_elements=forbidden_backbone_elements,
            max_non_carbon_ratio=max_non_carbon_ratio,
            allow_same_carbon=allow_same_carbon
        )

    def validate_molecules(self) -> Dict[str, Any]:
        """Validate chemical completeness of all organic spacer/A-site molecules.

        Requires ``analyze()`` first. Iterates spacer and a_site nodes in the
        structural graph, skips inorganic-only sites, and runs ``mol_validate``
        chemistry-only on each organic subgraph.

        Returns
        -------
        dict
            ``is_valid``: True iff every organic molecule passes stage 1
            ``molecules``: list of per-molecule results
            ``n_organic`` / ``n_failed``: counts
        """
        self._ensure_analyzed()
        from .graph_construction import get_molecule_atoms
        from ..molecular_processing.molecule_chemistry import is_organic_molecule

        graph = self._graph
        results = []
        n_failed = 0

        for node, data in graph.nodes(data=True):
            if data.get("node_type") not in ("spacer", "a_site"):
                continue
            atom_ids = get_molecule_atoms(graph, node)
            if not atom_ids:
                continue
            sub = graph.subgraph(atom_ids).copy()
            if not is_organic_molecule(sub):
                continue

            mol_result = self.mol_validate(sub, spacer_type=None)
            entry = {
                "node": node,
                "node_type": data.get("node_type"),
                "is_valid": mol_result.is_valid,
                "reason": mol_result.reason,
            }
            results.append(entry)
            if not mol_result.is_valid:
                n_failed += 1

        return {
            "is_valid": n_failed == 0,
            "molecules": results,
            "n_organic": len(results),
            "n_failed": n_failed,
        }

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
        from ..molecular_processing.molecule_candidates import convert_nh2_to_nh3, clean_molecule

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
            all_ligands = oct['terminal_atoms'] + oct['interlayer_atoms'] + oct['intralayer_atoms']
            for idx in all_ligands:
                if idx < len(atom_symbols):
                    x_ions.add(atom_symbols[idx])
        x_ions = list(x_ions)

        a_sites = self.get_a_sites()
        a_ions = list(set(a['symbol'] for a in a_sites))

        spacers = self.get_spacers()
        spacer = spacers if spacers else None

        slab_layer_map = {
            sid: info.get('layer_ids', [])
            for sid, info in self.get_slabs().items()
        }

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
            n_slabs=self.n_slabs,
            is_twister=self.is_twister,
            slab_layer_map=slab_layer_map,
        )
    
    def export_graph_data(self, output_path: Optional[str] = None) -> str:
        """
        Export graph visualization data as JSON.

        Exports the structural analysis as a JSON file containing:
        - Nodes (octahedra, layers, atoms, molecules, structure, cavities) with ALL properties
        - Edges (connections between components) with ALL properties
        - Structure metadata (type, counts, formula)
        - Atom details for detailed inspection

        All properties from nodes and edges are exported automatically by iterating
        over the graph data dictionaries (similar to what's shown in hover tooltips).

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

        # Export ALL nodes with ALL their properties - just iterate over the dictionary
        for node_id, node_data in self._graph.nodes(data=True):
            # Start with the node ID and type
            node_entry = {
                'id': int(node_counter),
                'original_id': str(node_id),
            }
            
            # Add ALL properties from node_data dictionary
            for key, value in node_data.items():
                node_entry[key] = to_native(value)
            
            node_id_map[node_id] = node_counter
            nodes.append(node_entry)
            node_counter += 1

        # Export ALL edges with ALL their properties - just iterate over the dictionary
        edges = []
        for source, target, edge_data in self._graph.edges(data=True):
            if source in node_id_map and target in node_id_map:
                # Start with connection info
                edge_entry = {
                    'from': int(node_id_map[source]),
                    'to': int(node_id_map[target]),
                    'source_id': str(source),
                    'target_id': str(target),
                }
                
                # Add ALL properties from edge_data dictionary
                for key, value in edge_data.items():
                    edge_entry[key] = to_native(value)
                
                edges.append(edge_entry)

        # Additional atom information (for backward compatibility)
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

        # Layer connections are now derived from shared atoms between layers
        # We keep this section for backward compatibility but it will be empty
        # since interlayer connections are now queryable via graph traversal
        layer_edges = []

        graph_data = {
            'metadata': {
                'experiment_name': str(self.experiment_name),
                'file_path': str(self.file_path) if self.file_path else None,
                'structure_type': str(self._structure_type),
                'total_atoms': int(len(self.cell)),
                'formula': str(self.cell.get_chemical_formula()),
                'octahedra_count': len([n for n in nodes if n.get('node_type') == 'octahedron']),
                'layers_count': len([n for n in nodes if n.get('node_type') == 'layer']),
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
    
    def export_graph_svg(
        self,
        output_path: Optional[str] = None,
        exclude_atoms: bool = True,
        show_properties: bool = True,
        figsize: Tuple[float, float] = (16, 12),
        dpi: int = 300,
    ) -> str:
        """
        Export structure graph as publication-ready SVG visualization.

        Creates an SVG file with:
        - Color-coded nodes by type (octahedra, layers, molecules, etc.)
        - Color-coded edges by relationship type (contains, bonded_to, shares_atoms, etc.)
        - Optional property annotations (formula, z-coordinate, etc.)
        - Automatic legend with node and edge types
        - Spring layout for natural clustering

        Parameters
        ----------
        output_path : str, optional
            Path to save the SVG file. If None, uses '{experiment_name}_graph.svg'
        exclude_atoms : bool, optional
            If True (default), exclude atom nodes for cleaner visualization.
            Show only structural nodes (octahedra, layers, molecules).
        show_properties : bool, optional
            If True (default), show key properties as annotations near nodes
        figsize : tuple, optional
            Figure size in inches (default: 16x12 for publication quality)
        dpi : int, optional
            DPI for output (default: 300 for publication)

        Returns
        -------
        str
            Absolute path to created SVG file

        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.vasp")
        >>> analyzer.analyze()
        >>> # Export with default settings (structural nodes only, with annotations)
        >>> svg_path = analyzer.export_graph_svg("structure_graph.svg")
        >>>
        >>> # Export with all atoms included
        >>> svg_path = analyzer.export_graph_svg("structure_graph_full.svg", exclude_atoms=False)
        >>>
        >>> # Export without property annotations
        >>> svg_path = analyzer.export_graph_svg("structure_graph_clean.svg", show_properties=False)
        """
        self._ensure_analyzed()

        from ...utils.other.graph_svg_export import export_structure_svg

        if output_path is None:
            output_path = f"{self.experiment_name}_graph.svg"

        return export_structure_svg(
            self._graph,
            output_path,
            exclude_atoms=exclude_atoms,
            show_properties=show_properties,
            figsize=figsize,
            dpi=dpi,
        )
    
    # ========================================================================
    # Spacer Molecule Export Methods
    # ========================================================================
    
    def export_molecules_individual(self) -> List[Atoms]:
        """Export each spacer molecule as a separate ASE Atoms object.
        
        Each molecule is reconstructed with correct PBC coordinates but returned
        as an isolated Atoms object without periodic boundary conditions.
        
        Returns
        -------
        list of ase.Atoms
            Each element is a separate spacer molecule in cartesian coordinates.
            Molecules are ordered by their appearance in the structure.
            
        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.cif")
        >>> analyzer.analyze()
        >>> molecules = analyzer.export_molecules_individual()
        >>> for i, mol in enumerate(molecules):
        ...     mol.write(f'molecule_{i}.xyz')
        """
        self._ensure_analyzed()
        
        spacers = self.get_spacers()
        if not spacers:
            return []
        
        molecules = []
        for spacer in spacers:
            try:
                mol = self._reconstruct_spacer_molecule(spacer)
                molecules.append(mol)
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to reconstruct spacer molecule: {e}")
                # Fall back to original spacer atoms without PBC reconstruction
                molecules.append(spacer.copy())
        
        return molecules
    
    def export_molecules_cartesian(self) -> Atoms:
        """Export all spacer molecules combined in cartesian coordinates.
        
        All spacer molecules are reconstructed with correct PBC coordinates
        and combined into a single Atoms object. The result has no periodic
        boundary conditions (all atoms in cartesian space).
        
        Returns
        -------
        ase.Atoms
            Combined spacer molecules in cartesian coordinates.
            
        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.cif")
        >>> analyzer.analyze()
        >>> all_molecules = analyzer.export_molecules_cartesian()
        >>> all_molecules.write('all_molecules.xyz')
        """
        self._ensure_analyzed()
        
        spacers = self.get_spacers()
        if not spacers:
            return Atoms()
        
        all_symbols = []
        all_positions = []
        
        for spacer in spacers:
            try:
                mol = self._reconstruct_spacer_molecule(spacer)
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to reconstruct spacer molecule: {e}")
                mol = spacer.copy()
            
            all_symbols.extend(mol.get_chemical_symbols())
            all_positions.extend(mol.get_positions())
        
        # Create combined Atoms object without PBC
        combined = Atoms(symbols=all_symbols, positions=all_positions)
        return combined
    
    def export_molecules_pbc(self, z_vacuum: float = 10.0) -> Atoms:
        """Export all spacer molecules with PBC (inherits XY cell, controllable Z vacuum).
        
        All spacer molecules are reconstructed and combined into a single Atoms object
        with periodic boundary conditions. The cell parameters are inherited from the
        original structure in the XY plane, and a new Z cell parameter is calculated
        to accommodate all molecules with the specified vacuum.
        
        B and X (halide) atoms are filtered out; only spacer molecule atoms are included.
        
        Parameters
        ----------
        z_vacuum : float, default=10.0
            Z-direction vacuum to add around the molecules in Angstroms.
            
        Returns
        -------
        ase.Atoms
            Combined spacer molecules with PBC (Z direction is periodic).
            Only contains molecule atoms (no B or X atoms).
            
        Examples
        --------
        >>> analyzer = q2D_analyzer("structure.cif")
        >>> analyzer.analyze()
        >>> # Z vacuum = 15 Å
        >>> pbc_molecules = analyzer.export_molecules_pbc(z_vacuum=15.0)
        >>> pbc_molecules.write('molecules_pbc.xyz')
        """
        self._ensure_analyzed()
        
        spacers = self.get_spacers()
        if not spacers:
            return Atoms()
        
        # Get original cell (XY plane)
        original_cell = self.cell.get_cell()
        a = original_cell[0]
        b = original_cell[1]
        
        # Reconstruct all spacers and collect positions
        all_symbols = []
        all_positions = []
        
        for spacer in spacers:
            try:
                mol = self._reconstruct_spacer_molecule(spacer)
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to reconstruct spacer molecule: {e}")
                mol = spacer.copy()
            
            all_symbols.extend(mol.get_chemical_symbols())
            all_positions.extend(mol.get_positions())
        
        # Calculate Z cell parameter based on molecule positions
        if all_positions:
            all_positions_array = np.array(all_positions)
            z_min = np.min(all_positions_array[:, 2])
            z_max = np.max(all_positions_array[:, 2])
            z_height = z_max - z_min + z_vacuum
        else:
            z_height = z_vacuum
        
        # Create new cell: inherit XY, new Z
        c = np.array([0, 0, z_height])
        new_cell = np.array([a, b, c])
        
        # Create combined Atoms object with PBC
        combined = Atoms(symbols=all_symbols, positions=all_positions, cell=new_cell, pbc=True)
        
        # Translate Z coordinates so that molecules are centered in the cell
        if all_positions:
            z_center = (z_min + z_max) / 2.0
            z_shift = z_height / 2.0 - z_center
            positions = combined.get_positions()
            positions[:, 2] += z_shift
            combined.set_positions(positions)
        
        return combined
    
    def _reconstruct_spacer_molecule(self, spacer: Atoms) -> Atoms:
        """Reconstruct a spacer molecule with correct PBC coordinates.
        
        Uses the PBC reconstruction module to handle molecules that span
        periodic boundaries. Starts from NH3 groups when available.
        
        Parameters
        ----------
        spacer : ase.Atoms
            Spacer molecule to reconstruct
            
        Returns
        -------
        ase.Atoms
            Reconstructed spacer in PBC-unwrapped coordinates (isolated molecule)
        """
        from ...utils.molecules.pbc_reconstruction import (
            reconstruct_molecule_pbc,
            reconstruct_molecule_from_nh3
        )
        
        original_indices = spacer.info.get('original_indices', list(range(len(spacer))))
        
        # Try to reconstruct from NH3 groups first
        try:
            # Get geometric center of spacer
            spacer_positions = spacer.get_positions()
            geometric_center = np.mean(spacer_positions, axis=0)
            
            # Try NH3-aware reconstruction
            reconstructed_dict = reconstruct_molecule_from_nh3(
                original_indices,
                geometric_center,
                self._graph,
                self.cell.get_positions(),
                self.cell.get_chemical_symbols(),
                self.cell.get_cell(),
                pbc=True
            )
        except Exception as e:
            # Fall back to regular reconstruction
            spacer_positions = spacer.get_positions()
            geometric_center = np.mean(spacer_positions, axis=0)
            
            reconstructed_dict = reconstruct_molecule_pbc(
                original_indices,
                geometric_center,
                self._graph,
                self.cell.get_positions(),
                self.cell.get_chemical_symbols(),
                self.cell.get_cell(),
                pbc=True
            )
        
        # Convert reconstructed positions to Atoms object
        symbols = [self.cell[idx].symbol for idx in original_indices]
        positions = []
        for idx in original_indices:
            if idx in reconstructed_dict:
                pos, _ = reconstructed_dict[idx]
                positions.append(pos)
            else:
                positions.append(self.cell[idx].position)
        
        reconstructed_atoms = Atoms(symbols=symbols, positions=positions)
        reconstructed_atoms.info['original_indices'] = original_indices
        
        return reconstructed_atoms
    
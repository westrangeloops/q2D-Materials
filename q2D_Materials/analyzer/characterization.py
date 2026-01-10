"""
Perovskite characterization functions for common analysis tasks.

This module provides predefined getters for perovskite characterization:
- Glazer pattern detection (inverse of glazer_tilting.py)
- BXB angle calculations
- Partial radial distribution functions (RDF)
- Graph query interface for custom analysis

This module acts as a compatibility layer that re-exports functions
from specialized modules and provides a unified query interface:
- glazer_detection: Glazer pattern detection
- angle_analysis: B-X-B angle calculations
- rdf_analysis: Radial distribution functions
- Graph query helpers: Custom graph queries using NetworkX

Functions
---------
_detect_glazer_pattern
    Detect Glazer notation from structure by analyzing octahedral tilting
_calculate_bxb_angles
    Calculate B-X-B bond angles using analyzer graph structure
_calculate_partial_rdf
    Compute partial radial distribution functions for element pairs
_gaussian_kernel_discrete_spectrum
    Smooth discrete spectrum with gaussian kernel for RDF
analyze
    Route standard analyses (glazer, rdf, bxb) to specialized modules
query_octahedra, query_layers, query_atoms
    Graph query helper functions for filtering nodes
get_octahedron_neighbors, get_layer_octahedra, find_paths, get_subgraph
    Graph traversal and analysis helper functions
"""

from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from .analyzer_class import q2D_analyzer

# Import from specialized modules
from .glazer_detection import (
    _detect_glazer_pattern,
    DEFAULT_TOLERANCE,
    DEFAULT_TILT_SIGNIFICANCE_THRESHOLD,
    DEFAULT_ZERO_TILT_THRESHOLD,
    DEFAULT_ZERO_TILT_RELATIVE_THRESHOLD,
    DEFAULT_MAGNITUDE_EQUIVALENCE_THRESHOLD,
    DEFAULT_BOND_SELECTION_THRESHOLD,
    DEFAULT_NUMERICAL_TOLERANCE,
)

from .angle_analysis import _calculate_bxb_angles

from .rdf_analysis import (
    _calculate_partial_rdf,
    _gaussian_kernel_discrete_spectrum,
)


def analyze(analyzer: "q2D_analyzer", analysis_type: str, **kwargs: Any) -> Any:
    """
    Route standard analyses to specialized modules.
    
    Provides a unified interface for accessing standard analyses:
    - 'glazer': Glazer pattern detection
    - 'rdf': Radial distribution functions
    - 'bxb': B-X-B angle calculations
    
    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzer instance with analyzed structure
    analysis_type : str
        Type of analysis: 'glazer', 'rdf', or 'bxb'
    **kwargs
        Arguments passed to the specific analysis function
    
    Returns
    -------
    Any
        Result from the specific analysis function
    
    Examples
    --------
    >>> result = analyze(analyzer, 'glazer', tolerance=0.1)
    >>> result = analyze(analyzer, 'rdf', element_pairs=[["Pb", "I"]])
    >>> result = analyze(analyzer, 'bxb', include_bp=True)
    """
    if analysis_type == 'glazer':
        return _detect_glazer_pattern(analyzer, **kwargs)
    elif analysis_type == 'rdf':
        return _calculate_partial_rdf(analyzer, **kwargs)
    elif analysis_type == 'bxb':
        bxb_angles, bxbp_angles = _calculate_bxb_angles(analyzer, **kwargs)
        result = {
            "bxb_angles": bxb_angles,
            "bxb_mean": float(np.mean(bxb_angles)) if bxb_angles is not None and len(bxb_angles) > 0 else None,
            "bxb_std": float(np.std(bxb_angles)) if bxb_angles is not None and len(bxb_angles) > 0 else None,
        }
        if kwargs.get('include_bp', False):
            result["bxbp_angles"] = bxbp_angles
            result["bxbp_mean"] = (
                float(np.mean(bxbp_angles)) if bxbp_angles is not None and len(bxbp_angles) > 0 else None
            )
        return result
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}. Must be 'glazer', 'rdf', or 'bxb'.")



def query_octahedra(analyzer: "q2D_analyzer", **filters: Any) -> List[Dict[str, Any]]:
    """
    Query octahedra nodes with attribute filters.
    
    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzer instance with analyzed structure
    **filters
        Attribute filters (e.g., central_atom=5, node_type='octahedron')
    
    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with 'node' and 'data' keys for each matching octahedron
    """
    graph = analyzer.get_graph()
    results = []
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'octahedron':
            if all(data.get(k) == v for k, v in filters.items()):
                results.append({'node': node, 'data': data})
    return results


def query_layers(analyzer: "q2D_analyzer", **filters: Any) -> List[Dict[str, Any]]:
    """
    Query layer nodes with attribute filters.
    
    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzer instance with analyzed structure
    **filters
        Attribute filters (e.g., position='Surface', z_coord=5.0)
    
    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with 'node' and 'data' keys for each matching layer
    """
    graph = analyzer.get_graph()
    results = []
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'layer':
            if all(data.get(k) == v for k, v in filters.items()):
                results.append({'node': node, 'data': data})
    return results


def query_atoms(analyzer: "q2D_analyzer", **filters: Any) -> List[Dict[str, Any]]:
    """
    Query atom nodes with attribute filters.
    
    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzer instance with analyzed structure
    **filters
        Attribute filters (e.g., symbol='Pb', x_atom_type='intralayer')
    
    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with 'node' and 'data' keys for each matching atom
    """
    graph = analyzer.get_graph()
    results = []
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'atom':
            if all(data.get(k) == v for k, v in filters.items()):
                results.append({'node': node, 'data': data})
    return results


def get_octahedron_neighbors(
    analyzer: "q2D_analyzer", octahedron_id: str, edge_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get neighbors of an octahedron node.
    
    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzer instance with analyzed structure
    octahedron_id : str
        Node ID of the octahedron (e.g., 'octahedron_0')
    edge_type : str, optional
        Filter by edge type (e.g., 'shares_atoms', 'contains')
    
    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with 'node', 'data', and 'edge_data' keys
    """
    graph = analyzer.get_graph()
    if octahedron_id not in graph:
        return []
    
    neighbors = []
    for neighbor in graph.neighbors(octahedron_id):
        edge_data = graph[octahedron_id][neighbor]
        if edge_type is None or edge_data.get('edge_type') == edge_type:
            neighbors.append({
                'node': neighbor,
                'data': graph.nodes[neighbor],
                'edge_data': edge_data
            })
    return neighbors


def get_layer_octahedra(analyzer: "q2D_analyzer", layer_id: str) -> List[Dict[str, Any]]:
    """
    Get all octahedra in a specific layer.
    
    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzer instance with analyzed structure
    layer_id : str
        Layer node ID (e.g., 'layer_0') or layer identifier
    
    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with 'node' and 'data' keys for octahedra in the layer
    """
    graph = analyzer.get_graph()
    layer_node = layer_id if layer_id.startswith('layer_') else f'layer_{layer_id}'
    
    if layer_node not in graph:
        return []
    
    octahedra = []
    for neighbor in graph.neighbors(layer_node):
        edge_data = graph[layer_node][neighbor]
        if edge_data.get('edge_type') == 'contains':
            node_data = graph.nodes[neighbor]
            if node_data.get('node_type') == 'octahedron':
                octahedra.append({'node': neighbor, 'data': node_data})
    return octahedra


def find_paths(
    analyzer: "q2D_analyzer", source: str, target: str, max_length: Optional[int] = None
) -> List[List[str]]:
    """
    Find paths between two nodes in the graph.
    
    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzer instance with analyzed structure
    source : str
        Source node ID
    target : str
        Target node ID
    max_length : int, optional
        Maximum path length. If None, finds all simple paths
    
    Returns
    -------
    List[List[str]]
        List of paths, where each path is a list of node IDs
    """
    graph = analyzer.get_graph()
    if source not in graph or target not in graph:
        return []
    
    if max_length is None:
        try:
            paths = list(nx.all_simple_paths(graph, source, target))
        except nx.NetworkXNoPath:
            paths = []
    else:
        try:
            paths = list(nx.all_simple_paths(graph, source, target, cutoff=max_length))
        except nx.NetworkXNoPath:
            paths = []
    
    return paths


def get_subgraph(analyzer: "q2D_analyzer", node_ids: List[str]) -> nx.Graph:
    """
    Extract a subgraph containing the specified nodes and their edges.
    
    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzer instance with analyzed structure
    node_ids : List[str]
        List of node IDs to include in the subgraph
    
    Returns
    -------
    networkx.Graph
        Subgraph containing the specified nodes and connecting edges
    """
    graph = analyzer.get_graph()
    valid_nodes = [n for n in node_ids if n in graph]
    return graph.subgraph(valid_nodes).copy()


# ============================================================================
# Query Builder Class (Optional, for method chaining)
# ============================================================================

class CharacterizationQuery:
    """
    Query builder for graph-based characterization analysis.
    
    Provides a fluent interface for querying the graph and routing
    standard analyses. Supports method chaining for convenience.
    
    Examples
    --------
    >>> # Standard analysis routing
    >>> result = analyzer.get_characterization().glazer(tolerance=0.1).execute()
    >>> result = analyzer.get_characterization().rdf([["Pb", "I"]]).execute()
    
    >>> # Graph queries with method chaining
    >>> result = (analyzer.get_characterization()
    ...          .octahedra()
    ...          .neighbors(edge_type='shares_atoms')
    ...          .to_list())
    """
    
    def __init__(self, analyzer: "q2D_analyzer") -> None:
        """
        Initialize the query builder.
        
        Parameters
        ----------
        analyzer : q2D_analyzer
            Analyzer instance with analyzed structure
        """
        self.analyzer: "q2D_analyzer" = analyzer
        self._current_nodes: Optional[List[str]] = None
        self._query_type: Optional[str] = None
        self._query_kwargs: Dict[str, Any] = {}
    
    def octahedra(self, **filters: Any) -> "CharacterizationQuery":
        """
        Query octahedra nodes with filters.
        
        Parameters
        ----------
        **filters
            Attribute filters for octahedra
        
        Returns
        -------
        CharacterizationQuery
            Self for method chaining
        """
        results = query_octahedra(self.analyzer, **filters)
        self._current_nodes = [r['node'] for r in results]
        self._query_type = 'graph'
        return self
    
    def layers(self, **filters: Any) -> "CharacterizationQuery":
        """
        Query layer nodes with filters.
        
        Parameters
        ----------
        **filters
            Attribute filters for layers
        
        Returns
        -------
        CharacterizationQuery
            Self for method chaining
        """
        results = query_layers(self.analyzer, **filters)
        self._current_nodes = [r['node'] for r in results]
        self._query_type = 'graph'
        return self
    
    def neighbors(self, edge_type: Optional[str] = None) -> "CharacterizationQuery":
        """
        Get neighbors of currently selected nodes.
        
        Parameters
        ----------
        edge_type : str, optional
            Filter by edge type
        
        Returns
        -------
        CharacterizationQuery
            Self for method chaining
        """
        if self._current_nodes is None:
            raise ValueError("No nodes selected. Call octahedra() or layers() first.")
        
        all_neighbors = []
        for node_id in self._current_nodes:
            neighbors = get_octahedron_neighbors(self.analyzer, node_id, edge_type)
            all_neighbors.extend([n['node'] for n in neighbors])
        
        self._current_nodes = list(set(all_neighbors))  # Remove duplicates
        return self
    
    def glazer(self, **kwargs: Any) -> "CharacterizationQuery":
        """
        Route to Glazer pattern detection.
        
        Parameters
        ----------
        **kwargs
            Arguments for _detect_glazer_pattern
        
        Returns
        -------
        CharacterizationQuery
            Self for method chaining
        """
        self._query_type = 'glazer'
        self._query_kwargs = kwargs
        return self
    
    def rdf(self, element_pairs: List[List[str]], **kwargs: Any) -> "CharacterizationQuery":
        """
        Route to radial distribution function calculation.
        
        Parameters
        ----------
        element_pairs : List[List[str]]
            List of element pairs for RDF
        **kwargs
            Additional arguments for _calculate_partial_rdf
        
        Returns
        -------
        CharacterizationQuery
            Self for method chaining
        """
        self._query_type = 'rdf'
        self._query_kwargs = {'element_pairs': element_pairs, **kwargs}
        return self
    
    def bxb(self, **kwargs: Any) -> "CharacterizationQuery":
        """
        Route to B-X-B angle calculation.
        
        Parameters
        ----------
        **kwargs
            Arguments for _calculate_bxb_angles
        
        Returns
        -------
        CharacterizationQuery
            Self for method chaining
        """
        self._query_type = 'bxb'
        self._query_kwargs = kwargs
        return self
    
    def execute(self) -> Any:
        """
        Execute the query and return results.
        
        Returns
        -------
        Any
            Results based on query type
        """
        if self._query_type == 'glazer':
            return _detect_glazer_pattern(self.analyzer, **self._query_kwargs)
        elif self._query_type == 'rdf':
            return _calculate_partial_rdf(self.analyzer, **self._query_kwargs)
        elif self._query_type == 'bxb':
            bxb_angles, bxbp_angles = _calculate_bxb_angles(self.analyzer, **self._query_kwargs)
            result = {
                "bxb_angles": bxb_angles,
                "bxb_mean": float(np.mean(bxb_angles)) if bxb_angles is not None and len(bxb_angles) > 0 else None,
                "bxb_std": float(np.std(bxb_angles)) if bxb_angles is not None and len(bxb_angles) > 0 else None,
            }
            if self._query_kwargs.get('include_bp', False):
                result["bxbp_angles"] = bxbp_angles
                result["bxbp_mean"] = (
                    float(np.mean(bxbp_angles)) if bxbp_angles is not None and len(bxbp_angles) > 0 else None
                )
            return result
        elif self._query_type == 'graph':
            if self._current_nodes is None:
                return []
            graph = self.analyzer.get_graph()
            return [{'node': n, 'data': graph.nodes[n]} for n in self._current_nodes]
        else:
            raise ValueError("No query specified. Call a query method first.")
    
    def to_list(self) -> List[Dict[str, Any]]:
        """
        Return current selection as a list of node dictionaries.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of node dictionaries
        """
        if self._current_nodes is None:
            return []
        graph = self.analyzer.get_graph()
        return [{'node': n, 'data': graph.nodes[n]} for n in self._current_nodes]
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Return current selection as a dictionary keyed by node ID.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping node IDs to node data
        """
        if self._current_nodes is None:
            return {}
        graph = self.analyzer.get_graph()
        return {n: graph.nodes[n] for n in self._current_nodes}
    
    def to_graph(self) -> nx.Graph:
        """
        Return current selection as a NetworkX subgraph.
        
        Returns
        -------
        networkx.Graph
            Subgraph containing selected nodes and connecting edges
        """
        if self._current_nodes is None:
            return nx.Graph()
        return get_subgraph(self.analyzer, self._current_nodes)


def create_query(analyzer: "q2D_analyzer") -> CharacterizationQuery:
    """
    Create a CharacterizationQuery instance.
    
    Convenience function for creating query builders.
    
    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzer instance with analyzed structure
    
    Returns
    -------
    CharacterizationQuery
        Query builder instance
    """
    return CharacterizationQuery(analyzer)


# Re-export for backward compatibility
__all__ = [
    '_detect_glazer_pattern',
    '_calculate_bxb_angles',
    '_calculate_partial_rdf',
    '_gaussian_kernel_discrete_spectrum',
    'DEFAULT_TOLERANCE',
    'DEFAULT_TILT_SIGNIFICANCE_THRESHOLD',
    'DEFAULT_ZERO_TILT_THRESHOLD',
    'DEFAULT_ZERO_TILT_RELATIVE_THRESHOLD',
    'DEFAULT_MAGNITUDE_EQUIVALENCE_THRESHOLD',
    'DEFAULT_BOND_SELECTION_THRESHOLD',
    'DEFAULT_NUMERICAL_TOLERANCE',
    # Graph query interface
    'analyze',
    'query_octahedra',
    'query_layers',
    'query_atoms',
    'get_octahedron_neighbors',
    'get_layer_octahedra',
    'find_paths',
    'get_subgraph',
    'create_query',
    'CharacterizationQuery',
]

"""
Shared graph utilities for NetworkX graph operations.

This module provides common graph operations that work on any NetworkX graph,
whether it's a structural graph (from crystal structures) or a molecular graph
(from SMILES or molecular structures).

These utilities are used by:
- Structural graph queries (characterization.py)
- Molecular graph analysis (molecule_candidates.py)
- Other graph-based analysis modules
"""

from typing import List, Optional, Set, Dict, Any
import networkx as nx


def find_shortest_path(graph: nx.Graph, source: Any, target: Any) -> Optional[List[Any]]:
    """
    Find shortest path between two nodes in a graph.
    
    Works on any NetworkX graph (structural or molecular).
    
    Parameters
    ----------
    graph : nx.Graph
        Any NetworkX graph
    source : Any
        Source node
    target : Any
        Target node
        
    Returns
    -------
    Optional[List[Any]]
        Shortest path as list of nodes, or None if no path exists
    """
    try:
        return nx.shortest_path(graph, source, target)
    except nx.NetworkXNoPath:
        return None


def find_all_paths(
    graph: nx.Graph,
    source: Any,
    target: Any,
    max_length: Optional[int] = None
) -> List[List[Any]]:
    """
    Find all simple paths between two nodes.
    
    Works on any NetworkX graph (structural or molecular).
    
    Parameters
    ----------
    graph : nx.Graph
        Any NetworkX graph
    source : Any
        Source node
    target : Any
        Target node
    max_length : int, optional
        Maximum path length. If None, finds all simple paths
        
    Returns
    -------
    List[List[Any]]
        List of paths, where each path is a list of nodes
    """
    if source not in graph or target not in graph:
        return []
    
    if max_length is None:
        try:
            return list(nx.all_simple_paths(graph, source, target))
        except nx.NetworkXNoPath:
            return []
    else:
        try:
            return list(nx.all_simple_paths(graph, source, target, cutoff=max_length))
        except nx.NetworkXNoPath:
            return []


def get_connected_components(graph: nx.Graph) -> List[Set[Any]]:
    """
    Get all connected components in a graph.
    
    Works on any NetworkX graph (structural or molecular).
    
    Parameters
    ----------
    graph : nx.Graph
        Any NetworkX graph
        
    Returns
    -------
    List[Set[Any]]
        List of connected components, each as a set of nodes
    """
    return list(nx.connected_components(graph))


def filter_nodes_by_attributes(
    graph: nx.Graph,
    **filters: Any
) -> List[Any]:
    """
    Filter nodes by their attributes.
    
    Works on any NetworkX graph (structural or molecular).
    
    Parameters
    ----------
    graph : nx.Graph
        Any NetworkX graph
    **filters
        Attribute filters (e.g., symbol='C', node_type='atom')
        
    Returns
    -------
    List[Any]
        List of node IDs matching all filters
    """
    results = []
    for node, data in graph.nodes(data=True):
        if all(data.get(k) == v for k, v in filters.items()):
            results.append(node)
    return results


def get_node_neighbors(
    graph: nx.Graph,
    node: Any,
    edge_type: Optional[str] = None
) -> List[Any]:
    """
    Get neighbors of a node, optionally filtered by edge type.
    
    Works on any NetworkX graph (structural or molecular).
    
    Parameters
    ----------
    graph : nx.Graph
        Any NetworkX graph
    node : Any
        Node ID
    edge_type : str, optional
        Filter by edge type attribute
        
    Returns
    -------
    List[Any]
        List of neighbor node IDs
    """
    if node not in graph:
        return []
    
    neighbors = []
    for neighbor in graph.neighbors(node):
        if edge_type is None:
            neighbors.append(neighbor)
        else:
            edge_data = graph.get_edge_data(node, neighbor, {})
            if edge_data.get('edge_type') == edge_type:
                neighbors.append(neighbor)
    
    return neighbors


def validate_path_continuity(graph: nx.Graph, path: List[Any]) -> bool:
    """
    Validate that all consecutive nodes in a path are connected by edges.
    
    Works on any NetworkX graph (structural or molecular).
    
    Parameters
    ----------
    graph : nx.Graph
        Any NetworkX graph
    path : List[Any]
        List of node IDs forming a path
        
    Returns
    -------
    bool
        True if path is continuous (all consecutive nodes are connected)
    """
    if len(path) < 2:
        return True
    
    for i in range(len(path) - 1):
        if not graph.has_edge(path[i], path[i + 1]):
            return False
    
    return True


def extract_subgraph(graph: nx.Graph, node_ids: List[Any]) -> nx.Graph:
    """
    Extract a subgraph containing specified nodes and their edges.
    
    Works on any NetworkX graph (structural or molecular).
    
    Parameters
    ----------
    graph : nx.Graph
        Any NetworkX graph
    node_ids : List[Any]
        List of node IDs to include in subgraph
        
    Returns
    -------
    nx.Graph
        Subgraph containing specified nodes and connecting edges
    """
    valid_nodes = [n for n in node_ids if n in graph]
    return graph.subgraph(valid_nodes).copy()


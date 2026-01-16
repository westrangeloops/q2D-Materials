"""
Graph exporter for pyvis visualization.

This module provides functionality to export NetworkX graphs to HTML format
using pyvis for interactive visualization.
"""

import networkx as nx
from pathlib import Path
from typing import Union, Dict, Any, Optional
import numpy as np

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    Network = None

def _get_element_color(symbol: str) -> str:
    """Get hex color for an element symbol."""
    element_colors = {
        'Pb': '#6272a4', 'Sn': '#8be9fd', 'Ge': '#ffb86c', 'Ti': '#bd93f9',
        'Zr': '#bd93f9', 'Hf': '#ff79c6', 'Nb': '#f1fa8c', 'Ta': '#6272a4',
        'I': '#ff5555', 'Br': '#ff79c6', 'Cl': '#50fa7b', 'F': '#f1fa8c', 'O': '#50fa7b',
        'Cs': '#50fa7b', 'Rb': '#50fa7b', 'K': '#50fa7b',
        'C': '#44475a', 'N': '#8be9fd', 'H': '#f8f8f2', 'S': '#ffb86c',
    }
    return element_colors.get(symbol, '#888888')

def _get_node_color(node_data: Dict[str, Any]) -> str:
    """Get color for a node based on its type and attributes."""
    node_type = node_data.get('node_type', 'unknown')
    
    if node_type == 'atom':
        symbol = node_data.get('symbol', '')
        return _get_element_color(symbol) if symbol else '#888888'
    elif node_type == 'octahedron':
        return '#bd93f9'  # Purple
    elif node_type == 'layer':
        return '#ff79c6'  # Pink
    elif node_type == 'molecule':
        return '#ffb86c'  # Orange
    return '#888888'

def _get_node_size(node_data: Dict[str, Any]) -> int:
    """Get size for a node based on its type."""
    node_type = node_data.get('node_type', 'unknown')
    if node_type == 'atom':
        return 10
    elif node_type == 'octahedron':
        return 25
    elif node_type == 'layer':
        return 30
    elif node_type == 'molecule':
        return 20
    return 15

def _create_node_label(node_id: str, node_data: Dict[str, Any]) -> str:
    """Create a meaningful label for a node."""
    node_type = node_data.get('node_type', 'unknown')
    
    if node_type == 'atom':
        symbol = node_data.get('symbol', '')
        vasp_idx = node_data.get('vasp_index', '')
        return f"{symbol}{vasp_idx}" if symbol and vasp_idx != '' else symbol or str(node_id)
    elif node_type == 'octahedron':
        oct_idx = node_id.split('_')[-1] if '_' in node_id else node_id
        return f"Oct{oct_idx}"
    elif node_type == 'layer':
        layer_id = node_id.split('_')[-1] if '_' in node_id else node_id
        return f"Layer {layer_id}"
    elif node_type == 'molecule':
        return node_data.get('formula', str(node_id).replace('_', ' '))
    return str(node_id).replace('_', ' ')

def export_structure_pyvis(
    graph: nx.Graph,
    output_path: Union[str, Path],
    exclude_cavities: bool = True
) -> str:
    """
    Export structure graph to HTML using pyvis.
    
    Parameters
    ----------
    graph : nx.Graph
        The structural graph (from analyzer.get_graph())
    output_path : str or Path
        Path to save HTML file
    exclude_cavities : bool, optional
        If True (default), exclude cavity nodes and their edges
        
    Returns
    -------
    str
        Absolute path to created file
    """
    if not PYVIS_AVAILABLE:
        raise ImportError("pyvis is required. Install with: pip install pyvis")
    
    path = Path(output_path)
    if not path.suffix:
        path = path.with_suffix('.html')
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a copy and filter cavities if needed
    G = graph.copy()
    if exclude_cavities:
        cavity_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'cavity']
        G.remove_nodes_from(cavity_nodes)
    
    # Create pyvis network
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    net.set_options("""
    {
      "nodes": {
        "font": {"size": 12},
        "scaling": {"min": 10, "max": 30}
      },
      "edges": {
        "width": 2,
        "color": {"inherit": true}
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.04,
          "damping": 0.09
        }
      }
    }
    """)
    
    # Add nodes
    for node_id, data in G.nodes(data=True):
        label = _create_node_label(node_id, data)
        color = _get_node_color(data)
        size = _get_node_size(data)
        
        # Build title with node info
        title_parts = [f"Type: {data.get('node_type', 'unknown')}"]
        if 'symbol' in data:
            title_parts.append(f"Symbol: {data['symbol']}")
        if 'vasp_index' in data:
            title_parts.append(f"Index: {data['vasp_index']}")
        title = "\\n".join(title_parts)
        
        net.add_node(
            str(node_id),
            label=label,
            color=color,
            size=size,
            title=title
        )
    
    # Add edges
    for u, v, data in G.edges(data=True):
        edge_type = data.get('edge_type', 'unknown')
        net.add_edge(str(u), str(v), title=edge_type)
    
    # Save to file
    net.save_graph(str(path))
    
    return str(path.absolute())

def export_cavity_pyvis(
    cavity_graph: nx.Graph,
    cavity_index: int,
    output_dir: Union[str, Path]
) -> str:
    """
    Export a single cavity subgraph to HTML using pyvis.
    
    Parameters
    ----------
    cavity_graph : nx.Graph
        The cavity subgraph
    cavity_index : int
        Index of the cavity
    output_dir : str or Path
        Directory to save the file
        
    Returns
    -------
    str
        Absolute path to created file
    """
    if not PYVIS_AVAILABLE:
        raise ImportError("pyvis is required. Install with: pip install pyvis")
    
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    filename = path / f"cavity_{cavity_index}.html"
    
    # Create pyvis network
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    net.set_options("""
    {
      "nodes": {
        "font": {"size": 12},
        "scaling": {"min": 10, "max": 30}
      },
      "edges": {
        "width": 2,
        "color": {"inherit": true}
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.04,
          "damping": 0.09
        }
      }
    }
    """)
    
    # Add nodes
    for node_id, data in cavity_graph.nodes(data=True):
        label = _create_node_label(node_id, data)
        color = _get_node_color(data)
        size = _get_node_size(data)
        
        title_parts = [f"Type: {data.get('node_type', 'unknown')}"]
        if 'symbol' in data:
            title_parts.append(f"Symbol: {data['symbol']}")
        title = "\\n".join(title_parts)
        
        net.add_node(
            str(node_id),
            label=label,
            color=color,
            size=size,
            title=title
        )
    
    # Add edges
    for u, v, data in cavity_graph.edges(data=True):
        edge_type = data.get('edge_type', 'unknown')
        net.add_edge(str(u), str(v), title=edge_type)
    
    # Save to file
    net.save_graph(str(filename))
    
    return str(filename.absolute())

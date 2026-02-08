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

try:
    import pydot
    PYDOT_AVAILABLE = True
except ImportError:
    PYDOT_AVAILABLE = False
    pydot = None

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import to_agraph
    PYGRAPHVIZ_AVAILABLE = True
except (ImportError, OSError):
    # OSError can occur if Graphviz is not installed
    PYGRAPHVIZ_AVAILABLE = False
    to_agraph = None

def _get_element_color(symbol: str) -> str:
    """Get hex color for an element symbol from theme.css."""
    element_colors = {
        'Pb': '#C3D1D1', 'Sn': '#C3D1D1', 'Ge': '#C3D1D1', 'Ti': '#C3D1D1',
        'Zr': '#C3D1D1', 'Hf': '#C3D1D1', 'Nb': '#C3D1D1', 'Ta': '#C3D1D1',
        'I': '#8F26AB', 'Br': '#8F26AB', 'Cl': '#8F26AB', 'F': '#8F26AB', 'O': '#8F26AB',
        'Cs': '#F5B102', 'Rb': '#F5B102', 'K': '#F5B102',
        'C': '#64748b', 'N': '#3b82f6', 'H': '#e2e8f0', 'S': '#fb923c',
    }
    return element_colors.get(symbol, '#94a3b8')

def _get_node_color(node_data: Dict[str, Any]) -> str:
    """Get color for a node based on its type using theme.css colors."""
    node_type = node_data.get('node_type', 'unknown')
    
    if node_type == 'atom':
        symbol = node_data.get('symbol', '')
        return _get_element_color(symbol) if symbol else '#94a3b8'
    elif node_type == 'octahedron':
        return '#C3D1D1'  # Light blue-gray (B-site color)
    elif node_type == 'layer':
        return '#49ADB6'  # Success/teal (layer color)
    elif node_type == 'a_site':
        return '#F5B102'  # Gold/yellow (A-site color)
    elif node_type == 'spacer':
        return '#039FE5'  # Blue (spacer color)
    elif node_type == 'cavity':
        return '#F5B102'  # Gold/yellow (cavity color)
    elif node_type == 'molecule':
        return '#F5B102'  # Gold/yellow (molecule color)
    elif node_type in ['cage', 'half_cage']:
        return '#F5B102'  # Gold/yellow (same as cavity)
    elif node_type == 'anchor':
        return '#E91E63'  # Pink for anchors
    return '#94a3b8'

def _get_node_size(node_data: Dict[str, Any]) -> int:
    """Get size for a node based on its type."""
    node_type = node_data.get('node_type', 'unknown')
    if node_type == 'atom':
        return 10
    elif node_type == 'octahedron':
        return 25
    elif node_type == 'layer':
        return 30
    elif node_type in ['a_site', 'spacer']:
        return 20
    elif node_type in ['cage', 'half_cage']:
        return 25
    elif node_type == 'anchor':
        return 10
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
    elif node_type in ['a_site', 'spacer']:
        return node_data.get('formula', str(node_id).replace('_', ' '))
    return str(node_id).replace('_', ' ')

def _get_node_shape(node_data: Dict[str, Any]) -> str:
    """Get Graphviz shape for a node based on its type."""
    node_type = node_data.get('node_type', 'unknown')
    
    if node_type == 'octahedron':
        return 'diamond'  # Rhombus/diamond (4 edges) for octahedra
    elif node_type == 'layer':
        return 'box'  # Rectangle (not square) for layers
    elif node_type == 'spacer':
        return 'ellipse'  # Ellipse for spacers/molecules
    elif node_type == 'molecule':
        return 'ellipse'  # Ellipse for molecules
    elif node_type == 'structure':
        return 'star'  # Star for structure
    elif node_type == 'atom':
        return 'dot'  # Dot for atoms (label outside)
    elif node_type in ['cage', 'half_cage']:
        return 'square'
    elif node_type == 'anchor':
        return 'triangle'
    return 'dot'

def _get_edge_color(edge_data: Dict[str, Any]) -> str:
    """Get color for an edge based on its type."""
    edge_type = edge_data.get('edge_type', 'unknown')
    color_map = {
        'contains': '#64748b',
        'bonded_to': '#f59e0b',
        'shares_atoms': '#ef4444',
        'is_contained_in': '#94a3b8',
        'proximity': '#10b981',
        'cavity_octahedron': '#fbbf24',
        'contains_a_site': '#ec4899',
    }
    return color_map.get(edge_type, '#cbd5e1')

def _get_edge_style(edge_data: Dict[str, Any]) -> str:
    """Get line style for an edge based on its type."""
    edge_type = edge_data.get('edge_type', 'unknown')
    if edge_type == 'shares_atoms':
        return 'dashed'
    elif edge_type == 'proximity':
        return 'dotted'
    return 'solid'

def _remove_svg_background(svg_path: str) -> None:
    """Remove white background polygon from Graphviz-generated SVG."""
    import re
    
    with open(svg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove white background polygon - matches the pattern from Graphviz output
    # Pattern 1: <polygon fill="white" stroke="none" points="..."/>
    # Pattern 2: <polygon fill="white" stroke="transparent" points="..."/>
    # Pattern 3: <polygon fill="white" ... points="..."/> (any stroke value)
    pattern = r'<polygon\s+fill="(?:white|#ffffff|#FFFFFF)"[^>]*points="[^"]*"[^>]*/>\s*\n?'
    content = re.sub(pattern, '', content, flags=re.MULTILINE)
    
    with open(svg_path, 'w', encoding='utf-8') as f:
        f.write(content)

def export_structure_pyvis(
    graph: nx.Graph,
    output_path: Union[str, Path],
    exclude_cavities: bool = True,
    export_cavities: Optional[Any] = None,
    cavity_output_dir: Optional[Union[str, Path]] = None
) -> str:
    """
    Export structure graph to HTML using pyvis.
    
    This function exports the whole structure graph by default. Cavities are
    excluded from the main graph visualization unless explicitly included.
    
    Note: This function only exports the main structural graph returned by
    analyzer.get_graph(). This graph contains layers, octahedra, atoms, a_sites,
    and spacers.
    
    Parameters
    ----------
    graph : nx.Graph
        The structural graph (from analyzer.get_graph()). This is the main
        structure graph containing layers, octahedra, atoms, a_sites, and spacers.
    output_path : str or Path
        Path to save HTML file for the structure graph
    exclude_cavities : bool, optional
        If True (default), exclude cavity nodes and their edges from the
        structure graph visualization. Set to False to include cavities in
        the main graph.
    export_cavities : CavityCollection or list, optional
        If provided, also export individual cavity graphs. Should be a
        CavityCollection from analyzer.get_cavities() or a list of Cavity objects.
        Default is None (no cavity export).
    cavity_output_dir : str or Path, optional
        Directory to save cavity HTML files. Only used if export_cavities is provided.
        If None and export_cavities is provided, uses the same directory as output_path.
        
    Returns
    -------
    str
        Absolute path to created structure graph file
        
    Examples
    --------
    >>> # Export whole structure only (default)
    >>> graph = analyzer.get_graph()
    >>> export_structure_pyvis(graph, 'structure.html')
    
    >>> # Export structure with cavities included in main graph
    >>> export_structure_pyvis(graph, 'structure.html', exclude_cavities=False)
    
    >>> # Export structure and also export individual cavity graphs
    >>> cavities = analyzer.get_cavities()
    >>> export_structure_pyvis(graph, 'structure.html', export_cavities=cavities)
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
    
    # Helper function to convert values to strings for tooltips
    def format_value(value):
        """Format a value for display in tooltip."""
        if value is None:
            return "None"
        elif isinstance(value, (list, tuple)):
            return f"[{', '.join(str(format_value(v)) for v in value[:5])}{'...' if len(value) > 5 else ''}]"
        elif isinstance(value, dict):
            return f"{{...}}"  # Don't show full dict in tooltip
        elif isinstance(value, (int, float)):
            return f"{value:.4f}" if isinstance(value, float) else str(value)
        else:
            return str(value)
    
    # Add nodes
    for node_id, data in G.nodes(data=True):
        label = _create_node_label(node_id, data)
        color = _get_node_color(data)
        size = _get_node_size(data)
        shape = _get_node_shape(data)
        
        # Build title with ALL node properties - iterate over dictionary
        title_parts = []
        for key, value in sorted(data.items()):  # Sort for consistent display
            formatted_value = format_value(value)
            title_parts.append(f"{key}: {formatted_value}")
        # Join with a readable separator for tooltips
        # HTML title attributes don't support newlines, so use a visual separator
        title = " | ".join(title_parts) if title_parts else "No properties"
        
        net.add_node(
            str(node_id),
            label=label,
            color=color,
            size=size,
            title=title,
            shape=shape
        )
    
    # Add edges
    for u, v, data in G.edges(data=True):
        # Build title with ALL edge properties - iterate over dictionary
        title_parts = []
        for key, value in sorted(data.items()):  # Sort for consistent display
            formatted_value = format_value(value)
            title_parts.append(f"{key}: {formatted_value}")
        # Join with a readable separator for tooltips
        # HTML title attributes don't support newlines, so use a visual separator
        title = " | ".join(title_parts) if title_parts else "No properties"
        
        # Create edge label showing distance if available
        edge_label = None
        if 'distance' in data:
            distance = data['distance']
            edge_label = f"{distance:.3f}Å"
        
        net.add_edge(str(u), str(v), title=title, label=edge_label)
    
    # Save to file
    net.save_graph(str(path))
    
    # Optionally export individual cavity graphs
    if export_cavities is not None:
        if cavity_output_dir is None:
            # Use same directory as output file
            cavity_output_dir = path.parent
        else:
            cavity_output_dir = Path(cavity_output_dir)
        
        # Handle both CavityCollection and list of cavities
        try:
            # Try to iterate (works for both list and CavityCollection)
            cavities_list = list(export_cavities)
        except (TypeError, AttributeError):
            raise ValueError(
                "export_cavities must be a CavityCollection (from analyzer.get_cavities()) "
                "or a list of Cavity objects"
            )
        
        for i, cavity in enumerate(cavities_list):
            if hasattr(cavity, 'subgraph') and cavity.subgraph is not None:
                try:
                    export_cavity_pyvis(
                        cavity.subgraph,
                        i,
                        cavity_output_dir
                    )
                except Exception as e:
                    import sys
                    print(
                        f"Warning: Could not export graph for cavity {i}: {e}",
                        file=sys.stderr
                    )
    
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
    
    # Helper function to convert values to strings for tooltips
    def format_value(value):
        """Format a value for display in tooltip."""
        if value is None:
            return "None"
        elif isinstance(value, (list, tuple)):
            return f"[{', '.join(str(format_value(v)) for v in value[:5])}{'...' if len(value) > 5 else ''}]"
        elif isinstance(value, dict):
            return f"{{...}}"  # Don't show full dict in tooltip
        elif isinstance(value, (int, float)):
            return f"{value:.4f}" if isinstance(value, float) else str(value)
        else:
            return str(value)
    
    # Add nodes
    for node_id, data in cavity_graph.nodes(data=True):
        label = _create_node_label(node_id, data)
        color = _get_node_color(data)
        size = _get_node_size(data)
        shape = _get_node_shape(data)
        
        # Build title with ALL node properties - iterate over dictionary
        title_parts = []
        for key, value in sorted(data.items()):  # Sort for consistent display
            formatted_value = format_value(value)
            title_parts.append(f"{key}: {formatted_value}")
        # Join with a readable separator for tooltips
        # HTML title attributes don't support newlines, so use a visual separator
        title = " | ".join(title_parts) if title_parts else "No properties"
        
        net.add_node(
            str(node_id),
            label=label,
            color=color,
            size=size,
            title=title,
            shape=shape
        )
    
    # Add edges
    for u, v, data in cavity_graph.edges(data=True):
        # Build title with ALL edge properties - iterate over dictionary
        title_parts = []
        for key, value in sorted(data.items()):  # Sort for consistent display
            formatted_value = format_value(value)
            title_parts.append(f"{key}: {formatted_value}")
        # Join with a readable separator for tooltips
        # HTML title attributes don't support newlines, so use a visual separator
        title = " | ".join(title_parts) if title_parts else "No properties"
        
        # Create edge label showing distance if available
        edge_label = None
        if 'distance' in data:
            distance = data['distance']
            edge_label = f"{distance:.3f}Å"
        
        net.add_edge(str(u), str(v), title=title, label=edge_label)
    
    # Save to file
    net.save_graph(str(filename))
    
    return str(filename.absolute())


def export_structure_svg(
    graph: nx.Graph,
    output_path: Union[str, Path],
    exclude_cavities: bool = True,
    exclude_atoms: bool = False,
    layout: str = 'sfdp',
    dpi: int = 300,
    spring_constant: float = 5.0,
    repulsive_force: float = 2.0
) -> str:
    """
    Export structure graph to SVG using pygraphviz/Graphviz.
    
    This function exports the graph as a static SVG file with proper node shapes,
    colors matching theme.css, and Graphviz layout algorithms.
    
    Parameters
    ----------
    graph : nx.Graph
        The structural graph (from analyzer.get_graph())
    output_path : str or Path
        Path to save SVG file
    exclude_cavities : bool, optional
        If True (default), exclude cavity nodes from visualization
    exclude_atoms : bool, optional
        If True, exclude atom nodes for cleaner visualization (default: False, atoms are shown)
    layout : str, optional
        Graphviz layout engine: 'sfdp' (default, scalable force-directed), 'neato', 'fdp', 'dot', 'circo'
    dpi : int, optional
        DPI for output (default: 300). Size is auto-calculated to fit all nodes.
    spring_constant : float, optional
        Spring constant K for sfdp layout - higher values increase edge forces (default: 5.0)
    repulsive_force : float, optional
        Repulsive force for sfdp layout - higher values increase node separation (default: 2.0)
        
    Returns
    -------
    str
        Absolute path to created SVG file
        
    Examples
    --------
    >>> graph = analyzer.get_graph()
    >>> export_structure_svg(graph, 'structure.svg')
    """
    if not PYGRAPHVIZ_AVAILABLE:
        raise ImportError("pygraphviz is required. Install with: pip install pygraphviz")
    
    path = Path(output_path)
    if not path.suffix:
        path = path.with_suffix('.svg')
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a copy and filter if needed
    G = graph.copy()
    if exclude_cavities:
        cavity_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'cavity']
        G.remove_nodes_from(cavity_nodes)
    
    if exclude_atoms:
        atom_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'atom']
        G.remove_nodes_from(atom_nodes)
    
    if len(G.nodes()) == 0:
        raise ValueError("Graph has no nodes after filtering")
    
    # Convert NetworkX graph to pygraphviz AGraph
    A = to_agraph(G)
    
    # Set graph attributes - no background, auto-size
    # For sfdp layout, use specific attributes for better results
    graph_attrs = {
        'dpi': str(dpi),
        'overlap': 'false',
        'splines': 'curved',  # Curved edges for better visualization
        'pad': '0.5',
        # No bgcolor - transparent
        # No size - auto-size
    }
    
    # Add sfdp-specific attributes if using sfdp layout
    if layout == 'sfdp':
        graph_attrs.update({
            'K': str(spring_constant),  # Spring constant - higher = stronger edge forces (pulls nodes together)
            'repulsiveforce': str(repulsive_force),  # Repulsive force power - higher = more repulsion, more node separation
            'beautify': 'true',  # Draw leaf nodes uniformly in a circle around root nodes
            'overlap_scaling': '4',  # Scale to reduce overlap
            'overlap_shrink': 'true',  # Compression pass
            'smoothing': 'triangle',  # Smoothing algorithm
        })
    else:
        graph_attrs['rankdir'] = 'TB'  # Top to bottom for other layouts
    
    A.graph_attr.update(graph_attrs)
    
    # Set default node style
    A.node_attr.update({
        'fontname': 'Arial',
        'fontsize': '14',  # Bigger font
        'fontcolor': '#1e293b',  # Dark text for transparent/light background
        'fontweight': 'bold',  # Bold text
        'style': 'filled',
        'penwidth': '2'
    })
    
    # Set default edge style
    A.edge_attr.update({
        'penwidth': '3.0',
        'color': '#94a3b8'  # Lighter gray for better visibility
    })
    
    # Set node attributes
    for node_id, data in G.nodes(data=True):
        label = _create_node_label(node_id, data)
        color = _get_node_color(data)
        size_val = _get_node_size(data)
        shape = _get_node_shape(data)
        
        # Map pyvis 'dot' to graphviz 'circle'
        if shape == 'dot':
            shape = 'circle'
        
        # Calculate node size
        node_size = max(size_val / 15.0, 0.3)  # Minimum 0.3 inches
        
        # Special handling for X-site atoms - add X to label
        if data.get('node_type') == 'atom' and data.get('is_X', False):
            label = f"{label} ✕"
        
        # Set node attributes
        node = A.get_node(str(node_id))
        node.attr['label'] = label
        node.attr['fillcolor'] = color
        node.attr['shape'] = shape
        
        # For rectangles (layers), make them wider than tall
        if shape == 'box':
            node.attr['width'] = str(node_size * 1.2)  # Wider
            node.attr['height'] = str(node_size / 1.5)  # Taller
        else:
            node.attr['width'] = str(node_size)
            node.attr['height'] = str(node_size)
        
        node.attr['fontcolor'] = '#1e293b'
        node.attr['fontsize'] = '14'  # Bigger font
        node.attr['fontweight'] = 'bold'  # Bold text
    
    # Set edge attributes
    for u, v, data in G.edges(data=True):
        edge_color = _get_edge_color(data)
        edge_style = _get_edge_style(data)
        
        edge = A.get_edge(str(u), str(v))
        edge.attr['color'] = edge_color
        edge.attr['style'] = edge_style
        edge.attr['penwidth'] = '3.0'
        
        # Add distance label if available
        if 'distance' in data:
            distance = data['distance']
            edge.attr['label'] = f"{distance:.3f}Å"
            edge.attr['fontsize'] = '10'
            edge.attr['fontcolor'] = '#1e293b'
    
    # Render and save as SVG
    A.draw(str(path), format='svg', prog=layout)
    
    # Post-process SVG to remove white background polygon
    _remove_svg_background(str(path))
    
    return str(path.absolute())


def export_cavity_svg(
    cavity_graph: nx.Graph,
    cavity_index: int,
    output_path: Union[str, Path],
    exclude_atoms: bool = False,
    layout: str = 'sfdp',
    dpi: int = 300,
    spring_constant: float = 5.0,
    repulsive_force: float = 2.0
) -> str:
    """
    Export a single cavity subgraph to SVG using pygraphviz/Graphviz.
    
    Parameters
    ----------
    cavity_graph : nx.Graph
        The cavity subgraph
    cavity_index : int
        Index of the cavity
    output_path : str or Path
        Path to save the SVG file (if directory, will create cavity_{index}.svg)
    exclude_atoms : bool, optional
        If True, exclude atom nodes (default: False, atoms are shown)
    layout : str, optional
        Graphviz layout engine (default: 'sfdp', scalable force-directed)
    dpi : int, optional
        DPI for output (default: 300). Size is auto-calculated to fit all nodes.
    spring_constant : float, optional
        Spring constant K for sfdp layout - higher values increase edge forces (default: 5.0)
    repulsive_force : float, optional
        Repulsive force for sfdp layout - higher values increase node separation (default: 2.0)
        
    Returns
    -------
    str
        Absolute path to created file
    """
    if not PYGRAPHVIZ_AVAILABLE:
        raise ImportError("pygraphviz is required. Install with: pip install pygraphviz")
    
    path = Path(output_path)
    if path.is_dir() or not path.suffix:
        path.mkdir(parents=True, exist_ok=True)
        filename = path / f"cavity_{cavity_index}.svg"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        filename = path
    
    # Create a copy and filter if needed
    G = cavity_graph.copy()
    if exclude_atoms:
        atom_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'atom']
        G.remove_nodes_from(atom_nodes)
    
    if len(G.nodes()) == 0:
        raise ValueError("Cavity graph has no nodes after filtering")
    
    # Convert NetworkX graph to pygraphviz AGraph
    A = to_agraph(G)
    
    # Set graph attributes - no background, auto-size
    # For sfdp layout, use specific attributes for better results
    graph_attrs = {
        'dpi': str(dpi),
        'overlap': 'false',
        'splines': 'curved',  # Curved edges for better visualization
        'pad': '0.5',
        # No bgcolor - transparent
        # No size - auto-size
    }
    
    # Add sfdp-specific attributes if using sfdp layout
    if layout == 'sfdp':
        graph_attrs.update({
            'K': str(spring_constant),  # Spring constant - higher = stronger edge forces (pulls nodes together)
            'repulsiveforce': str(repulsive_force),  # Repulsive force power - higher = more repulsion, more node separation
            'beautify': 'true',  # Draw leaf nodes uniformly in a circle around root nodes
            'overlap_scaling': '4',  # Scale to reduce overlap
            'overlap_shrink': 'true',  # Compression pass
            'smoothing': 'triangle',  # Smoothing algorithm
        })
    
    A.graph_attr.update(graph_attrs)
    
    # Set default node style
    A.node_attr.update({
        'fontname': 'Arial',
        'fontsize': '14',  # Bigger font
        'fontcolor': '#1e293b',  # Dark text for transparent/light background
        'fontweight': 'bold',  # Bold text
        'style': 'filled',
        'penwidth': '2'
    })
    
    # Set default edge style
    A.edge_attr.update({
        'penwidth': '3.0',
        'color': '#94a3b8'  # Lighter gray for better visibility
    })
    
    # Set node attributes
    for node_id, data in G.nodes(data=True):
        label = _create_node_label(node_id, data)
        color = _get_node_color(data)
        size_val = _get_node_size(data)
        shape = _get_node_shape(data)
        
        # Map pyvis 'dot' to graphviz 'circle'
        if shape == 'dot':
            shape = 'circle'
        
        # Calculate node size
        node_size = max(size_val / 15.0, 0.3)  # Minimum 0.3 inches
        
        # Special handling for X-site atoms - add X to label
        if data.get('node_type') == 'atom' and data.get('is_X', False):
            label = f"{label} ✕"
        
        # Set node attributes
        node = A.get_node(str(node_id))
        node.attr['label'] = label
        node.attr['fillcolor'] = color
        node.attr['shape'] = shape
        
        # For rectangles (layers), make them wider than tall
        if shape == 'box':
            node.attr['width'] = str(node_size * 1.5)  # Wider
            node.attr['height'] = str(node_size)  # Taller
        else:
            node.attr['width'] = str(node_size)
            node.attr['height'] = str(node_size)
        
        node.attr['fontcolor'] = '#1e293b'
        node.attr['fontsize'] = '14'  # Bigger font
        node.attr['fontweight'] = 'bold'  # Bold text
    
    # Set edge attributes
    for u, v, data in G.edges(data=True):
        edge_color = _get_edge_color(data)
        edge_style = _get_edge_style(data)
        
        edge = A.get_edge(str(u), str(v))
        edge.attr['color'] = edge_color
        edge.attr['style'] = edge_style
        edge.attr['penwidth'] = '3.0'
        
        # Add distance label if available
        if 'distance' in data:
            distance = data['distance']
            edge.attr['label'] = f"{distance:.3f}Å"
            edge.attr['fontsize'] = '10'
            edge.attr['fontcolor'] = '#1e293b'
    
    # Render and save as SVG
    A.draw(str(filename), format='svg', prog=layout)
    
    # Post-process SVG to remove white background polygon
    _remove_svg_background(str(filename))
    
    return str(filename.absolute())

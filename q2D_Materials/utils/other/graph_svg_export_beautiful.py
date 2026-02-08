"""
Beautiful publication-ready SVG graph export with pyvis-inspired styling.

This module provides functionality to export NetworkX graphs to SVG format
with beautiful styling inspired by pyvis, including property panels, better layouts,
and modern color schemes.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Ellipse, RegularPolygon, FancyArrowPatch
from matplotlib.collections import LineCollection
from pathlib import Path
from typing import Union, Dict, Any, Optional, List, Tuple
import numpy as np
from ase.data import vdw_radii, atomic_numbers


def _get_element_color(symbol: str) -> str:
    """Get hex color for an element symbol - matching layer_arquitect.html palette."""
    element_colors = {
        # B-site (Blue)
        'Pb': '#3b82f6', 'Sn': '#3b82f6', 'Ge': '#3b82f6', 'Ti': '#3b82f6',
        'Zr': '#3b82f6', 'Hf': '#3b82f6', 'Nb': '#3b82f6', 'Ta': '#3b82f6',
        # X-site (Red)
        'I': '#ef4444', 'Br': '#ef4444', 'Cl': '#ef4444', 'F': '#ef4444', 'O': '#ef4444',
        # A-site (Green)
        'Cs': '#10b981', 'Rb': '#10b981', 'K': '#10b981',
        # Organic
        'C': '#64748b', 'N': '#8b5cf6', 'H': '#e2e8f0', 'S': '#fb923c',
    }
    return element_colors.get(symbol, '#94a3b8')


def _get_node_color(node_data: Dict[str, Any]) -> str:
    """Get color for a node based on its type and attributes."""
    node_type = node_data.get('node_type', 'unknown')
    
    if node_type == 'atom':
        symbol = node_data.get('symbol', '')
        return _get_element_color(symbol) if symbol else '#94a3b8'
    elif node_type == 'octahedron':
        return '#3b82f6'  # Blue (B-site)
    elif node_type == 'layer':
        return '#83D3DC'  # Teal (Layer Architect grid color)
    elif node_type == 'a_site':
        return '#10b981'  # Green (A-site)
    elif node_type == 'spacer':
        return '#f97316'  # Orange (Spacer)
    elif node_type == 'cavity':
        return '#f59e0b'  # Amber/Yellow
    elif node_type == 'molecule':
        return '#f97316'  # Orange
    elif node_type in ['cage', 'half_cage']:
        return '#f59e0b'  # Amber/Yellow (same as cavity)
    elif node_type == 'anchor':
        return '#E91E63'  # Pink
    return '#94a3b8'


def _get_node_size(node_data: Dict[str, Any]) -> float:
    """Get size for a node based on its type."""
    node_type = node_data.get('node_type', 'unknown')
    # Sizes are relative scaling factors for patches
    if node_type == 'atom':
        symbol = node_data.get('symbol', '')
        radius = 1.7
        if symbol in atomic_numbers:
            r = vdw_radii[atomic_numbers[symbol]]
            if not np.isnan(r):
                radius = r
        return radius * 2000.0
    elif node_type == 'octahedron':
        return 5400
    elif node_type == 'layer':
        return 7200
    elif node_type in ['a_site', 'spacer', 'molecule']:
        return 6000
    elif node_type == 'cavity':
        return 4800
    elif node_type in ['cage', 'half_cage']:
        return 4800
    elif node_type == 'anchor':
        return 3600
    elif node_type == 'structure':
        return 9000
    return 3000


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
        return f"L{layer_id}"
    elif node_type in ['a_site', 'spacer', 'molecule']:
        formula = node_data.get('formula', '')
        if formula and len(formula) > 10:
            return formula[:10] + '...'
        return formula or node_type
    elif node_type == 'cavity':
        cav_idx = node_id.split('_')[-1] if '_' in node_id else node_id
        return f"Cav{cav_idx}"
    
    return str(node_id).replace('_', ' ')[:15]


def _add_property_panel(ax, node_id: str, node_data: Dict[str, Any], 
                        panel_x: float, panel_y: float, panel_width: float,
                        anchor: str = 'top'):
    """Add a property panel for a node."""
    # Filter out large/complex properties
    skip_keys = {'position', 'positions', 'neighbors', 'subgraph', 'graph'}
    
    properties = []
    for key, value in sorted(node_data.items()):
        if key in skip_keys:
            continue
        
        # Format value
        if isinstance(value, (list, tuple)):
            if len(value) > 3:
                val_str = f"[{len(value)} items]"
            else:
                val_str = str(value)[:30]
        elif isinstance(value, dict):
            val_str = f"{{{len(value)} keys}}"
        elif isinstance(value, float):
            val_str = f"{value:.3f}"
        elif isinstance(value, (int, str, bool)):
            val_str = str(value)[:30]
        else:
            val_str = str(type(value).__name__)
        
        properties.append(f"{key}: {val_str}")
    
    if not properties:
        return
    
    # Draw panel background (white with border for any background)
    panel_height = min(0.15 + len(properties) * 0.035, 0.5)
    
    if anchor == 'bottom':
        y_start = panel_y
        top_y = panel_y + panel_height
    else:
        y_start = panel_y - panel_height
        top_y = panel_y
        
    panel = FancyBboxPatch(
        (panel_x, y_start),
        panel_width, panel_height,
        boxstyle="round,pad=0.01",
        facecolor='white',
        edgecolor='#64748b',
        linewidth=2,
        alpha=0.92,
        transform=ax.transAxes,
        zorder=1000
    )
    ax.add_patch(panel)
    
    # Add title
    label = _create_node_label(node_id, node_data)
    ax.text(
        panel_x + panel_width/2, top_y - 0.02,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        color='#1e293b',
        ha='center',
        va='top',
        zorder=1001
    )
    
    # Add properties
    y_offset = 0.05
    for prop in properties[:8]:  # Limit to 8 properties
        ax.text(
            panel_x + 0.01, top_y - y_offset,
            prop,
            transform=ax.transAxes,
            fontsize=12,
            color='#334155',
            ha='left',
            va='top',
            family='monospace',
            zorder=1001
        )
        y_offset += 0.025


def _add_legend_panel(ax, graph: nx.Graph):
    """Add a beautiful legend panel."""
    # Collect unique node and edge types
    node_types_in_graph = set()
    edge_types_in_graph = set()
    
    for node, data in graph.nodes(data=True):
        node_types_in_graph.add(data.get('node_type', 'unknown'))
    
    for u, v, data in graph.edges(data=True):
        edge_types_in_graph.add(data.get('edge_type', 'unknown'))
    
    # Node types legend
    node_type_order = ['octahedron', 'layer', 'a_site', 'spacer', 'cavity', 'molecule', 'atom']
    node_legend_data = []
    
    for node_type in node_type_order:
        if node_type in node_types_in_graph:
            dummy_data = {'node_type': node_type, 'symbol': 'X'}
            color = _get_node_color(dummy_data)
            node_legend_data.append((node_type.replace('_', ' ').title(), color))
    
    # Draw node legend panel (white background)
    if node_legend_data:
        panel_height = 0.05 + len(node_legend_data) * 0.03
        panel = FancyBboxPatch(
            (0.02, 0.98 - panel_height),
            0.15, panel_height,
            boxstyle="round,pad=0.01",
            facecolor='white',
            edgecolor='#64748b',
            linewidth=2,
            alpha=0.92,
            transform=ax.transAxes,
            zorder=1000
        )
        ax.add_patch(panel)
        
        # Title
        ax.text(
            0.095, 0.97,
            'Node Types',
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            color='#1e293b',
            ha='center',
            va='top',
            zorder=1001
        )
        
        # Legend items
        y_pos = 0.95
        for label, color in node_legend_data:
            # Draw circle
            circle = Circle(
                (0.035, y_pos - 0.01),
                0.008,
                facecolor=color,
                edgecolor='#1e293b',
                linewidth=1.5,
                transform=ax.transAxes,
                zorder=1001
            )
            ax.add_patch(circle)
            
            # Draw label
            ax.text(
                0.055, y_pos - 0.01,
                label,
                transform=ax.transAxes,
                fontsize=12,
                color='#334155',
                ha='left',
                va='center',
                zorder=1001
            )
            y_pos -= 0.03
    
    # Edge types legend
    edge_type_order = ['contains', 'bonded_to', 'shares_atoms', 'is_contained_in', 
                      'proximity', 'cavity_octahedron']
    edge_legend_data = []
    
    line_styles = {
        'contains': '-',
        'bonded_to': '-',
        'shares_atoms': '--',
        'is_contained_in': '-',
        'proximity': ':',
        'cavity_octahedron': '-'
    }
    
    for edge_type in edge_type_order:
        if edge_type in edge_types_in_graph:
            dummy_edge_data = {'edge_type': edge_type}
            color = _get_edge_color(dummy_edge_data)
            linestyle = line_styles.get(edge_type, '-')
            edge_legend_data.append((edge_type.replace('_', ' ').title(), color, linestyle))
    
    # Draw edge legend panel (white background)
    if edge_legend_data:
        panel_height = 0.05 + len(edge_legend_data) * 0.03
        panel = FancyBboxPatch(
            (0.83, 0.98 - panel_height),
            0.15, panel_height,
            boxstyle="round,pad=0.01",
            facecolor='white',
            edgecolor='#64748b',
            linewidth=2,
            alpha=0.92,
            transform=ax.transAxes,
            zorder=1000
        )
        ax.add_patch(panel)
        
        # Title
        ax.text(
            0.905, 0.97,
            'Edge Types',
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            color='#1e293b',
            ha='center',
            va='top',
            zorder=1001
        )
        
        # Legend items
        y_pos = 0.95
        for label, color, linestyle in edge_legend_data:
            # Draw line
            ax.plot(
                [0.845, 0.885], [y_pos - 0.01, y_pos - 0.01],
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                transform=ax.transAxes,
                zorder=1001
            )
            
            # Draw label
            ax.text(
                0.895, y_pos - 0.01,
                label,
                transform=ax.transAxes,
                fontsize=12,
                color='#334155',
                ha='left',
                va='center',
                zorder=1001
            )
            y_pos -= 0.03


def _add_stats_panel(ax, graph: nx.Graph):
    """Add graph statistics panel in bottom right."""
    stats = [
        f"Nodes: {len(graph.nodes())}",
        f"Edges: {len(graph.edges())}",
    ]
    
    # Panel config
    panel_width = 0.15
    panel_height = 0.05 + len(stats) * 0.035
    panel_x = 0.83
    panel_y = 0.02  # Bottom edge
    
    # Draw panel
    panel = FancyBboxPatch(
        (panel_x, panel_y),
        panel_width, panel_height,
        boxstyle="round,pad=0.01",
        facecolor='white',
        edgecolor='#64748b',
        linewidth=2,
        alpha=0.92,
        transform=ax.transAxes,
        zorder=1000
    )
    ax.add_patch(panel)
    
    # Title
    top_y = panel_y + panel_height
    ax.text(
        panel_x + panel_width/2, top_y - 0.02,
        'Graph Stats',
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        color='#1e293b',
        ha='center',
        va='top',
        zorder=1001
    )
    
    # Items
    y_offset = 0.05
    for stat in stats:
        ax.text(
            panel_x + 0.02, top_y - y_offset,
            stat,
            transform=ax.transAxes,
            fontsize=12,
            color='#334155',
            ha='left',
            va='top',
            family='monospace',
            zorder=1001
        )
        y_offset += 0.025


def _add_structure_info_panel(ax, graph: nx.Graph):
    """Add structure info panel in bottom left."""
    # Count node types
    counts = {}
    for n, d in graph.nodes(data=True):
        nt = d.get('node_type', 'unknown')
        counts[nt] = counts.get(nt, 0) + 1
    
    properties = []
    if hasattr(graph, 'name') and graph.name:
        properties.append(f"Name: {graph.name}")
    if hasattr(graph, 'graph') and 'formula' in graph.graph:
        properties.append(f"Formula: {graph.graph['formula']}")
        
    for nt, count in sorted(counts.items()):
        properties.append(f"{nt.replace('_', ' ').title()}: {count}")
        
    # Use generic property panel function
    # Create a dummy node data dict to reuse the function
    dummy_data = {p.split(':')[0]: p.split(':')[1].strip() for p in properties}
    
    # Manually draw it to control title
    _add_property_panel(
        ax, 
        "Structure Info", 
        dummy_data, 
        0.02, 0.02, 0.20, 
        anchor='bottom'
    )


def _draw_node_shape(ax, x, y, node_type, size, color, alpha=1.0, edgecolor='#1e293b', zorder=3, scale_divisor=600.0):
    """Draw specific shape for node type."""
    # Scale factor for converting abstract size to data coordinates
    # Assuming layout is roughly -1 to 1, sizes like 300-1000 need to be scaled down
    s = size / scale_divisor 
    
    p = None
    if node_type == 'atom':
        # Circle
        p = Circle((x, y), radius=s, facecolor=color, edgecolor=edgecolor, linewidth=2, alpha=alpha, zorder=zorder)
    elif node_type == 'octahedron':
        # Rhombus (Diamond)
        p = RegularPolygon((x, y), numVertices=4, radius=s*1.6, orientation=0, 
                           facecolor=color, edgecolor=edgecolor, linewidth=2, alpha=alpha, zorder=zorder)
    elif node_type == 'layer':
        # Rectangle
        w, h = s * 4.0, s * 2.0
        p = Rectangle((x - w/2, y - h/2), w, h, facecolor=color, edgecolor=edgecolor, linewidth=2, alpha=alpha, zorder=zorder)
    elif node_type in ['molecule', 'spacer']:
        # Ellipse
        w, h = s * 4.0, s * 2.0
        p = Ellipse((x, y), width=w, height=h, facecolor=color, edgecolor=edgecolor, linewidth=2, alpha=alpha, zorder=zorder)
    elif node_type == 'structure':
        # Square
        w = s * 3.0
        p = Rectangle((x - w/2, y - w/2), w, w, facecolor=color, edgecolor=edgecolor, linewidth=2, alpha=alpha, zorder=zorder)
    elif node_type == 'cavity':
        # Dashed Circle
        p = Circle((x, y), radius=s*1.3, facecolor=color, edgecolor=edgecolor, linewidth=2, linestyle='--', alpha=alpha, zorder=zorder)
    elif node_type == 'half_cage':
        # Triangle
        p = RegularPolygon((x, y), numVertices=3, radius=s*1.5, orientation=0, 
                           facecolor=color, edgecolor=edgecolor, linewidth=2, alpha=alpha, zorder=zorder)
    elif node_type == 'cage':
        # Square
        w = s * 2.5
        p = Rectangle((x - w/2, y - w/2), w, w, facecolor=color, edgecolor=edgecolor, linewidth=2, alpha=alpha, zorder=zorder)
    elif node_type == 'anchor':
        # Star (5 points)
        r_outer = s * 1.5
        r_inner = s * 0.6
        angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, 11)
        verts = [(x + (r_outer if i % 2 == 0 else r_inner) * np.cos(a), 
                  y + (r_outer if i % 2 == 0 else r_inner) * np.sin(a)) for i, a in enumerate(angles[:-1])]
        p = mpatches.Polygon(verts, closed=True, facecolor=color, edgecolor=edgecolor, linewidth=2, alpha=alpha, zorder=zorder)
    else:
        # Default Circle
        p = Circle((x, y), radius=s, facecolor=color, edgecolor=edgecolor, linewidth=2, alpha=alpha, zorder=zorder)
    
    if p:
        ax.add_patch(p)


def _draw_curved_edge(ax, u_pos, v_pos, color, linewidth, linestyle, rad=0.1):
    """Draw a curved edge."""
    e = FancyArrowPatch(
        u_pos, v_pos,
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=0.6,
        zorder=1,
        arrowstyle='-',
        shrinkA=0, shrinkB=0
    )
    ax.add_patch(e)


def export_structure_svg_beautiful(
    graph: nx.Graph,
    output_path: Union[str, Path],
    exclude_atoms: bool = True,
    show_properties: bool = True,
    selected_node: Optional[str] = None,
    figsize: Tuple[float, float] = (20, 14),
    dpi: int = 300,
    layout: str = 'spring',
) -> str:
    """
    Export structure graph to beautiful SVG with pyvis-inspired styling.
    
    Parameters
    ----------
    graph : nx.Graph
        The structural graph from analyzer.get_graph()
    output_path : str or Path
        Path to save SVG file
    exclude_atoms : bool, optional
        If True (default), exclude atom nodes for cleaner visualization
    show_properties : bool, optional
        If True (default), show property panel for selected node
    selected_node : str, optional
        Node ID to show properties for. If None, shows properties for a central node
    figsize : tuple, optional
        Figure size in inches (default: 20x14 for publication)
    dpi : int, optional
        DPI for output (default: 300 for publication)
    layout : str, optional
        Layout algorithm: 'spring' (default), 'kamada_kawai', or 'circular'
        
    Returns
    -------
    str
        Absolute path to created SVG file
    """
    path = Path(output_path)
    if not path.suffix:
        path = path.with_suffix('.svg')
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a copy and filter if needed
    G = graph.copy()
    
    if exclude_atoms:
        atom_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'atom']
        G.remove_nodes_from(atom_nodes)
    
    if len(G.nodes()) == 0:
        raise ValueError("Graph has no nodes after filtering")
    
    # Compute layout with better spacing (like pyvis)
    # Use higher k value for more separation between nodes
    if layout == 'kamada_kawai':
        try:
            pos = nx.kamada_kawai_layout(G, scale=170.0)
        except:
            pos = nx.spring_layout(G, k=55.0, scale=170.0, iterations=1000, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G, scale=170.0)
    else:  # spring (default - best for showing connections)
        # Higher k = more spacing, more iterations = better convergence
        pos = nx.spring_layout(G, k=55.0, scale=170.0, iterations=1000, seed=42)
    
    # Create figure with TRANSPARENT background for use in any document
    fig, ax = plt.subplots(figsize=figsize, facecolor='none', edgecolor='none')
    ax.set_facecolor('none')
    
    # Draw edges with curved lines
    for u, v, edge_data in G.edges(data=True):
        u_pos = pos[u]
        v_pos = pos[v]
        
        color = _get_edge_color(edge_data)
        edge_type = edge_data.get('edge_type', 'unknown')
        
        # Vary line style and width by edge type
        linestyle = '-'
        linewidth = 10.0
        if edge_type == 'shares_atoms':
            linestyle = '--'
            linewidth = 12.0
        elif edge_type == 'proximity':
            linestyle = ':'
            linewidth = 6.0
        
        _draw_curved_edge(ax, u_pos, v_pos, color, linewidth, linestyle, rad=0.1)
    
    # Draw nodes with specific shapes
    for node in G.nodes():
        node_data = G.nodes[node]
        color = _get_node_color(node_data)
        size = _get_node_size(node_data)
        node_type = node_data.get('node_type', 'unknown')
        x, y = pos[node]
        
        # Draw glow
        _draw_node_shape(ax, x, y, node_type, size*1.4, color, alpha=0.15, edgecolor='none', zorder=2, scale_divisor=600.0)
        # Draw main node
        _draw_node_shape(ax, x, y, node_type, size, color, alpha=0.95, edgecolor='#1e293b', zorder=3, scale_divisor=600.0)
    
    # Draw node labels with better contrast (works on any background)
    for node in G.nodes():
        node_data = G.nodes[node]
        label = _create_node_label(node, node_data)
        x, y = pos[node]
        
        # Text with semi-transparent background
        ax.text(x, y, label,
               fontsize=32, fontweight='bold', color='#1e293b',
               ha='center', va='center', zorder=4,
               bbox=dict(boxstyle='round,pad=0.35', facecolor='white', 
                        edgecolor='#64748b', linewidth=1.5, alpha=0.85))
    
    # Set limits explicitly to ensure everything is visible (patches don't auto-scale)
    x_values = [p[0] for p in pos.values()]
    y_values = [p[1] for p in pos.values()]
    if x_values and y_values:
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        
        w = x_max - x_min
        h = y_max - y_min
        if w == 0: w = 1.0
        if h == 0: h = 1.0
        
        # Add 30% margin for labels and large nodes
        pad_x = w * 0.3
        pad_y = h * 0.3
        
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

    # Add panels
    if show_properties:
        # Bottom Left: Structure Info
        _add_structure_info_panel(ax, G)
        
        # Bottom Right: Stats
        _add_stats_panel(ax, G)
    
    # Add legend
    _add_legend_panel(ax, G)
    
    # Clean up axes
    ax.axis('off')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(str(path), format='svg', dpi=dpi, bbox_inches='tight', 
                facecolor='none', edgecolor='none', transparent=True)
    plt.close(fig)
    
    return str(path.absolute())


def export_subgraph_svg_beautiful(
    subgraph: nx.Graph,
    output_path: Union[str, Path],
    title: str = "Graph Subgraph",
    show_properties: bool = True,
    figsize: Tuple[float, float] = (16, 12),
    dpi: int = 300,
    layout: str = 'kamada_kawai',
    node_size_multiplier: float = 1.0,
) -> str:
    """
    Export a subgraph (e.g., cavity) to beautiful SVG.
    
    Parameters
    ----------
    subgraph : nx.Graph
        The subgraph to export
    output_path : str or Path
        Path to save SVG file
    title : str, optional
        Title for the visualization
    show_properties : bool, optional
        If True, show property panels
    figsize : tuple, optional
        Figure size in inches
    dpi : int, optional
        DPI for output
    layout : str, optional
        Layout algorithm: 'spring', 'kamada_kawai' (default)
    node_size_multiplier : float, optional
        Multiplier for node sizes (default: 1.0). Use 1.3 for A-site cavities.
        
    Returns
    -------
    str
        Absolute path to created SVG file
    """
    path = Path(output_path)
    if not path.suffix:
        path = path.with_suffix('.svg')
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute layout with better spacing for cavities
    if layout == 'kamada_kawai':
        try:
            pos = nx.kamada_kawai_layout(subgraph, scale=300.0)
        except:
            pos = nx.spring_layout(subgraph, k=50.0, scale=300.0, iterations=2000, seed=42)
    else:
        pos = nx.spring_layout(subgraph, k=50.0, scale=300.0, iterations=2000, seed=42)
    
    # Create figure with TRANSPARENT background
    fig, ax = plt.subplots(figsize=figsize, facecolor='none', edgecolor='none')
    ax.set_facecolor('none')
    
    # Draw edges
    for u, v, edge_data in subgraph.edges(data=True):
        u_pos = pos[u]
        v_pos = pos[v]
        
        color = _get_edge_color(edge_data)
        edge_type = edge_data.get('edge_type', 'unknown')
        
        linewidth = 6.0
        linestyle = '-'
        if edge_type == 'shares_atoms':
            linestyle = '--'
            linewidth = 7.0
        elif edge_type == 'proximity':
            linestyle = ':'
            linewidth = 4.0
        
        _draw_curved_edge(ax, u_pos, v_pos, color, linewidth, linestyle, rad=0.1)
    
    # Draw nodes
    for node in subgraph.nodes():
        node_data = subgraph.nodes[node]
        color = _get_node_color(node_data)
        size = _get_node_size(node_data) * node_size_multiplier
        node_type = node_data.get('node_type', 'unknown')
        x, y = pos[node]
        
        # Draw glow
        _draw_node_shape(ax, x, y, node_type, size*1.4, color, alpha=0.15, edgecolor='none', zorder=2, scale_divisor=400.0)
        # Draw main node
        _draw_node_shape(ax, x, y, node_type, size, color, alpha=0.95, edgecolor='#1e293b', zorder=3, scale_divisor=400.0)
    
    # Draw labels (works on any background)
    for node in subgraph.nodes():
        node_data = subgraph.nodes[node]
        label = _create_node_label(node, node_data)
        x, y = pos[node]
        
        ax.text(x, y, label,
               fontsize=12, fontweight='bold', color='#1e293b',
               ha='center', va='center', zorder=4,
               bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                        edgecolor='#64748b', linewidth=1.5, alpha=0.85))
    
    # Set limits explicitly to ensure everything is visible
    x_values = [p[0] for p in pos.values()]
    y_values = [p[1] for p in pos.values()]
    if x_values and y_values:
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        
        w = x_max - x_min
        h = y_max - y_min
        if w == 0: w = 1.0
        if h == 0: h = 1.0
        
        # Add 25% margin for labels and large nodes (more space for cavities)
        pad_x = w * 0.25
        pad_y = h * 0.25
        
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

    # Add panels
    if show_properties:
        # Bottom Left: Structure Info
        _add_structure_info_panel(ax, subgraph)
        
        # Bottom Right: Stats
        _add_stats_panel(ax, subgraph)
    
    ax.axis('off')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(str(path), format='svg', dpi=dpi, bbox_inches='tight', 
                facecolor='none', edgecolor='none', transparent=True)
    plt.close(fig)
    
    return str(path.absolute())

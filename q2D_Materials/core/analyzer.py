"""
Refactored q2D Analyzer - Main analysis class using modular components.

This is the main analyzer class that orchestrates the octahedral analysis
using specialized modules for geometry, angular analysis, and connectivity.
"""

import pandas as pd
import numpy as np
import os
from ase.io import read
from ase import Atoms
from ..utils.geometry import _graph_inorganic_ontology
from ..utils.cell_analysis import _get_cell_properties

# Use matplotlib without x server
import matplotlib
matplotlib.use('Agg')


class q2D_analyzer:
    """
    Refactored analyzer for 2D quantum materials using VASP results.
    
    Creates a unified ontology:
    Experiment -> Cell Properties -> Octahedra (with atomic indices and properties)
    -> Vector Analysis (if salt structure available)
    
    Uses modular components for:
    - Geometry calculations (GeometryCalculator)
    - Angular analysis (AngularAnalyzer) 
    - Connectivity analysis (ConnectivityAnalyzer)
    - Vector analysis (VectorAnalyzer)
    """
    
    def __init__(self, file_path=None, b='Pb', x='Cl', cutoff_ref_ligand=3.5):
        """
        Initialize the q2D_analyzer.
        
        Parameters:
        file_path (str, optional): Path to the VASP file to load initially
        b (str): Central atom symbol (e.g., 'Pb')
        x (str): Ligand atom symbol (e.g., 'Cl')
        cutoff_ref_ligand (float): Distance cutoff for identifying ligands
        """
        # Basic properties
        self.file_path = file_path
        self.experiment_name = file_path.split('/')[-1].split('.')[0]
        self.cell = read(file_path)
        
        # Ensure PBC is enabled for crystal structures
        self.cell.pbc = True

        # The unified ontology
        self.unified_ontology = self.get_unified_ontology()
        
    def get_unified_ontology(self):
        """Get the unified ontology connecting cell and octahedral properties."""
        cell_properties = _get_cell_properties(self.cell)
        inorganic_ontology_graph = _graph_inorganic_ontology(
            self.cell.positions, 
            self.cell.get_chemical_symbols(), 
            cell=self.cell.get_cell()
        )
        
        # Add unit cell node to the graph with cell properties
        inorganic_ontology_graph.add_node('unit_cell', 
                                         node_type='unit_cell',
                                         **cell_properties)
        
        # Connect unit cell to all layers
        for node in inorganic_ontology_graph.nodes():
            if inorganic_ontology_graph.nodes[node].get('node_type') == 'layer':
                inorganic_ontology_graph.add_edge('unit_cell', node, edge_type='contains')
        
        return {
            "cell_properties": cell_properties,
            "inorganic_ontology": inorganic_ontology_graph
        }

    def visualize_unified_ontology(self, output_file=None):
        """
        Visualize the unified ontology graph using pyvis.
        
        Parameters:
        output_file: str - optional file path to save the HTML visualization
        
        Returns:
        pyvis.Network: the network object
        """
        from pyvis.network import Network
        
        # Get the graph
        graph = self.unified_ontology["inorganic_ontology"]
        
        # Create pyvis network
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
        
        # Define node colors and sizes by type
        node_configs = {
            'unit_cell': {'color': '#ff4444', 'size': 30, 'shape': 'star'},
            'layer': {'color': '#4444ff', 'size': 25, 'shape': 'triangle'},
            'octahedron': {'color': '#44ff44', 'size': 20, 'shape': 'diamond'},
            'atom': {'color': '#ffff44', 'size': 15, 'shape': 'dot'}
        }
        
        # Define edge colors by type
        edge_configs = {
            'contains': {'color': '#ffffff', 'width': 3},
            'contains_atom': {'color': '#ffaa44', 'width': 2},
            'has_center': {'color': '#ff44aa', 'width': 3},
            'is_center_of': {'color': '#ff44aa', 'width': 3},
            'shares_atoms': {'color': '#44ffaa', 'width': 2},
            'covalent_bond': {'color': '#ff88ff', 'width': 2},
            'hydrogen_bond': {'color': '#44aaff', 'width': 2, 'dashes': True}
        }
        
        # Add nodes
        for node in graph.nodes():
            node_type = graph.nodes[node].get('node_type', 'unknown')
            config = node_configs.get(node_type, {'color': '#888888', 'size': 10, 'shape': 'dot'})
            
            # Create label
            if node_type == 'atom':
                symbol = graph.nodes[node].get('symbol', '?')
                vasp_idx = graph.nodes[node].get('vasp_index', '?')
                label = f"{symbol}_{vasp_idx}"
            elif node_type == 'octahedron':
                central_atom = graph.nodes[node].get('central_atom', '?')
                label = f"Oct_{central_atom}"
            elif node_type == 'layer':
                position = graph.nodes[node].get('position', '?')
                label = f"Layer_{position}"
            else:
                label = node.replace('_', '\n')
            
            net.add_node(node, label=label, color=config['color'], 
                        size=config['size'], shape=config['shape'])
        
        # Add edges
        for u, v, data in graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            config = edge_configs.get(edge_type, {'color': '#888888', 'width': 1})
            
            net.add_edge(u, v, color=config['color'], width=config['width'], 
                        title=f"{edge_type}")
        
        # Configure physics
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
              "gravitationalConstant": -2000,
              "centralGravity": 0.1,
              "springLength": 200,
              "springConstant": 0.05
            }
          }
        }
        """)
        
        # Save or show
        if output_file:
            net.save_graph(output_file)
            print(f"Graph visualization saved to {output_file}")
        else:
            net.show(f"{self.experiment_name}_ontology.html")
            print(f"Graph visualization opened in browser")
        
        return net

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
from ..utils.geometry import _octahedra_ontology
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
        octahedral_graph = _octahedra_ontology(
            self.cell.positions, 
            self.cell.get_chemical_symbols(), 
            cell=self.cell.get_cell()
        )
        return {
            "cell_properties": cell_properties,
            "octahedral_graph": octahedral_graph
        }
    
    def graph_export_html(self, filename=None):
        """Export the octahedral graph to HTML format."""
        if filename is None:
            filename = f"{self.experiment_name}_octahedral_graph.html"
        
        import json
        with open(filename, 'w') as f:
            json.dump(self.unified_ontology['octahedral_graph'], f, indent=2)
        
        print(f"Octahedral graph exported to {filename}")

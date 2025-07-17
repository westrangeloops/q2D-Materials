import pandas as pd
import numpy as np
import os
from ase.io import read
from ..utils.file_handlers import vasp_load, save_vasp
from ..utils.plots import plot_gaussian_projection, plot_multi_element_comparison, save_beautiful_plot, create_summary_plot
from octadist.src.io import extract_octa
from octadist.src.calc import CalcDistortion

class q2D_analyzer:
    """
    A class for analyzing 2D quantum materials using VASP results.
    """
    
    def __init__(self, file_path=None, b='Pb', x='Cl', cutoff_ref_ligand=3.5):
        """
        Initialize the q2D_analyzer.
        
        Parameters:
        file_path (str, optional): Path to the VASP file to load initially
        """
        # Load the structure as atom object
        self.atoms = read(file_path)
        self.b = b
        self.x = x
        self.all_octahedra = self.find_all_octahedra()

    def find_all_octahedra(self):
        """
        Automatically identify all octahedra in the given ASE Atoms object.
        
        Args:
            atoms: ASE Atoms object
            central_symbols: List of atomic symbols that can be octahedral centers
            cutoff_ref_ligand: Distance cutoff for identifying ligands
            
        Returns:
            List of tuples: (central_atom_symbol, central_index, atom_octa, coord_octa)
        """
        all_octa = []
        
        # Central symbols
        central_symbols = self.B

        # Get atomic symbols and coordinates
        atom_symbols = self.atoms.get_chemical_symbols()
        coord = self.atoms.get_positions()
        
        # Find indices of potential central atoms
        central_indices = [i for i, sym in enumerate(atom_symbols) if sym in central_symbols]
        
    for ref_index in central_indices:
        # Extract octahedron around this center
        atom_octa, coord_octa = extract_octa(atom_symbols, coord, ref_index=ref_index, cutoff_ref_ligand=self.cutoff_ref_ligand)
        
        # Verify
        dist = CalcDistortion(coord_octa)
        if dist.non_octa:
            print(f"Warning: Non-octahedral structure around {atom_symbols[ref_index]} at index {ref_index}")
            continue
        
        all_octa.append((atom_symbols[ref_index], ref_index, atom_octa, coord_octa))
    
    return all_octa

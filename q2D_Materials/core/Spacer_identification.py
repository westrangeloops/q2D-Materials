"""
Refactored q2D Analyzer - Main analysis class using modular components.

This is the main analyzer class that orchestrates the octahedral analysis
using specialized modules for geometry, angular analysis, and connectivity.
"""

import pandas as pd
import numpy as np
import os
from ase.io import read
from ..utils.file_handlers import vasp_load, save_vasp
from ..utils.plots import plot_gaussian_projection, plot_multi_element_comparison, save_beautiful_plot, create_summary_plot
from ..utils.octadist.io import extract_octa
from ..utils.octadist.calc import CalcDistortion
from ..utils.octadist.linear import angle_btw_vectors

# Import our modular components
from .geometry import GeometryCalculator
from .angular_analysis import AngularAnalyzer
from .connectivity import ConnectivityAnalyzer
from .layers_analysis import LayersAnalyzer

# Use matplotlib without x server
import matplotlib
matplotlib.use('Agg')


class q2D_analyzer:
    """
    Refactored analyzer for 2D quantum materials using VASP results.
    
    Creates a unified ontology:
    Experiment -> Cell Properties -> Octahedra (with atomic indices and properties)
    
    Uses modular components for:
    - Geometry calculations (GeometryCalculator)
    - Angular analysis (AngularAnalyzer) 
    - Connectivity analysis (ConnectivityAnalyzer)
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
        self.atoms = read(file_path)
        self.b = b
        self.x = x
        self.chemical_symbols = self.atoms.get_chemical_symbols()
        self.coord = self.atoms.get_positions()
        self.cutoff_ref_ligand = cutoff_ref_ligand
        
        # Initialize modular components
        self.geometry_calc = GeometryCalculator(atoms=self.atoms)
        self.angular_analyzer = AngularAnalyzer(self.geometry_calc)
        self.connectivity_analyzer = ConnectivityAnalyzer(self.geometry_calc)
        self.layers_analyzer = LayersAnalyzer(z_window=2.0)
        
        # Initialize analysis
        self.all_octahedra = self.find_all_octahedra()
        self.ordered_octahedra = self.order_octahedra()
        
        # Create unified ontology
        self.ontology = self._create_unified_ontology()

    def find_all_octahedra(self):
        """
        Automatically identify all octahedra in the given ASE Atoms object.
        
        Returns:
            List of tuples: (central_atom_symbol, central_index, atom_octa, coord_octa)
        """
        all_octa = []
        central_symbols = [self.b]
        atom_symbols = self.chemical_symbols
        coord = self.coord
        
        # Get cell and PBC information from ASE atoms object
        cell = self.atoms.get_cell()
        pbc = self.atoms.get_pbc()
        use_pbc = np.any(pbc) and self.atoms.get_volume() > 0
        
        # Find indices of potential central atoms
        central_indices = [i for i, sym in enumerate(atom_symbols) if sym in central_symbols]
        
        for ref_index in central_indices:
            # Extract octahedron around this center with PBC support
            if use_pbc:
                atom_octa, coord_octa = extract_octa(
                    atom_symbols, coord, 
                    ref_index=ref_index, 
                    cutoff_ref_ligand=self.cutoff_ref_ligand,
                    cell=cell, pbc=pbc
                )
            else:
                atom_octa, coord_octa = extract_octa(
                    atom_symbols, coord, 
                    ref_index=ref_index, 
                    cutoff_ref_ligand=self.cutoff_ref_ligand
                )
            
            # Verify if it's a proper octahedron
            dist = CalcDistortion(coord_octa)
            if dist.non_octa:
                print(f"Warning: Non-octahedral structure around {atom_symbols[ref_index]} at index {ref_index}")
                continue
            
            all_octa.append((atom_symbols[ref_index], ref_index, atom_octa, coord_octa))
        
        return all_octa

    def order_octahedra(self):
        """
        Order octahedra systematically for comparison between structures.
        Orders by Z coordinate first, then by X, then by Y coordinates.
        
        Returns:
            List of ordered octahedra with consistent indexing
        """
        if not self.all_octahedra:
            return []
        
        # Extract central atom coordinates for sorting
        octahedra_with_coords = []
        for i, (symbol, index, atom_octa, coord_octa) in enumerate(self.all_octahedra):
            central_coord = coord_octa[0]
            octahedra_with_coords.append({
                'original_index': i,
                'symbol': symbol,
                'global_index': index,
                'atom_octa': atom_octa,
                'coord_octa': coord_octa,
                'central_coord': central_coord,
                'z': central_coord[2],
                'x': central_coord[0],
                'y': central_coord[1]
            })
        
        # Sort by Z first, then by X, then by Y
        ordered_octahedra = sorted(octahedra_with_coords, 
                                 key=lambda octa: (octa['z'], octa['x'], octa['y']))
        
        # Add ordered index for reference
        for i, octa in enumerate(ordered_octahedra):
            octa['ordered_index'] = i
        
        return ordered_octahedra

    def _get_cell_properties(self):
        """
        Extract cell properties in a standardized format.
        
        Returns:
            dict: Cell properties including lattice parameters and composition
        """
        cell_lengths_angles = self.atoms.get_cell_lengths_and_angles()
        a, b, c, alpha, beta, gamma = cell_lengths_angles
        
        return {
            "lattice_parameters": {
                "A": float(a),
                "B": float(b), 
                "C": float(c),
                "Alpha": float(alpha),
                "Beta": float(beta),
                "Gamma": float(gamma)
            },
            "composition": {
                "metal_B": str(self.b),
                "halogen_X": str(self.x),
                "number_of_atoms": len(self.atoms),
                "number_of_octahedra": len(self.ordered_octahedra)
            },
            "structure_info": {
                "cell_volume": float(self.atoms.get_volume()),
                "cutoff_ref_ligand": float(self.cutoff_ref_ligand)
            }
        }

    def _process_octahedron_data(self, octa_data):
        """
        Process individual octahedron data and calculate all properties.
        
        Parameters:
            octa_data: Dictionary containing octahedron information
            
        Returns:
            dict: Complete octahedron analysis with atomic indices and properties
        """
        coord_octa = octa_data['coord_octa']
        atom_symbols = octa_data['atom_octa']
        central_coord = np.array(octa_data['central_coord'])
        global_index = octa_data['global_index']
        
        # Calculate distortion parameters
        dist_calc = CalcDistortion(coord_octa)
        
        # Identify axial and equatorial positions using geometry calculator
        axial_positions, equatorial_positions = self.geometry_calc.identify_axial_equatorial_positions(
            coord_octa, atom_symbols, central_coord
        )
        
        # Get global indices for all ligand atoms using PBC-aware distance calculation
        # Process each ligand exactly once to avoid duplicates
        ligand_global_indices = []
        axial_global_indices = []
        equatorial_global_indices = []
        
        # Get cell and PBC information for distance calculations
        cell = self.atoms.get_cell()
        pbc = self.atoms.get_pbc()
        use_pbc = np.any(pbc) and self.atoms.get_volume() > 0
        
        # Process all ligands from coord_octa (skip index 0 which is central atom)
        for i in range(1, len(coord_octa)):
            ligand_coord = coord_octa[i]
            ligand_symbol = atom_symbols[i]
            
            # Find the global index using PBC-aware distance calculation
            global_atom_index = self._find_global_index_pbc_aware(
                ligand_coord, central_coord, ligand_symbol, cell, pbc, use_pbc
            )
            
            if global_atom_index is not None:
                ligand_global_indices.append(global_atom_index)
                
                # Check if this ligand is axial or equatorial
                # Find matching position in axial_positions
                is_axial = False
                is_equatorial = False
                
                for pos in axial_positions:
                    if pos['local_index'] == i - 1:  # -1 because local_index doesn't include central atom
                        axial_global_indices.append(global_atom_index)
                        is_axial = True
                        break
                
                if not is_axial:
                    for pos in equatorial_positions:
                        if pos['local_index'] == i - 1:  # -1 because local_index doesn't include central atom
                            equatorial_global_indices.append(global_atom_index)
                            is_equatorial = True
                            break
                
                # If not found in either, this indicates an issue with octahedron identification
                if not is_axial and not is_equatorial:
                    print(f"Warning: Ligand {i} ({ligand_symbol}) at global index {global_atom_index} "
                          f"not classified as axial or equatorial for octahedron {global_index}")
            else:
                # If we can't find the global index, this octahedron is incomplete
                print(f"Error: Could not find global index for ligand {i} ({ligand_symbol}) "
                      f"at coord {ligand_coord} for central atom {global_index}")
                print(f"This octahedron will be marked as non-octahedral.")
                # Mark as non-octahedral if we can't find all ligands
                dist_calc.non_octa = True
        
        # Verify we have exactly 6 ligands
        if len(ligand_global_indices) != 6:
            print(f"Warning: Found {len(ligand_global_indices)} ligands instead of 6 for octahedron {global_index}")
            dist_calc.non_octa = True
        
        # Check for duplicates
        if len(set(ligand_global_indices)) != len(ligand_global_indices):
            print(f"Error: Duplicate ligand indices found for octahedron {global_index}: {ligand_global_indices}")
            print(f"Unique indices: {list(set(ligand_global_indices))}")
            dist_calc.non_octa = True
        
        # Sort ligand indices for consistency
        ligand_global_indices.sort()
        axial_global_indices.sort()
        equatorial_global_indices.sort()
        
        # Create angles dictionary with atom indices
        cis_angles_with_indices = {}
        trans_angles_with_indices = {}
        
        if hasattr(dist_calc, 'cis_angle'):
            for i, angle_val in enumerate(list(dist_calc.cis_angle)):
                cis_angles_with_indices[f"cis_angle_{i+1}"] = {
                    "value": float(angle_val),
                    "atom_indices": f"See ligand_global_indices for atoms involved"
                }
        
        if hasattr(dist_calc, 'trans_angle'):
            for i, angle_val in enumerate(list(dist_calc.trans_angle)):
                trans_angles_with_indices[f"trans_angle_{i+1}"] = {
                    "value": float(angle_val),
                    "atom_indices": f"See ligand_global_indices for atoms involved"
                }
        
        # Create bond distances with atom indices
        bond_distances_with_indices = {}
        bond_distance_list = dist_calc.bond_dist.tolist()
        for i, distance in enumerate(bond_distance_list):
            ligand_idx = ligand_global_indices[i] if i < len(ligand_global_indices) else None
            bond_distances_with_indices[f"bond_{i+1}"] = {
                "distance": float(distance),
                "central_atom_index": int(global_index),
                "ligand_atom_index": int(ligand_idx) if ligand_idx is not None else None
            }
        
        # Create eight_theta data with indices
        eight_theta_with_indices = {}
        if hasattr(dist_calc, 'eight_theta'):
            for i, theta_val in enumerate(list(dist_calc.eight_theta)):
                eight_theta_with_indices[f"theta_face_{i+1}"] = {
                    "value": float(theta_val),
                    "description": f"Theta angle for octahedral face {i+1}"
                }
        
        # Calculate comprehensive angular analysis using angular analyzer
        angular_analysis = self.angular_analyzer.calculate_angular_analysis(
            coord_octa, axial_positions, equatorial_positions, central_coord, 
            octa_data['ordered_index'], self.ordered_octahedra, 
            self.coord, self.chemical_symbols, self.cutoff_ref_ligand
        )
        
        return {
            "central_atom": {
                "global_index": int(global_index),
                "symbol": str(octa_data['symbol']),
                "coordinates": {
                    "x": float(central_coord[0]),
                    "y": float(central_coord[1]),
                    "z": float(central_coord[2])
                }
            },
            "ligand_atoms": {
                "axial_global_indices": [int(idx) for idx in axial_global_indices],
                "equatorial_global_indices": [int(idx) for idx in equatorial_global_indices],
                "all_ligand_global_indices": [int(idx) for idx in sorted(ligand_global_indices)]
            },
            "distortion_parameters": {
                "zeta": float(dist_calc.zeta),
                "delta": float(dist_calc.delta),
                "sigma": float(dist_calc.sigma),
                "theta_mean": float(dist_calc.theta),
                "theta_min": float(dist_calc.theta_min),
                "theta_max": float(dist_calc.theta_max),
                "eight_theta": eight_theta_with_indices
            },
            "bond_distances": bond_distances_with_indices,
            "bond_distance_analysis": {
                "mean_bond_distance": float(dist_calc.d_mean),
                "bond_distance_variance": float(np.var(dist_calc.bond_dist))
            },
            "bond_angles": {
                "cis_angles": cis_angles_with_indices,
                "trans_angles": trans_angles_with_indices
            },
            "geometric_properties": {
                "octahedral_volume": float(dist_calc.oct_vol),
                "is_octahedral": bool(not dist_calc.non_octa)
            },
            "detailed_atom_info": {
                "all_ligand_symbols": [atom_symbols[i] for i in range(1, len(atom_symbols))],
                "axial_atom_symbols": [pos['symbol'] for pos in axial_positions],
                "equatorial_atom_symbols": [pos['symbol'] for pos in equatorial_positions],
                "ligand_coordinates": [coord_octa[i].tolist() for i in range(1, len(coord_octa))]
            },
            "angular_analysis": angular_analysis
        }

    def _find_global_index_pbc_aware(self, ligand_coord, central_coord, ligand_symbol, cell, pbc, use_pbc):
        """
        Find global index of ligand atom using PBC-aware distance calculation.
        
        Parameters:
        ligand_coord: coordinates of ligand atom from octahedron extraction
        central_coord: coordinates of central atom
        ligand_symbol: expected symbol of ligand atom
        cell: unit cell matrix (not used - kept for compatibility)
        pbc: periodic boundary conditions (not used - kept for compatibility)
        use_pbc: whether to use PBC calculations (not used - kept for compatibility)
        
        Returns:
        global index of ligand atom, or None if not found
        """
        ligand_coord = np.array(ligand_coord)
        central_coord = np.array(central_coord)
        tolerance = 1e-3
        
        best_match_index = None
        min_distance = float('inf')
        
        for i, coord in enumerate(self.coord):
            # Skip if atom symbol doesn't match
            if self.chemical_symbols[i] != ligand_symbol:
                continue
            
            coord = np.array(coord)
            
            # Use geometry calculator's PBC-aware distance calculation
            distance_to_ligand = self.geometry_calc.calculate_distance(coord, ligand_coord)
            dist_to_central = self.geometry_calc.calculate_distance(coord, central_coord)
            
            if dist_to_central <= self.cutoff_ref_ligand and distance_to_ligand < min_distance:
                if distance_to_ligand < tolerance:
                    min_distance = distance_to_ligand
                    best_match_index = i
        
        return best_match_index

    def _create_unified_ontology(self):
        """
        Create the unified ontology structure:
        Experiment -> Cell Properties -> Octahedra (with atomic indices and properties)
        
        Returns:
            dict: Complete unified ontology
        """
        # Get cell properties
        cell_properties = self._get_cell_properties()
        
        # Process all octahedra
        octahedra_data = {}
        for octa_data in self.ordered_octahedra:
            octahedron_index = octa_data['ordered_index']
            octahedra_data[f"octahedron_{octahedron_index + 1}"] = self._process_octahedron_data(octa_data)
        
        # Analyze connectivity between octahedra using connectivity analyzer
        connectivity_analysis = self.connectivity_analyzer.analyze_octahedra_connectivity(
            octahedra_data, self.coord, self.chemical_symbols
        )
        
        # Analyze layers using layers analyzer with connectivity information
        layers_analysis = self.layers_analyzer.identify_layers(octahedra_data, connectivity_analysis)
        
        # Create unified structure
        unified_ontology = {
            "experiment": {
                "name": str(self.experiment_name),
                "file_path": str(self.file_path),
                "timestamp": str(pd.Timestamp.now())
            },
            "cell_properties": cell_properties,
            "octahedra": octahedra_data,
            "connectivity_analysis": connectivity_analysis,
            "layers_analysis": layers_analysis
        }
        
        return unified_ontology

    def get_ontology(self):
        """
        Get the unified ontology structure.
        
        Returns:
            dict: Complete unified ontology
        """
        return self.ontology


    def export_ontology_json(self, filename=None):
        """
        Export the unified ontology to JSON format.
        
        Parameters:
            filename (str, optional): Output filename
        """
        if filename is None:
            filename = f"{self.experiment_name}_unified_ontology.json"
        
        import json
        with open(filename, 'w') as f:
            json.dump(self.ontology, f, indent=2)
        
        print(f"Unified ontology exported to {filename}")
        return self.ontology

    def print_ontology_summary(self):
        """
        Print a summary of the ontology structure.
        """
        print(f"\n=== UNIFIED ONTOLOGY SUMMARY ===")
        print(f"Experiment: {self.ontology['experiment']['name']}")
        print(f"\nCell Properties:")
        cell_props = self.ontology['cell_properties']
        lattice = cell_props['lattice_parameters']
        print(f"  A={lattice['A']:.3f}, B={lattice['B']:.3f}, C={lattice['C']:.3f}")
        print(f"  α={lattice['Alpha']:.1f}°, β={lattice['Beta']:.1f}°, γ={lattice['Gamma']:.1f}°")
        print(f"  Volume: {cell_props['structure_info']['cell_volume']:.3f}")
        print(f"  Composition: {cell_props['composition']['metal_B']}-{cell_props['composition']['halogen_X']}")
        
        print(f"\nOctahedra Analysis:")
        for oct_key, oct_data in self.ontology['octahedra'].items():
            central = oct_data['central_atom']
            distortion = oct_data['distortion_parameters']
            ligands = oct_data['ligand_atoms']
            bond_analysis = oct_data['bond_distance_analysis']
            geometric = oct_data['geometric_properties']
            detailed = oct_data['detailed_atom_info']
            angular_data = oct_data.get('angular_analysis', {})
            
            print(f"  {oct_key}: Central atom {central['symbol']} (index {central['global_index']})")
            print(f"    Distortion: ζ={distortion['zeta']:.4f}, δ={distortion['delta']:.4f}, σ={distortion['sigma']:.4f}")
            print(f"    Theta: mean={distortion['theta_mean']:.2f}°, min={distortion['theta_min']:.2f}°, max={distortion['theta_max']:.2f}°")
            print(f"    Bond distances: mean={bond_analysis['mean_bond_distance']:.3f}Å, variance={bond_analysis['bond_distance_variance']:.6f}")
            print(f"    Volume: {geometric['octahedral_volume']:.3f}Å³, Is octahedral: {geometric['is_octahedral']}")
            print(f"    Ligand indices: {ligands['all_ligand_global_indices']}")
            print(f"    Ligand symbols: {detailed['all_ligand_symbols']}")
            
            # Add angular analysis information
            if angular_data:
                if 'axial_central_axial' in angular_data:
                    aca = angular_data['axial_central_axial']
                    print(f"    Axial-Central-Axial angle: {aca['angle_degrees']:.2f}° (deviation from 180°: {aca['deviation_from_180']:.2f}°)")
                    print(f"    Is linear: {aca['is_linear']}")
                
                if 'central_axial_central' in angular_data:
                    cac_angles = angular_data['central_axial_central']
                    if cac_angles:
                        print(f"    Central-Axial-Central bridges: {len(cac_angles)}")
                        for i, cac in enumerate(cac_angles):
                            print(f"      Bridge {i+1}: {cac['angle_degrees']:.2f}° to {cac['connected_octahedron']} via atom {cac['axial_atom_global_index']}")
                
                if 'central_equatorial_central' in angular_data:
                    cec_angles = angular_data['central_equatorial_central']
                    if cec_angles:
                        print(f"    Central-Equatorial-Central bridges: {len(cec_angles)}")
                        for i, cec in enumerate(cec_angles):
                            print(f"      Bridge {i+1}: {cec['angle_degrees']:.2f}° to {cec['connected_octahedron']} via atom {cec['equatorial_atom_global_index']}")
                
                if 'summary' in angular_data:
                    summary = angular_data['summary']
                    print(f"    Angular summary: {summary['total_axial_bridges']} axial + {summary['total_equatorial_bridges']} equatorial bridges")
                    if summary['average_central_axial_central_angle'] > 0:
                        print(f"    Average C-A-C angle: {summary['average_central_axial_central_angle']:.2f}°")
                    if summary['average_central_equatorial_central_angle'] > 0:
                        print(f"    Average C-E-C angle: {summary['average_central_equatorial_central_angle']:.2f}°")
            
            if ligands['axial_global_indices'] and ligands['equatorial_global_indices']:
                print(f"    Axial: {ligands['axial_global_indices']} ({detailed['axial_atom_symbols']})")
                print(f"    Equatorial: {ligands['equatorial_global_indices']} ({detailed['equatorial_atom_symbols']})")
            
            # Show connectivity information
            connectivity = self.ontology.get('connectivity_analysis', {})
            connections = connectivity.get('octahedra_connections', {}).get(oct_key, [])
            if connections:
                shared_info = []
                for conn in connections:
                    shared_info.append(f"{conn['connected_octahedron']} (atom {conn['shared_atom_index']}:{conn['atom_symbol']})")
                print(f"    Connected to: {', '.join(shared_info)}")
            print()
        
        # Show overall connectivity summary using connectivity analyzer
        connectivity = self.ontology.get('connectivity_analysis', {})
        if connectivity:
            print(self.connectivity_analyzer.get_connectivity_summary(connectivity))
        
        # Show layer analysis summary
        layers_analysis = self.ontology.get('layers_analysis', {})
        if layers_analysis:
            print(self.layers_analyzer.get_layer_summary())
            
    def get_layers_analysis(self):
        """
        Get the layers analysis results.
        
        Returns:
            dict: Layers analysis data with distortion statistics
        """
        return self.ontology.get('layers_analysis', {})
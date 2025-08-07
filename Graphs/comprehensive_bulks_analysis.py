#!/usr/bin/env python3
"""
Comprehensive bulk analysis script for DION-JACOBSON perovskite structures.
Processes all CONTCAR files in the BULKS directory structure and creates
comprehensive analysis folders with all structural components and properties.

Features:
- Octahedral distortion analysis
- Molecular identification and analysis  
- Vector/plane analysis of salt structures
- Penetration depth analysis for spacer molecules
- Comprehensive dataset generation for ML applications

Based on test_octahedral_analysis.py but adapted for directory traversal and
bulk processing of real experimental data.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
import re
from datetime import datetime

# Add the parent directory to the path to import SVC_materials
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from SVC_materials.core.analyzer import q2D_analyzer
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Error: Could not import SVC_materials analyzer: {e}")
    ANALYSIS_AVAILABLE = False

def extract_octahedral_properties(ontology_data, structure_name, vector_analysis=None, penetration_analysis=None):
    """
    Extract interesting properties from the ontology data for each octahedron.
    
    Args:
        ontology_data (dict): The loaded ontology JSON data
        structure_name (str): Name of the structure 
        vector_analysis (dict, optional): Vector analysis results from analyzer
        penetration_analysis (dict, optional): Penetration depth analysis results
    
    Returns:
        list: List of dictionaries containing octahedral properties
    """
    properties_list = []
    
    # Extract general experiment info
    experiment_info = ontology_data.get('experiment', {})
    cell_props = ontology_data.get('cell_properties', {})
    lattice_params = cell_props.get('lattice_parameters', {})
    composition = cell_props.get('composition', {})
    structure_info = cell_props.get('structure_info', {})
    
    # Extract octahedra data
    octahedra = ontology_data.get('octahedra', {})
    
    for oct_id, oct_data in octahedra.items():
        # Basic experiment information
        properties = {
            'structure_name': structure_name,
            'entity_id': oct_id,
            'entity_type': 'octahedron',
            'experiment_name': experiment_info.get('name', ''),
            'timestamp': experiment_info.get('timestamp', ''),
            
            # Lattice parameters
            'lattice_a': lattice_params.get('A', np.nan),
            'lattice_b': lattice_params.get('B', np.nan),
            'lattice_c': lattice_params.get('C', np.nan),
            'alpha': lattice_params.get('Alpha', np.nan),
            'beta': lattice_params.get('Beta', np.nan),
            'gamma': lattice_params.get('Gamma', np.nan),
            
            # Composition
            'metal_B': composition.get('metal_B', ''),
            'halogen_X': composition.get('halogen_X', ''),
            'total_atoms': composition.get('number_of_atoms', 0),
            'total_octahedra': composition.get('number_of_octahedra', 0),
            
            # Cell properties
            'cell_volume': structure_info.get('cell_volume', np.nan),
            'cutoff_ref_ligand': structure_info.get('cutoff_ref_ligand', np.nan),
        }
        
        # Central atom properties
        central_atom = oct_data.get('central_atom', {})
        properties.update({
            'central_atom_index': central_atom.get('global_index', np.nan),
            'central_atom_symbol': central_atom.get('symbol', ''),
            'central_x': central_atom.get('coordinates', {}).get('x', np.nan),
            'central_y': central_atom.get('coordinates', {}).get('y', np.nan),
            'central_z': central_atom.get('coordinates', {}).get('z', np.nan),
        })
        
        # Distortion parameters
        distortion = oct_data.get('distortion_parameters', {})
        properties.update({
            'zeta': distortion.get('zeta', np.nan),
            'delta': distortion.get('delta', np.nan),
            'sigma': distortion.get('sigma', np.nan),
            'theta_mean': distortion.get('theta_mean', np.nan),
            'theta_min': distortion.get('theta_min', np.nan),
            'theta_max': distortion.get('theta_max', np.nan),
        })
        
        # Bond distance analysis
        bond_analysis = oct_data.get('bond_distance_analysis', {})
        properties.update({
            'mean_bond_distance': bond_analysis.get('mean_bond_distance', np.nan),
            'bond_distance_variance': bond_analysis.get('bond_distance_variance', np.nan),
        })
        
        # Geometric properties
        geom_props = oct_data.get('geometric_properties', {})
        properties.update({
            'octahedral_volume': geom_props.get('octahedral_volume', np.nan),
            'is_octahedral': geom_props.get('is_octahedral', False),
        })
        
        # Ligand atom information
        ligand_atoms = oct_data.get('ligand_atoms', {})
        properties.update({
            'num_axial_ligands': len(ligand_atoms.get('axial_global_indices', [])),
            'num_equatorial_ligands': len(ligand_atoms.get('equatorial_global_indices', [])),
            'total_ligands': len(ligand_atoms.get('all_ligand_global_indices', [])),
        })
        
        # Angular analysis
        angular_analysis = oct_data.get('angular_analysis', {})
        axial_central_axial = angular_analysis.get('axial_central_axial', {})
        properties.update({
            'axial_central_axial_angle': axial_central_axial.get('angle_degrees', np.nan),
            'deviation_from_180': axial_central_axial.get('deviation_from_180', np.nan),
            'is_linear': axial_central_axial.get('is_linear', False),
        })
        
        # Summary statistics from angular analysis
        summary = angular_analysis.get('summary', {})
        properties.update({
            'total_axial_bridges': summary.get('total_axial_bridges', 0),
            'total_equatorial_bridges': summary.get('total_equatorial_bridges', 0),
            'avg_axial_central_axial_angle': summary.get('average_axial_central_axial_angle', np.nan),
            'avg_central_equatorial_central_angle': summary.get('average_central_equatorial_central_angle', np.nan),
        })
        
        # Bond angle statistics (cis and trans)
        bond_angles = oct_data.get('bond_angles', {})
        cis_angles = bond_angles.get('cis_angles', {})
        trans_angles = bond_angles.get('trans_angles', {})
        
        # Extract cis angle values
        cis_values = [angle_data.get('value', np.nan) for angle_data in cis_angles.values()]
        if cis_values:
            properties.update({
                'cis_angle_mean': np.mean(cis_values),
                'cis_angle_std': np.std(cis_values),
                'cis_angle_min': np.min(cis_values),
                'cis_angle_max': np.max(cis_values),
            })
        else:
            properties.update({
                'cis_angle_mean': np.nan,
                'cis_angle_std': np.nan,
                'cis_angle_min': np.nan,
                'cis_angle_max': np.nan,
            })
        
        # Extract trans angle values
        trans_values = [angle_data.get('value', np.nan) for angle_data in trans_angles.values()]
        if trans_values:
            properties.update({
                'trans_angle_mean': np.mean(trans_values),
                'trans_angle_std': np.std(trans_values),
                'trans_angle_min': np.min(trans_values),
                'trans_angle_max': np.max(trans_values),
            })
        else:
            properties.update({
                'trans_angle_mean': np.nan,
                'trans_angle_std': np.nan,
                'trans_angle_min': np.nan,
                'trans_angle_max': np.nan,
            })
        
        # Individual bond distances
        bond_distances = oct_data.get('bond_distances', {})
        bond_dist_values = [bond.get('distance', np.nan) for bond in bond_distances.values()]
        if bond_dist_values:
            properties.update({
                'bond_distance_min': np.min(bond_dist_values),
                'bond_distance_max': np.max(bond_dist_values),
                'bond_distance_range': np.max(bond_dist_values) - np.min(bond_dist_values),
            })
        else:
            properties.update({
                'bond_distance_min': np.nan,
                'bond_distance_max': np.nan,
                'bond_distance_range': np.nan,
            })
        
        # Add molecular properties (set to defaults for octahedra)
        properties.update({
            'molecular_formula': '',
            'num_atoms': np.nan,
            'molecular_size_max': np.nan,
            'molecular_size_mean': np.nan,
            'is_spacer_molecule': False,
            'is_a_site_molecule': False,
        })
        
        # Add element composition (set to 0 for octahedra)
        for element in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']:
            properties[f'mol_{element}_count'] = 0
        
        # Add vector analysis properties (same for all octahedra in a structure)
        if vector_analysis and 'error' not in vector_analysis:
            vector_results = vector_analysis.get('vector_analysis_results', {})
            plane_analysis = vector_analysis.get('plane_analysis', {})
            
            properties.update({
                'vector_angle_between_planes_deg': vector_results.get('angle_between_planes_degrees', np.nan),
                'vector_distance_between_centers_angstrom': vector_results.get('distance_between_plane_centers_angstrom', np.nan),
                'vector_low_plane_vs_z_axis_deg': plane_analysis.get('low_plane', {}).get('angle_vs_z_axis_degrees', np.nan),
                'vector_high_plane_vs_z_axis_deg': plane_analysis.get('high_plane', {}).get('angle_vs_z_axis_degrees', np.nan),
                'vector_analysis_available': True
            })
        else:
            properties.update({
                'vector_angle_between_planes_deg': np.nan,
                'vector_distance_between_centers_angstrom': np.nan,
                'vector_low_plane_vs_z_axis_deg': np.nan,
                'vector_high_plane_vs_z_axis_deg': np.nan,
                'vector_analysis_available': False
            })
        
        # Add penetration depth analysis properties (same for all octahedra in a structure)
        if penetration_analysis and 'error' not in penetration_analysis:
            # Extract penetration data for spacer molecules (if available)
            mol_1_data = penetration_analysis.get('molecule_1', penetration_analysis.get('molecule_0', {}))
            mol_2_data = penetration_analysis.get('molecule_2', penetration_analysis.get('molecule_1', {}))
            
            if 'penetration_segments' in mol_1_data:
                segments_1 = mol_1_data['penetration_segments']
                properties.update({
                    'penetration_mol1_formula': mol_1_data.get('formula', ''),
                    'penetration_mol1_n1_to_low': segments_1.get('n1_to_low_plane', np.nan),
                    'penetration_mol1_low_to_high': segments_1.get('low_plane_to_high_plane', np.nan),
                    'penetration_mol1_high_to_n2': segments_1.get('high_plane_to_n2', np.nan),
                    'penetration_mol1_total_length': segments_1.get('molecular_length', np.nan),
                })
            else:
                properties.update({
                    'penetration_mol1_formula': '',
                    'penetration_mol1_n1_to_low': np.nan,
                    'penetration_mol1_low_to_high': np.nan,
                    'penetration_mol1_high_to_n2': np.nan,
                    'penetration_mol1_total_length': np.nan,
                })
            
            if 'penetration_segments' in mol_2_data:
                segments_2 = mol_2_data['penetration_segments']
                properties.update({
                    'penetration_mol2_formula': mol_2_data.get('formula', ''),
                    'penetration_mol2_n1_to_low': segments_2.get('n1_to_low_plane', np.nan),
                    'penetration_mol2_low_to_high': segments_2.get('low_plane_to_high_plane', np.nan),
                    'penetration_mol2_high_to_n2': segments_2.get('high_plane_to_n2', np.nan),
                    'penetration_mol2_total_length': segments_2.get('molecular_length', np.nan),
                })
            else:
                properties.update({
                    'penetration_mol2_formula': '',
                    'penetration_mol2_n1_to_low': np.nan,
                    'penetration_mol2_low_to_high': np.nan,
                    'penetration_mol2_high_to_n2': np.nan,
                    'penetration_mol2_total_length': np.nan,
                })
            
            # Add comparative analysis
            if 'comparative_analysis' in penetration_analysis:
                comp_data = penetration_analysis['comparative_analysis']['penetration_comparison']
                properties.update({
                    'penetration_n1_to_low_diff': comp_data.get('n1_to_low_diff', np.nan),
                    'penetration_low_to_high_diff': comp_data.get('low_to_high_diff', np.nan),
                    'penetration_high_to_n2_diff': comp_data.get('high_to_n2_diff', np.nan),
                    'penetration_analysis_available': True
                })
            else:
                properties.update({
                    'penetration_n1_to_low_diff': np.nan,
                    'penetration_low_to_high_diff': np.nan,
                    'penetration_high_to_n2_diff': np.nan,
                    'penetration_analysis_available': True
                })
        else:
            # No penetration analysis available
            properties.update({
                'penetration_mol1_formula': '',
                'penetration_mol1_n1_to_low': np.nan,
                'penetration_mol1_low_to_high': np.nan,
                'penetration_mol1_high_to_n2': np.nan,
                'penetration_mol1_total_length': np.nan,
                'penetration_mol2_formula': '',
                'penetration_mol2_n1_to_low': np.nan,
                'penetration_mol2_low_to_high': np.nan,
                'penetration_mol2_high_to_n2': np.nan,
                'penetration_mol2_total_length': np.nan,
                'penetration_n1_to_low_diff': np.nan,
                'penetration_low_to_high_diff': np.nan,
                'penetration_high_to_n2_diff': np.nan,
                'penetration_analysis_available': False
            })
        
        properties_list.append(properties)
    
    return properties_list

def extract_molecule_properties_for_dataset(analyzer, structure_name, vector_analysis=None, penetration_analysis=None):
    """
    Extract molecule properties in the same format as octahedral properties for dataset creation.
    
    Args:
        analyzer: q2D_analyzer instance with molecule analysis completed
        structure_name (str): Name of the structure
        vector_analysis (dict, optional): Vector analysis results from analyzer
        penetration_analysis (dict, optional): Penetration depth analysis results
    
    Returns:
        list: List of dictionaries containing molecule properties (similar to octahedral format)
    """
    molecule_properties = []
    
    if not hasattr(analyzer, 'spacer_molecules') or not analyzer.spacer_molecules:
        return molecule_properties
    
    molecules = analyzer.spacer_molecules.get('molecules', {})
    if not molecules:
        return molecule_properties
    
    # Get basic experiment info from analyzer
    experiment_name = analyzer.experiment_name
    timestamp = str(pd.Timestamp.now())
    
    # Get cell and spacer information
    spacer_composition = analyzer._get_spacer_composition() if analyzer.spacer else {}
    cell_info = analyzer.atoms.get_cell_lengths_and_angles()
    a, b, c, alpha, beta, gamma = cell_info
    
    # Separate molecules by size to identify spacer vs A-sites
    spacer_molecules, a_site_molecules = analyzer.separate_molecules_by_size()
    
    for mol_id, mol_data in molecules.items():
        # Determine molecule type
        is_spacer_molecule = mol_id in spacer_molecules
        is_a_site_molecule = mol_id in a_site_molecules
        
        # Calculate molecule properties
        symbols = mol_data.get('symbols', [])
        coordinates = mol_data.get('coordinates', [])
        formula = mol_data.get('formula', '')
        
        # Calculate center of mass
        if coordinates:
            coords_array = np.array(coordinates)
            center_of_mass = np.mean(coords_array, axis=0)
        else:
            center_of_mass = [np.nan, np.nan, np.nan]
        
        # Count elements
        element_counts = {}
        for symbol in symbols:
            element_counts[symbol] = element_counts.get(symbol, 0) + 1
        
        # Calculate molecular geometry properties
        if len(coordinates) > 1:
            coords_array = np.array(coordinates)
            # Calculate molecular size (max distance between any two atoms)
            distances = []
            for i in range(len(coordinates)):
                for j in range(i+1, len(coordinates)):
                    dist = np.linalg.norm(coords_array[i] - coords_array[j])
                    distances.append(dist)
            max_distance = max(distances) if distances else 0
            mean_distance = np.mean(distances) if distances else 0
        else:
            max_distance = 0
            mean_distance = 0
        
        # Create molecule properties dictionary (similar to octahedral format)
        properties = {
            # Basic identification (similar to octahedral)
            'structure_name': structure_name,
            'entity_id': f"molecule_{mol_id}",
            'entity_type': 'molecule',
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            
            # Lattice parameters (same as octahedral)
            'lattice_a': float(a),
            'lattice_b': float(b),
            'lattice_c': float(c),
            'alpha': float(alpha),
            'beta': float(beta),
            'gamma': float(gamma),
            
            # Composition (adapted for molecules)
            'metal_B': analyzer.b,
            'halogen_X': analyzer.x,
            'total_atoms': len(analyzer.atoms),
            'total_octahedra': len(analyzer.ordered_octahedra),
            
            # Cell properties
            'cell_volume': float(analyzer.atoms.get_volume()),
            'cutoff_ref_ligand': float(analyzer.cutoff_ref_ligand),
            
            # Central position (analogous to central atom)
            'central_atom_index': np.nan,
            'central_atom_symbol': 'MOL',
            'central_x': float(center_of_mass[0]),
            'central_y': float(center_of_mass[1]),
            'central_z': float(center_of_mass[2]),
            
            # Molecular properties (analogous to distortion parameters)
            'molecular_formula': formula,
            'num_atoms': len(symbols),
            'molecular_size_max': float(max_distance),
            'molecular_size_mean': float(mean_distance),
            'is_spacer_molecule': bool(is_spacer_molecule),
            'is_a_site_molecule': bool(is_a_site_molecule),
            
            # Set octahedral-specific properties to NaN for molecules
            'zeta': np.nan,
            'delta': np.nan,
            'sigma': np.nan,
            'theta_mean': np.nan,
            'theta_min': np.nan,
            'theta_max': np.nan,
            'mean_bond_distance': np.nan,
            'bond_distance_variance': np.nan,
            'octahedral_volume': np.nan,
            'is_octahedral': False,
            'num_axial_ligands': 0,
            'num_equatorial_ligands': 0,
            'total_ligands': 0,
            'axial_central_axial_angle': np.nan,
            'deviation_from_180': np.nan,
            'is_linear': False,
            'total_axial_bridges': 0,
            'total_equatorial_bridges': 0,
            'avg_axial_central_axial_angle': np.nan,
            'avg_central_equatorial_central_angle': np.nan,
            'cis_angle_mean': np.nan,
            'cis_angle_std': np.nan,
            'cis_angle_min': np.nan,
            'cis_angle_max': np.nan,
            'trans_angle_mean': np.nan,
            'trans_angle_std': np.nan,
            'trans_angle_min': np.nan,
            'trans_angle_max': np.nan,
            'bond_distance_min': np.nan,
            'bond_distance_max': np.nan,
            'bond_distance_range': np.nan,
        }
        
        # Add element composition
        for element in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']:
            properties[f'mol_{element}_count'] = element_counts.get(element, 0)
        
        # Add vector analysis properties
        if vector_analysis and 'error' not in vector_analysis:
            vector_results = vector_analysis.get('vector_analysis_results', {})
            plane_analysis = vector_analysis.get('plane_analysis', {})
            
            properties.update({
                'vector_angle_between_planes_deg': vector_results.get('angle_between_planes_degrees', np.nan),
                'vector_distance_between_centers_angstrom': vector_results.get('distance_between_plane_centers_angstrom', np.nan),
                'vector_low_plane_vs_z_axis_deg': plane_analysis.get('low_plane', {}).get('angle_vs_z_axis_degrees', np.nan),
                'vector_high_plane_vs_z_axis_deg': plane_analysis.get('high_plane', {}).get('angle_vs_z_axis_degrees', np.nan),
                'vector_analysis_available': True
            })
        else:
            properties.update({
                'vector_angle_between_planes_deg': np.nan,
                'vector_distance_between_centers_angstrom': np.nan,
                'vector_low_plane_vs_z_axis_deg': np.nan,
                'vector_high_plane_vs_z_axis_deg': np.nan,
                'vector_analysis_available': False
            })
        
        # Add penetration depth analysis properties (specific to this molecule if it's a spacer)
        if penetration_analysis and 'error' not in penetration_analysis and is_spacer_molecule:
            # Find penetration data for this specific molecule
            mol_penetration_data = None
            for key, mol_data in penetration_analysis.items():
                if key.startswith('molecule_') and 'formula' in mol_data:
                    if mol_data['formula'] == formula:  # Match by formula
                        mol_penetration_data = mol_data
                        break
            
            if mol_penetration_data and 'penetration_segments' in mol_penetration_data:
                segments = mol_penetration_data['penetration_segments']
                properties.update({
                    'penetration_n1_to_low': segments.get('n1_to_low_plane', np.nan),
                    'penetration_low_to_high': segments.get('low_plane_to_high_plane', np.nan),
                    'penetration_high_to_n2': segments.get('high_plane_to_n2', np.nan),
                    'penetration_total_length': segments.get('molecular_length', np.nan),
                    'penetration_length_difference': segments.get('length_difference', np.nan),
                    'penetration_analysis_available': True
                })
                
                # Add N1 and N2 positions
                if 'n1_position' in mol_penetration_data:
                    n1_pos = mol_penetration_data['n1_position']
                    properties.update({
                        'penetration_n1_x': float(n1_pos[0]),
                        'penetration_n1_y': float(n1_pos[1]),
                        'penetration_n1_z': float(n1_pos[2]),
                    })
                else:
                    properties.update({
                        'penetration_n1_x': np.nan,
                        'penetration_n1_y': np.nan,
                        'penetration_n1_z': np.nan,
                    })
                
                if 'n2_position' in mol_penetration_data:
                    n2_pos = mol_penetration_data['n2_position']
                    properties.update({
                        'penetration_n2_x': float(n2_pos[0]),
                        'penetration_n2_y': float(n2_pos[1]),
                        'penetration_n2_z': float(n2_pos[2]),
                    })
                else:
                    properties.update({
                        'penetration_n2_x': np.nan,
                        'penetration_n2_y': np.nan,
                        'penetration_n2_z': np.nan,
                    })
            else:
                # Spacer molecule but no penetration data found
                properties.update({
                    'penetration_n1_to_low': np.nan,
                    'penetration_low_to_high': np.nan,
                    'penetration_high_to_n2': np.nan,
                    'penetration_total_length': np.nan,
                    'penetration_length_difference': np.nan,
                    'penetration_n1_x': np.nan,
                    'penetration_n1_y': np.nan,
                    'penetration_n1_z': np.nan,
                    'penetration_n2_x': np.nan,
                    'penetration_n2_y': np.nan,
                    'penetration_n2_z': np.nan,
                    'penetration_analysis_available': False
                })
        else:
            # Not a spacer molecule or no penetration analysis
            properties.update({
                'penetration_n1_to_low': np.nan,
                'penetration_low_to_high': np.nan,
                'penetration_high_to_n2': np.nan,
                'penetration_total_length': np.nan,
                'penetration_length_difference': np.nan,
                'penetration_n1_x': np.nan,
                'penetration_n1_y': np.nan,
                'penetration_n1_z': np.nan,
                'penetration_n2_x': np.nan,
                'penetration_n2_y': np.nan,
                'penetration_n2_z': np.nan,
                'penetration_analysis_available': False
            })
        
        # Add spacer composition context
        for element, count in spacer_composition.items():
            properties[f'spacer_total_{element}'] = count
        
        molecule_properties.append(properties)
    
    return molecule_properties

def create_analysis_folder(analyzer, structure_name, output_base_dir="BULKS_RESULTS"):
    """
    Create a comprehensive analysis folder with all files organized in a single directory.
    """
    # Create main analysis folder
    analysis_dir = os.path.join(output_base_dir, f"{structure_name}_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(f"  📁 Creating comprehensive analysis folder: {analysis_dir}")
    
    # 1. Export octahedral ontology (layers analysis)
    layers_ontology_file = os.path.join(analysis_dir, f"{structure_name}_layers_ontology.json")
    analyzer.export_ontology_json(layers_ontology_file)
    print(f"    ✓ Layers ontology: {os.path.basename(layers_ontology_file)}")
    
    # 2. Export molecule ontology
    if hasattr(analyzer, 'molecule_ontology') and analyzer.molecule_ontology:
        molecule_ontology_file = os.path.join(analysis_dir, f"{structure_name}_molecule_ontology.json")
        analyzer.export_molecule_ontology_json(molecule_ontology_file)
        print(f"    ✓ Molecule ontology: {os.path.basename(molecule_ontology_file)}")
    
    # 3. Save spacer structure (original spacer with all molecules)
    if analyzer.spacer is not None:
        spacer_file = os.path.join(analysis_dir, f"{structure_name}_spacer.vasp")
        analyzer.save_spacer_structure(spacer_file)
        print(f"    ✓ Spacer structure: {os.path.basename(spacer_file)}")
    
    # 4. Create separated structures
    if hasattr(analyzer, 'spacer_molecules') and analyzer.spacer_molecules:
        spacer_molecules, a_site_molecules = analyzer.separate_molecules_by_size()
        
        if spacer_molecules:
            large_molecules_file = os.path.join(analysis_dir, f"{structure_name}_large_molecules.xyz")
            spacer_structure = analyzer.create_spacer_structure(spacer_molecules)
            if spacer_structure:
                spacer_structure.write(large_molecules_file, format='xyz')
                print(f"    ✓ Large molecules (spacer): {os.path.basename(large_molecules_file)}")
        
        # Salt structure (large molecules + terminal halogens)
        if spacer_molecules:
            salt_file = os.path.join(analysis_dir, f"{structure_name}_salt.vasp")
            salt_structure = analyzer.create_salt_structure(spacer_molecules)
            if salt_structure:
                salt_structure.write(salt_file, format='vasp')
                print(f"    ✓ Salt structure (spacer + terminals): {os.path.basename(salt_file)}")
        
        # Small molecules (A-sites)
        if a_site_molecules:
            small_molecules_file = os.path.join(analysis_dir, f"{structure_name}_small_molecules.xyz")
            a_sites_structure = analyzer.create_a_sites_structure(a_site_molecules)
            if a_sites_structure:
                a_sites_structure.write(small_molecules_file, format='xyz')
                print(f"    ✓ Small molecules (A-sites): {os.path.basename(small_molecules_file)}")
    
    # 5. Create layers-only structure
    if hasattr(analyzer, 'spacer_molecules') and analyzer.spacer_molecules:
        layers_only_file = os.path.join(analysis_dir, f"{structure_name}_layers_only.vasp")
        layers_structure = analyzer.create_layers_only_structure()
        if layers_structure:
            layers_structure.write(layers_only_file, format='vasp')
            print(f"    ✓ Layers only (no large molecules): {os.path.basename(layers_only_file)}")
    
    # 6. Save individual molecules
    if hasattr(analyzer, 'spacer_molecules') and analyzer.spacer_molecules:
        molecules = analyzer.spacer_molecules.get('molecules', {})
        if molecules:
            isolated_molecules = analyzer.isolate_individual_molecules()
            
            for mol_id, mol_data in molecules.items():
                formula = mol_data.get('formula', f'molecule_{mol_id}')
                mol_file = os.path.join(analysis_dir, f"molecule_{mol_id}_{formula}.vasp")
                
                if mol_id in isolated_molecules:
                    isolated_molecules[mol_id]['atoms'].write(mol_file, format='vasp')
            
            print(f"    ✓ Individual molecules: {len(molecules)} files")
    
    # 7. Vector analysis (if salt structure available)
    if hasattr(analyzer, 'vector_analysis') and 'error' not in analyzer.vector_analysis:
        vector_json_file = os.path.join(analysis_dir, f"{structure_name}_vector_analysis.json")
        analyzer.export_vector_analysis_json(vector_json_file)
        print(f"    ✓ Vector analysis: {os.path.basename(vector_json_file)}")
        
        vector_plot_file = os.path.join(analysis_dir, f"{structure_name}_vector_plot.html")
        plot_result = analyzer.create_vector_plot(vector_plot_file)
        if plot_result.endswith('.html'):
            print(f"    ✓ Interactive vector plot: {os.path.basename(vector_plot_file)}")
        else:
            print(f"    ⚠ Vector plot creation failed: {plot_result}")
    else:
        print(f"    ⚠ Vector analysis not available")
    
    # 8. Penetration depth analysis (if available)
    penetration_results = analyzer.get_penetration_depth_analysis()
    if 'error' not in penetration_results:
        penetration_json_file = os.path.join(analysis_dir, f"{structure_name}_penetration_analysis.json")
        analyzer.export_penetration_analysis_json(penetration_json_file)
        print(f"    ✓ Penetration depth analysis: {os.path.basename(penetration_json_file)}")
        
        # Create penetration summary
        penetration_summary_file = os.path.join(analysis_dir, f"{structure_name}_penetration_summary.txt")
        with open(penetration_summary_file, 'w') as f:
            f.write(analyzer.get_penetration_summary())
        print(f"    ✓ Penetration depth summary: {os.path.basename(penetration_summary_file)}")
    else:
        print(f"    ⚠ Penetration depth analysis: {penetration_results['error']}")
    
    # 9. Create summary report
    summary_file = os.path.join(analysis_dir, f"{structure_name}_molecule_summary.txt")
    analyzer._create_molecule_summary_report(summary_file)
    print(f"    ✓ Analysis summary: {os.path.basename(summary_file)}")
    
    return analysis_dir

def parse_structure_info(path_str):
    """Parse perovskite and experiment information from path."""
    parts = Path(path_str).parts
    
    # Extract perovskite name (e.g., MAPbBr3)
    perovskite_name = None
    experiment_name = None
    
    for part in parts:
        if 'Pb' in part and any(x in part for x in ['Cl', 'Br', 'I']):
            if '_n' in part:  # This is the experiment name
                experiment_name = part
                # Extract perovskite name from experiment name
                perovskite_name = part.split('_')[0]
            else:  # This is just the perovskite name
                perovskite_name = part
    
    # Parse halogen from perovskite name
    halogen = None
    if perovskite_name:
        halogen_match = re.search(r'(Br|Cl|I)', perovskite_name)
        halogen = halogen_match.group(1) if halogen_match else None
    
    # Parse n_slab from experiment name
    n_slab = None
    if experiment_name:
        n_match = re.search(r'_n(\d+)_', experiment_name)
        n_slab = int(n_match.group(1)) if n_match else None
    
    # Parse molecule name
    molecule_name = None
    if experiment_name and n_slab is not None:
        parts = experiment_name.split('_')
        if len(parts) > 2:
            molecule_name = '_'.join(parts[2:])
    
    return {
        'perovskite_name': perovskite_name,
        'experiment_name': experiment_name,
        'halogen': halogen,
        'n_slab': n_slab,
        'molecule_name': molecule_name
    }

def find_all_contcar_files(root_dir):
    """Find all CONTCAR files in the directory structure."""
    contcar_files = []
    root_path = Path(root_dir)
    
    for contcar_path in root_path.rglob('CONTCAR'):
        if contcar_path.is_file():
            contcar_files.append(contcar_path)
    
    return contcar_files

def comprehensive_bulks_analysis():
    """
    Comprehensive analysis of all CONTCAR files in DION-JACOBSON/BULKS directory.
    """
    if not ANALYSIS_AVAILABLE:
        print("❌ SVC_materials analysis not available. Cannot run analysis.")
        return None
    
    print("=" * 80)
    print("COMPREHENSIVE DION-JACOBSON BULKS ANALYSIS")
    print("=" * 80)
    
    # Define the BULKS directory
    bulks_dir = '/home/dotempo/Documents/DION-JACOBSON/BULKS'
    output_dir = '/home/dotempo/Documents/REPO/SVC-Materials/Graphs/BULKS_RESULTS'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CONTCAR files
    print(f"\nScanning for CONTCAR files in: {bulks_dir}")
    contcar_files = find_all_contcar_files(bulks_dir)
    
    if not contcar_files:
        print("❌ No CONTCAR files found!")
        return None
    
    print(f"Found {len(contcar_files)} CONTCAR files")
    print("-" * 60)
    
    # Parameters for analysis
    central_atom = 'Pb'
    ligand_atom = 'Cl'  # Will be updated per structure
    cutoff = 3.5
    
    # Collect properties for unified dataset
    all_unified_properties = []
    successful_analyses = 0
    failed_analyses = 0
    
    for i, contcar_path in enumerate(contcar_files, 1):
        print(f"\n[{i}/{len(contcar_files)}] Processing: {contcar_path}")
        print("*" * 60)
        
        try:
            # Parse structure information from path
            structure_info = parse_structure_info(str(contcar_path))
            perovskite_name = structure_info['perovskite_name']
            experiment_name = structure_info['experiment_name']
            halogen = structure_info['halogen']
            n_slab = structure_info['n_slab']
            molecule_name = structure_info['molecule_name']
            
            if not halogen:
                print(f"  ⚠ Could not determine halogen from path, skipping")
                failed_analyses += 1
                continue
            
            # Update ligand atom based on structure
            ligand_atom = halogen
            
            # Create structure name for output
            structure_name = f"{perovskite_name}_n{n_slab}" if n_slab else perovskite_name
            if molecule_name:
                # Truncate long molecule names for file naming
                short_mol_name = molecule_name[:30] if len(molecule_name) > 30 else molecule_name
                structure_name += f"_{short_mol_name}"
            
            print(f"  Structure: {perovskite_name}")
            print(f"  Halogen: {halogen}")
            print(f"  N-slab: {n_slab}")
            print(f"  Molecule: {molecule_name}")
            
            # Create analyzer
            analyzer = q2D_analyzer(
                file_path=str(contcar_path),
                b=central_atom,
                x=ligand_atom,
                cutoff_ref_ligand=cutoff
            )
            
            # Display basic information
            print(f"  Original structure: {len(analyzer.atoms)} atoms")
            if analyzer.spacer is not None:
                print(f"  Spacer structure: {len(analyzer.spacer)} atoms")
                spacer_comp = analyzer._get_spacer_composition()
                print(f"  Spacer composition: {spacer_comp}")
            print(f"  Octahedra identified: {len(analyzer.ordered_octahedra)}")
            
            # Check vector analysis
            vector_analysis = analyzer.get_vector_analysis()
            if 'error' not in vector_analysis:
                vector_results = vector_analysis['vector_analysis_results']
                print(f"  ✓ Vector analysis available:")
                print(f"    • Angle between planes: {vector_results['angle_between_planes_degrees']:.2f}°")
                print(f"    • Distance between centers: {vector_results['distance_between_plane_centers_angstrom']:.3f} Å")
            else:
                print(f"  ⚠ Vector analysis: {vector_analysis['error']}")
                vector_analysis = None
            
            # Check penetration depth analysis
            penetration_analysis = analyzer.get_penetration_depth_analysis()
            if 'error' not in penetration_analysis:
                print(f"  ✓ Penetration depth analysis available:")
                mol_count = len([k for k in penetration_analysis.keys() if k.startswith('molecule_')])
                print(f"    • Analyzed molecules: {mol_count}")
                if 'comparative_analysis' in penetration_analysis:
                    comp_data = penetration_analysis['comparative_analysis']['penetration_comparison']
                    print(f"    • Max segment difference: {max(comp_data.values()):.3f} Å")
            else:
                print(f"  ⚠ Penetration depth analysis: {penetration_analysis['error']}")
                penetration_analysis = None
            
            # Create comprehensive analysis folder
            analysis_dir = create_analysis_folder(analyzer, structure_name, output_dir)
            
            # Show molecule distribution if available
            if analyzer.spacer_molecules and analyzer.spacer_molecules.get('total_molecules', 0) > 0:
                molecules = analyzer.spacer_molecules.get('molecules', {})
                print(f"  ✓ Molecules identified: {len(molecules)}")
                
                distribution = analyzer._get_molecule_distribution()
                print("    Molecule types:")
                for formula, count in sorted(distribution.items()):
                    print(f"      {formula}: {count} molecules")
            else:
                print("  - No molecules found in spacer structure")
            
            # Extract properties for unified dataset
            layers_ontology_file = os.path.join(analysis_dir, f"{structure_name}_layers_ontology.json")
            with open(layers_ontology_file, 'r') as f:
                octahedral_ontology_data = json.load(f)
            
            # Get vector analysis from ontology if available
            ontology_vector_analysis = octahedral_ontology_data.get('vector_analysis', None)
            if ontology_vector_analysis and 'error' not in ontology_vector_analysis:
                vector_analysis = ontology_vector_analysis
            
            # Get penetration analysis from ontology if available
            ontology_penetration_analysis = octahedral_ontology_data.get('penetration_analysis', None)
            if ontology_penetration_analysis and 'error' not in ontology_penetration_analysis:
                penetration_analysis = ontology_penetration_analysis
            
            octahedral_props = extract_octahedral_properties(octahedral_ontology_data, structure_name, vector_analysis, penetration_analysis)
            
            # Add spacer composition and structure metadata to octahedral properties
            spacer_composition = analyzer._get_spacer_composition() if analyzer.spacer else {}
            for oct_prop in octahedral_props:
                for element, count in spacer_composition.items():
                    oct_prop[f'spacer_total_{element}'] = count
                
                # Add parsed structure metadata
                oct_prop.update({
                    'perovskite_name': perovskite_name,
                    'halogen': halogen,
                    'n_slab': n_slab,
                    'molecule_name': molecule_name,
                    'contcar_path': str(contcar_path)
                })
            
            all_unified_properties.extend(octahedral_props)
            print(f"  ✓ Octahedral properties extracted: {len(octahedral_props)} octahedra")
            
            # Extract molecule properties for unified dataset
            if analyzer.spacer_molecules:
                molecule_props = extract_molecule_properties_for_dataset(analyzer, structure_name, vector_analysis, penetration_analysis)
                
                # Add structure metadata to molecular properties
                for mol_prop in molecule_props:
                    mol_prop.update({
                        'perovskite_name': perovskite_name,
                        'halogen': halogen,
                        'n_slab': n_slab,
                        'molecule_name': molecule_name,
                        'contcar_path': str(contcar_path)
                    })
                
                all_unified_properties.extend(molecule_props)
                print(f"  ✓ Molecule properties extracted: {len(molecule_props)} molecules")
            
            successful_analyses += 1
            print(f"  ✅ Analysis completed successfully")
            
        except Exception as e:
            print(f"  ✗ Error processing {contcar_path}: {str(e)}")
            failed_analyses += 1
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("CREATING UNIFIED DATASET")
    print("=" * 80)
    
    # Create unified dataset
    if all_unified_properties:
        unified_df = pd.DataFrame(all_unified_properties)
        unified_df = unified_df.sort_values(['perovskite_name', 'n_slab', 'entity_type', 'entity_id']).reset_index(drop=True)
        
        # Add timestamp
        unified_df['analysis_timestamp'] = datetime.now().isoformat()
        
        unified_csv = os.path.join(output_dir, "comprehensive_bulks_dataset.csv")
        unified_df.to_csv(unified_csv, index=False)
        
        print(f"\n📊 UNIFIED DATASET (Octahedra + Molecules):")
        print(f"   File: {unified_csv}")
        print(f"   Total entities: {len(unified_df)}")
        print(f"   Properties per entity: {len(unified_df.columns)}")
        
        # Analysis summary
        print(f"\n   Processing summary:")
        print(f"     Successful analyses: {successful_analyses}")
        print(f"     Failed analyses: {failed_analyses}")
        print(f"     Success rate: {successful_analyses/(successful_analyses+failed_analyses)*100:.1f}%")
        
        # Entity type distribution
        print(f"\n   Entity type distribution:")
        for entity_type, count in unified_df['entity_type'].value_counts().items():
            print(f"     {entity_type}: {count} entities")
        
        # Perovskite distribution
        print(f"\n   Perovskite distribution:")
        for perovskite, count in unified_df['perovskite_name'].value_counts().items():
            print(f"     {perovskite}: {count} entities")
        
        # Halogen distribution
        print(f"\n   Halogen distribution:")
        for halogen, count in unified_df['halogen'].value_counts().items():
            print(f"     {halogen}: {count} entities")
        
        # N-slab distribution
        print(f"\n   N-slab distribution:")
        for n_slab, count in unified_df['n_slab'].value_counts().items():
            print(f"     n={n_slab}: {count} entities")
        
        # Distortion statistics for octahedra
        octahedra_only = unified_df[unified_df['entity_type'] == 'octahedron']
        if len(octahedra_only) > 0:
            print(f"\n   Key distortion parameters for octahedra (mean ± std):")
            distortion_cols = ['zeta', 'delta', 'sigma', 'theta_mean']
            for col in distortion_cols:
                mean_val = octahedra_only[col].mean()
                std_val = octahedra_only[col].std()
                print(f"     {col}: {mean_val:.4f} ± {std_val:.4f}")
            
            # Penetration depth statistics for octahedra
            penetration_available = octahedra_only['penetration_analysis_available'].sum()
            if penetration_available > 0:
                print(f"\n   Penetration depth analysis (for {penetration_available} structures):")
                penetration_cols = ['penetration_mol1_n1_to_low', 'penetration_mol1_low_to_high', 'penetration_mol1_high_to_n2',
                                   'penetration_mol2_n1_to_low', 'penetration_mol2_low_to_high', 'penetration_mol2_high_to_n2',
                                   'penetration_n1_to_low_diff', 'penetration_low_to_high_diff', 'penetration_high_to_n2_diff']
                for col in penetration_cols:
                    valid_data = octahedra_only[col].dropna()
                    if len(valid_data) > 0:
                        mean_val = valid_data.mean()
                        std_val = valid_data.std()
                        print(f"     {col.replace('penetration_', '').replace('_', ' ')}: {mean_val:.3f} ± {std_val:.3f} Å")
        
        # Molecular statistics
        molecules_only = unified_df[unified_df['entity_type'] == 'molecule']
        if len(molecules_only) > 0:
            print(f"\n   Molecular statistics:")
            print(f"     Total molecules: {len(molecules_only)}")
            spacer_molecules = len(molecules_only[molecules_only['is_spacer_molecule'] == True])
            a_site_molecules = len(molecules_only[molecules_only['is_a_site_molecule'] == True])
            print(f"     Spacer molecules: {spacer_molecules}")
            print(f"     A-site molecules: {a_site_molecules}")
            print(f"     Mean molecular size: {molecules_only['molecular_size_max'].mean():.2f} Å")
            
            # Penetration depth statistics for spacer molecules
            spacer_molecules_data = molecules_only[molecules_only['is_spacer_molecule'] == True]
            if len(spacer_molecules_data) > 0:
                penetration_available_spacers = spacer_molecules_data['penetration_analysis_available'].sum()
                if penetration_available_spacers > 0:
                    print(f"\n     Penetration depth for spacer molecules ({penetration_available_spacers} molecules):")
                    penetration_spacer_cols = ['penetration_n1_to_low', 'penetration_low_to_high', 'penetration_high_to_n2', 'penetration_total_length']
                    for col in penetration_spacer_cols:
                        valid_data = spacer_molecules_data[col].dropna()
                        if len(valid_data) > 0:
                            mean_val = valid_data.mean()
                            std_val = valid_data.std()
                            print(f"       {col.replace('penetration_', '').replace('_', ' ')}: {mean_val:.3f} ± {std_val:.3f} Å")
    
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ Successfully analyzed {successful_analyses}/{len(contcar_files)} structures")
    print(f"📁 Output directory: {output_dir}")
    print(f"   • Each structure has its own analysis folder")
    print(f"   • All files organized in individual structure folders")
    
    print(f"\n📄 Files created per structure:")
    print(f"   • <structure>_layers_ontology.json - Octahedral analysis")
    print(f"   • <structure>_molecule_ontology.json - Molecular analysis")
    print(f"   • <structure>_vector_analysis.json - Vector/plane analysis")
    print(f"   • <structure>_penetration_analysis.json - Penetration depth analysis")
    print(f"   • <structure>_vector_plot.html - Interactive visualization")
    print(f"   • <structure>_penetration_summary.txt - Penetration depth summary")
    print(f"   • <structure>_spacer.vasp - All spacer molecules")
    print(f"   • <structure>_large_molecules.xyz - Largest molecules")
    print(f"   • <structure>_salt.vasp - Large molecules + terminals")
    print(f"   • <structure>_small_molecules.xyz - Small molecules (A-sites)")
    print(f"   • <structure>_layers_only.vasp - Structure without large molecules")
    print(f"   • molecule_<id>_<formula>.vasp - Individual molecules")
    print(f"   • <structure>_molecule_summary.txt - Human-readable summary")
    
    if all_unified_properties:
        print(f"\n📊 Comprehensive dataset: {unified_csv}")
        print(f"   • Ready for statistical analysis and machine learning")
        print(f"   • Includes both octahedral and molecular properties")
        print(f"   • Complete structure metadata and paths")
    
    return unified_df if all_unified_properties else None

if __name__ == "__main__":
    # Run the comprehensive bulk analysis
    print("🔬 STARTING COMPREHENSIVE BULKS ANALYSIS")
    print("Processing all CONTCAR files in DION-JACOBSON/BULKS directory")
    print("=" * 80)
    
    unified_results = comprehensive_bulks_analysis()
    
    if unified_results is not None:
        print("\n" + "=" * 80)
        print("COMPREHENSIVE BULKS ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"   📁 Analysis folders: BULKS_RESULTS/<structure>_analysis/")
        print(f"   📊 Unified dataset: BULKS_RESULTS/comprehensive_bulks_dataset.csv")
        print(f"   📈 Total entities analyzed: {len(unified_results)}")
        print(f"   🔬 Octahedra: {len(unified_results[unified_results['entity_type'] == 'octahedron'])}")
        print(f"   🧪 Molecules: {len(unified_results[unified_results['entity_type'] == 'molecule'])}")
        
        print(f"\n   🚀 Ready for:")
        print(f"   • Advanced visualization and statistical analysis")
        print(f"   • Structure-property relationship studies")
        print(f"   • Machine learning applications")
        print(f"   • High-throughput materials screening")
        print(f"   • Vector/angular analysis of salt structures")
        print(f"   • Penetration depth studies of spacer molecules")
    else:
        print("\n❌ Analysis failed or no data extracted.") 
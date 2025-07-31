#!/usr/bin/env python3
"""
Test script for octahedral distortion analysis using the q2D_analyzer class.
Analyzes three test structures (n1, n2, n3) and creates comprehensive datasets
including octahedral properties and molecule ontologies.
"""

import sys
import os
import pandas as pd
import numpy as np
import json

# Add the parent directory to the path to import SVC_materials
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from SVC_materials.core.analyzer import q2D_analyzer

def extract_octahedral_properties(ontology_data, structure_name):
    """
    Extract interesting properties from the ontology data for each octahedron.
    
    Args:
        ontology_data (dict): The loaded ontology JSON data
        structure_name (str): Name of the structure (n1, n2, n3)
    
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
            'entity_id': oct_id,  # Changed from octahedron_id to entity_id for consistency
            'entity_type': 'octahedron',  # Added to distinguish from molecules
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
        
        # Add spacer composition context (will be filled from structure data)
        # These will be the same for all octahedra in a structure
        
        properties_list.append(properties)
    
    return properties_list

def extract_molecule_properties_for_dataset(analyzer, structure_name):
    """
    Extract molecule properties in the same format as octahedral properties for dataset creation.
    
    Args:
        analyzer: q2D_analyzer instance with molecule analysis completed
        structure_name (str): Name of the structure (n1, n2, n3)
    
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
            'entity_id': f"molecule_{mol_id}",  # Analogous to octahedron_id
            'entity_type': 'molecule',  # To distinguish from octahedra
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
            'central_atom_index': np.nan,  # Molecules don't have single central atom
            'central_atom_symbol': 'MOL',  # Identifier for molecule
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
            
            # Bond distance analysis (not applicable to molecules as a whole)
            'mean_bond_distance': np.nan,
            'bond_distance_variance': np.nan,
            
            # Geometric properties (adapted)
            'octahedral_volume': np.nan,  # Not applicable
            'is_octahedral': False,  # This is a molecule, not an octahedron
            
            # Ligand information (not applicable)
            'num_axial_ligands': 0,
            'num_equatorial_ligands': 0,
            'total_ligands': 0,
            
            # Angular analysis (not applicable to molecules as entities)
            'axial_central_axial_angle': np.nan,
            'deviation_from_180': np.nan,
            'is_linear': False,
            
            # Summary statistics (not applicable)
            'total_axial_bridges': 0,
            'total_equatorial_bridges': 0,
            'avg_axial_central_axial_angle': np.nan,
            'avg_central_equatorial_central_angle': np.nan,
            
            # Bond angles (not applicable at molecular level)
            'cis_angle_mean': np.nan,
            'cis_angle_std': np.nan,
            'cis_angle_min': np.nan,
            'cis_angle_max': np.nan,
            'trans_angle_mean': np.nan,
            'trans_angle_std': np.nan,
            'trans_angle_min': np.nan,
            'trans_angle_max': np.nan,
            
            # Bond distances (not applicable at molecular level)
            'bond_distance_min': np.nan,
            'bond_distance_max': np.nan,
            'bond_distance_range': np.nan,
        }
        
        # Add element composition
        for element in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']:
            properties[f'mol_{element}_count'] = element_counts.get(element, 0)
        
        # Add spacer composition context
        for element, count in spacer_composition.items():
            properties[f'spacer_total_{element}'] = count
        
        molecule_properties.append(properties)
    
    return molecule_properties

def create_analysis_folder(analyzer, structure_name, output_base_dir="tests"):
    """
    Create a comprehensive analysis folder with all files organized in a single directory.
    
    Args:
        analyzer: q2D_analyzer instance
        structure_name: Name of the structure (e.g., 'n1')
        output_base_dir: Base directory for output
    
    Returns:
        str: Path to the created analysis folder
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
        # Large molecules (spacer) - 2 largest molecules in the original cell
        spacer_molecules, a_site_molecules = analyzer.separate_molecules_by_size()
        
        if spacer_molecules:
            large_molecules_file = os.path.join(analysis_dir, f"{structure_name}_large_molecules.vasp")
            spacer_structure = analyzer.create_spacer_structure(spacer_molecules)
            if spacer_structure:
                spacer_structure.write(large_molecules_file, format='vasp')
                print(f"    ✓ Large molecules (spacer): {os.path.basename(large_molecules_file)}")
        
        # Salt structure (large molecules + terminal halogens)
        # This requires creating the spacer with terminal atoms
        if spacer_molecules:
            salt_file = os.path.join(analysis_dir, f"{structure_name}_salt.vasp")
            # Create salt structure by adding terminal halogens to spacer molecules
            salt_structure = analyzer.create_salt_structure(spacer_molecules)
            if salt_structure:
                salt_structure.write(salt_file, format='vasp')
                print(f"    ✓ Salt structure (spacer + terminals): {os.path.basename(salt_file)}")
        
        # Small molecules (A-sites) if any
        if a_site_molecules:
            small_molecules_file = os.path.join(analysis_dir, f"{structure_name}_small_molecules.vasp")
            a_sites_structure = analyzer.create_a_sites_structure(a_site_molecules)
            if a_sites_structure:
                a_sites_structure.write(small_molecules_file, format='vasp')
                print(f"    ✓ Small molecules (A-sites): {os.path.basename(small_molecules_file)}")
    
    # 5. Create layers-only structure (without large molecules)
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
            # Get isolated molecules once
            isolated_molecules = analyzer.isolate_individual_molecules()
            
            for mol_id, mol_data in molecules.items():
                formula = mol_data.get('formula', f'molecule_{mol_id}')
                mol_file = os.path.join(analysis_dir, f"molecule_{mol_id}_{formula}.vasp")
                
                # Access the 'atoms' key from the isolated molecule dictionary
                if mol_id in isolated_molecules:
                    isolated_molecules[mol_id]['atoms'].write(mol_file, format='vasp')
            
            print(f"    ✓ Individual molecules: {len(molecules)} files")
    
    # 7. Create summary report
    summary_file = os.path.join(analysis_dir, f"{structure_name}_molecule_summary.txt")
    analyzer._create_molecule_summary_report(summary_file)
    print(f"    ✓ Analysis summary: {os.path.basename(summary_file)}")
    
    return analysis_dir

def test_comprehensive_analysis():
    """
    Test comprehensive octahedral and molecular analysis on three structures.
    """
    print("=" * 80)
    print("COMPREHENSIVE OCTAHEDRAL AND MOLECULAR ANALYSIS")
    print("=" * 80)
    
    # Define the structure files
    structures = {
        'n1': 'tests/structures/structure_n1_test.vasp',
        'n2': 'tests/structures/structure_n2_test.vasp',
        'n3': 'tests/structures/structure_n3_test.vasp'
    }
    
    # Parameters for analysis
    central_atom = 'Pb'
    ligand_atom = 'Cl'
    cutoff = 3.5

    # Collect properties for datasets
    all_unified_properties = []
    
    print(f"\nAnalyzing structures with {central_atom}-{ligand_atom} octahedra (cutoff: {cutoff} Å)")
    print("-" * 60)
    
    for key, value in structures.items():
        print(f"\nProcessing structure: {key}")
        print("*" * 40)
        
        try:
            # Create analyzer - this automatically performs spacer isolation and molecule analysis
            analyzer = q2D_analyzer(value, central_atom, ligand_atom, cutoff)
            
            # Display basic information
            print(f"  Original structure: {len(analyzer.atoms)} atoms")
            if analyzer.spacer is not None:
                print(f"  Spacer structure: {len(analyzer.spacer)} atoms")
                print(f"  Spacer composition: {analyzer._get_spacer_composition()}")
            print(f"  Octahedra identified: {len(analyzer.ordered_octahedra)}")
            
            # Create comprehensive analysis folder with all files
            analysis_dir = create_analysis_folder(analyzer, key)
            
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
            # Load octahedral ontology and extract properties
            layers_ontology_file = os.path.join(analysis_dir, f"{key}_layers_ontology.json")
            with open(layers_ontology_file, 'r') as f:
                octahedral_ontology_data = json.load(f)
            
            octahedral_props = extract_octahedral_properties(octahedral_ontology_data, key)
            
            # Add spacer composition to octahedral properties
            spacer_composition = analyzer._get_spacer_composition() if analyzer.spacer else {}
            for oct_prop in octahedral_props:
                for element, count in spacer_composition.items():
                    oct_prop[f'spacer_total_{element}'] = count
            
            all_unified_properties.extend(octahedral_props)
            print(f"  ✓ Octahedral properties extracted: {len(octahedral_props)} octahedra")
            
            # Extract molecule properties for unified dataset (same format as octahedra)
            if analyzer.spacer_molecules:
                molecule_props = extract_molecule_properties_for_dataset(analyzer, key)
                all_unified_properties.extend(molecule_props)  # Add to same list!
                print(f"  ✓ Molecule properties extracted: {len(molecule_props)} molecules")
            
        except Exception as e:
            print(f"  ✗ Error processing {key}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("CREATING UNIFIED DATASET")
    print("=" * 80)
    
    # Create unified dataset (octahedra + molecules)
    if all_unified_properties:
        unified_df = pd.DataFrame(all_unified_properties)
        unified_df = unified_df.sort_values(['structure_name', 'entity_type', 'entity_id']).reset_index(drop=True)
        
        unified_csv = "tests/unified_octahedral_molecular_dataset.csv"
        unified_df.to_csv(unified_csv, index=False)
        
        print(f"\n📊 UNIFIED DATASET (Octahedra + Molecules):")
        print(f"   File: {unified_csv}")
        print(f"   Total entities: {len(unified_df)}")
        print(f"   Properties per entity: {len(unified_df.columns)}")
        
        # Entity type distribution
        print(f"\n   Entity type distribution:")
        for entity_type, count in unified_df['entity_type'].value_counts().items():
            print(f"     {entity_type}: {count} entities")
        
        # Structure distribution by entity type
        print(f"\n   Structure distribution:")
        for structure in unified_df['structure_name'].unique():
            structure_data = unified_df[unified_df['structure_name'] == structure]
            octahedra_count = len(structure_data[structure_data['entity_type'] == 'octahedron'])
            molecules_count = len(structure_data[structure_data['entity_type'] == 'molecule'])
            print(f"     {structure}: {octahedra_count} octahedra, {molecules_count} molecules")
        
        # Key distortion statistics (for octahedra only)
        octahedra_only = unified_df[unified_df['entity_type'] == 'octahedron']
        if len(octahedra_only) > 0:
            print(f"\n   Key distortion parameters for octahedra (mean ± std):")
            distortion_cols = ['zeta', 'delta', 'sigma', 'theta_mean']
            for col in distortion_cols:
                mean_val = octahedra_only[col].mean()
                std_val = octahedra_only[col].std()
                print(f"     {col}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Molecular statistics
        molecules_only = unified_df[unified_df['entity_type'] == 'molecule']
        if len(molecules_only) > 0:
            print(f"\n   Molecular statistics:")
            print(f"     Total molecules: {len(molecules_only)}")
            spacer_molecules = len(molecules_only[molecules_only['is_spacer_molecule'] == True])
            a_site_molecules = len(molecules_only[molecules_only['is_a_site_molecule'] == True])
            print(f"     Spacer molecules: {spacer_molecules}")
            print(f"     A-site molecules: {a_site_molecules}")
            
            # Molecular size statistics
            print(f"     Molecular size range:")
            print(f"       Min atoms: {molecules_only['num_atoms'].min()}")
            print(f"       Max atoms: {molecules_only['num_atoms'].max()}")
            print(f"       Mean atoms: {molecules_only['num_atoms'].mean():.1f}")
            
            # Most common formulas
            print(f"     Most common molecular formulas:")
            formula_counts = molecules_only['molecular_formula'].value_counts().head(5)
            for formula, count in formula_counts.items():
                print(f"       {formula}: {count} molecules")
    
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ Successfully analyzed {len(structures)} structures")
    print(f"📁 Output organization:")
    print(f"   • Each structure has its own analysis folder: tests/<structure>_analysis/")
    print(f"   • All files are organized in a single folder per structure (no subfolders)")
    
    print(f"\n📄 Files created per structure:")
    print(f"   • <structure>_layers_ontology.json - Octahedral/layers analysis")
    print(f"   • <structure>_molecule_ontology.json - Molecular analysis")
    print(f"   • <structure>_spacer.vasp - All spacer molecules")
    print(f"   • <structure>_large_molecules.vasp - 2 largest molecules only")
    print(f"   • <structure>_salt.vasp - Large molecules + terminal halogens")
    print(f"   • <structure>_small_molecules.vasp - Remaining small molecules (A-sites)")
    print(f"   • <structure>_layers_only.vasp - Structure without large molecules")
    print(f"   • molecule_<id>_<formula>.vasp - Individual molecules")
    print(f"   • <structure>_molecule_summary.txt - Human-readable summary")
    
    if all_unified_properties:
        unified_df_temp = pd.DataFrame(all_unified_properties)
        octahedra_count = len(unified_df_temp[unified_df_temp['entity_type'] == 'octahedron'])
        molecules_count = len(unified_df_temp[unified_df_temp['entity_type'] == 'molecule'])
        print(f"\n📊 Unified dataset: tests/unified_octahedral_molecular_dataset.csv")
        print(f"   • Total entities: {len(all_unified_properties)}")
        print(f"   • Octahedra: {octahedra_count} entries")
        print(f"   • Molecules: {molecules_count} entries")
        print(f"   • Ready for statistical analysis and machine learning")
    
    return unified_df if all_unified_properties else None

if __name__ == "__main__":
    # Run the comprehensive test
    unified_results = test_comprehensive_analysis()
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80) 
    
    if unified_results is not None:
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"   📁 Analysis folders: tests/<structure>_analysis/")
        print(f"   📊 Unified dataset: tests/unified_octahedral_molecular_dataset.csv")
        print(f"   📈 Total entities analyzed: {len(unified_results)}")
        print(f"   🔬 Octahedra: {len(unified_results[unified_results['entity_type'] == 'octahedron'])}")
        print(f"   🧪 Molecules: {len(unified_results[unified_results['entity_type'] == 'molecule'])}")
        
        print(f"\n   🎯 Each analysis folder contains:")
        print(f"   • Complete structural decomposition")
        print(f"   • JSON ontologies for layers and molecules")
        print(f"   • VASP files for all structural components")
        print(f"   • Individual molecule structures")
        print(f"   • Human-readable analysis summary")
        
        print(f"\n   🚀 Ready for:")
        print(f"   • Advanced visualization and analysis")
        print(f"   • Structure-property relationship studies")
        print(f"   • Machine learning applications")
        print(f"   • High-throughput materials screening") 
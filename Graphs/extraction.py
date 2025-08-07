import pandas as pd
from pathlib import Path
import re
from ase.io import read
from typing import Optional, Dict, List
import sys
import os
import json
import numpy as np

# Add the parent directory to the path to import SVC_materials
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from SVC_materials.core.analyzer import q2D_analyzer
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SVC_materials analyzer: {e}")
    ANALYSIS_AVAILABLE = False

def extract_comprehensive_properties(root_path: str, cutoff_ref_ligand: float = 3.5) -> pd.DataFrame:
    """
    Extracts comprehensive octahedral and molecular properties using the new ontology system.
    
    Args:
        root_path: The absolute path to the main directory containing the perovskite folders.
        cutoff_ref_ligand: Distance cutoff for identifying ligands (default: 3.5 Å)
                   
    Returns:
        A pandas DataFrame with unified octahedral and molecular properties.
    """
    if not ANALYSIS_AVAILABLE:
        print("Error: SVC_materials analysis not available. Cannot extract properties.")
        return pd.DataFrame()
    
    all_entity_data = []
    base_dir = Path(root_path)
    
    if not base_dir.is_dir():
        print(f"Error: The path '{root_path}' does not exist or is not a directory.")
        return pd.DataFrame()
        
    print(f"Scanning directory: {root_path}")
    print("=" * 80)
    
    for perovskite_dir in base_dir.iterdir():
        if perovskite_dir.is_dir():
            perovskite_name = perovskite_dir.name
            
            # Parse halogen from perovskite_name
            halogen_match = re.search(r'(Br|Cl|I)', perovskite_name)
            halogen = halogen_match.group(1) if halogen_match else None
            
            print(f"\nProcessing perovskite: {perovskite_name}")
            print("-" * 60)
            
            for experiment_dir in perovskite_dir.iterdir():
                if experiment_dir.is_dir():
                    experiment_name = experiment_dir.name
                    
                    print(f"  Analyzing experiment: {experiment_name}")
                    
                    # Parse experiment information
                    parts = experiment_name.split('_')
                    n_slab = None
                    molecule_name = None
                    
                    if len(parts) >= 3 and parts[1].startswith('n'):
                        try:
                            n_slab = int(parts[1][1:])
                            molecule_name = '_'.join(parts[2:])
                        except (ValueError, IndexError):
                            print(f"    Warning: Could not parse n_slab from '{experiment_name}'")
                    
                    # Look for structure files
                    structure_files = ['CONTCAR', 'POSCAR', 'vasprun.xml']
                    structure_path = None
                    
                    for filename in structure_files:
                        potential_path = experiment_dir / filename
                        if potential_path.exists():
                            structure_path = potential_path
                            break
                    
                    if not structure_path:
                        print(f"    ⚠ No structure file found")
                        continue
                    
                    if not halogen:
                        print(f"    ⚠ Could not determine halogen")
                        continue
                    
                    # Extract energy from XML if available
                    energy_data = extract_energy_properties(experiment_dir, halogen, n_slab)
                    
                    try:
                        # Initialize analyzer with comprehensive analysis
                        analyzer = q2D_analyzer(
                            file_path=str(structure_path),
                            b='Pb',  # Central atom
                            x=halogen,  # Ligand atom
                            cutoff_ref_ligand=cutoff_ref_ligand
                        )
                        
                        # Get unified ontology
                        ontology = analyzer.get_ontology()
                                
                        # Extract properties from ontology
                        entity_properties = extract_properties_from_ontology(
                            ontology, analyzer, experiment_name, perovskite_name, 
                            halogen, n_slab, molecule_name, energy_data
                        )
                        
                        all_entity_data.extend(entity_properties)
                        
                        octahedra_count = len([e for e in entity_properties if e['entity_type'] == 'octahedron'])
                        molecules_count = len([e for e in entity_properties if e['entity_type'] == 'molecule'])
                                
                        print(f"    ✓ Extracted {octahedra_count} octahedra and {molecules_count} molecules")
                                
                    except Exception as e:
                        print(f"    ✗ Analysis failed: {str(e)}")
                        continue
    
    print("\n" + "=" * 80)
    print(f"EXTRACTION COMPLETE - Total entities: {len(all_entity_data)}")
    print("=" * 80)
    
    return pd.DataFrame(all_entity_data)

def extract_energy_properties(experiment_dir: Path, halogen: str, n_slab: Optional[int]) -> Dict:
    """Extract energy properties from vasprun.xml if available."""
    xml_path = experiment_dir / 'vasprun.xml'
    energy_data = {
        'energy': None,
        'eslab': None,
        'total_atoms': None,
        'cell_a': None,
        'cell_b': None,
        'cell_c': None,
        'cell_alpha': None,
        'cell_beta': None,
        'cell_gamma': None
    }
    
    if xml_path.exists():
        try:
            atoms = read(str(xml_path), format='vasp-xml')
            energy_data['energy'] = atoms.get_potential_energy()
            energy_data['total_atoms'] = len(atoms)
            
            cell_lengths_angles = atoms.get_cell_lengths_and_angles()
            a, b, c, alpha, beta, gamma = cell_lengths_angles
            energy_data.update({
                'cell_a': float(a),
                'cell_b': float(b),
                'cell_c': float(c),
                'cell_alpha': float(alpha),
                'cell_beta': float(beta),
                'cell_gamma': float(gamma)
            })
            
            # Calculate formation energy if possible
            if halogen and n_slab in [1, 2, 3]:
                halogen_energies = {
                    'Br': -25.57722651 / 2,
                    'Cl': -11.54663106 / 2,
                    'I': -47.93611241 / 2
                }
                EPb = -56.04078139 / 2
                EMA = -38.22939357
                
                EX = halogen_energies[halogen]
                
                if n_slab == 1:
                    NX, NB, NMA = 8, 2, 0
                elif n_slab == 2:
                    NX, NB, NMA = 14, 4, 2
                elif n_slab == 3:
                    NX, NB, NMA = 20, 6, 4
                
                energy_data['eslab'] = energy_data['energy'] - (NX * EX + NB * EPb + NMA * EMA)
                
        except Exception as e:
            print(f"      Warning: Could not extract energy: {str(e)}")
    
    return energy_data

def extract_properties_from_ontology(ontology: Dict, analyzer, experiment_name: str, 
                                   perovskite_name: str, halogen: str, n_slab: Optional[int],
                                   molecule_name: Optional[str], energy_data: Dict) -> List[Dict]:
    """
    Extract comprehensive properties from the unified ontology.
    """
    all_entities = []
    
    # Common experiment information
    common_info = {
        'experiment_name': experiment_name,
        'perovskite_name': perovskite_name,
        'halogen': halogen,
        'n_slab': n_slab,
        'molecule_name': molecule_name,
        'timestamp': ontology.get('experiment', {}).get('timestamp', ''),
    }
    
    # Add energy information
    common_info.update(energy_data)
    
    # Add cell properties
    cell_props = ontology.get('cell_properties', {})
    lattice_params = cell_props.get('lattice_parameters', {})
    composition = cell_props.get('composition', {})
    structure_info = cell_props.get('structure_info', {})
    
    common_info.update({
        'lattice_a': lattice_params.get('A', np.nan),
        'lattice_b': lattice_params.get('B', np.nan),
        'lattice_c': lattice_params.get('C', np.nan),
        'lattice_alpha': lattice_params.get('Alpha', np.nan),
        'lattice_beta': lattice_params.get('Beta', np.nan),
        'lattice_gamma': lattice_params.get('Gamma', np.nan),
        'metal_B': composition.get('metal_B', ''),
        'halogen_X': composition.get('halogen_X', ''),
        'total_structure_atoms': composition.get('number_of_atoms', 0),
        'total_octahedra': composition.get('number_of_octahedra', 0),
        'cell_volume': structure_info.get('cell_volume', np.nan),
        'cutoff_ref_ligand': structure_info.get('cutoff_ref_ligand', np.nan),
    })
    
    # Extract octahedral entities
    octahedra = ontology.get('octahedra', {})
    for oct_id, oct_data in octahedra.items():
        entity = common_info.copy()
        entity.update({
            'entity_id': oct_id,
            'entity_type': 'octahedron',
            'entity_index': int(oct_id.split('_')[-1]) if '_' in oct_id else 0,
        })
        
        # Add octahedral-specific properties
        entity.update(extract_octahedral_properties(oct_data))
        
        # Set molecular properties to defaults for octahedra
        entity.update(get_default_molecular_properties())
        
        all_entities.append(entity)
    
    # Extract molecular entities if available
    if hasattr(analyzer, 'spacer_molecules') and analyzer.spacer_molecules:
        molecules = analyzer.spacer_molecules.get('molecules', {})
        
        # Get spacer information
        spacer_composition = analyzer._get_spacer_composition() if analyzer.spacer else {}
        
        # Separate molecules by size
        spacer_molecules, a_site_molecules = analyzer.separate_molecules_by_size()
        
        for mol_id, mol_data in molecules.items():
            entity = common_info.copy()
            entity.update({
                'entity_id': f"molecule_{mol_id}",
                'entity_type': 'molecule',
                'entity_index': int(mol_id) if str(mol_id).isdigit() else 0,
            })
            
            # Add molecular-specific properties
            entity.update(extract_molecular_properties(mol_data, spacer_composition, 
                                                     mol_id in spacer_molecules, 
                                                     mol_id in a_site_molecules))
            
            # Set octahedral properties to defaults for molecules
            entity.update(get_default_octahedral_properties())
            
            all_entities.append(entity)
    
    return all_entities

def extract_octahedral_properties(oct_data: Dict) -> Dict:
    """Extract octahedral-specific properties from ontology data."""
    properties = {}
    
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
    
    # Ligand information
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
    
    # Bond angle statistics
    bond_angles = oct_data.get('bond_angles', {})
    cis_angles = bond_angles.get('cis_angles', {})
    trans_angles = bond_angles.get('trans_angles', {})
    
    # Calculate angle statistics
    cis_values = [angle_data.get('value', np.nan) for angle_data in cis_angles.values()]
    trans_values = [angle_data.get('value', np.nan) for angle_data in trans_angles.values()]
    
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
    
    # Bond distance range
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
    
    return properties

def extract_molecular_properties(mol_data: Dict, spacer_composition: Dict, 
                                is_spacer: bool, is_a_site: bool) -> Dict:
    """Extract molecular-specific properties."""
    properties = {}
    
    # Basic molecular properties
    symbols = mol_data.get('symbols', [])
    coordinates = mol_data.get('coordinates', [])
    formula = mol_data.get('formula', '')
    
    # Calculate center of mass
    if coordinates:
        coords_array = np.array(coordinates)
        center_of_mass = np.mean(coords_array, axis=0)
    else:
        center_of_mass = [np.nan, np.nan, np.nan]
    
    # Calculate molecular geometry
    if len(coordinates) > 1:
        coords_array = np.array(coordinates)
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
    
    # Count elements
    element_counts = {}
    for symbol in symbols:
        element_counts[symbol] = element_counts.get(symbol, 0) + 1
    
    properties.update({
        'molecular_formula': formula,
        'num_atoms': len(symbols),
        'molecular_size_max': float(max_distance),
        'molecular_size_mean': float(mean_distance),
        'is_spacer_molecule': bool(is_spacer),
        'is_a_site_molecule': bool(is_a_site),
        'center_of_mass_x': float(center_of_mass[0]),
        'center_of_mass_y': float(center_of_mass[1]),
        'center_of_mass_z': float(center_of_mass[2]),
    })
    
    # Add element composition
    for element in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']:
        properties[f'mol_{element}_count'] = element_counts.get(element, 0)
    
    # Add spacer composition context
    for element, count in spacer_composition.items():
        properties[f'spacer_total_{element}'] = count
    
    return properties

def get_default_octahedral_properties() -> Dict:
    """Get default octahedral properties for molecules."""
    return {
        'central_atom_index': np.nan,
        'central_atom_symbol': '',
        'central_x': np.nan,
        'central_y': np.nan,
        'central_z': np.nan,
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

def get_default_molecular_properties() -> Dict:
    """Get default molecular properties for octahedra."""
    properties = {
        'molecular_formula': '',
        'num_atoms': np.nan,
        'molecular_size_max': np.nan,
        'molecular_size_mean': np.nan,
        'is_spacer_molecule': False,
        'is_a_site_molecule': False,
        'center_of_mass_x': np.nan,
        'center_of_mass_y': np.nan,
        'center_of_mass_z': np.nan,
    }
    
    # Add element composition (all zeros for octahedra)
    for element in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']:
        properties[f'mol_{element}_count'] = 0
    
    return properties

if __name__ == '__main__':
    # Replace this path with the actual absolute path to your data directory
    vasp_root_directory = '/home/dotempo/Documents/DION-JACOBSON/BULKS'

    print("🔬 COMPREHENSIVE PEROVSKITE ANALYSIS")
    print("Using unified octahedral and molecular ontology system")
    print("=" * 80)
    
    # Extract comprehensive properties using new ontology system
    results_df = extract_comprehensive_properties(vasp_root_directory)
    
    if not results_df.empty:
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"   Total entities extracted: {len(results_df)}")
        
        # Show entity distribution
        entity_counts = results_df['entity_type'].value_counts()
        for entity_type, count in entity_counts.items():
            print(f"   {entity_type.title()}: {count} entities")
        
        # Show structure distribution
        print(f"\n   Structure distribution:")
        structure_counts = results_df['perovskite_name'].value_counts()
        for structure, count in structure_counts.head(10).items():
            print(f"     {structure}: {count} entities")
        
        # Show halogen distribution
        print(f"\n   Halogen distribution:")
        halogen_counts = results_df['halogen'].value_counts()
        for halogen, count in halogen_counts.items():
            print(f"     {halogen}: {count} entities")

        # Calculate some statistics
        octahedra_df = results_df[results_df['entity_type'] == 'octahedron']
        molecules_df = results_df[results_df['entity_type'] == 'molecule']
        
        if len(octahedra_df) > 0:
            print(f"\n   Octahedral distortion statistics:")
            print(f"     Mean Zeta: {octahedra_df['zeta'].mean():.4f}")
            print(f"     Mean Delta: {octahedra_df['delta'].mean():.6f}")
            print(f"     Mean Sigma: {octahedra_df['sigma'].mean():.4f}°")
            print(f"     Mean bond distance: {octahedra_df['mean_bond_distance'].mean():.3f} Å")
        
        if len(molecules_df) > 0:
            print(f"\n   Molecular statistics:")
            print(f"     Total molecules: {len(molecules_df)}")
            spacer_count = len(molecules_df[molecules_df['is_spacer_molecule'] == True])
            a_site_count = len(molecules_df[molecules_df['is_a_site_molecule'] == True])
            print(f"     Spacer molecules: {spacer_count}")
            print(f"     A-site molecules: {a_site_count}")
            print(f"     Mean molecular size: {molecules_df['molecular_size_max'].mean():.2f} Å")

        # Save to CSV
        output_file = 'comprehensive_perovskite_properties.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")
        print(f"   Columns: {len(results_df.columns)}")
        print(f"   Rows: {len(results_df)}")
        
        print(f"\n📁 This file contains:")
        print(f"   • Both octahedral and molecular entities as rows")
        print(f"   • Comprehensive distortion parameters for octahedra")
        print(f"   • Molecular composition and size data")
        print(f"   • Energy properties where available")
        print(f"   • Cell parameters and structural information")
        print(f"   • Ready for advanced statistical analysis and visualization")
        
    else:
        print("❌ No data extracted. Check your directory path and file structure.")

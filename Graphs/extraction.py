import pandas as pd
from pathlib import Path
import re
from ase.io import read
from typing import Optional
import sys
import os

# Add the parent directory to the path to import SVC_materials
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from SVC_materials.core.analyzer import q2D_analyzer
    DISTORTION_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import distortion analysis: {e}")
    DISTORTION_ANALYSIS_AVAILABLE = False

def name_properties(root_path: str) -> pd.DataFrame:
    """
    Parses a directory of VASP calculations and extracts information into a pandas DataFrame.

    The script assumes a directory structure like:
    - BULKS/
      - MAPbBr3/
        - MAPbBr3_n1_MoleculeName1
        - MAPbBr3_n2_MoleculeName2
      - MAPbCl3/
        - ...

    Args:
        root_path: The absolute path to the main directory containing the perovskite folders.
                   Example: '/home/dotempo/Documents/DION-JACOBSON/BULKS'

    Returns:
        A pandas DataFrame with columns: 'Experiment', 'Perovskite', 'Halogen', 'N_Slab', 'Molecule'.
    """
    # A list to hold the dictionaries of data for each calculation
    all_calculations_data = []

    # Convert the input string path to a Path object for easier manipulation
    base_dir = Path(root_path)

    # Check if the provided path exists and is a directory
    if not base_dir.is_dir():
        print(f"Error: The path '{root_path}' does not exist or is not a directory.")
        return pd.DataFrame() # Return an empty DataFrame

    # Iterate through the first level of directories (e.g., MAPbBr3, MAPbCl3)
    for perovskite_dir in base_dir.iterdir():
        if perovskite_dir.is_dir():
            perovskite_name = perovskite_dir.name

            # --- Halogen Extraction ---
            # Use regex to find Br, Cl, or I in the perovskite name
            halogen_match = re.search(r'(Br|Cl|I)', perovskite_name)
            halogen = halogen_match.group(1) if halogen_match else None

            # Iterate through the second level of directories (the actual calculation folders)
            for experiment_dir in perovskite_dir.iterdir():
                if experiment_dir.is_dir():
                    experiment_name = experiment_dir.name

                    # --- Data Extraction using split() ---
                    parts = experiment_name.split('_')
                    
                    n_slab = None
                    molecule_name = None # Default to None if parsing fails

                    # Expected format: Perovskite_nSlab_Molecule
                    if len(parts) >= 3 and parts[1].startswith('n'):
                        try:
                            # Extract n_slab from 'n1', 'n2', etc. by removing the 'n'
                            n_slab = int(parts[1][1:])
                            # Reconstruct the molecule name from all remaining parts
                            molecule_name = '_'.join(parts[2:])
                        except (ValueError, IndexError):
                            # This handles cases where the part after 'n' is not a number
                            print(f"Warning: Could not parse n_slab from '{experiment_name}'.")
                    else:
                        print(f"Warning: Unexpected experiment name format: '{experiment_name}'.")

                    # Append the extracted data as a dictionary to our list
                    all_calculations_data.append({
                        'Experiment': experiment_name,
                        'Perovskite': perovskite_name,
                        'Halogen': halogen,
                        'N_Slab': n_slab,
                        'Molecule': molecule_name,
                    })

    # Create the pandas DataFrame from the list of dictionaries
    # The columns will be created automatically from the keys
    df = pd.DataFrame(all_calculations_data)
    return df

def distortion_properties(root_path: str, cutoff_ref_ligand: float = 3.5) -> pd.DataFrame:
    """
    Extracts octahedral distortion parameters from VASP calculations.
    
    Args:
        root_path: The absolute path to the main directory containing the perovskite folders.
        cutoff_ref_ligand: Distance cutoff for identifying ligands (default: 3.5 Å)
                   
    Returns:
        A pandas DataFrame with octahedral distortion parameters.
    """
    if not DISTORTION_ANALYSIS_AVAILABLE:
        print("Warning: Distortion analysis not available. Returning empty DataFrame.")
        return pd.DataFrame()
    
    all_distortion_data = []
    base_dir = Path(root_path)
    
    if not base_dir.is_dir():
        print(f"Error: The path '{root_path}' does not exist or is not a directory.")
        return pd.DataFrame()
        
    for perovskite_dir in base_dir.iterdir():
        if perovskite_dir.is_dir():
            perovskite_name = perovskite_dir.name
            
            # Parse halogen from perovskite_name
            halogen_match = re.search(r'(Br|Cl|I)', perovskite_name)
            halogen = halogen_match.group(1) if halogen_match else None
            
            for experiment_dir in perovskite_dir.iterdir():
                if experiment_dir.is_dir():
                    experiment_name = experiment_dir.name
                    
                    # Look for CONTCAR or POSCAR files
                    structure_files = ['CONTCAR', 'POSCAR', 'vasprun.xml']
                    structure_path = None
                    
                    for filename in structure_files:
                        potential_path = experiment_dir / filename
                        if potential_path.exists():
                            structure_path = potential_path
                            break
                    
                    # Initialize distortion data with default values
                    distortion_data = {
                        'Experiment': experiment_name,
                        'Perovskite': perovskite_name,
                        'NumOctahedra': 0,
                        'MeanZeta': None,
                        'StdZeta': None,
                        'MinZeta': None,
                        'MaxZeta': None,
                        'MeanDelta': None,
                        'StdDelta': None,
                        'MinDelta': None,
                        'MaxDelta': None,
                        'MeanSigma': None,
                        'StdSigma': None,
                        'MinSigma': None,
                        'MaxSigma': None,
                        'MeanTheta': None,
                        'StdTheta': None,
                        'MinTheta': None,
                        'MaxTheta': None,
                        'MeanBondDistance': None,
                        'StdBondDistance': None,
                        'MeanOctaVolume': None,
                        'StdOctaVolume': None,
                        'DistortionAnalysisSuccess': False
                    }
                    
                    if structure_path and halogen:
                        try:
                            # Initialize analyzer
                            analyzer = q2D_analyzer(
                                file_path=str(structure_path),
                                b='Pb',  # Central atom
                                x=halogen,  # Ligand atom
                                cutoff_ref_ligand=cutoff_ref_ligand
                            )
                            
                            # Calculate distortions
                            distortions = analyzer.calculate_octahedral_distortions()
                            
                            if distortions:
                                # Get summary DataFrame
                                summary_df = analyzer.get_distortion_summary()
                                
                                # Calculate statistics for each parameter
                                distortion_data.update({
                                    'NumOctahedra': len(summary_df),
                                    'MeanZeta': summary_df['zeta'].mean(),
                                    'StdZeta': summary_df['zeta'].std(),
                                    'MinZeta': summary_df['zeta'].min(),
                                    'MaxZeta': summary_df['zeta'].max(),
                                    'MeanDelta': summary_df['delta'].mean(),
                                    'StdDelta': summary_df['delta'].std(),
                                    'MinDelta': summary_df['delta'].min(),
                                    'MaxDelta': summary_df['delta'].max(),
                                    'MeanSigma': summary_df['sigma'].mean(),
                                    'StdSigma': summary_df['sigma'].std(),
                                    'MinSigma': summary_df['sigma'].min(),
                                    'MaxSigma': summary_df['sigma'].max(),
                                    'MeanTheta': summary_df['theta_mean'].mean(),
                                    'StdTheta': summary_df['theta_mean'].std(),
                                    'MinTheta': summary_df['theta_mean'].min(),
                                    'MaxTheta': summary_df['theta_mean'].max(),
                                    'MeanBondDistance': summary_df['mean_bond_distance'].mean(),
                                    'StdBondDistance': summary_df['mean_bond_distance'].std(),
                                    'MeanOctaVolume': summary_df['octahedral_volume'].mean(),
                                    'StdOctaVolume': summary_df['octahedral_volume'].std(),
                                    'DistortionAnalysisSuccess': True
                                })
                                
                                print(f"✓ Analyzed {len(summary_df)} octahedra in {experiment_name}")
                            else:
                                print(f"⚠ No octahedra found in {experiment_name}")
                                
                        except Exception as e:
                            print(f"✗ Distortion analysis failed for {experiment_name}: {str(e)}")
                    else:
                        if not structure_path:
                            print(f"⚠ No structure file found for {experiment_name}")
                        if not halogen:
                            print(f"⚠ Could not determine halogen for {experiment_name}")
                    
                    all_distortion_data.append(distortion_data)
    
    return pd.DataFrame(all_distortion_data)

def xml_properties(root_path: str) -> pd.DataFrame:
    """
    Extracts energy information from vasprun.xml files in VASP calculations.
    
    Args:
        root_path: The absolute path to the main directory containing the perovskite folders.
                   Example: '/home/dotempo/Documents/DION-JACOBSON/BULKS'
                   
    Returns:
        A pandas DataFrame with columns: 'Experiment', 'Perovskite', 'Energy'.
    """
    all_calculations_data = []
    base_dir = Path(root_path)
    
    if not base_dir.is_dir():
        print(f"Error: The path '{root_path}' does not exist or is not a directory.")
        return pd.DataFrame()
        
    for perovskite_dir in base_dir.iterdir():
        if perovskite_dir.is_dir():
            perovskite_name = perovskite_dir.name
            
            for experiment_dir in perovskite_dir.iterdir():
                if experiment_dir.is_dir():
                    experiment_name = experiment_dir.name
                    xml_path = experiment_dir / 'vasprun.xml'
                    
                    energy: Optional[float] = None
                    eslab: Optional[float] = None
                    total_atoms: Optional[int] = None
                    
                    # Parse halogen from perovskite_name
                    halogen_match = re.search(r'(Br|Cl|I)', perovskite_name)
                    halogen = halogen_match.group(1) if halogen_match else None
                    
                    # Parse n_slab from experiment_name
                    parts = experiment_name.split('_')
                    n_slab = None
                    if len(parts) >= 3 and parts[1].startswith('n'):
                        try:
                            n_slab = int(parts[1][1:])
                        except ValueError:
                            pass
                    
                    if xml_path.exists():
                        try:
                            atoms = read(str(xml_path), format='vasp-xml')
                            energy = atoms.get_potential_energy()
                            
                            if halogen and n_slab in [1, 2, 3]:
                                # Hardcoded values
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
                                
                                eslab = energy - (NX * EX + NB * EPb + NMA * EMA)
                                total_atoms = len(atoms)
                                cell_lengths_angles = atoms.get_cell_lengths_and_angles()
                                a, b, c, alpha, beta, gamma = cell_lengths_angles
                        except Exception as e:
                            print(f"Warning: Could not read energy from {xml_path}: {str(e)}")
                    
                    all_calculations_data.append({
                        'Experiment': experiment_name,
                        'Perovskite': perovskite_name,
                        'Eslab': eslab,
                        'TotalAtoms': total_atoms if 'total_atoms' in locals() else None,
                        'A': a if 'a' in locals() else None,
                        'B': b if 'b' in locals() else None,
                        'C': c if 'c' in locals() else None,
                        'Alpha': alpha if 'alpha' in locals() else None,
                        'Beta': beta if 'beta' in locals() else None,
                        'Gamma': gamma if 'gamma' in locals() else None
                    })
    
    return pd.DataFrame(all_calculations_data)

if __name__ == '__main__':
    # Replace this path with the actual absolute path to your 'BULKS' directory
    vasp_root_directory = '/home/dotempo/Documents/DION-JACOBSON/BULKS'

    # Get all property DataFrames
    print("Extracting name properties...")
    names_df = name_properties(vasp_root_directory)
    
    print("Extracting energy properties...")
    energies_df = xml_properties(vasp_root_directory)
    
    print("Extracting octahedral distortion properties...")
    distortion_df = distortion_properties(vasp_root_directory)
    
    # Merge the DataFrames on common columns
    if not names_df.empty and not energies_df.empty:
        # First merge names and energies
        results_df = pd.merge(
            names_df, 
            energies_df[['Experiment', 'Eslab', 'TotalAtoms', 'A', 'B', 'C', 'Alpha', 'Beta', 'Gamma']], 
            on='Experiment',
            how='left'
        )
        
        # Then merge with distortion data if available
        if not distortion_df.empty:
            distortion_columns = [col for col in distortion_df.columns if col != 'Perovskite']
            results_df = pd.merge(
                results_df,
                distortion_df[distortion_columns],
                on='Experiment',
                how='left'
            )
            print(f"✓ Added {len(distortion_columns)-1} distortion analysis columns")
        else:
            print("⚠ No distortion data available")
        
        print("Successfully created merged DataFrame:")
        print(results_df.to_string())

        # Load molecule data
        try:
            molecule_df = pd.read_csv('molecule_data.csv')
            molecule_df = molecule_df.rename(columns={
                'molecule': 'Molecule',
                'energy_eV': 'Emol',
                'numb_carbons': 'NCarbonsMol',
                'numb_nitrogens': 'NNitrogensMol',
                'numb_hydrogens': 'NHydrogensMol',
                'family': 'FamilyMol',
                'parity': 'ParityMol'
            })

            # Merge with molecule properties
            results_df = pd.merge(
                results_df,
                molecule_df,
                on='Molecule',
                how='left'
            )

            # Calculate NormEnergy
            results_df['NormEnergy'] = (results_df['Eslab'] - 2 * results_df['Emol']) / results_df['TotalAtoms']

            # Calculate normalized Emol
            results_df['MolAtoms'] = results_df['NCarbonsMol'] + results_df['NNitrogensMol'] + results_df['NHydrogensMol']
            results_df['NormEmol'] = results_df['Emol'] / results_df['MolAtoms']

            print("✓ Successfully merged with molecule data")
        except FileNotFoundError:
            print("⚠ molecule_data.csv not found, skipping molecule properties")

        # Save to CSV
        results_df.to_csv('perovskites.csv', index=False)
        print(f"✓ Results saved to perovskites.csv with {len(results_df.columns)} columns")
        
        # Print summary of distortion analysis
        if 'DistortionAnalysisSuccess' in results_df.columns:
            successful_analyses = results_df['DistortionAnalysisSuccess'].sum()
            total_experiments = len(results_df)
            print(f"\n=== Distortion Analysis Summary ===")
            print(f"Successful analyses: {successful_analyses}/{total_experiments}")
            
            if successful_analyses > 0:
                print(f"Average number of octahedra per structure: {results_df['NumOctahedra'].mean():.1f}")
                print(f"Average Zeta parameter: {results_df['MeanZeta'].mean():.4f}")
                print(f"Average Delta parameter: {results_df['MeanDelta'].mean():.6f}")
                print(f"Average Sigma parameter: {results_df['MeanSigma'].mean():.4f}°")
    else:
        print("Error: One or both DataFrames are empty")

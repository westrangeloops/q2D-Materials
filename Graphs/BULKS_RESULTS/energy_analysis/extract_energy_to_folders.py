import os
import json
from ase.io import read
from ase.calculators.vasp import Vasp

def extract_energy_from_vasp_folder(vasp_folder_path, experiment_name):
    """
    Extract normalized energy and ASE properties directly from VASP experiment folder.
    Uses the same normalization algorithm as extraction.py
    
    Returns:
        dict: Dictionary with normalized energy and other ASE properties
    """
    energy_data = {}
    
    try:
        # Look for vasprun.xml first (preferred), then OUTCAR
        vasprun_path = os.path.join(vasp_folder_path, 'vasprun.xml')
        outcar_path = os.path.join(vasp_folder_path, 'OUTCAR')
        
        atoms = None
        total_energy = None
        
        # Try to read from vasprun.xml first
        if os.path.exists(vasprun_path):
            try:
                atoms = read(vasprun_path, format='vasp-xml')
                total_energy = atoms.get_potential_energy()
            except Exception as e:
                print(f"Failed to read vasprun.xml: {e}")
                atoms = None
        
        # Fallback to OUTCAR if vasprun.xml failed
        if atoms is None and os.path.exists(outcar_path):
            try:
                # Try to read structure from CONTCAR
                contcar_path = os.path.join(vasp_folder_path, 'CONTCAR')
                if os.path.exists(contcar_path):
                    atoms = read(contcar_path, format='vasp')
                
                # Extract energy from OUTCAR
                with open(outcar_path, 'r') as f:
                    for line in f:
                        if 'free  energy   TOTEN' in line:
                            total_energy = float(line.split()[-2])  # Energy in eV
                            break
            except Exception as e:
                print(f"Failed to read OUTCAR: {e}")
        
        if atoms is not None and total_energy is not None:
            # Extract halogen and n_layers from experiment name
            halogen = None
            n_layers = None
            
            # Parse experiment name to get halogen and layers
            if 'Cl' in experiment_name:
                halogen = 'Cl'
            elif 'Br' in experiment_name:
                halogen = 'Br'
            elif 'I' in experiment_name:
                halogen = 'I'
            
            # Extract n_layers
            import re
            n_match = re.search(r'_n(\d+)_', experiment_name)
            if n_match:
                n_layers = int(n_match.group(1))
            
            # Calculate basic properties
            n_atoms = len(atoms)
            
            # Get chemical formula and atom counts
            formula = atoms.get_chemical_formula()
            symbols = atoms.get_chemical_symbols()
            
            atom_counts = {}
            for symbol in set(symbols):
                atom_counts[f'n_{symbol}'] = symbols.count(symbol)
            
            # Calculate formation energy
            e_nergy = None
            if halogen and n_layers in [1, 2, 3]:
                # Reference energies from extraction.py
                halogen_energies = {
                    'Br': -25.57722651 / 2,
                    'Cl': -11.54663106 / 2,
                    'I': -47.93611241 / 2
                }
                EPb = -56.04078139 / 2
                EMA = -38.22939357
                
                EX = halogen_energies[halogen]
                
                # Number of atoms by layer (from extraction.py)
                if n_layers == 1:
                    NX, NB, NMA = 8, 2, 0
                elif n_layers == 2:
                    NX, NB, NMA = 14, 4, 2
                elif n_layers == 3:
                    NX, NB, NMA = 20, 6, 4
                
                # Calculate formation energy
                e_nergy = total_energy - (NX * EX + NB * EPb + NMA * EMA)
            
            # Create energy data dictionary (non-structural properties only)
            energy_data = {
                'total_energy_eV': total_energy,
                'energy_per_atom_eV': e_nergy / n_atoms,
                'n_total_atoms': n_atoms,
                'chemical_formula': formula,
                'halogen': halogen,
                'n_layers': n_layers,
                **atom_counts  # Add atom counts (e.g., n_Pb, n_Br, n_C, etc.)
            }
            
            return energy_data
        else:
            print(f"Could not extract energy from {vasp_folder_path}")
            return None
            
    except Exception as e:
        print(f"Error extracting from {vasp_folder_path}: {e}")
        return None

def find_vasp_folders_and_extract(base_dir):
    """
    Find VASP experiment folders and extract energy data to analysis folders.
    Handles truncated analysis directory names by fuzzy matching with VASP folders.
    """
    print(f"Scanning for VASP folders in: {base_dir}")
    
    created_files = 0
    failed_extractions = 0
    skipped_dirs = 0
    
    # Look for analysis directories (exclude utility directories)
    analysis_dirs = [d for d in os.listdir(base_dir) if d.endswith('_analysis')]
    utility_dirs = ['energy_analysis', 'lattice_parameter_analysis', 'penetration_depth']
    analysis_dirs = [d for d in analysis_dirs if d not in utility_dirs]
    
    print(f"Found {len(analysis_dirs)} experiment analysis directories to process")
    
    # Build a mapping of halogen folders to their VASP experiments
    vasp_base_dir = '/home/dotempo/Documents/DION-JACOBSON/BULKS'
    halogen_to_experiments = {}
    
    for halogen_folder in ['MAPbCl3', 'MAPbBr3', 'MAPbI3']:
        halogen_path = os.path.join(vasp_base_dir, halogen_folder)
        if os.path.exists(halogen_path):
            experiments = [d for d in os.listdir(halogen_path) 
                         if os.path.isdir(os.path.join(halogen_path, d))]
            halogen_to_experiments[halogen_folder] = experiments
            print(f"Found {len(experiments)} experiments in {halogen_folder}")
    
    for analysis_dir in analysis_dirs:
        try:
            # Get the experiment name (remove _analysis suffix)
            experiment_name = analysis_dir.replace('_analysis', '')
            
            # Path to the analysis folder
            analysis_path = os.path.join(base_dir, analysis_dir)
            
            # Extract halogen from experiment name to determine subfolder
            halogen_folder = None
            if 'Cl' in experiment_name:
                halogen_folder = 'MAPbCl3'
            elif 'Br' in experiment_name:
                halogen_folder = 'MAPbBr3'
            elif 'I' in experiment_name:
                halogen_folder = 'MAPbI3'
            
            if not halogen_folder:
                print(f"Could not determine halogen for {experiment_name}")
                failed_extractions += 1
                continue
            
            # Try exact match first
            vasp_folder = None
            exact_path = os.path.join(vasp_base_dir, halogen_folder, experiment_name)
            
            if os.path.exists(exact_path):
                vasp_folder = exact_path
            else:
                # Try fuzzy matching for truncated names
                if halogen_folder in halogen_to_experiments:
                    for vasp_exp in halogen_to_experiments[halogen_folder]:
                        # Check if the analysis name is a truncated version of the vasp name
                        if vasp_exp.startswith(experiment_name):
                            candidate_path = os.path.join(vasp_base_dir, halogen_folder, vasp_exp)
                            if (os.path.exists(os.path.join(candidate_path, 'vasprun.xml')) or 
                                os.path.exists(os.path.join(candidate_path, 'OUTCAR'))):
                                vasp_folder = candidate_path
                                print(f"Matched truncated '{experiment_name}' to full '{vasp_exp}'")
                                break
            
            if vasp_folder:
                # Use the full VASP folder name for energy extraction
                full_experiment_name = os.path.basename(vasp_folder)
                print(f"Processing {full_experiment_name}...")
                
                # Extract energy data
                energy_data = extract_energy_from_vasp_folder(vasp_folder, full_experiment_name)
                
                if energy_data:
                    # Save as JSON in the analysis folder
                    json_path = os.path.join(analysis_path, 'energy_properties.json')
                    
                    with open(json_path, 'w') as f:
                        json.dump(energy_data, f, indent=2)
                    
                    created_files += 1
                    if created_files % 10 == 0:
                        print(f"Created {created_files} energy JSON files...")
                else:
                    failed_extractions += 1
                    print(f"Failed to extract energy data from {vasp_folder}")
            else:
                failed_extractions += 1
                print(f"No VASP folder found for {experiment_name}")
                
        except Exception as e:
            print(f"Error processing {analysis_dir}: {e}")
            failed_extractions += 1
            continue
    
    print(f"\nExtraction complete!")
    print(f"Created {created_files} energy JSON files")
    print(f"Failed extractions: {failed_extractions}")
    
    return created_files

def main():
    base_dir = '/home/dotempo/Documents/REPO/SVC-Materials/Graphs/BULKS_RESULTS'
    vasp_base_dir = '/home/dotempo/Documents/DION-JACOBSON/BULKS'
    
    print("Starting energy extraction directly from VASP folders...")
    print(f"Analysis results directory: {base_dir}")
    print(f"VASP calculations directory: {vasp_base_dir}")
    print("Will search for VASP experiment folders and extract energy + ASE properties")
    
    created = find_vasp_folders_and_extract(base_dir)
    
    if created > 0:
        print(f"\nSuccessfully created {created} energy_properties.json files!")
        print("Each analysis folder now contains:")
        print("- energy_properties.json with VASP energies and atom counts")
        print("- No structural properties (those are handled separately)")
    else:
        print("No energy files were created.")
        print("Please check that VASP folders exist and contain OUTCAR files.")

if __name__ == "__main__":
    main() 
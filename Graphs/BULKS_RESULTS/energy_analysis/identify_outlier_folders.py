import pandas as pd
import os
import shutil
from pathlib import Path

def identify_and_move_outlier_folders():
    """
    Identify folders with energy values outside -2.5 to 0 range and move them to error_calculations.
    """
    # Load the CSV data
    csv_path = '/home/dotempo/Documents/REPO/SVC-Materials/Graphs/perovskites.csv'
    base_dir = '/home/dotempo/Documents/REPO/SVC-Materials/Graphs/BULKS_RESULTS'
    error_dir = os.path.join(base_dir, 'error_calculations')
    
    print(f"Loading energy data from CSV: {csv_path}")
    
    try:
        df_csv = pd.read_csv(csv_path)
        print(f"Loaded {len(df_csv)} energy records from CSV")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Create error_calculations directory if it doesn't exist
    os.makedirs(error_dir, exist_ok=True)
    print(f"Created/verified error directory: {error_dir}")
    
    # Get analysis directories
    analysis_dirs = [d for d in os.listdir(base_dir) if d.endswith('_analysis')]
    utility_dirs = ['energy_analysis', 'lattice_parameter_analysis', 'penetration_depth', 'error_calculations']
    analysis_dirs = [d for d in analysis_dirs if d not in utility_dirs]
    
    print(f"Found {len(analysis_dirs)} experiment analysis directories to check")
    
    # Track statistics
    outlier_folders = []
    good_folders = []
    unmatched_folders = []
    
    # Define energy range
    min_energy = -2.5
    max_energy = 0.0
    
    for analysis_dir in analysis_dirs:
        try:
            # Get the experiment name (remove _analysis suffix)
            experiment_name = analysis_dir.replace('_analysis', '')
            
            # Try exact match first
            csv_match = df_csv[df_csv['Experiment'] == experiment_name]
            
            if csv_match.empty:
                # Try fuzzy matching for truncated names
                possible_matches = df_csv[df_csv['Experiment'].str.startswith(experiment_name)]
                
                if len(possible_matches) == 1:
                    csv_match = possible_matches
                elif len(possible_matches) > 1:
                    # If multiple matches, take the shortest one (most likely match)
                    csv_match = possible_matches.loc[[possible_matches['Experiment'].str.len().idxmin()]]
            
            if not csv_match.empty:
                row = csv_match.iloc[0]
                norm_energy = row['NormEnergy']
                experiment_full = row['Experiment']
                
                # Check if energy is outside the acceptable range
                if norm_energy < min_energy or norm_energy > max_energy:
                    outlier_folders.append({
                        'analysis_dir': analysis_dir,
                        'experiment_full': experiment_full,
                        'norm_energy': norm_energy,
                        'reason': f"Energy {norm_energy:.3f} outside range [{min_energy}, {max_energy}]"
                    })
                    print(f"OUTLIER: {analysis_dir} -> Energy: {norm_energy:.3f} eV/atom")
                else:
                    good_folders.append({
                        'analysis_dir': analysis_dir,
                        'experiment_full': experiment_full,
                        'norm_energy': norm_energy
                    })
            else:
                unmatched_folders.append(analysis_dir)
                print(f"UNMATCHED: No CSV match found for {analysis_dir}")
                
        except Exception as e:
            print(f"Error processing {analysis_dir}: {e}")
            unmatched_folders.append(analysis_dir)
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print("ENERGY OUTLIER ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total folders analyzed: {len(analysis_dirs)}")
    print(f"Good folders (energy in range): {len(good_folders)}")
    print(f"Outlier folders (energy out of range): {len(outlier_folders)}")
    print(f"Unmatched folders: {len(unmatched_folders)}")
    
    # Show outlier details
    if outlier_folders:
        print(f"\n{'='*60}")
        print("OUTLIER FOLDERS TO BE MOVED:")
        print(f"{'='*60}")
        for outlier in outlier_folders:
            print(f"Folder: {outlier['analysis_dir']}")
            print(f"  Full name: {outlier['experiment_full']}")
            print(f"  Energy: {outlier['norm_energy']:.3f} eV/atom")
            print(f"  Reason: {outlier['reason']}")
            print()
    
    # Ask for confirmation before moving
    if outlier_folders:
        response = input(f"\nDo you want to move {len(outlier_folders)} outlier folders to error_calculations? (y/n): ")
        
        if response.lower() == 'y':
            moved_count = 0
            for outlier in outlier_folders:
                source_path = os.path.join(base_dir, outlier['analysis_dir'])
                dest_path = os.path.join(error_dir, outlier['analysis_dir'])
                
                try:
                    if os.path.exists(source_path):
                        shutil.move(source_path, dest_path)
                        print(f"Moved: {outlier['analysis_dir']} -> error_calculations/")
                        moved_count += 1
                    else:
                        print(f"Source not found: {source_path}")
                except Exception as e:
                    print(f"Error moving {outlier['analysis_dir']}: {e}")
            
            print(f"\nSuccessfully moved {moved_count} folders to error_calculations/")
            print(f"Remaining good folders: {len(good_folders)}")
        else:
            print("Operation cancelled. No folders were moved.")
    else:
        print("\nNo outlier folders found. All energies are within the acceptable range!")
    
    # Save report
    report_path = os.path.join(base_dir, 'energy_outlier_report.txt')
    with open(report_path, 'w') as f:
        f.write("ENERGY OUTLIER ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Energy range: [{min_energy}, {max_energy}] eV/atom\n")
        f.write(f"Total folders: {len(analysis_dirs)}\n")
        f.write(f"Good folders: {len(good_folders)}\n")
        f.write(f"Outlier folders: {len(outlier_folders)}\n")
        f.write(f"Unmatched folders: {len(unmatched_folders)}\n\n")
        
        if outlier_folders:
            f.write("OUTLIER FOLDERS:\n")
            f.write("-" * 30 + "\n")
            for outlier in outlier_folders:
                f.write(f"{outlier['analysis_dir']}: {outlier['norm_energy']:.3f} eV/atom\n")
        
        f.write(f"\nGOOD FOLDERS:\n")
        f.write("-" * 20 + "\n")
        for good in good_folders:
            f.write(f"{good['analysis_dir']}: {good['norm_energy']:.3f} eV/atom\n")
    
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    identify_and_move_outlier_folders() 
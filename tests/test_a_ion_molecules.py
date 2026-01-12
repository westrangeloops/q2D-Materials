"""Test script for analyzing all A-ion molecules from CSV as DJ and RP spacers.

This script reads all molecules from A-ion_data.csv and tests them as:
- DJ spacers (bifunctional, requires 2 terminal NH2/NH3 groups)
- RP spacers (monofunctional, requires 1+ terminal NH2/NH3 groups)

Usage:
    nix develop -c python3 test_a_ion_molecules.py

Output:
    - Console summary with statistics
    - a_ion_molecules_full_results.json: Complete results for all molecules
    - dj_spacers_2_nh3.json: DJ spacers with exactly 2 NH3 groups
    - rp_spacers_1_nh3.json: RP spacers with exactly 1 NH3 group
    - a_ion_molecules_summary.json: Summary statistics
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
from q2D_Materials.analyzer import q2D_analyzer


def read_molecules_from_csv(csv_path: str) -> List[Dict]:
    """Read molecules from CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file
        
    Returns
    -------
    List[Dict]
        List of molecule dictionaries with ID, Abbreviation, SMILE, etc.
    """
    molecules = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows without SMILES
            if row.get('SMILE') and row['SMILE'].strip():
                molecules.append({
                    'ID': row.get('ID', ''),
                    'Abbreviation': row.get('Abbreviation', ''),
                    'Common_name': row.get('Common_name', ''),
                    'SMILE': row['SMILE'].strip(),
                    'Molecular_formula': row.get('Molecular_formula', ''),
                })
    
    return molecules


def analyze_molecules(molecules: List[Dict], verbose: bool = False) -> Dict:
    """Analyze all molecules as DJ and RP spacers.
    
    Parameters
    ----------
    molecules : List[Dict]
        List of molecule dictionaries
    verbose : bool
        Print detailed information for each molecule
        
    Returns
    -------
    Dict
        Results dictionary with statistics and detailed results
    """
    analyzer = q2D_analyzer()
    
    results = {
        'total_molecules': len(molecules),
        'processed': 0,
        'failed_smiles': 0,
        'valid_dj': [],
        'valid_rp': [],
        'invalid_dj': [],
        'invalid_rp': [],
        'dj_with_2_nh3': [],
        'rp_with_1_nh3': [],
        'statistics': {
            'valid_dj_count': 0,
            'valid_rp_count': 0,
            'dj_with_2_nh3_count': 0,
            'rp_with_1_nh3_count': 0,
            'invalid_dj_count': 0,
            'invalid_rp_count': 0,
        }
    }
    
    print(f"Analyzing {len(molecules)} molecules from CSV...")
    print("=" * 80)
    
    for i, mol in enumerate(molecules, 1):
        smiles = mol['SMILE']
        mol_id = mol.get('ID', 'N/A')
        abbrev = mol.get('Abbreviation', 'N/A')
        name = mol.get('Common_name', 'N/A')
        
        if verbose:
            print(f"\n[{i}/{len(molecules)}] {abbrev} ({name})")
            print(f"  SMILES: {smiles}")
        
        try:
            # Test as DJ spacer
            result_dj = analyzer.analyze_molecule_as_dj_spacer(smiles, allow_nh2=True)
            
            # Test as RP spacer
            result_rp = analyzer.analyze_molecule_as_rp_spacer(smiles, allow_nh2=True)
            
            results['processed'] += 1
            
            # Count NH3 groups (not NH2)
            nh3_groups_dj = [g for g in result_dj.terminal_groups if g.group_type == "NH3"]
            nh3_groups_rp = [g for g in result_rp.terminal_groups if g.group_type == "NH3"]
            
            mol_result = {
                'ID': mol_id,
                'Abbreviation': abbrev,
                'Common_name': name,
                'SMILE': smiles,
                'Molecular_formula': mol.get('Molecular_formula', ''),
                'dj_valid': result_dj.is_valid,
                'dj_reason': result_dj.reason,
                'dj_terminal_count': len(result_dj.terminal_groups),
                'dj_nh3_count': len(nh3_groups_dj),
                'dj_nh2_count': len(result_dj.terminal_groups) - len(nh3_groups_dj),
                'dj_paths_count': len(result_dj.valid_paths),
                'rp_valid': result_rp.is_valid,
                'rp_reason': result_rp.reason,
                'rp_terminal_count': len(result_rp.terminal_groups),
                'rp_nh3_count': len(nh3_groups_rp),
                'rp_nh2_count': len(result_rp.terminal_groups) - len(nh3_groups_rp),
            }
            
            # Categorize results
            if result_dj.is_valid:
                results['valid_dj'].append(mol_result)
                results['statistics']['valid_dj_count'] += 1
                
                # Check if has exactly 2 NH3 groups
                if len(nh3_groups_dj) == 2:
                    results['dj_with_2_nh3'].append(mol_result)
                    results['statistics']['dj_with_2_nh3_count'] += 1
            else:
                results['invalid_dj'].append(mol_result)
                results['statistics']['invalid_dj_count'] += 1
            
            if result_rp.is_valid:
                results['valid_rp'].append(mol_result)
                results['statistics']['valid_rp_count'] += 1
                
                # Check if has exactly 1 NH3 group
                if len(nh3_groups_rp) == 1:
                    results['rp_with_1_nh3'].append(mol_result)
                    results['statistics']['rp_with_1_nh3_count'] += 1
            else:
                results['invalid_rp'].append(mol_result)
                results['statistics']['invalid_rp_count'] += 1
            
            if verbose:
                print(f"  DJ: {'✓' if result_dj.is_valid else '✗'} "
                      f"({len(result_dj.terminal_groups)} terminals, "
                      f"{len(nh3_groups_dj)} NH3, {len(result_dj.valid_paths)} paths)")
                print(f"  RP: {'✓' if result_rp.is_valid else '✗'} "
                      f"({len(result_rp.terminal_groups)} terminals, "
                      f"{len(nh3_groups_rp)} NH3)")
                if not result_dj.is_valid:
                    print(f"    DJ reason: {result_dj.reason}")
                if not result_rp.is_valid:
                    print(f"    RP reason: {result_rp.reason}")
        
        except Exception as e:
            results['failed_smiles'] += 1
            if verbose:
                print(f"  ✗ Failed to process: {e}")
    
    return results


def print_summary(results: Dict):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    stats = results['statistics']
    
    print(f"\nTotal molecules in CSV: {results['total_molecules']}")
    print(f"Successfully processed: {results['processed']}")
    print(f"Failed to process: {results['failed_smiles']}")
    
    print(f"\n{'='*80}")
    print("DJ SPACER RESULTS (Bifunctional - requires 2 terminal NH2/NH3)")
    print(f"{'='*80}")
    print(f"Valid DJ spacers: {stats['valid_dj_count']}")
    print(f"  - With exactly 2 NH3 groups: {stats['dj_with_2_nh3_count']}")
    print(f"Invalid DJ spacers: {stats['invalid_dj_count']}")
    
    print(f"\n{'='*80}")
    print("RP SPACER RESULTS (Monofunctional - requires 1+ terminal NH2/NH3)")
    print(f"{'='*80}")
    print(f"Valid RP spacers: {stats['valid_rp_count']}")
    print(f"  - With exactly 1 NH3 group: {stats['rp_with_1_nh3_count']}")
    print(f"Invalid RP spacers: {stats['invalid_rp_count']}")
    
    # Show some examples
    if results['dj_with_2_nh3']:
        print(f"\n{'='*80}")
        print("EXAMPLES: DJ Spacers with 2 NH3 groups")
        print(f"{'='*80}")
        for mol in results['dj_with_2_nh3'][:10]:  # Show first 10
            print(f"  {mol['Abbreviation']:10s} | {mol['Common_name']:40s} | {mol['SMILE']}")
        if len(results['dj_with_2_nh3']) > 10:
            print(f"  ... and {len(results['dj_with_2_nh3']) - 10} more")
    
    if results['rp_with_1_nh3']:
        print(f"\n{'='*80}")
        print("EXAMPLES: RP Spacers with 1 NH3 group")
        print(f"{'='*80}")
        for mol in results['rp_with_1_nh3'][:10]:  # Show first 10
            print(f"  {mol['Abbreviation']:10s} | {mol['Common_name']:40s} | {mol['SMILE']}")
        if len(results['rp_with_1_nh3']) > 10:
            print(f"  ... and {len(results['rp_with_1_nh3']) - 10} more")


def save_results(results: Dict, output_dir: str = "."):
    """Save results to JSON files.
    
    Parameters
    ----------
    results : Dict
        Results dictionary
    output_dir : str
        Output directory for results files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save full results
    full_results_file = output_path / "a_ion_molecules_full_results.json"
    with open(full_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Full results saved to: {full_results_file}")
    
    # Save DJ spacers with 2 NH3
    dj_file = output_path / "dj_spacers_2_nh3.json"
    with open(dj_file, 'w', encoding='utf-8') as f:
        json.dump(results['dj_with_2_nh3'], f, indent=2, ensure_ascii=False)
    print(f"✓ DJ spacers (2 NH3) saved to: {dj_file}")
    
    # Save RP spacers with 1 NH3
    rp_file = output_path / "rp_spacers_1_nh3.json"
    with open(rp_file, 'w', encoding='utf-8') as f:
        json.dump(results['rp_with_1_nh3'], f, indent=2, ensure_ascii=False)
    print(f"✓ RP spacers (1 NH3) saved to: {rp_file}")
    
    # Save summary statistics
    summary_file = output_path / "a_ion_molecules_summary.json"
    summary = {
        'statistics': results['statistics'],
        'total_molecules': results['total_molecules'],
        'processed': results['processed'],
        'failed_smiles': results['failed_smiles'],
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Summary statistics saved to: {summary_file}")


def main():
    """Main test function."""
    # Path to CSV file (go up one level from tests/ to project root)
    csv_path = Path(__file__).parent.parent / "q2D_Materials" / "data" / "tables" / "A-ion_data.csv"
    
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found at {csv_path}")
        return
    
    print("A-Ion Molecules Spacer Candidate Analysis")
    print("=" * 80)
    print(f"Reading molecules from: {csv_path}")
    
    # Read molecules
    molecules = read_molecules_from_csv(str(csv_path))
    print(f"Found {len(molecules)} molecules with SMILES strings")
    
    # Analyze molecules (set verbose=True for detailed output)
    results = analyze_molecules(molecules, verbose=False)
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results)
    
    print("\n" + "=" * 80)
    print("✓ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()


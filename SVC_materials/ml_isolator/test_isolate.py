"""
Test script demonstrating the molecule isolation functionality.
"""

from pathlib import Path
from SVC_materials.utils.isolate_molecule import process_molecules

def main():
    # --- USER-DEFINED PATHS ---
    INPUT_DIR = Path("SVC_materials/ml_isolator/data/original")
    OUTPUT_DIR = Path("SVC_materials/ml_isolator/data/molecules")
    TEMPLATE_DIR = Path("~/Documents/DION-JACOBSON/MOLECULES/")
    
    # Process all molecules
    stats = process_molecules(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        template_dir=TEMPLATE_DIR,
        debug=True
    )
    
    # Print summary
    print(f"\n--- Final Summary ---")
    print(f"Total input files: {stats['total_files']}")
    print(f"Successfully processed: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Output directory: {stats['output_dir']}")

if __name__ == "__main__":
    main() 
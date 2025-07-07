#!/usr/bin/env python3
"""
Test script for gaussian projection analysis functionality.

This script demonstrates how to use the q2D_analyzer class to perform
gaussian projection analysis on atomic structures.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the parent directory to the path to import SVC_materials
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SVC_materials.core.analyzer import q2D_analyzer, get_ionic_radius, ionic_radius_to_sigma


def create_test_data():
    """
    Create synthetic test data for demonstration purposes.
    This simulates a layered structure with different elements at different z-heights.
    """
    np.random.seed(42)  # For reproducible results
    
    # Create a synthetic layered structure
    data = []
    
    # Layer 1: Lead (Pb) atoms at z ~ 0
    pb_atoms = []
    for i in range(20):
        x = np.random.uniform(0, 10)
        y = np.random.uniform(0, 10)
        z = np.random.normal(0, 0.2)  # Centered at z=0 with small spread
        pb_atoms.append(['Pb', x, y, z])
    
    # Layer 2: Iodine (I) atoms at z ~ 3
    i_atoms = []
    for i in range(40):
        x = np.random.uniform(0, 10)
        y = np.random.uniform(0, 10)
        z = np.random.normal(3, 0.3)  # Centered at z=3 with small spread
        i_atoms.append(['I', x, y, z])
    
    # Layer 3: Nitrogen (N) atoms at z ~ 6
    n_atoms = []
    for i in range(15):
        x = np.random.uniform(0, 10)
        y = np.random.uniform(0, 10)
        z = np.random.normal(6, 0.1)  # Centered at z=6 with very small spread
        n_atoms.append(['N', x, y, z])
    
    # Layer 4: Carbon (C) atoms at z ~ 8
    c_atoms = []
    for i in range(30):
        x = np.random.uniform(0, 10)
        y = np.random.uniform(0, 10)
        z = np.random.normal(8, 0.4)  # Centered at z=8 with larger spread
        c_atoms.append(['C', x, y, z])
    
    # Combine all atoms
    all_atoms = pb_atoms + i_atoms + n_atoms + c_atoms
    
    # Create DataFrame
    df = pd.DataFrame(all_atoms, columns=['Element', 'X', 'Y', 'Z'])
    
    return df


def test_gaussian_projection():
    """
    Test the gaussian projection functionality with synthetic data.
    """
    print("Testing Gaussian Projection Analysis")
    print("=" * 50)
    
    # Create analyzer instance
    analyzer = q2D_analyzer()
    
    # Generate test data
    print("Creating synthetic test data...")
    test_data = create_test_data()
    
    # Manually set the data (simulating loaded VASP data)
    analyzer.data = test_data
    analyzer.file_path = "synthetic_test_data"
    
    # Show summary
    print("\nStructure Summary:")
    analyzer.summary()
    
    # Perform gaussian projection analysis
    print("\nPerforming Gaussian Projection Analysis...")
    analyzer.gaussian_projection()
    
    print("\nTest completed successfully!")
    print("Check the 'tests' directory for generated plots.")


def test_with_vasp_file(vasp_file_path):
    """
    Test the gaussian projection functionality with a real VASP file.
    
    Parameters:
    vasp_file_path (str): Path to the VASP file to analyze
    """
    print(f"Testing Gaussian Projection Analysis with VASP file: {vasp_file_path}")
    print("=" * 70)
    
    # Create analyzer instance and load file
    analyzer = q2D_analyzer(vasp_file_path)
    
    if analyzer.data is None:
        print("Failed to load VASP file. Using synthetic data instead.")
        test_gaussian_projection()
        return
    
    # Show summary
    print("\nStructure Summary:")
    analyzer.summary()
    
    # Perform gaussian projection analysis
    print("\nPerforming Gaussian Projection Analysis...")
    analyzer.gaussian_projection()
    
    print("\nAnalysis completed successfully!")
    print("Check the 'tests' directory for generated plots.")


def test_batch_processing():
    """
    Test batch processing of all VASP files in the structures directory.
    Now with gaussian base width = 2 × ionic radius for physically meaningful peaks.
    """
    print("Testing Batch Gaussian Projection Analysis with base width = 2 × ionic radius")
    print("=" * 75)
    
    # Show some examples of ionic radii used
    print("\nIonic Radii Examples (commonly found in structures):")
    example_elements = ['H', 'C', 'N', 'O', 'Pb', 'I', 'Cl', 'Br']
    
    for elem in example_elements:
        r_ionic = get_ionic_radius(elem)
        sigma = ionic_radius_to_sigma(r_ionic)  # This will now equal r_ionic / 3
        base_width = 6 * sigma  # Should equal 2 × r_ionic
        print(f"  {elem}: r_ionic = {r_ionic:.3f} Å, σ = {sigma:.3f} Å, base_width = {base_width:.3f} Å")
    
    print("\nPhysical Meaning:")
    print("- σ = ionic_radius / 3")
    print("- Base width (±3σ) = 2 × ionic_radius")
    print("- Gaussian effectively reaches zero at ±ionic_radius from center")
    print("- This makes the analysis more physically meaningful!")
    
    # Create analyzer instance
    analyzer = q2D_analyzer()
    
    # Run batch processing
    analyzer.batch_gaussian_projection()
    
    print("\nBatch processing test completed successfully!")
    print("Check the 'tests' directory for organized folders with beautiful plots.")
    print("Each plot now shows base width = 2 × ionic radius for focused gaussian peaks!")


if __name__ == "__main__":
    # Check if a VASP file path was provided as command line argument
    if len(sys.argv) > 1:
        if sys.argv[1] == "--batch":
            test_batch_processing()
        else:
            vasp_file_path = sys.argv[1]
            if os.path.exists(vasp_file_path):
                test_with_vasp_file(vasp_file_path)
            else:
                print(f"Error: File {vasp_file_path} not found.")
                print("Using synthetic test data instead.")
                test_gaussian_projection()
    else:
        print("Usage: python test_gaussian_projection.py [vasp_file_path|--batch]")
        print("Options:")
        print("  vasp_file_path  : Process a specific VASP file")
        print("  --batch         : Process all VASP files in tests/structures/")
        print("  (no arguments)  : Use synthetic test data")
        print()
        
        # Ask user what they want to do
        choice = input("Choose an option (1=synthetic, 2=batch, 3=quit): ")
        if choice == "1":
            test_gaussian_projection()
        elif choice == "2":
            test_batch_processing()
        else:
            print("Goodbye!") 
#!/usr/bin/env python3
"""
Example script demonstrating octahedral distortion analysis using the q2D_analyzer.

This script shows how to:
1. Load a structure and analyze octahedral distortions
2. Get systematic ordering of octahedra for comparison
3. Calculate comprehensive distortion parameters
4. Compare distortions between different structures
5. Export results for further analysis
"""

from SVC_materials.core.analyzer import q2D_analyzer
import pandas as pd
import numpy as np

def main():
    # Example 1: Basic distortion analysis
    print("=== Example 1: Basic Octahedral Distortion Analysis ===")
    
    # Load a structure (replace with your actual file path)
    try:
        analyzer = q2D_analyzer(
            file_path="structure_test_salt.vasp",  # Replace with your file
            b='Pb',  # Central atom
            x='Cl',  # Ligand atom
            cutoff_ref_ligand=3.5  # Distance cutoff in Angstroms
        )
        
        # Calculate distortions
        distortions = analyzer.calculate_octahedral_distortions()
        
        # Print summary
        analyzer.print_distortion_summary()
        
        # Get summary DataFrame
        summary_df = analyzer.get_distortion_summary()
        print(f"\nDetailed summary DataFrame:")
        print(summary_df.head())
        
        # Export to CSV
        analyzer.export_distortion_data("example_distortions.csv")
        
    except FileNotFoundError:
        print("Structure file not found. Please provide a valid VASP file path.")
        return
    
    # Example 2: Detailed analysis of specific octahedron
    print("\n=== Example 2: Detailed Analysis of Specific Octahedron ===")
    
    try:
        # Get detailed info for the first octahedron
        first_octa = analyzer.get_octahedron_by_index(0)
        
        print(f"Octahedron 0 details:")
        print(f"  Central atom: {first_octa['central_symbol']}")
        print(f"  Position: {first_octa['central_coord']}")
        print(f"  Bond distances: {first_octa['bond_distances']}")
        print(f"  Mean bond distance: {first_octa['mean_bond_distance']:.4f} Å")
        print(f"  Zeta parameter: {first_octa['zeta']:.4f}")
        print(f"  Delta parameter: {first_octa['delta']:.6f}")
        print(f"  Sigma parameter: {first_octa['sigma']:.4f}°")
        print(f"  Theta mean: {first_octa['theta_mean']:.4f}°")
        print(f"  Volume: {first_octa['octahedral_volume']:.2f} Å³")
        print(f"  Is octahedral: {first_octa['is_octahedral']}")
        
    except (IndexError, ValueError) as e:
        print(f"Could not analyze specific octahedron: {e}")
    
    # Example 3: Structure comparison (if you have two structures)
    print("\n=== Example 3: Structure Comparison ===")
    
    try:
        # Load a second structure for comparison
        analyzer2 = q2D_analyzer(
            file_path="structure_test_spacers.vasp",  # Replace with second file
            b='Pb',
            x='Cl',
            cutoff_ref_ligand=3.5
        )
        
        # Compare distortions
        comparison_df = analyzer.compare_distortions(analyzer2)
        print("Comparison between structures:")
        print(comparison_df.head())
        
        # Save comparison
        comparison_df.to_csv("structure_comparison.csv", index=False)
        print("Comparison saved to structure_comparison.csv")
        
    except FileNotFoundError:
        print("Second structure file not found. Skipping comparison.")
    
    # Example 4: Statistical analysis
    print("\n=== Example 4: Statistical Analysis ===")
    
    try:
        summary_df = analyzer.get_distortion_summary()
        
        print("Statistical analysis of distortion parameters:")
        print(f"Number of octahedra: {len(summary_df)}")
        
        # Analyze zeta parameter distribution
        zeta_values = summary_df['zeta']
        print(f"\nZeta parameter statistics:")
        print(f"  Mean: {zeta_values.mean():.4f}")
        print(f"  Std:  {zeta_values.std():.4f}")
        print(f"  Min:  {zeta_values.min():.4f}")
        print(f"  Max:  {zeta_values.max():.4f}")
        
        # Analyze delta parameter distribution
        delta_values = summary_df['delta']
        print(f"\nDelta parameter statistics:")
        print(f"  Mean: {delta_values.mean():.6f}")
        print(f"  Std:  {delta_values.std():.6f}")
        print(f"  Min:  {delta_values.min():.6f}")
        print(f"  Max:  {delta_values.max():.6f}")
        
        # Analyze sigma parameter distribution
        sigma_values = summary_df['sigma']
        print(f"\nSigma parameter statistics:")
        print(f"  Mean: {sigma_values.mean():.4f}°")
        print(f"  Std:  {sigma_values.std():.4f}°")
        print(f"  Min:  {sigma_values.min():.4f}°")
        print(f"  Max:  {sigma_values.max():.4f}°")
        
    except Exception as e:
        print(f"Statistical analysis failed: {e}")

def demonstrate_ordering():
    """
    Demonstrate the systematic ordering of octahedra.
    """
    print("\n=== Octahedra Ordering Demonstration ===")
    
    try:
        analyzer = q2D_analyzer(
            file_path="structure_test_salt.vasp",
            b='Pb',
            x='Cl',
            cutoff_ref_ligand=3.5
        )
        
        print("Octahedra ordering (Z, then X, then Y coordinates):")
        for i, octa in enumerate(analyzer.ordered_octahedra):
            coord = octa['central_coord']
            print(f"  Octahedron {i}: ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f})")
            print(f"    Global index: {octa['global_index']}")
            print(f"    Original index: {octa['original_index']}")
        
    except FileNotFoundError:
        print("Structure file not found for ordering demonstration.")

if __name__ == "__main__":
    main()
    demonstrate_ordering() 
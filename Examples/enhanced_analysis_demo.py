#!/usr/bin/env python3
"""
Enhanced Analysis Demo for q2D Materials.

This script demonstrates the enhanced analysis capabilities inspired by Pyrovskite,
including:
- Enhanced distortion analysis (delta, sigma, lambda)
- Tolerance factor calculations
- Comprehensive plotting capabilities
- Angular distribution analysis
- Cis/trans angle analysis
"""

import sys
import os
import numpy as np

# Add the parent directory to the path to import q2D_Materials
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q2D_Materials.core.analyzer import q2D_analyzer


def main():
    """
    Main function demonstrating enhanced analysis capabilities.
    """
    print("=" * 60)
    print("q2D Materials - Enhanced Analysis Demo")
    print("Inspired by Pyrovskite Package")
    print("=" * 60)
    
    # Test with one of the existing analysis files
    test_file = "tests/n3_analysis/n3_layers_only.vasp"
    
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found!")
        print("Available test files:")
        for root, dirs, files in os.walk("tests"):
            for file in files:
                if file.endswith('.vasp'):
                    print(f"  {os.path.join(root, file)}")
        return
    
    print(f"\nAnalyzing structure: {test_file}")
    print("-" * 40)
    
    try:
        # Initialize analyzer with larger cutoff to ensure we get 6 ligands
        analyzer = q2D_analyzer(
            file_path=test_file,
            b='Pb',
            x='Cl',
            cutoff_ref_ligand=4.0  # Increased cutoff to ensure we get all 6 ligands
        )
        
        # Perform comprehensive analysis
        print("Performing comprehensive analysis...")
        ontology = analyzer.get_ontology()
        
        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        analyzer.print_ontology_summary()
        
        # Demonstrate distortion analysis
        print("\n" + "=" * 60)
        print("DISTORTION ANALYSIS")
        print("=" * 60)
        
        distortion_analysis = ontology.get('distortion_analysis', {})
        if distortion_analysis and 'error' not in distortion_analysis:
            print("✓ Distortion analysis completed successfully!")
            
            # Delta distortion
            delta_analysis = distortion_analysis.get('delta_analysis', {})
            if 'overall_delta' in delta_analysis and delta_analysis['overall_delta'] is not None:
                print(f"  Delta distortion (bond length variation): {delta_analysis['overall_delta']:.6f}")
            
            # Sigma distortion
            sigma_analysis = distortion_analysis.get('sigma_analysis', {})
            if 'overall_sigma' in sigma_analysis and sigma_analysis['overall_sigma'] is not None:
                print(f"  Sigma distortion (angular deviation): {sigma_analysis['overall_sigma']:.2f}°")
            
            # Lambda distortion
            lambda_analysis = distortion_analysis.get('lambda_analysis', {})
            if 'lambda_3' in lambda_analysis and lambda_analysis['lambda_3'] is not None:
                print(f"  Lambda-3 distortion: {lambda_analysis['lambda_3']:.6f}")
            if 'lambda_2' in lambda_analysis and lambda_analysis['lambda_2'] is not None:
                print(f"  Lambda-2 distortion: {lambda_analysis['lambda_2']:.6f}")
            
            # Tolerance factors
            tolerance_factors = distortion_analysis.get('tolerance_factors', {})
            if 'goldschmidt_tolerance' in tolerance_factors and tolerance_factors['goldschmidt_tolerance'] is not None:
                print(f"  Goldschmidt tolerance factor: {tolerance_factors['goldschmidt_tolerance']:.4f}")
            if 'octahedral_tolerance' in tolerance_factors and tolerance_factors['octahedral_tolerance'] is not None:
                print(f"  Octahedral tolerance factor: {tolerance_factors['octahedral_tolerance']:.4f}")
        else:
            print("✗ Distortion analysis failed or not available")
            if 'error' in distortion_analysis:
                print(f"  Error: {distortion_analysis['error']}")
        
        # Demonstrate angular analysis
        print("\n" + "=" * 60)
        print("ANGULAR ANALYSIS")
        print("=" * 60)
        
        octahedra_data = ontology.get('octahedra', {})
        if octahedra_data:
            # Get angular distribution statistics
            angular_stats = analyzer.angular_analyzer.get_angular_distribution_statistics(octahedra_data)
            
            print("✓ Angular analysis completed successfully!")
            
            # Cis angles
            cis_stats = angular_stats.get('cis_angles', {})
            if cis_stats['count'] > 0:
                print(f"  Cis angles: {cis_stats['count']} angles")
                print(f"    Mean: {cis_stats['mean']:.2f}°")
                print(f"    Std: {cis_stats['std']:.2f}°")
                print(f"    Deviation from 90°: {cis_stats['deviation_from_90']:.2f}°")
            
            # Trans angles
            trans_stats = angular_stats.get('trans_angles', {})
            if trans_stats['count'] > 0:
                print(f"  Trans angles: {trans_stats['count']} angles")
                print(f"    Mean: {trans_stats['mean']:.2f}°")
                print(f"    Std: {trans_stats['std']:.2f}°")
                print(f"    Deviation from 180°: {trans_stats['deviation_from_180']:.2f}°")
            
            # Axial-central-axial angles
            aca_stats = angular_stats.get('axial_central_axial', {})
            if aca_stats['count'] > 0:
                print(f"  Axial-Central-Axial angles: {aca_stats['count']} angles")
                print(f"    Mean: {aca_stats['mean']:.2f}°")
                print(f"    Std: {aca_stats['std']:.2f}°")
                print(f"    Deviation from 180°: {aca_stats['deviation_from_180']:.2f}°")
        else:
            print("✗ No octahedra data available for angular analysis")
        
        # Demonstrate plotting capabilities
        print("\n" + "=" * 60)
        print("PLOTTING CAPABILITIES")
        print("=" * 60)
        
        try:
            # Create angle distribution plot
            print("Creating angle distribution plot...")
            angle_fig = analyzer.create_angle_distribution_plot(
                smearing=1.0, gridpoints=300, show=False, save=True, 
                filename="enhanced_angle_distributions.png"
            )
            if angle_fig:
                print("✓ Angle distribution plot created: enhanced_angle_distributions.png")
            
            # Create distance distribution plot
            print("Creating distance distribution plot...")
            distance_fig = analyzer.create_distance_distribution_plot(
                smearing=0.02, gridpoints=300, show=False, save=True,
                filename="enhanced_distance_distributions.png"
            )
            if distance_fig:
                print("✓ Distance distribution plot created: enhanced_distance_distributions.png")
            
            # Create octahedral distortion plot
            print("Creating octahedral distortion plot...")
            distortion_fig = analyzer.create_octahedral_distortion_plot(
                show=False, save=True, filename="enhanced_octahedral_distortion.png"
            )
            if distortion_fig:
                print("✓ Octahedral distortion plot created: enhanced_octahedral_distortion.png")
            
            # Create distortion comparison plot
            print("Creating distortion comparison plot...")
            distortion_fig = analyzer.create_distortion_plot(
                show=False, save=True, filename="distortion_comparison.png"
            )
            if distortion_fig:
                print("✓ Distortion comparison plot created: distortion_comparison.png")
            
            # Create comprehensive analysis plot
            print("Creating comprehensive analysis plot...")
            comprehensive_fig = analyzer.create_comprehensive_analysis_plot(
                show=False, save=True, filename="comprehensive_analysis.png"
            )
            if comprehensive_fig:
                print("✓ Comprehensive analysis plot created: comprehensive_analysis.png")
                
        except Exception as e:
            print(f"✗ Plotting failed: {str(e)}")
        
        # Demonstrate export capabilities
        print("\n" + "=" * 60)
        print("EXPORT CAPABILITIES")
        print("=" * 60)
        
        try:
            # Export ontology
            ontology_file = "enhanced_analysis_ontology.json"
            analyzer.export_ontology(ontology_file)
            print(f"✓ Ontology exported to: {ontology_file}")
            
            # Export layer analysis
            layer_file = "enhanced_layer_analysis.json"
            analyzer.export_layer_analysis(layer_file)
            print(f"✓ Layer analysis exported to: {layer_file}")
            
            # Export molecule ontology
            molecule_file = "enhanced_molecule_ontology.json"
            analyzer.export_molecule_ontology(molecule_file)
            print(f"✓ Molecule ontology exported to: {molecule_file}")
            
        except Exception as e:
            print(f"✗ Export failed: {str(e)}")
        
        print("\n" + "=" * 60)
        print("ENHANCED ANALYSIS DEMO COMPLETED")
        print("=" * 60)
        print("Generated files:")
        print("  - enhanced_angle_distributions.png")
        print("  - enhanced_distance_distributions.png")
        print("  - enhanced_octahedral_distortion.png")
        print("  - distortion_comparison.png")
        print("  - comprehensive_analysis.png")
        print("  - enhanced_analysis_ontology.json")
        print("  - enhanced_layer_analysis.json")
        print("  - enhanced_molecule_ontology.json")
        print("\nFeatures demonstrated:")
        print("  ✓ Distortion analysis (delta, sigma, lambda)")
        print("  ✓ Tolerance factor calculations")
        print("  ✓ Cis/trans angle analysis")
        print("  ✓ Angular distribution statistics")
        print("  ✓ Comprehensive plotting capabilities")
        print("  ✓ Gaussian smearing for smooth distributions")
        print("  ✓ Export capabilities")
        print("\nInspired by Pyrovskite package:")
        print("  - Stanton, R., & Trivedi, D. (2023). Pyrovskite: A software package")
        print("    for the high throughput construction, analysis, and featurization")
        print("    of two- and three-dimensional perovskite systems.")
        print("    Journal of Applied Physics, 133(24), 244701.")
        print("    https://doi.org/10.1063/5.0159407")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from SVC_materials.core.creator import q2D_creator
from SVC_materials.core.analyzer import q2D_analysis
import os
import numpy as np

# Example coordinates
P1 = "6.88270  -0.00111  10.43607"
P2 = "5.49067  -0.00113   9.90399"
P3 = "6.88270  -0.00111   4.49028"
Q1 = "0.93691   5.94468  10.43607"
Q2 = "-0.45512   5.94466   9.90399"
Q3 = "  0.93691   5.94468   4.49028"

# File paths
mol_file = 'Examples/benzyl_di_ammonium.xyz'  # Molecule file, it could be a mol2 or xyz file
perov_file = 'Examples/SuperCell_MAPbI3.vasp'  # This need to be a .vasp file with cartesian coordinates

def create_structures():
    """Create and display the initial structures"""
    # Create the q2D_creator instance
    mol = q2D_creator(
        B='Pb',  # B is the metal center
        X='I',   # X are the octahedra vertex elements
        molecule_xyz=mol_file,  # Molecule path
        perov_vasp=perov_file,  # Perovskite path
        P1=P1, P2=P2, P3=P3, Q1=Q1, Q2=Q2, Q3=Q3,  # Desired positions to put the organic molecule in the perovskite
        name='Perov1',  # Name of the system, is used save the files
        vac=10  # Amount of vacuum in Z direction for the SVC
    )

    # Display the structures
    #mol.show_bulk(slab=2, hn=0.3)  # Slab is the number of perovskite slabs, and hn is the penetration of the molecule in the slab
    #mol.show_svc()
    #mol.show_iso()

    # Save the structures
    #mol.write_iso()
    #mol.write_svc()
    #mol.write_bulk(slab=2, hn=0.3, order=['N', 'C', 'H', 'Pb', 'I'])  # Order specify the desired element order in the .vasp file

def analyze_structures():
    """Analyze and isolate components from the structures"""
    # Get the current working directory
    cwd = os.getcwd()
    
    # Create analyzer instance for the bulk structure
    structures_path = os.path.join(cwd, 'tests/structures')
    for bulk_file in os.listdir(structures_path):
        if bulk_file.endswith('.vasp'):  # Only process VASP files
            full_path = os.path.join(structures_path, bulk_file)
            if not os.path.exists(full_path):
                print(f"Error: Bulk structure file not found at {full_path}")
                continue
                
            print(f"\nAnalyzing structure: {bulk_file}")
            bulk_analyzer = q2D_analysis(
                B='Pb',  # Metal center
                X='Br',   # Halogen
                crystal=full_path  # Path to the bulk structure
            )

            # Show and save the isolated components
            try:
                #bulk_analyzer.show_original()  # Show the original bulk structure
                
                # Analyze perovskite structure
                print("\nAnalyzing perovskite structure parameters...")
                bulk_structure_data = bulk_analyzer.analyze_perovskite_structure()
                if bulk_structure_data:
                    print("\nBulk Structure Analysis Summary:")
                    print(f"Number of octahedra analyzed: {len(bulk_structure_data['per_octahedron'])}")
                    print(f"Average axial angle: {np.mean(bulk_structure_data['axial_angles']):.2f}°")
                    print(f"Average equatorial angle: {np.mean(bulk_structure_data['equatorial_angles']):.2f}°")
                    print(f"Average axial length: {np.mean(bulk_structure_data['axial_lengths']):.3f} Å")
                    print(f"Average equatorial length: {np.mean(bulk_structure_data['equatorial_lengths']):.3f} Å")
                    print(f"Average out-of-plane distortion: {np.mean(bulk_structure_data['out_of_plane_distortions']):.2f}°")
                
                # Isolate and visualize the spacer
                print("\nIsolating spacer...")
                #bulk_analyzer.show_spacer()
                #bulk_analyzer.save_spacer(name=os.path.join(cwd, 'bulk_spacer.vasp'))  # Save with custom name
                
                # Isolate and visualize the salt
                print("\nIsolating salt...")
                #bulk_analyzer.show_salt()
                #bulk_analyzer.save_salt(name=os.path.join(cwd, 'bulk_salt.vasp'))  # Save with custom name
            except Exception as e:
                print(f"Error analyzing bulk structure: {e}")

    # Create analyzer instance for the SVC structure
    svc_file = os.path.join(cwd, '/home/dotempo/Documents/REPOS/SVC-Materials/tests/structures/post_slab1.vasp')
    if not os.path.exists(svc_file):
        print(f"Error: SVC structure file not found at {svc_file}")
        return
        
    svc_analyzer = q2D_analysis(
        B='Pb',
        X='Br',
        crystal=svc_file  # Path to the SVC structure
    )
    
    # Analyze SVC structure
    print("\nAnalyzing SVC structure...")
    svc_analyzer.show_original()

def main():
    """Main function to run the examples"""
    print("Creating structures...")
    create_structures()
    
    print("\nAnalyzing structures...")
    analyze_structures()

if __name__ == "__main__":
    main() 
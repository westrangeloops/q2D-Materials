import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SVC_materials import q2D_creator

# Example coordinates
P1 = "6.88270  -0.00111  10.43607"
P2 = "5.49067  -0.00113   9.90399"
P3 = "6.88270  -0.00111   4.49028"
Q1 = "0.93691   5.94468  10.43607"
Q2 = "-0.45512   5.94466   9.90399"
Q3 = "  0.93691   5.94468   4.49028"

# File paths
mol_file = '../Examples/benzyl_di_ammonium.xyz'  # Updated path to be relative to tests directory
perov_file = '../Examples/SuperCell_MAPbI3.vasp'  # Updated path to be relative to tests directory

# Create the q2D_creator instance
mol = q2D_creator(
    B='Pb',
    X='I',  # B is the metal center and X are the octahedra vertex elements
    molecule_xyz=mol_file,  # Molecule path
    perov_vasp=perov_file,  # Perovskite path
    P1=P1, P2=P2, P3=P3, Q1=Q1, Q2=Q2, Q3=Q3,  # Desired positions to put the organic molecule in the perovskite
    name='Perov1',  # Name of the system, is used save the files 
    vac=10  # Amount of vacuum in Z direction for the SVC
)

# Test the visualization methods
mol.show_bulk(slab=1, hn=0.3)  # Slab is the number of perovskite slabs, and hn is the penetration of the molecule in the slab
mol.show_svc()
mol.show_iso()

# Test the file writing methods
mol.write_iso()
mol.write_svc()
mol.write_bulk(slab=1, hn=0.3, order=['N', 'C', 'H', 'Pb', 'Br'])  # Order specify the desired element order in the .vasp file 
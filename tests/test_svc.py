import sys
import os
import pytest
from pathlib import Path

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
mol_file = '../Examples/benzyl_di_ammonium.xyz'
perov_file = '../Examples/SuperCell_MAPbI3.vasp'

# Expected output files
EXPECTED_FILES = [
    'test_iso.vasp',
    'svc_test.vasp',
    'bulk_test.vasp'
]

@pytest.fixture
def q2d_instance():
    """Fixture to create and clean up q2D_creator instance"""
    mol = q2D_creator(
        B='Pb',
        X='I',
        molecule_xyz=mol_file,
        perov_vasp=perov_file,
        P1=P1, P2=P2, P3=P3, Q1=Q1, Q2=Q2, Q3=Q3,
        name='test',
        vac=10
    )
    yield mol
    # Cleanup after tests
    for file in EXPECTED_FILES:
        if os.path.exists(file):
            os.remove(file)

def test_file_creation(q2d_instance):
    """Test that all expected files are created"""
    # Create the files
    q2d_instance.write_iso()
    q2d_instance.write_svc()
    q2d_instance.write_bulk(slab=1, hn=0.3, order=['N', 'C', 'H', 'Pb', 'Br'])
    
    # Check if all files exist
    for file in EXPECTED_FILES:
        assert os.path.exists(file), f"File {file} was not created"
        
        # Basic content validation
        with open(file, 'r') as f:
            content = f.read()
            assert len(content) > 0, f"File {file} is empty"
            assert 'Cartesian' in content, f"File {file} does not contain Cartesian coordinates"

def test_file_content_structure(q2d_instance):
    """Test the structure of created files"""
    # Create the files
    q2d_instance.write_iso()
    q2d_instance.write_svc()
    q2d_instance.write_bulk(slab=1, hn=0.3, order=['N', 'C', 'H', 'Pb', 'Br'])
    
    for file in EXPECTED_FILES:
        with open(file, 'r') as f:
            lines = f.readlines()
            # Check basic VASP file structure
            assert len(lines) > 5, f"File {file} has insufficient content"
            assert 'Cartesian' in lines[7], f"File {file} does not specify Cartesian coordinates"
            
            # Check for element counts
            elements_line = lines[5].strip().split()
            counts_line = lines[6].strip().split()
            assert len(elements_line) == len(counts_line), f"File {file} has mismatched elements and counts"
            
            # Verify all counts are positive integers
            for count in counts_line:
                assert count.isdigit(), f"File {file} has invalid count: {count}"
                assert int(count) > 0, f"File {file} has zero or negative count: {count}"

def test_cleanup(q2d_instance):
    """Test that files are properly cleaned up"""
    # Create the files
    q2d_instance.write_iso()
    q2d_instance.write_svc()
    q2d_instance.write_bulk(slab=1, hn=0.3, order=['N', 'C', 'H', 'Pb', 'Br'])
    
    # Verify files exist
    for file in EXPECTED_FILES:
        assert os.path.exists(file), f"File {file} was not created"
    
    # Cleanup happens automatically after the test due to the fixture
    # This test just verifies the cleanup worked
    for file in EXPECTED_FILES:
        assert not os.path.exists(file), f"File {file} was not cleaned up" 
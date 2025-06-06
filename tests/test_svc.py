import sys
import os
import pytest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SVC_materials import q2D_creator
from SVC_materials.utils.file_handlers import vasp_load

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

def cleanup_files():
    """Helper function to clean up test files"""
    for file in EXPECTED_FILES:
        if os.path.exists(file):
            os.remove(file)

@pytest.fixture(autouse=True)
def setup_teardown():
    """Fixture to handle setup and teardown for all tests"""
    # Setup: Clean up any existing files
    cleanup_files()
    yield
    # Teardown: Clean up files after each test
    cleanup_files()

@pytest.fixture
def q2d_instance():
    """Fixture to create q2D_creator instance"""
    return q2D_creator(
        B='Pb',
        X='I',
        molecule_xyz=mol_file,
        perov_vasp=perov_file,
        P1=P1, P2=P2, P3=P3, Q1=Q1, Q2=Q2, Q3=Q3,
        name='test',
        vac=10
    )

def test_file_creation(q2d_instance):
    """Test that all expected files are created and can be read"""
    # Create the files
    q2d_instance.write_iso()
    q2d_instance.write_svc()
    q2d_instance.write_bulk(slab=1, hn=0.3)
    
    # Check if all files exist and can be read
    for file in EXPECTED_FILES:
        assert os.path.exists(file), f"File {file} was not created"
        
        # Try to read the VASP file
        try:
            df, box = vasp_load(file)
            assert not df.empty, f"File {file} has no atomic data"
            
            # Box is a list of two elements: [box_vectors, metadata]
            assert len(box) == 2, f"File {file} has invalid box structure"
            box_vectors = box[0]
            metadata = box[1]
            
            # Check box vectors
            assert len(box_vectors) == 3, f"File {file} has invalid box dimensions"
            assert all(len(row) == 3 for row in box_vectors), f"File {file} has invalid box vectors"
            
            # Check metadata
            assert len(metadata) == 3, f"File {file} has invalid metadata"
            assert 'Cartesian' in metadata[2], f"File {file} does not specify Cartesian coordinates"
            
        except Exception as e:
            pytest.fail(f"Failed to read {file}: {str(e)}")

def test_file_content_structure(q2d_instance):
    """Test the structure of created files"""
    # Create the files
    q2d_instance.write_iso()
    q2d_instance.write_svc()
    q2d_instance.write_bulk(slab=1, hn=0.3)
    
    for file in EXPECTED_FILES:
        # Read the VASP file
        df, box = vasp_load(file)
        
        # Check DataFrame structure
        assert 'Element' in df.columns, f"File {file} missing Element column"
        assert all(col in df.columns for col in ['X', 'Y', 'Z']), f"File {file} missing coordinate columns"
        
        # Check for non-empty data
        assert len(df) > 0, f"File {file} has no atomic data"
        
        # Check element counts
        element_counts = df['Element'].value_counts()
        assert all(count > 0 for count in element_counts), f"File {file} has zero counts for some elements"
        
        # Check box structure
        assert len(box) == 2, f"File {file} has invalid box structure"
        box_vectors = box[0]
        metadata = box[1]
        
        # Check box vectors
        assert len(box_vectors) == 3, f"File {file} has invalid box dimensions"
        assert all(len(row) == 3 for row in box_vectors), f"File {file} has invalid box vectors"
        
        # Check metadata
        assert len(metadata) == 3, f"File {file} has invalid metadata"
        assert 'Cartesian' in metadata[2], f"File {file} does not specify Cartesian coordinates"
        
        # Check coordinates are within reasonable bounds
        assert df['X'].between(-100, 100).all(), f"File {file} has X coordinates out of bounds"
        assert df['Y'].between(-100, 100).all(), f"File {file} has Y coordinates out of bounds"
        assert df['Z'].between(-100, 100).all(), f"File {file} has Z coordinates out of bounds"

def test_cleanup(q2d_instance):
    """Test that files are properly cleaned up"""
    # Create the files
    q2d_instance.write_iso()
    q2d_instance.write_svc()
    q2d_instance.write_bulk(slab=1, hn=0.3)
    
    # Verify files exist and can be read
    for file in EXPECTED_FILES:
        assert os.path.exists(file), f"File {file} was not created"
        df, box = vasp_load(file)
        assert not df.empty, f"File {file} has no atomic data"
        assert len(box) == 2, f"File {file} has invalid box structure"
    
    # Clean up files
    cleanup_files()
    
    # Verify files are cleaned up
    for file in EXPECTED_FILES:
        assert not os.path.exists(file), f"File {file} was not cleaned up" 
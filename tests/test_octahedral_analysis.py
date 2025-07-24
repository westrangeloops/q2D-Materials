#!/usr/bin/env python3
"""
Test script for octahedral distortion analysis using the q2D_analyzer class.
Analyzes three test structures (n1, n2, n3) and creates a comprehensive dataset.
"""

import sys
import os
import pandas as pd
import numpy as np
import json

# Add the parent directory to the path to import SVC_materials
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from SVC_materials.core.analyzer import q2D_analyzer

def test_octahedral_analysis():
    """
    Test octahedral distortion analysis on three structures and create a dataset.
    """
    print("=" * 80)
    print("OCTAHEDRAL DISTORTION ANALYSIS TEST")
    print("=" * 80)
    
    # Define the structure files
    structures = {
        'n1': 'tests/structures/structure_n1_test.vasp',
        'n2': 'tests/structures/structure_n2_test.vasp',
        'n3': 'tests/structures/structure_n3_test.vasp'
    }
    
    # Parameters for analysis
    central_atom = 'Pb'
    ligand_atom = 'Cl'
    cutoff = 3.5

    # Lets see what are the octahedra in the structure
    for key, value in structures.items():
        analyzer = q2D_analyzer(value, central_atom, ligand_atom, cutoff)
        analyzer.export_ontology_json(f"tests/{key}_ontology.json")
    
    


if __name__ == "__main__":
    # Run the test
    results_df = test_octahedral_analysis()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80) 
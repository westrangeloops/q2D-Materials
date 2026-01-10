"""
Diagnostic script for Glazer tilt detection.
Verifies local tilt angle calculations and correlation logic.
"""
import numpy as np
from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.analyzer.characterization import _get_reference_axes

def debug_glazer():
    print("="*60)
    print("GLAZER DETECTION DIAGNOSTIC")
    print("="*60)
    
    # Create a structure with known tilt: a-a-a- (10 degrees)
    print("\n1. Creating test structure (a-a-a-, 10 deg)...")
    creator = q2D_creator()
    structure = creator.create_structure(
        A_ions='Cs', B_ions='Pb', X_ions='I',
        structure_type='bulk', thickness=2, xy_expansion=(2, 2),
        glazer_pattern='a-a-a-', glazer_angles=[10, 10, 10]
    )
    
    print("2. Analyzing structure...")
    analyzer = q2D_analyzer(structure)
    analyzer.analyze()
    
    # Extract internals
    octahedra = analyzer.get_octahedra()
    atom_positions = analyzer.cell.get_positions()
    cell = np.array(analyzer.cell.get_cell())
    inv_cell = np.linalg.inv(cell)
    
    print(f"   Found {len(octahedra)} octahedra")
    
    # Get reference axes
    ref_axes = _get_reference_axes(atom_positions, cell, octahedra)
    print(f"\n3. Reference Axes:\n{ref_axes}")
    
    print("\n4. Local Tilt Analysis (First 4 Octahedra):")
    print("-" * 80)
    print(f"{'ID':<4} {'Alpha (X)':<20} {'Beta (Y)':<20} {'Gamma (Z)':<20}")
    print("-" * 80)
    
    def get_angle(y, x):
        return np.arctan2(y, x) * 180.0 / np.pi
    
    for i in range(min(4, len(octahedra))):
        oct_data = octahedra[i]
        b_idx = oct_data.get("central_atom_index")
        b_pos = atom_positions[b_idx]
        
        # Get neighbors and vectors
        x_indices = (oct_data.get("terminal_atoms", []) + 
                     oct_data.get("interlayer_atoms", []) + 
                     oct_data.get("intralayer_atoms", []))
        
        vectors = []
        for x_idx in x_indices:
            v = atom_positions[x_idx] - b_pos
            v_frac = v @ inv_cell.T
            v_frac -= np.round(v_frac)
            v = v_frac @ cell
            vectors.append(v)
            
        # Identify axes
        v_x, v_y, v_z = None, None, None
        v_x_proj, v_y_proj, v_z_proj = 0.0, 0.0, 0.0

        for v in vectors:
            projs = [np.dot(v, ref_axes[k]) for k in range(3)]
            abs_projs = np.abs(projs)
            max_idx = np.argmax(abs_projs)

            if max_idx == 0 and projs[0] > v_x_proj:
                v_x = v; v_x_proj = projs[0]
            elif max_idx == 1 and projs[1] > v_y_proj:
                v_y = v; v_y_proj = projs[1]
            elif max_idx == 2 and projs[2] > v_z_proj:
                v_z = v; v_z_proj = projs[2]
                
        # Transform
        f_x = v_x @ ref_axes.T if v_x is not None else None
        f_y = v_y @ ref_axes.T if v_y is not None else None
        f_z = v_z @ ref_axes.T if v_z is not None else None
        
        # Calculate angles (Current vs Clean)
        # Alpha
        a1 = get_angle(f_y[2], f_y[1]) if f_y is not None else 0 # Clean (f_y z-comp)
        a2 = get_angle(-f_z[1], f_z[2]) if f_z is not None else 0 # Dirty (f_z y-comp)
        
        # Beta
        b1 = get_angle(f_z[0], f_z[2]) if f_z is not None else 0 # Dirty (f_z x-comp)
        b2 = get_angle(-f_x[2], f_x[0]) if f_x is not None else 0 # Clean (f_x z-comp)
        
        # Gamma
        g1 = get_angle(f_x[1], f_x[0]) if f_x is not None else 0 # Clean (f_x y-comp)
        g2 = get_angle(-f_y[0], f_y[1]) if f_y is not None else 0 # Dirty (f_y x-comp)
        
        print(f"{i:<4} {a1:>6.1f} / {a2:>6.1f}     {b1:>6.1f} / {b2:>6.1f}     {g1:>6.1f} / {g2:>6.1f}")
        print(f"     (Clean/Dirty)        (Dirty/Clean)        (Clean/Dirty)")

if __name__ == "__main__":
    debug_glazer()
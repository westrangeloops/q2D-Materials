#!/usr/bin/env python3
"""
Test supercell expansion to see if atoms are being lost.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from q2D_Materials.core.creator import q2D_creator
from ase.io import write

print("Creating 1×1 DJ structure (n=2)")

q2d = q2D_creator()

structure = q2d.create_structure(
    structure_type="bulk",
    template="cubic",
    layer_sequence="DJ",
    thickness=2,
    A_ions="MA",
    B_ions="Pb",
    X_ions="I",
    spacer="[NH3+]CCC[NH3+]",
    glazer_angles=[0, 0, 0],
    glazer_pattern=["0", "0", "0"],
)

print(f"\n1×1 structure:")
print(f"  Atoms: {len(structure)}")
print(f"  Formula: {structure.get_chemical_formula()}")
print(f"  Cell: {structure.cell.cellpar()}")

# Count atom types
from collections import Counter
symbols = Counter(structure.get_chemical_symbols())
print(f"  Composition: {dict(symbols)}")

# Save 1x1
write(ROOT / "tests" / "dj_1x1_n2.vasp", structure, format="vasp")
print(f"  Saved: tests/dj_1x1_n2.vasp")

# Create 2x2
print("\nCreating 2×2 supercell...")
s2x2 = structure * (2, 2, 1)
print(f"  Atoms: {len(s2x2)} (expected: {len(structure) * 4})")
print(f"  Formula: {s2x2.get_chemical_formula()}")
symbols2 = Counter(s2x2.get_chemical_symbols())
print(f"  Composition: {dict(symbols2)}")
write(ROOT / "tests" / "dj_2x2_n2.vasp", s2x2, format="vasp")
print(f"  Saved: tests/dj_2x2_n2.vasp")

# Create 3x3
print("\nCreating 3×3 supercell...")
s3x3 = structure * (3, 3, 1)
print(f"  Atoms: {len(s3x3)} (expected: {len(structure) * 9})")
print(f"  Formula: {s3x3.get_chemical_formula()}")
symbols3 = Counter(s3x3.get_chemical_symbols())
print(f"  Composition: {dict(symbols3)}")
write(ROOT / "tests" / "dj_3x3_n2.vasp", s3x3, format="vasp")
print(f"  Saved: tests/dj_3x3_n2.vasp")

# Create 4x4
print("\nCreating 4×4 supercell...")
s4x4 = structure * (4, 4, 1)
print(f"  Atoms: {len(s4x4)} (expected: {len(structure) * 16})")
print(f"  Formula: {s4x4.get_chemical_formula()}")
symbols4 = Counter(s4x4.get_chemical_symbols())
print(f"  Composition: {dict(symbols4)}")
write(ROOT / "tests" / "dj_4x4_n2.vasp", s4x4, format="vasp")
print(f"  Saved: tests/dj_4x4_n2.vasp")

# Create 5x5
print("\nCreating 5×5 supercell...")
s5x5 = structure * (5, 5, 1)
print(f"  Atoms: {len(s5x5)} (expected: {len(structure) * 25})")
print(f"  Formula: {s5x5.get_chemical_formula()}")
symbols5 = Counter(s5x5.get_chemical_symbols())
print(f"  Composition: {dict(symbols5)}")
write(ROOT / "tests" / "dj_5x5_n2.vasp", s5x5, format="vasp")
print(f"  Saved: tests/dj_5x5_n2.vasp")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"1×1: {len(structure)} atoms")
print(f"2×2: {len(s2x2)} atoms (expected {len(structure)*4}, diff: {len(s2x2) - len(structure)*4})")
print(f"3×3: {len(s3x3)} atoms (expected {len(structure)*9}, diff: {len(s3x3) - len(structure)*9})")
print(f"4×4: {len(s4x4)} atoms (expected {len(structure)*16}, diff: {len(s4x4) - len(structure)*16})")
print(f"5×5: {len(s5x5)} atoms (expected {len(structure)*25}, diff: {len(s5x5) - len(structure)*25})")

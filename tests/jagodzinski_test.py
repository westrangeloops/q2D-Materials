#!/usr/bin/env python3
"""Test Jagodzinski stacking - outputs VASP files for visual inspection."""

import sys
sys.path.insert(0, '/home/dotempo/Documents/DJ/q2D-Materials')

from q2D_Materials.core.creator import q2D_creator
from ase.io import write

creator = q2D_creator()

print("Jagodzinski Stacking Tests")
print("=" * 40)

sequences = [
    ('ch', 'Simple cubic-hexagonal'),
    ('hh', 'Pure hexagonal'),
    ('chhcc', 'Mixed sequence'),
    ('chcchc', 'Alternating'),
    ('hhchhchhc', 'Extended mixed'),
]

for seq, desc in sequences:
    print(f"\n{seq}: {desc}")
    s = creator.create_perovskite('bulk',
        A_ions='Ca', B_ions='Ti', X_ions='O',
        jagodzinski_sequence=seq
    )
    fname = f'jag_{seq}.vasp'
    write(fname, s, format='vasp', sort=True)
    print(f"   {len(s)} atoms, layer sequence: {s.jagodzinski_sequence}")

print("\n" + "=" * 40)
print("Files written:")
for seq, _ in sequences:
    print(f"  - jag_{seq}.vasp")

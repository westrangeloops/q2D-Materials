#!/usr/bin/env python3
"""
Test script for the modular perovskite pipeline.
"""

import sys
sys.path.insert(0, '/home/dotempo/Documents/DJ/q2D-Materials')

from ase.io import write
from q2D_Materials.utils.pipeline import (
    build_cubic_primitive,
    build_monolayer,
    build_rp,
    build_dj,
    build_aci,
    create_structure,
    orient_structure,
    slice_layers,
    add_vacancies,
    substitute_sites,
    SITE_ROLE_KEY,
    SITE_A,
    SITE_B,
    SITE_X,
)
from q2D_Materials.core.creator import q2D_creator


def test_cubic_primitive():
    """Test building cubic primitive with site tags."""
    print("\n=== Test: Cubic Primitive ===")
    atoms = build_cubic_primitive('Cs', 'Pb', 'I', a0=6.3, supercell=(2, 2, 2))
    
    print(f"Number of atoms: {len(atoms)}")
    print(f"Cell: {atoms.cell.lengths()}")
    print(f"Symbols: {set(atoms.get_chemical_symbols())}")
    
    # Check site roles
    roles = atoms.arrays.get(SITE_ROLE_KEY)
    if roles is not None:
        unique_roles = set(roles)
        print(f"Site roles: {unique_roles}")
        for role in unique_roles:
            count = sum(1 for r in roles if r == role)
            print(f"  {role}: {count} atoms")
    
    write('tests/creator_outputs/pipeline_cubic.vasp', atoms, format='vasp')
    print("Saved to tests/creator_outputs/pipeline_cubic.vasp")


def test_monolayer():
    """Test building monolayer structure."""
    print("\n=== Test: Monolayer ===")
    atoms = build_monolayer('Cs', 'Pb', 'I', a0=6.3, n_layers=2, supercell_xy=(1, 1), vacuum=15.0)
    
    print(f"Number of atoms: {len(atoms)}")
    print(f"Cell: {atoms.cell.lengths()}")
    print(f"Symbols: {set(atoms.get_chemical_symbols())}")
    
    write('tests/creator_outputs/pipeline_monolayer.vasp', atoms, format='vasp')
    print("Saved to tests/creator_outputs/pipeline_monolayer.vasp")


def test_rp_phase():
    """Test building RP phase structure."""
    print("\n=== Test: RP Phase ===")
    atoms = build_rp('Cs', 'Pb', 'I', a0=6.3, n_layers=1, supercell_xy=(1, 1), interlayer_gap=3.0)
    
    print(f"Number of atoms: {len(atoms)}")
    print(f"Cell: {atoms.cell.lengths()}")
    print(f"Phase type: {atoms.info.get('phase_type')}")
    print(f"Spacer template: {atoms.info.get('spacer_template')}")
    
    write('tests/creator_outputs/pipeline_rp.vasp', atoms, format='vasp')
    print("Saved to tests/creator_outputs/pipeline_rp.vasp")


def test_dj_phase():
    """Test building DJ phase structure."""
    print("\n=== Test: DJ Phase ===")
    atoms = build_dj('Cs', 'Pb', 'I', a0=6.3, n_layers=1, supercell_xy=(1, 1), interlayer_gap=3.0)
    
    print(f"Number of atoms: {len(atoms)}")
    print(f"Cell: {atoms.cell.lengths()}")
    print(f"Phase type: {atoms.info.get('phase_type')}")
    
    write('tests/creator_outputs/pipeline_dj.vasp', atoms, format='vasp')
    print("Saved to tests/creator_outputs/pipeline_dj.vasp")


def test_aci_phase():
    """Test building ACI phase structure."""
    print("\n=== Test: ACI Phase ===")
    atoms = build_aci('Cs', 'Pb', 'I', a0=6.3, n_layers=1, supercell_xy=(1, 1), interlayer_gap=3.0)
    
    print(f"Number of atoms: {len(atoms)}")
    print(f"Cell: {atoms.cell.lengths()}")
    print(f"Phase type: {atoms.info.get('phase_type')}")
    
    write('tests/creator_outputs/pipeline_aci.vasp', atoms, format='vasp')
    print("Saved to tests/creator_outputs/pipeline_aci.vasp")


def test_orientation():
    """Test structure orientation."""
    print("\n=== Test: Orientation ===")
    cubic = build_cubic_primitive('Ca', 'Ti', 'O', a0=4.0, supercell=(2, 2, 2))
    
    print("Original cubic:")
    print(f"  Cell: {cubic.cell.lengths()}")
    
    oriented_100 = orient_structure(cubic, plane='100')
    print("Oriented (100):")
    print(f"  Cell: {oriented_100.cell.lengths()}")
    
    oriented_110 = orient_structure(cubic, plane='110')
    print("Oriented (110):")
    print(f"  Cell: {oriented_110.cell.lengths()}")


def test_vacancies():
    """Test vacancy creation."""
    print("\n=== Test: Vacancies ===")
    atoms = build_cubic_primitive('Cs', 'Pb', 'I', a0=6.3, supercell=(2, 2, 2))
    
    n_before = len(atoms)
    print(f"Before vacancies: {n_before} atoms")
    
    atoms_with_vac = add_vacancies(atoms, SITE_X, fraction=0.25, seed=42)
    n_after = len(atoms_with_vac)
    print(f"After 25% X-site vacancies: {n_after} atoms")
    print(f"Removed: {n_before - n_after} atoms")


def test_substitution():
    """Test site substitution."""
    print("\n=== Test: Substitution ===")
    atoms = build_cubic_primitive('Cs', 'Pb', 'I', a0=6.3, supercell=(2, 2, 2))
    
    symbols_before = atoms.get_chemical_symbols()
    n_pb = sum(1 for s in symbols_before if s == 'Pb')
    print(f"Before: {n_pb} Pb atoms")
    
    atoms_sub = substitute_sites(atoms, 'Pb', 'Sn', site_type=SITE_B, fraction=0.5, seed=42)
    
    symbols_after = atoms_sub.get_chemical_symbols()
    n_pb_after = sum(1 for s in symbols_after if s == 'Pb')
    n_sn = sum(1 for s in symbols_after if s == 'Sn')
    print(f"After 50% Pb->Sn substitution: {n_pb_after} Pb, {n_sn} Sn")


def test_unified_create_structure():
    """Test the unified create_structure function."""
    print("\n=== Test: Unified create_structure ===")
    
    # Bulk
    bulk = create_structure('bulk', 'Cs', 'Pb', 'I', supercell=(2, 2, 2))
    print(f"Bulk: {len(bulk)} atoms")
    
    # Monolayer
    mono = create_structure('monolayer', 'Cs', 'Pb', 'I', n_layers=1, vacuum=15.0)
    print(f"Monolayer: {len(mono)} atoms")
    
    # RP
    rp = create_structure('rp', 'Cs', 'Pb', 'I', n_layers=1, interlayer_gap=3.0)
    print(f"RP: {len(rp)} atoms")
    
    # DJ
    dj = create_structure('dj', 'Cs', 'Pb', 'I', n_layers=1, interlayer_gap=3.0)
    print(f"DJ: {len(dj)} atoms")


def test_creator_class():
    """Test the q2D_creator class with new pipeline."""
    print("\n=== Test: q2D_creator Class ===")
    
    creator = q2D_creator()
    
    # Bulk structure
    bulk = creator.create_perovskite(
        structure_type='bulk',
        A_ions='Cs',
        B_ions='Pb',
        X_ions='I',
        supercell_size=(2, 2, 2),
    )
    print(f"Bulk: {len(bulk)} atoms, type={bulk.structure_type}")
    
    # Monolayer
    mono = creator.create_perovskite(
        structure_type='monolayer',
        A_ions='Cs',
        B_ions='Pb',
        X_ions='I',
        supercell=[1, 1, 1],
        vacuum=15.0,
    )
    print(f"Monolayer: {len(mono)} atoms, type={mono.structure_type}")
    
    # RP phase
    rp = creator.create_perovskite(
        structure_type='RP',
        A_ions='Cs',
        B_ions='Pb',
        X_ions='I',
        supercell=[1, 1, 1],
        interlayer_gap=3.0,
    )
    print(f"RP: {len(rp)} atoms, type={rp.structure_type}")
    
    # DJ phase
    dj = creator.create_perovskite(
        structure_type='DJ',
        A_ions='Cs',
        B_ions='Pb',
        X_ions='I',
        supercell=[1, 1, 1],
        interlayer_gap=3.0,
    )
    print(f"DJ: {len(dj)} atoms, type={dj.structure_type}")


def test_glazer_tilting():
    """Test Glazer tilting through the creator."""
    print("\n=== Test: Glazer Tilting ===")
    
    creator = q2D_creator()
    
    tilted = creator.create_perovskite(
        structure_type='bulk',
        A_ions='Cs',
        B_ions='Pb',
        X_ions='I',
        supercell_size=(2, 2, 2),
        glazer_notation='a-a-a-',
    )
    print(f"Glazer tilted: {len(tilted)} atoms")
    print(f"Glazer pattern: {tilted.glazer_pattern if hasattr(tilted, 'glazer_pattern') else 'N/A'}")
    
    write('tests/creator_outputs/pipeline_glazer.vasp', tilted, format='vasp')
    print("Saved to tests/creator_outputs/pipeline_glazer.vasp")


def test_jagodzinski():
    """Test Jagodzinski stacking through the creator."""
    print("\n=== Test: Jagodzinski Stacking ===")
    
    creator = q2D_creator()
    
    jag = creator.create_perovskite(
        structure_type='bulk',
        A_ions='Ca',
        B_ions='Ti',
        X_ions='O',
        jagodzinski_sequence='ch',
    )
    print(f"Jagodzinski 'ch': {len(jag)} atoms")
    print(f"Layer sequence: {jag.jagodzinski_sequence if hasattr(jag, 'jagodzinski_sequence') else 'N/A'}")
    
    write('tests/creator_outputs/pipeline_jagodzinski.vasp', jag, format='vasp')
    print("Saved to tests/creator_outputs/pipeline_jagodzinski.vasp")


if __name__ == '__main__':
    print("=" * 60)
    print("Modular Perovskite Pipeline Tests")
    print("=" * 60)
    
    test_cubic_primitive()
    test_monolayer()
    test_rp_phase()
    test_dj_phase()
    test_aci_phase()
    test_orientation()
    test_vacancies()
    test_substitution()
    test_unified_create_structure()
    test_creator_class()
    test_glazer_tilting()
    test_jagodzinski()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


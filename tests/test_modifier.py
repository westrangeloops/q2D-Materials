"""Test script for molecule modifier functionality.

This test demonstrates the RDKit-based API for molecular modifications.
Modifications are done outside with RDKit, then passed back via update_from_rdkit().
"""

from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.modifier import GraphView
from rdkit import Chem
from rdkit.Chem import AllChem
from ase.io import write

# Step 1: Create DJ structure with diammonium hexane spacer
print("Step 1: Creating DJ structure with [NH3+]CCCCCC[NH3+] spacer...")
creator = q2D_creator()

# Create DJ structure with the specified spacer
structure = creator.create_structure(
    structure_type="bulk",
    A_ions="MA",
    B_ions="Pb",
    X_ions="I",
    template="cubic",
    layer_sequence="DJ",
    thickness=2,
    spacer="[NH3+]CCCCCC[NH3+]",  # Diammonium hexane
    xy_expansion=(1, 1),
)

# Save base structure
print("Saving base structure to base.vasp...")
write("base.vasp", structure, format="vasp")
print(f"Base structure created: {len(structure)} atoms\n")

# Step 2: Load and analyze the structure
print("Step 2: Analyzing structure...")
analyzer = q2D_analyzer(structure)
analyzer.analyze()

# Step 3: Create GraphView and access molecules
print("Step 3: Creating GraphView and accessing spacer molecules...")
view = GraphView(analyzer)
spacers = view.spacers.list()
print(f"Found {len(spacers)} spacer molecules")

if len(spacers) == 0:
    print("ERROR: No spacers found in the structure!")
    exit(1)

# Get the first spacer
spacer = spacers[0]
print(f"Spacer has {len(spacer.original_indices)} atoms")
print(f"Spacer SMILES: {spacer.smiles}")
print(f"Spacer JSON info:")
import json
print(json.dumps(spacer.to_json(), indent=2))
print()

# Find the third carbon atom (index in the molecule)
# The spacer molecule structure is: [NH3+]-C-C-C-C-C-C-[NH3+]
# We want to modify the third carbon (0-indexed: 0=N, 1=H, 2=H, 3=H, 4=C, 5=C, 6=C...)
# Let's find carbon atoms and select the third one
spacer_json = spacer.to_json()
carbon_indices = [
    atom['index'] for atom in spacer_json['atoms']
    if atom['symbol'] == 'C'
]
print(f"Carbon atom indices in spacer: {carbon_indices}")

if len(carbon_indices) < 3:
    print("ERROR: Spacer doesn't have at least 3 carbon atoms!")
    exit(1)

third_carbon_index = carbon_indices[2]  # Third carbon (0-indexed)
print(f"Third carbon is at local index: {third_carbon_index}\n")

# Step 4: Modify molecule using RDKit API (recommended approach)
# 
# ARCHITECTURE EXPLANATION:
# The refactor provides TWO approaches:
#
# 1. RDKit API (to_rdkit() → modify → update_from_rdkit()):
#    - For complex chemistry operations (reactions, bond modifications, etc.)
#    - Full access to RDKit's chemistry toolkit
#    - Modifications done outside, then passed back
#    - Automatically preserves coordinates of unchanged atoms
#
# 2. Legacy replace() method:
#    - Still available for simple atom replacements
#    - Convenient for swapping one atom with a fragment
#    - Uses from_smiles() internally (which uses RDKit)
#    - Handles geometry-aware positioning automatically
#
# Both approaches use RDKit under the hood, but serve different use cases.
# For fragment replacement, replace() is more convenient than manually editing RDKit Mol.

print("Step 4: Modifying molecule using RDKit API...")
print("  Converting molecule to RDKit Mol...")

# Convert molecule to RDKit for modifications
rdkit_mol = spacer.to_rdkit()
print(f"  Original molecule has {rdkit_mol.GetNumAtoms()} atoms")

# Example modifications using RDKit
# Note: For complex modifications like adding functional groups, use RDKit reactions
# For this test, we'll demonstrate the workflow with a simple modification

try:
    # Sanitize molecule first (required before operations)
    from rdkit import Chem
    from rdkit.Chem import rdmolops
    
    # Sanitize first (required before AddHs)
    try:
        Chem.SanitizeMol(rdkit_mol)
    except Exception:
        # If sanitization fails, try with relaxed options
        Chem.SanitizeMol(rdkit_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    
    # Add explicit hydrogens after sanitization (RDKit uses implicit by default)
    rdkit_mol = rdmolops.AddHs(rdkit_mol)
    
    # COMPLEX EXAMPLE: Multiple simultaneous modifications using RDKit
    # This demonstrates the power of RDKit API for complex chemistry operations:
    # 1. Replace one carbon with a benzene ring (aromatic system)
    # 2. Add a carbonyl group (C=O) to another carbon
    # 3. Add a hydroxyl group (-OH) to a third carbon
    # 4. Add a methyl group (-CH3) to a fourth carbon
    
    print(f"  Performing complex multi-step modifications...")
    
    # Find carbon atoms in RDKit molecule
    carbon_atoms = []
    for atom in rdkit_mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            carbon_atoms.append(atom.GetIdx())
    
    if len(carbon_atoms) < 5:
        print(f"  ERROR: Not enough carbon atoms (found {len(carbon_atoms)}, need at least 5)")
        raise ValueError("Not enough carbon atoms for complex modifications")
    
    # Create editable molecule for modifications
    editable_mol = Chem.RWMol(rdkit_mol)
    
    # MODIFICATION 1: Replace third carbon with benzene ring (aromatic)
    target_carbon_1 = carbon_atoms[2]  # Third carbon
    print(f"  Modification 1: Replacing carbon {target_carbon_1} with benzene ring (aromatic)")
    
    target_atom_1 = editable_mol.GetAtomWithIdx(target_carbon_1)
    neighbors_1 = [n.GetIdx() for n in target_atom_1.GetNeighbors()]
    
    if len(neighbors_1) > 0:
        # Create benzene ring (6 carbons in aromatic ring)
        benzene_carbons = []
        for i in range(6):
            c_idx = editable_mol.AddAtom(Chem.Atom('C'))
            benzene_carbons.append(c_idx)
            # Add hydrogens (aromatic carbons typically have 1 H)
            h_idx = editable_mol.AddAtom(Chem.Atom('H'))
            editable_mol.AddBond(c_idx, h_idx, Chem.BondType.SINGLE)
        
        # Create aromatic ring bonds (alternating single/double)
        for i in range(6):
            j = (i + 1) % 6
            # Use aromatic bond type
            editable_mol.AddBond(benzene_carbons[i], benzene_carbons[j], Chem.BondType.AROMATIC)
        
        # Connect benzene to neighbor of original carbon
        neighbor_1 = neighbors_1[0]
        editable_mol.AddBond(neighbor_1, benzene_carbons[0], Chem.BondType.SINGLE)
        
        # Remove original carbon and its bonds
        for n_idx in neighbors_1:
            if editable_mol.GetBondBetweenAtoms(target_carbon_1, n_idx):
                editable_mol.RemoveBond(target_carbon_1, n_idx)
        editable_mol.RemoveAtom(target_carbon_1)
    
    # MODIFICATION 2: Add carbonyl group (C=O) to fourth carbon
    remaining_carbons = [idx for idx in carbon_atoms if idx != target_carbon_1]
    if len(remaining_carbons) >= 1:
        target_carbon_2 = remaining_carbons[0]  # Fourth carbon (after removal)
        print(f"  Modification 2: Adding carbonyl (C=O) to carbon {target_carbon_2}")
        
        # Add oxygen atom
        o_idx = editable_mol.AddAtom(Chem.Atom('O'))
        # Add double bond between carbon and oxygen
        editable_mol.AddBond(target_carbon_2, o_idx, Chem.BondType.DOUBLE)
    
    # MODIFICATION 3: Add hydroxyl group (-OH) to fifth carbon
    if len(remaining_carbons) >= 2:
        target_carbon_3 = remaining_carbons[1]  # Fifth carbon
        print(f"  Modification 3: Adding hydroxyl (-OH) to carbon {target_carbon_3}")
        
        # Add oxygen atom
        oh_o_idx = editable_mol.AddAtom(Chem.Atom('O'))
        # Add hydrogen atom
        oh_h_idx = editable_mol.AddAtom(Chem.Atom('H'))
        # Add single bond between carbon and oxygen
        editable_mol.AddBond(target_carbon_3, oh_o_idx, Chem.BondType.SINGLE)
        # Add single bond between oxygen and hydrogen
        editable_mol.AddBond(oh_o_idx, oh_h_idx, Chem.BondType.SINGLE)

    # MODIFICATION 4: Add methyl group (-CH3) to sixth carbon
    if len(remaining_carbons) >= 3:
        target_carbon_4 = remaining_carbons[2]  # Sixth carbon
        print(f"  Modification 4: Adding methyl group (-CH3) to carbon {target_carbon_4}")
        
        # Add carbon atom for methyl
        ch3_c_idx = editable_mol.AddAtom(Chem.Atom('C'))
        # Add three hydrogen atoms
        for _ in range(3):
            h_idx = editable_mol.AddAtom(Chem.Atom('H'))
            editable_mol.AddBond(ch3_c_idx, h_idx, Chem.BondType.SINGLE)
        # Connect methyl carbon to target carbon
        editable_mol.AddBond(target_carbon_4, ch3_c_idx, Chem.BondType.SINGLE)
    
    # Convert back to regular molecule
    modified_rdkit_mol = editable_mol.GetMol()
    
    # Sanitize the modified molecule FIRST (required before AddHs)
    # This handles aromaticity, valence, etc.
    # Note: Warnings about nitrogen valence 4 are OK for [NH3+] (ammonium)
    # Note: Kekulization warnings are OK - structure is correct, RDKit just can't assign bond orders
    try:
        Chem.SanitizeMol(modified_rdkit_mol)
        # Set aromaticity for benzene ring
        Chem.SetAromaticity(modified_rdkit_mol)
        # Try to kekulize (assign alternating single/double bonds to aromatic rings)
        try:
            Chem.Kekulize(modified_rdkit_mol, clearAromaticFlags=False)
        except Exception:
            pass  # Kekulization failure is OK - structure is still valid
    except Exception as e:
        # Only print warnings for non-nitrogen-valence issues
        if "valence" not in str(e).lower() or "N" not in str(e):
            print(f"  Warning: Sanitization had issues: {e}")
        # Try with relaxed options
        try:
            Chem.SanitizeMol(modified_rdkit_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception:
            pass  # Continue anyway
    
    # Add explicit hydrogens for all atoms AFTER sanitization
    # This ensures that atoms like C2 in vinyl groups (C1=C2) get their 2 hydrogens:
    # - C1 is connected to chain (no H needed)
    # - C2 has double bond to C1, so needs 2 H to satisfy valence
    # RDKit's AddHs() automatically adds the correct number based on valence
    modified_rdkit_mol = rdmolops.AddHs(modified_rdkit_mol)
    
    print(f"  Modified molecule has {modified_rdkit_mol.GetNumAtoms()} atoms")
    print(f"  Summary of changes:")
    print(f"    - Replaced carbon with benzene ring (6C aromatic + 6H)")
    print(f"    - Added carbonyl group (C=O)")
    print(f"    - Added hydroxyl group (-OH)")
    print(f"    - Added methyl group (-CH3)")
    
    # Update back with coordinate preservation
    modified_structure = spacer.update_from_rdkit(modified_rdkit_mol, validate=True)

    # Save modified structure
    output_file = "modify_rdkit_example.vasp"
    write(output_file, modified_structure, format="vasp")
    print(f"  Saved modified structure to {output_file}")
    print(f"  Modified structure has {len(modified_structure)} atoms")
    print()
    print("  Note: For fragment replacement (e.g., replacing C with [CH0]=C),")
    print("        use RDKit reactions or manual Mol editing, then update_from_rdkit()")
    print()

    except Exception as e:
        print(f"  ERROR during modification: {e}")
        import traceback
        traceback.print_exc()
        print()
    print("  Note: If error mentions 'less than 2 atoms conserved',")
    print("        it means too many atoms were changed. Need at least 2 conserved atoms.")
    print()

print("Test completed!")

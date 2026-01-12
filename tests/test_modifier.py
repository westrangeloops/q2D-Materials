"""Test script for molecule modifier functionality."""

from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.modifier import GraphView, from_smiles
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

# Step 4: Create modifications with different fragments
modifications = [
    ("[CH0]=C", "vinyl"),      # Ethene/vinyl group
    ("CC(F)(F)F", "trifluoromethyl"), # Trifluoromethyl
    ("[CH0]=O", "carbonyl"),   # Carbonyl group
]

for smiles, name in modifications:
    print(f"Step 4.{modifications.index((smiles, name)) + 1}: Modifying with {name} ({smiles})...")

    try:
        # Convert SMILES to fragment graph
        fragment = from_smiles(smiles)
        print(f"  Fragment has {len(fragment.nodes())} atoms")

        # Replace the third carbon with the fragment
        # fragment_index=0 uses the first atom of the fragment as attachment point
        modified_structure = spacer.replace(
            atom_index=third_carbon_index,
            fragment_graph=fragment,
            fragment_index=0,
        )

        # Save modified structure
        output_file = f"modify_{smiles.replace('=', '_')}.vasp"
        write(output_file, modified_structure, format="vasp")
        print(f"  Saved modified structure to {output_file}")
        print(f"  Modified structure has {len(modified_structure)} atoms")
        print()

    except Exception as e:
        print(f"  ERROR during modification: {e}")
        import traceback
        traceback.print_exc()
        print()

print("Test completed!")

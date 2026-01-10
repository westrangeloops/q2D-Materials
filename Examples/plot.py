from pathlib import Path
import sys

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    plot_atoms = None

from ase.io import write
from q2D_Materials.core.creator import q2D_creator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

IMAGES = Path(__file__).resolve().parent / "images"
IMAGES.mkdir(exist_ok=True)

q2d = q2D_creator()

rotation_iso = "45x,45y,0z"
rotation_top = "0x,0y,0z"

# Bases
cubic_base = dict(structure_type="bulk", A_ions="MA", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="cubic")
jago_base = dict(structure_type="bulk", A_ions="MA", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="jagodinxky")
dj_base = dict(structure_type="bulk", A_ions="MA", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="cubic")

# Creator / Templates (cubic)
# Atomic A-site (Cs)
creator_L1_Cs = q2d.create_structure(**(cubic_base | {"A_ions": "Cs"}), layer_sequence=["L1"])
write(IMAGES / "creator-L1-Cs.png", creator_L1_Cs, rotation=rotation_iso, show_unit_cell=2)

creator_L1_L2_Cs = q2d.create_structure(**(cubic_base | {"A_ions": "Cs"}), layer_sequence=["L1", "L2"])
write(IMAGES / "creator-L1-L2-Cs.png", creator_L1_L2_Cs, rotation=rotation_iso, show_unit_cell=2)

creator_L1_L2_L1_Cs = q2d.create_structure(**(cubic_base | {"A_ions": "Cs"}), layer_sequence=["L1", "L2", "L1"])
write(IMAGES / "creator-L1-L2-L1-Cs.png", creator_L1_L2_L1_Cs, rotation=rotation_iso, show_unit_cell=2)

# Molecular A-site (MA)
creator_L1 = q2d.create_structure(**cubic_base, layer_sequence=["L1"])
write(IMAGES / "creator-L1.png", creator_L1, rotation=rotation_iso, show_unit_cell=2)

creator_L1_L2 = q2d.create_structure(**cubic_base, layer_sequence=["L1", "L2"])
write(IMAGES / "creator-L1-L2.png", creator_L1_L2, rotation=rotation_iso, show_unit_cell=2)

creator_L1_L2_L1 = q2d.create_structure(**cubic_base, layer_sequence=["L1", "L2", "L1"])
write(IMAGES / "creator-L1-L2-L1.png", creator_L1_L2_L1, rotation=rotation_iso, show_unit_cell=2)

# Jagodzinski storyboard
jago_a = q2d.create_structure(**jago_base, layer_sequence=["a"])
write(IMAGES / "jago-a.png", jago_a, rotation=rotation_iso, show_unit_cell=2)

jago_aC = q2d.create_structure(**jago_base, layer_sequence=["a", "C"])
write(IMAGES / "jago-aC.png", jago_aC, rotation=rotation_iso, show_unit_cell=2)

jago_aCc = q2d.create_structure(**jago_base, layer_sequence=["a", "C", "c"])
write(IMAGES / "jago-aCc.png", jago_aCc, rotation=rotation_iso, show_unit_cell=2)

# Glazer: untilted vs tilted (4x4, top-down)
glazer_untilted = q2d.create_structure(**(cubic_base | {"xy_expansion": (4, 4), "glazer_angles": [0, 0, 0], "glazer_pattern": ["0", "0", "0"]}), layer_sequence=["L1", "L2"])
write(IMAGES / "glazer-untitled-top.png", glazer_untilted, rotation=rotation_top, show_unit_cell=2)

glazer_tilted = q2d.create_structure(**(cubic_base | {"xy_expansion": (4, 4), "glazer_angles": [0, 0, 10], "glazer_pattern": ["0", "0", "+"]}), layer_sequence=["L1", "L2"])
write(IMAGES / "glazer-tilted-top.png", glazer_tilted, rotation=rotation_top, show_unit_cell=2)

# -----------------------------------------------------------------------------
# All 15 Glazer Space Groups (Howard & Stokes, 1998)
# -----------------------------------------------------------------------------
# First 10 using cubic template, last 5 using reduced template

# Mapping of all 15 conventional Glazer patterns to space groups
GLAZER_15_SYSTEMS = [
    # 1-10: Using cubic template
    ("a0a0a0", "Pm-3m", "cubic", "Cubic - No tilting"),
    ("a0a0c+", "P4/mbm", "cubic", "Tetragonal - In-phase tilt about c"),
    ("a0b+b+", "I4/mmm", "cubic", "Tetragonal - In-phase tilts about b and c"),
    ("a+a+a+", "Im-3", "cubic", "Cubic - In-phase tilts about all axes"),
    ("a+b+c+", "Immm", "cubic", "Orthorhombic - In-phase tilts, all different"),
    ("a0a0c-", "I4/mcm", "cubic", "Tetragonal - Out-of-phase tilt about c"),
    ("a0b-b-", "Imma", "cubic", "Orthorhombic - Out-of-phase tilts about b and c"),
    ("a-a-a-", "R-3c", "cubic", "Rhombohedral - Out-of-phase tilts about all axes"),
    ("a0b-c-", "C2/m", "cubic", "Monoclinic - Out-of-phase tilts about b and c"),
    ("a-b-b-", "C2/c", "cubic", "Monoclinic - Out-of-phase tilts, b=c"),
    # 11-15: Using reduced template
    ("a-b-c-", "P-1", "reduced", "Triclinic - Out-of-phase tilts, all different"),
    ("a0b+c-", "Cmcm", "reduced", "Orthorhombic - Mixed phases, b+ c-"),
    ("a+b-b-", "Pnma", "reduced", "Orthorhombic - Mixed phases, a+ b- c-"),
    ("a+b-c-", "P21/m", "reduced", "Monoclinic - Mixed phases, a+ b- c-"),
    ("a+a+c-", "P42/nmc", "reduced", "Tetragonal - Mixed phases, a+ a+ c-"),
]

print("\n" + "="*70)
print("Generating all 15 Glazer Space Group Systems")
print("="*70)

for idx, system_data in enumerate(GLAZER_15_SYSTEMS, 1):
    # Unpack system data (view_axis may be None initially)
    if len(system_data) == 5:
        notation, space_group, template, description, _ = system_data
    else:
        notation, space_group, template, description = system_data[:4]
    print(f"\n[{idx}/15] {space_group} ({notation}) - {description}")
    
    # Determine base configuration - use monolayer for all
    if template == "cubic":
        base_config = dict(
            structure_type="monolayer",
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            xy_expansion=(1, 1),
            template="cubic",
            vacuum=15.0
        )
    else:  # reduced
        base_config = dict(
            structure_type="monolayer",
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            xy_expansion=(2, 2),
            template="reduced",
            vacuum=15.0
        )
    
    # Determine supercell size based on pattern
    # Out-of-phase patterns need 2x2 supercell
    has_out_of_phase = "-" in notation
    if has_out_of_phase:
        xy_exp = (2, 2)
        thickness = 2
    else:
        xy_exp = (2, 2) if idx <= 10 else (2, 2)  # Use 2x2 for visibility
        thickness = 2
    
    base_config["xy_expansion"] = xy_exp
    base_config["thickness"] = thickness
    
    # Create structure with appropriate angles for visibility
    # Parse notation to determine angles for each axis
    def get_angles_from_notation(notat):
        """Extract tilt angles from Glazer notation.
        
        Ensures axes with the same magnitude letter get the same angle value.
        """
        if notat == "a0a0a0":
            return None
        
        # First pass: determine angle for each unique magnitude letter
        mag_to_angle_plus = {"a": 10.0, "b": 12.0, "c": 15.0}
        mag_to_angle_minus = {"a": 12.0, "b": 14.0, "c": 16.0}
        
        # Track which magnitude letters we've seen and their assigned angles
        mag_angles = {}
        
        angles = [0.0, 0.0, 0.0]
        # Parse notation: mag1 phase1 mag2 phase2 mag3 phase3
        for i in range(3):
            mag_idx = i * 2
            phase_idx = i * 2 + 1
            mag = notat[mag_idx]
            phase = notat[phase_idx]
            
            if phase == "0" or mag == "0":
                angles[i] = 0.0
            else:
                # Use cached angle if we've seen this magnitude before
                if mag in mag_angles:
                    angles[i] = mag_angles[mag]
                else:
                    # Assign new angle based on phase and magnitude
                    if phase == "+":
                        angles[i] = mag_to_angle_plus[mag]
                    elif phase == "-":
                        angles[i] = mag_to_angle_minus[mag]
                    # Cache this assignment
                    mag_angles[mag] = angles[i]
        
        return angles
    
    angles = get_angles_from_notation(notation)
    
    try:
        structure = q2d.create_structure(
            **base_config,
            glazer_pattern=notation,  # Use notation string directly
            glazer_angles=angles if angles else None,
        )
        
        # Generate three views for each space group: [001], [010], [100]
        view_configs = [
            ("001", "0x,0y,0z", "[0 0 1]"),    # Top view - looking down z-axis
            ("010", "90x,90y,0z", "[0 1 0]"),   # Front view - looking along y-axis
            ("100", "90x,0y,0z", "[1 0 0]"),   # Side view - looking along x-axis
        ]
        
        saved_files = []
        for view_code, rot, view_axis in view_configs:
            # Generate filename with view code
            filename = f"glazer_{idx:02d}_{space_group.lower().replace('/', '_')}_{view_code}.png"
            write(IMAGES / filename, structure, rotation=rot, show_unit_cell=2)
            saved_files.append((view_code, filename, view_axis))
            print(f"  ✓ Saved: {filename} (view: {view_axis})")
        
    except Exception as e:
        print(f"  ✗ Error creating {space_group}: {e}")

print("\n" + "="*70)
print("Completed generating all 15 Glazer Space Group Systems")
print("="*70 + "\n")


def save_multiview(atoms, filename, rotations, titles=None, radii=0.25, show_bonds=True):
    """Save multi-view visualization with optional bonds.
    
    Bonds are drawn for specific pairs with cutoffs:
    - N-H: 1.2 Å
    - Pb-I: 4.6 Å
    - C-N: 1.6 Å
    """
    if not MATPLOTLIB_AVAILABLE:
        print(f"Matplotlib not available; skipping {filename}")
        return
    
    # Define bond cutoffs for specific pairs
    BOND_CUTOFFS = {
        ('N', 'H'): 1.2,
        ('H', 'N'): 1.2,
        ('Pb', 'I'): 4.6,
        ('I', 'Pb'): 4.6,
        ('C', 'N'): 1.6,
        ('N', 'C'): 1.6,
    }
    
    # Detect bonds if requested
    bonds = []
    if show_bonds:
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        cell = atoms.get_cell()
        
        # Use PBC if cell is defined
        if cell is not None and np.any(cell):
            from q2D_Materials.builders.optimizers import _get_uc_neighbor_offsets
            uc_offsets = _get_uc_neighbor_offsets(cell)
        else:
            uc_offsets = np.array([[0., 0., 0.]])
        
        for i in range(len(atoms)):
            pos_i = positions[i]
            sym_i = symbols[i]
            pos_i_images = pos_i + uc_offsets
            
            for j in range(i + 1, len(atoms)):
                sym_j = symbols[j]
                pair = (sym_i, sym_j)
                
                # Check if this pair should have bonds
                if pair not in BOND_CUTOFFS:
                    continue
                
                cutoff = BOND_CUTOFFS[pair]
                pos_j = positions[j]
                
                # Check distance to all periodic images
                dists = np.linalg.norm(pos_i_images - pos_j, axis=1)
                min_dist = np.min(dists)
                
                if min_dist <= cutoff:
                    bonds.append((i, j))
    
    fig, axarr = plt.subplots(1, len(rotations), figsize=(4 * len(rotations), 4))
    if len(rotations) == 1:
        axarr = [axarr]
    
    for i, (ax, rot) in enumerate(zip(axarr, rotations)):
        # Plot atoms
        plot_atoms(atoms, ax, radii=radii, rotation=rot, show_unit_cell=0)
        
        # Draw bonds if available
        if show_bonds and bonds:
            positions = atoms.get_positions()
            symbols = atoms.get_chemical_symbols()
            
            # Parse rotation string to determine which axes to project
            # For axis-aligned views: [001] = 0x,0y,0z, [010] = 0x,90y,0z, [100] = 90x,0y,0z
            rot_parts = rot.split(',') if rot else []
            proj_axes = [0, 1]  # Default: project x and y
            
            # Determine projection axes based on rotation
            if '90x' in rot or '90X' in rot:
                # Side view [100]: project y and z
                proj_axes = [1, 2]
            elif '90y' in rot or '90Y' in rot:
                # Front view [010]: project x and z
                proj_axes = [0, 2]
            else:
                # Top view [001]: project x and y
                proj_axes = [0, 1]
            
            # Draw bonds
            for bond_i, bond_j in bonds:
                pos_i = positions[bond_i]
                pos_j = positions[bond_j]
                
                # Get bond color and style based on atom types
                sym_i, sym_j = symbols[bond_i], symbols[bond_j]
                if ('N' in (sym_i, sym_j) and 'H' in (sym_i, sym_j)):
                    color = 'gray'
                    linewidth = 1.0
                elif ('Pb' in (sym_i, sym_j) and 'I' in (sym_i, sym_j)):
                    color = 'orange'
                    linewidth = 1.5
                elif ('C' in (sym_i, sym_j) and 'N' in (sym_i, sym_j)):
                    color = 'blue'
                    linewidth = 1.0
                else:
                    color = 'gray'
                    linewidth = 0.5
                
                # Draw line using the appropriate projection axes
                ax.plot([pos_i[proj_axes[0]], pos_j[proj_axes[0]]], 
                       [pos_i[proj_axes[1]], pos_j[proj_axes[1]]], 
                       color=color, linewidth=linewidth, alpha=0.6, zorder=0)
        
        ax.set_axis_off()
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Monolayer: single and double thickness
mono1 = q2d.create_structure(structure_type="monolayer", A_ions="MA", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="cubic", vacuum=15.0, thickness=1, layer_sequence=["L1", "L2"])
write(IMAGES / "mono-1layer.png", mono1, rotation=rotation_iso, show_unit_cell=2)

mono2 = q2d.create_structure(structure_type="monolayer", A_ions="MA", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="cubic", vacuum=15.0, thickness=2, layer_sequence=["L1", "L2", "L1"])
write(IMAGES / "mono-2layer.png", mono2, rotation=rotation_iso, show_unit_cell=2)

# DJ (Dion–Jacobson) spacer example (bulk, spacer between S# layers)
dj = q2d.create_structure(
    **dj_base,
    layer_sequence="DJ",
    thickness=2,
    spacer="[NH3+]CCCC[NH3+]",
    glazer_angles=[0, 0, 3],
    glazer_pattern=["0", "0", "+"],
)
write(IMAGES / "dj_bulk.png", dj, rotation=rotation_iso, show_unit_cell=2)

# DJ atomic spacer example (Cs)
dj_atomic = q2d.create_structure(
    **(dj_base | {"xy_expansion": (2, 2)}),
    layer_sequence="DJ",
    thickness=3,
    spacer="Cs",
    glazer_angles=[3, 2, 6],
    glazer_pattern=["+", "-", "-"],
)
write(IMAGES / "dj_atomic.png", dj_atomic, rotation=rotation_iso, show_unit_cell=2)

# Ruddlesden–Popper storyboard (three representative shots)
rp_base = q2d.create_structure(
    structure_type="bulk",
    template="cubic",
    layer_sequence="RP",
    thickness=2,
    xy_expansion=(1, 1),
    A_ions="MA",
    B_ions="Pb",
    X_ions="I",
    spacer="[NH3+]CCCC[NH3+]",
    glazer_angles=[0, 0, 5],
    glazer_pattern=["0", "0", "+"],
)
write(IMAGES / "rp_base_side.png", rp_base, rotation=rotation_iso, show_unit_cell=2)

rp_glazer = q2d.create_structure(
    structure_type="bulk",
    template="cubic",
    layer_sequence="RP",
    thickness=3,
    xy_expansion=(2, 2),
    A_ions="MA",
    B_ions="Pb",
    X_ions="I",
    spacer=["[NH3+]CCCCCCC", "[NH3+]CCCC[NH3+]"],
    glazer_angles=[0, 2, 12],
    glazer_pattern=["0", "+", "+"],
    penetration=[-0.3, 0.3],
)
write(IMAGES / "rp_glazer_top.png", rp_glazer, rotation=rotation_top, show_unit_cell=2)

rp_atomic = q2d.create_structure(
    structure_type="bulk",
    template="cubic",
    layer_sequence="RP",
    thickness=4,
    xy_expansion=(2, 2),
    A_ions="MA",
    B_ions="Pb",
    X_ions="I",
    spacer="Cs",
    glazer_angles=[3, 2, 6],
    glazer_pattern=["+", "-", "-"],
    penetration=0.2,
)
write(IMAGES / "rp_atomic_side.png", rp_atomic, rotation=rotation_iso, show_unit_cell=2)


if MATPLOTLIB_AVAILABLE:
    # DJ (reduced template) Matplotlib views
    dj_reduced = q2d.create_structure(
        structure_type="bulk",
        template="reduced",
        layer_sequence="L2-M1-RP1-RP2-RP1-M1",
        thickness=2,
        A_ions="MA",
        B_ions="Pb",
        X_ions="I",
        spacer="[NH3+]CCCC[NH3+]",
        glazer_angles=[0, 0, 3],
        glazer_pattern=["0", "0", "+"],
    )
    save_multiview(
        dj_reduced,
        IMAGES / "dj_reduced_single.png",
        rotations=["90x,45y,0z"],
    )

    # RP (reduced template) Matplotlib storyboard
    rp_reduced = q2d.create_structure(
        structure_type="bulk",
        template="reduced",
        layer_sequence="L2-M1-RP1-RP2-RP1-M1",
        thickness=3,
        A_ions="MA",
        B_ions="Pb",
        X_ions="I",
        spacer=["C=CCC=CCC[NH3+]", "[NH3+]CCCC[NH3+]"],
        glazer_angles=[2, 4, 8],
        glazer_pattern=["0", "+", "+"],
        penetration=0.2,
    )
    save_multiview(
        rp_reduced,
        IMAGES / "rp_reduced_views.png",
        rotations=[
            "0x,0y,0z",
            "45x,0y,0z",
            "0x,0y,45z",
            "90x,45y,0z",
        ],
        titles=["Front", "Tilt", "Plan", "Iso"],
    )
else:
    print("Matplotlib not installed; skip DJ/RP multiview renders.")

# Twist storyboard: build two monolayers and twist them
twist_mono1 = q2d.create_structure(structure_type="monolayer", A_ions="MA", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="cubic", vacuum=12.0)
twist_mono2 = q2d.create_structure(structure_type="monolayer", A_ions="FA", B_ions="Sn", X_ions="Br", xy_expansion=(1, 1), template="cubic", vacuum=12.0)

write(IMAGES / "twist-mono1.png", twist_mono1, rotation=rotation_top, show_unit_cell=2)
write(IMAGES / "twist-mono2.png", twist_mono2, rotation=rotation_top, show_unit_cell=2)

twisted = q2d.twist(monolayers=[twist_mono1, twist_mono2], twist_angles=[(3, 1)], interlayer_distances=[8.0], vacuum=12.0)
write(IMAGES / "twist-bilayer.png", twisted, rotation=rotation_iso, show_unit_cell=2)

# Salts template: Visualize S# site vectors showing quadrant-based direction calculation
if MATPLOTLIB_AVAILABLE:
    from q2D_Materials.builders.optimizers import _find_directional_xy_pbc_vector
    import numpy as np
    
    # Create a salts structure to extract S# positions
    salts_structure = q2d.create_structure(
        template="salts",
        X_ions="I",
        spacer="[NH3+]CCC[NH3+]",
        lattice_multipliers=[4.0, 4.0],
        layer_sequence="L1-L2",
        structure_type="bulk",
        optimizer="Off",
        vacuum=0.0,
    )
    
    # Extract S# site positions from the structure
    # We need to get positions from the floor schema before population
    from q2D_Materials.builders.templates import build_floor_schema, flatten_floor_schema
    from q2D_Materials.pipeline.common import build_cell_positions
    
    # Build the floor schema to get S# positions
    schema = build_floor_schema(
        template_name="salts",
        BX_dist=3.0,
        layer_sequence="L1-L2",
        lattice_multipliers=[4.0, 4.0],
    )
    
    positions_by_site, site_labels = flatten_floor_schema(schema)
    cell = schema.cell
    
    # Get S# positions from L1 and L2
    s1_l1_positions = []
    s1_l2_positions = []
    s2_l1_positions = []
    s2_l2_positions = []
    s3_l1_positions = []
    s3_l2_positions = []
    s4_l1_positions = []
    s4_l2_positions = []
    
    # Extract positions from floors (L1 is floor 1, L2 is floor 2)
    for floor_key, entries in schema.floors.items():
        floor_name = schema.floor_names[floor_key]
        for site, x, y, z in entries:
            if site == "S1":
                if floor_name == "L1":
                    s1_l1_positions.append(np.array([x, y, z]))
                elif floor_name == "L2":
                    s1_l2_positions.append(np.array([x, y, z]))
            elif site == "S2":
                if floor_name == "L1":
                    s2_l1_positions.append(np.array([x, y, z]))
                elif floor_name == "L2":
                    s2_l2_positions.append(np.array([x, y, z]))
            elif site == "S3":
                if floor_name == "L1":
                    s3_l1_positions.append(np.array([x, y, z]))
                elif floor_name == "L2":
                    s3_l2_positions.append(np.array([x, y, z]))
            elif site == "S4":
                if floor_name == "L1":
                    s4_l1_positions.append(np.array([x, y, z]))
                elif floor_name == "L2":
                    s4_l2_positions.append(np.array([x, y, z]))
    
    # Collect all positions to calculate bounding box for zooming
    all_positions = []
    all_vectors = []
    
    # Color map for different S# labels
    colors = {'S1': 'red', 'S2': 'blue', 'S3': 'green', 'S4': 'orange'}
    
    # Plot S# positions and vectors
    s_pairs = [
        ('S1', s1_l1_positions, s1_l2_positions),
        ('S2', s2_l1_positions, s2_l2_positions),
        ('S3', s3_l1_positions, s3_l2_positions),
        ('S4', s4_l1_positions, s4_l2_positions),
    ]
    
    # First pass: collect all positions and vectors to determine bounds
    for label, l1_positions, l2_positions in s_pairs:
        for pos in l1_positions:
            all_positions.append(pos[:2])
        for i, l2_pos in enumerate(l2_positions):
            if i < len(l1_positions):
                l1_pos = l1_positions[i]
                vector = _find_directional_xy_pbc_vector(l1_pos, l2_pos, cell)
                all_positions.append(l2_pos[:2])
                all_positions.append((l1_pos[:2] + vector[:2]))
                all_vectors.append((l1_pos, vector))
    
    # Calculate bounding box with padding
    all_positions = np.array(all_positions)
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    
    # Add 15% padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_pad = x_range * 0.15
    y_pad = y_range * 0.15
    
    # Create visualization with better space usage
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot unit cell boundaries (only show relevant cells)
    a_vec = cell[0]
    b_vec = cell[1]
    cell_corners = np.array([
        [0, 0],
        [a_vec[0], a_vec[1]],
        [a_vec[0] + b_vec[0], a_vec[1] + b_vec[1]],
        [b_vec[0], b_vec[1]],
        [0, 0]
    ])
    
    # Draw unit cells only in the visible region
    for i in range(-1, 3):
        for j in range(-1, 3):
            offset = i * a_vec[:2] + j * b_vec[:2]
            shifted_corners = cell_corners + offset
            ax.plot(shifted_corners[:, 0], shifted_corners[:, 1], 'k--', alpha=0.2, linewidth=0.5)
    
    # Second pass: plot positions and vectors
    for label, l1_positions, l2_positions in s_pairs:
        color = colors[label]
        
        # Plot L1 positions
        for idx, pos in enumerate(l1_positions):
            ax.scatter(pos[0], pos[1], c=color, s=150, marker='o', 
                      edgecolors='black', linewidths=2, label=f'{label} L1' if idx == 0 else '', zorder=3)
            ax.text(pos[0] + 0.4, pos[1] + 0.4, f'{label}\nL1', 
                   fontsize=10, ha='left', va='bottom', color=color, weight='bold', zorder=4)
        
        # Plot L2 positions and draw vectors
        for i, l2_pos in enumerate(l2_positions):
            if i < len(l1_positions):
                l1_pos = l1_positions[i]
                
                # Calculate vector using the same function used in populate
                vector = _find_directional_xy_pbc_vector(l1_pos, l2_pos, cell)
                
                # Plot L2 position
                ax.scatter(l2_pos[0], l2_pos[1], c=color, s=150, marker='s', 
                          edgecolors='black', linewidths=2, label=f'{label} L2' if i == 0 else '', zorder=3)
                ax.text(l2_pos[0] + 0.4, l2_pos[1] + 0.4, f'{label}\nL2', 
                       fontsize=10, ha='left', va='bottom', color=color, weight='bold', zorder=4)
                
                # Draw arrow showing vector direction - same color as label, thicker
                arrow_length = np.linalg.norm(vector[:2])
                head_width = max(0.8, arrow_length * 0.08)
                head_length = max(0.6, arrow_length * 0.1)
                
                ax.arrow(l1_pos[0], l1_pos[1], vector[0], vector[1],
                        head_width=head_width, head_length=head_length, 
                        fc=color, ec=color,  # Same color for fill and edge
                        linewidth=4, alpha=0.9, length_includes_head=True, zorder=2)
    
    # Set zoomed limits
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    
    ax.set_xlabel('X (Å)', fontsize=14, weight='bold')
    ax.set_ylabel('Y (Å)', fontsize=14, weight='bold')
    ax.set_title('S# Site Vectors: Quadrant-Based Direction Calculation\n(salts template, L1→L2 connections)', 
                fontsize=16, weight='bold', pad=20)
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add explanation text
    explanation = (
        "Vectors are calculated using fractional coordinates:\n"
        "• L1 S# sites are in the center cell (0,0)\n"
        "• L2 S# sites with frac X>1 or Y>1 are in adjacent cells\n"
        "• Arrows show the periodic boundary-aware connection vectors"
    )
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.tight_layout()
    fig.savefig(IMAGES / "salts_site_vectors.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ Created salts_site_vectors.png showing S# site direction vectors")
else:
    print("Matplotlib not available; skipping S# vectors visualization")

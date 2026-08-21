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

# Helper function to save atoms as SVG using matplotlib
def save_atoms_svg(filename, atoms, rotation="0x,0y,0z", show_unit_cell=0):
    """Save atoms structure as SVG using matplotlib."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("Matplotlib not available")
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_atoms(atoms, ax, rotation=rotation, show_unit_cell=show_unit_cell)
    ax.set_axis_off()
    plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig)

# Import publication-quality plotting style
try:
    from plot_style import (
        apply_publication_style,
        create_figure_with_style,
        style_axes,
        save_figure,
        PUBLICATION_DPI,
        HALOGEN_COLORS,
        ORGANIC_COLOR,
    )
    STYLE_AVAILABLE = True
except ImportError:
    STYLE_AVAILABLE = False
    print("⚠ Warning: plot_style module not available, using default matplotlib style")


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
save_atoms_svg(IMAGES / "creator-L1-Cs.svg", creator_L1_Cs, rotation=rotation_iso, show_unit_cell=2)

creator_L1_L2_Cs = q2d.create_structure(**(cubic_base | {"A_ions": "Cs"}), layer_sequence=["L1", "L2"])
save_atoms_svg(IMAGES / "creator-L1-L2-Cs.svg", creator_L1_L2_Cs, rotation=rotation_iso, show_unit_cell=2)

creator_L1_L2_L1_Cs = q2d.create_structure(**(cubic_base | {"A_ions": "Cs"}), layer_sequence=["L1", "L2", "L1"])
save_atoms_svg(IMAGES / "creator-L1-L2-L1-Cs.svg", creator_L1_L2_L1_Cs, rotation=rotation_iso, show_unit_cell=2)

# Molecular A-site (MA)
creator_L1 = q2d.create_structure(**cubic_base, layer_sequence=["L1"])
save_atoms_svg(IMAGES / "creator-L1.svg", creator_L1, rotation=rotation_iso, show_unit_cell=2)

creator_L1_L2 = q2d.create_structure(**cubic_base, layer_sequence=["L1", "L2"])
save_atoms_svg(IMAGES / "creator-L1-L2.svg", creator_L1_L2, rotation=rotation_iso, show_unit_cell=2)

creator_L1_L2_L1 = q2d.create_structure(**cubic_base, layer_sequence=["L1", "L2", "L1"])
save_atoms_svg(IMAGES / "creator-L1-L2-L1.svg", creator_L1_L2_L1, rotation=rotation_iso, show_unit_cell=2)

# Jagodzinski storyboard
jago_a = q2d.create_structure(**jago_base, layer_sequence=["a"])
save_atoms_svg(IMAGES / "jago-a.svg", jago_a, rotation=rotation_iso, show_unit_cell=2)

jago_aC = q2d.create_structure(**jago_base, layer_sequence=["a", "C"])
save_atoms_svg(IMAGES / "jago-aC.svg", jago_aC, rotation=rotation_iso, show_unit_cell=2)

jago_aCc = q2d.create_structure(**jago_base, layer_sequence=["a", "C", "c"])
save_atoms_svg(IMAGES / "jago-aCc.svg", jago_aCc, rotation=rotation_iso, show_unit_cell=2)

# Glazer: untilted vs tilted (4x4, top-down)
glazer_untilted = q2d.create_structure(**(cubic_base | {"xy_expansion": (4, 4), "glazer_angles": [0, 0, 0], "glazer_pattern": ["0", "0", "0"]}), layer_sequence=["L1", "L2"])
save_atoms_svg(IMAGES / "glazer-untitled-top.svg", glazer_untilted, rotation=rotation_top, show_unit_cell=2)

glazer_tilted = q2d.create_structure(**(cubic_base | {"xy_expansion": (4, 4), "glazer_angles": [0, 0, 10], "glazer_pattern": ["0", "0", "+"]}), layer_sequence=["L1", "L2"])
save_atoms_svg(IMAGES / "glazer-tilted-top.svg", glazer_tilted, rotation=rotation_top, show_unit_cell=2)

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
            filename = f"glazer_{idx:02d}_{space_group.lower().replace('/', '_')}_{view_code}.svg"
            save_atoms_svg(IMAGES / filename, structure, rotation=rot, show_unit_cell=2)
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
save_atoms_svg(IMAGES / "mono-1layer.svg", mono1, rotation=rotation_iso, show_unit_cell=2)

mono2 = q2d.create_structure(structure_type="monolayer", A_ions="MA", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="cubic", vacuum=15.0, thickness=2, layer_sequence=["L1", "L2", "L1"])
save_atoms_svg(IMAGES / "mono-2layer.svg", mono2, rotation=rotation_iso, show_unit_cell=2)

# DJ (Dion–Jacobson) spacer example (bulk, spacer between S# layers)
dj = q2d.create_structure(
    **dj_base,
    layer_sequence="DJ",
    thickness=2,
    spacer="[NH3+]CCCC[NH3+]",
    glazer_angles=[0, 0, 3],
    glazer_pattern=["0", "0", "+"],
)
save_atoms_svg(IMAGES / "dj_bulk.svg", dj, rotation=rotation_iso, show_unit_cell=2)

# DJ atomic spacer example (Cs)
dj_atomic = q2d.create_structure(
    **(dj_base | {"xy_expansion": (2, 2)}),
    layer_sequence="DJ",
    thickness=3,
    spacer="Cs",
    glazer_angles=[3, 2, 6],
    glazer_pattern=["+", "-", "-"],
)
save_atoms_svg(IMAGES / "dj_atomic.svg", dj_atomic, rotation=rotation_iso, show_unit_cell=2)

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
save_atoms_svg(IMAGES / "rp_base_side.svg", rp_base, rotation=rotation_iso, show_unit_cell=2)

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
save_atoms_svg(IMAGES / "rp_glazer_top.svg", rp_glazer, rotation=rotation_top, show_unit_cell=2)

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
save_atoms_svg(IMAGES / "rp_atomic_side.svg", rp_atomic, rotation=rotation_iso, show_unit_cell=2)


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
        IMAGES / "dj_reduced_single.svg",
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
        IMAGES / "rp_reduced_views.svg",
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

save_atoms_svg(IMAGES / "twist-mono1.svg", twist_mono1, rotation=rotation_top, show_unit_cell=2)
save_atoms_svg(IMAGES / "twist-mono2.svg", twist_mono2, rotation=rotation_top, show_unit_cell=2)

twisted = q2d.twist(monolayers=[twist_mono1, twist_mono2], twist_angles=[(3, 1)], interlayer_distances=[8.0], vacuum=12.0)
save_atoms_svg(IMAGES / "twist-bilayer.svg", twisted, rotation=rotation_iso, show_unit_cell=2)

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
    fig.savefig(IMAGES / "salts_site_vectors.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ Created salts_site_vectors.svg showing S# site direction vectors")
else:
    print("Matplotlib not available; skipping S# vectors visualization")

# -----------------------------------------------------------------------------
# Molecule Modifier Examples
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("Generating Molecule Modifier Examples")
print("="*70)

from q2D_Materials.modifier import GraphView, from_smiles

# Create base structure with spacer
modifier_base = q2d.create_structure(
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

# Analyze structure
from q2D_Materials.analyzer import q2D_analyzer
analyzer = q2D_analyzer(modifier_base)
analyzer.analyze()

# Create GraphView
view = GraphView(analyzer)
spacers = view.spacers.list()

if len(spacers) > 0:
    spacer = spacers[0]
    
    # Get molecule info
    spacer_info = spacer.to_json()
    carbon_indices = [
        atom['index'] for atom in spacer_info['atoms']
        if atom['symbol'] == 'C'
    ]
    
    if len(carbon_indices) >= 3:
        target_carbon = carbon_indices[2]
        
        # Save base structure
        save_atoms_svg(IMAGES / "modifier_base.svg", modifier_base, rotation=rotation_iso, show_unit_cell=2)
        print("✓ Created modifier_base.svg")
        
        # Fragment modifications
        fragments = [
            ("[CH0]=C", "vinyl"),
            ("CC(F)(F)F", "trifluoromethyl"),
            ("[CH0]=O", "carbonyl"),
        ]
        
        for smiles, name in fragments:
            try:
                fragment = from_smiles(smiles)
                modified = spacer.replace(
                    atom_index=target_carbon,
                    fragment_graph=fragment,
                    fragment_index=0,
                )
                filename = f"modifier_{name}.svg"
                save_atoms_svg(IMAGES / filename, modified, rotation=rotation_iso, show_unit_cell=2)
                print(f"✓ Created {filename}")
            except Exception as e:
                print(f"✗ Error creating modifier_{name}.svg: {e}")
    else:
        print("✗ Not enough carbon atoms for modification example")
else:
    print("✗ No spacers found for modification example")

print("="*70 + "\n")

# -----------------------------------------------------------------------------
# RDF Plot Generation (Example 14)
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("Generating RDF Plot Examples")
print("="*70)

if MATPLOTLIB_AVAILABLE:
    from q2D_Materials.analyzer import q2D_analyzer
    
    # Create structure for RDF analysis
    rdf_structure = q2d.create_structure(
        structure_type="monolayer",
        A_ions="MA", B_ions="Pb", X_ions="I",
        xy_expansion=(2, 2), template="cubic",
        thickness=2, vacuum=15.0,
    )
    
    # Analyze structure
    rdf_analyzer = q2D_analyzer(rdf_structure)
    rdf_analyzer.analyze()
    
    # Apply publication style if available
    if STYLE_AVAILABLE:
        apply_publication_style()
    
    # 1. Single element pair RDF - Get data and plot
    rdf_data = rdf_analyzer.get_partial_rdf(element_pairs=[["Pb", "I"]], max_dist=12.0)
    
    if STYLE_AVAILABLE:
        fig = create_figure_with_style('standard')
        ax = fig.add_subplot(111)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'PbI_x' in rdf_data and 'PbI_rdf' in rdf_data:
        ax.plot(rdf_data['PbI_x'], rdf_data['PbI_rdf'], label='Pb-I', linewidth=2, color='#ff5555')
    ax.set_xlabel('Distance (Å)')
    ax.set_ylabel('g(r)')
    ax.set_title('Partial Radial Distribution Function')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 12.0)
    
    if STYLE_AVAILABLE:
        style_axes(ax)
        save_figure(fig, IMAGES / "rdf_single_pair.svg", format='svg')
    else:
        plt.tight_layout()
        plt.savefig(IMAGES / "rdf_single_pair.svg", format='svg', dpi=300, bbox_inches='tight')
        plt.close()
    print("✓ Created rdf_single_pair.svg")
    
    # 2. Multiple element pairs RDF - Get data and plot
    rdf_data = rdf_analyzer.get_partial_rdf(
        element_pairs=[["Pb", "I"], ["Pb", "Pb"], ["I", "I"]],
        max_dist=12.0
    )
    
    if STYLE_AVAILABLE:
        fig = create_figure_with_style('standard')
        ax = fig.add_subplot(111)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 3))
    pairs = [("PbI", "Pb-I"), ("PbPb", "Pb-Pb"), ("II", "I-I")]
    for idx, (pair_key, pair_label) in enumerate(pairs):
        x_key = f"{pair_key}_x"
        rdf_key = f"{pair_key}_rdf"
        if x_key in rdf_data and rdf_key in rdf_data:
            ax.plot(rdf_data[x_key], rdf_data[rdf_key], label=pair_label,
                   color=colors[idx], linewidth=2)
    ax.set_xlabel('Distance (Å)')
    ax.set_ylabel('g(r)')
    ax.set_title('Partial Radial Distribution Functions')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 12.0)
    
    if STYLE_AVAILABLE:
        style_axes(ax)
        save_figure(fig, IMAGES / "rdf_multiple_pairs.svg", format='svg')
    else:
        plt.tight_layout()
        plt.savefig(IMAGES / "rdf_multiple_pairs.svg", format='svg', dpi=300, bbox_inches='tight')
        plt.close()
    print("✓ Created rdf_multiple_pairs.svg")
    
    print("="*70 + "\n")
else:
    print("Matplotlib not available; skipping RDF plots\n")

# -----------------------------------------------------------------------------
# Angle and Distance Plot Generation (Example 15)
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("Generating Angle and Distance Plot Examples")
print("="*70)

if MATPLOTLIB_AVAILABLE:
    # Apply publication style if available
    if STYLE_AVAILABLE:
        apply_publication_style()
    
    # Use same structure
    # 1. B-X-B angles - Get data and plot
    bxb_data = rdf_analyzer.get_bxb_angles()
    angles = bxb_data['bxb_angles']
    if angles is not None and len(angles) > 0:
        # Adjust bins if needed
        unique_values = len(np.unique(angles))
        adjusted_bins = min(60, max(unique_values, 1)) if unique_values < 60 else 60
        
        if STYLE_AVAILABLE:
            fig = create_figure_with_style('standard')
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(angles, bins=adjusted_bins, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(bxb_data['bxb_mean'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {bxb_data['bxb_mean']:.2f}°')
        ax.set_xlabel('B-X-B Angle (degrees)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'B-X-B Bond Angles (σ = {bxb_data['bxb_std']:.2f}°)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        if STYLE_AVAILABLE:
            style_axes(ax)
            save_figure(fig, IMAGES / "bxb_angles_histogram.svg", format='svg')
        else:
            plt.tight_layout()
            plt.savefig(IMAGES / "bxb_angles_histogram.svg", format='svg', dpi=300, bbox_inches='tight')
            plt.close()
        print("✓ Created bxb_angles_histogram.svg")
    
    # 2. X-B-X angles - Get data and plot
    from q2D_Materials.analyzer.characterization.distortions import _compute_octahedral_distortions
    distortions = _compute_octahedral_distortions(rdf_analyzer)
    xbx_angles = distortions['bond_angles']
    if len(xbx_angles) > 0:
        # Adjust bins if needed
        unique_values = len(np.unique(xbx_angles))
        adjusted_bins = min(80, max(unique_values, 1)) if unique_values < 80 else 80
        
        if STYLE_AVAILABLE:
            fig = create_figure_with_style('standard')
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(xbx_angles, bins=adjusted_bins, alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(distortions['mean_angle'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {distortions['mean_angle']:.2f}°')
        ax.axvline(90, color='green', linestyle=':', linewidth=1.5,
                   label='Ideal 90°', alpha=0.7)
        ax.axvline(180, color='blue', linestyle=':', linewidth=1.5,
                   label='Ideal 180°', alpha=0.7)
        ax.set_xlabel('X-B-X Angle (degrees)')
        ax.set_ylabel('Frequency')
        ax.set_title('X-B-X Bond Angles')
        ax.legend()
        ax.grid(alpha=0.3)
        
        if STYLE_AVAILABLE:
            style_axes(ax)
            save_figure(fig, IMAGES / "xbx_angles_histogram.svg", format='svg')
        else:
            plt.tight_layout()
            plt.savefig(IMAGES / "xbx_angles_histogram.svg", format='svg', dpi=300, bbox_inches='tight')
            plt.close()
        print("✓ Created xbx_angles_histogram.svg")
    
    # 3. B-X bond lengths - Get data and plot
    bond_lengths = distortions['bond_lengths']
    if len(bond_lengths) > 0:
        # Adjust bins if data range is too small
        data_range = np.max(bond_lengths) - np.min(bond_lengths)
        unique_values = len(np.unique(bond_lengths))
        # Use fewer bins if range is very small or we have few unique values
        if data_range < 1e-6 or unique_values < 40:
            adjusted_bins = min(40, max(unique_values, 1))
        else:
            adjusted_bins = 40
        
        if STYLE_AVAILABLE:
            fig = create_figure_with_style('standard')
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(bond_lengths, bins=adjusted_bins, alpha=0.7, color='mediumseagreen', edgecolor='black')
        ax.axvline(distortions['mean_bond_length'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {distortions['mean_bond_length']:.3f} Å')
        ax.set_xlabel('B-X Bond Length (Å)')
        ax.set_ylabel('Frequency')
        std_dist = np.std(bond_lengths)
        ax.set_title(f'B-X Bond Lengths (σ = {std_dist:.3f} Å)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        if STYLE_AVAILABLE:
            style_axes(ax)
            save_figure(fig, IMAGES / "bx_bond_lengths_histogram.svg", format='svg')
        else:
            plt.tight_layout()
            plt.savefig(IMAGES / "bx_bond_lengths_histogram.svg", format='svg', dpi=300, bbox_inches='tight')
            plt.close()
        print("✓ Created bx_bond_lengths_histogram.svg")
    
    # 4. All pairwise distances - Get data and plot
    from ase.neighborlist import neighbor_list
    i, j, d = neighbor_list('ijd', rdf_analyzer.cell, cutoff=8.0)
    
    if STYLE_AVAILABLE:
        fig = create_figure_with_style('standard')
        ax = fig.add_subplot(111)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(d, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Distance (Å)')
    ax.set_ylabel('Frequency')
    ax.set_title('All Pairwise Distances')
    ax.grid(alpha=0.3)
    
    if STYLE_AVAILABLE:
        style_axes(ax)
        save_figure(fig, IMAGES / "all_distances_histogram.svg", format='svg')
    else:
        plt.tight_layout()
        plt.savefig(IMAGES / "all_distances_histogram.svg", format='svg', dpi=300, bbox_inches='tight')
        plt.close()
    print("✓ Created all_distances_histogram.svg")
    
    print("="*70 + "\n")
else:
    print("Matplotlib not available; skipping angle/distance plots\n")

# -----------------------------------------------------------------------------
# Graph Visualization Generation (Example 16)
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("Generating Graph Visualization Examples")
print("="*70)

if MATPLOTLIB_AVAILABLE:
    import networkx as nx
    
    # Create structure for graph analysis
    graph_structure = q2d.create_structure(
        structure_type="monolayer",
        A_ions="MA", B_ions="Pb", X_ions="I",
        xy_expansion=(2, 2), template="cubic",
        thickness=2, vacuum=15.0,
    )
    
    graph_analyzer = q2D_analyzer(graph_structure)
    graph_analyzer.analyze()
    graph = graph_analyzer.get_graph()
    
    # 1. Graph structure overview
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get node positions using spring layout
    # Create subgraph for layout (use octahedra and layers for cleaner visualization)
    viz_nodes = [n for n in graph.nodes() 
                 if graph.nodes[n].get('node_type') in ['octahedron', 'layer']]
    viz_subgraph = graph.subgraph(viz_nodes)
    
    if len(viz_subgraph) > 0:
        pos = nx.spring_layout(viz_subgraph, k=2, iterations=50)
        
        # Color nodes by type
        node_colors = []
        for node in viz_subgraph.nodes():
            node_type = graph.nodes[node].get('node_type', 'unknown')
            if node_type == 'octahedron':
                node_colors.append('#ff5555')  # Red for octahedra
            elif node_type == 'layer':
                node_colors.append('#50fa7b')  # Green for layers
            else:
                node_colors.append('#6272a4')  # Gray for others
        
        # Draw nodes
        nx.draw_networkx_nodes(viz_subgraph, pos, node_color=node_colors, 
                              node_size=500, alpha=0.8, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(viz_subgraph, pos, alpha=0.3, width=1.5, ax=ax)
        
        # Draw labels for layers only (to avoid clutter)
        layer_labels = {n: n.replace('layer_', 'L') for n in viz_subgraph.nodes() 
                       if graph.nodes[n].get('node_type') == 'layer'}
        nx.draw_networkx_labels(viz_subgraph, pos, labels=layer_labels, 
                               font_size=10, font_weight='bold', ax=ax)
    
    ax.set_title('Graph Structure Overview\n(Octahedra and Layers)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff5555', label='Octahedra'),
        Patch(facecolor='#50fa7b', label='Layers'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(IMAGES / "graph_structure_overview.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created graph_structure_overview.svg")
    
    # 2. Octahedra network visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create octahedra-only subgraph
    octahedra_nodes = [n for n in graph.nodes() 
                      if graph.nodes[n].get('node_type') == 'octahedron']
    octahedra_subgraph = graph.subgraph(octahedra_nodes)
    
    if len(octahedra_subgraph) > 0:
        # Use spring layout
        pos = nx.spring_layout(octahedra_subgraph, k=1.5, iterations=50)
        
        # Color by B-site element
        octahedra = graph_analyzer.get_octahedra()
        oct_to_element = {oct['id']: oct.get('central_atom_symbol', 'Unknown') 
                          for oct in octahedra}
        
        node_colors = []
        for node in octahedra_subgraph.nodes():
            element = oct_to_element.get(node, 'Unknown')
            if element == 'Pb':
                node_colors.append('#ff5555')
            elif element == 'Sn':
                node_colors.append('#50fa7b')
            else:
                node_colors.append('#6272a4')
        
        nx.draw_networkx_nodes(octahedra_subgraph, pos, node_color=node_colors,
                              node_size=600, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(octahedra_subgraph, pos, alpha=0.4, width=2, ax=ax)
        
        # Add labels (show first few)
        labels = {list(octahedra_subgraph.nodes())[i]: f"O{i}" 
                 for i in range(min(5, len(octahedra_subgraph)))}
        nx.draw_networkx_labels(octahedra_subgraph, pos, labels, 
                               font_size=9, font_weight='bold', ax=ax)
    
    ax.set_title('Octahedra Connectivity Network\n(Shared X-site connections)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(IMAGES / "graph_octahedra_network.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created graph_octahedra_network.svg")
    
    # 3. Layer structure visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    layers = graph_analyzer.get_layers()
    layer_nodes = [f'layer_{lid}' for lid in layers.keys()]
    layer_subgraph = graph.subgraph(layer_nodes + octahedra_nodes[:min(20, len(octahedra_nodes))])
    
    if len(layer_subgraph) > 0:
        # Hierarchical layout: layers on left, octahedra on right
        pos = {}
        layer_y_positions = {}
        
        # Position layers vertically
        for i, layer_id in enumerate(sorted(layers.keys())):
            layer_node = f'layer_{layer_id}'
            y_pos = len(layers) - i
            pos[layer_node] = (0, y_pos)
            layer_y_positions[layer_id] = y_pos
        
        # Position octahedra based on their layer
        for oct in octahedra:
            oct_node = oct['id']
            if oct_node in layer_subgraph:
                layer_id = None
                for lid, layer_info in layers.items():
                    if oct_node in layer_info.get('octahedra', []):
                        layer_id = lid
                        break
                
                if layer_id is not None and layer_id in layer_y_positions:
                    y_base = layer_y_positions[layer_id]
                    # Spread octahedra horizontally
                    oct_in_layer = [o for o in octahedra 
                                  if any(o['id'] in layers[lid].get('octahedra', []) 
                                        for lid in layers.keys() if lid == layer_id)]
                    idx = next((i for i, o in enumerate(oct_in_layer) if o['id'] == oct_node), 0)
                    pos[oct_node] = (1 + idx * 0.3, y_base + (idx % 3 - 1) * 0.2)
                else:
                    pos[oct_node] = (1, 0)
        
        # Draw
        layer_colors = ['#50fa7b' if 'layer' in n else '#ff5555' for n in layer_subgraph.nodes()]
        node_sizes = [800 if 'layer' in str(n) else 400 for n in layer_subgraph.nodes()]
        nx.draw_networkx_nodes(layer_subgraph, pos, node_color=layer_colors,
                              node_size=node_sizes, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(layer_subgraph, pos, alpha=0.3, width=1.5, ax=ax)
        
        # Labels
        layer_labels = {n: n.replace('layer_', 'L') for n in layer_subgraph.nodes() 
                       if 'layer' in n}
        nx.draw_networkx_labels(layer_subgraph, pos, labels=layer_labels,
                               font_size=12, font_weight='bold', ax=ax)
    
    ax.set_title('Layer-Based Graph Organization', fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(IMAGES / "graph_layer_structure.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created graph_layer_structure.svg")
    
    print("="*70 + "\n")
else:
    print("Matplotlib not available; skipping graph visualizations\n")

# -----------------------------------------------------------------------------
# Molecular Visualization Generation (Example 20 - Validator)
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("Generating Molecular Validator Examples")
print("="*70)

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import AllChem
    
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not available; skipping molecular visualizations\n")

if RDKIT_AVAILABLE and MATPLOTLIB_AVAILABLE:
    # 1. Valid DJ spacer (NCCCCN)
    dj_smiles = "NCCCCN"
    dj_mol = Chem.MolFromSmiles(dj_smiles)
    if dj_mol:
        AllChem.Compute2DCoords(dj_mol)
        svg = Draw.MolToSVG(dj_mol, width=600, height=400)
        with open(IMAGES / "validator_dj_spacer.svg", 'w') as f:
            f.write(svg)
        print("✓ Created validator_dj_spacer.svg")
    
    # 2. Valid RP spacer (C[NH3+])
    rp_smiles = "C[NH3+]"
    rp_mol = Chem.MolFromSmiles(rp_smiles)
    if rp_mol:
        AllChem.Compute2DCoords(rp_mol)
        svg = Draw.MolToSVG(rp_mol, width=400, height=400)
        with open(IMAGES / "validator_rp_spacer.svg", 'w') as f:
            f.write(svg)
        print("✓ Created validator_rp_spacer.svg")
    
    # 3. Pattern matching visualization
    # Show molecule with pattern matches highlighted
    pattern_smiles = "NCCCCN"
    pattern = "[NH3+]C"
    mol = Chem.MolFromSmiles(pattern_smiles)
    if mol:
        # Create pattern molecule
        pattern_mol = Chem.MolFromSmarts(pattern)
        if pattern_mol:
            # Highlight matches
            matches = mol.GetSubstructMatches(pattern_mol)
            if matches:
                highlight_atoms = [atom for match in matches for atom in match]
                svg = Draw.MolToSVG(mol, width=600, height=400, highlightAtoms=highlight_atoms)
                with open(IMAGES / "validator_pattern_matching.svg", 'w') as f:
                    f.write(svg)
                print("✓ Created validator_pattern_matching.svg")
    
    print("="*70 + "\n")
else:
    if not RDKIT_AVAILABLE:
        print("RDKit not available; skipping molecular visualizations\n")
    else:
        print("Matplotlib not available; skipping molecular visualizations\n")

# -----------------------------------------------------------------------------
# Backbone Visualization Generation (Example 21)
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("Generating Backbone Visualization Examples")
print("="*70)

if MATPLOTLIB_AVAILABLE:
    from q2D_Materials.modifier import GraphView
    
    # Create DJ structure with spacer for backbone analysis
    backbone_structure = q2d.create_structure(
        structure_type="bulk",
        A_ions="MA", B_ions="Pb", X_ions="I",
        template="cubic",
        layer_sequence="DJ",
        thickness=2,
        spacer="[NH3+]CCCCCC[NH3+]",  # Hexanediammonium
        xy_expansion=(1, 1),
    )
    
    backbone_analyzer = q2D_analyzer(backbone_structure)
    backbone_analyzer.analyze()
    
    view = GraphView(backbone_analyzer)
    spacers = view.spacers.list()
    
    if len(spacers) > 0:
        spacer = spacers[0]
        result = spacer.analyze_spacer()
        backbone = result.backbone
        
        # 1. Backbone structure visualization
        # Extract backbone atoms and visualize
        backbone_indices = backbone.indices()
        if backbone_indices:
            from ase import Atoms
            backbone_atoms = Atoms([backbone_structure[i].symbol for i in backbone_indices],
                                  positions=[backbone_structure[i].position for i in backbone_indices])
            save_atoms_svg(IMAGES / "backbone_structure.svg", backbone_atoms, rotation=rotation_iso, show_unit_cell=0)
            print("✓ Created backbone_structure.svg")
        
        # 2. Dihedral angles plot
        dihedrals = backbone.dihedrals()
        if dihedrals:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(len(dihedrals)), dihedrals, 'o-', linewidth=2, markersize=8, color='#ff5555')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=180, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=-180, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Dihedral Index', fontsize=12, fontweight='bold')
            ax.set_ylabel('Dihedral Angle (degrees)', fontsize=12, fontweight='bold')
            ax.set_title('Backbone Dihedral Angles', fontsize=14, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(IMAGES / "backbone_dihedrals_plot.svg", format='svg', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created backbone_dihedrals_plot.svg")
        
        # 3. Bond lengths and angles visualization
        bond_lengths = backbone.bond_lengths()
        bond_angles = backbone.bond_angles()
        
        if bond_lengths and bond_angles:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Bond lengths
            ax1.bar(range(len(bond_lengths)), bond_lengths, color='#50fa7b', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Bond Index', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Bond Length (Å)', fontsize=12, fontweight='bold')
            ax1.set_title('Backbone Bond Lengths', fontsize=13, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            
            # Bond angles
            ax2.bar(range(len(bond_angles)), bond_angles, color='#bd93f9', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Angle Index', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Bond Angle (degrees)', fontsize=12, fontweight='bold')
            ax2.set_title('Backbone Bond Angles', fontsize=13, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(IMAGES / "backbone_geometry.svg", format='svg', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created backbone_geometry.svg")
    
    print("="*70 + "\n")
else:
    print("Matplotlib not available; skipping backbone visualizations\n")

# -----------------------------------------------------------------------------
# Distortion Analysis Plot Generation (Example 22)
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("Generating Distortion Analysis Plot Examples")
print("="*70)

if MATPLOTLIB_AVAILABLE:
    # Use structure with some distortion
    distortion_structure = q2d.create_structure(
        structure_type="monolayer",
        A_ions="MA", B_ions="Pb", X_ions="I",
        xy_expansion=(2, 2), template="cubic",
        thickness=2, vacuum=15.0,
        glazer_pattern="a+b-b-",  # Add some tilting for distortion
    )
    
    distortion_analyzer = q2D_analyzer(distortion_structure)
    distortion_analyzer.analyze()
    
    # Apply publication style if available
    if STYLE_AVAILABLE:
        apply_publication_style()
    
    # 1. Complete 6-panel analysis
    if STYLE_AVAILABLE:
        fig = create_figure_with_style((15, 10))
    else:
        fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: B-X-B angles - Get data and plot
    from q2D_Materials.analyzer.characterization.distortions import _compute_octahedral_distortions
    bxb_data = distortion_analyzer.get_bxb_angles()
    angles = bxb_data['bxb_angles']
    ax1 = plt.subplot(2, 3, 1)
    if angles is not None and len(angles) > 0:
        ax1.hist(angles, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(bxb_data['bxb_mean'], color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('B-X-B Angle (degrees)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('B-X-B Angles')
    ax1.grid(alpha=0.3)
    
    # Plot 2: X-B-X angles - Get data and plot
    distortions = _compute_octahedral_distortions(distortion_analyzer)
    xbx_angles = distortions['bond_angles']
    ax2 = plt.subplot(2, 3, 2)
    if len(xbx_angles) > 0:
        ax2.hist(xbx_angles, bins=50, alpha=0.7, color='coral', edgecolor='black')
        ax2.axvline(distortions['mean_angle'], color='red', linestyle='--', linewidth=2)
        ax2.axvline(90, color='green', linestyle=':', linewidth=1, alpha=0.7)
        ax2.axvline(180, color='blue', linestyle=':', linewidth=1, alpha=0.7)
    ax2.set_xlabel('X-B-X Angle (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('X-B-X Angles')
    ax2.grid(alpha=0.3)
    
    # Plot 3: B-X bond lengths - Get data and plot
    bond_lengths = distortions['bond_lengths']
    ax3 = plt.subplot(2, 3, 3)
    if len(bond_lengths) > 0:
        ax3.hist(bond_lengths, bins=50, alpha=0.7, color='mediumseagreen', edgecolor='black')
        ax3.axvline(distortions['mean_bond_length'], color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('B-X Bond Length (Å)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('B-X Bond Lengths')
    ax3.grid(alpha=0.3)
    
    # Plot 4: All distances - Get data and plot
    from ase.neighborlist import neighbor_list
    i, j, d = neighbor_list('ijd', distortion_analyzer.cell, cutoff=8.0)
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(d, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Distance (Å)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('All Pairwise Distances')
    ax4.grid(alpha=0.3)
    if STYLE_AVAILABLE:
        style_axes(ax4)
    
    # Plot 5: RDF - Get data and plot
    # Auto-detect element pairs (B-X pairs)
    from q2D_Materials.analyzer.utils.geometry_helpers import get_all_x_atoms_from_octahedron
    octahedra = distortion_analyzer.get_octahedra()
    atom_symbols = distortion_analyzer.cell.get_chemical_symbols()
    
    b_elements = set()
    x_elements = set()
    
    for oct in octahedra:
        if oct['central_atom_symbol']:
            b_elements.add(oct['central_atom_symbol'])
        for idx in get_all_x_atoms_from_octahedron(oct):
            x_elements.add(atom_symbols[idx])
    
    # Create B-X pairs
    element_pairs = [[b, x] for b in sorted(b_elements) for x in sorted(x_elements)]
    
    if not element_pairs:
        # Fallback: use all unique elements (limit to 3 most common)
        unique_elements = sorted(set(atom_symbols))[:3]
        element_pairs = [[e1, e2] for i, e1 in enumerate(unique_elements)
                        for e2 in unique_elements[i:]]
    
    rdf_data = distortion_analyzer.get_partial_rdf(element_pairs=element_pairs, max_dist=10.0)
    ax5 = plt.subplot(2, 3, 5)
    colors = plt.cm.tab10(np.linspace(0, 1, len([k for k in rdf_data.keys() if k.endswith('_rdf')])))
    idx = 0
    for key in sorted(rdf_data.keys()):
        if key.endswith('_rdf'):
            pair_key = key.replace('_rdf', '')
            x_key = f"{pair_key}_x"
            if x_key in rdf_data:
                ax5.plot(rdf_data[x_key], rdf_data[key], 
                        label=f"{pair_key[0]}-{pair_key[1]}" if len(pair_key) == 2 else pair_key,
                        color=colors[idx], linewidth=2)
                idx += 1
    ax5.set_xlabel('Distance (Å)')
    ax5.set_ylabel('g(r)')
    ax5.set_title('Radial Distribution Function')
    ax5.legend(loc='best')
    ax5.grid(alpha=0.3)
    ax5.set_xlim(0, 10.0)
    if STYLE_AVAILABLE:
        style_axes(ax5)
    
    # Plot 6: Glazer pattern
    ax6 = plt.subplot(2, 3, 6)
    try:
        glazer = distortion_analyzer.get_glazer_pattern()
        ax6.text(0.5, 0.5, f"Glazer: {glazer['notation']}\n\nTilt Angles:\n{glazer['tilt_angles']}°",
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                transform=ax6.transAxes)
    except Exception:
        ax6.text(0.5, 0.5, "Glazer pattern\nnot detected",
                ha='center', va='center', fontsize=14, transform=ax6.transAxes)
    ax6.axis('off')
    ax6.set_title('Glazer Pattern')
    if STYLE_AVAILABLE:
        style_axes(ax6)
    
    plt.tight_layout()
    
    if STYLE_AVAILABLE:
        style_axes(ax1)
        style_axes(ax2)
        style_axes(ax3)
        style_axes(ax4)
        style_axes(ax5)
        save_figure(fig, IMAGES / "distortion_complete_analysis.svg", format='svg')
    else:
        plt.savefig(IMAGES / "distortion_complete_analysis.svg", format='svg', dpi=300, bbox_inches='tight')
        plt.close()
    print("✓ Created distortion_complete_analysis.svg")
    
    # 2. Layer-based distortion comparison
    try:
        delta_by_layer = distortion_analyzer.compute_delta(group_by='layer')
        sigma_by_layer = distortion_analyzer.compute_sigma(group_by='layer')
        lambda_by_layer = distortion_analyzer.compute_lambda(group_by='layer')
        
        layer_ids = [k for k in delta_by_layer.keys() if k != 'global']
        
        if len(layer_ids) > 0:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            x_pos = np.arange(len(layer_ids))
            width = 0.6
            
            # Delta
            delta_values = [delta_by_layer[lid] for lid in layer_ids]
            ax1.bar(x_pos, delta_values, width, color='#ff5555', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Δ (Bond Length Distortion)', fontsize=12, fontweight='bold')
            ax1.set_title('Delta by Layer', fontsize=13, fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([f'L{lid}' for lid in layer_ids])
            ax1.grid(axis='y', alpha=0.3)
            
            # Sigma
            sigma_values = [sigma_by_layer[lid] for lid in layer_ids]
            ax2.bar(x_pos, sigma_values, width, color='#50fa7b', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
            ax2.set_ylabel('σ² (Bond Length Variance)', fontsize=12, fontweight='bold')
            ax2.set_title('Sigma by Layer', fontsize=13, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([f'L{lid}' for lid in layer_ids])
            ax2.grid(axis='y', alpha=0.3)
            
            # Lambda
            lambda_values = [lambda_by_layer[lid] for lid in layer_ids]
            ax3.bar(x_pos, lambda_values, width, color='#bd93f9', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Layer', fontsize=12, fontweight='bold')
            ax3.set_ylabel('λ² (Bond Angle Variance)', fontsize=12, fontweight='bold')
            ax3.set_title('Lambda by Layer', fontsize=13, fontweight='bold')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([f'L{lid}' for lid in layer_ids])
            ax3.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(IMAGES / "distortion_layer_comparison.svg", format='svg', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created distortion_layer_comparison.svg")
    except Exception as e:
        print(f"⚠ Could not create layer comparison (may need multilayer structure): {e}")
    
    # 3. Parameters summary
    try:
        delta = distortion_analyzer.compute_delta()
        sigma = distortion_analyzer.compute_sigma()
        lambda_param = distortion_analyzer.compute_lambda()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        params = ['Δ (Delta)', 'σ² (Sigma)', 'λ² (Lambda)']
        values = [delta, sigma, lambda_param]
        colors = ['#ff5555', '#50fa7b', '#bd93f9']
        
        bars = ax.bar(params, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Distortion Parameter Value', fontsize=12, fontweight='bold')
        ax.set_title('Octahedral Distortion Parameters Summary', fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(IMAGES / "distortion_parameters_summary.svg", format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created distortion_parameters_summary.svg")
    except Exception as e:
        print(f"⚠ Could not create parameters summary: {e}")
    
    print("="*70 + "\n")
else:
    print("Matplotlib not available; skipping distortion plots\n")

# -----------------------------------------------------------------------------
# Cavity Visualization
# -----------------------------------------------------------------------------
if MATPLOTLIB_AVAILABLE:
    print("\n" + "="*70)
    print("Creating Cavity Visualization...")
    print("="*70)
    
    try:
        from q2D_Materials.analyzer import q2D_analyzer
        from q2D_Materials.analyzer.cavities_processing.cavity_tracing import (
            calculate_cavity_deformation,
            calculate_cavity_volume,
        )
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Create 4x4x2 cubic structure
        structure = q2d.create_structure(
            A_ions='MA',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(4, 4),
        )
        
        # Save temporarily and analyze
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vasp', delete=False) as tmp:
            tmp_path = tmp.name
            write(tmp_path, structure)
        
        analyzer = None
        try:
            analyzer = q2D_analyzer(tmp_path)
            analyzer.analyze()
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        if analyzer is None:
            raise Exception("Failed to create analyzer")
        
        # Get cavities
        cavities = analyzer.get_cavities()
        atom_positions = analyzer.cell.get_positions()
        atom_symbols = analyzer.cell.get_chemical_symbols()
        cell = analyzer.cell.get_cell()
        
        # Build octahedra data
        octahedra_data = {}
        for node, data in analyzer._graph.nodes(data=True):
            if data.get('node_type') == 'octahedron':
                oct_idx = int(node.replace('octahedron_', ''))
                octahedra_data[oct_idx] = {
                    'central_atom': data.get('central_atom'),
                    'terminal_atoms': data.get('terminal_atoms', []),
                    'interlayer_atoms': data.get('interlayer_atoms', []),
                    'intralayer_atoms': data.get('intralayer_atoms', []),
                }
        
        # Find a cavity that's fully inside the cell (not PBC-wrapped)
        selected_cavity = None
        for cavity in cavities:
            if not getattr(cavity, 'is_pbc_wrapped', True):
                selected_cavity = cavity
                break
        
        # If all are wrapped, use the first one
        if selected_cavity is None and cavities:
            selected_cavity = cavities[0]
        
        if selected_cavity:
            # Get all atoms in the cavity using Cavity object attributes
            # Get octahedra from octahedra_info
            octahedra_info = getattr(selected_cavity, 'octahedra_info', {})
            all_octs = list(octahedra_info.keys()) if octahedra_info else []
            
            # Collect cavity X atoms
            cavity_x_atoms = set()
            cavity_b_atoms = set()
            cavity_a_atoms = set()
            
            # Use b_atom_indices and x_atom_indices directly from Cavity
            cavity_b_atoms.update(getattr(selected_cavity, 'b_atom_indices', []))
            cavity_x_atoms.update(getattr(selected_cavity, 'x_atom_indices', []))
            
            # Get A-site atoms in this cavity
            a_site_indices = getattr(selected_cavity, 'a_site_indices', [])
            cavity_a_atoms.update(a_site_indices)
            
            # Create figure with 3D subplot
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Color scheme
            x_color = '#ff6b6b'  # Red for X atoms (I)
            b_color = '#4ecdc4'   # Cyan for B atoms (Pb)
            a_color = '#ffe66d'   # Yellow for A atoms (MA)
            other_color = '#95a5a6'  # Gray for other atoms
            
            # Plot non-cavity atoms with alpha
            for i, (pos, symbol) in enumerate(zip(atom_positions, atom_symbols)):
                if i in cavity_x_atoms or i in cavity_b_atoms or i in cavity_a_atoms:
                    continue  # Skip cavity atoms (will plot them separately)
                
                # Determine color based on element
                if symbol in ['I', 'Br', 'Cl', 'F']:
                    color = x_color
                elif symbol in ['Pb', 'Sn', 'Ge']:
                    color = b_color
                elif symbol in ['C', 'N', 'H']:
                    color = a_color
                else:
                    color = other_color
                
                ax.scatter(pos[0], pos[1], pos[2], 
                          c=color, s=30, alpha=0.15, edgecolors='none')
            
            # Plot cavity atoms with full opacity
            # X atoms (cavity walls)
            for i in cavity_x_atoms:
                if i < len(atom_positions):
                    pos = atom_positions[i]
                    ax.scatter(pos[0], pos[1], pos[2], 
                              c=x_color, s=200, alpha=1.0, 
                              edgecolors='black', linewidths=1.5, 
                              label='Cavity X atoms' if i == list(cavity_x_atoms)[0] else '')
            
            # B atoms (octahedra centers)
            for i in cavity_b_atoms:
                if i < len(atom_positions):
                    pos = atom_positions[i]
                    ax.scatter(pos[0], pos[1], pos[2], 
                              c=b_color, s=250, alpha=1.0, 
                              edgecolors='black', linewidths=1.5,
                              marker='^', label='Cavity B atoms' if i == list(cavity_b_atoms)[0] else '')
            
            # A atoms (molecules in cavity)
            for i in cavity_a_atoms:
                if i < len(atom_positions):
                    pos = atom_positions[i]
                    ax.scatter(pos[0], pos[1], pos[2], 
                              c=a_color, s=150, alpha=1.0, 
                              edgecolors='black', linewidths=1.5,
                              marker='s', label='Cavity A atoms' if i == list(cavity_a_atoms)[0] else '')
            
            # Set labels and title
            ax.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y (Å)', fontsize=12, fontweight='bold')
            ax.set_zlabel('Z (Å)', fontsize=12, fontweight='bold')
            ax.set_title('Cavity Visualization (4×4×2 Cubic Perovskite)\n' +
                        f'Cavity with {len(cavity_x_atoms)} X atoms, {len(cavity_b_atoms)} B atoms, {len(cavity_a_atoms)} A atoms',
                        fontsize=14, fontweight='bold', pad=20)
            
            # Set equal aspect ratio
            max_range = np.array([atom_positions[:, 0].max() - atom_positions[:, 0].min(),
                                 atom_positions[:, 1].max() - atom_positions[:, 1].min(),
                                 atom_positions[:, 2].max() - atom_positions[:, 2].min()]).max() / 2.0
            mid_x = (atom_positions[:, 0].max() + atom_positions[:, 0].min()) * 0.5
            mid_y = (atom_positions[:, 1].max() + atom_positions[:, 1].min()) * 0.5
            mid_z = (atom_positions[:, 2].max() + atom_positions[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            # Remove duplicates
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), 
                     loc='upper left', fontsize=10, framealpha=0.9)
            
            # Set viewing angle
            ax.view_init(elev=20, azim=45)
            
            plt.tight_layout()
            plt.savefig(IMAGES / "cavity_visualization.svg", format='svg', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created cavity_visualization.svg")
            
            # Print cavity analysis metrics
            try:
                deformation = calculate_cavity_deformation(
                    selected_cavity, octahedra_data, atom_positions, cell
                )
                volume = calculate_cavity_volume(
                    selected_cavity, octahedra_data, atom_positions, cell
                )
                print(f"  Cavity metrics:")
                if deformation.get('symmetry_score') is not None:
                    print(f"    Symmetry score: {deformation['symmetry_score']:.3f}")
                if volume > 0:
                    print(f"    Volume: {volume:.2f} Ų")
            except Exception as e:
                print(f"  Could not compute cavity metrics: {e}")
        else:
            print("⚠ No cavities found in structure")
            
    except Exception as e:
        print(f"⚠ Could not create cavity visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*70 + "\n")

# -----------------------------------------------------------------------------
# Cavity Visualization (Example 22)
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("Generating Cavity Visualization Examples")
print("="*70)

from q2D_Materials.analyzer import q2D_analyzer

# Create the same structure as test_graph_expoert.py
cavity_structure = q2d.create_structure(
    A_ions="MA",
    B_ions="Pb",
    X_ions="I",
    template="reduced",
    layer_sequence="DJ",
    thickness=2,
    spacer=["[NH3+]CCCC[NH3+]", "[NH3+]C1C=CCC=CC1C"],
    glazer_angles=[0, 0, 3],
    glazer_pattern=["0", "0", "+"],
)

# Analyze structure
cavity_analyzer = q2D_analyzer(cavity_structure)
cavity_analyzer.analyze()

# Get cavities using on-demand detection
cavities = cavity_analyzer.get_cavities()
print(f"Found {len(cavities)} cavities")

# Convert all cavities to ASE Atoms objects
cavity_atoms_list = cavities.to_atoms()

# Save specific cavities as PNG images: cavity_0, cavity_2, cavity_3
cavity_indices_to_plot = [0, 2, 3]

for cavity_idx in cavity_indices_to_plot:
    if cavity_idx < len(cavity_atoms_list):
        cavity_atoms = cavity_atoms_list[cavity_idx]
        
        # Get cavity info for filename and title
        if cavity_idx < len(cavities):
            cavity = cavities[cavity_idx]
            cavity_type = getattr(cavity, 'cavity_type', 'unknown')
            
            # Count atoms from subgraph using graph structure
            subgraph = getattr(cavity, 'subgraph', None)
            if subgraph is not None:
                # Get B atom indices from octahedra_info (B atoms are octahedron centers)
                octahedra_info = getattr(cavity, 'octahedra_info', {})
                b_atom_indices = set(octahedra_info.keys())  # Keys are B atom indices
                
                # Find A atoms: nodes connected to molecule nodes via CONTAINS edges
                a_atom_nodes = set()
                for node in subgraph.nodes():
                    node_data = subgraph.nodes.get(node, {})
                    if node_data.get('node_type') == 'molecule':
                        # Find all atoms connected to this molecule
                        for neighbor in subgraph.neighbors(node):
                            edge_data = subgraph.get_edge_data(node, neighbor)
                            if edge_data and edge_data.get('edge_type') == 'contains':
                                a_atom_nodes.add(neighbor)
                
                # Count atoms from subgraph
                b_count = 0
                x_count = 0
                a_count = 0
                
                for node in subgraph.nodes():
                    node_data = subgraph.nodes.get(node, {})
                    if node_data.get('node_type') == 'atom':
                        original_idx = node_data.get('original_index')
                        
                        # A atoms: connected to molecule nodes
                        if node in a_atom_nodes:
                            a_count += 1
                        # B atoms: octahedron centers (original_index in octahedra_info keys)
                        elif original_idx is not None and original_idx in b_atom_indices:
                            b_count += 1
                        # X atoms: everything else
                        else:
                            x_count += 1
            else:
                # Fallback if subgraph not available
                b_count = len(getattr(cavity, 'b_atom_indices', []))
                x_count = len(getattr(cavity, 'x_atom_indices', []))
                a_count = len(getattr(cavity, 'a_site_indices', []))
            
            # Determine cavity label
            if getattr(cavity, 'contains_a_site', False):
                cav_label = "A-site"
            elif cavity_type == 'spacer_dj':
                cav_label = "DJ Spacer"
            elif cavity_type == 'spacer_rp':
                cav_label = "RP Spacer"
            else:
                cav_label = "Cavity"
            
            filename = f"cavity_{cavity_idx}.svg"
            save_atoms_svg(IMAGES / filename, cavity_atoms, 
                  rotation=rotation_iso, show_unit_cell=0)
            print(f"✓ Created {filename} ({cav_label}: {b_count}B + {x_count}X + {a_count}A)")
        else:
            filename = f"cavity_{cavity_idx}.svg"
            save_atoms_svg(IMAGES / filename, cavity_atoms, 
                  rotation=rotation_iso, show_unit_cell=0)
            print(f"✓ Created {filename}")
    else:
        print(f"⚠ Cavity {cavity_idx} not found (only {len(cavity_atoms_list)} cavities)")

print("="*70 + "\n")

print("\n" + "="*70)
print("All Example Images Generated Successfully!")
print("="*70 + "\n")

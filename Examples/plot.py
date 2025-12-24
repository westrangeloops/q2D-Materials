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
    xy_expansion=(1, 1),
    A_ions="MA",
    B_ions="Pb",
    X_ions="I",
    spacer="Cs",
    glazer_angles=[3, 2, 6],
    glazer_pattern=["+", "-", "-"],
    penetration=0.2,
)
write(IMAGES / "rp_atomic_side.png", rp_atomic, rotation=rotation_iso, show_unit_cell=2)


def save_multiview(atoms, filename, rotations, titles=None, radii=0.25):
    if not MATPLOTLIB_AVAILABLE:
        print(f"Matplotlib not available; skipping {filename}")
        return
    fig, axarr = plt.subplots(1, len(rotations), figsize=(4 * len(rotations), 4))
    if len(rotations) == 1:
        axarr = [axarr]
    for i, (ax, rot) in enumerate(zip(axarr, rotations)):
        plot_atoms(atoms, ax, radii=radii, rotation=rot)
        ax.set_axis_off()
        if titles and i < len(titles):
            ax.set_title(titles[i])
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


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

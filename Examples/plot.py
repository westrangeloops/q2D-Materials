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
    sharp_spacer="[NH3+]CCCC[NH3+]",
    glazer_angles=[0, 0, 3],
    glazer_pattern=["0", "0", "+"],
)
write(IMAGES / "dj_bulk.png", dj, rotation=rotation_iso, show_unit_cell=2)

# DJ atomic spacer example (Cs)
dj_atomic = q2d.create_structure(
    **(dj_base | {"xy_expansion": (2, 2)}),
    layer_sequence="DJ",
    thickness=3,
    sharp_spacer="Cs",
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
    sharp_spacer="[NH3+]CCCC[NH3+]",
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
    sharp_spacer=["[NH3+]CCCCCCC", "[NH3+]CCCC[NH3+]"],
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
    sharp_spacer="Cs",
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
        sharp_spacer="[NH3+]CCCC[NH3+]",
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
        sharp_spacer=["C=CCC=CCC[NH3+]", "[NH3+]CCCC[NH3+]"],
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

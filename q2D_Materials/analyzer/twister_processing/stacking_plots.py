"""Heatmap visualization for twister stacking registry."""

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from .stacking_analysis import StackingRegistryResult

RATIO_LABEL_NORMALIZED = (
    r"$R = \frac{2\,d_{X \to cation}}{d_{X \to X} + d_{X \to cation}}$"
)
RATIO_LABEL_UNNORMALIZED = r"$r = \frac{d_{X \to cation}}{d_{X \to X}}$"


def unnormalized_ratio(registry: StackingRegistryResult) -> np.ndarray:
    """Notebook per-atom ratio ``d(X→cation) / d(X→X)`` from raw distances."""
    d_cat = np.asarray(registry.raw.get("d_x_to_cation", []), dtype=float)
    d_xx = np.asarray(registry.raw.get("d_x_to_x", []), dtype=float)
    n = min(len(d_cat), len(d_xx))
    if n == 0:
        return np.array([])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d_xx[:n] > 0, d_cat[:n] / d_xx[:n], np.nan)


def plot_stacking_heatmap(
    registry: StackingRegistryResult,
    output_path: Optional[str] = None,
    show_atoms: bool = True,
    grid_pts: int = 300,
    smoothing: Optional[float] = None,
    colorbar_label: Optional[str] = None,
    dpi: int = 600,
) -> None:
    """Plot continuous R(x,y) stacking ratio heatmap from a registry result."""
    from scipy.interpolate import RBFInterpolator

    ratio = np.asarray(registry.ratio_per_x)
    xy_I = np.asarray(registry.xy_x_interface)
    lat = registry.lattice

    n = min(len(ratio), len(xy_I))
    if n == 0:
        raise ValueError("Registry has no interface X atoms / ratios to plot.")
    ratio = ratio[:n]
    xy_I = xy_I[:n]
    if colorbar_label is None:
        colorbar_label = RATIO_LABEL_NORMALIZED

    a1 = lat[0, :2]
    a2 = lat[1, :2]

    shifts = np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)])
    xy_exp = np.vstack([xy_I + s[0] * a1 + s[1] * a2 for s in shifts])
    r_exp = np.tile(ratio, len(shifts))

    frac_u = np.linspace(0, 1, grid_pts)
    frac_v = np.linspace(0, 1, grid_pts)
    Fu, Fv = np.meshgrid(frac_u, frac_v)
    Xg = Fu * a1[0] + Fv * a2[0]
    Yg = Fu * a1[1] + Fv * a2[1]
    xy_grid = np.column_stack([Xg.ravel(), Yg.ravel()])

    eps = smoothing if smoothing is not None else 1.0
    rbf = RBFInterpolator(xy_exp, r_exp, kernel='thin_plate_spline', smoothing=eps * len(xy_exp))
    R_grid = rbf(xy_grid).reshape(grid_pts, grid_pts)

    cmap = plt.cm.RdBu_r
    vmax = max(abs(ratio - 1.0).max() * 1.05, 0.05) + 1.0
    vmin = 2.0 - vmax
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(4, 4))
    pc = ax.pcolormesh(Xg, Yg, R_grid, cmap=cmap, norm=norm, shading='gouraud', rasterized=True)
    cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(colorbar_label, fontsize=13)
    cb.ax.axhline(1.0, color='k', lw=1.2, ls='--')

    if show_atoms:
        ax.scatter(
            xy_I[:, 0], xy_I[:, 1],
            c=ratio, cmap=cmap, norm=norm,
            s=30, edgecolors='k', linewidths=0.6, zorder=5,
        )

    corners_frac = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    corners_xy = corners_frac @ np.vstack([a1, a2])
    ax.plot(corners_xy[:, 0], corners_xy[:, 1], 'k-', lw=1.0, alpha=0.6)

    ax.set_xlabel('x  (Å)', fontsize=11)
    ax.set_ylabel('y  (Å)', fontsize=11)
    ax.set_aspect('equal')
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)

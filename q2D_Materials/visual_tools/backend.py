# backend.py - FastAPI backend for Layer Architect
from fastapi import FastAPI, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
import json
import io

# Matplotlib setup - MUST be before pyplot import
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

app = FastAPI(title="Layer Architect API")

# Enable CORS for Vue local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for JSmol
# This serves files from the 'static' directory at /static
BACKEND_DIR = Path(__file__).parent
STATIC_DIR = BACKEND_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- State Management (In-Memory) ---
state = {
    "layers": {"L1": []},  # "L1": [ ["A", 0.5, 0.5], ... ]
    "layer_stack": [],
    "lattice": {"a": 5.0, "b": 5.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    "current_layer": "L1",
    "grid_options": {
        "cartesian": True,
        "diagonal": False,
        "step_x": 0.2,
        "step_y": 0.2,
        "step_diag": 0.2
    }
}

# --- Helper Functions ---
def get_lattice_vectors(a_mult, b_mult, gamma_deg):
    """Calculate 2D lattice vectors."""
    gamma_rad = np.radians(gamma_deg)
    v1 = np.array([a_mult, 0.0])
    v2 = np.array([b_mult * np.cos(gamma_rad), b_mult * np.sin(gamma_rad)])
    return v1, v2

def frac_to_cart(u, v, v1, v2):
    return u * v1 + v * v2

# --- Data Models ---
class AtomModel(BaseModel):
    code: str
    u: float
    v: float

class LayerCreate(BaseModel):
    name: str

class LatticeUpdate(BaseModel):
    a: float
    b: float
    alpha: Optional[float] = 90.0
    beta: Optional[float] = 90.0
    gamma: Optional[float] = 90.0

class GridOptions(BaseModel):
    cartesian: bool = True
    diagonal: bool = False
    step_x: float = 0.2    # Default step for U (Vertical lines)
    step_y: float = 0.2    # Default step for V (Horizontal lines)
    step_diag: float = 0.2 # Default step for Diagonals

# --- Endpoints ---

# Get the directory where this file is located
BACKEND_DIR = Path(__file__).parent
HTML_FILE = BACKEND_DIR / "index.html"

@app.get("/")
async def read_root():
    """Serve the main HTML file at the root URL."""
    if HTML_FILE.exists():
        return FileResponse(HTML_FILE)
    else:
        return {"message": "Frontend HTML file not found. Please ensure index.html is in the same directory as backend.py"}

@app.get("/state")
def get_state():
    """Returns the full UI state to sync Vue."""
    return state

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/lattice")
def update_lattice(params: LatticeUpdate):
    """Update lattice parameters."""
    state["lattice"].update(params.dict(exclude_unset=True))
    return state["lattice"]

@app.post("/grid")
def update_grid_options(options: GridOptions):
    """Update grid display options."""
    state["grid_options"] = options.dict()
    return state["grid_options"]

@app.post("/layer/select/{name}")
def select_layer(name: str):
    """Select active layer."""
    if name in state["layers"]:
        state["current_layer"] = name
    return {"current_layer": state["current_layer"]}

@app.post("/layer/create")
def create_layer(layer: LayerCreate):
    """Create a new layer."""
    if layer.name not in state["layers"]:
        state["layers"][layer.name] = []
        state["current_layer"] = layer.name
    return state

@app.post("/atom/add")
def add_atom(atom: AtomModel):
    """Add atom to current layer."""
    layer = state["current_layer"]
    if layer not in state["layers"]:
        state["layers"][layer] = []
    # Make sure we're appending to the existing list, not replacing it
    state["layers"][layer] = state["layers"][layer] + [[atom.code, atom.u, atom.v]]
    return state["layers"][layer]

@app.post("/atom/delete")
def delete_atom(idx: int = Body(..., embed=True)):
    """Delete atom from current layer."""
    layer = state["current_layer"]
    if 0 <= idx < len(state["layers"][layer]):
        state["layers"][layer].pop(idx)
    return state["layers"][layer]

@app.post("/atom/update")
def update_atom(idx: int = Body(..., embed=True), u: float = Body(..., embed=True), v: float = Body(..., embed=True)):
    """Update atom position in current layer."""
    layer = state["current_layer"]
    if 0 <= idx < len(state["layers"][layer]):
        # Keep the same code, just update u and v
        code = state["layers"][layer][idx][0]
        state["layers"][layer][idx] = [code, u, v]
    return state["layers"][layer]

@app.post("/stack/add")
def add_to_stack():
    """Add current layer to stack."""
    if state["current_layer"] not in state["layer_stack"]:
        state["layer_stack"].append(state["current_layer"])
    return state["layer_stack"]

@app.post("/stack/remove/{idx}")
def remove_from_stack(idx: int):
    """Remove layer from stack by index."""
    if 0 <= idx < len(state["layer_stack"]):
        state["layer_stack"].pop(idx)
    return state["layer_stack"]

@app.post("/stack/clear")
def clear_stack():
    """Clear the layer stack."""
    state["layer_stack"] = []
    return state["layer_stack"]

@app.get("/structure/3d")
def get_3d_structure(spacing: float = 3.0):
    """Generate XYZ string for Jmol visualization."""
    if not state["layer_stack"]:
        return {"xyz": ""}
    
    v1, v2 = get_lattice_vectors(
        state["lattice"]["a"],
        state["lattice"]["b"],
        state["lattice"]["gamma"]
    )
    
    # Map your codes to real Elements for Jmol visualization
    element_map = {
        "A": "Cs",  # Cesium (Big Green)
        "B": "Pb",  # Lead (Grey/Blue)
        "X": "I",   # Iodine (Purple/Red)
        "S1": "C",  # Carbon (Grey)
        "S2": "N"   # Nitrogen (Blue)
    }
    
    atoms_list = []
    
    # Build atom list
    for i, layer_name in enumerate(state["layer_stack"]):
        z = i * spacing
        if layer_name in state["layers"]:
            for atom in state["layers"][layer_name]:
                code, u, v = atom[:3]
                xy = frac_to_cart(u, v, v1, v2)
                
                # Handle spacers or unknown codes
                elem = element_map.get(code, "C")
                if code.startswith("S") and code not in element_map:
                    elem = "C"  # Default spacers to Carbon
                
                atoms_list.append(f"{elem} {xy[0]:.4f} {xy[1]:.4f} {z:.4f}")
    
    # Construct XYZ String
    # Line 1: Atom Count
    # Line 2: Comment
    # Line 3+: Element X Y Z
    xyz_str = f"{len(atoms_list)}\nGenerated by Layer Architect\n" + "\n".join(atoms_list)
    
    return {"xyz": xyz_str}

@app.post("/export")
def export_structure():
    """Export current structure as JSON."""
    export_obj = {
        "named_layers": state["layers"],
        "layer_stack": state["layer_stack"],
        "lattice_multipliers": [state["lattice"]["a"], state["lattice"]["b"]],
        "angles": [
            state["lattice"]["alpha"],
            state["lattice"]["beta"],
            state["lattice"]["gamma"]
        ]
    }
    return export_obj

@app.post("/import")
async def import_structure(file: UploadFile = File(...)):
    """Import structure from JSON file."""
    try:
        content = await file.read()
        data = json.loads(content)
        
        if "named_layers" in data:
            state["layers"] = data["named_layers"]
        if "layer_stack" in data:
            state["layer_stack"] = data["layer_stack"]
        if "lattice_multipliers" in data:
            state["lattice"]["a"] = data["lattice_multipliers"][0]
            state["lattice"]["b"] = data["lattice_multipliers"][1]
        if "angles" in data:
            state["lattice"]["alpha"] = data["angles"][0]
            state["lattice"]["beta"] = data["angles"][1]
            state["lattice"]["gamma"] = data["angles"][2]
        
        # Set current layer to first available
        if state["layers"]:
            state["current_layer"] = list(state["layers"].keys())[0]
        
        return {"status": "success", "state": state}
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )

@app.get("/render/3d")
def render_3d_view(angle: float = 45.0, elev: float = 30.0, spacing: float = 3.0):
    """
    Generates a static PNG of the 3D structure at a specific azimuth angle and elevation.
    angle: azimuth rotation (0-360 degrees)
    elev: elevation angle (0-180 degrees, 0=from below, 90=side, 180=from above)
    """
    if not state["layer_stack"]:
        # Return a transparent empty image
        fig = plt.figure(figsize=(4, 4), dpi=100)
        fig.patch.set_alpha(0.0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        return Response(content=buf.getvalue(), media_type="image/png")

    # 1. Build Data (Reuse logic)
    v1, v2 = get_lattice_vectors(
        state["lattice"]["a"],
        state["lattice"]["b"],
        state["lattice"]["gamma"]
    )
    # Color map - handle all spacer types dynamically
    color_map = {"A": "#50fa7b", "B": "#8be9fd", "X": "#ff5555"}
    # Spacers (S1, S2, S3, etc.) all use orange
    spacer_color = "#ffb86c"
    
    xs, ys, zs, c_list, s_list, labels_list = [], [], [], [], [], []
    
    for i, layer_name in enumerate(state["layer_stack"]):
        z = i * spacing
        if layer_name in state["layers"]:
            for atom in state["layers"][layer_name]:
                code, u, v = atom[:3]
                xy = frac_to_cart(u, v, v1, v2)
                
                xs.append(float(xy[0]))
                ys.append(float(xy[1]))
                zs.append(float(z))
                # Use spacer color for any code starting with 'S', otherwise use color_map
                if code.startswith('S'):
                    c_list.append(spacer_color)
                else:
                    c_list.append(color_map.get(code, "#6272a4"))  # Default to comment color
                # Size: A=200, B=160, X=130, Spacers=110 (5px larger proportionally)
                if code == "A":
                    s_list.append(200)
                    labels_list.append("")  # No label for A
                elif code == "B":
                    s_list.append(160)
                    labels_list.append("")  # No label for B
                elif code == "X":
                    s_list.append(130)
                    labels_list.append("")  # No label for X
                else:  # Spacers
                    s_list.append(110)
                    labels_list.append(code)  # Show label for spacers (S1, S2, S3, etc.)

    # 2. Plot with Matplotlib
    # Use a larger figsize for better detail
    fig = plt.figure(figsize=(6, 6), dpi=120, facecolor='#272935')
    ax = fig.add_subplot(111, projection='3d')
    
    # Dark background matching Dracula theme
    fig.patch.set_facecolor('#272935')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#44475a')
    ax.yaxis.pane.set_edgecolor('#44475a')
    ax.zaxis.pane.set_edgecolor('#44475a')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    if xs and len(xs) > 0:
        # Plot atoms with different markers: A=circle, B=square, X=X, Spacers=triangle
        # We need to group by type to use different markers
        from collections import defaultdict
        atom_groups = defaultdict(lambda: {'x': [], 'y': [], 'z': [], 'c': [], 's': [], 'labels': []})
        
        # Build list of codes in same order as xs, ys, zs
        codes_list = []
        for layer_name in state["layer_stack"]:
            if layer_name in state["layers"]:
                for atom in state["layers"][layer_name]:
                    codes_list.append(atom[0])
        
        # Group atoms by type
        for i, code in enumerate(codes_list):
            atom_groups[code]['x'].append(xs[i])
            atom_groups[code]['y'].append(ys[i])
            atom_groups[code]['z'].append(zs[i])
            atom_groups[code]['c'].append(c_list[i])
            atom_groups[code]['s'].append(s_list[i])
            atom_groups[code]['labels'].append(labels_list[i])
        
        # Plot each type with appropriate marker
        for code, group in atom_groups.items():
            if code.startswith('A'):
                marker = 'o'      # Circle
            elif code.startswith('B'):
                marker = 's'      # Square
            elif code.startswith('X'):
                marker = 'x'      # X
            else:  # Spacers
                marker = '^'      # Triangle
            
            if len(group['x']) > 0:
                ax.scatter(
                    group['x'], group['y'], group['z'],
                    c=group['c'], s=group['s'],
                    marker=marker,
                    edgecolors='#272935', linewidth=1.2,
                    depthshade=True, alpha=0.95
                )
                
                # Add text labels for spacers only (S1, S2, S3, etc.)
                # Position labels above the triangles so they're visible
                if code.startswith('S'):
                    for j in range(len(group['x'])):
                        label = group['labels'][j]
                        if label:  # Only label if not empty
                            # Calculate offset to position label above the triangle
                            # Use a small offset in the Z direction (upward)
                            z_offset = 0.15  # Offset in Angstroms to position label above triangle
                            ax.text(
                                group['x'][j], group['y'][j], group['z'][j] + z_offset,
                                label,
                                fontsize=8,
                                color='#f8f8f2',
                                ha='center',
                                va='bottom',  # Align to bottom so text sits above the point
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='#272935', alpha=0.7, edgecolor='#6272a4', linewidth=0.5)
                            )
        
        # Calculate bounds with minimal padding (like Streamlit version)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        # Add minimal padding (1% on each side, max 0.05)
        padding_x = max(x_range * 0.01, 0.05) if x_range > 0 else 0.05
        padding_y = max(y_range * 0.01, 0.05) if y_range > 0 else 0.05
        padding_z = max(z_range * 0.01, 0.05) if z_range > 0 else 0.05
        
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)
        ax.set_zlim(z_min - padding_z, z_max + padding_z)
        
        # Draw unit cell boundaries in the XY plane (bottom layer)
        if len(xs) > 0:
            # Get lattice vectors for the bottom layer
            v1, v2 = get_lattice_vectors(
                state["lattice"]["a"],
                state["lattice"]["b"],
                state["lattice"]["gamma"]
            )
            
            # Draw unit cell outline at bottom (min_z)
            cell_z = z_min - padding_z * 0.3
            cell_corners = [
                [0, 0, cell_z],
                [v1[0], v1[1], cell_z],
                [v1[0] + v2[0], v1[1] + v2[1], cell_z],
                [v2[0], v2[1], cell_z],
                [0, 0, cell_z]  # Close the loop
            ]
            
            # Draw cell edges (purple, like in 2D view)
            for i in range(len(cell_corners) - 1):
                ax.plot3D(
                    [cell_corners[i][0], cell_corners[i+1][0]],
                    [cell_corners[i][1], cell_corners[i+1][1]],
                    [cell_corners[i][2], cell_corners[i+1][2]],
                    color='#bd93f9', linewidth=2.0, alpha=0.7
                )
            
            # Draw vertical lines at corners to show stacking direction
            if len(state["layer_stack"]) > 1:
                for corner in cell_corners[:-1]:  # Exclude duplicate last point
                    ax.plot3D(
                        [corner[0], corner[0]],
                        [corner[1], corner[1]],
                        [cell_z, z_max + padding_z * 0.5],
                        color='#6272a4', linewidth=1.0, alpha=0.4, linestyle='--'
                    )
            
            # Also draw unit cell at top layer for reference
            if len(state["layer_stack"]) > 1:
                top_cell_z = z_max + padding_z * 0.3
                for i in range(len(cell_corners) - 1):
                    ax.plot3D(
                        [cell_corners[i][0], cell_corners[i+1][0]],
                        [cell_corners[i][1], cell_corners[i+1][1]],
                        [top_cell_z, top_cell_z],
                        color='#bd93f9', linewidth=1.5, alpha=0.5, linestyle=':'
                    )
        
        # Style axes for better visibility
        ax.xaxis.label.set_color('#8be9fd')
        ax.yaxis.label.set_color('#8be9fd')
        ax.zaxis.label.set_color('#8be9fd')
        ax.tick_params(colors='#6272a4', labelsize=8)
        ax.set_xlabel('X (Å)', color='#8be9fd', fontsize=9)
        ax.set_ylabel('Y (Å)', color='#8be9fd', fontsize=9)
        ax.set_zlabel('Z (Å)', color='#8be9fd', fontsize=9)
        
        # Grid for reference
        ax.grid(True, color='#44475a', alpha=0.2, linestyle=':')
    else:
        # Empty state - still show axes
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.xaxis.label.set_color('#8be9fd')
        ax.yaxis.label.set_color('#8be9fd')
        ax.zaxis.label.set_color('#8be9fd')
        ax.tick_params(colors='#6272a4', labelsize=8)
        ax.set_xlabel('X (Å)', color='#8be9fd', fontsize=9)
        ax.set_ylabel('Y (Å)', color='#8be9fd', fontsize=9)
        ax.set_zlabel('Z (Å)', color='#8be9fd', fontsize=9)
        ax.grid(True, color='#44475a', alpha=0.2, linestyle=':')
    
    # Set rotation (elevation and azimuth)
    ax.view_init(elev=elev, azim=angle)
    
    # 3. Save to Buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)  # Important: free memory
    
    return Response(content=buf.getvalue(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    # Use import string for reload to work properly
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)

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
import uuid
from typing import Dict, Any

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

# Mount current directory for static files including SVG, CSS, etc.
app.mount("/static", StaticFiles(directory=str(BACKEND_DIR)), name="visual_tools_static")

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

# --- Generated Structures Storage ---
generated_structures = {}  # job_id -> {"vasp_content": str, "metadata": dict}

# --- Helper Functions ---
def get_lattice_vectors(a_mult, b_mult, gamma_deg):
    """Calculate 2D lattice vectors."""
    gamma_rad = np.radians(gamma_deg)
    v1 = np.array([a_mult, 0.0])
    v2 = np.array([b_mult * np.cos(gamma_rad), b_mult * np.sin(gamma_rad)])
    return v1, v2

def frac_to_cart(u, v, v1, v2):
    return u * v1 + v * v2

# --- Structure Generation Helpers ---
def convert_layer_stack_to_sequence(layer_stack: list, layer_sequence: str = "") -> str:
    """Convert layer stack array to layer sequence string."""
    if layer_sequence:
        return layer_sequence

    # If no custom sequence, use the stack order
    return "-".join(layer_stack)

def build_ion_arrays_from_mapping(layers: dict, layer_stack: list, ion_mapping: dict) -> tuple:
    """Build A, B, X ion arrays from per-layer mapping."""
    A_ions = []
    B_ions = []
    X_ions = []
    spacer_dict = {}

    for layer_name in layer_stack:
        if layer_name not in layers or layer_name not in ion_mapping:
            continue

        layer_mapping = ion_mapping[layer_name]

        # Get ion counts for this layer
        a_count = sum(1 for atom in layers[layer_name] if atom[0] == 'A')
        b_count = sum(1 for atom in layers[layer_name] if atom[0] == 'B')
        x_count = sum(1 for atom in layers[layer_name] if atom[0] == 'X')

        # Build ion arrays
        if layer_mapping.get('A'):
            A_ions.extend([layer_mapping['A']] * a_count)
        if layer_mapping.get('B'):
            B_ions.extend([layer_mapping['B']] * b_count)
        if layer_mapping.get('X'):
            X_ions.extend([layer_mapping['X']] * x_count)

        # Handle spacers
        for atom in layers[layer_name]:
            code = atom[0]
            if code.startswith('S') and code in layer_mapping:
                if code not in spacer_dict:
                    spacer_dict[code] = []
                spacer_dict[code].append(layer_mapping[code])

    return A_ions, B_ions, X_ions, spacer_dict

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

@app.get("/layer_arquitect.html")
async def serve_layer_architect():
    """Serve the Layer Architect module HTML file."""
    layer_architect_file = BACKEND_DIR / "layer_arquitect.html"
    if layer_architect_file.exists():
        return FileResponse(layer_architect_file)
    else:
        return JSONResponse(
            status_code=404,
            content={"detail": "Layer Architect module not found"}
        )

@app.get("/structure_generator.html")
async def serve_structure_generator():
    """Serve the Structure Generator module HTML file."""
    structure_generator_file = BACKEND_DIR / "structure_generator.html"
    if structure_generator_file.exists():
        return FileResponse(structure_generator_file)
    else:
        return JSONResponse(
            status_code=404,
            content={"detail": "Structure Generator module not found"}
        )

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
    """Add current layer to stack (allows duplicates)."""
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

@app.get("/structure/3d/data")
def get_3d_data(spacing: float = 3.0):
    """Get 3D structure data as JSON for Three.js visualization."""
    # Always return lattice data even if stack is empty, so we can draw the grid base
    v1, v2 = get_lattice_vectors(
        state["lattice"]["a"],
        state["lattice"]["b"],
        state["lattice"]["gamma"]
    )

    # Calculate unit cell corners (Z=0)
    # c0 = origin, c1 = a, c2 = a+b, c3 = b
    corners = [
        {"x": 0.0, "y": 0.0},
        {"x": float(v1[0]), "y": float(v1[1])},
        {"x": float(v1[0] + v2[0]), "y": float(v1[1] + v2[1])},
        {"x": float(v2[0]), "y": float(v2[1])}
    ]
    
    response = {
        "atoms": [], 
        "arrows": [], 
        "grid": {
            "corners": corners,
            "total_height": max(len(state["layer_stack"]) - 1, 0) * spacing
        }
    }

    if not state["layer_stack"]:
        return response
    
    # Color map
    color_map = {"A": "#50fa7b", "B": "#8be9fd", "X": "#ff5555"}
    spacer_color = "#ffb86c"
    
    atoms = []
    spacer_positions = {}  # {spacer_code: {layer_idx: (x, y, z)}}
    
    # Build atom list and collect spacer positions
    for i, layer_name in enumerate(state["layer_stack"]):
        z = i * spacing
        if layer_name in state["layers"]:
            for atom in state["layers"][layer_name]:
                code, u, v = atom[:3]
                xy = frac_to_cart(u, v, v1, v2)
                x, y = float(xy[0]), float(xy[1])
                
                # Determine color and size
                if code.startswith('S'):
                    color = spacer_color
                    size = 0.35 
                    if code not in spacer_positions:
                        spacer_positions[code] = {}
                    spacer_positions[code][i] = (x, y, z)
                else:
                    color = color_map.get(code, "#6272a4")
                    if code == "A":
                        size = 0.5
                    elif code == "B":
                        size = 0.45
                    elif code == "X":
                        size = 0.35
                    else:
                        size = 0.3
                
                atoms.append({
                    "x": x,
                    "y": y,
                    "z": z,
                    "code": code,
                    "color": color,
                    "size": size
                })
    
    # Build arrows connecting spacers between consecutive layers
    arrows = []
    for spacer_code, layer_positions in spacer_positions.items():
        sorted_layers = sorted(layer_positions.keys())
        for idx in range(len(sorted_layers) - 1):
            layer_i = sorted_layers[idx]
            layer_j = sorted_layers[idx + 1]
            
            if layer_j == layer_i + 1:  # Only consecutive layers
                x1, y1, z1 = layer_positions[layer_i]
                x2, y2, z2 = layer_positions[layer_j]
                
                arrows.append({
                    "from": {"x": x1, "y": y1, "z": z1},
                    "to": {"x": x2, "y": y2, "z": z2},
                    "color": spacer_color
                })
    
    response["atoms"] = atoms
    response["arrows"] = arrows
    
    return response

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

# --- Structure Generation Endpoints ---

class StructureGenerateRequest(BaseModel):
    layers: Dict[str, list]
    layer_stack: list
    lattice: Dict[str, float]
    ion_mapping: Dict[str, Dict[str, str]]
    config: Dict[str, Any]

@app.post("/structure/generate")
def generate_structure(request: StructureGenerateRequest):
    """Generate a structure from layer definitions and ion mapping."""
    try:
        # Convert layer stack to sequence
        layer_sequence = convert_layer_stack_to_sequence(
            request.layer_stack,
            request.config.get('layer_sequence', '')
        )

        # Build ion arrays from per-layer mapping
        A_ions, B_ions, X_ions, spacer_dict = build_ion_arrays_from_mapping(
            request.layers, request.layer_stack, request.ion_mapping
        )

        # Prepare sharp_spacer list
        sharp_spacer = []
        for spacer_list in spacer_dict.values():
            sharp_spacer.extend(spacer_list)

        # Import q2D_creator (lazy import to avoid circular dependencies)
        from q2D_Materials.core.creator import q2D_creator

        # Create the structure
        creator = q2D_creator()
        structure = creator.create_structure(
            A_ions=A_ions if A_ions else None,
            B_ions=B_ions if B_ions else None,
            X_ions=X_ions if X_ions else None,
            xy_expansion=tuple(request.config.get('xy_expansion', [1, 1])),
            template=request.config.get('template', 'cubic'),
            structure_type=request.config.get('structure_type', 'bulk'),
            vacuum=request.config.get('vacuum', 15.0),
            layer_sequence=layer_sequence,
            sharp_spacer=sharp_spacer if sharp_spacer else None,
            penetration=request.config.get('penetration', 0.0),
            attachment_end=request.config.get('attachment_end') or None,
            optimizer=request.config.get('optimizer', 'KS'),
            thickness=request.config.get('thickness', 1),
            glazer_angles=request.config.get('glazer_angles'),
            glazer_pattern=request.config.get('glazer_pattern'),
            lattice_multipliers=[request.lattice['a'], request.lattice['b']]
        )

        # Convert to VASP string
        from io import StringIO
        vasp_buffer = StringIO()
        structure.write(vasp_buffer, format='vasp', sort=True)
        vasp_content = vasp_buffer.getvalue()

        # Generate job ID and store
        job_id = str(uuid.uuid4())
        generated_structures[job_id] = {
            "vasp_content": vasp_content,
            "metadata": {
                "atom_count": len(structure),
                "layer_count": len(request.layer_stack),
                "template": request.config.get('template', 'cubic'),
                "structure_type": request.config.get('structure_type', 'bulk'),
                "xy_expansion": request.config.get('xy_expansion', [1, 1]),
                "timestamp": str(np.datetime64('now'))
            }
        }

        return {
            "status": "success",
            "job_id": job_id,
            "atom_count": len(structure),
            "message": f"Structure generated with {len(structure)} atoms"
        }

    except Exception as e:
        import traceback
        print(f"Structure generation failed: {e}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.get("/structure/download/{job_id}")
def download_structure(job_id: str):
    """Download generated structure as VASP file."""
    if job_id not in generated_structures:
        return JSONResponse(
            status_code=404,
            content={"detail": "Structure not found"}
        )

    structure_data = generated_structures[job_id]
    vasp_content = structure_data["vasp_content"]

    return Response(
        content=vasp_content,
        media_type="chemical/x-vasp",
        headers={
            "Content-Disposition": f"attachment; filename=structure_{job_id}.vasp"
        }
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
        
        # Draw arrows connecting spacers between layers (S1->S1, S2->S2, etc.)
        # Collect spacer positions by layer and spacer code
        spacer_positions = {}  # {spacer_code: {layer_idx: (x, y, z)}}
        
        for i, layer_name in enumerate(state["layer_stack"]):
            z = i * spacing
            if layer_name in state["layers"]:
                for atom in state["layers"][layer_name]:
                    code, u, v = atom[:3]
                    if code.startswith('S'):  # It's a spacer
                        xy = frac_to_cart(u, v, v1, v2)
                        if code not in spacer_positions:
                            spacer_positions[code] = {}
                        spacer_positions[code][i] = (float(xy[0]), float(xy[1]), float(z))
        
        # Draw arrows for each spacer code connecting consecutive layers
        from mpl_toolkits.mplot3d.proj3d import proj_transform
        from matplotlib.patches import FancyArrowPatch
        
        class Arrow3D(FancyArrowPatch):
            def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
                super().__init__((0, 0), (0, 0), *args, **kwargs)
                self._xyz = (x, y, z)
                self._dxdydz = (dx, dy, dz)
            
            def draw(self, renderer):
                x1, y1, z1 = self._xyz
                dx, dy, dz = self._dxdydz
                x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
                
                xs, ys, zs = proj_transform([x1, x2], [y1, y2], [z1, z2], self.axes.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                super().draw(renderer)
            
            def do_3d_projection(self, renderer=None):
                x1, y1, z1 = self._xyz
                dx, dy, dz = self._dxdydz
                x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
                
                xs, ys, zs = proj_transform([x1, x2], [y1, y2], [z1, z2], self.axes.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                return min(zs) if zs else 0
        
        # Draw arrows for each spacer code
        for spacer_code, layer_positions in spacer_positions.items():
            # Sort by layer index
            sorted_layers = sorted(layer_positions.keys())
            for idx in range(len(sorted_layers) - 1):
                layer_i = sorted_layers[idx]
                layer_j = sorted_layers[idx + 1]
                
                x1, y1, z1 = layer_positions[layer_i]
                x2, y2, z2 = layer_positions[layer_j]
                
                # Only draw if they're consecutive layers (not skipping layers)
                if layer_j == layer_i + 1:
                    dx = x2 - x1
                    dy = y2 - y1
                    dz = z2 - z1
                    
                    arrow = Arrow3D(
                        x1, y1, z1, dx, dy, dz,
                        mutation_scale=15, arrowstyle='->', 
                        color='#ffb86c', alpha=0.7, linewidth=2
                    )
                    ax.add_artist(arrow)
        
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
    import sys
    import platform
    
    # Use 127.0.0.1 for Windows compatibility (browsers can't access 0.0.0.0)
    # On Linux, 0.0.0.0 works but localhost/127.0.0.1 is still preferred for browser access
    host = "127.0.0.1"
    port = 8000
    
    print(f"\n{'='*60}")
    print(f"Layer Architect Server Starting...")
    print(f"{'='*60}")
    print(f"Platform: {platform.system()}")
    print(f"Access the application at: http://{host}:{port}/")
    print(f"{'='*60}\n")
    
    # Use import string for reload to work properly
    uvicorn.run("backend:app", host=host, port=port, reload=True)

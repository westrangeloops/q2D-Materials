import numpy as np
from ase.io import read
import plotly.graph_objects as go
import plotly.express as px

def create_plane_from_vectors(p1, p2, z_axis):
    """
    Calculates a plane defined by two points and a Z-axis vector.

    Args:
        p1 (list or numpy.array): The first 3D point [x, y, z].
        p2 (list or numpy.array): The second 3D point [x, y, z].
        z_axis (list or numpy.array): The Z-axis vector [x, y, z].

    Returns:
        tuple: A tuple containing:
            - numpy.array: A point on the plane (p1).
            - numpy.array: The normal vector of the plane.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    z_axis = np.array(z_axis)

    # 1. Calculate the vector between the couple (p2 - p1)
    couple_vector = p2 - p1

    # 2. Calculate the vectorial product of the couple_vector and the z_axis vector
    result_vector = np.cross(couple_vector, z_axis)

    # 3. Generate the plane from the two vectors
    plane_vector = np.cross(result_vector, couple_vector)

    return plane_vector

#  create a vector from the point on the plane to the point on the plane + 100 in the direction of the normal vector
# Read the structure from the vasp file
structure = read('/home/dotempo/Documents/REPO/SVC-Materials/tests/n2_analysis/n2_salt.vasp')

# Get the positions of the atoms
positions = structure.get_positions()

# Get the cell
cell = structure.get_cell()

# Get all Cl atoms and sort by z-coordinate
cl_atoms = structure[structure.get_atomic_numbers() == 17]
cl_atoms_sorted = cl_atoms[cl_atoms.positions[:, 2].argsort()]

print(f"Found {len(cl_atoms_sorted)} Cl atoms")
for i, atom in enumerate(cl_atoms_sorted):
    print(f"Cl{i+1}: {atom.position} (z = {atom.position[2]:.3f})")

# Get the z axis vector from the cell box
z_axis = cell[2]

# Create two planes:
# Plane 1: Using the two lowest Cl atoms
cl_low_1, cl_low_2 = cl_atoms_sorted[0], cl_atoms_sorted[1]
vector_low = cl_low_2.position - cl_low_1.position
plane_vector_low = create_plane_from_vectors(cl_low_1.position, cl_low_2.position, z_axis)

# Plane 2: Using the two highest Cl atoms  
cl_high_1, cl_high_2 = cl_atoms_sorted[-2], cl_atoms_sorted[-1]
vector_high = cl_high_2.position - cl_high_1.position
plane_vector_high = create_plane_from_vectors(cl_high_1.position, cl_high_2.position, z_axis)

# Normalize the plane vectors (normal vectors)
normal_vector_low = plane_vector_low / np.linalg.norm(plane_vector_low)
normal_vector_high = plane_vector_high / np.linalg.norm(plane_vector_high)

# Create Plotly figure
fig = go.Figure()

# Color mapping for atoms
colors = {'Cl': 'green', 'Na': 'purple', 'other': 'gray'}
symbols = structure.get_chemical_symbols()

# Group atoms by type for better legend handling
unique_symbols = list(set(symbols))
for symbol in unique_symbols:
    mask = np.array(symbols) == symbol
    atom_positions = positions[mask]
    color = colors.get(symbol, colors['other'])
    
    fig.add_trace(go.Scatter3d(
        x=atom_positions[:, 0],
        y=atom_positions[:, 1],
        z=atom_positions[:, 2],
        mode='markers',
        marker=dict(size=8, color=color, opacity=0.8),
        name=f'{symbol} atoms',
        hovertemplate=f'{symbol}<br>X: %{{x:.2f}} Å<br>Y: %{{y:.2f}} Å<br>Z: %{{z:.2f}} Å<extra></extra>'
    ))

# Highlight the Cl atoms used for both planes
# Low plane Cl atoms
fig.add_trace(go.Scatter3d(
    x=[cl_low_1.position[0]],
    y=[cl_low_1.position[1]],
    z=[cl_low_1.position[2]],
    mode='markers',
    marker=dict(size=15, color='red', symbol='circle'),
    name='Cl1 (low plane)',
    hovertemplate='Cl1 (low plane)<br>X: %{x:.2f} Å<br>Y: %{y:.2f} Å<br>Z: %{z:.2f} Å<extra></extra>'
))

fig.add_trace(go.Scatter3d(
    x=[cl_low_2.position[0]],
    y=[cl_low_2.position[1]],
    z=[cl_low_2.position[2]],
    mode='markers',
    marker=dict(size=15, color='darkred', symbol='circle'),
    name='Cl2 (low plane)',
    hovertemplate='Cl2 (low plane)<br>X: %{x:.2f} Å<br>Y: %{y:.2f} Å<br>Z: %{z:.2f} Å<extra></extra>'
))

# High plane Cl atoms
fig.add_trace(go.Scatter3d(
    x=[cl_high_1.position[0]],
    y=[cl_high_1.position[1]],
    z=[cl_high_1.position[2]],
    mode='markers',
    marker=dict(size=15, color='blue', symbol='circle'),
    name='Cl3 (high plane)',
    hovertemplate='Cl3 (high plane)<br>X: %{x:.2f} Å<br>Y: %{y:.2f} Å<br>Z: %{z:.2f} Å<extra></extra>'
))

fig.add_trace(go.Scatter3d(
    x=[cl_high_2.position[0]],
    y=[cl_high_2.position[1]],
    z=[cl_high_2.position[2]],
    mode='markers',
    marker=dict(size=15, color='darkblue', symbol='circle'),
    name='Cl4 (high plane)',
    hovertemplate='Cl4 (high plane)<br>X: %{x:.2f} Å<br>Y: %{y:.2f} Å<br>Z: %{z:.2f} Å<extra></extra>'
))

# Draw the vectors between the Cl atoms for both planes
fig.add_trace(go.Scatter3d(
    x=[cl_low_1.position[0], cl_low_2.position[0]],
    y=[cl_low_1.position[1], cl_low_2.position[1]],
    z=[cl_low_1.position[2], cl_low_2.position[2]],
    mode='lines',
    line=dict(color='red', width=8),
    name='Low Cl-Cl vector',
    hovertemplate='Low Cl-Cl vector<extra></extra>'
))

fig.add_trace(go.Scatter3d(
    x=[cl_high_1.position[0], cl_high_2.position[0]],
    y=[cl_high_1.position[1], cl_high_2.position[1]],
    z=[cl_high_1.position[2], cl_high_2.position[2]],
    mode='lines',
    line=dict(color='blue', width=8),
    name='High Cl-Cl vector',
    hovertemplate='High Cl-Cl vector<extra></extra>'
))

# Create and plot both planes
# Determine the extent of the structure to size the planes appropriately
min_coords = np.min(positions, axis=0)
max_coords = np.max(positions, axis=0)
plane_size = np.max(max_coords - min_coords) * 0.8

# ===== LOW PLANE =====
# Use the midpoint between the two low Cl atoms as the plane center
plane_center_low = (cl_low_1.position + cl_low_2.position) / 2

# Create two orthogonal vectors in the low plane
v1_low = cl_low_2.position - cl_low_1.position
v1_low = v1_low / np.linalg.norm(v1_low)
v2_low = np.cross(normal_vector_low, v1_low)
v2_low = v2_low / np.linalg.norm(v2_low)

# Create a mesh for the low plane
u = np.linspace(-plane_size/2, plane_size/2, 20)
v = np.linspace(-plane_size/2, plane_size/2, 20)
U, V = np.meshgrid(u, v)

# Calculate low plane points
plane_points_low = plane_center_low[:, np.newaxis, np.newaxis] + U[np.newaxis, :, :] * v1_low[:, np.newaxis, np.newaxis] + V[np.newaxis, :, :] * v2_low[:, np.newaxis, np.newaxis]

# Plot the low plane as a surface
fig.add_trace(go.Surface(
    x=plane_points_low[0],
    y=plane_points_low[1],
    z=plane_points_low[2],
    opacity=0.3,
    colorscale=[[0, 'yellow'], [1, 'yellow']],
    showscale=False,
    name='Low Plane',
    hovertemplate='Low Plane<br>X: %{x:.2f} Å<br>Y: %{y:.2f} Å<br>Z: %{z:.2f} Å<extra></extra>'
))

# ===== HIGH PLANE =====
# Use the midpoint between the two high Cl atoms as the plane center
plane_center_high = (cl_high_1.position + cl_high_2.position) / 2

# Create two orthogonal vectors in the high plane
v1_high = cl_high_2.position - cl_high_1.position
v1_high = v1_high / np.linalg.norm(v1_high)
v2_high = np.cross(normal_vector_high, v1_high)
v2_high = v2_high / np.linalg.norm(v2_high)

# Calculate high plane points
plane_points_high = plane_center_high[:, np.newaxis, np.newaxis] + U[np.newaxis, :, :] * v1_high[:, np.newaxis, np.newaxis] + V[np.newaxis, :, :] * v2_high[:, np.newaxis, np.newaxis]

# Plot the high plane as a surface
fig.add_trace(go.Surface(
    x=plane_points_high[0],
    y=plane_points_high[1],
    z=plane_points_high[2],
    opacity=0.3,
    colorscale=[[0, 'lightblue'], [1, 'lightblue']],
    showscale=False,
    name='High Plane',
    hovertemplate='High Plane<br>X: %{x:.2f} Å<br>Y: %{y:.2f} Å<br>Z: %{z:.2f} Å<extra></extra>'
))

# Draw the normal vectors from both plane centers
normal_scale = plane_size * 0.3

# Low plane normal vector
normal_end_low = plane_center_low + normal_vector_low * normal_scale
fig.add_trace(go.Scatter3d(
    x=[plane_center_low[0], normal_end_low[0]],
    y=[plane_center_low[1], normal_end_low[1]],
    z=[plane_center_low[2], normal_end_low[2]],
    mode='lines+markers',
    line=dict(color='orange', width=6),
    marker=dict(size=[8, 12], color='orange', symbol=['circle', 'diamond']),
    name='Low Normal vector',
    hovertemplate='Low Normal vector<extra></extra>'
))

# High plane normal vector
normal_end_high = plane_center_high + normal_vector_high * normal_scale
fig.add_trace(go.Scatter3d(
    x=[plane_center_high[0], normal_end_high[0]],
    y=[plane_center_high[1], normal_end_high[1]],
    z=[plane_center_high[2], normal_end_high[2]],
    mode='lines+markers',
    line=dict(color='purple', width=6),
    marker=dict(size=[8, 12], color='purple', symbol=['circle', 'diamond']),
    name='High Normal vector',
    hovertemplate='High Normal vector<extra></extra>'
))

# Draw the z-axis vector for reference (from origin)
z_axis_normalized = z_axis / np.linalg.norm(z_axis)
origin = np.mean(positions, axis=0)  # Use center of structure as origin
z_axis_end = origin + z_axis_normalized * normal_scale * 0.8

fig.add_trace(go.Scatter3d(
    x=[origin[0], z_axis_end[0]],
    y=[origin[1], z_axis_end[1]],
    z=[origin[2], z_axis_end[2]],
    mode='lines+markers',
    line=dict(color='cyan', width=4),
    marker=dict(size=[6, 10], color='cyan', symbol=['circle', 'diamond']),
    name='Z-axis',
    hovertemplate='Z-axis vector<extra></extra>'
))

# Update layout for better visualization
fig.update_layout(
    title='Interactive Crystal Structure with Two Planes Defined by Cl Atoms',
    scene=dict(
        xaxis_title='X (Å)',
        yaxis_title='Y (Å)',
        zaxis_title='Z (Å)',
        aspectmode='cube',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    width=1200,
    height=800,
    showlegend=True
)

# Save as HTML file and show
fig.write_html('interactive_structure.html')
fig.show()

# Print detailed information about both planes
print("\n" + "="*60)
print("PLANE ANALYSIS RESULTS")
print("="*60)

print(f"\nLOW PLANE (Yellow):")
print(f"  Normal vector: {normal_vector_low}")
print(f"  Passes through points:")
print(f"    Cl1: {cl_low_1.position}")
print(f"    Cl2: {cl_low_2.position}")
print(f"  Plane center: {plane_center_low}")
print(f"  Distance between Cl atoms: {np.linalg.norm(vector_low):.3f} Å")

print(f"\nHIGH PLANE (Light Blue):")
print(f"  Normal vector: {normal_vector_high}")
print(f"  Passes through points:")
print(f"    Cl3: {cl_high_1.position}")
print(f"    Cl4: {cl_high_2.position}")
print(f"  Plane center: {plane_center_high}")
print(f"  Distance between Cl atoms: {np.linalg.norm(vector_high):.3f} Å")

# Calculate angle between the two planes
dot_product = np.dot(normal_vector_low, normal_vector_high)
angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
angle_deg = np.degrees(angle_rad)
print(f"\nANGLE BETWEEN PLANES: {angle_deg:.2f}°")

# Calculate the angle between the planes vs z-axis
angle_z_axis_low = np.arccos(np.dot(z_axis_normalized, normal_vector_low))
angle_z_axis_deg_low = np.degrees(angle_z_axis_low)
angle_z_axis_high = np.arccos(np.dot(z_axis_normalized, normal_vector_high))
angle_z_axis_deg_high = np.degrees(angle_z_axis_high)
print(f"ANGLE BETWEEN LOW PLANE VS Z-AXIS: {angle_z_axis_deg_low:.2f}°")
print(f"ANGLE BETWEEN HIGH PLANE VS Z-AXIS: {angle_z_axis_deg_high:.2f}°") 

# Calculate distance between plane centers
plane_separation = np.linalg.norm(plane_center_high - plane_center_low)
print(f"DISTANCE BETWEEN PLANE CENTERS: {plane_separation:.3f} Å")

print("="*60)



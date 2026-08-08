# This example is used to write structures to images in an automated manner.
# It is a very specific script! Not intended for general use.
# Uses Crystal Toolkit to render a q2D-Materials structure and save a PNG screenshot.
from __future__ import annotations

import base64
from pathlib import Path
from time import sleep

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

from pymatgen.io.ase import AseAtomsAdaptor

import crystal_toolkit.components as ctc
from crystal_toolkit.settings import SETTINGS

# Add project root for q2D_Materials
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from q2D_Materials.core.creator import q2D_creator

SCREENSHOT_PATH = Path(__file__).resolve().parent / "images"
SCREENSHOT_PATH.mkdir(exist_ok=True)

# Build a monolayer with q2D-Materials and convert to pymatgen
q2d = q2D_creator()
atoms = q2d.create_structure(
    structure_type="monolayer",
    A_ions="MA",
    B_ions="Pb",
    X_ions="I",
    xy_expansion=(1, 1),
    template="cubic",
    vacuum=15.0,
    thickness=1,
    layer_sequence=["L1", "L2"],
)
adaptor = AseAtomsAdaptor()
structure = adaptor.get_structure(atoms)

app = dash.Dash(assets_folder=SETTINGS.ASSETS_PATH)
server = app.server

structure_component = ctc.StructureMoleculeComponent(
    structure,
    id="q2d_structure",
    show_compass=False,
    bonded_sites_outside_unit_cell=True,
    scene_settings={"zoomToFit2D": True},
)

layout = html.Div(
    [
        structure_component.layout(),
        dcc.Location(id="url"),
        html.Div(id="dummy-output"),
    ]
)


@app.callback(
    Output(structure_component.id("scene"), "imageRequest"),
    Input(structure_component.id("graph"), "data"),
)
def trigger_image_request(data):
    sleep(1)
    return {"filetype": "png"}


@app.callback(
    Output("dummy-output", "children"),
    Input(structure_component.id("scene"), "imageDataTimestamp"),
    State(structure_component.id("scene"), "imageData"),
)
def save_image(image_data_timestamp, image_data):
    if image_data:
        # image_data is "data:image/png;base64,..."
        if "," in image_data:
            b64 = image_data.split(",", 1)[1]
        else:
            b64 = image_data
        image_bytes = base64.b64decode(b64)
        out_path = SCREENSHOT_PATH / "mono-1layer_crystaltoolkit.png"
        out_path.write_bytes(image_bytes)
        return html.Div(f"Saved to {out_path}", style={"padding": "1rem"})


ctc.register_crystal_toolkit(app=app, layout=layout)

if __name__ == "__main__":
    print("Crystal Toolkit: open http://127.0.0.1:8050 (screenshot saves automatically)")
    app.run(debug=True, port=8050)

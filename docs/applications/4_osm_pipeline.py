"""OpenStreetMap → Sionna RT → DeepMIMO: No-Blender Urban Pipeline."""
# %% [markdown]
# # OpenStreetMap → Sionna RT → DeepMIMO: No-Blender Urban Pipeline
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepMIMO/DeepMIMO/blob/main/docs/applications/4_osm_pipeline.py)
# &nbsp;
# [![GitHub](https://img.shields.io/badge/Open_on-GitHub-181717?logo=github&style=for-the-badge)](https://github.com/DeepMIMO/DeepMIMO/blob/main/docs/applications/4_osm_pipeline.py)
#
# ---
#
# **What this notebook covers:**
# 1. Define a GPS bounding box around any urban area
# 2. Download building footprints from **OpenStreetMap** — no Blender required
# 3. Extrude footprints to 3D and write a **Mitsuba scene** directly
# 4. Run **Sionna RT 2.0** ray tracing on that scene
# 5. Export and convert the results to a **DeepMIMO** dataset
# 6. Load, inspect, and visualise the channel data
#
# **Why no Blender?**
# The traditional pipeline uses Blender + the blosm addon to convert OSM data
# into a Mitsuba XML scene.  This notebook skips Blender entirely:
# `deepmimo.pipelines.osm_to_mitsuba.generate_scene` queries the Overpass API
# directly, extrudes building polygons with NumPy, and writes a valid Mitsuba
# scene file that Sionna RT can load immediately.
#
# **Requirements:**
# ```bash
# pip install 'deepmimo[sionna]'
# ```
#
# ---

# %%
# %pip install 'deepmimo[sionna]'  # uncomment if not installed

# %% [markdown]
# ## Imports

# %%
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sionna.rt as sionna_rt
from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter

import deepmimo as dm
from deepmimo.exporters.sionna_exporter import sionna_exporter
from deepmimo.pipelines.osm_to_mitsuba import generate_scene
from deepmimo.pipelines.txrx_placement import gen_plane_grid
from deepmimo.pipelines.utils.geo_utils import convert_GpsBBox2CartesianBBox
from deepmimo.pipelines.utils.pipeline_utils import get_origin_coords

# %% [markdown]
# ## Configuration
#
# Pick any GPS bounding box.  A block of ~300-600 m on a side works well -
# large enough to capture reflections, small enough to run quickly.
#
# The example below covers part of **Midtown Manhattan**, New York City.

# %%
# GPS bounding box of the area to simulate
BBOX = {
    "minlat": 40.7520,
    "minlon": -73.9820,
    "maxlat": 40.7560,
    "maxlon": -73.9770,
}

# Carrier frequency
CARRIER_FREQ = 3.5e9   # 3.5 GHz

# Sionna RT ray-tracing settings
MAX_DEPTH    = 3       # maximum reflection/refraction bounces
N_SAMPLES    = 2_000_000  # Monte-Carlo rays per transmitter

# Transmitter (base station) position in local Cartesian metres
# (0, 0) is the centre of the bounding box; z is height above ground
TX_POS = np.array([[0.0, 0.0, 25.0]])   # single BS at 25 m height

# UE grid settings
UE_HEIGHT    = 1.5    # metres above ground
GRID_SPACING = 20.0   # metres between UE positions

# %% [markdown]
# ## Step 1 — Generate the Mitsuba Scene from OSM
#
# `generate_scene` queries the Overpass API for building footprints, extrudes
# them into 3D PLY meshes, and writes a `scene.xml` that Sionna can load.
# An `osm_gps_origin.txt` file records the local coordinate origin so that
# GPS coordinates can be recovered later.

# %%
scene_folder = tempfile.mkdtemp()   # use any writable directory

osm_scene_folder = generate_scene(
    minlat=BBOX["minlat"],
    minlon=BBOX["minlon"],
    maxlat=BBOX["maxlat"],
    maxlon=BBOX["maxlon"],
    scene_folder=str(Path(scene_folder) / "osm_scene"),
    verbose=True,
)

print(f"\nScene folder: {osm_scene_folder}")
print("Files created:")
for f in sorted(Path(osm_scene_folder).rglob("*")):
    if f.is_file():
        size_kb = f.stat().st_size / 1024
        print(f"  {f.relative_to(osm_scene_folder)!s:<45}  {size_kb:7.1f} KB")

# %% [markdown]
# ## Step 2 — Load the Scene in Sionna RT

# %%
xml_path = str(Path(osm_scene_folder) / "scene.xml")
scene = sionna_rt.load_scene(xml_path, merge_shapes=False)
scene.frequency = CARRIER_FREQ

print(f"Scene loaded: {len(scene.objects)} objects")
print(f"Sample materials: {[o.radio_material.name for o in list(scene.objects.values())[:4]]}")

# Single isotropic antenna for both TX and RX
single_ant = PlanarArray(
    num_rows=1, num_cols=1,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="iso", polarization="V",
)
scene.tx_array = single_ant
scene.rx_array = single_ant

# %% [markdown]
# ## Step 3 — Place Transmitter and Receivers
#
# The base station is placed at the centre of the area at rooftop height.
# UEs are placed on a regular grid at street level.

# %%
# Read the origin saved by generate_scene
origin_lat, origin_lon = get_origin_coords(osm_scene_folder)
print(f"Local coordinate origin: ({origin_lat:.6f}, {origin_lon:.6f})")

# Bounding box in local Cartesian
xmin, ymin, xmax, ymax = convert_GpsBBox2CartesianBBox(
    BBOX["minlat"], BBOX["minlon"], BBOX["maxlat"], BBOX["maxlon"],
    origin_lat, origin_lon,
)
print(f"Scene extent: X=[{xmin:.0f}, {xmax:.0f}] m,  Y=[{ymin:.0f}, {ymax:.0f}] m")

# Add transmitter
scene.add(Transmitter("tx_0", position=TX_POS[0].tolist()))
print(f"TX at {TX_POS[0]}")

# Generate UE grid and add receivers
rx_pos = gen_plane_grid(xmin + 10, xmax - 10, ymin + 10, ymax - 10,
                        GRID_SPACING, UE_HEIGHT)
print(f"UE grid: {len(rx_pos)} positions")

for i, pos in enumerate(rx_pos):
    scene.add(Receiver(f"rx_{i}", position=pos.tolist()))
print(f"Added {len(rx_pos)} receivers")

# %% [markdown]
# ## Step 4 — Run Sionna RT Ray Tracing

# %%
RT_PARAMS = {
    "max_depth":           MAX_DEPTH,
    "los":                 True,
    "specular_reflection": True,
    "diffuse_reflection":  False,
    "refraction":          True,
    "samples_per_src":     N_SAMPLES,
}

print("Running ray tracing...")
p_solver = PathSolver()
paths = p_solver(scene=scene, **RT_PARAMS)

print(f"Path delays shape (tau): {paths.tau.shape}")
print("  (num_rx, num_tx, num_paths)")

# %% [markdown]
# ## Step 5 — Export with `sionna_exporter`

# %%
rt_save_folder = str(Path(scene_folder) / "sionna_rt_export")
sionna_exporter(scene, paths, RT_PARAMS, rt_save_folder)

print(f"\nExported files in {rt_save_folder}:")
for f in sorted(Path(rt_save_folder).iterdir()):
    print(f"  {f.name:<40}  {f.stat().st_size / 1024:7.1f} KB")

# %% [markdown]
# ## Step 6 — Convert to DeepMIMO Format

# %%
scenario_name = dm.convert(rt_save_folder, overwrite=True)
print(f"Converted scenario: {scenario_name}")

# %% [markdown]
# ## Step 7 — Load and Inspect the DeepMIMO Dataset

# %%
dataset = dm.load(scenario_name)
print(dataset)

# %% [markdown]
# ## Step 8 — Compute Channels and Visualise

# %%
n_paths_result = max(*dataset.num_paths, 1)
ch_params = dm.ChannelParameters()
ch_params.num_paths = n_paths_result

channels = dataset.compute_channels(ch_params)
print(f"Channel tensor: {channels.shape}  (n_ue, n_rx_ant, n_tx_ant, n_paths)")

# Per-UE received power (sum over paths, dB scale)
pwr_linear = np.sum(np.abs(channels[:, 0, 0, :]) ** 2, axis=-1)
pwr_db = 10 * np.log10(pwr_linear + 1e-30)

# Scatter plot: UE position coloured by received power
fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(
    dataset.rx_pos[:, 0], dataset.rx_pos[:, 1],
    c=pwr_db, cmap="viridis", s=40, vmin=np.percentile(pwr_db, 5),
)
ax.scatter(*TX_POS[0, :2], c="red", marker="^", s=200, zorder=5, label="TX")
plt.colorbar(sc, ax=ax, label="Received power [dB]")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(f"Received power — OSM scene\n({len(rx_pos)} UEs, 3.5 GHz)")
ax.legend()
ax.grid(visible=True, alpha=0.3)
plt.tight_layout()
plt.savefig("osm_pipeline_power_map.png", dpi=100, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# | Step | Tool | Output |
# |------|------|--------|
# | 1. OSM → scene | `generate_scene` | `scene.xml` + PLY meshes |
# | 2. Load scene | `sionna.rt.load_scene` | Sionna `Scene` object |
# | 3. Place antennas | `Transmitter` / `Receiver` | TX + UE grid |
# | 4. Ray trace | `PathSolver` | `Paths` object |
# | 5. Export | `sionna_exporter` | `.pkl` files |
# | 6. Convert | `dm.convert` | DeepMIMO scenario |
# | 7. Load | `dm.load` | `Dataset` object |
# | 8. Channels | `dataset.compute_channels` | Complex channel tensor |
#
# **Key design choices:**
# - **No Blender** — `generate_scene` fetches OSM data via the Overpass API and
#   writes Mitsuba PLY meshes directly in Python.
# - **ITU radio materials** — buildings use `itu-radio-material type=concrete`,
#   matching the material convention of Sionna's built-in scenes.
# - **Local coordinate system** — the centre of the GPS bounding box is the
#   origin; `osm_gps_origin.txt` records this for downstream use.
# - **Reusable scene folder** — the generated `scene.xml` folder can be fed
#   directly into `raytrace_sionna()` for large-scale batch processing.

"""Sionna RT 2.0 → DeepMIMO: Run, Export, Convert."""
# %% [markdown]
# # Sionna RT 2.0 → DeepMIMO: Run, Export, Convert
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepMIMO/DeepMIMO/blob/main/docs/applications/2_sionna_rt_downstream.py)
# &nbsp;
# [![GitHub](https://img.shields.io/badge/Open_on-GitHub-181717?logo=github&style=for-the-badge)](https://github.com/DeepMIMO/DeepMIMO/blob/main/docs/applications/2_sionna_rt_downstream.py)
#
# ---
#
# **What this notebook covers:**
# 1. Load Sionna RT 2.0's built-in Munich scene and visualize it
# 2. Run specular-reflection ray tracing with timing
# 3. Inspect propagation paths directly from Sionna
# 4. Export results to disk with DeepMIMO's `sionna_exporter`
# 5. Convert to a DeepMIMO scenario with `dm.convert`
# 6. Explore channels: delay profiles and ray visualization
#
# **Why this workflow?**
# Sionna RT does not natively persist ray tracing results to disk.
# DeepMIMO's exporter serializes all path data (delays, angles, vertices,
# interaction types) so expensive simulations can be reused without re-running
# the ray tracer. The converter maps that data into the standardized DeepMIMO
# format, unlocking the full DeepMIMO toolchain.
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
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sionna.rt as sionna_rt
from sionna.rt import Camera, PathSolver, PlanarArray, Receiver, Transmitter

import deepmimo as dm
from deepmimo.exporters.sionna_exporter import sionna_exporter

# %% [markdown]
# ## Scene Configuration
#
# We use Sionna's built-in **Munich** scene: a realistic urban environment
# with buildings, streets, and varied geometry. The transmitter is placed
# on a rooftop; receivers are placed at street level.
#
# We use `max_depth=2` with specular reflections only — enough to capture the
# dominant LoS and first-order reflected paths without the cost of diffractions
# or refractions.

# %%
CARRIER_FREQ = 3.5e9  # 3.5 GHz

# Ray-tracing parameters forwarded to PathSolver
RT_PARAMS = {
    "max_depth": 2,
    "los": True,
    "specular_reflection": True,
    "diffuse_reflection": False,
    "refraction": False,
    "samples_per_src": 2_000_000,
}

# Transmitter — rooftop position
TX_POS = [-210.0, 73.0, 105.0]

# Receivers — four street-level positions with confirmed propagation paths
RX_POSITIONS = [
    [40.0, 20.0, 1.5],
    [20.0, 50.0, 1.5],
    [30.0, 60.0, 1.5],
    [50.0, 50.0, 1.5],
]

# %% [markdown]
# ## Build the Scene

# %%
scene = sionna_rt.load_scene(sionna_rt.scene.munich)
scene.frequency = CARRIER_FREQ

# Single-element isotropic antenna at TX and RX
single_ant = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V",
)
scene.tx_array = single_ant
scene.rx_array = single_ant

# Add transmitter
scene.add(Transmitter("tx_0", position=TX_POS))

# Add receivers — set display_radius so devices render at a reasonable size
for i, pos in enumerate(RX_POSITIONS):
    rx = Receiver(f"rx_{i}", position=pos)
    rx.display_radius = 8.0  # metres; default heuristic is too large for Munich
    scene.add(rx)

print(f"TX position : {TX_POS}")
print(f"RX count    : {len(RX_POSITIONS)}")

# %% [markdown]
# ## Visualize the Scene
#
# Sionna RT renders the 3D scene with device positions before any ray tracing.
# The top-down view shows the full Munich scene geometry; the oblique view gives
# a sense of building heights relative to the rooftop TX.

# %%
# Top-down view centered between TX and receiver cluster
cam_top = Camera(position=[-80.0, 40.0, 600.0], look_at=[-80.0, 40.0, 0.0])
fig = scene.render(camera=cam_top, show_devices=True)
fig.suptitle("Munich Scene — Top View (TX = red triangle, RX = blue dots)")
plt.show()

# %%
# Oblique perspective from the south-west
cam_oblique = Camera(position=[-500.0, -300.0, 400.0], look_at=[-80.0, 40.0, 50.0])
fig = scene.render(camera=cam_oblique, show_devices=True)
fig.suptitle("Munich Scene — Perspective View")
plt.show()

# %% [markdown]
# ## Run Ray Tracing

# %%
p_solver = PathSolver()

t0 = time.perf_counter()
paths = p_solver(scene=scene, **RT_PARAMS)
elapsed = time.perf_counter() - t0

n_rx = paths.tau.shape[0]
n_tx = paths.tau.shape[1]
max_paths = paths.tau.shape[2]

print(f"Ray tracing completed in {elapsed:.1f} s")
print(f"Receiver count  : {n_rx}")
print(f"Transmitter count: {n_tx}")
print(f"Max paths / pair : {max_paths}")
print(f"Channel coeff (a[0]) shape: {paths.a[0].shape}")

# %% [markdown]
# ## Visualize Propagation Paths
#
# Render the scene with the computed ray paths overlaid.
# Each colored line segment represents a propagation path from TX to RX.

# %%
fig = scene.render(camera=cam_top, paths=paths, show_devices=True)
fig.suptitle("Propagation Paths — Top View")
plt.show()

# %%
# Perspective view of paths
fig = scene.render(camera=cam_oblique, paths=paths, show_devices=True)
fig.suptitle("Propagation Paths — Perspective View")
plt.show()

# %% [markdown]
# ## Power-Delay Profiles (Sionna)
#
# Plot the per-receiver power-delay profile directly from the `Paths` object,
# before any conversion.  Delays are in nanoseconds; power is derived from the
# complex channel coefficient `a`.

# %%
# Complex channel coefficients — shape: (n_rx, n_tx, n_rx_ant, n_tx_ant, max_paths)
a_complex = paths.a[0].numpy() + 1j * paths.a[1].numpy()
tau_np = paths.tau.numpy()  # (n_rx, n_tx, max_paths)

fig, axes = plt.subplots(1, n_rx, figsize=(4 * n_rx, 4), sharey=True)
for rx_idx in range(n_rx):
    ax = axes[rx_idx]
    delays_ns = tau_np[rx_idx, 0, :] * 1e9
    power_lin = np.abs(a_complex[rx_idx, 0, 0, 0, :]) ** 2
    valid = delays_ns > 0
    if valid.any():
        ax.stem(
            delays_ns[valid],
            10 * np.log10(power_lin[valid] + 1e-30),
            basefmt="none",
            markerfmt="C0o",
            linefmt="C0-",
        )
    ax.set_title(f"RX {rx_idx}")
    ax.set_xlabel("Delay (ns)")
    if rx_idx == 0:
        ax.set_ylabel("Power (dBW)")

plt.suptitle("Power-Delay Profiles (from Sionna)", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Export with `sionna_exporter`
#
# `sionna_exporter` serializes paths, materials, RT parameters, and full scene
# geometry (vertices **and** face connectivity) into `.pkl` files.  Face
# connectivity is used by the converter to split merged city meshes into
# individual building components.

# %%
save_folder = str(Path(tempfile.mkdtemp()) / "munich_sionna_rt")

sionna_exporter(scene, paths, RT_PARAMS, save_folder)

print(f"Exported to: {save_folder}")
for f in sorted(Path(save_folder).iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:40s}  {size_kb:7.1f} KB")

# %% [markdown]
# ## Convert to DeepMIMO Format
#
# `dm.convert` auto-detects the Sionna `.pkl` files and routes them through
# the Sionna RT converter, producing a scenario folder in the DeepMIMO
# scenarios directory.

# %%
scenario_name = dm.convert(save_folder, overwrite=True)
print(f"Converted scenario: {scenario_name}")

# %% [markdown]
# ## Load and Inspect the DeepMIMO Dataset

# %%
dataset = dm.load(scenario_name)
print(dataset)

# %% [markdown]
# ## Ray Visualization (DeepMIMO)
#
# DeepMIMO stores the full 3D interaction geometry for each path.
# `dataset.plot_rays` draws TX → bounce points → RX segments, color-coded by
# interaction type (LoS = green, reflected = red), with the reconstructed scene
# overlaid automatically.

# %%
n_ue = len(dataset.rx_pos)
fig, axes = plt.subplots(1, n_ue, figsize=(5 * n_ue, 5), subplot_kw={"projection": "3d"})
for u_idx, ax in enumerate(axes):
    dataset.plot_rays(u_idx, ax=ax)
    ax.set_title(f"UE {u_idx}")
plt.suptitle("Propagation Rays — DeepMIMO", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# The complete **Sionna RT 2.0 → DeepMIMO** pipeline:
#
# | Step | Tool | Output |
# |------|------|--------|
# | 1. Load scene | `sionna_rt.load_scene` | `Scene` object |
# | 2. Visualize | `scene.render` | RGBA renders |
# | 3. Ray trace | `PathSolver` | `Paths` object |
# | 4. Export | `sionna_exporter` | `.pkl` files on disk |
# | 5. Convert | `dm.convert` | DeepMIMO scenario folder |
# | 6. Load | `dm.load` | `Dataset` object |
# | 7. Explore | `plot_rays`, delay profiles | Figures |
#
# From here you can use any DeepMIMO tool: channel generation, beamforming,
# dataset manipulation, Doppler, and ML training pipelines.

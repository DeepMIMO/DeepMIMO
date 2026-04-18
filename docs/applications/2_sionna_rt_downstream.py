"""Sionna RT 2.0 → DeepMIMO: Run, Export, Convert."""
# %% [markdown]
# # Sionna RT 2.0 → DeepMIMO: Run, Export, Convert.
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepMIMO/DeepMIMO/blob/main/docs/applications/2_sionna_rt_downstream.py)
# &nbsp;
# [![GitHub](https://img.shields.io/badge/Open_on-GitHub-181717?logo=github&style=for-the-badge)](https://github.com/DeepMIMO/DeepMIMO/blob/main/docs/applications/2_sionna_rt_downstream.py)
#
# ---
#
# **What this notebook covers:**
# 1. Run Sionna RT 2.0 ray tracing on a built-in scene
# 2. Export results with DeepMIMO's `sionna_exporter`
# 3. Convert to a DeepMIMO scenario with `dm.convert`
# 4. Load, inspect, and generate channels from the resulting dataset
#
# **Why this workflow?**
# Sionna RT does not natively persist ray tracing results to disk.
# DeepMIMO's exporter serialises all path data (delays, angles, vertices,
# interaction types) so that expensive simulations can be reused without
# re-running the ray tracer. The converter then maps that data into the
# standardised DeepMIMO format, unlocking the full DeepMIMO toolchain.
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
from pathlib import Path

import numpy as np
import sionna.rt as sionna_rt
from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter

import deepmimo as dm
from deepmimo.exporters.sionna_exporter import sionna_exporter

# %% [markdown]
# ## Scene Configuration
#
# We use Sionna's built-in **Munich** scene: a realistic urban environment
# with buildings, streets, and varied geometry. The transmitter is placed
# on a rooftop; receivers are placed at street level.

# %%
CARRIER_FREQ = 3.5e9   # 3.5 GHz
MAX_DEPTH    = 3       # Maximum reflection/refraction bounces
N_SAMPLES    = 2_000_000  # Monte-Carlo rays per source

# Transmitter position (rooftop)
TX_POS = [-210.0, 73.0, 105.0]

# Street-level receiver positions (x, y, z)
RX_POSITIONS = [
    [ 40.0,  20.0, 1.5],
    [ 20.0,  10.0, 1.5],
    [ 20.0,  50.0, 1.5],
    [ 10.0,  40.0, 1.5],
    [ 30.0,  60.0, 1.5],
    [-10.0,  30.0, 1.5],
    [  0.0,  60.0, 1.5],
    [ 50.0,  50.0, 1.5],
]

# Ray-tracing parameters forwarded to PathSolver
RT_PARAMS = {
    "max_depth":           MAX_DEPTH,
    "los":                 True,
    "specular_reflection": True,
    "diffuse_reflection":  False,
    "refraction":          True,
    "samples_per_src":     N_SAMPLES,
}

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
print(f"TX at {TX_POS}")

# Add receivers
for i, pos in enumerate(RX_POSITIONS):
    scene.add(Receiver(f"rx_{i}", position=pos))
print(f"Added {len(RX_POSITIONS)} receivers at street level")

# %% [markdown]
# ## Run Ray Tracing

# %%
p_solver = PathSolver()
paths = p_solver(scene=scene, **RT_PARAMS)

n_rx   = paths.tau.shape[0]
n_tx   = paths.tau.shape[1]
n_paths = paths.tau.shape[2]
print(f"Paths found: {n_paths} (across {n_rx} RX x {n_tx} TX)")
print(f"Channel coeff shape (a[0]):  {paths.a[0].shape}")
print(f"Delay shape (tau):           {paths.tau.shape}")
print(f"Interaction shape:           {paths.interactions.shape}")

# %% [markdown]
# ## Export with `sionna_exporter`
#
# `sionna_exporter` serialises paths, materials, RT parameters, and scene
# geometry into `.pkl` files inside a save folder.

# %%
save_folder = str(Path(tempfile.mkdtemp()) / "munich_sionna_rt")

sionna_exporter(scene, paths, RT_PARAMS, save_folder)

print(f"\nExported files in {save_folder}:")
for f in sorted(Path(save_folder).iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:40s}  {size_kb:7.1f} KB")

# %% [markdown]
# ## Convert to DeepMIMO Format
#
# `dm.convert` auto-detects `.pkl` files and routes them through the
# Sionna RT converter, producing a scenario folder in the DeepMIMO
# scenarios directory.

# %%
scenario_name = dm.convert(save_folder, overwrite=True)
print(f"\nConverted scenario: {scenario_name}")

# %% [markdown]
# ## Load and Inspect the DeepMIMO Dataset

# %%
dataset = dm.load(scenario_name)
print(dataset)

# %% [markdown]
# ## Generate Channels

# %%
params = dm.ChannelParameters()
params.num_paths = n_paths if n_paths > 0 else 1
channels = dataset.compute_channels(params)

print(f"\nChannel shape: {channels.shape}")
print("  (n_ue, n_rx_ant, n_tx_ant, n_paths)")

# %% [markdown]
# ## Sanity Check: Per-User Path Count and Power

# %%
n_paths_per_ue = dataset.num_paths
powers_db = 20 * np.log10(np.abs(channels[:, 0, 0, :]) + 1e-30)

print("\nPer-UE statistics:")
print(f"  {'UE':>4}  {'RX pos':>30}  {'#paths':>7}  {'peak power (dB)':>16}")
for i in range(dataset.n_ue):
    rx = dataset.rx_pos[i]
    pos_str = f"({rx[0]:.0f}, {rx[1]:.0f}, {rx[2]:.1f})"
    peak_db = float(np.max(powers_db[i]))
    print(f"  {i:>4}  {pos_str:>30}  {n_paths_per_ue[i]:>7}  {peak_db:>16.1f}")

# %% [markdown]
# ## Summary
#
# The complete **Sionna RT 2.0 → DeepMIMO** pipeline ran successfully:
#
# | Step | Tool | Output |
# |------|------|--------|
# | 1. Ray trace | `sionna.rt.PathSolver` | `sionna.rt.Paths` object |
# | 2. Export | `sionna_exporter` | `.pkl` files on disk |
# | 3. Convert | `dm.convert` | DeepMIMO scenario folder |
# | 4. Load | `dm.load` | `Dataset` object |
# | 5. Channels | `dataset.compute_channels` | Complex channel matrix |
#
# From here you can use any DeepMIMO tool: visualisation, beamforming,
# dataset manipulation, ML training pipelines, and more.

"""# Doppler and Mobility.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/5_doppler_mobility.py)
&nbsp;
[![GitHub](https://img.shields.io/badge/Open_on-GitHub-181717?logo=github&style=for-the-badge)](https://github.com/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/5_doppler_mobility.py)

---

**Tutorial Overview:**
There are three ways to configure Doppler effects. In order of increasing complexity:
1. Set Doppler Directly - Configure Doppler shifts (Hz) manually, per user and path
2. Set Speeds - Define user/object velocities (m/s), which will be converted to Doppler shifts
per user and path depending on the paths that interact with the user/object.
3. Set Timestamps - Configure time evolution between scenes - this computes the velocities of
users/objects across scenes given the timestamps (Note: requires dynamic datasets)

**Related Video:** [Doppler Video](https://youtu.be/xsl6gjTEu2U)

---
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

import deepmimo as dm

# %%
# Load dataset
scen_name = "asu_campus_3p5"
dm.download(scen_name)
dataset = dm.load(scen_name)

# %% [markdown]
# ## Set Doppler Directly
#
# Manually configure Doppler frequency shifts.

# %%
# Create channel parameters with Doppler enabled
ch_params = dm.ChannelParameters()
ch_params.enable_doppler = True

print(f"Doppler enabled: {ch_params.enable_doppler}")

# %%
# Set Doppler shifts directly for each user and path
num_users = len(dataset.power)
num_paths = dataset.power.shape[1]

# Example: Set constant Doppler shift
doppler_shifts = np.ones((num_users, num_paths)) * 100  # 100 Hz
dataset.set_doppler(doppler_shifts)

print(f"Doppler shape: {dataset.doppler.shape}")
print(f"Sample Doppler values: {dataset.doppler[0, :5]} Hz")

# %% [markdown]
# ## Set Speeds
#
# Define velocities for users or objects.

# %% [markdown]
# ### User Velocity

# %%
# Set velocity for receivers (users)
# Velocity in [vx, vy, vz] format (m/s)
user_velocity = np.array([5.0, 0.0, 0.0])  # 5 m/s in x-direction

# Apply to all users
velocities = np.tile(user_velocity, (num_users, 1))
dataset.rx_vel = velocities

print(f"User velocities shape: {velocities.shape}")
print(f"First user velocity: {velocities[0]} m/s")

# %% [markdown]
# ### Object Velocity

# %%
# Set velocity for transmitter (BS)
tx_velocity = np.array([0.0, 0.0, 0.0])  # Stationary BS

dataset.tx_vel = tx_velocity

# %% [markdown]
# ### Calculate Doppler from Velocities

# %%
# Compute Doppler shifts based on velocities
ch_params.doppler = True
dataset.compute_channels(ch_params)
channels_with_doppler = dataset.channel

# Access computed Doppler shifts
if hasattr(dataset, "doppler"):
    print(f"Computed Doppler shifts: {dataset.doppler[0, :5]} Hz")

# %% [markdown]
# ## Understanding Dynamic Datasets
#
# **Dynamic scenarios** are series of ray-tracing snapshots where at least one element"s
# properties change between scenes. A scenario is "dynamic" when at least one property of
# at least one element changes, requiring a new ray-tracing simulation.
#
# ### What Can Change Between Scenes?
#
# Elements in the scene are either:
# - **Objects** (no antennas): Properties that affect ray-tracing are **position**,
#   **orientation**, and **material**
# - **Transmitters/Receivers**: Properties that affect ray-tracing are **position**,
#   **orientation**, and **antenna**
#
# In practice, **position changes are by far the most common**. Each scene is ray-traced
# independently after the property changes, capturing how propagation evolves as elements move.
#
# ### Ray-Tracing and Time
#
# **Ray-tracing has no concept of time—only space.** Ray-tracing is fully deterministic,
# not stochastic. Dynamic scenarios simply provide consecutive snapshots of a changing scene.
# When elements move, they interact differently with propagation, and consecutive snapshots
# capture those changes for more realistic modeling.
#
# **Time is a construct you apply in your simulations.** In DeepMIMO, Doppler effects are
# added afterwards based on configured velocities, providing maximum flexibility without
# loss of generality.
#
# ### When to Use Dynamic vs Static Scenarios
#
# Many studies accept using **static scenarios with user sampling along trajectories** as an
# approximation. For example, if you want to model a user moving between buildings without
# considering interaction with moving cars, a static scenario is sufficient. This provides
# a good trade-off between simulation complexity and realism.
#
# **Important:** Current dynamic dataset support for channel generation across scenes is
# limited. **We recommend using static datasets with mobility modeling** (as shown in this
# tutorial) for most applications including channel prediction, beam tracking, and channel aging.
#
# Below is a demonstration of loading a dynamic dataset to visualize position changes.

# %%
# Example: Load dynamic dataset (demonstration only)
dyn_scen_name = "asu_campus_3p5_dyn"

print("Loading dynamic scenario...")

# UNCOMMENT TO RUN. Currently commented to avoid 1.6 GB download during pytests.

# Note: Load only one transmitter to avoid MacroDataset structure
# which is not fully supported in dynamic scenarios yet

# dataset_dyn = dm.load(dyn_scen_name, tx_sets={1: [0]})
# print(f"Number of scenes in dynamic dataset: {len(dataset_dyn)}")

dataset_dyn = None

# %% [markdown]
# ### Visualize Position Changes Across Scenes
#
# Dynamic datasets contain multiple scenes. When timestamps are applied, the velocity
# between consecutive user positions across scenes will vary based on temporal spacing.

# %%
# In this dynamic dataset, the property that changes across scenes is the position of one of the
# transmitters.

if dataset_dyn is not None:
    # Visualize position evolution for a transmitter across scenes
    tx1_pos = np.array(dataset_dyn.tx_pos)

    plt.figure(figsize=(10, 6), tight_layout=True)
    dataset_dyn[0].scene.plot(proj_3D=False)
    plt.scatter(
        tx1_pos[:, 0],
        tx1_pos[:, 1],
        c="blue",
        marker="o",
        s=50,
        alpha=0.6,
        label="Transmitter Trajectory",
    )
    plt.legend()
    plt.show()
    print(f"Transmitter moved across {len(tx1_pos)} scene snapshots")

# %%

# Apply timestamps to compute velocities between scenes
if dataset_dyn is not None:
    dataset_dyn.set_timestamps(1)  # 1 second between consecutive scenes
    # If the position differences between scenes are constant, velocities will be constant too
    # set_timestamps accepts a list of different time deltas between scenes as well.

    # Timestamps are computed given a constant time delta of 1 second
    print(f"timestamps: {dataset_dyn.timestamps}")
    # Constant (zero) rx velocities because rx positions are constant
    print(f"rx_vel: {dataset_dyn.rx_vel}")
    # Constant (non-zero) tx velocities due to constant 5 meter position differences between scenes
    print(f"tx_vel: {dataset_dyn.tx_vel}")
    # Constant (zero) object velocities - object positions never change across scenes
    print(f"obj_vel: {[obj.vel for obj in dataset_dyn.scene.objects]}")

    # Note: The Dynamic dataset should not contain MacroDatasets, since tx-rx pair Datasets.
    # A Dataset will contain a single tx-rx pair, when we pre-load it with tx_sets={1: [0]}.
    print(f"Transmitter velocity (m/s) in scene 1: {dataset_dyn[1].tx_vel}")

# %% [markdown]
# **Note:** While dynamic datasets show realistic element movements, for most applications
# (channel prediction, beam tracking, etc.), **static scenarios with user sampling along
# trajectories** provide sufficient realism with simpler implementation. See the
# [user sampling documentation](https://deepmimo.net/docs/manual_full.html#user-sampling)
# for trajectory-based mobility in static scenarios.

# %% [markdown]
# ## Doppler Spectrum

# %%
# Analyze Doppler spectrum
if hasattr(dataset, "doppler") and dataset.doppler is not None:
    doppler_values = dataset.doppler[dataset.doppler != 0]

    plt.figure(figsize=(10, 5))
    plt.hist(doppler_values, bins=50, edgecolor="black")
    plt.xlabel("Doppler Shift (Hz)")
    plt.ylabel("Count")
    plt.title("Doppler Spectrum Distribution")
    plt.grid(visible=True)
    plt.show()

# %% [markdown]
# ## Mobility Patterns

# %% [markdown]
# ### Linear Motion

# %%
# Define linear motion pattern
start_pos = dataset.rx_pos[0]
velocity = np.array([10.0, 5.0, 0.0])  # m/s
time_steps = np.linspace(0, 1.0, 10)  # 0 to 1 second, 10 steps

# Calculate positions over time
positions_over_time = np.array([start_pos + velocity * t for t in time_steps])

plt.figure(figsize=(10, 6))
plt.plot(positions_over_time[:, 0], positions_over_time[:, 1], "o-")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Linear Motion Trajectory")
plt.grid(visible=True)
plt.show()

# %% [markdown]
# ### Circular Motion

# %%
# Define circular motion
center = np.array([0, 0, 1.5])
radius = 50  # meters
angular_velocity = 2 * np.pi / 10  # rad/s (complete circle in 10 seconds)
time_steps = np.linspace(0, 10, 50)  # 10 seconds, 50 steps

circular_positions = np.array(
    [
        [
            center[0] + radius * np.cos(angular_velocity * t),
            center[1] + radius * np.sin(angular_velocity * t),
            center[2],
        ]
        for t in time_steps
    ]
)

plt.figure(figsize=(8, 8))
plt.plot(circular_positions[:, 0], circular_positions[:, 1], "o-")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Circular Motion Trajectory")
plt.axis("equal")
plt.grid(visible=True)
plt.show()

# %% [markdown]
# ## Maximum Doppler Shift

# %%
# Calculate maximum Doppler shift
carrier_freq = 3.5e9  # 3.5 GHz
speed_of_light = 3e8  # m/s
max_velocity = 30  # m/s (108 km/h)

max_doppler = (max_velocity / speed_of_light) * carrier_freq

print(f"Carrier frequency: {carrier_freq / 1e9} GHz")
print(f"Maximum velocity: {max_velocity} m/s")
print(f"Maximum Doppler shift: {max_doppler:.2f} Hz")

# %% [markdown]
# ---
#
# ## Next Steps
#
# Continue with:
# - **Tutorial 6: Beamforming** - Implement beamforming and spatial processing
# - **Tutorial 7: Convert & Upload Ray-tracing dataset** - Work with external ray tracers

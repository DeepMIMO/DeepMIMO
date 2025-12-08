"""# Doppler and Mobility

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/5_doppler_mobility.py)
&nbsp;
[![GitHub](https://img.shields.io/badge/Open_on-GitHub-181717?logo=github&style=for-the-badge)](https://github.com/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/5_doppler_mobility.py)

---

**Tutorial Overview:**
- Set Doppler Directly - Configure Doppler shifts manually
- Set Speeds - Define user/object velocities
- Set Timestamps - Configure time evolution of the scene

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
# ## Set Timestamps
#
# Configure time evolution for time-varying channels.

# %%
# Note: Timestamp-based channel generation requires dynamic datasets
# which track temporal changes. For static scenarios, we generate channels
# with the configured Doppler parameters directly.

print("Note: Time-varying channel generation requires dynamic datasets")
print("For static scenarios, Doppler is applied based on configured velocities")

# %% [markdown]
# ### Visualize Time-Varying Channel
#
# Note: This section requires dynamic datasets to visualize channel evolution over time.
# For static scenarios with Doppler, the phase varies but requires explicit time-stepping.

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
    plt.grid(True)
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
plt.grid(True)
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
plt.grid(True)
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

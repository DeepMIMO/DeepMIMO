"""
# Doppler and Mobility

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
dataset.set_user_velocity(velocities)

print(f"User velocities shape: {velocities.shape}")
print(f"First user velocity: {velocities[0]} m/s")

# %% [markdown]
# ### Object Velocity

# %%
# Set velocity for transmitter or scattering objects
tx_velocity = np.array([0.0, 0.0, 0.0])  # Stationary BS

dataset.set_obj_vel(tx_velocity)

# %% [markdown]
# ### Calculate Doppler from Velocities

# %%
# Compute Doppler shifts based on velocities
ch_params.enable_doppler = True
channels_with_doppler = dataset.get_time_domain_channel(ch_params=ch_params)

# Access computed Doppler shifts
if hasattr(dataset, 'doppler'):
    print(f"Computed Doppler shifts: {dataset.doppler[0, :5]} Hz")

# %% [markdown]
# ## Set Timestamps
#
# Configure time evolution for time-varying channels.

# %%
# Define timestamps for channel snapshots
num_snapshots = 10
timestamps = np.linspace(0, 1.0, num_snapshots)  # 0 to 1 second

dataset.set_timestamps(timestamps)

print(f"Timestamps: {timestamps} s")

# %% [markdown]
# ### Generate Time-Varying Channels

# %%
# Generate channels at different time instances
time_varying_channels = []

for t_idx, t in enumerate(timestamps):
    # Update dataset timestamp
    dataset.current_time = t
    
    # Generate channel
    ch = dataset.get_time_domain_channel(ch_params=ch_params)
    time_varying_channels.append(ch)

print(f"Generated {len(time_varying_channels)} channel snapshots")

# %% [markdown]
# ### Visualize Time-Varying Channel

# %%
# Plot channel magnitude over time for one user and path
user_idx = 0
rx_ant = 0
tx_ant = 0
path_idx = 0

channel_evolution = [np.abs(ch[user_idx, rx_ant, tx_ant, path_idx]) 
                     for ch in time_varying_channels]

plt.figure(figsize=(10, 5))
plt.plot(timestamps, channel_evolution, 'o-')
plt.xlabel('Time (s)')
plt.ylabel('|h|')
plt.title(f'Channel Evolution - User {user_idx}, Path {path_idx}')
plt.grid(True)
plt.show()

# %% [markdown]
# ## Doppler Spectrum

# %%
# Analyze Doppler spectrum
if hasattr(dataset, 'doppler') and dataset.doppler is not None:
    doppler_values = dataset.doppler[dataset.doppler != 0]
    
    plt.figure(figsize=(10, 5))
    plt.hist(doppler_values, bins=50, edgecolor='black')
    plt.xlabel('Doppler Shift (Hz)')
    plt.ylabel('Count')
    plt.title('Doppler Spectrum Distribution')
    plt.grid(True)
    plt.show()

# %% [markdown]
# ## Mobility Patterns

# %% [markdown]
# ### Linear Motion

# %%
# Define linear motion pattern
start_pos = dataset.user_positions[0]
velocity = np.array([10.0, 5.0, 0.0])  # m/s

# Calculate positions over time
positions_over_time = []
for t in timestamps:
    pos = start_pos + velocity * t
    positions_over_time.append(pos)

positions_over_time = np.array(positions_over_time)

plt.figure(figsize=(10, 6))
plt.plot(positions_over_time[:, 0], positions_over_time[:, 1], 'o-')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Linear Motion Trajectory')
plt.grid(True)
plt.show()

# %% [markdown]
# ### Circular Motion

# %%
# Define circular motion
center = np.array([0, 0, 1.5])
radius = 50  # meters
angular_velocity = 2 * np.pi / 10  # rad/s (complete circle in 10 seconds)

circular_positions = []
for t in timestamps:
    angle = angular_velocity * t
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    z = center[2]
    circular_positions.append([x, y, z])

circular_positions = np.array(circular_positions)

plt.figure(figsize=(8, 8))
plt.plot(circular_positions[:, 0], circular_positions[:, 1], 'o-')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Circular Motion Trajectory')
plt.axis('equal')
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

print(f"Carrier frequency: {carrier_freq/1e9} GHz")
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


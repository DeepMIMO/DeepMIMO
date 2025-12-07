"""
# Beamforming

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/6_beamforming.py)
&nbsp;
[![GitHub](https://img.shields.io/badge/Open_on-GitHub-181717?logo=github&style=for-the-badge)](https://github.com/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/6_beamforming.py)

---

**Tutorial Overview:**
- Computing Beamformers - Calculate received power with beamforming
- Steering Vectors - Generate steering vectors for different angles
- Beamforming Visualization - Visualize beamforming patterns and performance

**Related Video:** [Beamforming Video](https://youtu.be/IPVnIW2vGLE)

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
# ## Computing Beamformers
#
# Calculate steering vectors and beamforming gains.

# %%
# Configure multi-antenna system
ch_params = dm.ChannelParameters()
ch_params.ant_bs.shape = [8, 1]  # 8-element linear array at BS
ch_params.ant_ue.shape = [1, 1]  # Single antenna at UE

# Generate channels
channels = dataset.get_time_domain_channel(ch_params=ch_params)
print(f"Channel shape: {channels.shape}")

# %% [markdown]
# ## Steering Vectors
#
# Generate steering vectors for specific angles.

# %%
# Array parameters
num_antennas = ch_params.ant_bs.shape[0]
antenna_spacing = ch_params.ant_bs.spacing  # in wavelengths

# Generate steering vector for a specific angle
theta = 30  # degrees
sv = dm.steering_vec(theta, num_antennas, antenna_spacing)

print(f"Steering vector shape: {sv.shape}")
print(f"Steering vector for {theta}°: {sv[:4]}...")

# %% [markdown]
# ### Steering Vectors for Multiple Angles

# %%
# Generate steering vectors for a range of angles
angles = np.arange(-90, 91, 5)  # -90° to 90° in 5° steps
steering_vectors = np.array([dm.steering_vec(angle, num_antennas, antenna_spacing) 
                              for angle in angles])

print(f"Steering vectors shape: {steering_vectors.shape}")

# %% [markdown]
# ## Beamforming Gain

# %%
# Compute beamforming gain for a specific user
user_idx = 0
h = channels[user_idx, :, 0, 0]  # Channel vector for user 0

# Matched filter beamformer (MF)
w_mf = h.conj() / np.linalg.norm(h)

# Compute received power with beamforming
bf_gain = np.abs(np.dot(w_mf.conj(), h))**2
no_bf_power = np.sum(np.abs(h)**2) / num_antennas

print(f"Power without beamforming: {10*np.log10(no_bf_power):.2f} dB")
print(f"Power with beamforming: {10*np.log10(bf_gain):.2f} dB")
print(f"Beamforming gain: {10*np.log10(bf_gain/no_bf_power):.2f} dB")

# %% [markdown]
# ## Beam Patterns

# %% [markdown]
# ### Array Factor Pattern

# %%
# Compute array factor for different angles
array_factor = []
for angle in angles:
    sv = dm.steering_vec(angle, num_antennas, antenna_spacing)
    # Beamformer pointed at 0 degrees
    w = dm.steering_vec(0, num_antennas, antenna_spacing)
    af = np.abs(np.dot(w.conj(), sv))
    array_factor.append(af)

array_factor = np.array(array_factor)

# Plot array factor in dB
plt.figure(figsize=(10, 6))
plt.plot(angles, 20*np.log10(array_factor / np.max(array_factor)))
plt.xlabel('Angle (degrees)')
plt.ylabel('Array Factor (dB)')
plt.title('Beamforming Array Factor Pattern')
plt.grid(True)
plt.ylim([-40, 0])
plt.show()

# %% [markdown]
# ### Polar Plot

# %%
# Polar plot of beam pattern
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
angles_rad = np.deg2rad(angles)
ax.plot(angles_rad, array_factor)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_title('Beam Pattern (Polar)')
ax.grid(True)
plt.show()

# %% [markdown]
# ## Beamforming Visualization

# %% [markdown]
# ### Received Power per Beam

# %%
# Scan through all angles and compute received power
num_users = min(100, len(dataset.power))  # Use first 100 users
beam_powers = np.zeros((num_users, len(angles)))

for user_idx in range(num_users):
    h = channels[user_idx, :, 0, 0]  # Channel vector
    
    for angle_idx, angle in enumerate(angles):
        w = dm.steering_vec(angle, num_antennas, antenna_spacing)
        power = np.abs(np.dot(w.conj(), h))**2
        beam_powers[user_idx, angle_idx] = power

# Plot heatmap
plt.figure(figsize=(12, 6))
plt.imshow(10*np.log10(beam_powers + 1e-10), 
           aspect='auto', cmap='hot', origin='lower',
           extent=[angles[0], angles[-1], 0, num_users])
plt.colorbar(label='Received Power (dBW)')
plt.xlabel('Beam Angle (degrees)')
plt.ylabel('User Index')
plt.title('Received Power per Beam and User')
plt.show()

# %% [markdown]
# ### Best Beam per Position

# %%
# Find best beam for each user
best_beams = np.argmax(beam_powers, axis=1)
best_angles = angles[best_beams]

plt.figure(figsize=(10, 6))
plt.scatter(dataset.user_positions[:num_users, 0],
            dataset.user_positions[:num_users, 1],
            c=best_angles, cmap='viridis', s=20)
plt.colorbar(label='Best Beam Angle (degrees)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Best Beam Angle per User Position')
plt.grid(True)
plt.show()

# %% [markdown]
# ### Max Received Power Map

# %%
# Maximum received power (best beam) for each user
max_powers = np.max(beam_powers, axis=1)

plt.figure(figsize=(10, 6))
plt.scatter(dataset.user_positions[:num_users, 0],
            dataset.user_positions[:num_users, 1],
            c=10*np.log10(max_powers + 1e-10), 
            cmap='viridis', s=20)
plt.colorbar(label='Max Received Power (dBW)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Maximum Received Power with Beamforming')
plt.grid(True)
plt.show()

# %% [markdown]
# ## Multi-User Beamforming

# %%
# Zero-forcing beamforming for multiple users
num_selected_users = min(num_antennas - 1, num_users)
selected_users = np.random.choice(num_users, num_selected_users, replace=False)

# Channel matrix for selected users
H = np.array([channels[u, :, 0, 0] for u in selected_users]).T

# Zero-forcing precoder
W_zf = np.linalg.pinv(H)

print(f"Channel matrix H shape: {H.shape}")
print(f"Zero-forcing precoder W shape: {W_zf.shape}")

# %% [markdown]
# ## Beamforming Gain Analysis

# %%
# Compare beamforming techniques
gains_mf = []
gains_uniform = []

for user_idx in range(num_users):
    h = channels[user_idx, :, 0, 0]
    
    # Matched filter
    w_mf = h.conj() / np.linalg.norm(h)
    gain_mf = np.abs(np.dot(w_mf.conj(), h))**2
    
    # Uniform weighting
    w_uniform = np.ones(num_antennas) / np.sqrt(num_antennas)
    gain_uniform = np.abs(np.dot(w_uniform.conj(), h))**2
    
    gains_mf.append(gain_mf)
    gains_uniform.append(gain_uniform)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.hist([10*np.log10(gains_uniform), 10*np.log10(gains_mf)], 
         bins=30, label=['Uniform', 'Matched Filter'], alpha=0.7)
plt.xlabel('Received Power (dBW)')
plt.ylabel('Count')
plt.title('Beamforming Gain Comparison')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ---
#
# ## Next Steps
#
# Continue with:
# - **Tutorial 7: Convert & Upload Ray-tracing dataset** - Work with external ray tracers
# - **Tutorial 8: Migration Guide** - Migrating from DeepMIMO v3 to v4


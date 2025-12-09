# ruff: noqa: N999
"""# Detailed Channel Generation.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/3_channel_generation.py)
&nbsp;
[![GitHub](https://img.shields.io/badge/Open_on-GitHub-181717?logo=github&style=for-the-badge)](https://github.com/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/3_channel_generation.py)

---

**Tutorial Overview:**
- Channel Parameters - Configuring channel generation
- Time Domain - Generate time-domain channel responses
- Frequency Domain (OFDM) - Generate OFDM channel responses
- Antenna Rotation - Adjust antenna orientations

**Related Video:** [Channel Generation Video](https://youtu.be/xsl6gjTEu2U)

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
# ## Channel Parameters
#
# Configure channel generation with custom parameters.

# %%
# Get help on channel parameters
dm.info("ch_params")

# %%
# Create channel parameters
ch_params = dm.ChannelParameters()
print("Default channel parameters:")
print(ch_params)

# %%
# Customize antenna array configuration
ch_params.bs_antenna.shape = [4, 4]  # 4x4 BS antenna array
ch_params.bs_antenna.spacing = 0.5  # Half-wavelength spacing
ch_params.ue_antenna.shape = [2, 1]  # 2x1 UE antenna array

print(f"BS antenna shape: {ch_params.bs_antenna.shape}")
print(f"UE antenna shape: {ch_params.ue_antenna.shape}")

# %%
# Set the number of paths to use
ch_params.num_paths = 20
print(f"Number of paths: {ch_params.num_paths}")

# %% [markdown]
# ## Time Domain
#
# Generate time-domain channel responses.

# %%
# Set to time domain
ch_params.freq_domain = False

# Generate time-domain channel
dataset.compute_channels(ch_params)
time_channels = dataset.channel
print(f"Time domain channel shape: {time_channels.shape}")
print("Shape: [num_rx, num_rx_ant, num_tx_ant, num_paths]")

# %%
# Visualize channel impulse response for one user
user_idx = 10
cir = time_channels[user_idx, 0, 0, :]  # First RX ant, first TX ant

plt.figure(figsize=(10, 5))
plt.stem(np.abs(cir))
plt.xlabel("Path Index")
plt.ylabel("|h|")
plt.title(f"Channel Impulse Response - User {user_idx}")
plt.grid(visible=True)
plt.show()

# %% [markdown]
# ## Frequency Domain (OFDM)
#
# Generate OFDM channel responses.

# %%
# Configure OFDM parameters
ch_params.freq_domain = True
ch_params.ofdm.bandwidth = 10e6  # 10 MHz
ch_params.ofdm.subcarriers = 512
ch_params.ofdm.selected_subcarriers = np.arange(64)  # Select first 64 subcarriers
ch_params.ofdm.rx_filter = False  # No low-pass filter

# Generate frequency-domain channel
dataset.compute_channels(ch_params)
freq_channels = dataset.channel
print(f"Frequency domain channel shape: {freq_channels.shape}")
print("Shape: [num_rx, num_rx_ant, num_tx_ant, num_subcarriers]")

# %%
# Visualize frequency response for one user
user_idx = 10
freq_response = freq_channels[user_idx, 0, 0, :]  # First RX ant, first TX ant

plt.figure(figsize=(10, 5))
plt.plot(np.abs(freq_response))
plt.xlabel("Subcarrier Index")
plt.ylabel("|H[k]|")
plt.title(f"Channel Frequency Response - User {user_idx}")
plt.grid(visible=True)
plt.show()

# %% [markdown]
# ### Sampling Subcarriers
#
# You can sample only specific subcarriers for efficiency.

# %%
# Sample every 4th subcarrier
ch_params.ofdm.selected_subcarriers = list(range(0, 512, 4))

dataset.compute_channels(ch_params)
freq_channels_sampled = dataset.channel
print(f"Sampled frequency domain channel shape: {freq_channels_sampled.shape}")

# %% [markdown]
# ## Antenna Rotation
#
# Adjust antenna orientations to simulate different deployment scenarios.

# %% [markdown]
# ### Azimuth Rotation

# %%
# Rotate BS antennas in azimuth
ch_params.bs_antenna.rotation = [45, 0, 0]  # 45° azimuth rotation

dataset.compute_channels(ch_params)
rotated_channels = dataset.channel
print(f"Channels with rotated BS antennas shape: {rotated_channels.shape}")

# %%
# Compare power before and after rotation
ch_params.bs_antenna.rotation = [0, 0, 0]
dataset.compute_channels(ch_params)
channels_no_rot = dataset.channel

ch_params.bs_antenna.rotation = [90, 0, 0]
dataset.compute_channels(ch_params)
channels_rot = dataset.channel

power_no_rot = np.mean(np.abs(channels_no_rot[0:100]) ** 2)
power_rot = np.mean(np.abs(channels_rot[0:100]) ** 2)

print(f"Average power (no rotation): {power_no_rot:.6f}")
print(f"Average power (90° rotation): {power_rot:.6f}")

# %% [markdown]
# ### Elevation Rotation

# %%
# Rotate BS antennas in elevation
ch_params.bs_antenna.rotation = [0, 15, 0]  # 15° elevation (downtilt)

dataset.compute_channels(ch_params)
tilted_channels = dataset.channel
print(f"Channels with tilted BS antennas shape: {tilted_channels.shape}")

# %% [markdown]
# ## Channel Comparison
#
# Compare different antenna configurations.

# %%
# Single antenna vs. array
ch_params_single = dm.ChannelParameters()
ch_params_single.bs_antenna.shape = [1, 1]
ch_params_single.ue_antenna.shape = [1, 1]

ch_params_array = dm.ChannelParameters()
ch_params_array.bs_antenna.shape = [8, 1]
ch_params_array.ue_antenna.shape = [4, 1]

dataset.compute_channels(ch_params_single)
channels_single = dataset.channel

dataset.compute_channels(ch_params_array)
channels_array = dataset.channel

print(f"Single antenna channel shape: {channels_single.shape}")
print(f"Array antenna channel shape: {channels_array.shape}")

# %% [markdown]
# ---
#
# ## Next Steps
#
# Continue with:
# - **Tutorial 4: User Selection and Dataset Manipulation** - Learn how to filter and sample users
# - **Tutorial 5: Doppler and Mobility** - Add time-varying effects to your channels

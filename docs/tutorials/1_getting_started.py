"""
# Getting Started with DeepMIMO

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/1_getting_started.py)
&nbsp;
[![GitHub](https://img.shields.io/badge/Open_on-GitHub-181717?logo=github&style=for-the-badge)](https://github.com/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/1_getting_started.py)

---

**Tutorial Overview:**
- **Part 1**: Hello World - Load a simple scenario, generate channels
- **Part 2**: Deep Dive - Explore complex scenarios with multiple base stations
- **Part 3**: Discovery - Learn how to discover more using `dm.info()`, aliases, and implicit computations

**Related Video:** [Getting Started Video](https://youtu.be/LDG6IPEHY54)

---
"""

# %% [markdown]
# ## Part 1: Hello World
#
# Let's start with the absolute basics: installing DeepMIMO, loading a simple scenario, and generating channels.

# %%
# Install DeepMIMO (if not already installed)
# %pip install --pre deepmimo

# %%
# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import deepmimo as dm

# %% [markdown]
# ### Load a Simple Scenario
#
# We'll use the ASU Campus scenario, which is a simple outdoor scenario.

# %%
# Download and load a scenario
scen_name = "asu_campus_3p5"
dm.download(scen_name)
dataset = dm.load(scen_name)

# %%
# Check what we loaded
print(f"Loaded scenario: {scen_name}")
print(f"Dataset type: {type(dataset)}")
print(dataset)

# %% [markdown]
# ### Generate Channels
#
# Now let's generate some basic channel matrices.

# %%
# Generate time-domain channels
channels = dataset.get_time_domain_channel()
print(f"Channel shape: {channels.shape}")
print(f"Channel type: {type(channels)}")

# %% [markdown]
# ### Basic Inspection
#
# Let's check some basic properties of our dataset.

# %%
# Check line-of-sight status
print(f"LOS status shape: {dataset.los.shape}")
print(f"Number of users with LOS: {np.sum(dataset.los == 1)}")
print(f"Number of users with NLOS: {np.sum(dataset.los == 0)}")
print(f"Number of users with no paths: {np.sum(dataset.los == -1)}")

# %%
# Check pathloss
pathloss = dataset.pathloss
print(f"Pathloss shape: {pathloss.shape}")
print(f"Average pathloss: {np.mean(pathloss[pathloss > -np.inf]):.2f} dB")

# %% [markdown]
# ---
#
# ## Part 2: Deep Dive with Complex Scenarios
#
# Now let's explore a more complex scenario with multiple base stations and dive deeper into the dataset structure.

# %%
# Load a city scenario with multiple base stations
city_scen = "city_18_denver"
dm.download(city_scen)
city_dataset = dm.load(city_scen)

# %% [markdown]
# ### Explore TX/RX Pairs and Sets

# %%
# Get information about transmitter and receiver sets
print("Available TX sets:", dm.get_txrx_sets(city_scen)["tx_sets"])
print("Available RX sets:", dm.get_txrx_sets(city_scen)["rx_sets"])

# %%
# Get TX/RX pair information
pairs = dm.get_txrx_pairs(city_scen)
print(f"Number of TX/RX pairs: {len(pairs)}")
print(f"First few pairs: {pairs[:5]}")

# %% [markdown]
# ### Explore Available Matrices
#
# Let's see what matrices/parameters are available in the dataset.

# %%
# List available fundamental matrices
print("Available dataset attributes:")
for attr in dir(city_dataset):
    if not attr.startswith('_') and not callable(getattr(city_dataset, attr)):
        print(f"  - {attr}")

# %%
# Explore some key matrices
print(f"Power matrix shape: {city_dataset.power.shape}")
print(f"Phase matrix shape: {city_dataset.phase.shape}")
print(f"Delay matrix shape: {city_dataset.delay.shape}")
print(f"AOA (azimuth) matrix shape: {city_dataset.aoa_az.shape}")
print(f"AOD (azimuth) matrix shape: {city_dataset.aod_az.shape}")

# %% [markdown]
# ### Scenario Summary
#
# Get a high-level overview of the scenario.

# %%
# Get scenario information
info = dm.get_scenario_info(city_scen)
print("Scenario Summary:")
for key, value in info.items():
    print(f"  {key}: {value}")

# %% [markdown]
# ---
#
# ## Part 3: Discovery - Learning More
#
# DeepMIMO provides powerful tools to help you discover and understand the available functions and parameters.

# %% [markdown]
# ### Using `dm.info()` for Parameter Documentation

# %%
# Get information about all available parameters
dm.info('all')

# %%
# Get information about a specific parameter
dm.info('power')

# %%
# Get information about channel parameters
dm.info('ch_params')

# %% [markdown]
# ### Inspecting Function Docstrings
#
# You can use `dm.info()` to inspect the docstrings of any function or object!

# %%
# Inspect the docstring of a function
dm.info(dm.load)

# %%
# Inspect the docstring of a method
dm.info(dataset.get_time_domain_channel)

# %% [markdown]
# ### Understanding Aliases
#
# DeepMIMO provides convenient aliases for common parameters.

# %%
# Check if an alias resolves to a parameter
dm.info('rx_pos')  # This is an alias

# %%
# Common aliases
print("Using aliases:")
print(f"  dataset.rx_pos is the same as dataset.user_positions")
print(f"  dataset.tx_pos is the same as dataset.bs_positions")
print(f"  Shapes match: {np.array_equal(dataset.rx_pos, dataset.user_positions)}")

# %% [markdown]
# ### Implicit Computations
#
# Some dataset attributes are computed on-the-fly when you access them.

# %%
# These are computed implicitly:
print(f"Number of paths per user: {dataset.num_paths[:10]}")  # Computed from power/delay
print(f"Pathloss (dB): {dataset.pathloss[0, :5]}")  # Computed from power
print(f"Distance (m): {dataset.dist[0, :5]}")  # Computed from delay

# %% [markdown]
# ### Discovering More Functions
#
# Explore the available functions in the DeepMIMO module.

# %%
# List key functions in the dm module
print("Key DeepMIMO functions:")
key_functions = [f for f in dir(dm) if not f.startswith('_') and callable(getattr(dm, f))]
for func in sorted(key_functions)[:20]:  # Show first 20
    print(f"  - dm.{func}()")

# %%
# Get help for any function
dm.info(dm.download)

# %% [markdown]
# ---
#
# ## Next Steps
#
# Now that you understand the basics, explore these tutorials:
#
# - **Tutorial 2: Visualization and Scene** - Learn how to visualize coverage maps, rays, and 3D scenes
# - **Tutorial 3: Detailed Channel Generation** - Deep dive into channel generation parameters
# - **Tutorial 4: User Selection and Dataset Manipulation** - Learn how to filter and sample users
# - **Tutorial 5: Doppler and Mobility** - Add time-varying effects to your channels
# - **Tutorial 6: Beamforming** - Implement beamforming and spatial processing
# - **Tutorial 7: Convert & Upload Ray-tracing dataset** - Work with external ray tracers
# - **Tutorial 8: Migration Guide** - Migrating from DeepMIMO v3 to v4


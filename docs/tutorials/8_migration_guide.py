"""# Migration Guide: DeepMIMO v3 to v4.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/8_migration_guide.py)
&nbsp;
[![GitHub](https://img.shields.io/badge/Open_on-GitHub-181717?logo=github&style=for-the-badge)](https://github.com/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/8_migration_guide.py)

---

**Tutorial Overview:**
- Key differences between v3 and v4
- Channel generation changes
- Data access and format changes
- User selection changes
- Best practices for migration

**Related Video:** [Migration Video](https://youtu.be/15nQWS15h3k)

---
"""

# %%

# %% [markdown]
# ## Overview
#
# DeepMIMO v4 introduces significant changes for improved performance, storage efficiency,
# and expanded capabilities. This guide helps you migrate from v3 to v4.

# %% [markdown]
# ## v3 Workflow (Old)

# %%
# V3 Example (DO NOT RUN - for reference only)
v3_example = """
# DeepMIMO v3 workflow:
import DeepMIMOv3

# Load default parameters
params = DeepMIMOv3.default_params()

# Configure scenario
params["dataset_folder"] = "."
params["scenario"] = "asu_campus1"
params["active_BS"] = np.array([1])
params["user_rows"] = np.arange(321)

# Configure antennas
params["ue_antenna"]["shape"] = np.array([1, 1])
params["bs_antenna"]["shape"] = np.array([8, 1])

# Generate dataset
dataset = DeepMIMOv3.generate_data(params)

# Access data
channels = dataset[0]["user"]["channel"]
"""

print("DeepMIMO v3 workflow (deprecated):")
print(v3_example)

# %% [markdown]
# ## v4 Workflow (New)

# %%
# DeepMIMO v4 workflow:
import deepmimo as dm

# Download and load scenario
scen_name = "asu_campus_3p5"
dm.download(scen_name)
dataset = dm.load(scen_name)

# Configure channel parameters
ch_params = dm.ChannelParameters()
ch_params.bs_antenna.shape = [8, 1]
ch_params.ue_antenna.shape = [1, 1]

# Generate channels
dataset.compute_channels(ch_params)
channels = dataset.channel

print("v4 Dataset loaded successfully")
print(f"Channel shape: {channels.shape}")

# %% [markdown]
# ## Key Differences
#
# ### 1. Installation

# %%
# v3: pip install deepmimov3
# v4: pip install --pre deepmimo

print("v3: pip install DeepMIMOv3")
print("v4: pip install --pre deepmimo")

# %% [markdown]
# ### 2. Import Statements

# %%
# v3: import DeepMIMOv3
# v4: import deepmimo as dm

print("v3: import DeepMIMOv3")
print("v4: import deepmimo as dm")

# %% [markdown]
# ### 3. Dataset Loading

# %%
comparison = """
# v3:
params = DeepMIMOv3.default_params()
params["dataset_folder"] = "."
params["scenario"] = "asu_campus1"
dataset = DeepMIMOv3.generate_data(params)

# v4:
dm.download("asu_campus_3p5")
dataset = dm.load("asu_campus_3p5")
"""

print("Dataset Loading:")
print(comparison)

# %% [markdown]
# ## Channel Generation
#
# Channel generation is now explicit and on-demand.

# %% [markdown]
# ### v3: Implicit Generation

# %%
v3_channel_gen = """
# v3: Channels generated during dataset loading
params["OFDM"]["subcarriers"] = 512
params["OFDM"]["bandwidth"] = 10e6
dataset = DeepMIMOv3.generate_data(params)
channels = dataset[0]["user"]["channel"]
"""

print("v3 channel generation (automatic):")
print(v3_channel_gen)

# %% [markdown]
# ### v4: Explicit Generation

# %%
# v4: Channels generated on-demand
ch_params = dm.ChannelParameters()
ch_params.freq_domain = True  # Frequency domain
ch_params.ofdm.subcarriers = 512
ch_params.ofdm.bandwidth = 10e6

dataset.compute_channels(ch_params)
channels = dataset.channel
print(f"v4 channel shape: {channels.shape}")

# %% [markdown]
# ## Data Access
#
# Data access is simplified in v4.

# %% [markdown]
# ### v3: Nested Dictionary

# %%
v3_data_access = """
# v3: Nested dictionary access
power = dataset[bs_idx]["user"]["power"]
delay = dataset[bs_idx]["user"]["delay"]
aoa = dataset[bs_idx]["user"]["DoA_phi"]
aod = dataset[bs_idx]["user"]["DoD_phi"]
"""

print("v3 data access (nested):")
print(v3_data_access)

# %% [markdown]
# ### v4: Direct Attributes

# %%
# v4: Direct attribute access
power = dataset.power
delay = dataset.delay
aoa_az = dataset.aoa_az
aod_az = dataset.aod_az

print("v4 data access (direct):")
print(f"  Power shape: {power.shape}")
print(f"  Delay shape: {delay.shape}")
print(f"  AOA (az) shape: {aoa_az.shape}")

# %% [markdown]
# ## User Selection
#
# User selection methods have changed.

# %% [markdown]
# ### v3: Row/Column Selection

# %%
v3_user_selection = """
# v3: Use row indices during parameter setup
params["user_rows"] = np.arange(0, 100, 10)  # Every 10th row
dataset = DeepMIMOv3.generate_data(params)
"""

print("v3 user selection (pre-generation):")
print(v3_user_selection)

# %% [markdown]
# ### v4: Post-Loading Selection

# %%
# v4: Select users after loading
row_idxs = dataset.get_idxs(mode="row", row_idxs=list(range(0, 100, 10)))
subset = dataset.trim(idxs=row_idxs)

print("v4 user selection (post-loading):")
print(f"  Selected {len(row_idxs)} users")

# %% [markdown]
# ## Parameter Names
#
# Many parameter names have changed.

# %%
parameter_mapping = {
    "DoA_phi / DoA_theta": "aoa_az / aoa_el",
    "DoD_phi / DoD_theta": "aod_az / aod_el",
    "active_BS": "tx_sets (in dm.load())",
    "user_rows": "get_row_idxs() method",
    "ue_antenna": "ant_ue (in ChannelParameters)",
    "bs_antenna": "ant_bs (in ChannelParameters)",
}

print("Parameter name mapping (v3 -> v4):")
for v3_name, v4_name in parameter_mapping.items():
    print(f"  {v3_name:25s} -> {v4_name}")

# %% [markdown]
# ## Storage Format
#
# v4 uses more efficient storage.

# %%
# v4 files are typically 50% smaller
print("Storage improvements:")
print("  - v3: Nested dictionaries in .mat files")
print("  - v4: Direct NumPy arrays in .npz files")
print("  - Average size reduction: ~50%")

# %% [markdown]
# ## Migration Checklist
#
# Use this checklist when migrating your code:
#
# - [ ] Update installation: `pip install --pre deepmimo`
# - [ ] Change import: `import deepmimo as dm`
# - [ ] Replace `default_params()` with `dm.load()`
# - [ ] Replace `generate_data()` with `get_time/freq_domain_channel()`
# - [ ] Update data access from `dataset[bs]["user"]["param"]` to `dataset.param`
# - [ ] Move user selection from params to `dataset.subset()`
# - [ ] Update parameter names (DoA/DoD -> aoa/aod, etc.)
# - [ ] Update antenna configuration to ChannelParameters
# - [ ] Test thoroughly with your existing workflows

# %% [markdown]
# ## Common Migration Patterns

# %% [markdown]
# ### Pattern 1: Basic Channel Generation

# %%
print("Pattern 1: Basic Channel Generation")
print("\nv3:")
print("""
params = DeepMIMOv3.default_params()
params["scenario"] = "O1_60"
params["bs_antenna"]["shape"] = [4, 4]
dataset = DeepMIMOv3.generate_data(params)
ch = dataset[0]["user"]["channel"]
""")

print("\nv4:")
print("""
dataset = dm.load("city_18_denver_3p5")
ch_params = dm.ChannelParameters()
ch_params.bs_antenna.shape = [4, 4]
dataset.compute_channels(ch_params)
ch = dataset.channel
""")

# %% [markdown]
# ### Pattern 2: OFDM Channels

# %%
print("\nPattern 2: OFDM Channels")
print("\nv3:")
print("""
params["OFDM"]["subcarriers"] = 512
params["OFDM"]["bandwidth"] = 10e6
dataset = DeepMIMOv3.generate_data(params)
""")

print("\nv4:")
print("""
ch_params.freq_domain = True
ch_params.ofdm.subcarriers = 512
ch_params.ofdm.bandwidth = 10e6
dataset.compute_channels(ch_params)
ch = dataset.channel
""")

# %% [markdown]
# ### Pattern 3: User Sampling

# %%
print("\nPattern 3: User Sampling")
print("\nv3:")
print("""
params["user_rows"] = [0, 10, 20]
dataset = DeepMIMOv3.generate_data(params)
""")

print("\nv4:")
print("""
dataset = dm.load(scenario)
row_idxs = dataset.get_idxs(mode='row', row_idxs=[0, 10, 20])
subset = dataset.trim(idxs=row_idxs)
""")

# %% [markdown]
# ## Benefits of v4
#
# 1. **Performance**: Faster loading and processing
# 2. **Storage**: 50% smaller dataset files
# 3. **Flexibility**: On-demand channel generation
# 4. **Simplicity**: Cleaner API and data access
# 5. **Features**: New visualization, sampling, and converter tools
# 6. **Documentation**: Comprehensive tutorials and examples

# %% [markdown]
# ## Need Help?
#
# - Check the [documentation](https://deepmimo.github.io/DeepMIMO/)
# - Ask questions on [GitHub Discussions](https://github.com/DeepMIMO/DeepMIMO/discussions)
# - Report bugs on [GitHub Issues](https://github.com/DeepMIMO/DeepMIMO/issues)
# - Join the community at [deepmimo.net](https://deepmimo.net)

# %% [markdown]
# ---
#
# ## Next Steps
#
# Now that you've migrated to v4, explore:
# - **Tutorial 1: Getting Started** - Learn v4 basics
# - **Tutorial 2: Visualization** - Use new visualization features
# - **Tutorial 3: Channel Generation** - Master channel generation

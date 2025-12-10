"""DeepMIMO Examples Manual.

Comprehensive reference manual with all DeepMIMO examples.

Open in Colab:
https://colab.research.google.com/github/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/manual.py

Open on GitHub:
https://github.com/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/manual.py

This manual covers:
- Migration from v3 to v4
- Installation (Python and MATLAB)
- Loading datasets
- Scenario information
- Visualization
- Channel generation
- Basic and advanced operations
- Scene and materials
- User sampling
- Beamforming
- Converting from other ray tracers
- Uploading scenarios
"""


# %% [markdown]
# # Examples Manual
#
# Open in Colab:
# https://colab.research.google.com/drive/1U-e2rLDJYW-VcbJ7C3H2dqC625JJH5FF
#
# Open on GitHub:
# https://github.com/DeepMIMO/DeepMIMO/blob/main/docs/manual.ipynb
#
# ---
# **How to use this script**:
# 1. Install DeepMIMO: `pip install --pre deepmimo`
# 2. Run sections interactively in your IDE
# 3. Jump to the section of interest (see table below)
# 4. Watch the video explaining the section in detail

# %%
# Install DeepMIMO (run this in your terminal or uncomment to run here)
# pip install --pre deepmimo

# Import manual-wide dependencies

import matplotlib.pyplot as plt
import numpy as np

import deepmimo as dm

# Load example scenario
scen_name = "asu_campus_3p5"
dm.download(scen_name)
dataset = dm.load(scen_name)

# %% Purely visual changes for this notebook

import pydoc

pydoc.pager = pydoc.plainpager  # when calling help(function), print instead of page w/ less


# %% [markdown]
# | Section | Video | Subsection | Description | DeepMIMO Functions |
# |---------|-------|------------|-------------|----------------------|
# | [Migrating from v3](#migrating-from-v3) | [Video](https://youtu.be/15nQWS15h3k) | [Generating v3 Dataset](#generating-v3-dataset) | Usual workflow with DeepMIMO v2/v3 | pip install DeepMIMOv3, default_params(), generate_data() |
# | | | [Generating v4 Dataset](#generating-v4-dataset) | Usual workflow with DeepMIMO v4 | dm.load(), dataset.compute_channels() |
# | | | [Comparing v3 & v4](#comparing-v3--v4) | Understand and adapt to new design | dataset.get_row_idxs() |
# | [Install DeepMIMO](#install-deepmimo) | [Video](https://youtu.be/Mx2aXu9J0pA) | [Python](#python) | Setup in Python using mamba and pip| pip install --pre deepmimo |
# | | | [Matlab](#matlab) | Setup in Matlab using pyenv | pyenv, pyrun, pyrunfile |
# | [Load Dataset](#load-dataset) | [Video](https://youtu.be/LDG6IPEHY54) | [Simple](#simple) | Basic dataset loading method | dm.download(), dm.load() |
# | | | [Detailed](#detailed) | Advanced dataset loading options | dm.load() with tx_sets, rx_sets, matrices |
# | [Scenario Information](#scenario-information) | [Video](https://youtu.be/AfjRkFvC5CI) | [Summary](#summary) | High-level overview of scenario | dm.get_scenario_info() |
# | | | [Transmitters and Receivers](#transmitters-and-receivers) | Information on TX and RX placement | dm.get_txrx_sets(), dm.get_txrx_pairs() |
# | | | [Ray Tracing Parameters](#ray-tracing-parameters) | Configuration of ray tracing settings | dm.get_available_scenarios(), dm.get_params_path() |
# | [Visualization](#visualization) | [Video](https://youtu.be/MO7h2shBhsc) | [Coverage Maps](#coverage-maps) | Visualizing signal coverage | dm.plot_coverage() |
# | | | [Rays](#rays) | Ray propagation visualization | dm.plot_rays() |
# | | | [Path Plots](#path-plots) | Visualization of different path components | dm.plot_coverage() with different path metrics |
# | | | [Overlays](#overlays) | Combine different plots | dataset.\<attr\>.plot(), dataset.scene.plot(), dataset.plot_rays() |
# | [Channel Generation](#channel-generation) | [Video](https://youtu.be/xsl6gjTEu2U) | [Parameters](#parameters) | Configuring channel generation | dm.ChannelParameters(), dm.set_channel_params() |
# | | | [Time Domain](#time-domain) | Generate time-domain channel responses | dataset.get_time_domain_channel() |
# | | | [Frequency Domain (OFDM)](#frequency-domain-ofdm) | Generate OFDM channel responses | dataset.get_freq_domain_channel() |
# | | | [Doppler](#doppler) | Add Doppler to Channels | dataset.set_doppler(), dataset.set_obj_vel(), dataset.set_timestamps() |
# | [Basic Operations](#basic-operations) | [Video](https://youtu.be/gv9qhC-c4ho) | [Line-of-Sight Status](#line-of-sight-status) | Check if paths are LOS or NLOS | dataset.los, dataset.num_paths |
# | | | [Pathloss](#pathloss) | Calculate pathloss values | dataset.compute_pathloss() |
# | | | [Implicit Computations](#implicit-computations) | Compute parameters automatically | dataset.\<attribute\> |
# | | | [Aliases](#aliases) | Shortcuts for dataset fields | dataset.\<attribute-alias\> |
# | | | [Attribute Access](#attribute-access) | Directly access dataset properties | dataset.\<attribute\> |
# | | | [Antenna Rotation](#antenna-rotation) | Adjust antenna orientations | dataset.rotate_antennas() |
# | [Advanced Operations](#advanced-operations) | [Video](https://youtu.be/PApPjG4HTHs) | [Field-of-View](#field-of-view) | FoV analysis for receivers | dataset.trim_by_fov() |
# | [Scene & Materials](#scene--materials) | [Video](https://youtu.be/gq7F2cUthuU) | [Visualization](#scene-visualization) | Show scene | dataset.scene, dataset.materials |
# | | | [Operations](#scene-operations) | Retrieve objects and materials | scene.get_objects(), objects.get_materials() |
# | [User Sampling](#user-sampling) | [Video](https://youtu.be/KV0LLp0jOFc) | [Dataset Trimming](#dataset-trimming) | Trim dataset based on conditions | dataset.get_idxs("active"), dataset.trim(idxs=...) |
# | | | [Uniform](#uniform) | Uniform user sampling | dataset.get_uniform_idxs() |
# | | | [Rows and Columns](#rows-and-columns) | Select users by row/col | dataset.get_row_idxs(), dataset.get_col_idxs() |
# | | | [Linear](#linear) | Linear user placement | dataset.get_idxs("linear", ...) |
# | | | [Rectangular Zones](#rectangular-zones) | Filtering in 3D bounding boxes | dm.get_idxs_with_limits() |
# | [Beamforming](#beamforming) | [Video](https://youtu.be/IPVnIW2vGLE) | [Computing Beamformers](#computing-beamformers) | Calculate received power with beamforming | dm.steering_vec() |
# | | | [Visualization](#beamforming-visualization) | Beamforming visualization methods | dm.plot_beamforming() |
# | [Convert to DeepMIMO](#convert-to-deepmimo) | [Video](https://youtu.be/kXY2bMWeDgg) | [From Wireless InSite](#from-wireless-insite) | Conversion from Wireless InSite | dm.convert() |
# | | | [From Sionna RT](#from-sionna-rt) | Conversion from Sionna RT | sionna_exporter() |
# | | | [From AODT](#from-aodt) | Conversion from AODT | aodt_exporter() |
# | [Upload to DeepMIMO](#upload-to-deepmimo) | [Video](https://youtu.be/tNF6TN_ueU4) | [Upload Scenario](#upload-scenario) | Upload dataset to DeepMIMO database | dm.upload() |
# | | | [Upload Images](#upload-images) | Upload additional scenario images | dm.upload_images() |
# | | | [Upload Raytracing Source](#upload-raytracing-source) | Upload RT simulation source files | dm.upload_rt_source() |
#

# %% [markdown]
# ## Migrating from v3

# %% [markdown]
# The DeepMIMO API changed drastically since version 3 for more efficient storage
# and processing. And to allow for the inclusion of more powerful tools, not only
# to manage, analyze, and operate on ray tracing datasets, but also to let users
# conveniently download, upload, and convert their own ray tracer datasets. See
# below the differences between DeepMIMO v2/v3 (very similar) and version 4.

# %% [markdown]
# ### Generating v3 Dataset

# %%
# pip install deepmimov3

# %%
# DeepMIMO Scenario Download & Unzip
# Link from: deepmimo.net/scenarios/asu-campus-1
import subprocess

subprocess.run([
    "wget", "-O", "deepmimo_scen.zip",
    "https://www.dropbox.com/scl/fi/unldvnar22cuxjh7db2rf/ASU_Campus1.zip?"
    "rlkey=rs2ofv3pt4ctafs2zi3vwogrh&dl=0"
], check=False)
dm.unzip("deepmimo_scen.zip")

# %% [markdown]
# Note that the ASU scenario from v3 is 50% larger than that of v4. That is because
# v3 has structures of dictionaries with very small matrices inside, which introduce
# additional overhead and prevent the matlab compression to work well on those files.
# DeepMIMO v4 has simpler file formats where each file is a simple 2D matrix (except
# `inter_pos` which is 4D) that can be opened and used directly.

# %%
import DeepMIMOv3

# Load the default parameters
params_v3 = DeepMIMOv3.default_params()

# Print the default parameters

# %%
# Set your scenario path
params_v3["dataset_folder"] = "./deepmimo_scen"
params_v3["scenario"] = "asu_campus1"

# Set BSs and rows to generate
params_v3["active_BS"] = np.array([1])
params_v3["user_rows"] = np.arange(321)  # 321 x 411

# Set user and BS antennas
params_v3["ue_antenna"]["shape"] = np.array([1, 1])
params_v3["bs_antenna"]["shape"] = np.array([8, 1])  ## Horizontal, Vertical

# Enable only user-to-BS channels
params_v3["enable_BS2BS"] = False

# Start generation
dataset_v3 = DeepMIMOv3.generate_data(params_v3)

# %% [markdown]
# ### Generating v4 Dataset

# %%
dataset = dm.load("asu_campus_3p5")

ch_params = dm.ChannelParameters()
ch_params.bs_antenna.shape = [8, 1]

# for v3 compatibility
ch_params.ofdm.bandwidth = 50e6
ch_params.num_paths = 5

dataset.compute_channels(ch_params)
print(f"Channel parameters: \n{ch_params}")

# %% [markdown]
# ### Comparing v3 & v4

# %% [markdown]
# #### Channel Generation

# %% [markdown]
# There are considerable differences in v4:
# - **10x faster** loading, thanks to the efficient matrix format
# - **2.5x faster** channel generation is vectorized leveraging matrices
# - Data can be accessed as attributes in v4
# - Some parameters changed their default values for more practical parameter
#   settings. These include the:
#   - number of paths (5 in v3 vs 25 in v4)
#   - bandwidth (50 MHz in v3 vs 10 MHz in v4)
#   - bandwidth units (GHz in v3 vs Hz in v4)
#   - no. user antenna elements (`[4,2]` in v3 vs `[1,1]` in v4)
#   - number of rows to generate (1 in v3 vs all in v4)
#   - BS2BS channels (yes vs no by default)
#   - User sampling (like row selection) and FoV moved to separate functions.
#     This allowed for better dataset consistency and organization. More
#     information further down.
#
# Importantly, for the same configurations, the channels *match exactly*.

# %%
ch_v3 = dataset_v3[0]["user"]["channel"]
ch_v4 = dataset.channel

print(f"v3 channels.shape = {ch_v3.shape}")
print(f"v4 channels.shape = {ch_v4.shape}")

# %%
mean_abs_diff = np.mean(np.abs(ch_v3 - ch_v4))
mean_abs_v3 = np.mean(np.abs(ch_v3))
mean_abs_v4 = np.mean(np.abs(ch_v4))

print(f"Mean absolute difference: {mean_abs_diff}")
print(f"Mean absolute v3: {mean_abs_v3}")
print(f"Mean absolute v4: {mean_abs_v4}")

# %% [markdown]
# #### Data Access and Format

# %% [markdown]
# The dimensions and data access also changed. Version 3 is heavily nested while
# version 4 is extremely flat. See below the data structures of the v3 and the v4
# datasets, and the pros and cons for wireless applications using ray tracing
# datasets.

# %%
print("--- V3 Dataset Structure ---")

print(f"dataset.keys(): {list(dataset_v3[0].keys())}")
print(f'dataset["basestation"]: {dataset_v3[0]["basestation"]}')
print(f'dataset["location"]: {dataset_v3[0]["location"]}')
print(f'dataset["user"].keys(): {list(dataset_v3[0]["user"].keys())}')
print(f'dataset["user"]["location"].shape: {dataset_v3[0]["user"]["location"].shape}')
print(f'dataset["user"]["channel"].shape: {dataset_v3[0]["user"]["channel"].shape}')
print(f'dataset["user"]["distance"].shape: {dataset_v3[0]["user"]["distance"].shape}')
print(f'dataset["user"]["pathloss"].shape: {dataset_v3[0]["user"]["pathloss"].shape}')
print(f'dataset["user"]["LoS"].shape: {dataset_v3[0]["user"]["LoS"].shape}')
print(f'len(dataset["user"]["paths"]): {len(dataset_v3[0]["user"]["paths"])}')
print(f'dataset["user"]["paths"][0].keys():\n{list(dataset_v3[0]["user"]["paths"][0].keys())}')

# %%
import textwrap

print("--- V4 Dataset Structure ---")

keys_v4 = [key for key in dataset if not key.startswith("_")]

# Group keys by type (filter out callable methods)
array_keys = [
    key for key in keys_v4
    if not callable(getattr(dataset, key, None)) and isinstance(dataset[key], np.ndarray)
]
dict_keys = [
    key for key in keys_v4
    if not callable(getattr(dataset, key, None)) and hasattr(dataset[key], "keys")
]
other_keys = [
    key for key in keys_v4
    if not callable(getattr(dataset, key, None))
    and key not in array_keys + dict_keys
]

# Print numpy arrays
print("\n=== Numpy Arrays ===")
for key in array_keys:
    print(f"dataset.{key}.shape: {dataset[key].shape}")

# Print dictionaries
print("\n=== Dictionaries ===")
for key in dict_keys:
    keys = dataset[key].keys()
    print(f"dataset.{key}: DotDict with {len(keys)} keys:")
    print("\t" + textwrap.fill(str(list(keys)), width=80, subsequent_indent="\t"))

# Print other types
print("\n=== Other Types ===")
for key in other_keys:
    print(f"dataset.{key}: {dataset[key]}")

# %% [markdown]
# **Why the change into v4 format**
# - All path data is stored in aligned NumPy arrays → enables fast slicing,
#   filtering, and batching.
# - Flat structure avoids deep indexing → makes code for ML training, plotting,
#   and analysis shorter and clearer.
# - Efficient memory layout and easier I/O → good for large datasets and GPU
#   pipelines.
# - Compatible with vectorized NumPy and PyTorch/TensorFlow operations.
# - Metadata (e.g., ray tracing settings, scene info) is easily accessible at
#   the top level.
# - Best for debugging and inspecting datasets interactively.
# - Ideal for single-modal data with uniform UE structures.
#
# **When v3 structure is better**
# - Multi-modal or per-user datasets like DeepVerse and DeepSense → easier to
#   manage heterogeneity.
# - Useful when each user/entity has unique fields or variable numbers of data
#   elements.
#
# The need for speed and simplicity with the ever growing ray tracing datasets
# required new tools to adapt. For DeepMIMO, this meant a complete redesign of
# the dataset format and backend processing. We have witnessed many advantages
# and we continuously expand this toolchain to fit the wireless community.
# Please let us know if you find places of improvement.

# %% [markdown]
# #### User Selection

# %% [markdown]
# **Decoupling User Sampling in v4**
#
# User sampling was separated to enable data-dependent selection.
# - Sampling is decoupled to allow user selection based on the dataset's
#   internal information (e.g., user positions, row/col density, active paths,
#   path types, etc.).
# - Since this information is only available **after loading**, sampling must
#   be done **after** the initial `load()` step.
# - This enables flexible user control — e.g., trimming users in a bounding box,
#   or selecting a regular grid subset.
#
# > In contrast, **Field of View (FoV)** trimming was separated primarily for
# > **data integrity**: to ensure that the available paths in the dataset match
# > those used for channel generation.
#
# ----
#
# To further enable separation of concerns, DeepMIMO v4 introduces a clean
# 3-step workflow:
#
# 1. **Loading**: `deepmimo.load()` uses `load_params` to load full path data
#    and metadata.
# 2. **Modification**: Optional user sampling, path filtering and dataset
#    trimming, done after loading, based on dataset contents.
# 3. **Channel Generation**: `deepmimo.compute_channels()` generates the channel
#    matrix. It takes an optional `dm.ChannelParameters()` object containing
#    parameters that **only affect the channel computation** (e.g., number of
#    antennas, polarization, combining, etc.).
#
# This decoupling allows users to first inspect and modify the dataset — for
# instance, selecting users based on position or number of active paths — before
# computing any channel data.
#
# The result is **more user sampling functions in DeepMIMOv4**:
#
# - `dataset.get_idxs("row", row_idxs=...)`
# - `dataset.get_idxs("col", col_idxs=...)`
# - `dataset.get_idxs("uniform", steps=...)`
# - `dataset.get_idxs("limits", x_min=..., x_max=..., ...)`
# - `dataset.get_idxs("active")`
# - `dataset.get_idxs("linear", start_pos=..., end_pos=..., n_steps=...)`
#
# These functions return user indices for trimming. Only the first function is
# supported in DeepMIMOv3.
#
# Below we see the differences between both versions to obtain rows between 40 and 50.

# %%
# User sampling in v3
params_v3_sampling = DeepMIMOv3.default_params()
# Set your scenario path
params_v3_sampling["dataset_folder"] = "./deepmimo_scen"
params_v3_sampling["scenario"] = "asu_campus1"

# Set BSs and rows to generate
params_v3_sampling["active_BS"] = np.array([1])
params_v3_sampling["user_rows"] = np.arange(40, 50)  # 321 x 411

# Enable only user-to-BS channels
params_v3_sampling["enable_BS2BS"] = False

# Start generation
dataset_v3_sampling = DeepMIMOv3.generate_data(params_v3_sampling)
print(f"\nNumber of users: {len(dataset_v3_sampling[0]['user']['LoS'])}")

# %%
# User sampling in v4
idxs = dataset.get_idxs("row", row_idxs=range(40, 50))
dataset_t = dataset.trim(idxs=idxs)
_ = dataset_t.n_ue
_ = dataset_t.channel.shape

# %% [markdown]
# #### Others
#
# - Field of View (FoV) filtering also moved from the parameters to its own
#   function `dataset.trim_by_fov(bs_fov, ue_fov)` or `dataset.trim(bs_fov=..., ue_fov=...)`.
#   The reason for this is to maintain maximum dataset consistency and integrity.
#   If FoV was in the channel parameters, it would affect the channel but then
#   the data inside the dataset would not match the data in the channel. Some
#   matrices would change but not others, very likely creating errors. Because of
#   that, FoV filtering returns a new dataset with paths outside the FoV removed.
#   For more information, see the FoV reference example and APIs.
# - Examples of Sionna Adapter using O1 scenario with v3 and v4:
#   - [v3 Sionna Adapter](https://github.com/NVlabs/sionna/blob/main/tutorials/phy/DeepMIMO.ipynb)
#   - [v4 Sionna Adapter](https://github.com/jmoraispk/sionna/blob/DeepMIMOv4_sionna_adapter/tutorials/phy/DeepMIMO.ipynb)

# %% [markdown]
# ## Install DeepMIMO
#

# %% [markdown]
# ### Python (pip)

# %% [markdown]
# The best way to use python is to use virtual environments. We recommend
# miniforge (smaller version of conda) to manage such environments.
#
# Here are the steps to create a Python environment with the DeepMIMO package:
#
# 1. Install **[miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install)**
#    (tip: select init now & then `conda config --set auto_activate_base false`)
# 2. In windows, open miniforge prompt from the start menu. In Linux/Mac, source
#    bash or restart the terminal.
# 3. Run the following commands to create and activate an environment:
#   1. `mamba create -n deepmimo_env python=3.11 expat=2.5.0`
#   2. `mamba activate deepmimo_env`
# 4. Install DeepMIMO:
#   - For Users: `pip install --pre deepmimo`
#   - For Developers: clone [DeepMIMO](https://github.com/DeepMIMO/DeepMIMO),
#     go into folder, `pip install -e .`

# %% [markdown]
# ### Matlab
#
# The way DeepMIMO "supports" execution in Matlab is via
# [Python in Matlab](https://www.mathworks.com/help/matlab/call-python-libraries.html).
#
# Check [MATLAB's Python Version Compatibility Table]
# (https://www.mathworks.com/support/requirements/python-compatibility.html) to
# make sure your Python version is supported in Matlab. DeepMIMO recommends
# Python 3.11

# %% [markdown]
# **Step 1**: Configure python environment with the interpreter path:
# - `pyenv("Version", "C:\Users\joao\mambaforge\envs\deepmimo_env\python.exe")`
#   to setup the interpreter in Matlab
# - `pyenv("ExecutionMode","OutOfProcess")` to separate the python and Matlab
#   processes - helps reduce library collisions (e.g. needed for plots)
#
# ***Tip***: To get the interpreter path, open the terminal, navigate to where
# python could be called (activate the environment if needed) and get the path
# via:
# - `where python` on Windows
# - `which python` on Linux & MacOS
#
#
# **Step 2**: Install DeepMIMO via Matlab (if not in current python environment yet):
# * `pipinstall("deepmimo")`
#
# **Step 3**: Run DeepMIMO in Matlab via one of the 3 options below.
#
# Option 1: Run file
# * `[channels, los] = pyrunfile("deepmimo_examples.py”, [channels, los])`
#
# Option 2: Run function from file
# * `out = pyrunfile("deepmimo_examples.py”, "get_chs_and_los()")`
# * `channels = out{1}`
#
# Option 3: Run individual lines of code
# * `pyrun("import deepmimo as dm")`
# * `pyrun("dataset = dm.load('asu_campus_3p5')")`
# * `py_chs = pyrun("chs = dataset.compute_channels()", "chs")`
#
# Note: You have to convert/cast variables that come from `pyrun` into matrices
# with a certain type. See a full example below:

# %% [markdown]
# ```python
# # Example code to run in Matlab
# pyrun("import deepmimo as dm")
# pyrun("dm.download('asu_campus_3p5')")
# pyrun("dataset = dm.load('asu_campus_3p5')")
# py_chs = pyrun("chs = dataset.compute_channels()", 'chs')
# chs = double(py_chs); % cast to complex array
# pyrun("import matplotlib.pyplot as plt")
# pyrun("dataset.los.plot(); plt.show()")
# ```

# %% [markdown]
# #### Troubleshooting MATLAB
#
# Sometimes, the environment may not install or force a wrong version of expat
# in python. This is needed by matlab. In that case, run with conda/mamba:
#
# 1. Remove expat cleanly
# `mamba remove expat --yes`
#
# 2. Reinstall compatible version
# `mamba install expat=2.5.0 --yes`
#
# 3. Reinstall compatible python (if previous command didn't)
# `mamba install python=3.11 --yes`
#

# %% [markdown]
# ## Load Dataset

# %% [markdown]
# ### Simple Load
#

# %%
import deepmimo as dm

scen_name = "asu_campus_3p5"
macro_dataset = dm.load(scen_name)

# %% [markdown]
# ### Detailed Load

# %%
city_scen_name = "city_0_newyork_3p5"
dm.download(city_scen_name)  # just to avoid prompting the user during load

tx_sets_dict = {3: [0]}  # Load first points from set 1
rx_sets_dict = {0: np.arange(10)}  # Load first 10 points from set 4

# Example 1: Load specific points of specific TX/RX sets using dictionaries
# (& limit paths and matrices)
dataset1 = dm.load(
    city_scen_name,
    tx_sets=tx_sets_dict,
    rx_sets=rx_sets_dict,
    matrices=["aoa_az", "aoa_el", "inter_pos", "inter"],
    max_paths=4,
)

# %%
# Example 2: Load all points of specific TX/RX sets using lists
dataset2 = dm.load(city_scen_name, tx_sets=[1], rx_sets=[2])

# %%
# Example 3: Load all TX/RX sets
dataset3 = dm.load(city_scen_name, tx_sets="all", rx_sets="all")
# This includes receiving BSs. By default rx_sets='rx_only', ignoring RX BSs

# %%
help(dm.load)

# %% [markdown]
# ## Scenario Information

# %% [markdown]
# ### Summary

# %%
# Like the information present in the scenario webpage
dm.summary("city_0_newyork_3p5")

# %%
dm.download("city_0_newyork_3p5_s")
try:
    dm.plot_summary("city_0_newyork_3p5_s")
except AttributeError as e:
    print(f"Plot summary skipped due to error: {e}")

# %%
dm.info()

# %% [markdown]
# ### Transmitters and Receivers

# %%
# Get all available TX-RX sets
txrx_sets = dm.get_txrx_sets(scen_name)
print(txrx_sets)

# %%
# Get all available TX to RX set pairs (pairs of rx sets to txs, not tx sets!)
pairs = dm.get_txrx_pairs(txrx_sets)
print(pairs)

# %%
dm.print_available_txrx_pair_ids(scen_name)

# %%
# (tx-set, tx, rx-set) IDs of the loaded matrices
print(dataset.txrx)

# %% [markdown]
# ### Ray Tracing Parameters

# %%
# This information is present in the scenario table and can be used to search and filter.
# (soon in dm.search())

from pathlib import Path

# Get all available scenarios
scenarios = dm.get_available_scenarios()
print(f"Found {len(scenarios)} scenarios\n")

for scen_name in scenarios:
    params_json_path = dm.get_params_path(scen_name)

    # Skip if params file doesn't exist
    if not Path(params_json_path).exists():
        print(f"Skipping {scen_name} - no params file found")
        continue

    params_dict = dm.load_dict_from_json(params_json_path)
    rt_params = params_dict[dm.consts.RT_PARAMS_PARAM_NAME]

    # Calculate sums
    max_reflections = rt_params[dm.consts.RT_PARAM_MAX_REFLECTIONS]
    max_diffractions = rt_params[dm.consts.RT_PARAM_MAX_DIFFRACTIONS]
    total_interactions = max_reflections + max_diffractions

    print(f"\nScenario: {scen_name}")
    print(f"Max Reflections: {max_reflections}")
    print(f"Max Diffractions: {max_diffractions}")
    print(f"Total Interactions: {total_interactions}")

# %% [markdown]
# ## Visualization

# %% [markdown]
# ### Coverage Maps

# %%
dm.info()

# %%
help(dm.plot_coverage)

# %%
main_keys = ["aoa_az", "aoa_el", "aod_az", "aod_el", "delay", "power", "phase", "los", "num_paths"]
NDIM_TWO = 2

cbar_lbls = [
    "azimuth of arrival (º)",
    "elevation of arrival (º)",
    "azimuth of departure  (º)",
    "elevation of departure (º)",
    "delay (s)",
    "power (dBW)",
    "phase (º)",
    "line-of-sight status",
    "number of paths",
]

for key in main_keys:
    plt_var = dataset[key][:, 0] if dataset[key].ndim == NDIM_TWO else dataset[key]
    # Example: dm.plot_coverage(dataset.rx_pos, plt_var, bs_pos=dataset.tx_pos.T,
    #                          title=key, cbar_title=cbar_lbls[main_keys.index(key)])
    dataset.plot_coverage(plt_var, title=key, cbar_title=cbar_lbls[main_keys.index(key)])
    break

# %%
# 3D version
dm.plot_coverage(
    dataset.rx_pos,
    dataset["los"],
    bs_pos=dataset.tx_pos.T,
    bs_ori=dataset.tx_ori,
    title="LoS",
    cbar_title="LoS status",
    proj_3D=True,
    scat_sz=0.1,
)

# %%
# Another shorter way of plotting
dataset.aoa_az.plot()
dataset.aoa_az.plot(path_idx=3)  # same as dataset.aoa_az[:,3].plot()
dataset.inter.plot(path_idx=3, interaction_idx=1)  # plot inter[:, 3, 1]

# %% [markdown]
# ### Rays

# %%
u_idx = np.where(dataset.los == 1)[0][100]
dataset.plot_rays(u_idx, proj_3D=False, dpi=100)

# %% [markdown]
# ### Path Plots
# Note: For simplicity, the analysis is restricted to the **main path**.

# %% [markdown]
# #### Percentage of the Power
#

# %%
pwr_in_first_path = dataset.lin_pwr[:, 0] / np.nansum(dataset.lin_pwr, axis=-1) * 100

dm.plot_coverage(
    dataset.rx_pos,
    pwr_in_first_path,
    bs_pos=dataset.tx_pos.T,
    title="Percentage of power in 1st path",
    cbar_title="Percentage of power [%]",
)

# %% [markdown]
# #### Number of Interactions
#

# %%
dm.plot_coverage(
    dataset.rx_pos,
    dataset.num_interactions[:, 0],
    bs_pos=dataset.tx_pos.T,
    title="Number of interactions in 1st path",
    cbar_title="Number of interactions",
)

# %% [markdown]
# #### First Interaction Type
#

# %%
dataset.inter_str[10]

# %%
first_bounce_codes = [
    code[0] if code else "" for code in dataset.inter_str[:, 0]
]  # 'n', '2', '1', ...

unique_first_bounces = ["n", "", "R", "D", "S"]

coded_data = np.array([unique_first_bounces.index(code) for code in first_bounce_codes])

viridis_colors = plt.cm.viridis(np.linspace(0, 1, 4))  # Get 4 colors from viridis

dm.plot_coverage(
    dataset.rx_pos,
    coded_data,
    bs_pos=dataset.tx_pos.T,
    title="Type of first bounce of first path",
    cmap=["white", *viridis_colors.tolist()],  # white for 'n'
    cbar_labels=["None", "LoS", "R", "D", "S"],
)

# %% [markdown]
# #### Full Bounce Profile
#

# %%
# Full bounce profile visualization
unique_profiles = np.unique(dataset.inter_str[:, 0])
print(f"\nUnique bounce profiles found: {unique_profiles}")

# Create mapping for full profiles
profile_to_idx = {profile: idx for idx, profile in enumerate(unique_profiles)}
full_profile_data = np.array([profile_to_idx[profile] for profile in dataset.inter_str[:, 0]])

# Create colormap with white for no interaction and viridis colors for the rest
n_profiles = len(unique_profiles)
viridis = plt.cm.viridis(np.linspace(0, 1, n_profiles - 1))  # Get colors for the rest

# Create decoded labels for the colorbar
profile_labels = ["-".join(p) if p else "LoS" for p in unique_profiles]

# Plot the full bounce profiles
dm.plot_coverage(
    dataset.rx_pos,
    full_profile_data,
    bs_pos=dataset.tx_pos.T,
    title="Full bounce profile of first path",
    cmap=[*viridis.tolist(), "white"],
    cbar_labels=profile_labels,
)

# %% [markdown]
# ### Plot Overlays

# %% [markdown]
# DeepMIMO's main plot functions are `plot_coverage()`, `plot_rays()` and
# `plot_scene()` (available via `dataset.scene.plot()`). These can be composed
# / overlayed on top of each other to obtain more insightful visualizations.

# %% [markdown]
# #### 2D Scene, Coverage & Rays Overlay

# %%
ax = dataset.scene.plot(proj_3D=False, figsize=(8, 4), dpi=150)
dataset.power.plot(ax=ax)
dataset.plot_rays(90385, ax=ax, proj_3D=False)  # one los user
ax.legend().set_visible(False)

# %% [markdown]
# #### 3D Scene & Rays Overlay

# %% [markdown]
# In 3D the overlay of coverage does not work well because DeepMIMO's plotting
# backend (Matplotlib) has severe limitations with rendering depth in 3D. More
# capable backends will become available in the future.

# %%
ax = dataset.plot_rays(365)  # another los user
dataset.scene.plot(ax=ax)
ax.set_zlim((-40, 50))
ax.legend().set_visible(False)
ax.view_init(elev=40, azim=-85)

# %% [markdown]
# ## Channel Generation

# %% [markdown]
# ### Parameters

# %%
dm.ChannelParameters()

# %%
# Create channel parameters with all options
ch_params = dm.ChannelParameters()

# Antenna parameters

# Base station antenna parameters
ch_params.bs_antenna.rotation = np.array([0, 0, 0])  # [az, el, pol] in degrees
ch_params.bs_antenna.fov = np.array([360, 180])  # [az, el] in degrees
ch_params.bs_antenna.shape = np.array([8, 1])  # [horizontal, vertical] elements
ch_params.bs_antenna.spacing = 0.5  # Element spacing in wavelengths

# User equipment antenna parameters
ch_params.ue_antenna.rotation = np.array([0, 0, 0])  # [az, el, pol] in degrees
ch_params.ue_antenna.fov = np.array([360, 180])  # [az, el] in degrees
ch_params.ue_antenna.shape = np.array([1, 1])  # [horizontal, vertical] elements
ch_params.ue_antenna.spacing = 0.5  # Element spacing in wavelengths

# Channel parameters
ch_params.freq_domain = True  # Whether to compute frequency domain channels
ch_params.num_paths = 25  # Number of paths
ch_params.doppler = False  # Whether to add Doppler to the channels

# OFDM parameters
ch_params.ofdm.bandwidth = 10e6  # Bandwidth in Hz
ch_params.ofdm.subcarriers = 512  # Number of subcarriers
ch_params.ofdm.selected_subcarriers = np.arange(1)  # Which subcarriers to generate
ch_params.ofdm.rx_filter = 0  # Receive Low Pass / ADC Filter

# Generate channels
dataset.compute_channels(ch_params)
print(f"Shape of channel matrix: {dataset.channel.shape}")

# %%
dm.info("channel")

# %%
dm.info("ch_params")

# %% [markdown]
# Below is a brief summary for the parameters. For more details, see [DeepMIMOv3 page](https://www.deepmimo.net/versions/deepmimo-v3-python/).
#
# | Parameter              | Default Value          | Description                                                                           |
# |------------------------|------------------------|---------------------------------------------------------------------------------------|
# | **doppler**     | 0                          | Enable Doppler shift                                |
# | **num_paths**          | 25                          | Number of maximum paths                                                            |
# | **OFDM_channels**      | 1                          | Generate OFDM (True) or time domain channels (False)                               |
# | | | |
# | **OFDM**               |                            | OFDM parameters (only applies if OFDM_channels is True)                            |
# |   - **subcarriers**    | 512                        | Total number of subcarriers                                                        |
# |   - **selected_subcarriers** | [0]                  | Subcarriers to be generated                                                        |
# |   - **bandwidth**      | 0.05                       | Bandwidth                                                                          |
# |   - **RX_filter**      | 0                          | Receive filter                                                                     |
# | | | |
# | **bs_antenna**/**ue_antenna**         |                            | BS/UE antenna properties                                            |
# |   - **radiation_pattern** | isotropic              | Radiation pattern applied to the antenna, in ['isotropic', 'halfwave-dipole']       |
# |   - **rotation**       | [0, 0, 0]                  | Rotation of the antenna - in compliance with 38.901                                |
# |   - **shape**          | [8, 1]                     | UPA panel shape in the shape of (horizontal elements, vertical elements)           |
# |   - **spacing**        | 0.5                        | Antenna spacing                                                                    |

# %% [markdown]
# ### Time Domain

# %%
# Channel computation parameters
ch_params.freq_domain = False  # Whether to compute frequency domain channels

dataset.compute_channels(ch_params)
print(f"Shape of channel matrix: {dataset.channel.shape}")  # as many taps as paths

# %%
# Plot CIR
user_idx = np.where(dataset.n_paths > 0)[0][0]
plt.figure(dpi=200)
plt.stem(dataset.delay[user_idx] * 10**6, dataset.power[user_idx], basefmt="none")
plt.xlabel("Time of arrival [us]")
plt.ylabel("Power per path [dBW]")
plt.grid()
plt.show()

# %% [markdown]
# ### Frequency Domain (OFDM)

# %%
ch_params = dm.ChannelParameters()

ch_params.num_paths = 5
ch_params.ofdm.bandwidth = 50e6
ch_params.ofdm.selected_subcarriers = np.arange(64)  # Which subcarriers to generate

channels = dataset.compute_channels(ch_params)

# Visualize channel magnitude response (NOTE: requires at >1 subcarriers and antennas)
user_idx = np.where(dataset.n_paths > 0)[0][0]
plt.imshow(np.abs(np.squeeze(channels[user_idx]).T))
plt.title("Channel Magnitude Response")
plt.xlabel("TX Antennas")
plt.ylabel("Subcarriers")
plt.show()

# NOTE: show the case of when there are too few subcarriers

# %%
# Plot CIR from channel
cir = np.fft.ifft(dataset.channels[user_idx, 0, 0, :])  # cir ant 0 of rx, ant 0 of tx
plt.plot(np.abs(cir))
plt.ylabel("CIR magnitude")
plt.xlabel("delays bins [us]")
delay_idxs = np.arange(len(cir))
delay_labels = delay_idxs / ch_params.ofdm.bandwidth * 1e6
ax = plt.gca()
n_xtickstep = 10
ax.set_xticks(delay_idxs[::n_xtickstep])
ax.set_xticklabels([f"{label:.1f}" for label in delay_labels[::n_xtickstep]])
plt.grid()
plt.show()

# %% [markdown]
# ### Doppler

# %% [markdown]
# Doppler can be added to the generated channels (in time or frequency domain)
# in three different ways:
# - Set Doppler directly: Set manually the Doppler frequencies per user (and
#   optionally, per path)
# - Set Speeds directly: Set manually the TX, RX or object speeds. This will
#   automatically compute Doppler Frequencies.
# - Set Time Reference: This will automatically compute, the TX, RX and object
#   speeds across scenes. Note that this method only works when using a Dynamic
#   Dataset.
#
# Note: For Doppler to be ADDED to the channel, the `enable_doppler` parameter
# must be set to `True` in the channel parameters
#

# %% [markdown]
# #### Set Doppler Directly

# %%
# Same Doppler shift for all users
dopplers1 = 10  # [Hz]
dataset.set_doppler(dopplers1)
dataset.compute_channels(dm.ChannelParameters(doppler=True))


# Different Doppler shift for different users
rng = np.random.default_rng()
dopplers2 = rng.integers(20, 51, size=(dataset.n_ue,))
dataset.set_doppler(dopplers2)
dataset.compute_channels(dm.ChannelParameters(doppler=True))


# Different Doppler shift for different users
dopplers3 = rng.integers(20, 51, size=(dataset.n_ue, dataset.max_paths))
dataset.set_doppler(dopplers3)
dataset.compute_channels(dm.ChannelParameters(doppler=True))

# %% [markdown]
# #### Set Speeds

# %%
# Set rx velocities manually (same for all users)
dataset.rx_vel = [5, 0, 0]  # (x, y, z) [m/s]

# Set rx velocities manually (different per users)
min_speed, max_speed = 0, 10
random_velocities = np.zeros((dataset.n_ue, 3))
rng = np.random.default_rng()
random_velocities[:, :2] = rng.uniform(min_speed, max_speed, size=(dataset.n_ue, 2))
dataset.rx_vel = random_velocities
# Note: z = 0 to assume users are always at ground level

# Set tx velocities manually
dataset.tx_vel = [0, 0, 0]

# Set object velocities manually
dataset.set_obj_vel(obj_idx=[1, 3, 6], vel=[[0, 5, 0], [0, 5, 6], [0, 0, 3]])
# Note: these object indices should match the indices/ids of the objects in
#       dataset.scene.objects

dataset.compute_channels(dm.ChannelParameters(doppler=True))

# %% [markdown]
# #### Set Scene Timestamps

# %%
# NOTE: requires a dynamic dataset to test. Currently there are no Dynamic Datasets
dyn_dataset = None
if dyn_dataset is not None:
    # Uniform snapshots
    dyn_dataset.set_timestamps(10)  # [seconds between scenes]

    print(f"timestamps: {dyn_dataset.timestamps}")
    print(f"rx_vel: {dyn_dataset.rx_vel}")
    print(f"tx_vel: {dyn_dataset.tx_vel}")
    print(f"obj_vel: {[obj.vel for obj in dyn_dataset.scene.objects]}")

    # Non-uniform snapshots
    times = [0, 1.5, 2.3, 4.4, 5.8, 7.1, 8.9, 10.2, 11.7, 13.0]
    dyn_dataset.set_timestamps(times)  # [timestamps of each scene]

    print(f"timestamps: {dyn_dataset.timestamps}")
    print(f"rx_vel: {dyn_dataset.rx_vel}")
    print(f"tx_vel: {dyn_dataset.tx_vel}")
    print(f"obj_vel: {[obj.vel for obj in dyn_dataset.scene.objects]}")

# %% [markdown]
# ## Basic Operations
#

# %% [markdown]
# ### Line-of-Sight Status

# %%
dm.info("los")

# %%
print(f"Shape of LOS matrix: {dataset.los.shape}")

# %%
dataset.plot_coverage(dataset.los)

# %%
active_mask = dataset.num_paths > 0
print(f"\nNumber of active positions: {np.sum(active_mask)}")
print(f"Number of inactive positions: {np.sum(~active_mask)}")

# Create scatter plot showing active vs inactive positions
plt.figure(figsize=(8, 6))
plt.scatter(
    dataset.rx_pos[~active_mask, 0],
    dataset.rx_pos[~active_mask, 1],
    alpha=0.5,
    s=1,
    c="red",
    label="Inactive",
)
plt.scatter(
    dataset.rx_pos[active_mask, 0],
    dataset.rx_pos[active_mask, 1],
    alpha=0.5,
    s=1,
    c="green",
    label="Active",
)
plt.legend()
plt.show()

# dm.plot_coverage(dataset['rx_pos'], dataset.los != -1, cmap=['red', 'green'])

# %%
dm.plot_coverage(dataset["rx_pos"], dataset.los != -1, cmap=["red", "green"])

# %% [markdown]
# ### Pathloss

# %%
non_coherent_pathloss = dataset.compute_pathloss(coherent=False)
coherent_pathloss = dataset.compute_pathloss(coherent=True)  # default

_, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
dataset.plot_coverage(non_coherent_pathloss, title="Non-Coherent pathloss", ax=axes[0])
dataset.plot_coverage(coherent_pathloss, title="Coherent pathloss", ax=axes[1])

# %% [markdown]
# ### Implicit Computations

# %%
# Implicit and lazy computations
# Functions are public when arguments are needed

# Public compute functions
_ = dataset.channels  # calls dataset.compute_channels()
_ = dataset.pathloss  # calls dataset.compute_pathloss()

# Hidden compute functions
_ = dataset.distance  # calls dataset._compute_distances()
_ = dataset.num_paths  # calls dataset._compute_num_paths()
_ = dataset.num_interactions  # calls dataset._compute_num_interactions()
_ = dataset.los  # calls dataset._compute_los()
_ = dataset.n_ue  # calls dataset._compute_n_ue()
_ = dataset.grid_size  # calls dataset._compute_grid_info()
_ = dataset.grid_spacing  # calls dataset._compute_grid_info()

# %% [markdown]
# ### Aliases

# %%
checks = [
    dataset.pwr is dataset.power,
    dataset.pl is dataset.pathloss,
    dataset.ch is dataset.channels,
    dataset.ch_params is dataset.channel_params,
    dataset.n_paths is dataset.num_paths,
    dataset.aoa_phi is dataset.aoa_az,
    dataset.bs_pos is dataset.tx_pos,
    dataset.toa is dataset.delay,
]

for check in checks:
    print(check)

# %% [markdown]
# <table>
#     <tr>
#         <th>Original</th>
#         <th>Aliases</th>
#     </tr>
#     <tr><td rowspan="1">los</td><td>los_status</td></tr>
#     <tr><td rowspan="3">channel</td><td>ch</td></tr>
#     <tr><td>chs</td></tr>
#     <tr><td>channels</td></tr>
#     <tr><td rowspan="1">ch_params</td><td>channel_params</td></tr>
#     <tr><td rowspan="2">power</td><td>pwr</td></tr>
#     <tr><td>powers</td></tr>
#     <tr><td rowspan="3">pwr_linear</td><td>lin_pwr</td></tr>
#     <tr><td>linear_power</td></tr>
#     <tr><td>pwr_lin</td></tr>
#     <tr><td rowspan="1">pwr_linear_ant_gain</td><td>pwr_ant_gain</td></tr>
#     <tr><td rowspan="4">rx_pos</td><td>ue_pos</td></tr>
#     <tr><td>rx_loc</td></tr>
#     <tr><td>rx_position</td></tr>
#     <tr><td>rx_locations</td></tr>
#     <tr><td rowspan="4">tx_pos</td><td>bs_pos</td></tr>
#     <tr><td>tx_loc</td></tr>
#     <tr><td>tx_position</td></tr>
#     <tr><td>tx_locations</td></tr>
#     <tr><td rowspan="2">pathloss</td><td>pl</td></tr>
#     <tr><td>path_loss</td></tr>
#     <tr><td rowspan="2">distance</td><td>dist</td></tr>
#     <tr><td>dists</td></tr>
#     <tr><td rowspan="2">aoa_az</td><td>aoa_phi</td></tr>
#     <tr><td>aoa_theta</td></tr>
#     <tr><td rowspan="2">aod_az</td><td>aod_phi</td></tr>
#     <tr><td>aod_theta</td></tr>
#     <tr><td rowspan="1">num_paths</td><td>n_paths</td></tr>
#     <tr><td rowspan="2">delay</td><td>toa</td></tr>
#     <tr><td>time_of_arrival</td></tr>
#     <tr><td rowspan="2">interactions</td><td>bounce_type</td></tr>
#     <tr><td>interactions</td></tr>
#     <tr><td rowspan="3">interactions_pos</td><td>bounce_pos</td></tr>
#     <tr><td>interaction_positions</td></tr>
#     <tr><td>interaction_locations</td></tr>
# </table>

# %% [markdown]
# ### Access Attributes

# %%
for var_name in ["pl", "rx_pos", "aoa_az", "channel"]:
    a = dataset[var_name]
    b = getattr(dataset, var_name)
    print(f"dataset['{var_name}'] == dataset.{var_name}: {a is b}")

# %% [markdown]
# ### Antenna Rotation

# %% [markdown]
# #### Azimuth

# %%
params = dm.ChannelParameters()

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)

# Define 3 different rotations to show
rotations = [
    np.array([0, 0, 0]),  # Facing +x
    np.array([0, 0, 180]),  # Facing -x
    np.array([0, 0, -135]),
]  # Facing 45º between -x and -y

titles = [
    "Orientation along +x (0°)",
    "Orientation along -x (180°)",
    "Orientation at 45º between -x and -y (-135°)",
]

# Plot each azimuth rotation
for i, (rot, title) in enumerate(zip(rotations, titles, strict=False)):
    # Update channel parameters with new rotation
    params.bs_antenna.rotation = rot
    dataset.set_channel_params(params)  # safest way to set params

    # Create coverage plot in current subplot
    dm.plot_coverage(
        dataset.rx_pos,
        dataset.los,
        bs_pos=dataset.tx_pos.T,
        bs_ori=dataset.tx_ori,
        ax=axes[i],
        title=title,
        cbar_title="LoS status",
    )

# %% [markdown]
# #### Elevation

# %%
params = dm.ChannelParameters()

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": "3d"}, tight_layout=True)

# Define 3 different rotations to show
rotations = [
    np.array([0, 0, -180]),  # Facing -x
    np.array([0, 30, -180]),  # Facing 30º below -x in XZ plane
    np.array([0, 60, -180]),
]  # Facing 60º below -x in XZ plane

titles = [
    "Orientation along -x (180°)",
    "Orientation at 30º between -x and -z",
    "Orientation at 60º between -x and -z",
]

# Plot each azimuth rotation
for i, (rot, title) in enumerate(zip(rotations, titles, strict=False)):
    # Update channel parameters with new rotation
    params.bs_antenna.rotation = rot
    dataset.set_channel_params(params)

    # Create coverage plot in current subplot
    dataset.plot_coverage(
        dataset.los,
        proj_3D=True,
        ax=axes[i],
        title=title,
        cbar_title="LoS status",
    )
    axes[i].view_init(elev=5, azim=-90)  # Set view to xz plane to see tilt
    axes[i].set_yticks([])  # Remove y-axis ticks to unclutter the plot

# %% [markdown]
# ## Advanced Operations

# %% [markdown]
# ### Field-of-View

# %% [markdown]
# #### Azimuth

# %%
# First plot with no FoV filtering (full coverage)
dataset.plot_coverage(dataset.los)

# %%
params = dm.ChannelParameters()
params["bs_antenna"]["rotation"] = np.array([0, 0, -135])
dataset.set_channel_params(params)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)

# Define 3 FoV
fovs = [
    np.array([180, 180]),  # Facing -x
    np.array([90, 180]),  # Facing 30º below -x in XZ plane
    np.array([60, 180]),
]  # Facing 60º below -x in XZ plane

titles = [f"FoV = {fov[0]} x {fov[1]}°" for fov in fovs]

# Plot each FoV setting
for i, (fov, title) in enumerate(zip(fovs, titles, strict=False)):
    print(f"Iteration {i}: Setting FoV to {fov}")
    # Create a temporary dataset with FoV applied
    dataset_fov = dataset.trim_by_fov(bs_fov=fov)
    dataset_fov.plot_coverage(dataset_fov.los, ax=axes[i], title=title, cbar_title="LoS status")

# Note: trim_by_fov returns a new dataset with paths outside FoV removed
# The original dataset remains unchanged

# %% [markdown]
# #### Elevation

# %%
params = dm.ChannelParameters()
params["bs_antenna"]["rotation"] = np.array([0, 30, -135])
dataset.set_channel_params(params)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)

# Define 3 FoV
fovs = [
    np.array([360, 90]),  # Facing -x
    np.array([360, 45]),  # Facing 30º below -x in XZ plane
    np.array([360, 30]),
]  # Facing 60º below -x in XZ plane

titles = [f"FoV = {fov[0]} x {fov[1]}°" for fov in fovs]

# Plot each FoV setting
for i, (fov, title) in enumerate(zip(fovs, titles, strict=False)):
    print(f"Iteration {i}: Setting FoV to {fov}")
    # Create a temporary dataset with FoV applied
    dataset_fov = dataset.trim_by_fov(bs_fov=fov)
    dataset_fov.plot_coverage(dataset_fov.los, ax=axes[i], title=title, cbar_title="LoS status")

# Note: trim_by_fov returns a new dataset with paths outside FoV removed
# To see path information affected by fov, index arrays with: dataset.los != -1

# %% [markdown]
# ## Scene & Materials

# %% [markdown]
# ### Visualization

# %%
# Plot the full scene
dataset.scene.plot()

# %%
# Plot the scene with triangular faces
dataset.scene.plot(mode="tri_faces")

# %% [markdown]
# ### Operations

# %%
print("\nScene and Materials Example")
print("-" * 50)

scene = dataset.scene

# 1. Basic scene information
print("\nScene Overview:")
print(f"- Total objects: {len(scene.objects)}")

# Get objects by category
buildings = scene.get_objects(label="buildings")
terrain = scene.get_objects("terrain")
vegetation = scene.get_objects("vegetation")

print(f"- Buildings: {len(buildings)}")
print(f"- Terrain: {len(terrain)}")
print(f"- Vegetation: {len(vegetation)}")

# 2. Materials and Filtering
materials = dataset.materials

# Get materials used by buildings
building_materials = buildings.get_materials()
print(f"\nMaterials used in buildings: {building_materials}")

# Different ways to filter objects
print("\nFiltering examples:")

# Filter by label only
buildings = scene.get_objects(label="buildings")
print(f"- Buildings: {len(buildings)}")

# Filter by material only
material_idx = building_materials[0]
objects_with_material = scene.get_objects(material=material_idx)
print(f"- Objects with material {material_idx}: {len(objects_with_material)}")

# Filter by both label and material
buildings_with_material = scene.get_objects(label="buildings", material=material_idx)
print(f"- Buildings with material {material_idx}: {len(buildings_with_material)}")

# Print material properties
material = materials[material_idx]
print(f"\nMaterial {material_idx} properties:")
print(f"- Name: {material.name}")
print(f"- Permittivity: {material.permittivity}")
print(f"- Conductivity: {material.conductivity}")

# 3. Object Properties
print("\nObject Properties:")
building = buildings[0]
print(f"- Building faces: {len(building.faces)}")
print(f"- Building height: {building.height:.2f}m")
print(f"- Building volume: {building.volume:.2f}m³")
print(f"- Building footprint area: {building.footprint_area:.2f}m²")

# 4. Bounding Boxes
print("\nBuildings Bounding Box:")
bb = buildings.bounding_box
print(f"- Width (X): {bb.width:.2f}m")
print(f"- Length (Y): {bb.length:.2f}m")
print(f"- Height (Z): {bb.height:.2f}m")

# %% [markdown]
# ## User Sampling

# %% [markdown]
# ### Dataset Trimming
# For sampling users, we always have to find first the indices of the users we want to keep
# Then, we can use them to index particular matrix, or the entire dataset -> `subset()` method
#

# %%
print("\nActive Users and Dataset Subsetting (Trimming) Example")
print("-" * 50)

# Get indices of active users (those with paths)
active_idxs = dataset.get_idxs("active")
print(f"Original dataset has {dataset.n_ue} UEs")
print(f"Found {len(active_idxs)} active UEs")

# Create new dataset with only active users
dataset_t = dataset.trim(idxs=active_idxs)
print(f"New dataset has {dataset_t.n_ue} UEs")

dataset_t.plot_coverage(dataset_t.aoa_az[:, 0])

# %% [markdown]
# ### Uniform

# %%
idxs = dataset.get_idxs("uniform", steps=[4, 4])
dm.plot_coverage(dataset.rx_pos[idxs], dataset.aoa_az[idxs, 0], dpi=150, bs_pos=dataset.tx_pos.T)

# %% [markdown]
# ### Rows and Columns

# %%
row_idxs = dataset.get_idxs("row", row_idxs=np.arange(40, 60))
col_idxs = dataset.get_idxs("col", col_idxs=np.arange(40, 60))

dataset_sub1 = dataset.trim(idxs=row_idxs)
dataset_sub2 = dataset.trim(idxs=col_idxs)

dataset.plot_coverage(dataset.los, title="Full dataset")
x_lim, y_lim = plt.xlim(), plt.ylim()

dataset_sub1.plot_coverage(dataset_sub1.los, title="Row subset")
plt.xlim(x_lim)
plt.ylim(y_lim)

dataset_sub2.plot_coverage(dataset_sub2.los, title="Column subset")
plt.xlim(x_lim)
plt.ylim(y_lim)

# %% [markdown]
# ### Linear

# %%
# Get the closest dataset positions for a given path
idxs1 = dataset.get_idxs("linear", start_pos=[100, 90], end_pos=[-50, 90], n_steps=75)
idxs2 = dataset.get_idxs("linear", start_pos=[100, 80], end_pos=[-50, 80], n_steps=75)
idxs3 = dataset.get_idxs("linear", start_pos=[30, 0], end_pos=[30, 150], n_steps=75)

dataset.plot_coverage(dataset.los, title="LoS with positions", cbar_title="LoS status")

plt.scatter(
    dataset.rx_pos[idxs1, 0],
    dataset.rx_pos[idxs1, 1],
    c="blue",
    label="path1",
    s=6,
    lw=0.1,
)
plt.scatter(
    dataset.rx_pos[idxs2, 0],
    dataset.rx_pos[idxs2, 1],
    c="cyan",
    label="path2",
    s=6,
    lw=0.1,
)
plt.scatter(dataset.rx_pos[idxs3, 0], dataset.rx_pos[idxs3, 1], c="red", label="path3", s=6, lw=0.1)
plt.legend()

# %%
# Feature variation across linear path

for var_name in ["los", "pathloss", "delay"]:
    plt.figure()
    data = dataset[var_name] if var_name != "delay" else dataset[var_name][:, 0]
    plt.plot(data[idxs1], ls="-", c="blue", label="path1", marker="*", markersize=7)
    plt.plot(data[idxs2], ls="-.", c="cyan", label="path2", marker="s", markerfacecolor="none")
    plt.plot(data[idxs3], ls="--", c="red", label="path3", marker="o", markerfacecolor="w")
    plt.xlabel("Position index")
    plt.ylabel(f"{var_name}")
    plt.grid()
    plt.legend()
    plt.show()

# %% [markdown]
# ### Rectangular Zones

# %%
idxs_a = dm.get_idxs_with_limits(dataset.rx_pos, x_min=-100, x_max=-60, y_min=0, y_max=40)

idxs_b = dm.get_idxs_with_limits(dataset.rx_pos, x_min=125, x_max=165, y_min=0, y_max=40)

# Plot boxes
dataset.plot_coverage(dataset.aoa_az[:, 0])

plt.scatter(
    dataset.rx_pos[idxs_a, 0],
    dataset.rx_pos[idxs_a, 1],
    label="box A",
    s=2,
    lw=0.1,
    alpha=0.3,
)
plt.scatter(
    dataset.rx_pos[idxs_b, 0],
    dataset.rx_pos[idxs_b, 1],
    label="box B",
    s=2,
    lw=0.1,
    alpha=0.3,
)
plt.legend()
plt.title("Dataset zones on AoA Azimuth [º]")

# %%
def plot_feat_dist(data_a: np.ndarray, data_b: np.ndarray, feat_name: str) -> None:
    """Plot histograms of coordinate distributions for two datasets.

    Args:
        data_a: Array of coordinates for dataset A
        data_b: Array of coordinates for dataset B
        feat_name: Label for the plotted quantity

    """
    hist_params = {"alpha": 0.5, "bins": 8, "zorder": 2}

    # dist on x
    plt.figure()
    plt.hist(data_a, **hist_params, label="A")
    plt.hist(data_b, **hist_params, label="B")
    plt.title(f"{feat_name} distribution")
    plt.xlabel(f"{feat_name}")
    plt.grid()
    plt.legend()
    plt.show()


plot_feat_dist(dataset.rx_pos[idxs_a, 0], dataset.rx_pos[idxs_b, 0], "x (m)")
plot_feat_dist(dataset.rx_pos[idxs_a, 1], dataset.rx_pos[idxs_b, 1], "y (m)")
plot_feat_dist(dataset.aoa_az[idxs_a, 0], dataset.aoa_az[idxs_b, 0], "AoA Azimuth [º]")

# %% [markdown]
# ### Path Type

# %%
dataset_t = dataset.trim_by_path_type(["LoS", "R"])

attr_name = ["los", "num_interactions", "num_paths", "inter"]
for attr in attr_name:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    dataset[attr].plot(title=f"Full dataset ({attr})", ax=axes[0])
    dataset_t[attr].plot(title=f"Trimmed dataset ({attr})", ax=axes[1])

# Visualize the change in the rays (user 342 has LoS)
fig, axes = plt.subplots(
    1,
    2,
    figsize=(9, 4),
    constrained_layout=True,
    subplot_kw={"projection": "3d"},
)
dataset.plot_rays(342, ax=axes[0])
dataset_t.plot_rays(342, ax=axes[1])

# %% [markdown]
# ### Path Depth

# %%
dataset_t = dataset.trim_by_path_depth(1)

attr_name = ["los", "num_interactions", "num_paths", "inter"]
for attr in attr_name:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    dataset[attr].plot(title=f"Full dataset ({attr})", ax=axes[0])
    dataset_t[attr].plot(title=f"Trimmed dataset ({attr})", ax=axes[1])

# Visualize the rays too
fig, axes = plt.subplots(
    1,
    2,
    figsize=(9, 4),
    constrained_layout=True,
    subplot_kw={"projection": "3d"},
)
dataset.plot_rays(342, ax=axes[0])
dataset_t.plot_rays(342, ax=axes[1])

# %% [markdown]
# ## Beamforming

# %% [markdown]
# ### Computing Beamformers

# %%
ch_params = dm.ChannelParameters()  # default array has 8 elements
ch_params.bs_antenna.rotation = np.array([0, 0, -135])
ch_params.bs_antenna.shape = np.array([32, 1])
dataset.compute_channels(ch_params)

n_beams = 16

beam_angles = np.around(np.linspace(-60, 60, n_beams), 2)
print(f"Beam angles: {beam_angles}")

# Compute Beamformers: F1 is [n_beams, n_ant]
F1 = np.array(
    [dm.steering_vec(dataset.ch_params.bs_antenna.shape, phi=azi).squeeze() for azi in beam_angles],
)

# Apply beamformers
recv_bf_pwr_dbm = np.zeros((dataset.n_ue, n_beams)) * np.nan
mean_amplitude = np.abs(F1 @ dataset.channel[dataset.los != -1]).mean(axis=1).mean(axis=-1)
# Avg over rx antennas and subcarriers, respectively

# Convert to dBm
recv_bf_pwr_dbm[dataset.los != -1] = np.around(20 * np.log10(mean_amplitude) + 30, 1)

# %% [markdown]
# ### Visualization

# %% [markdown]
# #### Plot Received Power per Beam

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300, tight_layout=True)

for plt_idx, beam_idx in enumerate([6, 8, 12]):
    dataset.plot_coverage(
        recv_bf_pwr_dbm[:, beam_idx],
        ax=axes[plt_idx],
        lims=[-180, -60],
        title=f"Beam # {beam_idx} ({beam_angles[beam_idx]:.1f}º)",
    )

# %% [markdown]
# #### Plot Best Beam per Position

# %%
# Average the power on each subband and get the index of the beam that delivers max pwr
best_beams = np.argmax(recv_bf_pwr_dbm, axis=1).astype(float)
best_beams[np.isnan(recv_bf_pwr_dbm[:, 0])] = np.nan

dm.plot_coverage(
    dataset.rx_pos,
    best_beams,
    bs_pos=dataset.tx_pos.T,
    bs_ori=dataset.tx_ori,
    title="Best Beams",
    cbar_title="Best beam index",
)

# %% [markdown]
# #### Plot Max Received Power

# %%
max_bf_pwr = np.max(recv_bf_pwr_dbm, axis=1)
dm.plot_coverage(
    dataset.rx_pos,
    max_bf_pwr,
    bs_pos=dataset.tx_pos.T,
    bs_ori=dataset.tx_ori,
    title="Best Beamformed Power (with grid of beams) ",
)

# %% [markdown]
# ## Convert to DeepMIMO

# %% [markdown]
# ### From Wireless InSite
#
#

# %%
import subprocess

subprocess.run([
    "wget", "-O", "asu_campus_p2m.zip",
    "https://www.dropbox.com/s/lgzw8am5v5qz06v/asu_campus_p2m.zip?"
    "e=1&st=pcon8w9l&dl=1"
], check=False)
dm.unzip("asu_campus_p2m.zip")

# %%
rt_folder = "./asu_campus_p2m/asu_campus"
scen_name_insite = dm.convert(rt_folder, scenario_name="asu_campus_insite")

# %%
dataset_insite = dm.load(scen_name_insite)

# NOTE: This will crash if sionna is not installed

# %% [markdown]
# ### From Sionna RT
# Sionna is a bit more complicated because it doesn't have standard saving
# methods. Because of that, we use DeepMIMO exporter for Sionna, that saves the
# Scene, Path and computation parameters.
#
# Below is an example of ray tracing a simple scene in Sionna and converting it
# to DeepMIMO.
#
# Note: This code was tested with Sionna 0.19. It should work with many previous
# versions too, but needs to be verified. An example for Sionna 1.x is coming
# soon.

# %%
# pip install sionna==0.19.1

# %%
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sionna
from sionna.rt import DirectivePattern, PlanarArray, Receiver, Transmitter, load_scene
from tqdm import tqdm


def compute_array_combinations(arrays: list) -> np.ndarray:
    """Compute cartesian product combinations for array parameters."""
    return np.stack(np.meshgrid(*arrays), -1).reshape(-1, len(arrays))


def gen_user_grid(box_corners: list, steps: list, box_offsets: list | None = None) -> np.ndarray:
    """Generate a grid of user positions.

    box_corners is = [bbox_min_corner, bbox_max_corner]
    steps = [x_step, y_step, z_step].
    """
    # Sample the ranges of coordinates
    ndim = len(box_corners[0])
    dim_ranges = []
    for dim in range(ndim):
        if steps[dim]:
            dim_range = np.arange(box_corners[0][dim], box_corners[1][dim], steps[dim])
        else:
            dim_range = np.array([box_corners[0][dim]])  # select just the first limit

        dim_ranges.append(dim_range + box_offsets[dim] if box_offsets else 0)

    pos = compute_array_combinations(dim_ranges)
    print(f"Total positions generated: {pos.shape[0]}")
    return pos


def create_base_scene(scene_path: str, center_frequency: float) -> Any:
    """Load a Sionna scene and apply frequency and array defaults."""
    scene = load_scene(scene_path)
    scene.frequency = center_frequency
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    scene.rx_array = scene.tx_array
    scene.synthetic_array = True

    return scene


# Save dict with compute path params to export later
my_compute_path_params = {
    "max_depth": 5,
    "num_samples": 1e6,
    "scattering": False,
    "diffraction": False,
}
carrier_freq = 3.5 * 1e9  # Hz

tx_pos = [-33, 11, 32.03]

# 0- Create/Fetch scene and get buldings in the scene
scene = create_base_scene(sionna.rt.scene.simple_street_canyon, center_frequency=carrier_freq)

# 1- Compute TX position
print("Computing BS position")
scene.add(Transmitter(name="tx", position=tx_pos, orientation=[0, 0, 0]))

# 2- Compute RXs positions
print("Computing UEs positions")
rxs = gen_user_grid(
    box_corners=[(-93, -60, 0), (93, 60, 0)],
    steps=[4, 4, 0],
    box_offsets=[0, 0, 2],
)

# 3- Add the first batch of receivers to the scene
n_rx = len(rxs)
n_rx_in_scene = 10  # to compute in parallel
print(f"Adding users to the scene ({n_rx_in_scene} at a time)")
for rx_idx in range(n_rx_in_scene):
    scene.add(Receiver(name=f"rx_{rx_idx}", position=rxs[rx_idx], orientation=[0, 0, 0]))

# 4- Enable scattering in the radio materials
if my_compute_path_params["scattering"]:
    for rm in scene.radio_materials.values():
        rm.scattering_coefficient = 1 / np.sqrt(3)  # [0,1]
        rm.scattering_pattern = DirectivePattern(alpha_r=10)

# 5- Compute the paths for each set of receiver positions
path_list = []
n_rx_remaining = n_rx
for x in tqdm(range(int(n_rx / n_rx_in_scene) + 1), desc="Path computation"):
    if n_rx_remaining > 0:
        n_rx_remaining -= n_rx_in_scene
    else:
        break
    if x != 0:
        # modify current RXs in scene
        for rx_idx in range(n_rx_in_scene):
            if rx_idx + n_rx_in_scene * x < n_rx:
                scene.receivers[f"rx_{rx_idx}"].position = rxs[rx_idx + n_rx_in_scene * x]
            else:
                # remove the last receivers in the scene
                scene.remove(f"rx_{rx_idx}")

    paths = scene.compute_paths(**my_compute_path_params)

    paths.normalize_delays = False  # sum min_tau to tau, or tau of 1st path is always = 0

    path_list.append(paths)

# %%
# Ensure deepmimo is installed if running this locally
from deepmimo.exporters import sionna_exporter

save_folder = "sionna_test_scen/"

sionna_exporter(scene, path_list, my_compute_path_params, save_folder)

# %%
# To download the scenario to try locally (Colab only)
zip_path = dm.zip("sionna_test_scen")
# Uncomment the following lines if running in Google Colab:
# from google.colab import files
# files.download(zip_path)

# %%
import deepmimo as dm

scen_name_sionna = dm.convert(save_folder, overwrite=True)

# %%
dataset_sionna = dm.load(scen_name_sionna)

# %%
main_keys = ["aoa_az", "aoa_el", "aod_az", "aod_el", "delay", "power", "phase", "los", "num_paths"]
NDIM_TWO = 2

for key in main_keys:
    mat = dataset_sionna[key]
    plt_var = mat[:, 0] if mat.ndim == NDIM_TWO else mat
    dataset_sionna.plot_coverage(plt_var, title=key, scat_sz=50)

# %% [markdown]
# ### From AODT
# Like Sionna, conversion from AODT requires an exporter in DeepMIMO. This
# exporter will save AODT data using similar methods to those presented in AODT
# `Export_Data.ipynb` notebook.
#
# After ray tracing with AODT, be it AODT on the cloud, locally or in a remote
# server, this code can be executed wherever a clickhouse client can be executed,
# to fetch data from the database and save the necessary tables in parquet
# format.

# %%
# Install DeepMIMO in the Notebook (with AODT dependencies)
# pip install --pre deepmimo[aodt]

# %%
try:
    # Load database client
    from clickhouse_driver import Client

    db_client = Client("clickhouse")

    # Export database to folder with parquet files
    import deepmimo as dm
    from deepmimo.exporters import aodt_exporter

    rt_folder = aodt_exporter(db_client)
    # dm.zip(rt_folder)  # further zip the file for manual download

    scenario_aodt = dm.convert(rt_folder, overwrite=True)

    aodt_dataset = dm.load(scenario_aodt)

except (ImportError, RuntimeError) as e:
    print("This should be executed in a machine with clickhouse server access.")
    print(f"Error: {e!s}")

# %% [markdown]
# ### Dynamic Dataset

# %% [markdown]
# Dynamic datasets are the same as normal datasets, but usually with fewer
# receivers, but these receivers (or the transmitters, or the objects in the
# scene) move across scenes. Therefore the DeepMIMO dataset will consist of
# multiple MacroDatasets (or Datasets - refer to the Loading Section for
# information about these objects), one for each *snapshot*.
#
# To ray trace and convert such a dataset to DeepMIMO is very simple. Ray trace
# each individual scene independently and export them to a common folder, and
# pass that folder (which effectively contains several independent ray tracer
# outputs) to the converter, and it will figure out that the dataset is a
# Dynamic Dataset if it finds folders instead of files, and files inside one of
# those folders. The folders will be sorted (using the `sorted()` Python
# built-in) and the snapshots will be saved in that order.

# %%
try:
    dyn_dataset = dm.convert("path to folder containing individual datasets")
    dyn_dataset.scene.plot()  # Draws the scene of each dataset
except (FileNotFoundError, RuntimeError, ValueError) as e:
    print("This should be executed when a Dynamic scenario exists in the "
          "database (coming soon)")
    print(f"Error: {e!s}")

# %% [markdown]
# ### Multi-antenna Dataset
#

# %% [markdown]
# Similar to Dynamic datasets, DeepMIMO supports multiple antenna raytracing
# scenarios. Currently, the easiest way to ray trace such scenarios is to define
# multiple transmitters, each with a single antenna. In the future DeepMIMO will
# support multi-antenna terminals, but this has proven to be a rare use-case that
# is most often unnecessary.

# %% [markdown]
# ## Upload to DeepMIMO

# %% [markdown]
# ### Upload Scenario Files

# %%
import shutil

# Rename scenario to not converge with other previously uploaded scenarios
rng = np.random.default_rng()
scen_name_to_upload = scen_name + f"_{rng.integers(0, 1e8)}"
old_folder = dm.get_scenario_folder(scen_name)
new_folder = dm.get_scenario_folder(scen_name_to_upload)

# Copy the scenario files to a new folder, as if we just converted it
shutil.copytree(old_folder, new_folder, dirs_exist_ok=True)

# %%
# Get key in DeepMIMO "Contribute" dashboard (this one will be inactive)
MY_DEEPMIMO_KEY = "c3e344d106ab46161fea04b929bca1ae4a92ef5a368022561faa08da6e59dab0"
dm.upload(scen_name_to_upload, key=MY_DEEPMIMO_KEY)

# %% [markdown]
# ### Upload Additional Images

# %%
# For example, a GPS image
import subprocess

subprocess.run([
    "wget", "https://deepmimo.net/images/1737161953000.jpg"
], check=False)

# %%
from IPython.display import Image
from IPython.display import display as show_display

scen_gps_file_name = "1737161953000.jpg"

# Display local file
show_display(Image(filename=scen_gps_file_name, width=600))

# %%
dm.upload_images(scen_name_to_upload, key=MY_DEEPMIMO_KEY, img_paths=[scen_gps_file_name])

# %% [markdown]
# ### Upload Ray Tracing Source

# %%
from pathlib import Path

rt_folder = "fake_rt_output_folder"
Path(rt_folder).mkdir(parents=True, exist_ok=True)
zip_path = dm.zip(rt_folder)
dm.upload_rt_source(scen_name_to_upload, zip_path, MY_DEEPMIMO_KEY)

# %% [markdown]
# If everything went well, you should see the following submission under your "contribute" dashboard

# %%
# download screenshot
import subprocess

subprocess.run([
    "wget", "-O", "submission_screenshot_example.png",
    "https://deepmimo.net/examples/submission_screenshot_example.png"
], check=False)

# %%
from IPython.display import Image
from IPython.display import display as show_display

# Display local file
show_display(Image(filename="submission_screenshot_example.png"))

"""Scratchpad script for DeepMIMO conversion and visualization workflows."""

# %% Imports

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import deepmimo as dm

# from api_keys import DEEPMIMO_API_KEY

# %% V4 Conversion

# Example usage
rt_folder = "./RT_SOURCES/asu_campus"

scen_name = Path(rt_folder).name
dm.convert(rt_folder, overwrite=True, scenario_name=scen_name, vis_scene=True)

# %%

dataset = dm.load("asu_campus_3p5")

# %% AODT Conversion

# aodt_scen_name = 'aerial_2025_6_18_16_43_21'  # new (1 user)
# aodt_scen_name = 'aerial_2025_6_22_16_10_16' # old (2 users)
aodt_scen_name = "aerial_2025_6_18_16_43_21_dyn"  # new (1 user, dynamic)
folder = f"aodt_scripts/{aodt_scen_name}"
# df = pd.read_parquet(str(Path(folder) / 'db_info.parquet'))

# df.head()
aodt_scen = dm.convert(folder, overwrite=True)

aodt_scen = dm.load(aodt_scen_name, max_paths=500)

# %%
rt_folder = "./RT_SOURCES/"
sionna_rt_path_syn_true = rt_folder + "sionna_test_scen_synthetic_true"
# sionna_rt_path_syn_false = rt_folder + 'sionna_test_scen_synthetic_false' # multi-rx ant
sionna_rt_path_syn_false = rt_folder + "sionna_test_scen_synthetic_False3"  # single-rx ant

# %% Synthetic True
scen_syn = dm.convert(sionna_rt_path_syn_true, overwrite=True)
d = dm.load(scen_syn)
d.los.plot(scat_sz=20)
d.inter.plot(scat_sz=20)

# %% Synthetic False

scen_syn = dm.convert(sionna_rt_path_syn_false, overwrite=True)
d = dm.load(scen_syn)
d[1].los.plot(scat_sz=20)
d[1].inter.plot(scat_sz=20)

# %%
dataset = dm.load("asu_campus_3p5")

# %% Unified trim() API example (before/after)

dataset = dm.load("asu_campus_3p5")

# Set a BS rotation to make FoV meaningful
ch_params = dm.ChannelParameters()
ch_params.bs_antenna.rotation = [0, 0, -135]
dataset.set_channel_params(ch_params)

active_ues_before = int((dataset.num_paths > 0).sum())
before_msg = (
    f"Before trim -> n_ue={dataset.n_ue}, max_paths={dataset.max_paths}, "
    f"active_ues={active_ues_before}"
)
print(before_msg)

# Apply unified trimming in order: idxs -> FoV -> path depth -> path type
trimmed = dataset.trim(
    idxs=np.arange(0, dataset.n_ue, 5),  # keep every other UE
    bs_fov=[90, 90],  # FoV at BS
    path_depth=1,  # at most 1 interaction
    path_types=["LoS", "R"],  # only LoS and reflections
)

active_ues_after = int((trimmed.num_paths > 0).sum())
after_msg = (
    f"After trim  -> n_ue={trimmed.n_ue}, max_paths={trimmed.max_paths}, "
    f"active_ues={active_ues_after}"
)
print(after_msg)

# Optional visualization
trimmed.los.plot(title="Unified trim result")

# %%
ch_params = dm.ChannelParameters()
ch_params.ofdm.subcarriers = 32
ch_params.ofdm.selected_subcarriers = np.arange(1)
ch_params.num_paths = 5

dataset = dm.load("asu_campus_3p5")

# %%

dataset_t = dataset.subset(dataset.get_uniform_idxs([3, 3]))
params = dm.ChannelParameters()
params["bs_antenna"]["rotation"] = np.array([0, 0, -135])
dataset_t.set_channel_params(dm.ChannelParameters({"bs_antenna": {"rotation": [0, 0, -135]}}))
# Notes on possible API aliases for ergonomics:
# - ChannelParameters() -> ChParams()
# - set_channel_params() -> set_ch_params(); could accept only dict inputs
# - compute_channels() -> compute_ch()

fig, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=300)
dataset_t.los.plot(ax=axs[0], title="Before FoV")
dataset_t.apply_fov(bs_fov=np.array([60, 180]))  # irreversible change!
dataset_t.los.plot(ax=axs[1], title="After FoV")

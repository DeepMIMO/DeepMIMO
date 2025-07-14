#%% Imports

import os
import numpy as np
import deepmimo as dm
import matplotlib.pyplot as plt

from api_keys import DEEPMIMO_API_KEY

#%% V4 Conversion

# Example usage
rt_folder = './RT_SOURCES/asu_campus'

scen_name = os.path.basename(rt_folder)
dm.convert(rt_folder, overwrite=True, scenario_name=scen_name, vis_scene=True)

dataset = dm.load('asu_campus_3p5')

#%% AODT Conversion
import os
import deepmimo as dm
import pandas as pd

# aodt_scen_name = 'aerial_2025_6_18_16_43_21'  # new (1 user)
# aodt_scen_name = 'aerial_2025_6_22_16_10_16' # old (2 users)
aodt_scen_name = 'aerial_2025_6_18_16_43_21_dyn'  # new (1 user, dynamic)
folder = f'aodt_scripts/{aodt_scen_name}'
# df = pd.read_parquet(os.path.join(folder, 'db_info.parquet'))

# df.head()
aodt_scen = dm.convert(folder, overwrite=True)

aodt_scen = dm.load(aodt_scen_name, max_paths=500)

#%%

dataset = dm.load('asu_campus_3p5')

idx_1 = 10
idx_2 = 11

dataset.plot_rays(idx_1, proj_3D=False)
dataset.plot_rays(idx_2, proj_3D=False)

# GOAL: interpolate between positions

# TODO: FOR EACH PATH in a index pair:
#       1) Identify whether it has a correspondent in the other index
#       2) SAME PATHS: [Linear Interpolation] each quantity
#       2.1) Provide percentages for new points in the interpolation, i.e. [0.1, 0.3, 0.7]
#       3) NOT same paths: what to do? interpolate only equal paths?
#       3.1) Provide percentages for new points in the interpolation, i.e. [0.1, 0.3, 0.7]
#       4) Organize paths again (sort by strongest pwr)
#       5) Return the new dataset??


#%%

dataset.print_rx(idx_1, path_idxs=[0])
dataset.print_rx(idx_2, path_idxs=[0])


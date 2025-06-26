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

#%% Trimming by path type

dataset = dm.load('asu_campus_3p5')

# dataset_t = dataset.trim_by_path_depth(1)
dataset_t = dataset.trim_by_path_type(['LoS', 'R'])

dataset.plot_coverage(dataset.los, title='Full dataset')
dataset_t.plot_coverage(dataset_t.los, title='Trimmed dataset')

# Num interactions
dataset.plot_coverage(dataset.num_interactions[:,0], title='Number of interactions')
dataset_t.plot_coverage(dataset_t.num_interactions[:,0], title='Number of interactions')

# Num paths
dataset.plot_coverage(dataset.num_paths, title='Number of paths')
dataset_t.plot_coverage(dataset_t.num_paths, title='Number of paths')

# Interaction type
dataset_t.plot_coverage(dataset_t.inter[:,0], title='Interaction type')

# Plot rays
dataset_t.plot_rays(9)


#%% PLOT RAYS with BUILDINGS

dm.plot_rays(dataset['rx_pos'][10], dataset['tx_pos'][0],
             dataset['inter_pos'][10], dataset['inter'][10],
             proj_3D=True, color_by_type=True)

# Next: determine which buildings interact with each ray. 
#       make a set of those buildings for all the rays in the user.
#       plot the buildings that matter to that user along with the rays.
#       (based on the building bounding boxes)
#       Use the PhysicalObjects class to plot a group of buildings.
#### NEXT: Make a plot of just SOME of the buildings
# buildings_scene = dm.Scene()
# for obj in buildings[:3]:
#     buildings_scene.add_object(obj)
    
# buildings_scene.plot()
#####

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

#%%

aodt_scen = dm.load(aodt_scen_name, max_paths=500)
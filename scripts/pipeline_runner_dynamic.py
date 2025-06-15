"""
Run a pipeline that uses dynamic user positions.

This file is used to ray tracer a dynamic dataset, i.e., a dataset where users move.

Steps:
1. Generate user positions
2. Run ray tracing for each time step, and convert each snapshot to DeepMIMO format
3. Load the dataset and plot the results
"""


#%% Imports

import os
import numpy as np

import deepmimo as dm  # type: ignore

from deepmimo.pipelines.TxRxPlacement import gen_rx_grid, gen_tx_pos, gen_plane_grid
from deepmimo.pipelines.utils.pipeline_utils import get_origin_coords, load_params_from_row

from deepmimo.pipelines.sionna_rt.sionna_raytracer import raytrace_sionna

from scripts.pipeline_params import *

# Absolute (!) Paths
OSM_ROOT = os.path.join(os.getcwd(), "osm_root")

#%% User position generation

n_time = 10
x_step = 1

# Generate user positions with 1-meter spacing along x-axis
x_coords = np.arange(0, n_time * x_step, x_step)  # 0 to 9 meters with 1m spacing
rx_pos_list = []
for x in x_coords:
    # Create array for two users at this x position
    user_pos = np.array([
        [x, 0, 1.5],    # User at y=0
        [x, 5, 1.5]    # User at y=10
    ])
    rx_pos_list.append(user_pos)

tx_pos = np.array([[0, 0, 10]])

for index in range(n_time):

	print(f"Processing time index: {index}")
	p['name'] = f'simple_reflector_time_{index}'
	osm_folder = os.path.join(OSM_ROOT, p['name'])
	rx_pos = rx_pos_list[index]
	
	print('Starting RT')
	# osm_folder = os.path.join(OSM_ROOT, "simple_reflector")

	# RT Phase 4: Run Wireless InSite ray tracing
	# rt_path = raytrace_insite(osm_folder, tx_pos, rx_pos, **p)
	rt_path = raytrace_sionna(osm_folder, tx_pos, rx_pos, **p)

	# RT Phase 5: Convert to DeepMIMO format
	scen_name = dm.convert(rt_path, overwrite=True)

	# RT Phase 6: Test Conversion
	dataset = dm.load(scen_name)
	dataset.plot_coverage(dataset.los, scat_sz=40)
	dataset.plot_coverage(dataset.pwr[:, 0], scat_sz=40)
    

#%% Load a single scenario

outer_folder = OSM_ROOT

scen_name = dm.convert(outer_folder + '/simple_reflector_time_0', overwrite=True)
dataset = dm.load(scen_name)


#%% Load a dynamic dataset
outer_folder = OSM_ROOT
dyn_dataset_name = dm.convert(outer_folder, scenario_name='scen1', 
                                    overwrite=True, vis_scene=False)

dyn_dataset = dm.load(dyn_dataset_name)


#%% Plot summary
dm.plot_summary(dataset=dyn_dataset[0], plot_idx=[1])
dyn_dataset[0].plot_summary(plot_idx=[1])

dm.plot_summary(dataset=dyn_dataset, plot_idx=[1])
dyn_dataset.plot_summary(plot_idx=[1])

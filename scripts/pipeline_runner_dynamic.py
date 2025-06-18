"""
Run a pipeline that uses dynamic user positions.

This file is used to ray tracer a dynamic dataset, i.e., a dataset where users move.

Steps:
1. Generate user and object positions / orientation / velocities
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

np.set_printoptions(precision=3, suppress=True)


def path_inspection(paths):
	"""
	Introspect the paths object - This function should run right after path computation.
	"""
	i = 0
	# print(paths.doppler.numpy()[i,0,:])
	# print(f'doppler shape: {paths.doppler.shape}')
	# print(paths.tau.numpy()[i,0,:])
	complex_a = paths.a[0][i,0,0,0,:].numpy() + 1j * paths.a[1][i,0,0,0,:].numpy()
	path_idxs = np.argsort(np.abs(complex_a))[::-1]
	print('reordered doppler & delay')
	# print('delay')
	# print(paths.tau.numpy()[i,0,path_idxs])
	# a1 = np.take_along_axis(a, paths_idxs_a, axis=0)
	# doppler_reordered = paths.doppler[:,0,path_idxs]
	print('doppler')
	print(paths.doppler.numpy()[i,0,path_idxs])

	# print('primitives')
	# print(paths.primitives.numpy()[:, i,0,path_idxs].swapaxes(0, -1))
	# print('vertices')
	# print(paths.vertices.numpy()[:, i, 0, path_idxs, :].swapaxes(0, -2))

p['path_inspection_func'] = path_inspection

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

#%% Set parameters

for index in range(n_time):

	# Define extra user parameters
	p['rx_ori'] = np.array([[0, 0, 0], [0, 0, 0]])
	p['tx_ori'] = np.array([0, 0, 0])
	p['rx_vel'] = np.array([[0, 0, 0], [0, 0, 0]])
	p['tx_vel'] = np.array([0, 0, 0])

	# Define extra object parameters (should match the scene.objects)
	p['obj_pos'] = None # np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
	p['obj_ori'] = None # np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
	p['obj_vel'] = None # np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

	print(f"Processing time index: {index}")
	p['name'] = f'simple_reflector_time_{index}'
	osm_folder = os.path.join(OSM_ROOT, p['name'])
	rx_pos = rx_pos_list[index]
	
	print('Starting RT')
	# osm_folder = os.path.join(OSM_ROOT, "simple_reflector")

	# Run Ray Tracing
	rt_path = raytrace_sionna(osm_folder, tx_pos, rx_pos, **p)

	# Convert to DeepMIMO format
	scen_name = dm.convert(rt_path, overwrite=True)

	# Test Conversion
	dataset = dm.load(scen_name)
	dataset.plot_coverage(dataset.los, scat_sz=40)
	dataset.plot_coverage(dataset.pwr[:, 0], scat_sz=40)
	break
    

#%% Load a single scenario

outer_folder = OSM_ROOT

scen_name = dm.convert(outer_folder + '/simple_reflector_time_0', overwrite=True)
dataset = dm.load(scen_name)

#%% Set manual velocities
# dataset.rx_vel = [[0, 0, 5], [0, 0, 0]]
# dataset.tx_vel = [0, 0, 0]
dataset.set_obj_vel(obj_idx=[1, 3, 6], vel=[[0, 5, 0], [0, 5, 6], [0, 0, 3]])
print(dataset.doppler[0])

#%% Load a dynamic dataset
outer_folder = OSM_ROOT
dyn_dataset_name = dm.convert(outer_folder, scenario_name='scen1', 
                                    overwrite=True, vis_scene=False)

dyn_dataset = dm.load(dyn_dataset_name)

#%%

import numpy as np
import deepmimo as dm  # type: ignore

#%%
a = dm.load('asu_campus_3p5')
a.set_doppler(10)
params = dm.ChannelParameters()
params.freq_domain = False
params.enable_doppler = True
a.compute_channels(params=params)

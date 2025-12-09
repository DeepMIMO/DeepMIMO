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

from scripts.pipeline_params import p

# Absolute (!) Paths
OSM_ROOT = os.path.join(os.getcwd(), "osm_root")

np.set_printoptions(precision=2, suppress=True)


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
	print(paths.tau.numpy()[i,0,path_idxs])
	# a1 = np.take_along_axis(a, paths_idxs_a, axis=0)
	# doppler_reordered = paths.doppler[:,0,path_idxs]
	print('doppler')
	print(paths.doppler.numpy()[i,0,path_idxs])

	# print('primitives')
	# print(paths.primitives.numpy()[:, i,0,path_idxs].swapaxes(0, -1))
	# print('vertices')
	# print(paths.vertices.numpy()[:, i, 0, path_idxs, :].swapaxes(0, -2))

def scene_edit(scene):
	"""
	Edit the scene before ray tracing.
	"""
	import mitsuba as mi
	import sionna.rt
	objs = [obj for obj_id, obj in scene.objects.items()]
	scene.edit(remove=objs[3])
	scene.edit(remove=objs[1])
	car_material = sionna.rt.ITURadioMaterial("car-material", "metal", 
	                                          thickness=0.01, color=(0.8, 0.1, 0.1))
	car_obj = sionna.rt.SceneObject(fname=sionna.rt.scene.low_poly_car, 
	                                name=f"car", radio_material=car_material)
	scene.edit(add=car_obj)
	car_obj.scaling = 5.0
	car_obj.position = mi.Vector3f(np.array([0, 0, 0]))
	# print(f'scene.objects: {scene.objects}')


p['path_inspection_func'] = path_inspection
p['scene_edit_func'] = scene_edit

#%% Set parameters
n_steps = 11
x1_coords = np.linspace(-20, 20, n_steps)
x2_coords = np.linspace(10, -10, n_steps)

for timestep in range(n_steps):
	# Define extra user parameters (these just need to match the number of users)

	rx_pos = np.array([[x1_coords[timestep], -50, 1.5],
	                   [x2_coords[timestep], -55, 1.5]])
	tx_pos = np.array([[0, 50, 10.5]])
	p['rx_ori'] = None # np.array([[0, 0, 0], [0, 0, 0]])
	p['tx_ori'] = None # np.array([0, 0, 0])
	p['rx_vel'] = None # np.array([[0, 0, 0], [0, 0, 0]])
	p['tx_vel'] = None # np.array([0, 0, 0])

	# Define extra object parameters (should match the scene.objects)
	p['obj_idx'] = ['car']
	p['obj_pos'] = [np.array([-50, 0, 0]) + np.array([10, 0, 0]) * timestep]
	# p['obj_ori'] = None # np.array([[0, 0, 0]])
	# p['obj_vel'] = None # np.array([[0, 0, 0]])

	print(f"Processing time index: {timestep}")
	p['name'] = f'simple_reflector_time_{timestep:02d}'  # 02d for 2 digits (needed for sorting)
	osm_folder = os.path.join(OSM_ROOT, p['name'])
	
	print('Starting RT')
	# osm_folder = os.path.join(OSM_ROOT, "simple_reflector")

	# Run Ray Tracing
	rt_path = raytrace_sionna(osm_folder, tx_pos, rx_pos, **p)

	# Convert to DeepMIMO format
	scen_name = dm.convert(rt_path, overwrite=True, vis_scene=False)

	# Test Conversion
	dataset = dm.load(scen_name)
	dataset.scene.plot(proj_3D=False)
	# dataset.plot_coverage(dataset.los, scat_sz=40)
	# dataset.plot_coverage(dataset.pwr[:, 0], scat_sz=40)
    

#%% Load a single scenario

outer_folder = OSM_ROOT

scen_name = dm.convert(outer_folder + '/simple_reflector_time_05', overwrite=True)
dataset = dm.load(scen_name)

#%% Load a dynamic dataset
outer_folder = OSM_ROOT
dyn_dataset_name = dm.convert(outer_folder, scenario_name='scen1', 
                              overwrite=True, vis_scene=False)

dyn_dataset = dm.load(dyn_dataset_name)

#%% Example 3: Dynamic Dataset

# Uniform snapshots
dyn_dataset.set_timestamps(10) # [seconds between scenes]

print(f'timestamps: {dyn_dataset.timestamps}')
print(f'rx_vel: {dyn_dataset.rx_vel}')
print(f'tx_vel: {dyn_dataset.tx_vel}')
print(f'obj_vel: {[obj.vel for obj in dyn_dataset.scene.objects]}')

# dataset.compute_channels(dm.ChannelParameters(doppler = True))

#%% Non-uniform snapshots
dyn_dataset.set_timestamps([0, 1.5, 2.3, 4.4, 5.8, 7.1, 8.9, 10.2, 11.7, 13.0, 13.1]) # [timestamps of each scene]

print(f'timestamps: {dyn_dataset.timestamps}')
print(f'rx_vel: {dyn_dataset.rx_vel}')
print(f'tx_vel: {dyn_dataset.tx_vel}')
print(f'obj_vel: {[obj.vel for obj in dyn_dataset.scene.objects]}')

# dataset.compute_channels(dm.ChannelParameters(doppler = True))


"""
Parameters for the pipeline.

Steps to run a pipeline:

1. pip install --pre deepmimo

2. Install dependencies
	- install miniforge (https://github.com/conda-forge/miniforge)
	- (recommended) mamba create -n dm_env python=3.10
	- (recommended) mamba activate dm_env
	- (recommended) pip install uv
	- (for insite and sionna 1.0.x)  uv pip install .[all]
	- (for insite and sionna 0.19.x) uv pip install .[sionna019]

3. Adjust parameters in this file. Particularly:
	- WI_ROOT: path to the Wireless InSite installation
	- Config versions:
		- dm.config('sionna_version', '0.19.1')  # E.g. '0.19.1', '1.0.2'
		- dm.config('wireless_insite_version', "4.0.1")  # E.g. '3.3.0', '4.0.1'
	- Materials
	- Ray tracing parameters in the p (parameters) dictionary

    Note: There may be parameters in the specific pipeline runner file 
    that need to be adjusted too. Particularly:
	- OSM_ROOT: path to output OSM and scenario data
	- API KEYS:
		- DEEPMIMO_API_KEY: your DeepMIMO API key
		- GMAPS_API_KEY: your Google Maps API key

(Optional, if running a OSM pipeline)
4. Create a CSV file with the following format:

	name,min_lat,min_lon,max_lat,max_lon,bs_lat,bs_lon,bs_height
	city_0_newyork_3p5,40.68503298,-73.84682129,40.68597435,-73.84336302,"40.68575894,40.68578827,40.685554","-73.8446499,-73.84567948,-73.844944","10,10,10"
	city_1_losangeles_3p5,34.06430723,-118.2630866,34.06560881,-118.2609365,"34.06501496,34.06473123,34.06504731","-118.261547,-118.2619665,-118.2625399","10,10,10"

	Note: bs_lat/bs_lon/bs_height are comma separated lists of floats, 
	and the number of elements in the list must match the number of BSs.

	Note: the file pipeline_csv_gen.py can be used to generate a CSV file from a list of cities.
	Such a list of cities can be found in https://simplemaps.com/data/world-cities

5. Run specific pipeline:
	- python pipeline_runner_static.py
    - python pipeline_runner_osm_static.py
    - python pipeline_runner_dynamic.py

--------------------------------

TODO:
- Add option to indicate running multiple ray tracers, and ensure they all use the same materials and the same positions
- WI_EXE, WI_MAT, and these materials inside raytracer ("itu concrete", should match both sionna and wireless insite)
- Support sionna 1.0 in the exporter and converter
- Expand sionna 0.19.1 support (materials, roads, labels)
- Enhance fetch_satellite_view to choose the zoom level based on the bounding box size
- Remove utm dependency?
- Remove lxml dependency?

"""

#%% Imports
import os

#%% Constants

# Wireless InSite
WI_ROOT = "C:/Program Files/Remcom/Wireless InSite 3.3.0.4"
WI_EXE = os.path.join(WI_ROOT, "bin/calc/wibatch.exe")
WI_MAT = os.path.join(WI_ROOT, "materials")
WI_LIC = "C:/Users/jmora/Documents/GitHub/DeepMIMO/executables/wireless insite"

# Material paths
BUILDING_MATERIAL_PATH = os.path.join(WI_MAT, "ITU Concrete 3.5 GHz.mtl")
ROAD_MATERIAL_PATH = os.path.join(WI_MAT, "Asphalt_1GHz.mtl")
TERRAIN_MATERIAL_PATH = os.path.join(WI_MAT, "ITU Wet earth 3.5 GHz.mtl")

# GPU definition (e.g. for Sionna)
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

#%% Parameters

p = {
	# Scenario parameters (to be loaded from CSV)
	'name': None,
	'city': None,
	'min_lat': None,
	'min_lon': None,
	'max_lat': None,
	'max_lon': None,
	'bs_lats': None, 
	'bs_lons': None,
	'bs_heights': None,

	# User placement parameters
	'ue_height': 1.5,
	'grid_spacing': .1,
	'pos_prec': 4, # Decimal places for coordinates

	# Paths required by Wireless InSite
	'wi_exe': WI_EXE,
	'wi_lic': WI_LIC,
	'building_material': BUILDING_MATERIAL_PATH,
	'road_material': ROAD_MATERIAL_PATH,
	'terrain_material': TERRAIN_MATERIAL_PATH,

	# Ray-tracing parameters -> Efficient if they match the dataclass in SetupEditor.py
	'carrier_freq': 3.5e9,  # Hz
	'bandwidth': 10e6,  # Hz
	'max_reflections': 5, # Sionna currently breaking with 4 or more max_depth.
	'max_paths': 10,
	'ray_spacing': 0.25,  # m
	'max_transmissions': 0,
	'max_diffractions': 0,
	'ds_enable': False,
	'ds_max_reflections': 2,
	'ds_max_transmissions': 0,
	'ds_max_diffractions': 1,
	'ds_final_interaction_only': True,
	'conform_to_terrain': False,  # Whether to conform the terrain to the ray tracing grid
								  # (if True, positions have added the terrain height)
	'bs2bs': True,  # Whether to compute path between BSs (True) or not (False)
	
	# Insite specific parameters
    'insite_force_points': False,
    
	# Sionna specific parameters
	'los': True,  # Whether to use LOS paths (True) or not (False)
	'synthetic_array': True,  # Whether to use a synthetic array (True) or a real array (False)
	'batch_size': 15,  # Number of users to compute at a time
					   # Heuristic: 1.5 per GB of GPU VRAM, if using scattering, 
					   # else 5-10 users per GB
	'use_builtin_scene': False,  # Whether to use a builtin scene (True) or a custom scene (False)
                                 # NOTE: when fetching OSM data, set this to False
	'builtin_scene_path': 'simple_street_canyon', # 'simple_reflector', 'simple_street_canyon'
	'path_inspection_func': None,  # Function to inspect the paths after computation 
	                               # (before filtering or saving)
	'scene_edit_func': None,  # Function to edit the scene before ray tracing
	'create_scene_folder': False,  # Whether to create an additional scene folder
                                   # inside OSM_ROOT with params in the name.
	                               # Set to False for Dynamic Datasets 
                                   # (no extra folders -> direct access)

	
	# Sionna 0.x parameters
	'scat_random_phases': True,
	'edge_diffraction': False,
	'scat_keep_prob': 0.001,

	# Sionna 1.x parameters
	'n_samples_per_src': 1_000_000,  # Number of ray sampling directions per source
	'max_paths_per_src': 1_000_000,  # Maximum number of paths per source
	'refraction': False,  # Whether to use refraction (True) or not (False)
	'cpu_offload': True,  # Whether to offload paths to CPU (True) or not (False)
	                      # (slower, but does not accumulate VRAM usage)
	'rx_ori': None,  # [n_ue, 3] [rad]  # (azimuth, elevation, roll)
	'rx_vel': None,  # [n_ue, 3] [m/s] # (x, y, z)
	'tx_ori': None,  # [n_bs, 3] [rad]  # (azimuth, elevation, roll)
	'tx_vel': None,  # [n_bs, 3] [m/s] # (x, y, z)
	'obj_idx': None,  # [n_obj] [int]  # (x, y, z)
	'obj_pos': None,  # [n_obj, 3] [m]  # (x, y, z)
	'obj_ori': None,  # [n_obj, 3] [rad] # (azimuth, elevation, roll)
	'obj_vel': None,  # [n_obj, 3] [m/s] # (x, y, z)
}

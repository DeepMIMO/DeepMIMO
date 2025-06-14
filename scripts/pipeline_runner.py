"""
Steps to run a pipeline:

1. pip install deepmimo

2. Install dependencies
	- install miniforge (https://github.com/conda-forge/miniforge)
	- (recommended) mamba create -n dm_env python=3.10
	- (recommended) mamba activate dm_env
	- (recommended) pip install uv
	- (for insite and sionna 1.0.x)  uv pip install .[all]
	- (for insite and sionna 0.19.x) uv pip install .[sionna019]

3. Adjust parameters in this file. Particularly:
	- OSM_ROOT: path to output OSM and scenario data
	- WI_ROOT: path to the Wireless InSite installation
	- API KEYS:
		- DEEPMIMO_API_KEY: your DeepMIMO API key
		- GMAPS_API_KEY: your Google Maps API key
	- Config versions:
		- dm.config('sionna_version', '0.19.1')  # E.g. '0.19.1', '1.0.2'
		- dm.config('wireless_insite_version', "4.0.1")  # E.g. '3.3.0', '4.0.1'
	- Materials
	- Ray tracing parameters in the p (parameters) dictionary

4. Create a CSV file with the following format:

	name,min_lat,min_lon,max_lat,max_lon,bs_lat,bs_lon,bs_height
	city_0_newyork_3p5,40.68503298,-73.84682129,40.68597435,-73.84336302,"40.68575894,40.68578827,40.685554","-73.8446499,-73.84567948,-73.844944","10,10,10"
	city_1_losangeles_3p5,34.06430723,-118.2630866,34.06560881,-118.2609365,"34.06501496,34.06473123,34.06504731","-118.261547,-118.2619665,-118.2625399","10,10,10"

	Note: bs_lat/bs_lon/bs_height are comma separated lists of floats, 
	and the number of elements in the list must match the number of BSs.

	Note: the file pipeline_csv_gen.py can be used to generate a CSV file from a list of cities.
	Such a list of cities can be found in https://simplemaps.com/data/world-cities

5. Run: python pipeline_runner.py

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
import pandas as pd
import numpy as np

import deepmimo as dm  # type: ignore

from deepmimo.pipelines.TxRxPlacement import gen_rx_grid, gen_tx_pos, gen_plane_grid
from deepmimo.pipelines.utils.pipeline_utils import get_origin_coords, load_params_from_row
from deepmimo.pipelines.blender_osm_export import fetch_osm_scene
from deepmimo.pipelines.utils.geo_utils import get_city_name, fetch_satellite_view

# API Keys
# GMAPS_API_KEY = ""
# if GMAPS_API_KEY == "":
# 	try:
# 		from api_keys import GMAPS_API_KEY
# 	except ImportError:
# 		print("Please create a api_keys.py file, with GMAPS_API_KEY defined")
# 		print("Disabling Google Maps services:\n"
# 			"  - city name extraction\n"
# 			"  - satellite view image save")

# # DeepMIMO API Key
# DEEPMIMO_API_KEY = ""
# if DEEPMIMO_API_KEY == "":
# 	try:
# 		from api_keys import DEEPMIMO_API_KEY
# 	except ImportError:
# 		print("Please create a api_keys.py file, with DEEPMIMO_API_KEY defined")
# 		print("Disabling DeepMIMO services: scenario upload (zip, images, rt source)")

# Configure Ray Tracing Versions (before importing the pipeline modules)
dm.config('wireless_insite_version', "4.0.1")  # E.g. '3.3.0', '4.0.1'

# from deepmimo.pipelines.wireless_insite.insite_raytracer import raytrace_insite
from deepmimo.pipelines.sionna_rt.sionna_raytracer import raytrace_sionna

# Absolute (!!) Paths
# OSM_ROOT = "/home/jamorais/osm_root" # Windows
# OSM_ROOT = OSM_ROOT.replace('C:', '/mnt/c') # WSL
OSM_ROOT = os.path.join(os.getcwd(), "osm_root")

# Wireless InSite
WI_ROOT = "C:/Program Files/Remcom/Wireless InSite 4.0.0"
WI_EXE = os.path.join(WI_ROOT, "bin/calc/wibatch.exe")
WI_MAT = os.path.join(WI_ROOT, "materials")
WI_LIC = "C:/Users/jmora/Documents/GitHub/DeepMIMO/executables/wireless insite"

# Material paths
BUILDING_MATERIAL_PATH = os.path.join(WI_MAT, "ITU Concrete 3.5 GHz.mtl")
ROAD_MATERIAL_PATH = os.path.join(WI_MAT, "Asphalt_1GHz.mtl")
TERRAIN_MATERIAL_PATH = os.path.join(WI_MAT, "ITU Wet earth 3.5 GHz.mtl")

COUNTER = 30
#%% Iterate over CSV file to extract the map, create TX/RX positions, and run RT

# df = pd.read_csv('./dev/params_20cities_t.csv')

# GPU definition (e.g. for Sionna)
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

# Parameters
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
	'grid_spacing': 1,
	'pos_prec': 4, # Decimal places for coordinates

	# Paths required by Wireless InSite
	'wi_exe': WI_EXE,
	'wi_lic': WI_LIC,
	'building_material': BUILDING_MATERIAL_PATH,
	'road_material': ROAD_MATERIAL_PATH,
	'terrain_material': TERRAIN_MATERIAL_PATH,

	# Sionna specific parameters
	'los': True,  # Whether to use LOS paths (True) or not (False)
	'synthetic_array': True,  # Whether to use a synthetic array (True) or a real array (False)
	'batch_size': 15,  # Number of users to compute at a time
					   # Heuristic: 1.5 per GB of GPU VRAM, if using scattering, 
					   # else 5-10 users per GB
	'use_builtin_scene': True,  # Whether to use a builtin scene (True) or a custom scene (False)
	'builtin_scene_path': 'simple_street_canyon', # 'simple_reflector', 'simple_street_canyon'
	
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

	# Ray-tracing parameters -> Efficient if they match the dataclass in SetupEditor.py
	'carrier_freq': 3.5e9,  # Hz
	'bandwidth': 10e6,  # Hz
	'max_reflections': 3, # Sionna currently breaking with 4 or more max_depth.
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
	'bs2bs': False,  # Whether to compute path between BSs (True) or not (False)
	
	# Miscellaneous parameters
	'create_scene_folder': False,  # Whether to create a scene folder (True) or not (False)
}

n_time = 10
x_step = 1

#%%
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

# for index, row in df.iterrows():
# for index in [1]:
for index in range(n_time):

	print(f"Processing time index: {index}")
	p['name'] = f'simple_reflector_time_{index}'
	osm_folder = os.path.join(OSM_ROOT, p['name'])
	rx_pos = rx_pos_list[index]
	# print(f"\n{'=' * 50}\nSTARTING SCENARIO {index + 1}/{len(df)}: {row['name']}\n{'=' * 50}")

	# # RT Phase 1: Load GPS coordinates from CSV
	# load_params_from_row(row, p)

	# # RT Phase 2: Extract OSM data, City Name, and Satellite View
	# COUNTER += 1
	# osm_folder = os.path.join(OSM_ROOT, row['name']) + f'_{COUNTER}'
	# fetch_osm_scene(p['min_lat'], p['min_lon'], p['max_lat'], p['max_lon'],
	# 				osm_folder, output_formats=['insite','sionna'])
	# p['origin_lat'], p['origin_lon'] = get_origin_coords(osm_folder)

	# p['city'] = get_city_name(p['origin_lat'], p['origin_lon'], GMAPS_API_KEY)
	# sat_view_path = fetch_satellite_view(p['min_lat'], p['min_lon'], p['max_lat'], p['max_lon'],
	# 									 GMAPS_API_KEY, osm_folder)
	
	# # break
	# # RT Phase 3: Generate RX and TX positions
	# rx_pos = gen_rx_grid(p)  # N x 3 (N ~ 100k)
	# tx_pos = gen_tx_pos(p)   # M x 3 (M ~ 3)
	
	# # Optional: Round positions (visually better)
	# rx_pos = np.round(rx_pos, p['pos_prec'])
	# tx_pos = np.round(tx_pos, p['pos_prec'])
	
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
	# break

	# RT Phase 7: Upload (zip rt source)
	# scen_name = dm.zip(rt_path)
	# dm.upload(scen_name, key=DEEPMIMO_API_KEY)
	# dm.upload_images(scen_name, img_paths=[sat_view_path],  key=DEEPMIMO_API_KEY)
	# dm.upload_rt_source(scen_name, rt_zip_path=dm.zip(rt_path), key=DEEPMIMO_API_KEY)

#%%

outer_folder = OSM_ROOT

dataset_time = dm.convert(outer_folder)
dataset

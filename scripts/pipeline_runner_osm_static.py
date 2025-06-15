"""
Run a pipeline that uses OSM data.

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
GMAPS_API_KEY = ""
if GMAPS_API_KEY == "":
	try:
		from api_keys import GMAPS_API_KEY
	except ImportError:
		print("Please create a api_keys.py file, with GMAPS_API_KEY defined")
		print("Disabling Google Maps services:\n"
			"  - city name extraction\n"
			"  - satellite view image save")

# DeepMIMO API Key
DEEPMIMO_API_KEY = ""
if DEEPMIMO_API_KEY == "":
	try:
		from api_keys import DEEPMIMO_API_KEY
	except ImportError:
		print("Please create a api_keys.py file, with DEEPMIMO_API_KEY defined")
		print("Disabling DeepMIMO services: scenario upload (zip, images, rt source)")

# Configure Ray Tracing Versions (before importing the pipeline modules)
dm.config('wireless_insite_version', "4.0.1")  # E.g. '3.3.0', '4.0.1'

# from deepmimo.pipelines.wireless_insite.insite_raytracer import raytrace_insite
from deepmimo.pipelines.sionna_rt.sionna_raytracer import raytrace_sionna

# Absolute (!!) Paths
# OSM_ROOT = "/home/jamorais/osm_root" # Windows
# OSM_ROOT = OSM_ROOT.replace('C:', '/mnt/c') # WSL
OSM_ROOT = os.path.join(os.getcwd(), "osm_root")

df = pd.read_csv('./dev/params_20cities_t.csv')

counter = 0 # to run the same pipeline multiple times

#%% Run pipeline

for index, row in df.iterrows():
	print(f"\n{'=' * 50}\nSTARTING SCENARIO {index + 1}/{len(df)}: {row['name']}\n{'=' * 50}")

	# RT Phase 1: Load GPS coordinates from CSV
	load_params_from_row(row, p)

	# RT Phase 2: Extract OSM data, City Name, and Satellite View
	counter += 1
	osm_folder = os.path.join(OSM_ROOT, row['name']) + f'_{counter}'
	fetch_osm_scene(p['min_lat'], p['min_lon'], p['max_lat'], p['max_lon'],
					osm_folder, output_formats=['insite','sionna'])
	p['origin_lat'], p['origin_lon'] = get_origin_coords(osm_folder)

	p['city'] = get_city_name(p['origin_lat'], p['origin_lon'], GMAPS_API_KEY)
	sat_view_path = fetch_satellite_view(p['min_lat'], p['min_lon'], p['max_lat'], p['max_lon'],
										 GMAPS_API_KEY, osm_folder)
	
	# RT Phase 3: Generate RX and TX positions
	rx_pos = gen_rx_grid(p)  # N x 3 (N ~ 100k)
	tx_pos = gen_tx_pos(p)   # M x 3 (M ~ 3)
	
	# Optional: Round positions (visually better)
	rx_pos = np.round(rx_pos, p['pos_prec'])
	tx_pos = np.round(tx_pos, p['pos_prec'])
	
	print('Starting RT')
	osm_folder = os.path.join(OSM_ROOT, "simple_reflector")

	# RT Phase 4: Run Wireless InSite ray tracing
	# rt_path = raytrace_insite(osm_folder, tx_pos, rx_pos, **p)
	rt_path = raytrace_sionna(osm_folder, tx_pos, rx_pos, **p)

	# RT Phase 5: Convert to DeepMIMO format
	scen_name = dm.convert(rt_path, overwrite=True)

	# RT Phase 6: Test Conversion
	dataset = dm.load(scen_name)
	dataset.plot_coverage(dataset.los, scat_sz=40)
	dataset.plot_coverage(dataset.pwr[:, 0], scat_sz=40)

	# RT Phase 7: Upload (zip rt source)
	scen_name = dm.zip(rt_path)
	dm.upload(scen_name, key=DEEPMIMO_API_KEY)
	dm.upload_images(scen_name, img_paths=[sat_view_path],  key=DEEPMIMO_API_KEY)
	dm.upload_rt_source(scen_name, rt_zip_path=dm.zip(rt_path), key=DEEPMIMO_API_KEY)
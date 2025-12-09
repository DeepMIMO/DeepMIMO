"""Run a pipeline that uses OSM data.

"""

#%% Imports

import logging
import os

import numpy as np
import pandas as pd

logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

import deepmimo as dm  # type: ignore
from deepmimo.pipelines.blender_osm import fetch_osm_scene
from deepmimo.pipelines.txrx_placement import gen_rx_grid
from deepmimo.pipelines.utils.pipeline_utils import get_origin_coords, load_params_from_row

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
dm.config("wireless_insite_version", "3.3.0")  # E.g. '3.3.0', '4.0.1'

# from deepmimo.pipelines.wireless_insite.insite_raytracer import raytrace_insite
from pipeline_params import p

from deepmimo.pipelines.sionna_rt.sionna_raytracer import raytrace_sionna

# from pipeline_params import p

# Absolute (!!) Paths
# OSM_ROOT = "/home/jamorais/osm_root" # Windows
# OSM_ROOT = OSM_ROOT.replace('C:', '/mnt/c') # WSL
OSM_ROOT = os.path.join(os.getcwd(), "osm_root")
os.makedirs(OSM_ROOT, exist_ok=True)

df = pd.read_csv("params_20cities_t.csv")

#%% Run pipeline

for index, row in df.iterrows():
	print(f"\n{'=' * 50}\nSTARTING SCENARIO {index + 1}/{len(df)}: {row['name']}\n{'=' * 50}")

	# RT Phase 1: Load GPS coordinates from CSV
	load_params_from_row(row, p)

	# RT Phase 2: Extract OSM data, City Name, and Satellite View
	osm_folder = os.path.join(OSM_ROOT, row["name"])
	fetch_osm_scene(p["min_lat"], p["min_lon"], p["max_lat"], p["max_lon"],
					osm_folder, output_formats=["sionna"])
	p["origin_lat"], p["origin_lon"] = get_origin_coords(osm_folder)

	# p['city'] = get_city_name(p['origin_lat'], p['origin_lon'], GMAPS_API_KEY)
	# sat_view_path = fetch_satellite_view(p['min_lat'], p['min_lon'], p['max_lat'], p['max_lon'],
	# 									 GMAPS_API_KEY, osm_folder)

	# RT Phase 3: Generate RX and TX positions
	rx_pos = gen_rx_grid(p)  # N x 3 (N ~ 100k)
	tx_pos = np.array([[190, 106, 20]]) # M x 3 (M ~ 3)
	# tx_pos = np.array([[190, 106, 20]]) # M x 3 (M ~ 3)

	# Optional: Round positions (visually better)
	rx_pos = np.round(rx_pos, p["pos_prec"])
	tx_pos = np.round(tx_pos, p["pos_prec"])

	# plt.scatter(rx_pos[:, 0], rx_pos[:, 1], s=5)
	# plt.scatter(tx_pos[:, 0], tx_pos[:, 1], c='r')
	# plt.scatter(tx_pos_seq[:, 0], tx_pos_seq[:, 1], c='r', s=5)
	# plt.show()

	print("Starting RT")

	# RT Phase 4: Run Wireless InSite ray tracing
	print("Starting RT")
	rt_path = raytrace_sionna(osm_folder, tx_pos, rx_pos, **p)
	break
	# NOTE: with sionna 1.0, the roads are not exported correctly
	# Uncomment the following line in deepmimo/pipelines/utils/blender_utils.py
	# Reject all roads because of sionna 1.1 material bug
	# REJECTED_ROAD_KEYWORDS += TIERS[1] + TIERS[2]

	# RT Phase 5: Convert to DeepMIMO format
	# scen_name = dm.convert(rt_path, scenario_name=row['name']+f'_{i}', overwrite=True)

	# RT Phase 6: Test Conversion
	# dataset = dm.load(scen_name)
	# dataset.los.plot()
	# dataset.pwr.plot()

	# RT Phase 7: Upload (zip rt source)
	# dm.upload(scen_name, key=DEEPMIMO_API_KEY)
	# dm.upload_images(scen_name, img_paths=[sat_view_path],  key=DEEPMIMO_API_KEY)
	# dm.upload_rt_source(scen_name, rt_zip_path=dm.zip(rt_path), key=DEEPMIMO_API_KEY)
	# break

# %%

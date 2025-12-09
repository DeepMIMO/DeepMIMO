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
from deepmimo.pipelines.utils.geo_utils import fetch_satellite_view, get_city_name
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

# from deepmimo.pipelines.sionna_rt.sionna_raytracer import raytrace_sionna
from pipeline_params import p

from deepmimo.pipelines.wireless_insite.insite_raytracer import raytrace_insite

# Absolute (!!) Paths
# OSM_ROOT = "/home/jamorais/osm_root" # Windows
# OSM_ROOT = OSM_ROOT.replace('C:', '/mnt/c') # WSL
OSM_ROOT = os.path.join(os.getcwd(), "osm_root")

df = pd.read_csv("../dev/bounding_boxes2.csv")

#%% Run pipeline
import pickle

for index, row in df.iterrows():
	print(f"\n{'=' * 50}\nSTARTING SCENARIO {index + 1}/{len(df)}: {row['name']}\n{'=' * 50}")

	# RT Phase 1: Load GPS coordinates from CSV
	load_params_from_row(row, p)

	# RT Phase 2: Extract OSM data, City Name, and Satellite View
	osm_folder = os.path.join(OSM_ROOT, row["name"])
	fetch_osm_scene(p["min_lat"], p["min_lon"], p["max_lat"], p["max_lon"],
					osm_folder, output_formats=["insite"])
	p["origin_lat"], p["origin_lon"] = get_origin_coords(osm_folder)

	p["city"] = get_city_name(p["origin_lat"], p["origin_lon"], GMAPS_API_KEY)
	sat_view_path = fetch_satellite_view(p["min_lat"], p["min_lon"], p["max_lat"], p["max_lon"],
										 GMAPS_API_KEY, osm_folder)

	break

p["osm_folder_backup"]	= osm_folder
p["sat_view_path"]	= "d:\\Joao\\DeepMIMO\\osm_root\\asu_campus_3p5_dyn\\satellite_view.png"
with open("p.pkl", "wb") as f:
	pickle.dump(p, f, protocol=pickle.HIGHEST_PROTOCOL)


#%%
import pickle

with open("p.pkl", "rb") as f:
	p = pickle.load(f)
osm_folder = p["osm_folder_backup"]
import deepmimo as dm

d = dm.load("asu_campus_3p5")

seq_idxs1 = dm.LinearPath(d.rx_pos, [170, -115], [170, 85], n_steps=200).idxs
seq_idxs2 = dm.LinearPath(d.rx_pos, [170, 85], [-130, 85], n_steps=300).idxs
seq_idxs = np.concatenate([seq_idxs1, seq_idxs2]) # 500 of them

tx_pos_seq = d.rx_pos[seq_idxs[::5]] # 100 of them
tx_pos_seq[:,2] += 1.5

for i, moving_tx_pos in enumerate(tx_pos_seq):

	# RT Phase 3: Generate RX and TX positions
	rx_pos = gen_rx_grid(p)  # N x 3 (N ~ 100k)
	tx_pos = np.array([[190, 106, 20], moving_tx_pos.tolist()]) # M x 3 (M ~ 3)
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
	rt_path = raytrace_insite(osm_folder, tx_pos, rx_pos, **p)

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

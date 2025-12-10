"""Run a pipeline that uses static user positions.

This file is used to ray tracer a static dataset, i.e., a dataset where users are fixed.

Steps:
1. Generate user positions
2. Run ray tracing
3. Convert to DeepMIMO format
4. Test Conversion
"""

#%% Imports

import os

import numpy as np

import deepmimo as dm  # type: ignore
from deepmimo.pipelines.txrx_placement import gen_plane_grid

# Configure Ray Tracing Versions (before importing the pipeline modules)
dm.config("wireless_insite_version", "4.0.1")  # E.g. '3.3.0', '4.0.1'

# from deepmimo.pipelines.wireless_insite.insite_raytracer import raytrace_insite
from deepmimo.pipelines.sionna_rt.sionna_raytracer import raytrace_sionna

# Absolute (!!) Paths
OUT_FOLDER = os.path.join(os.getcwd(), "osm_root")

from pipeline_params import p

#%% Run pipeline

out_folder = os.path.join(OUT_FOLDER, "simple_reflector")

rx_pos = gen_plane_grid(0, 40, 0, 40, 2, 1.5)

tx_pos = np.array([[0, 0, 10]])

# Run Wireless InSite ray tracing
# rt_path = raytrace_insite(out_folder, tx_pos, rx_pos, **p)
rt_path = raytrace_sionna(out_folder, tx_pos, rx_pos, **p)

# Convert to DeepMIMO format
scen_name = dm.convert(rt_path, overwrite=True)

# Test Conversion
dataset = dm.load(scen_name)
dataset.plot_coverage(dataset.los, scat_sz=40)
dataset.plot_coverage(dataset.pwr[:, 0], scat_sz=40)

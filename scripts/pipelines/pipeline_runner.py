"""OSM -> Sionna RT -> DeepMIMO pipeline runner.

Reads a CSV of city bounding boxes and runs the full pipeline for each row:
  1. Download OSM building footprints and generate a Mitsuba scene
  2. Place TX and RX grids in local Cartesian coordinates
  3. Run Sionna RT ray tracing
  4. Convert to DeepMIMO format

Usage:
    python pipeline_runner.py [bounding_boxes.csv]

CSV format (see pipeline_csv_gen.py to generate it):
    name,min_lat,min_lon,max_lat,max_lon,bs_lat,bs_lon,bs_height
    city_0_munich_3p5,48.1355,11.5735,48.1395,11.5795,48.137,11.576,25

Set OSM_ROOT below (or CUDA_VISIBLE_DEVICES) before running.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import deepmimo as dm
from deepmimo.pipelines.osm_to_mitsuba import generate_scene
from deepmimo.pipelines.sionna_rt.sionna_raytracer import raytrace_sionna
from deepmimo.pipelines.txrx_placement import gen_rx_grid, gen_tx_pos
from deepmimo.pipelines.utils.pipeline_utils import get_origin_coords, load_params_from_row

from pipeline_params import p  # noqa: E402

# --- Configuration -----------------------------------------------------------

# Root folder for OSM scenes and ray-tracing exports
OSM_ROOT = os.path.join(os.getcwd(), "osm_root")
os.makedirs(OSM_ROOT, exist_ok=True)

# GPU to use; set "" to run on CPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# -----------------------------------------------------------------------------


def _scenario_exists(name: str) -> bool:
    return Path(dm.get_scenarios_dir(), name).is_dir()


def run_scenario(row: pd.Series, params: dict) -> str:
    """Run the full pipeline for one CSV row and return the scenario name."""
    load_params_from_row(row, params)
    osm_folder = os.path.join(OSM_ROOT, row["name"])

    # Step 1: OSM -> Mitsuba scene
    generate_scene(
        minlat=params["min_lat"],
        minlon=params["min_lon"],
        maxlat=params["max_lat"],
        maxlon=params["max_lon"],
        scene_folder=osm_folder,
    )
    params["origin_lat"], params["origin_lon"] = get_origin_coords(osm_folder)

    # Step 2: TX and RX placement
    rx_pos = np.round(gen_rx_grid(params), params["pos_prec"])
    tx_pos = np.round(gen_tx_pos(params), params["pos_prec"])

    # Step 3: Ray tracing
    rt_path = raytrace_sionna(osm_folder, tx_pos, rx_pos, **params)

    # Step 4: Convert to DeepMIMO
    return dm.convert(rt_path, scenario_name=row["name"], overwrite=True)


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} scenarios from {csv_path}\n")

    for index, row in df.iterrows():
        name = row["name"]
        print(f"{'=' * 60}")
        print(f"Scenario {index + 1}/{len(df)}: {name}")
        print(f"{'=' * 60}")

        if _scenario_exists(name):
            print(f"  Skipping — '{name}' already converted.\n")
            continue

        try:
            scen_name = run_scenario(row, p)
            print(f"  Done: {scen_name}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR in '{name}': {exc}\n")


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "bounding_boxes.csv"
    main(csv_file)

"""Script for generating bounding box CSV files from city coordinates.

This script:
1. Reads city coordinates from worldcities.csv
2. For each coordinate, generates a bounding box with base stations
3. Validates BS locations against building footprints
4. Saves valid scenarios to a CSV file

Configuration parameters control:
- BS placement strategy (on_top or outside buildings)
- Number of base stations per scenario
- BS height defaults
- City population threshold
- Bounding box dimensions

Todo:
    - Configure cell splitting feature:
        - Use cell splitting to split bounding box into smaller cells
        - Configure cell size to be ~80m longer in x and y compared to NY (~200x400m)
        - Save individual cells in bbox_cells csv file
    - Determine cell size (in meters) (not from constants in another file) 
    - osm_utils.py has many constants that are unnecessary
    - Test and improve BS placement algorithm 

"""

#%%
import random
import time

import pandas as pd

from deepmimo.pipelines.utils.osm_utils import get_buildings
from deepmimo.pipelines.utils.pipeline_utils import (
    ScenarioBboxInfo,
    generate_bs_positions,
    plot_scenario,
)

# Configuration parameters
DEFAULT_BS_HEIGHT = 10.0  # Default base station height in meters
MIN_CITY_POPULATION = 5000000  # Minimum city population to consider
NUM_BS = 3  # Number of base stations per scenario
BS_ALGORITHM = "uniform"  # 'uniform' or 'random'
BS_PLACEMENT = "outside"  # 'outside' or 'on_top'
PLOT_ENABLED = False  # Whether to generate plots

# Bounding box dimensions
DELTA_LAT = 0.003  # delta degrees in latitude (for bounding box size)
DELTA_LON = 0.003  # delta degrees in longitude (for bounding box size)

#%%
if __name__ == "__main__":
    """Main function to generate and save bounding boxes."""
    random.seed(42)

    # Load city coordinates
    cities = pd.read_csv("./scripts/worldcities_t.csv")
    urban_cities = cities[cities["population"] > MIN_CITY_POPULATION]

    # Generate bounding boxes
    bounding_boxes = []
    skipped = 0

    for box_idx, (_, city) in enumerate(urban_cities[:1].iterrows()):
        city_lat, city_lon = city["lat"], city["lng"]
        city_name = city["city"].lower().replace(" ", "")

        # First check if there are any significant buildings in the area
        buildings = get_buildings(city_lat, city_lon)
        if not buildings:
            print(f"Could not fetch buildings for {city['city']}, skipping")
            skipped += 1
            continue

        # Generate BS positions
        bs_lats, bs_lons, bs_heights = generate_bs_positions(
            city_lat, city_lon,
            num_bs=NUM_BS,
            buildings=buildings,
            algorithm=BS_ALGORITHM,
            placement=BS_PLACEMENT,
            default_height=DEFAULT_BS_HEIGHT,
            delta_lat=DELTA_LAT,
            delta_lon=DELTA_LON
        )

        if not bs_lats:  # Skip if no valid BS locations found
            print(f"Warning: No valid BS locations found for {city['city']}, skipping")
            skipped += 1
            continue

        # Create scenario info
        bbox = ScenarioBboxInfo(
            name=f"city_{box_idx}_{city_name}_3p5",
            minlat=city_lat - DELTA_LAT/2,
            maxlat=city_lat + DELTA_LAT/2,
            minlon=city_lon - DELTA_LON/2,
            maxlon=city_lon + DELTA_LON/2,
            bs_lats=bs_lats,
            bs_lons=bs_lons,
            bs_heights=bs_heights
        )

        bounding_boxes.append(bbox.to_dict())

        # Be kind to the OSM server
        time.sleep(1)
        break # do only for one city for now

    print(f"Generated {len(bounding_boxes)} valid boxes, skipped {skipped} boxes with no buildings")

    # Save to CSV
    df = pd.DataFrame(bounding_boxes)
    print("DataFrame columns:", df.columns)  # Debug print
    df.to_csv("./scripts/bounding_boxes.csv", index=False)
    print(f"Saved {len(bounding_boxes)} valid bounding boxes to bounding_boxes.csv")

    # Plot one scenario
    plot_scenario(bounding_boxes[0])


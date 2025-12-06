"""Utility functions for DeepMIMO pipeline operations.

This module provides utilities used across different pipeline stages in the DeepMIMO project:

Pipeline Runner Utilities:
- Command execution and output streaming (used in pipeline_runner.py)
- Origin coordinate handling for OSM/InSite integration
- Parameter loading from scenario configurations

CSV Generation Utilities:
- Base station positioning and validation (used in pipeline_csv_gen.py)
- City-based scenario generation for creating bounding box CSVs
- Scenario visualization and validation for CSV entries

The utilities here are designed to be reusable across different pipeline stages
and provide consistent handling of coordinates, parameters, and system operations.
"""

import os
import random
import subprocess
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

from deepmimo.pipelines.utils.osm_utils import (
    find_nearest_clear_location,
    is_point_clear_of_buildings,
)

###############################################################################
# Pipeline Runner Utilities
# Used in pipeline_runner.py for InSite execution and coordinate management
###############################################################################


def run_command(command: list[str], description: str) -> None:
    """Run a shell command and stream output in real-time.

    Args:
        command (List[str]): Command to run
        description (str): Description of the command for logging

    """
    print(f"\n🚀 Starting: {description}...\n")
    print("\t Running: ", " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # Stream the output in real-time
    for line in iter(process.stdout.readline, ""):
        print(line, end="")  # Print each line as it arrives

    process.stdout.close()
    process.wait()

    print(f"\n✅ {description} completed!\n")


def get_origin_coords(osm_folder: str) -> tuple[float, float]:
    """Read the origin coordinates from the OSM folder.

    Args:
        osm_folder (str): Path to the OSM folder

    Returns:
        Tuple[float, float]: Origin coordinates (latitude, longitude)

    """
    origin_file = os.path.join(osm_folder, "osm_gps_origin.txt")
    # Check if the file exists
    if not os.path.exists(origin_file):
        raise FileNotFoundError(
            f"❌ Origin coordinates file not found at {origin_file}\n"
            "Ensure that Blender has been run successfully.",
        )

    with open(origin_file) as f:
        origin_coords = f.read().split("\n")
    return float(origin_coords[0]), float(origin_coords[1])


def _split_coords(x: str) -> np.ndarray:
    """Split comma-separated coordinates into float array."""
    return np.array(x.split(",") if type(x) == str else [x]).astype(np.float32)


def load_params_from_row(row, params_dict):
    """Load parameters from a DataFrame row into a parameters dictionary.

    Args:
        row (pandas.Series): Row from a DataFrame containing parameters
        params_dict (Dict): Dictionary of parameters to update

    """
    # Update parameters that exist in both the row and params dict
    for key in params_dict.keys():
        if key in row.index:
            params_dict[key] = row[key]

    # Handle base station coordinates separately
    params_dict["bs_lats"] = _split_coords(row["bs_lat"])
    params_dict["bs_lons"] = _split_coords(row["bs_lon"])
    params_dict["bs_heights"] = _split_coords(row["bs_height"])


###############################################################################
# Scenario Data Structures
# Core data classes for CSV scenario information management
###############################################################################


@dataclass
class ScenarioBboxInfo:
    """Class to store scenario information including bounding box and multiple BS parameters."""

    name: str
    minlat: float
    minlon: float
    maxlat: float
    maxlon: float
    bs_lats: list[float]  # List of BS latitudes
    bs_lons: list[float]  # List of BS longitudes
    bs_heights: list[float]  # List of BS heights

    def to_dict(self) -> dict[str, str]:
        """Convert scenario info to dictionary format for CSV storage."""
        # Convert lists of coordinates to comma-separated strings
        bs_lats_str = ",".join([f"{lat:.8f}" for lat in self.bs_lats])
        bs_lons_str = ",".join([f"{lon:.8f}" for lon in self.bs_lons])
        bs_heights_str = ",".join([f"{h:.1f}" for h in self.bs_heights])

        return {
            "name": self.name,
            "min_lat": f"{self.minlat:.8f}",
            "min_lon": f"{self.minlon:.8f}",
            "max_lat": f"{self.maxlat:.8f}",
            "max_lon": f"{self.maxlon:.8f}",
            "bs_lat": bs_lats_str,
            "bs_lon": bs_lons_str,
            "bs_height": bs_heights_str,
        }


###############################################################################
# Base Station Generation and Validation
# Core functions for generating and validating BS positions for CSV scenarios
###############################################################################


def validate_and_adjust_point(
    lat: float,
    lon: float,
    buildings: list,
    placement: str = "outside",
    default_height: float = 10.0,
) -> tuple[float, float, float, bool]:
    """Validate point and adjust location based on placement strategy.

    Args:
        lat (float): Initial latitude to check
        lon (float): Initial longitude to check
        buildings (List): List of building polygons
        placement (str): Placement strategy ('outside' or 'on_top'). Defaults to 'outside'.
        default_height (float): Default height for BS placement in meters

    Returns:
        Tuple[float, float, float, bool]: Tuple containing:
            - latitude of valid location (or last attempt)
            - longitude of valid location (or last attempt)
            - height of BS placement
            - True if location is valid, False if no valid location found

    """
    point = Point(lon, lat)

    if placement == "outside":
        # Try to find location outside buildings (max 3 attempts)
        for _ in range(3):
            if is_point_clear_of_buildings(point, buildings):
                return lat, lon, default_height, True

            # Find new location clear of buildings
            new_lat, new_lon = find_nearest_clear_location(lat, lon, buildings)

            if is_point_clear_of_buildings(Point(new_lon, new_lat), buildings):
                return new_lat, new_lon, default_height, True

            lat, lon = new_lat, new_lon

        return lat, lon, default_height, False

    if placement == "on_top":
        # Find building at point
        for building in buildings:
            if Point(lon, lat).within(building.geometry):
                # Add 2 meters to building height for BS placement
                bs_height = building.height + 2.0
                return lat, lon, bs_height, True

        # If no building found at point, try nearest building
        min_dist = float("inf")
        best_location = None
        best_height = default_height

        for building in buildings:
            dist = Point(lon, lat).distance(building.geometry)
            if dist < min_dist:
                min_dist = dist
                # Get centroid of building for BS placement
                centroid = building.geometry.centroid
                best_location = (centroid.y, centroid.x)
                best_height = building.height + 2.0

        if best_location:
            return best_location[0], best_location[1], best_height, True

        return lat, lon, default_height, False

    raise ValueError(f"Unknown placement strategy: {placement}")


def generate_uniform_positions(
    city_lat: float,
    city_lon: float,
    num_bs: int,
    delta_lat: float,
    delta_lon: float,
) -> list[tuple[float, float]]:
    """Generate uniformly spaced positions in the bounding box for CSV generation.

    For different numbers of BS:
    - 1 BS: Center
    - 2 BS: Diagonal corners
    - 3 BS: Triangle formation
    - 4 BS: Square formation

    Args:
        city_lat (float): Center latitude
        city_lon (float): Center longitude
        num_bs (int): Number of base stations
        delta_lat (float): Latitude span of bounding box
        delta_lon (float): Longitude span of bounding box

    Returns:
        List[Tuple[float, float]]: List of (lat, lon) positions

    """
    # Calculate box boundaries (80% of full box size to keep BS away from edges)
    margin = 0.1  # 10% margin from edges
    lat_range = delta_lat * (1 - 2 * margin)
    lon_range = delta_lon * (1 - 2 * margin)
    min_lat = city_lat - lat_range / 2
    min_lon = city_lon - lon_range / 2

    positions = []

    if num_bs == 1:
        # Center position
        positions.append((city_lat, city_lon))

    elif num_bs == 2:
        # Diagonal corners
        positions.extend(
            [
                (min_lat, min_lon),
                (min_lat + lat_range, min_lon + lon_range),
            ],
        )

    elif num_bs == 3:
        # Triangle formation
        positions.extend(
            [
                (min_lat, min_lon),  # Bottom left
                (min_lat, min_lon + lon_range),  # Bottom right
                (min_lat + lat_range, min_lon + lon_range / 2),  # Top center
            ],
        )

    elif num_bs == 4:
        # Square formation
        positions.extend(
            [
                (min_lat, min_lon),  # Bottom left
                (min_lat, min_lon + lon_range),  # Bottom right
                (min_lat + lat_range, min_lon),  # Top left
                (min_lat + lat_range, min_lon + lon_range),  # Top right
            ],
        )

    else:
        raise NotImplementedError(
            f"Number of BSs {num_bs} not supported. Maximum number of BSs is 4.",
        )

    return positions


def generate_bs_positions(
    city_lat: float,
    city_lon: float,
    num_bs: int,
    buildings: list,
    algorithm: str = "uniform",
    placement: str = "outside",
    default_height: float = 10.0,
    delta_lat: float = 0.003,
    delta_lon: float = 0.003,
) -> tuple[list[float], list[float], list[float]]:
    """Generate and validate base station positions for CSV scenario generation.

    Args:
        city_lat (float): City center latitude
        city_lon (float): City center longitude
        num_bs (int): Number of base stations to generate
        buildings (List): List of building polygons in the area
        algorithm (str, optional): BS positioning algorithm ('uniform' or 'random'). Defaults to 'uniform'.
        placement (str, optional): BS placement strategy ('outside' or 'on_top'). Defaults to 'outside'.
        default_height (float): Default height for BS placement in meters
        delta_lat (float): Latitude span of bounding box
        delta_lon (float): Longitude span of bounding box

    Returns:
        Tuple[List[float], List[float], List[float]]: Lists of BS latitudes, longitudes, and heights

    """
    bs_lats, bs_lons, bs_heights = [], [], []

    if algorithm == "random":
        # Random positioning
        for _ in range(num_bs):
            offset_lat = random.uniform(-delta_lat / 4, delta_lat / 4)
            offset_lon = random.uniform(-delta_lon / 4, delta_lon / 4)
            test_lat = city_lat + offset_lat
            test_lon = city_lon + offset_lon

            bs_lat, bs_lon, height, is_valid = validate_and_adjust_point(
                test_lat,
                test_lon,
                buildings,
                placement,
                default_height,
            )
            if is_valid:
                bs_lats.append(bs_lat)
                bs_lons.append(bs_lon)
                bs_heights.append(height)

    else:  # uniform positioning
        # Generate uniform positions
        positions = generate_uniform_positions(city_lat, city_lon, num_bs, delta_lat, delta_lon)

        # Validate and adjust each position
        for test_lat, test_lon in positions:
            bs_lat, bs_lon, height, is_valid = validate_and_adjust_point(
                test_lat,
                test_lon,
                buildings,
                placement,
                default_height,
            )
            if is_valid:
                bs_lats.append(bs_lat)
                bs_lons.append(bs_lon)
                bs_heights.append(height)

    return bs_lats, bs_lons, bs_heights


###############################################################################
# Visualization Utilities
# Plotting functions for CSV scenario validation and debugging
###############################################################################


def plot_scenario(bbox_info: dict[str, str]):
    """Plot the bounding box and BS positions for a CSV scenario.

    Args:
        bbox_info (Dict[str, str]): Dictionary containing bounding box information

    """
    # Extract coordinates
    minlat, maxlat = float(bbox_info["min_lat"]), float(bbox_info["max_lat"])
    minlon, maxlon = float(bbox_info["min_lon"]), float(bbox_info["max_lon"])
    bs_lats = [float(x) for x in bbox_info["bs_lat"].split(",")]
    bs_lons = [float(x) for x in bbox_info["bs_lon"].split(",")]

    # Create figure
    plt.figure(figsize=(8, 8))

    # Plot bounding box
    plt.plot(
        [minlon, maxlon, maxlon, minlon, minlon],
        [minlat, minlat, maxlat, maxlat, minlat],
        "k-",
        label="Bounding Box",
    )

    # Plot BS positions
    plt.scatter(bs_lons, bs_lats, c="red", marker="^", s=100, label="Base Stations")

    # Add labels and title
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Scenario: {bbox_info['name']}\n{len(bs_lats)} Base Stations")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

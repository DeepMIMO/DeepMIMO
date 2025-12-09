"""Wireless Insite Ray Tracing Parameters.

This module provides parameter handling for Wireless Insite ray tracing simulations.

This module provides:
- Parameter parsing from Wireless Insite setup files
- Standardized parameter representation
- Parameter validation and conversion utilities
- Default parameter configuration

The module serves as the interface between Wireless Insite's parameter format
and DeepMIMO's standardized ray tracing parameters.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deepmimo.config import config
from deepmimo.consts import BBOX_PAD, RAYTRACER_NAME_WIRELESS_INSITE
from deepmimo.rt_params import RayTracingParameters

from .setup_parser import parse_file


def read_rt_params(sim_folder: str | Path) -> dict:
    """Read Wireless Insite RT parameters from a folder."""
    return InsiteRayTracingParameters.read_rt_params(sim_folder).to_dict()


def _get_gps_bbox(
    origin_lat: float,
    origin_lon: float,
    studyarea_vertices: np.ndarray,
    pad: float = BBOX_PAD,
) -> tuple[float, float, float, float]:
    """Get the GPS bounding box of a Wireless Insite simulation.

    This is an approximated method that considers the earth round.
    For a typical scenario, the error in latitude should be < 20 micro degress,
    and the error in longitude should be < 10 micro degress, which corresponds to
    an error of < 2 m in the vertical size of the bounding box, and < 1 m in the
    horizontal size of the bounding box.

    Args:
        origin_lat (float): Latitude of the origin
        origin_lon (float): Longitude of the origin
        studyarea_vertices (np.ndarray): Vertices of the study area
        pad (float): Padding for the bounding box in degrees

    Returns:
        Tuple[float, float, float, float]: Bounding box of the study area

    """
    if origin_lat == 0 and origin_lon == 0:
        return (0, 0, 0, 0)  # Default bounding box if origin is not available

    min_vertex = np.min(studyarea_vertices, axis=0)[:2]
    max_vertex = np.max(studyarea_vertices, axis=0)[:2]

    study_min_x, study_min_y = min_vertex
    study_max_x, study_max_y = max_vertex

    # Get min and max latitude and longitude
    x_range = study_max_x - study_min_x - 2 * pad
    y_range = study_max_y - study_min_y - 2 * pad

    # Transform xy distances to approximate lat/lon distances
    meter_per_degree_lat = 111320
    meter_per_degree_lon = 111320 * np.cos(np.radians(origin_lat))

    lat_range = y_range / meter_per_degree_lat
    lon_range = x_range / meter_per_degree_lon

    # Get min and max latitude and longitude
    min_lat = origin_lat - lat_range / 2
    max_lat = origin_lat + lat_range / 2
    min_lon = origin_lon - lon_range / 2
    max_lon = origin_lon + lon_range / 2

    return min_lat, min_lon, max_lat, max_lon


# Public alias for tests and external callers.
get_gps_bbox = _get_gps_bbox


@dataclass
class InsiteRayTracingParameters(RayTracingParameters):
    """Wireless Insite ray tracing parameter representation.

    This class extends the base RayTracingParameters with Wireless Insite-specific
    settings for antenna configuration, APG acceleration, and diffuse scattering.

    Attributes:
        raytracer_name (str): Name of ray tracing engine (from constants)
        raytracer_version (str): Version of ray tracing engine
        frequency (float): Center frequency in Hz
        max_path_depth (int): Maximum number of interactions (R + D + S + T)
        max_reflections (int): Maximum number of reflections (R)
        max_diffractions (int): Maximum number of diffractions (D)
        max_scattering (int): Maximum number of diffuse scattering events (S)
        max_transmissions (int): Maximum number of transmissions (T)
        diffuse_reflections (int): Reflections allowed in paths with diffuse scattering
        diffuse_diffractions (int): Diffractions allowed in paths with diffuse scattering
        diffuse_transmissions (int): Transmissions allowed in paths with diffuse scattering
        diffuse_final_interaction_only (bool): Whether to only consider diffuse scattering
            at the final interaction
        diffuse_random_phases (bool): Whether to use random phases for diffuse scattering
        terrain_reflection (bool): Whether to allow reflections on terrain
        terrain_diffraction (bool): Whether to allow diffractions on terrain
        terrain_scattering (bool): Whether to allow scattering on terrain
        num_rays (int): Number of rays to launch per antenna
        ray_casting_method (str): Method for casting rays ('uniform' or other)
        synthetic_array (bool): Whether to use a synthetic array
        ray_casting_range_az (float): Ray casting range in azimuth (degrees)
        ray_casting_range_el (float): Ray casting range in elevation (degrees)
        raw_params (Dict): Original parameters from Wireless Insite

    Notes:
        All required parameters must come before optional ones in dataclasses.
        First come the base class required parameters (inherited), then the class-specific
        required parameters, then all optional parameters.

    """

    @classmethod
    def read_rt_params(cls, sim_folder: str | Path) -> "InsiteRayTracingParameters":
        """Read a Wireless Insite setup file and return a parameters object.

        Args:
            sim_folder (str | Path): Path to simulation folder containing .setup file

        Returns:
            InsiteRayTracingParameters: Object containing standardized parameters

        Raises:
            ValueError: If no .setup file found or multiple .setup files found
            FileNotFoundError: If simulation folder does not exist

        """
        sim_folder = Path(sim_folder)
        if not sim_folder.exists():
            msg = f"Simulation folder does not exist: {sim_folder}"
            raise ValueError(msg)

        # Find .setup file
        setup_files = list(sim_folder.glob("*.setup"))
        if not setup_files:
            msg = f"No .setup file found in {sim_folder}"
            raise ValueError(msg)
        if len(setup_files) > 1:
            msg = f"Multiple .setup files found in {sim_folder}"
            raise ValueError(msg)

        # Parse setup file
        setup_file = str(setup_files[0])
        document = parse_file(setup_file)

        # Select study area
        prim = next(iter(document.keys()))

        prim_vals = document[prim].values
        antenna_vals = prim_vals["antenna"].values
        waveform_vals = prim_vals["Waveform"].values
        studyarea_vals = prim_vals["studyarea"].values
        model_vals = studyarea_vals["model"].values
        apg_accel_vals = studyarea_vals["apg_acceleration"].values
        diffuse_scat_vals = studyarea_vals["diffuse_scattering"].values

        # Defaults that sometimes are not present in the setup file
        model_vals["ray_spacing"] = model_vals.get("ray_spacing", 0.25)
        model_vals["terrain_diffractions"] = model_vals.get("terrain_diffractions", "No")

        # Diffractions
        if "max_wedge_diffractions" in model_vals:
            pass  # all good, information present
        else:
            default_diffractions = diffuse_scat_vals.get("diffuse_diffractions", 0)
            if default_diffractions == 0:
                default_diffractions = 1 if model_vals["terrain_diffractions"] == "Yes" else 0
            model_vals["max_wedge_diffractions"] = default_diffractions

        # Transmissions
        model_vals["max_transmissions"] = model_vals.get("max_transmissions", 0)

        # Store raw parameters
        raw_params = {
            "antenna": antenna_vals,
            "waveform": waveform_vals,
            "studyarea": studyarea_vals,
            "model": model_vals,
            "apg_acceleration": apg_accel_vals,
            "diffuse_scattering": diffuse_scat_vals,
        }

        num_rays = int(360 // model_vals["ray_spacing"] * 180)

        # Compute the path depth from the number of interactions
        computed_path_depth_no_scattering = sum(
            [
                model_vals["max_reflections"],
                model_vals["max_wedge_diffractions"],
                model_vals["max_transmissions"],
            ],
        )

        # When scattering is enabled, the number of interactions may be smaller
        computed_path_depth_scattering = 0
        if diffuse_scat_vals["enabled"]:
            computed_path_depth_scattering = sum(
                [
                    diffuse_scat_vals["diffuse_reflections"],
                    diffuse_scat_vals["diffuse_diffractions"],
                    diffuse_scat_vals["diffuse_transmissions"],
                ],
            )

        actual_max_path_depth = min(
            apg_accel_vals["path_depth"],
            max(computed_path_depth_no_scattering, computed_path_depth_scattering),
        )

        # Get GPS Bounding Box
        origin_lat = studyarea_vals["boundary"].values["reference"].values["latitude"]
        origin_lon = studyarea_vals["boundary"].values["reference"].values["longitude"]
        studyarea_vertices = studyarea_vals["boundary"].data
        gps_bbox = _get_gps_bbox(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            studyarea_vertices=np.array(studyarea_vertices),
        )

        # Build standardized parameter dictionary
        params_dict = {
            # Ray Tracing Engine info
            "raytracer_name": RAYTRACER_NAME_WIRELESS_INSITE,
            "raytracer_version": config.get("wireless_insite_version"),
            # Frequency
            "frequency": waveform_vals["CarrierFrequency"],
            # Ray tracing interaction settings
            "max_path_depth": actual_max_path_depth,
            "max_reflections": model_vals["max_reflections"],
            "max_diffractions": model_vals["max_wedge_diffractions"],
            "max_scattering": int(diffuse_scat_vals["enabled"]),  # 1 if enabled, 0 if not
            "max_transmissions": model_vals[
                "max_transmissions"
            ],  # Insite does not support transmissions in our setup
            # Details on diffraction, scattering, and transmission
            "diffuse_reflections": diffuse_scat_vals["diffuse_reflections"],
            "diffuse_diffractions": diffuse_scat_vals["diffuse_diffractions"],
            "diffuse_transmissions": diffuse_scat_vals["diffuse_transmissions"],
            "diffuse_final_interaction_only": diffuse_scat_vals["final_interaction_only"],
            "diffuse_random_phases": False,  # Insite does not support random phases
            # Terrain interaction settings
            "terrain_reflection": bool(model_vals.get("terrain_reflections", 1)),
            "terrain_diffraction": model_vals["terrain_diffractions"] == "Yes",
            "terrain_scattering": bool(model_vals.get("terrain_scattering", 0)),
            # Ray casting settings
            "num_rays": num_rays,  # Insite uses ray spacing instead of explicit ray count
            "ray_casting_method": "uniform",  # Insite uses uniform ray casting
            "synthetic_array": True,  # Currently only synthetic arrays are supported
            # GPS Bounding Box
            "gps_bbox": gps_bbox,
            # Store raw parameters
            "raw_params": raw_params,
        }

        # Create and return parameters object
        return cls.from_dict(params_dict)


if __name__ == "__main__":
    # Test directory with setup files
    test_dir = r"./P2Ms/simple_street_canyon_test/"

    # Find .setup file in test directory
    setup_file = None
    for root, _, filenames in os.walk(test_dir):
        for filename in filenames:
            if filename.endswith(".setup"):
                setup_file = str(Path(root) / filename)
                break
        if setup_file:
            break

    if not setup_file:
        print(f"No .setup file found in {test_dir}")
        sys.exit(1)

    print(f"\nTesting setup extraction from: {setup_file}")
    print("-" * 50)

    # Extract setup information and print in a nicely formatted way
    setup_dict = InsiteRayTracingParameters.read_rt_params(setup_file)

    # Filter out raw_params to keep output cleaner

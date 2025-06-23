"""
AODT Ray Tracing Parameters.

This module provides parameter handling for AODT (Aerial Optical Digital Twin) ray tracing simulations.

This module provides:
- Parameter parsing from AODT parquet files
- Standardized parameter representation
- Parameter validation and conversion utilities
- Default parameter configuration

The module serves as the interface between AODT's parameter format
and DeepMIMO's standardized ray tracing parameters.
"""

import os
import pandas as pd
from dataclasses import dataclass
from typing import Dict
from pathlib import Path

from ...rt_params import RayTracingParameters
from ...consts import RAYTRACER_NAME_AODT
from ...config import config


def read_rt_params(rt_folder: str) -> Dict:
    """Read AODT RT parameters from a folder."""
    return AODTRayTracingParameters.read_rt_params(rt_folder).to_dict()


@dataclass
class AODTRayTracingParameters(RayTracingParameters):
    """AODT ray tracing parameter representation.
    
    This class extends the base RayTracingParameters with AODT-specific
    settings for ray tracing configuration and interaction handling.
    
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
        diffuse_final_interaction_only (bool): Whether to only consider diffuse scattering at final interaction
        diffuse_random_phases (bool): Whether to use random phases for diffuse scattering
        terrain_reflection (bool): Whether to allow reflections on terrain
        terrain_diffraction (bool): Whether to allow diffractions on terrain
        terrain_scattering (bool): Whether to allow scattering on terrain
        num_rays (int): Number of rays to launch per antenna
        ray_casting_method (str): Method for casting rays ('uniform' or other)
        synthetic_array (bool): Whether to use a synthetic array
        ray_casting_range_az (float): Ray casting range in azimuth (degrees)
        ray_casting_range_el (float): Ray casting range in elevation (degrees)
        raw_params (Dict): Original parameters from AODT
        
    Notes:
        All required parameters must come before optional ones in dataclasses.
        First come the base class required parameters (inherited), then the class-specific
        required parameters, then all optional parameters.
    """
    
    @classmethod
    def read_rt_params(cls, rt_folder: str | Path) -> 'AODTRayTracingParameters':
        """Read AODT RT parameters and return a parameters object.
        
        Args:
            rt_folder (str | Path): Path to folder containing AODT parameter files
            
        Returns:
            AODTRayTracingParameters: Object containing standardized parameters
            
        Raises:
            FileNotFoundError: If parameter files not found
            ValueError: If required parameters are missing or invalid
        """
        # Load scenario parameters
        scenario_file = os.path.join(rt_folder, 'scenario.parquet')
        if not os.path.exists(scenario_file):
            raise FileNotFoundError(f"scenario.parquet not found in {rt_folder}")
            
        df = pd.read_parquet(scenario_file)
        if len(df) == 0:
            raise ValueError("scenario.parquet is empty")
            
        # Get first row since parameters are the same for all rows
        params = df.iloc[0]
        
        # Store raw parameters
        raw_params = params.to_dict()
        
        # Create standardized parameters
        params_dict = {
            # Ray Tracing Engine info
            'raytracer_name': RAYTRACER_NAME_AODT,
            'raytracer_version': config.get('aodt_version', '1.0'),
            
            # Frequency
            'frequency': 0, # panel frequency - needs to be set in the future
            
            # Ray tracing interaction settings
            'max_path_depth': int(params['num_scene_interactions_per_ray']),
            'max_reflections': int(params['num_scene_interactions_per_ray']),  # AODT allows all interactions to be reflections
            'max_diffractions': 1,  # AODT only allows one diffraction per path
            'max_scattering': 1,  # AODT only allows one scattering per path
            'max_transmissions': int(params['num_scene_interactions_per_ray']),  # AODT allows all interactions to be transmissions
            
            # Details on diffraction, scattering, and transmission
            'diffuse_reflections': 1,  # AODT specify this in the documentation
            'diffuse_diffractions': 0,  # AODT doesn't specify this
            'diffuse_transmissions': 0,  # AODT doesn't specify this
            'diffuse_final_interaction_only': False,  # AODT allows diffuse scattering at any interaction
            'diffuse_random_phases': False,  # AODT doesn't specify this
            
            # Terrain interaction settings
            'terrain_reflection': True,  # AODT allows reflections on any surface
            'terrain_diffraction': False,  # AODT allows diffractions on any surface
            'terrain_scattering': True,  # AODT allows scattering on any surface
            
            # Ray casting settings
            'num_rays': int(params['num_emitted_rays_in_thousands'] * 1000),
            'ray_casting_method': 'uniform',  # AODT uses uniform ray casting
            'synthetic_array': True,  # AODT does not use synthetic arrays, but we take only one element
            'ray_casting_range_az': 360.0,  # AODT casts rays in all directions
            'ray_casting_range_el': 180.0,  # AODT casts rays in all directions
            
            # GPS Bounding Box
            'gps_bbox': (0, 0, 0, 0),  # AODT doesn't provide this yet
            
            # Store raw parameters
            'raw_params': raw_params,
        }
        
        # Create and return parameters object
        return cls.from_dict(params_dict) 
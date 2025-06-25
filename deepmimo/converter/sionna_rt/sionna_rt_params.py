"""
Sionna Ray Tracing Parameters.

This module provides parameter handling for Sionna ray tracing simulations.

This module provides:
- Parameter parsing from Sionna configuration files
- Standardized parameter representation
- Parameter validation and conversion utilities
- Default parameter configuration

The module serves as the interface between Sionna's parameter format
and DeepMIMO's standardized ray tracing parameters.
"""

import os
from dataclasses import dataclass
from typing import Dict

from ...general_utils import load_pickle
from ...rt_params import RayTracingParameters
from ...consts import RAYTRACER_NAME_SIONNA
from ...config import config


def read_rt_params(load_folder: str) -> Dict:
    """Read Sionna RT parameters from a folder."""
    return SionnaRayTracingParameters.read_rt_params(load_folder).to_dict()


@dataclass
class SionnaRayTracingParameters(RayTracingParameters):
    """Sionna ray tracing parameter representation.
    
    This class extends the base RayTracingParameters with Sionna-specific
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
        raw_params (Dict): Original parameters from Sionna
        
    Notes:
        All required parameters must come before optional ones in dataclasses.
        First come the base class required parameters (inherited), then the class-specific
        required parameters, then all optional parameters.
    """
    
    @classmethod
    def read_rt_params(cls, load_folder: str) -> 'SionnaRayTracingParameters':
        """Read Sionna RT parameters and return a parameters object.
        
        Args:
            load_folder (str): Path to folder containing Sionna parameter files
            
        Returns:
            SionnaRayTracingParameters: Object containing standardized parameters
            
        Raises:
            FileNotFoundError: If parameter files not found
            ValueError: If required parameters are missing or invalid
        """
        # Load original parameters
        raw_params = load_pickle(os.path.join(load_folder, 'sionna_rt_params.pkl'))
        
        # Raise error if los is not present
        if 'los' not in raw_params or not raw_params['los']:
            raise ValueError("los not found in Sionna RT parameters")
        
        # Raise error if arrays are not synthetic
        # if not raw_params['synthetic_array']:
        #     raise ValueError("arrays are not synthetic in Sionna RT parameters. "
        #                      "Multi-antenna arrays are not supported yet.")
        
        # NOTE: Sionna distributes these samples across antennas AND TXs
        n_tx, n_tx_ant = raw_params['tx_array_size'], raw_params['tx_array_num_ant']
        n_emmitters = n_tx * n_tx_ant
        n_rays = raw_params['num_samples'] // n_emmitters
        
        # Check if GPS origin is present (average point of scene, in geographic coordinates)
        if raw_params.get('min_lat', 0) != 0:
            gps_bbox = (raw_params['min_lat'], raw_params['min_lon'],
                        raw_params['max_lat'], raw_params['max_lon'])
        else:
            gps_bbox = (0,0,0,0) # default
        
        rt_method = raw_params.get('method', 'fibonacci')

        # Create standardized parameters
        params_dict = {
            # Ray Tracing Engine info
            'raytracer_name': RAYTRACER_NAME_SIONNA,
            'raytracer_version': raw_params.get('raytracer_version', config.get('sionna_version')),

            # Base required parameters
            'frequency': int(raw_params['frequency']),
            
            # Ray tracing interaction settings
            'max_path_depth': int(raw_params['max_depth']),
            'max_reflections': int(raw_params['max_depth']) if raw_params['reflection'] else 0,
            'max_diffractions': int(raw_params['diffraction']),  # Sionna only supports 1 diffraction event
            'max_scattering': int(raw_params['scattering']),   # Sionna only supports 1 scattering event
            'max_transmissions': 0, # Sionna does not support transmissions

            # Terrain interaction settings
            'terrain_reflection': bool(raw_params['reflection']), 
            'terrain_diffraction': raw_params['diffraction'],  # Sionna only supports 1 diffraction, may be on terrain
            'terrain_scattering': raw_params['scattering'],

            # Details on diffraction, scattering, and transmission
            'diffuse_reflections': int(raw_params['max_depth']) - 1, # Sionna only supports diffuse reflections
            'diffuse_diffractions': 0, # Sionna only supports 1 diffraction event, with no diffuse scattering
            'diffuse_transmissions': 0, # Sionna does not support transmissions
            'diffuse_final_interaction_only': True, # Sionna only supports diffuse scattering at final interaction
            'diffuse_random_phases': raw_params.get('scat_random_phases', True),

            'synthetic_array': raw_params.get('synthetic_array', True),
            'num_rays': -1 if rt_method != 'fibonacci' else n_rays, 
            'ray_casting_method': rt_method.replace('fibonacci', 'uniform'),
            # The alternative to fibonacci is exhaustive, for which the number of rays is not predictable

            # GPS Bounding Box
            'gps_bbox': gps_bbox,

            # Store raw parameters
            'raw_params': raw_params,
        }
        
        # Create and return parameters object
        return cls.from_dict(params_dict)
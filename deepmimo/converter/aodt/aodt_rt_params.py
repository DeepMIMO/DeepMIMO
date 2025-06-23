"""
AODT Ray Tracing Parameters Module.

This module handles reading and processing ray tracing parameters from the
scenario.parquet file in AODT format.
"""

import os
import pandas as pd
from typing import Dict, Any

def read_rt_params(rt_folder: str) -> Dict[str, Any]:
    """Read ray tracing parameters from scenario.parquet.

    Args:
        rt_folder (str): Path to folder containing scenario.parquet.

    Returns:
        Dict[str, Any]: Dictionary containing ray tracing parameters including:
            - num_emitted_rays: Number of emitted rays (in thousands)
            - num_scene_interactions: Maximum interactions per ray
            - max_paths: Maximum paths per RU-UE pair
            - ray_sparsity: Ray sparsity parameter
            - rx_sphere_radius: Receiver sphere radius in meters
            - diffuse_type: Type of diffuse scattering
            - enable_wideband: Whether wideband CFRs are enabled
            - duration: Simulation duration
            - interval: Time interval between snapshots
            - seed: Random seed if simulation is seeded

    Raises:
        FileNotFoundError: If scenario.parquet is not found.
        ValueError: If required parameters are missing.
    """
    scenario_file = os.path.join(rt_folder, 'scenario.parquet')
    if not os.path.exists(scenario_file):
        raise FileNotFoundError(f"scenario.parquet not found in {rt_folder}")

    # Read scenario parameters
    df = pd.read_parquet(scenario_file)
    if len(df) == 0:
        raise ValueError("scenario.parquet is empty")

    # Get first row since parameters are the same for all rows
    params = df.iloc[0]

    # Convert parameters to dictionary
    rt_params = {
        'num_emitted_rays': int(params['num_emitted_rays_in_thousands'] * 1000),
        'num_scene_interactions': int(params['num_scene_interactions_per_ray']),
        'max_paths': int(params['max_paths_per_ru_ue_pair']),
        'ray_sparsity': float(params['ray_sparsity']),
        'rx_sphere_radius': float(params['rx_sphere_radius_m']),
        'diffuse_type': str(params['diffuse_type']),
        'enable_wideband': bool(params['enable_wideband_cfrs']),
        'duration': float(params['duration']),
        'interval': float(params['interval']),
        'seed': int(params['seed']) if params['is_seeded'] else None
    }

    return rt_params 
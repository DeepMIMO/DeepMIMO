"""
AODT Ray Tracing Parameters Module.

This module handles reading and processing:
1. Ray tracing parameters from scenario.parquet
2. RAN configuration from ran_config.parquet
"""

import os
import pandas as pd
from typing import Dict, Any

def read_ran_config(rt_folder: str) -> Dict[str, Any]:
    """Read RAN configuration parameters.

    Args:
        rt_folder (str): Path to folder containing ran_config.parquet.

    Returns:
        Dict[str, Any]: Dictionary containing RAN configuration parameters.
    """
    ran_file = os.path.join(rt_folder, 'ran_config.parquet')
    if not os.path.exists(ran_file):
        return {}

    df = pd.read_parquet(ran_file)
    if len(df) == 0:
        return {}

    # Get first row since parameters are the same for all rows
    params = df.iloc[0]

    ran_params = {
        'tdd_pattern': params['tdd_pattern'],
        'srs_slots': params['srs_slots'],
        'pusch_slots': params['pusch_slots'],
        'harq': {
            'dl_enabled': bool(params['dl_harq_enabled']),
            'ul_enabled': bool(params['ul_harq_enabled'])
        },
        'csi': {
            'beamforming': bool(params['beamforming_csi']),
            'mac': bool(params['mac_csi'])
        },
        'pusch_channel_estimation': bool(params['pusch_channel_estimation']),
        'scheduler_mode': params['scheduler_mode'],
        'mu_mimo_enabled': bool(params['mu_mimo_enabled']),
        'snr_thresholds': {
            'dl_srs': float(params['dl_srs_snr_thr']),
            'ul_srs': float(params['ul_srs_snr_thr'])
        },
        'chan_corr_thresholds': {
            'dl': float(params['dl_chan_corr_thr']),
            'ul': float(params['ul_chan_corr_thr'])
        },
        'beamforming': {
            'enabled': bool(params['beamforming_enabled']),
            'scheme': params['beamforming_scheme']
        }
    }

    return ran_params

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
            - ue_params: UE-related parameters
                - num_ues: Number of UEs
                - height: UE height
                - speed: Min/max speed
                - indoor_percentage: Percentage of indoor UEs
            - simulation: Simulation parameters
                - num_batches: Number of batches
                - slots_per_batch: Slots per batch
                - symbols_per_slot: Symbols per slot
            - ran_config: RAN configuration parameters

    Raises:
        FileNotFoundError: If scenario.parquet is not found.
        ValueError: If required parameters are missing.
    """
    scen_file_name = 'scenario.parquet'
    scenario_file = os.path.join(rt_folder, scen_file_name)
    if not os.path.exists(scenario_file):
        raise FileNotFoundError(f"{scen_file_name} not found in {rt_folder}")

    # Read scenario parameters
    df = pd.read_parquet(scenario_file)
    if len(df) == 0:
        raise ValueError(f"{scen_file_name} is empty")

    # Get first row since parameters are the same for all rows
    params = df.iloc[0]

    # Read RAN configuration
    ran_params = read_ran_config(rt_folder)

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
        'seed': int(params['seed']) if params['is_seeded'] else None,
        'ue_params': {
            'num_ues': int(params['num_ues']),
            'height': float(params['ue_height']),
            'speed': {
                'min': float(params['ue_min_speed']),
                'max': float(params['ue_max_speed'])
            },
            'indoor_percentage': float(params['percentage_indoor_ues'])
        },
        'simulation': {
            'num_batches': int(params['num_batches']),
            'slots_per_batch': int(params['slots_per_batch']),
            'symbols_per_slot': int(params['symbols_per_slot'])
        },
        'ran_config': ran_params
    }

    return rt_params 
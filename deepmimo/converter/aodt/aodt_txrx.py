"""
AODT Transmitter/Receiver Configuration Module.

This module handles reading and processing:
1. Distributed Unit (DU) configurations from dus.parquet
2. Radio Unit (RU) configurations from rus.parquet
3. User Equipment (UE) configurations from ues.parquet
4. Antenna panel configurations from panels.parquet
5. Antenna patterns from patterns.parquet
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def read_panels(rt_folder: str) -> Dict[str, Any]:
    """Read antenna panel configurations.

    Args:
        rt_folder (str): Path to folder containing panels.parquet.

    Returns:
        Dict[str, Any]: Dictionary mapping panel IDs to configurations.
    """
    panels_file = os.path.join(rt_folder, 'panels.parquet')
    if not os.path.exists(panels_file):
        return {}

    df = pd.read_parquet(panels_file)
    if len(df) == 0:
        return {}

    panels = {}
    for _, panel in df.iterrows():
        panel_dict = {
            'name': panel['panel_name'],
            'antenna_names': panel['antenna_names'],
            'pattern_indices': panel['antenna_pattern_indices'],
            'frequencies': np.array(panel['frequencies']),
            'thetas': np.array(panel['thetas']),
            'phis': np.array(panel['phis']),
            'reference_freq': float(panel['reference_freq']),
            'dual_polarized': bool(panel['dual_polarized']),
            'array_config': {
                'num_horz': int(panel['num_loc_antenna_horz']),
                'num_vert': int(panel['num_loc_antenna_vert']),
                'spacing_horz': float(panel['antenna_spacing_horz']),
                'spacing_vert': float(panel['antenna_spacing_vert']),
                'roll_angle_first': float(panel['antenna_roll_angle_first_polz']),
                'roll_angle_second': float(panel['antenna_roll_angle_second_polz'])
            }
        }
        panels[panel['panel_id']] = panel_dict
    return panels

def read_patterns(rt_folder: str) -> Dict[str, Any]:
    """Read antenna patterns.

    Args:
        rt_folder (str): Path to folder containing patterns.parquet.

    Returns:
        Dict[str, Any]: Dictionary mapping pattern IDs to configurations.
    """
    patterns_file = os.path.join(rt_folder, 'patterns.parquet')
    if not os.path.exists(patterns_file):
        return {}

    df = pd.read_parquet(patterns_file)
    if len(df) == 0:
        return {}

    patterns = {}
    for _, pattern in df.iterrows():
        pattern_dict = {
            'type': pattern['pattern_type'],
            'e_theta': np.array(pattern['e_theta_re']) + 1j * np.array(pattern['e_theta_im']),
            'e_phi': np.array(pattern['e_phi_re']) + 1j * np.array(pattern['e_phi_im'])
        }
        patterns[pattern['pattern_id']] = pattern_dict
    return patterns

def read_dus(rt_folder: str) -> Dict[str, Any]:
    """Read DU configurations.

    Args:
        rt_folder (str): Path to folder containing dus.parquet.

    Returns:
        Dict[str, Any]: Dictionary mapping DU IDs to configurations.
    """
    dus_file = os.path.join(rt_folder, 'dus.parquet')
    if not os.path.exists(dus_file):
        return {}

    df = pd.read_parquet(dus_file)
    if len(df) == 0:
        return {}

    dus = {}
    for _, du in df.iterrows():
        du_dict = {
            'scs': int(du['subcarrier_spacing']),
            'fft_size': int(du['fft_size']),
            'num_antennas': int(du['num_antennas']),
            'max_bw': float(du['max_channel_bandwidth']),
            'position': np.array(du['position'])
        }
        dus[du['ID']] = du_dict
    return dus

def read_txrx(rt_folder: str, rt_params: Dict[str, Any]) -> Dict[str, Any]:
    """Read transmitter and receiver configurations.

    Args:
        rt_folder (str): Path to folder containing configuration files.

    Returns:
        Dict[str, Any]: Dictionary containing TX/RX configurations including:
            panels: Dictionary of antenna panel configurations
            patterns: Dictionary of antenna patterns
            dus: Dictionary of DU configurations
            transmitters: List of transmitter (RU) dictionaries
            receivers: List of receiver (UE) dictionaries

    Raises:
        FileNotFoundError: If required files are not found.
        ValueError: If required parameters are missing.
    """
    # Read antenna configurations
    panels = read_panels(rt_folder)
    patterns = read_patterns(rt_folder)
    dus = read_dus(rt_folder)

    # Read frequency from rt_params
    rt_params['frequency'] = panels[0]['reference_freq']

    # Read RUs file
    rus_file = os.path.join(rt_folder, 'rus.parquet')
    if not os.path.exists(rus_file):
        raise FileNotFoundError(f"rus.parquet not found in {rt_folder}")
    
    rus_df = pd.read_parquet(rus_file)
    if len(rus_df) == 0:
        raise ValueError("rus.parquet is empty")

    # Read UEs file
    ues_file = os.path.join(rt_folder, 'ues.parquet')
    if not os.path.exists(ues_file):
        raise FileNotFoundError(f"ues.parquet not found in {rt_folder}")
    
    ues_df = pd.read_parquet(ues_file)
    if len(ues_df) == 0:
        raise ValueError("ues.parquet is empty")

    # Process RUs
    transmitters = []
    for _, ru in rus_df.iterrows():
        tx = {
            'id': int(ru['ID']),
            'position': np.array(ru['position']),
            'power': float(ru['radiated_power']),
            'mech_tilt': float(ru['mech_tilt']),
            'mech_azimuth': float(ru['mech_azimuth']),
            'scs': int(ru['subcarrier_spacing']),
            'fft_size': int(ru['fft_size']),
            'panel': [panels[i] for i in ru['panel']][0], # use only first panel
            'du_id': int(ru['du_id']) if not pd.isna(ru['du_id']) else None,
            'du_manual_assign': bool(ru['du_manual_assign'])
        }
        transmitters.append(tx)

    # Process UEs
    receivers = []
    for _, ue in ues_df.iterrows():
        rx = {
            'id': int(ue['ID']),
            'is_manual': bool(ue['is_manual']),
            'is_manual_mobility': bool(ue['is_manual_mobility']),
            'power': float(ue['radiated_power']),
            'height': float(ue['height']),
            'mech_tilt': float(ue['mech_tilt']),
            'panel': [panels[i] for i in ue['panel']][0],
            'indoor': bool(ue['is_indoor_mobility']),
            'bler_target': float(ue['bler_target']),
            'mobility': {
                'batch_indices': np.array(ue['batch_indices']),
                'waypoints': {
                    'ids': np.array(ue['waypoint_ids']),
                    'points': np.array(ue['waypoint_points']),
                    'stops': np.array(ue['waypoint_stops']),
                    'speeds': np.array(ue['waypoint_speeds'])
                },
                'trajectory': {
                    'ids': np.array(ue['trajectory_ids']),
                    'points': np.array(ue['trajectory_points']),
                    'stops': np.array(ue['trajectory_stops']),
                    'speeds': np.array(ue['trajectory_speeds'])
                },
                'route': {
                    'positions': np.array(ue['route_positions']),
                    'orientations': np.array(ue['route_orientations']),
                    'speeds': np.array(ue['route_speeds']),
                    'times': np.array(ue['route_times'])
                }
            }
        }
        receivers.append(rx)

    return {
        'panels': panels,
        'patterns': patterns,
        'dus': dus,
        'transmitters': transmitters,
        'receivers': receivers
    } 
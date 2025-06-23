"""
AODT Transmitter/Receiver Configuration Module.

This module handles reading and processing transmitter (RU) and receiver (UE)
configurations from rus.parquet and ues.parquet files.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any

def read_txrx(rt_folder: str) -> Dict[str, Any]:
    """Read transmitter and receiver configurations.

    Args:
        rt_folder (str): Path to folder containing rus.parquet and ues.parquet.

    Returns:
        Dict[str, Any]: Dictionary containing TX/RX configurations including:
            transmitters: List of transmitter (RU) dictionaries with:
                - id: RU ID
                - position: [x, y, z] coordinates
                - power: Radiated power in dBm
                - height: Height in meters
                - panel: Antenna panel configuration
                - mech_tilt: Mechanical tilt angle
                - mech_azimuth: Mechanical azimuth angle
                - scs: Subcarrier spacing
                - fft_size: FFT size
            receivers: List of receiver (UE) dictionaries with:
                - id: UE ID
                - position: [x, y, z] coordinates from trajectory
                - power: Radiated power in dBm
                - height: Height in meters
                - panel: Antenna panel configuration
                - mech_tilt: Mechanical tilt angle
                - indoor: Whether UE is indoor
                - trajectory: List of trajectory points

    Raises:
        FileNotFoundError: If required files are not found.
        ValueError: If required parameters are missing.
    """
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
            'height': float(ru['height']),
            'panel': ru['panel'],
            'mech_tilt': float(ru['mech_tilt']),
            'mech_azimuth': float(ru['mech_azimuth']),
            'scs': int(ru['subcarrier_spacing']),
            'fft_size': int(ru['fft_size'])
        }
        transmitters.append(tx)

    # Process UEs
    receivers = []
    for _, ue in ues_df.iterrows():
        rx = {
            'id': int(ue['ID']),
            'position': np.array(ue['route_positions'][0]) if len(ue['route_positions']) > 0 else None,
            'power': float(ue['radiated_power']),
            'height': float(ue['height']),
            'panel': ue['panel'],
            'mech_tilt': float(ue['mech_tilt']),
            'indoor': bool(ue['is_indoor_mobility']),
            'trajectory': {
                'positions': np.array(ue['route_positions']),
                'orientations': np.array(ue['route_orientations']),
                'speeds': np.array(ue['route_speeds']),
                'times': np.array(ue['route_times'])
            }
        }
        receivers.append(rx)

    return {
        'transmitters': transmitters,
        'receivers': receivers
    } 
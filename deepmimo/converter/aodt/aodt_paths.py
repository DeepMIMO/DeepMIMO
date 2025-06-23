"""
AODT Ray Paths Module.

This module handles reading and processing ray path data from raypaths.parquet,
including interaction points, types, and power information.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any

def read_paths(rt_folder: str, output_folder: str, txrx_dict: Dict[str, Any]) -> None:
    """Read and process ray paths from raypaths.parquet.

    Args:
        rt_folder (str): Path to folder containing raypaths.parquet.
        output_folder (str): Path to folder where processed paths will be saved.
        txrx_dict (Dict[str, Any]): Dictionary containing TX/RX configurations.

    Raises:
        FileNotFoundError: If raypaths.parquet is not found.
        ValueError: If required parameters are missing.
    """
    paths_file = os.path.join(rt_folder, 'raypaths.parquet')
    if not os.path.exists(paths_file):
        raise FileNotFoundError(f"raypaths.parquet not found in {rt_folder}")

    # Read paths data
    df = pd.read_parquet(paths_file)
    if len(df) == 0:
        raise ValueError("raypaths.parquet is empty")

    # Create output paths folder
    paths_folder = os.path.join(output_folder, 'paths')
    os.makedirs(paths_folder, exist_ok=True)

    # Group paths by time index
    for time_idx in df['time_idx'].unique():
        time_df = df[df['time_idx'] == time_idx]
        
        # Group by RU-UE pair
        for ru_id in time_df['ru_id'].unique():
            ru_df = time_df[time_df['ru_id'] == ru_id]
            
            for ue_id in ru_df['ue_id'].unique():
                paths = ru_df[ru_df['ue_id'] == ue_id].iloc[0]
                
                # Process path data
                path_data = {
                    'interaction_points': np.array(paths['points']),
                    'interaction_types': np.array(paths['interaction_types']),
                    'interaction_normals': np.array(paths['normals']),
                    'path_powers': np.array(paths['tap_power'])
                }

                # Save path data
                filename = f'paths_t{time_idx}_ru{ru_id}_ue{ue_id}.npz'
                np.savez(os.path.join(paths_folder, filename), **path_data)

    # Also read channel data if available
    read_channel_data(rt_folder, output_folder, txrx_dict)

def read_channel_data(rt_folder: str, output_folder: str, txrx_dict: Dict[str, Any]) -> None:
    """Read and process channel data from cfrs.parquet and cirs.parquet.

    Args:
        rt_folder (str): Path to folder containing channel data files.
        output_folder (str): Path to folder where processed data will be saved.
        txrx_dict (Dict[str, Any]): Dictionary containing TX/RX configurations.
    """
    # Create output channels folder
    channels_folder = os.path.join(output_folder, 'channels')
    os.makedirs(channels_folder, exist_ok=True)

    # Read CFR data if available
    cfrs_file = os.path.join(rt_folder, 'cfrs.parquet')
    if os.path.exists(cfrs_file):
        cfrs_df = pd.read_parquet(cfrs_file)
        if len(cfrs_df) > 0:
            for time_idx in cfrs_df['time_idx'].unique():
                time_df = cfrs_df[cfrs_df['time_idx'] == time_idx]
                
                for ru_id in time_df['ru_id'].unique():
                    ru_df = time_df[time_df['ru_id'] == ru_id]
                    
                    for ue_id in ru_df['ue_id'].unique():
                        cfrs = ru_df[ru_df['ue_id'] == ue_id]
                        
                        # Combine real and imaginary parts
                        cfr_data = np.array(cfrs['cfr_re']) + 1j * np.array(cfrs['cfr_im'])
                        
                        # Save CFR data
                        filename = f'cfr_t{time_idx}_ru{ru_id}_ue{ue_id}.npz'
                        np.savez(os.path.join(channels_folder, filename), cfr=cfr_data)

    # Read CIR data if available
    cirs_file = os.path.join(rt_folder, 'cirs.parquet')
    if os.path.exists(cirs_file):
        cirs_df = pd.read_parquet(cirs_file)
        if len(cirs_df) > 0:
            for time_idx in cirs_df['time_idx'].unique():
                time_df = cirs_df[cirs_df['time_idx'] == time_idx]
                
                for ru_id in time_df['ru_id'].unique():
                    ru_df = time_df[time_df['ru_id'] == ru_id]
                    
                    for ue_id in ru_df['ue_id'].unique():
                        cirs = ru_df[ru_df['ue_id'] == ue_id]
                        
                        # Combine real and imaginary parts
                        cir_data = np.array(cirs['cir_re']) + 1j * np.array(cirs['cir_im'])
                        cir_delays = np.array(cirs['cir_delay'])
                        
                        # Save CIR data
                        filename = f'cir_t{time_idx}_ru{ru_id}_ue{ue_id}.npz'
                        np.savez(os.path.join(channels_folder, filename), 
                                cir=cir_data, delays=cir_delays) 
"""
AODT Ray Paths Module.

This module handles reading and processing:
1. Ray path data from raypaths.parquet
2. Channel Impulse Response (CIR) from cirs.parquet
3. Channel Frequency Response (CFR) from cfrs.parquet
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any

def read_paths(rt_folder: str, output_folder: str, txrx_dict: Dict[str, Any]) -> None:
    """Read and process ray paths and channel responses.

    Args:
        rt_folder (str): Path to folder containing parquet files.
        output_folder (str): Path to folder where processed paths will be saved.
        txrx_dict (Dict[str, Any]): Dictionary containing TX/RX configurations.

    Raises:
        FileNotFoundError: If required files are not found.
        ValueError: If required parameters are missing.
    """
    # Create output folders
    paths_folder = os.path.join(output_folder, 'paths')
    channels_folder = os.path.join(output_folder, 'channels')
    os.makedirs(paths_folder, exist_ok=True)
    os.makedirs(channels_folder, exist_ok=True)

    # Read paths data
    paths_file = os.path.join(rt_folder, 'raypaths.parquet')
    if os.path.exists(paths_file):
        df = pd.read_parquet(paths_file)
        if len(df) > 0:
            process_raypaths(df, paths_folder)

    # Read CIR data
    cirs_file = os.path.join(rt_folder, 'cirs.parquet')
    if os.path.exists(cirs_file):
        df = pd.read_parquet(cirs_file)
        if len(df) > 0:
            process_cirs(df, channels_folder)

    # Read CFR data
    cfrs_file = os.path.join(rt_folder, 'cfrs.parquet')
    if os.path.exists(cfrs_file):
        df = pd.read_parquet(cfrs_file)
        if len(df) > 0:
            process_cfrs(df, channels_folder)

def process_raypaths(df: pd.DataFrame, output_folder: str) -> None:
    """Process ray paths data.
    
    Args:
        df (pd.DataFrame): DataFrame containing ray paths.
        output_folder (str): Output folder path.
    """
    for time_idx in df['time_idx'].unique():
        time_df = df[df['time_idx'] == time_idx]
        
        for ru_id in time_df['ru_id'].unique():
            ru_df = time_df[time_df['ru_id'] == ru_id]
            
            for ue_id in ru_df['ue_id'].unique():
                paths = ru_df[ru_df['ue_id'] == ue_id].iloc[0]
                
                path_data = {
                    'interaction_points': np.array(paths['points']),
                    'interaction_types': np.array(paths['interaction_types']),
                    'interaction_normals': np.array(paths['normals']),
                    'path_powers': np.array(paths['tap_power'])
                }

                filename = f'paths_t{time_idx}_ru{ru_id}_ue{ue_id}.npz'
                np.savez(os.path.join(output_folder, filename), **path_data)

def process_cirs(df: pd.DataFrame, output_folder: str) -> None:
    """Process Channel Impulse Response data.
    
    Args:
        df (pd.DataFrame): DataFrame containing CIRs.
        output_folder (str): Output folder path.
    """
    for time_idx in df['time_idx'].unique():
        time_df = df[df['time_idx'] == time_idx]
        
        for ru_id in time_df['ru_id'].unique():
            ru_df = time_df[time_df['ru_id'] == ru_id]
            
            for ue_id in ru_df['ue_id'].unique():
                cirs = ru_df[ru_df['ue_id'] == ue_id]
                
                # Group by antenna elements
                for ru_ant_el in cirs['ru_ant_el'].unique():
                    for ue_ant_el in cirs['ue_ant_el'].unique():
                        ant_cirs = cirs[
                            (cirs['ru_ant_el'] == ru_ant_el) & 
                            (cirs['ue_ant_el'] == ue_ant_el)
                        ]
                        
                        # Combine real and imaginary parts
                        cir_data = np.array(ant_cirs['cir_re']) + 1j * np.array(ant_cirs['cir_im'])
                        cir_delays = np.array(ant_cirs['cir_delay'])
                        
                        filename = f'cir_t{time_idx}_ru{ru_id}_ue{ue_id}_ruant{ru_ant_el}_ueant{ue_ant_el}.npz'
                        np.savez(os.path.join(output_folder, filename), 
                                cir=cir_data, delays=cir_delays)

def process_cfrs(df: pd.DataFrame, output_folder: str) -> None:
    """Process Channel Frequency Response data.
    
    Args:
        df (pd.DataFrame): DataFrame containing CFRs.
        output_folder (str): Output folder path.
    """
    for time_idx in df['time_idx'].unique():
        time_df = df[df['time_idx'] == time_idx]
        
        for ru_id in time_df['ru_id'].unique():
            ru_df = time_df[time_df['ru_id'] == ru_id]
            
            for ue_id in ru_df['ue_id'].unique():
                cfrs = ru_df[ru_df['ue_id'] == ue_id]
                
                # Group by antenna elements
                for ru_ant_el in cfrs['ru_ant_el'].unique():
                    for ue_ant_el in cfrs['ue_ant_el'].unique():
                        ant_cfrs = cfrs[
                            (cfrs['ru_ant_el'] == ru_ant_el) & 
                            (cfrs['ue_ant_el'] == ue_ant_el)
                        ]
                        
                        # Combine real and imaginary parts
                        cfr_data = np.array(ant_cfrs['cfr_re']) + 1j * np.array(ant_cfrs['cfr_im'])
                        
                        filename = f'cfr_t{time_idx}_ru{ru_id}_ue{ue_id}_ruant{ru_ant_el}_ueant{ue_ant_el}.npz'
                        np.savez(os.path.join(output_folder, filename), cfr=cfr_data) 
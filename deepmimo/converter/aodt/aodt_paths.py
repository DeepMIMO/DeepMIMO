"""
AODT Ray Paths Module.

This module handles reading and processing:
1. Ray path data from raypaths.parquet
2. Channel Impulse Response (CIR) from cirs.parquet
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from ... import consts as c
from ... import general_utils as gu
from .. import converter_utils as cu
from . import aodt_utils as au

# AODT interaction type mapping
AODT_TYPE_TO_NUM = {
    'emission': 0,
    'reflection': 1,
    'diffraction': 2,
    'scattering': 3,
    'diffuse': 3,  # alias for scattering
    'reception': 4,
    'transmission': 5
}

AODT_INTERACTIONS_MAP = {
    0: None,  # emission - not counted as interaction
    1: c.INTERACTION_REFLECTION,  # reflection
    2: c.INTERACTION_DIFFRACTION,  # diffraction
    3: c.INTERACTION_SCATTERING,  # diffuse scattering
    4: None,  # reception - not counted as interaction
    5: c.INTERACTION_TRANSMISSION  # transmission
}

def _transform_interaction_types(types: np.ndarray) -> float:
    """Transform AODT interaction types array into a single DeepMIMO interaction code.
    
    Args:
        types: Array of AODT interaction types where:
              - First element is always 'emission'
              - Last element is always 'reception'
              - Middle elements can be:
                'reflection', 'diffraction', 'scattering'/'diffuse', 'transmission'
              
    Returns:
        float: Single number representing concatenated interaction types.
               For example: [0, 1, 2, 1, 4] -> 121 (reflection-diffraction-reflection)
               LoS paths return c.INTERACTION_LOS
               
    Example:
        ['emission', 'reflection', 'reflection', 'reception'] -> 11 (two reflections)
        ['emission', 'diffraction', 'reception'] -> 2 (single diffraction)
        ['emission', 'reception'] -> 0 (LoS)
    """
    # If only emission and reception, it's LoS
    if len(types) <= 2:
        return c.INTERACTION_LOS
        
    # Take only middle interactions (exclude first and last)
    interactions = types[1:-1]
    
    # Convert string types to numeric codes
    if interactions.dtype == 'O': # str
        numeric_types = [AODT_TYPE_TO_NUM[t.lower()] for t in interactions]
    else:
        numeric_types = interactions
        
    # Map AODT types to DeepMIMO types and concatenate
    mapped = [str(AODT_INTERACTIONS_MAP[t]) for t in numeric_types if AODT_INTERACTIONS_MAP[t] is not None]
    if not mapped:  # If all interactions were mapped to None
        return c.INTERACTION_LOS
        
    return float(''.join(mapped))

def _preallocate_data(n_rx: int, n_paths: int = c.MAX_PATHS) -> Dict:
    """Pre-allocate data for path conversion.
    
    Args:
        n_rx: Number of RXs
        n_paths: Number of paths to allocate. Defaults to c.MAX_PATHS.

    Returns:
        data: Dictionary containing pre-allocated data
    """
    data = {
        c.RX_POS_PARAM_NAME: np.zeros((n_rx, 3), dtype=c.FP_TYPE),
        c.TX_POS_PARAM_NAME: np.zeros((1, 3), dtype=c.FP_TYPE),
        c.AOA_AZ_PARAM_NAME: np.zeros((n_rx, n_paths), dtype=c.FP_TYPE) * np.nan,
        c.AOA_EL_PARAM_NAME: np.zeros((n_rx, n_paths), dtype=c.FP_TYPE) * np.nan,
        c.AOD_AZ_PARAM_NAME: np.zeros((n_rx, n_paths), dtype=c.FP_TYPE) * np.nan,
        c.AOD_EL_PARAM_NAME: np.zeros((n_rx, n_paths), dtype=c.FP_TYPE) * np.nan,
        c.DELAY_PARAM_NAME:  np.zeros((n_rx, n_paths), dtype=c.FP_TYPE) * np.nan,
        c.POWER_PARAM_NAME:  np.zeros((n_rx, n_paths), dtype=c.FP_TYPE) * np.nan,
        c.PHASE_PARAM_NAME:  np.zeros((n_rx, n_paths), dtype=c.FP_TYPE) * np.nan,
        c.INTERACTIONS_PARAM_NAME:  np.zeros((n_rx, n_paths), dtype=c.FP_TYPE) * np.nan,
        c.INTERACTIONS_POS_PARAM_NAME: np.zeros((n_rx, n_paths, c.MAX_INTER_PER_PATH, 3), dtype=c.FP_TYPE) * np.nan,
    }
    
    return data

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
    # Read both parquet files
    paths_file = os.path.join(rt_folder, 'raypaths.parquet')
    cirs_file = os.path.join(rt_folder, 'cirs.parquet')
    
    if not os.path.exists(paths_file) or not os.path.exists(cirs_file):
        raise FileNotFoundError("Both raypaths.parquet and cirs.parquet are required")
        
    paths_df = pd.read_parquet(paths_file)
    cirs_df = pd.read_parquet(cirs_file)
    
    if len(paths_df) == 0 or len(cirs_df) == 0:
        raise ValueError("Empty parquet files")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Check if we have single UE with multiple TX/RX pairs
    unique_ues = paths_df['ue_id'].unique()
    is_single_ue_multi_pair = len(unique_ues) == 1 and len(cirs_df['ue_ant_el']) > 1
    
    if is_single_ue_multi_pair:
        print(f"\nDetected single UE ({unique_ues[0]}) with multiple TX/RX pairs.")
        print("Currently using only first TX/RX pair. Multi-pair support coming soon.")
        # TODO: In the future, we'll process all pairs here
        # For now, just filter to first RU
    
    # Build mapping from RU/UE IDs to txrx_set IDs
    tx_id_map = {}  # ru_id -> (tx_set_id, tx_idx)
    rx_id_map = {}  # ue_id -> rx_idx
    
    # Map transmitters
    for key, txrx_set in txrx_dict.items():
        if txrx_set['is_tx']:
            tx_id_map[txrx_set['id_orig']] = (txrx_set['id'], 0)  # tx_idx is always 0 since each TX is its own set
    
    # Get the single receiver set
    rx_set = next(v for v in txrx_dict.values() if v['is_rx'])
    rx_set_id = rx_set['id']
    
    # Map receivers to their indices in the order they appear in paths_df
    for idx, ue_id in enumerate(paths_df['ue_id'].unique()):
        rx_id_map[ue_id] = idx
    
    time_idx = paths_df['time_idx'].unique()[0] # take first time index
    paths_time_df = paths_df[paths_df['time_idx'] == time_idx]
    cirs_time_df = cirs_df[cirs_df['time_idx'] == time_idx]
    
    for ru_id in paths_time_df['ru_id'].unique():
        paths_ru_df = paths_time_df[paths_time_df['ru_id'] == ru_id]
        cirs_ru_df = cirs_time_df[cirs_time_df['ru_id'] == ru_id]
        
        # Get number of UEs (receivers) for this RU
        ue_ids = paths_ru_df['ue_id'].unique()
        n_rx = len(ue_ids)
        n_paths = c.MAX_PATHS if not is_single_ue_multi_pair else len(paths_ru_df)
        data = _preallocate_data(n_rx, n_paths)
        
        # Process each UE (receiver)
        for rx_idx, ue_id in enumerate(ue_ids):
            # Get all paths and CIRs for this RU-UE pair
            paths = paths_ru_df[paths_ru_df['ue_id'] == ue_id]
            cirs = cirs_ru_df[cirs_ru_df['ue_id'] == ue_id]
            
            if len(cirs) == 0:
                print(f"Warning: No CIR data for RU {ru_id} UE {ue_id}")
                continue
            
            # Process paths first to get positions and angles
            for path_idx, path in enumerate(paths.itertuples()):
                # Process interaction points
                interaction_points = au.process_points(path.points)
                
                # Convert from cm to m
                interaction_points = interaction_points / 100.0
                
                # First point is TX, last point is RX
                if path_idx == 0:  # Only need to set positions once per UE
                    tx_pos = interaction_points[0]
                    rx_pos = interaction_points[-1]
                    if rx_idx == 0:  # Only set TX position once for first UE
                        data[c.TX_POS_PARAM_NAME][0] = tx_pos
                    data[c.RX_POS_PARAM_NAME][rx_idx] = rx_pos
                
                # Calculate angles
                # Departure angles - vector from TX to first interaction point
                departure_vector = interaction_points[1] - tx_pos
                departure_angles = gu.cartesian_to_spherical(departure_vector.reshape(1, -1))[0]
                data[c.AOD_AZ_PARAM_NAME][rx_idx, path_idx] = np.rad2deg(departure_angles[1])
                data[c.AOD_EL_PARAM_NAME][rx_idx, path_idx] = np.rad2deg(departure_angles[2])
                
                # Arrival angles - vector from last interaction point to RX
                arrival_vector = rx_pos - interaction_points[-2]
                arrival_angles = gu.cartesian_to_spherical(arrival_vector.reshape(1, -1))[0]
                data[c.AOA_AZ_PARAM_NAME][rx_idx, path_idx] = np.rad2deg(arrival_angles[1])
                data[c.AOA_EL_PARAM_NAME][rx_idx, path_idx] = np.rad2deg(arrival_angles[2])
                
                # Store interaction data - skip first (TX) and last (RX) points
                actual_interactions = interaction_points[1:-1]
                if len(actual_interactions) > 0:
                    data[c.INTERACTIONS_POS_PARAM_NAME][rx_idx, path_idx, :len(actual_interactions), :] = actual_interactions
                
                # Transform interaction types to DeepMIMO format
                data[c.INTERACTIONS_PARAM_NAME][rx_idx, path_idx] = _transform_interaction_types(path.interaction_types)
            
            # Now process CIRs for power, phase and delay
            # For now just take first antenna pair (first row)
            ant_cirs = cirs.iloc[[0]]
            
            # Combine real and imaginary parts
            cir_data = ant_cirs['cir_re'].to_numpy()[0] + 1j * ant_cirs['cir_im'].to_numpy()[0]
            
            # Calculate power and phase from complex CIR
            cir_power = 20 * np.log10(np.abs(cir_data))
            data[c.POWER_PARAM_NAME][rx_idx, :len(cir_data)] = cir_power
            data[c.PHASE_PARAM_NAME][rx_idx, :len(cir_data)] = np.angle(cir_data, deg=True)
            data[c.DELAY_PARAM_NAME][rx_idx, :len(cir_data)] = ant_cirs['cir_delay'].to_numpy()[0]
        
        # Get TX/RX set IDs for saving
        tx_set_id, tx_idx = tx_id_map[ru_id]
        
        # Compress data before saving
        data = cu.compress_path_data(data)
        
        # Save data for all UEs of this RU
        for key in data.keys():
            cu.save_mat(data[key], key, output_folder, tx_set_id, tx_idx, rx_set_id)

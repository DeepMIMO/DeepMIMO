"""
Sionna Ray Tracing Paths Module.

This module handles loading and converting path data from Sionna's format to DeepMIMO's format.
"""

import os
import numpy as np
from tqdm import tqdm
from typing import Dict
from ... import consts as c
from ..converter_utils import save_mat, compress_path_data
from ...general_utils import load_pickle

# Interaction Type Map for Sionna
INTERACTIONS_MAP = {
    0:  c.INTERACTION_LOS,           # LoS
    1:  c.INTERACTION_REFLECTION,    # Reflection
    2:  c.INTERACTION_DIFFRACTION,   # Diffraction
    3:  c.INTERACTION_SCATTERING,    # Diffuse Scattering
    4:  None,  # Sionna RIS is not supported yet
}

def _is_sionna_v1(sionna_version: str):
    """Determine if Sionna version is 1.x or higher."""
    return sionna_version.startswith('1.')

def _preallocate_data(n_rx: int) -> Dict:
    """Pre-allocate data for path conversion.
    
    Args:
        n_rx: Number of RXs

    Returns:
        data: Dictionary containing pre-allocated data
    """
    data = {
        c.RX_POS_PARAM_NAME: np.zeros((n_rx, 3), dtype=c.FP_TYPE),
        c.TX_POS_PARAM_NAME: np.zeros((1, 3), dtype=c.FP_TYPE),
        c.AOA_AZ_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.AOA_EL_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.AOD_AZ_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.AOD_EL_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.DELAY_PARAM_NAME:  np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.POWER_PARAM_NAME:  np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.PHASE_PARAM_NAME:  np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.INTERACTIONS_PARAM_NAME:  np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.INTERACTIONS_POS_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS, c.MAX_INTER_PER_PATH, 3), dtype=c.FP_TYPE) * np.nan,
    }
    
    return data
    
def _process_paths_batch(paths_dict: Dict, data: Dict, b: int, t: int,
                         batch_size: int, targets: np.ndarray, rx_pos: np.ndarray,
                         sionna_version: str) -> int:
    """Process a batch of paths from Sionna format and store in DeepMIMO format.
    
    Args:
        paths_dict: Dictionary containing Sionna path data
        data: Dictionary to store processed path data
        b: Batch index
        t: Transmitter index in current paths dictionary
        batch_size: Number of receivers in current batch
        targets: Array of target positions
        rx_pos: Array of RX positions
        
    Returns:
        int: Number of inactive receivers found in this batch
    """
    inactive_count = 0
    
    a = paths_dict['a']
    tau = paths_dict['tau']
    phi_r = paths_dict['phi_r'] 
    phi_t = paths_dict['phi_t']
    theta_r = paths_dict['theta_r']
    theta_t = paths_dict['theta_t']
    vertices = paths_dict['vertices']

    # Sionna 0.x, uses 'types' & Sionna 1.x, uses 'interactions'
    types = _get_path_key(paths_dict, 'types', 'interactions')

    # Notes for single and multi antenna, in Sionna 0.x and Sionna 1.x
    # DIM_TYPE_1: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
    # DIM_TYPE_2: [batch_size, num_rx, num_tx, max_num_paths]
    
    # Sionna 0.x:
    # - a:        DIM_TYPE_1
    # - tau:      DIM_TYPE_1 or DIM_TYPE_2
    # - phi_r:    DIM_TYPE_1 or DIM_TYPE_2
    # - vertices: DIM_TYPE_1 or DIM_TYPE_2 + (,3) (but with max_depth instead of batch_size)
    # - types:    ...
    # Sionna 1.x: (the same but without batch dimension)
    # - types:    DIM_TYPE_1 or DIM_TYPE_2 (but with max_depth instead of batch_size)
    # Currently, we only support DIM_TYPE_2 (no multi antenna)
    sionna_v1 = _is_sionna_v1(sionna_version)
    if not sionna_v1:
        a = a[b, ..., 0]
        tau = tau[b, ...]
        phi_r = phi_r[b, ...]
        phi_t = phi_t[b, ...]
        theta_r = theta_r[b, ...]
        theta_t = theta_t[b, ...]
        types = types[b, ...]

    # Check if single antenna (this changes the dimensions of the arrays)
    if theta_r.ndim == 3:
        rx_ant_idx = 0
        tx_ant_idx = 0
        tx_idx = t
    else:
        rx_ant_range = range(a.shape[1])
        tx_ant_range = range(a.shape[3])
        raise NotImplementedError('Multi antenna support is not implemented yet.')
    
    n_rx = targets.shape[0]

    for rel_rx_idx in range(n_rx):

        abs_idx_arr = np.where(np.all(rx_pos == targets[rel_rx_idx], axis=1))[0]
        if len(abs_idx_arr) == 0:
            # RX position not found in global RX list, skip
            continue
        abs_idx = abs_idx_arr[0]

        amp = a[rel_rx_idx, rx_ant_idx, tx_idx, tx_ant_idx, :]
        non_zero_path_idxs = np.where(amp != 0)[0][:c.MAX_PATHS]
        n_paths = len(non_zero_path_idxs)
        if n_paths == 0:
            inactive_count += 1
            continue
        # Ensure that the paths are sorted by amplitude
        sorted_path_idxs = np.argsort(np.abs(amp))[::-1]
        path_idxs = sorted_path_idxs[:n_paths]
        
        data[c.POWER_PARAM_NAME][abs_idx, :n_paths] = 20 * np.log10(np.abs(amp[path_idxs]))
        data[c.PHASE_PARAM_NAME][abs_idx, :n_paths] = np.angle(amp[path_idxs], deg=True)
        
        data[c.AOA_AZ_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(phi_r[rel_rx_idx, tx_idx, path_idxs])
        data[c.AOD_AZ_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(phi_t[rel_rx_idx, tx_idx, path_idxs])
        data[c.AOA_EL_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(theta_r[rel_rx_idx, tx_idx, path_idxs])
        data[c.AOD_EL_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(theta_t[rel_rx_idx, tx_idx, path_idxs])
        
        data[c.DELAY_PARAM_NAME][abs_idx, :n_paths] = tau[rel_rx_idx, tx_idx, path_idxs]

        # Interaction positions and types
        inter_pos_rx = vertices[:, rel_rx_idx, tx_idx, path_idxs, :].swapaxes(0,1)
        n_interactions = inter_pos_rx.shape[1]
        inter_pos_rx[inter_pos_rx == 0] = np.nan
        # NOTE: this is a workaround to handle no interaction positions
        data[c.INTERACTIONS_POS_PARAM_NAME][abs_idx, :n_paths, :n_interactions, :] = inter_pos_rx
        if sionna_v1:
            # For Sionna v1, types is (max_depth, n_rx, n_tx, max_paths)
            # We need to get (n_paths, max_depth) for the current rx/tx pair
            path_types = types[:, rel_rx_idx, tx_idx, path_idxs].swapaxes(0,1)
            inter_types = _transform_interaction_types(path_types)
        else:
            inter_types = _get_sionna_interaction_types(types[path_idxs], inter_pos_rx)
        
        data[c.INTERACTIONS_PARAM_NAME][abs_idx, :n_paths] = inter_types
        
    return inactive_count

def _get_path_key(paths_dict, key, fallback_key=None, default=None):
    if key in paths_dict:
        return paths_dict[key]
    elif fallback_key and fallback_key in paths_dict:
        return paths_dict[fallback_key]
    elif default is not None:
        return default
    else:
        raise KeyError(f"Neither '{key}' nor '{fallback_key}' found in paths_dict.")

def _transform_interaction_types(types: np.ndarray) -> np.ndarray:
    """Transform a (n_paths, max_depth) interaction types array into a (n_paths,) array
    where each element is an integer formed by concatenating the interaction type digits.
    
    Args:
        types: Array of shape (n_paths, max_depth) containing interaction types:
              0 for LoS, 1 for Reflection, 2 for Diffraction, 3 for Scattering
              
    Returns:
        np.ndarray: Array of shape (n_paths,) where each element is an integer
                   representing the concatenated interaction types.
                   
    Example:
        [[0, 0, 0],      ->  [0,      # LoS
         [1, 1, 0],           11,     # Two reflections
         [1, 3, 0],           13,     # Reflection followed by scattering
         [2, 0, 0]]           2]      # Single diffraction
    
    Note: This function is only used for Sionna 1.x.
    """
    n_paths = types.shape[0]
    result = np.zeros(n_paths, dtype=np.float32)
    
    for i in range(n_paths):
        # Get non-zero interactions (ignoring trailing zeros)
        path = types[i]
        if np.all(path == 0):
            # All zeros means LoS
            result[i] = c.INTERACTION_LOS
            continue
            
        # Find first zero after a non-zero (if any)
        non_zero_mask = path != 0
        if np.any(non_zero_mask):
            # Get indices where we have non-zero values
            non_zero_indices = np.where(non_zero_mask)[0]
            # Take all interactions up to the last non-zero
            valid_interactions = path[: non_zero_indices[-1] + 1]
            # Convert to string and remove any zeros
            interaction_str = ''.join(str(int(x)) for x in valid_interactions if x != 0)
            result[i] = float(interaction_str)
            
    return result

def _get_sionna_interaction_types(types: np.ndarray, inter_pos: np.ndarray) -> np.ndarray:
    """
    Convert Sionna interaction types to DeepMIMO interaction codes.
    
    Args:
        types: Array of interaction types from Sionna (N_PATHS,)
        inter_pos: Array of interaction positions (N_PATHS x MAX_INTERACTIONS x 3)

    Returns:
        np.ndarray: Array of DeepMIMO interaction codes (N_PATHS,)
    
    Note: This function is only used for Sionna 0.x.
    """
    # Ensure types is a numpy array
    types = np.asarray(types)
    if types.ndim == 0:
        types = np.array([types])
    
    # Get number of paths
    n_paths = len(types)
    result = np.zeros(n_paths, dtype=np.float32)
    
    # For each path
    for path_idx in range(n_paths):
        # Skip if no type (nan or 0)
        if np.isnan(types[path_idx]) or types[path_idx] == 0:
            continue
            
        sionna_type = int(types[path_idx])
        
        # Handle LoS case (type 0)
        if sionna_type == 0:
            result[path_idx] = c.INTERACTION_LOS
            continue
            
        # Count number of actual interactions by checking non-nan positions
        if inter_pos.ndim == 2:  # Single path case
            n_interactions = np.nansum(~np.isnan(inter_pos[:, 0]))
        else:  # Multiple paths case
            n_interactions = np.nansum(~np.isnan(inter_pos[path_idx, :, 0]))
            
        if n_interactions == 0:  # Skip if no interactions
            continue
            
        # Handle different Sionna interaction types
        if sionna_type == 1:  # Pure reflection path
            # Create string of '1's with length = number of reflections
            code = '1' * n_interactions
            result[path_idx] = np.float32(code)
            
        elif sionna_type == 2:  # Single diffraction path
            # Always just '2' since Sionna only allows single diffraction
            result[path_idx] = c.INTERACTION_DIFFRACTION
            
        elif sionna_type == 3:  # Scattering path with possible reflections
            # Create string of '1's for reflections + '3' at the end for scattering
            if n_interactions > 1:
                code = '1' * (n_interactions - 1) + '3'
            else:
                code = '3'
            result[path_idx] = np.float32(code)
            
        else:
            if sionna_type == 4:
                raise NotImplementedError('RIS code not supported yet')
            else:
                raise ValueError(f'Unknown Sionna interaction type: {sionna_type}')
    
    return result 

def read_paths(load_folder: str, save_folder: str, txrx_dict: Dict, sionna_version: str) -> None:
    """Read and convert path data from Sionna format.
    
    Args:
        load_folder: Path to folder containing Sionna path files
        save_folder: Path to save converted path data
        txrx_dict: Dictionary containing TX/RX set information from read_txrx
        sionna_version: Sionna version string
        
    Notes:
        - Each path dictionary can contain one or more transmitters
        - Transmitters are identified by their positions across all path dictionaries
        - RX positions maintain their relative order across path dictionaries
    
    -- Information about the Sionna paths (from https://nvlabs.github.io/sionna/api/rt.html#paths) --

    [Amplitude]
    - paths_dict['a'] is the amplitude of the path
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps]

    [Delay]
    - paths_dict['tau'] is the delay of the path
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or 
        [batch_size, num_rx, num_tx, max_num_paths], float

    [Angles]
    - paths_dict['phi_r'] is the azimuth angle of the arrival of the path
    - paths_dict['theta_r'] is the elevation angle of the arrival of the path
    - paths_dict['phi_t'] is the azimuth angle of the departure of the path
    - paths_dict['theta_t'] is the elevation angle of the departure of the path
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or 
        [batch_size, num_rx, num_tx, max_num_paths], float

    [Types]
    - paths_dict['types'] is the type of the path
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or 
        [batch_size, num_rx, num_tx, max_num_paths], float

    [Vertices]
    - paths_dict['vertices'] is the vertices of the path
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or 
        [batch_size, num_rx, num_tx, max_num_paths], float

    """
    path_dict_list = load_pickle(os.path.join(load_folder, 'sionna_paths.pkl'))

    # Collect all unique TX positions from all path dictionaries
    all_tx_pos = np.unique(np.vstack([
        _get_path_key(paths_dict, 'sources', 'src_positions') for paths_dict in path_dict_list
    ]), axis=0)
    n_tx = len(all_tx_pos)

    # Collect all RX positions while maintaining order and removing duplicates
    all_rx_pos = np.vstack([
        _get_path_key(paths_dict, 'targets', 'tgt_positions') for paths_dict in path_dict_list
    ])
    
    _, unique_indices = np.unique(all_rx_pos, axis=0, return_index=True)
    rx_pos = all_rx_pos[np.sort(unique_indices)]  # Sort indices to maintain original order
    n_rx = len(rx_pos)

    # Initialize inactive indices list
    rx_inactive_idxs_count = 0
    bs_bs_paths = False
    for tx_idx, tx_pos_target in enumerate(all_tx_pos):
        # Pre-allocate matrices
        data = _preallocate_data(n_rx)

        data[c.RX_POS_PARAM_NAME], data[c.TX_POS_PARAM_NAME] = rx_pos, tx_pos_target
        
        # Create progress bar
        pbar = tqdm(total=n_rx, desc=f"Processing receivers for TX {tx_idx}")
        
        b = 0  # batch index 
        # Process each batch of paths
        for path_dict_idx, paths_dict in enumerate(path_dict_list):
            sources = _get_path_key(paths_dict, 'sources', 'src_positions')
            tx_idx_in_dict = np.where(np.all(sources == tx_pos_target, axis=1))[0]
            if len(tx_idx_in_dict) == 0:
                continue
            if path_dict_idx == 0:
                targets = _get_path_key(paths_dict, 'targets', 'tgt_positions')
                if np.array_equal(sources, targets):
                    bs_bs_paths = True
                    continue
            t = tx_idx_in_dict[0]
            batch_size = targets.shape[0]
            targets = _get_path_key(paths_dict, 'targets', 'tgt_positions')
            inactive_count = _process_paths_batch(paths_dict, data, b, t, batch_size, 
                                                  targets, rx_pos, sionna_version)
            if tx_idx == 0:
                rx_inactive_idxs_count += inactive_count
            pbar.update(batch_size)

        pbar.close()

        # Compress data before saving
        data = compress_path_data(data)
        
        # Save each data key
        for key in data.keys():
            save_mat(data[key], key, save_folder, 0, tx_idx, 1)  # Static for Sionna
        
        if bs_bs_paths:
            print(f'BS-BS paths found for TX {tx_idx}')
            
            paths_dict = path_dict_list[0]
            all_bs_pos = _get_path_key(paths_dict, 'sources', 'src_positions')
            num_bs = len(all_bs_pos)
            data_bs_bs = _preallocate_data(num_bs)
            data_bs_bs[c.RX_POS_PARAM_NAME] = all_bs_pos
            data_bs_bs[c.TX_POS_PARAM_NAME] = tx_pos_target
            
            # Process BS-BS paths using helper function
            _process_paths_batch(paths_dict, data_bs_bs, b, t, 0, all_bs_pos, rx_pos)
            
            # Compress data before saving
            data_bs_bs = compress_path_data(data_bs_bs)
            
            # Save each data key
            for key in data_bs_bs.keys():
                save_mat(data_bs_bs[key], key, save_folder, 
                         tx_set_idx=0, # BS INDEX
                         tx_idx=tx_idx, # ANTENNA INDEX
                         rx_set_idx=0)  # Same RX & TX set
    
    if bs_bs_paths:
        txrx_dict['txrx_set_0']['is_rx'] = True  # add BS set also as RX

    # Update txrx_dict with tx and rx numbers 
    txrx_dict['txrx_set_0']['num_points'] = n_tx
    txrx_dict['txrx_set_0']['num_active_points'] = n_tx
    
    txrx_dict['txrx_set_1']['num_points'] = n_rx
    txrx_dict['txrx_set_1']['num_active_points'] = n_rx - rx_inactive_idxs_count



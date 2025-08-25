"""
DeepMIMO Converter Utilities.

This module provides common utilities used by various ray tracing converters.

This module provides:
- File I/O operations for different formats
- Data type conversion and validation
- Path manipulation and validation
- Common mathematical operations

The module serves as a shared utility library for all DeepMIMO converters.
"""

import os
from typing import List, Dict, Optional, Any
import numpy as np
import shutil

from ..general_utils import zip, save_dict_as_json
from .. import consts as c

def check_scenario_exists(scenarios_folder: str, scen_name: str, overwrite: Optional[bool] = None) -> bool:
    """Check if a scenario exists and handle overwrite prompts.
    
    Args:
        scenarios_folder (str): Path to the scenarios folder
        scen_name (str): Name of the scenario
        overwrite (Optional[bool]): Whether to overwrite if exists. If None, prompts user.
        
    Returns:
        bool: True if scenario should be overwritten, False if should be skipped
    """
    if os.path.exists(os.path.join(scenarios_folder, scen_name)):
        if overwrite is None:
            print(f'Scenario with name "{scen_name}" already exists in '
                  f'{scenarios_folder}. Delete? (Y/n)')
            ans = input()
            overwrite = False if 'n' in ans.lower() else True
        return overwrite
    return True
        
def ext_in_list(extension: str, file_list: List[str]) -> List[str]:
    """Filter files by extension.
    
    This function filters a list of filenames to only include those that end with
    the specified extension.
    
    Args:
        extension (str): File extension to filter by (e.g. '.txt')
        file_list (List[str]): List of filenames to filter
        
    Returns:
        List[str]: Filtered list containing only filenames ending with extension
    """
    return [el for el in file_list if el.endswith(extension)]

def save_rt_source_files(sim_folder: str, source_exts: List[str]) -> None:
    """Save raytracing source files to a new directory and create a zip archive.
    
    Args:
        sim_folder (str): Path to simulation folder.
        source_exts (List[str]): List of file extensions to copy.
        verbose (bool): Whether to print progress messages. Defaults to True.
    """
    rt_source_folder = os.path.basename(sim_folder) + '_raytracing_source'
    files_in_sim_folder = os.listdir(sim_folder)
    print(f'Copying raytracing source files to {rt_source_folder}')
    zip_temp_folder = os.path.join(sim_folder, rt_source_folder)
    os.makedirs(zip_temp_folder)
    
    for ext in source_exts:
        # copy all files with extensions to temp folder
        for file in ext_in_list(ext, files_in_sim_folder):
            curr_file_path = os.path.join(sim_folder, file)
            new_file_path  = os.path.join(zip_temp_folder, file)
            
            # vprint(f'Adding {file}')
            shutil.copy(curr_file_path, new_file_path)
    
    # Zip the temp folder
    zip(zip_temp_folder)
    
    # Delete the temp folder (not the zip)
    shutil.rmtree(zip_temp_folder)

    return

def save_scenario(sim_folder: str, target_folder: str = c.SCENARIOS_FOLDER) -> Optional[str]:
    """Save scenario to the DeepMIMO scenarios folder.
    
    Args:
        sim_folder (str): Path to simulation folder.
        target_folder (str): Path to target folder. Defaults to DeepMIMO scenarios folder.
        overwrite (Optional[bool]): Whether to overwrite existing scenario. Defaults to None.
        
    Returns:
        Optional[str]: Name of the exported scenario.
    """
    # Remove conversion suffix
    new_scen_folder = sim_folder.replace(c.DEEPMIMO_CONVERSION_SUFFIX, '')
    
    # Get output scenario folder
    scen_name = os.path.basename(new_scen_folder)
    scen_path = os.path.join(target_folder, scen_name)
    
    # Delete scenario if it exists
    if os.path.exists(scen_path):
        shutil.rmtree(scen_path)
    
    # Move simulation folder to scenarios folder
    shutil.move(sim_folder, scen_path)
    return scen_name

################################################################################
### Utils for compressing path data (likely to be moved outward to paths.py) ###
################################################################################

def compress_path_data(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Remove unused paths and interactions to optimize memory usage.
    
    This function compresses the path data by:
    1. Finding the maximum number of actual paths used
    2. Computing maximum number of interactions (bounces)
    3. Trimming arrays to remove unused entries
    
    Args:
        data (Dict[str, np.ndarray]): Dictionary containing path information arrays
        num_paths_key (str): Key in data dict containing number of paths. Defaults to 'n_paths'
        
    Returns:
        Dict[str, np.ndarray]: Compressed data dictionary with unused entries removed
    """
    # Compute max paths
    max_paths = get_max_paths(data)

    # Compute max bounces if interaction data exists
    max_bounces = 0
    if c.INTERACTIONS_PARAM_NAME in data:
        max_bounces = np.max(comp_next_pwr_10(data[c.INTERACTIONS_PARAM_NAME]))
    

    # Compress arrays to not take more than that space
    for key in data.keys():
        if key in [c.RX_POS_PARAM_NAME, c.TX_POS_PARAM_NAME]:
            continue
        if len(data[key].shape) >= 2:
            data[key] = data[key][:, :max_paths, ...]
        if len(data[key].shape) >= 3:
            data[key] = data[key][:, :max_paths, :max_bounces]
    
    return data

def comp_next_pwr_10(arr: np.ndarray) -> np.ndarray:
    """Calculate number of interactions from interaction codes.
    
    This function computes the number of interactions (bounces) from the
    interaction code array by calculating the number of digits.
    
    Args:
        arr (np.ndarray): Array of interaction codes
        
    Returns:
        np.ndarray: Array containing number of interactions for each path
    """
    # Handle zero separately
    result = np.zeros_like(arr, dtype=int)
    
    # For non-zero values, calculate order
    non_zero = arr > 0
    result[non_zero] = np.floor(np.log10(arr[non_zero])).astype(int) + 1
    
    return result

def get_max_paths(arr: Dict[str, np.ndarray], angle_key: str = c.AOA_AZ_PARAM_NAME) -> int:
    """Find maximum number of valid paths in the dataset.
    
    This function determines the maximum number of valid paths by finding
    the first path index where all entries (across all receivers) are NaN.
    
    Args:
        arr (Dict[str, np.ndarray]): Dictionary containing path information arrays
        angle_key (str): Key to use for checking valid paths. Defaults to AOA_AZ
        
    Returns:
        int: Maximum number of valid paths, or actual number of paths if all contain data
    """
    # The first path index with all entries at NaN
    all_nans_per_path_idx = np.all(np.isnan(arr[angle_key]), axis=0)
    n_max_paths = np.where(all_nans_per_path_idx)[0]
    
    if len(n_max_paths):
        # Found first all-NaN path index
        return n_max_paths[0]
    else:
        # All paths contain data, return actual number of paths
        return arr[angle_key].shape[1]

def save_params(params_dict: Dict[str, Any], output_folder: str) -> None:
    """Save parameters dictionary to JSON format.
    
    This function saves the parameters dictionary to a standardized location
    using the proper JSON serialization for numeric types.
    
    Args:
        params_dict: Dictionary containing all parameters
        output_folder: Output directory path
    """
    # Get standardized path for params.json
    params_path = os.path.join(output_folder, c.PARAMS_FILENAME + '.json')
    
    # Save using JSON serializer that properly handles numeric types
    save_dict_as_json(params_path, params_dict)


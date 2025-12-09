"""DeepMIMO Converter Utilities.

This module provides common utilities used by various ray tracing converters.

This module provides:
- File I/O operations for different formats
- Data type conversion and validation
- Path manipulation and validation
- Common mathematical operations

The module serves as a shared utility library for all DeepMIMO converters.
"""
# ruff: noqa: I001

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from deepmimo import consts as c
from deepmimo.general_utils import save_dict_as_json, zip as zip_folder

TWO_DIMS = 2
THREE_DIMS = 3


def check_scenario_exists(
    scenarios_folder: str,
    scen_name: str,
    *,
    overwrite: bool | None = None,
) -> bool:
    """Check if a scenario exists and handle overwrite prompts.

    Args:
        scenarios_folder (str): Path to the scenarios folder
        scen_name (str): Name of the scenario
        overwrite (Optional[bool]): Whether to overwrite if exists. If None, prompts user.

    Returns:
        bool: True if scenario should be overwritten, False if should be skipped

    """
    if (Path(scenarios_folder) / scen_name).exists():
        if overwrite is None:
            print(
                f'Scenario with name "{scen_name}" already exists in '
                f"{scenarios_folder}. Delete? (Y/n)",
            )
            ans = input()
            overwrite = "n" not in ans.lower()
        return overwrite
    return True


def ext_in_list(extension: str, file_list: list[str]) -> list[str]:
    """Filter files by extension.

    This function filters a list of filenames to only include those that end with
    the specified extension.

    Args:
        extension (str): File extension to filter by (e.g. '.txt')
        file_list (list[str]): List of filenames to filter

    Returns:
        list[str]: Filtered list containing only filenames ending with extension

    """
    return [el for el in file_list if el.endswith(extension)]


def save_rt_source_files(sim_folder: str, source_exts: list[str]) -> None:
    """Save raytracing source files to a new directory and create a zip archive.

    Args:
        sim_folder (str): Path to simulation folder.
        source_exts (list[str]): List of file extensions to copy.
        verbose (bool): Whether to print progress messages. Defaults to True.

    """
    sim_folder_path = Path(sim_folder)
    rt_source_folder = sim_folder_path.name + "_raytracing_source"
    files_in_sim_folder = [f.name for f in sim_folder_path.iterdir()]
    print(f"Copying raytracing source files to {rt_source_folder}")
    zip_temp_folder = sim_folder_path / rt_source_folder
    zip_temp_folder.mkdir(parents=True)

    for ext in source_exts:
        # copy all files with extensions to temp folder
        for file in ext_in_list(ext, files_in_sim_folder):
            curr_file_path = sim_folder_path / file
            new_file_path = zip_temp_folder / file

            # vprint(f'Adding {file}')
            shutil.copy(str(curr_file_path), str(new_file_path))

    # Zip the temp folder
    zip_folder(zip_temp_folder)

    # Delete the temp folder (not the zip)
    shutil.rmtree(zip_temp_folder)


def save_scenario(sim_folder: str, target_folder: str = c.SCENARIOS_FOLDER) -> str | None:
    """Save scenario to the DeepMIMO scenarios folder.

    Args:
        sim_folder (str): Path to simulation folder.
        target_folder (str): Path to target folder. Defaults to DeepMIMO scenarios folder.
        overwrite (Optional[bool]): Whether to overwrite existing scenario. Defaults to None.

    Returns:
        Optional[str]: Name of the exported scenario.

    """
    # Remove conversion suffix
    new_scen_folder = sim_folder.replace(c.DEEPMIMO_CONVERSION_SUFFIX, "")

    # Get output scenario folder
    scen_name = Path(new_scen_folder).name
    scen_path = Path(target_folder) / scen_name

    # Delete scenario if it exists
    if scen_path.exists():
        shutil.rmtree(scen_path)

    # Move simulation folder to scenarios folder
    shutil.move(sim_folder, scen_path)
    return scen_name


################################################################################
### Utils for compressing path data (likely to be moved outward to paths.py) ###
################################################################################


def compress_path_data(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remove unused paths and interactions to optimize memory usage.

    This function compresses the path data by:
    1. Finding the maximum number of actual paths used
    2. Computing maximum number of interactions (bounces)
    3. Trimming arrays to remove unused entries

    Args:
        data (dict[str, np.ndarray]): Dictionary containing path information arrays
        num_paths_key (str): Key in data dict containing number of paths. Defaults to 'n_paths'

    Returns:
        dict[str, np.ndarray]: Compressed data dictionary with unused entries removed

    """
    # Compute max paths
    max_paths = get_max_paths(data)

    # Compute max bounces if interaction data exists
    max_bounces = 0
    if c.INTERACTIONS_PARAM_NAME in data:
        max_bounces = np.max(comp_next_pwr_10(data[c.INTERACTIONS_PARAM_NAME]))

    # Compress arrays to not take more than that space
    for key, value in data.items():
        if key in [c.RX_POS_PARAM_NAME, c.TX_POS_PARAM_NAME]:
            continue
        compressed = value
        if compressed.ndim >= TWO_DIMS:
            compressed = compressed[:, :max_paths, ...]
        if compressed.ndim >= THREE_DIMS:
            compressed = compressed[:, :max_paths, :max_bounces]
        data[key] = compressed

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


def get_max_paths(arr: dict[str, np.ndarray], angle_key: str = c.AOA_AZ_PARAM_NAME) -> int:
    """Find maximum number of valid paths in the dataset.

    This function determines the maximum number of valid paths by finding
    the first path index where all entries (across all receivers) are NaN.

    Args:
        arr (dict[str, np.ndarray]): Dictionary containing path information arrays
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
    # All paths contain data, return actual number of paths
    return arr[angle_key].shape[1]


def save_params(params_dict: dict[str, Any], output_folder: str) -> None:
    """Save parameters dictionary to JSON format.

    This function saves the parameters dictionary to a standardized location
    using the proper JSON serialization for numeric types.

    Args:
        params_dict: Dictionary containing all parameters
        output_folder: Output directory path

    """
    # Get standardized path for params.json
    params_path = str(Path(output_folder) / (c.PARAMS_FILENAME + ".json"))

    # Save using JSON serializer that properly handles numeric types
    save_dict_as_json(params_path, params_dict)

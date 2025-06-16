"""
General utility functions and classes for the DeepMIMO dataset generation.

This module provides utility functions and classes for handling printing,
file naming, string ID generation, and dictionary utilities used across 
the DeepMIMO toolkit.
"""

# ============================================================================
# Imports and Constants
# ============================================================================

import numpy as np
from pprint import pformat
from typing import Dict, Any, TypeVar, Mapping, Optional
from . import consts as c
import os
from tqdm import tqdm
import zipfile
import json
from .config import config

K = TypeVar("K", bound=str)
V = TypeVar("V")

# Headers for HTTP requests
HEADERS = {
    'User-Agent': 'DeepMIMO-Python/1.0',
    'Accept': '*/*'
}

# ============================================================================
# File System and Path Utilities
# ============================================================================

def check_scen_name(scen_name: str) -> None:
    """Check if a scenario name is valid.
    
    Args:
        scen_name (str): The scenario name to check
    
    """
    if np.any([char in scen_name for char in c.SCENARIO_NAME_INVALID_CHARS]):
        raise ValueError(f"Invalid scenario name: {scen_name}.\n"
                         f"Contains one of the following invalid characters: {c.SCENARIO_NAME_INVALID_CHARS}")
    return 

def get_scenarios_dir() -> str:
    """Get the absolute path to the scenarios directory.
    
    This directory contains the extracted scenario folders ready for use.
    
    Returns:
        str: Absolute path to the scenarios directory
    """
    return os.path.join(os.getcwd(), config.get('scenarios_folder'))

def get_scenario_folder(scenario_name: str) -> str:
    """Get the absolute path to a specific scenario folder.
    
    Args:
        scenario_name: Name of the scenario
        
    Returns:
        str: Absolute path to the scenario folder
    """
    check_scen_name(scenario_name)
    return os.path.join(get_scenarios_dir(), scenario_name)

def get_params_path(scenario_name: str) -> str:
    """Get the absolute path to a scenario's params file.
    
    Args:
        scenario_name: Name of the scenario
        
    Returns:
        str: Absolute path to the scenario's params file
    
    Raises:
        FileNotFoundError: If the scenario folder or params file is not found
    """
    check_scen_name(scenario_name)
    scenario_folder = get_scenario_folder(scenario_name)
    if not os.path.exists(scenario_folder):
        raise FileNotFoundError(f"Scenario folder not found: {scenario_name}")
    
    # Check if there is a params file in the scenario folder
    path = os.path.join(scenario_folder, f'{c.PARAMS_FILENAME}.json')
    if not os.path.exists(path):
        # Check if there are multiple scene folders
        subdirs = [d for d in os.listdir(scenario_folder)
                   if os.path.isdir(os.path.join(scenario_folder, d))]
        if len(subdirs):
            # Check if there is a params file in each subdirectory
            path = os.path.join(scenario_folder, subdirs[0], f'{c.PARAMS_FILENAME}.json')

    if not os.path.exists(path):
        raise FileNotFoundError(f"Params file not found for scenario: {scenario_name}")

    return path

def get_available_scenarios() -> list:
    """Get a list of all available scenarios in the scenarios directory.
    
    Returns:
        list: List of scenario names (folder names in the scenarios directory)
    """
    scenarios_dir = get_scenarios_dir()
    if not os.path.exists(scenarios_dir):
        return []
    
    # Get all subdirectories in the scenarios folder
    scenarios = [f for f in os.listdir(scenarios_dir) 
                if os.path.isdir(os.path.join(scenarios_dir, f))]
    return sorted(scenarios)

# ============================================================================
# Dictionary and Data Structure Utilities
# ============================================================================

def save_dict_as_json(output_path: str, data_dict: Dict[str, Any]) -> None:
    """Save dictionary as JSON, handling NumPy arrays and other non-JSON types.
    
    Args:
        output_path: Path to save JSON file
        data_dict: Dictionary to save
    """
    numpy_handler = lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)
    with open(output_path, 'w') as f:
        json.dump(data_dict, f, indent=2, default=numpy_handler)

def load_dict_from_json(file_path: str) -> Dict[str, Any]:
    """Load dictionary from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary containing loaded data
    """
    with open(file_path, 'r') as f:
        return json.load(f)

class DotDict(Mapping[K, V]):
    """A dictionary subclass that supports dot notation access to nested dictionaries.

    This class allows accessing dictionary items using both dictionary notation (d['key'])
    and dot notation (d.key). It automatically converts nested dictionaries to DotDict
    instances to maintain dot notation access at all levels.

    Example:
        >>> d = DotDict({'a': 1, 'b': {'c': 2}})
        >>> d.a
        1
        >>> d.b.c
        2
        >>> d['b']['c']
        2
        >>> list(d.keys())
        ['a', 'b']
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """Initialize DotDict with a dictionary.

        Args:
            dictionary: Dictionary to convert to DotDict
        """
        # Store protected attributes in a set
        self._data = {}
        if data:
            for key, value in data.items():
                if isinstance(value, dict):
                    self._data[key] = DotDict(value)
                else:
                    self._data[key] = value

    def __getattr__(self, key: str) -> Any:
        """Enable dot notation access to dictionary items."""
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        """Enable dot notation assignment."""
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self[key] = value

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Enable dictionary-style assignment."""
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        """Enable dictionary-style deletion."""
        del self._data[key]

    def update(self, other: Dict[str, Any]) -> None:
        """Update the dictionary with elements from another dictionary."""
        # Convert any nested dicts to DotDicts first
        processed = {
            k: DotDict(v) if isinstance(v, dict) and not isinstance(v, DotDict) else v
            for k, v in other.items()
        }
        self._data.update(processed)

    def __len__(self) -> int:
        """Return the length of the underlying data dictionary."""
        return len(self._data)

    def __iter__(self):
        """Return an iterator over the data dictionary keys."""
        return iter(self._data)

    def __dir__(self):
        """Return list of valid attributes."""
        return list(set(list(super().__dir__()) + list(self._data.keys())))

    def keys(self):
        """Return dictionary keys."""
        return self._data.keys()

    def values(self):
        """Return dictionary values."""
        return self._data.values()

    def items(self):
        """Return dictionary items as (key, value) pairs."""
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Get value for key, returning default if key doesn't exist."""
        return self._data.get(key, default)

    def to_dict(self) -> Dict:
        """Convert DotDict back to a regular dictionary.

        Returns:
            dict: Regular dictionary representation
        """
        result = {}
        for key, value in self._data.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def deepcopy(self) -> 'DotDict':
        """Create a deep copy of the DotDict instance.
        
        This method creates a completely independent copy of the DotDict,
        including nested dictionaries and numpy arrays. This ensures that
        modifications to the copy won't affect the original.
        
        Returns:
            DotDict: A deep copy of this instance
        """
        result = {}
        for key, value in self._data.items():
            if isinstance(value, DotDict):
                result[key] = value.deepcopy()
            elif isinstance(value, dict):
                result[key] = DotDict(value).deepcopy()
            elif isinstance(value, np.ndarray):
                result[key] = value.copy()
            else:
                result[key] = value
        return type(self)(result)  # Use the same class type as self

    def __repr__(self) -> str:
        """Return string representation of dictionary."""
        return pformat(self._data)

# ============================================================================
# Printing and Logging Utilities
# ============================================================================

class PrintIfVerbose:
    """A callable class that conditionally prints messages based on verbosity setting.

    The only purpose of this class is to avoid repeating "if verbose:" all the time. 
    
    Usage: 
        vprint = PrintIfVerbose(verbose);
        vprint(message)

    Args:
        verbose (bool): Flag to control whether messages should be printed.
    """

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose

    def __call__(self, message: str) -> None:
        """Print the message if verbose mode is enabled.

        Args:
            message (str): The message to potentially print.
        """
        if self.verbose:
            print(message)

# ============================================================================
# String Generation Utilities for TXRX ID and MAT Files
# ============================================================================

def get_txrx_str_id(tx_set_idx: int, tx_idx: int, rx_set_idx: int) -> str:
    """Generate a standardized string identifier for TX-RX combinations.

    Args:
        tx_set_idx (int): Index of the transmitter set.
        tx_idx (int): Index of the transmitter within its set.
        rx_set_idx (int): Index of the receiver set.

    Returns:
        str: Formatted string identifier in the form 't{tx_set_idx}_tx{tx_idx}_r{rx_set_idx}'.
    """
    return f"t{tx_set_idx:03}_tx{tx_idx:03}_r{rx_set_idx:03}"


def get_mat_filename(key: str, tx_set_idx: int, tx_idx: int, rx_set_idx: int) -> str:
    """Generate a .mat filename for storing DeepMIMO data.

    Args:
        key (str): The key identifier for the data type.
        tx_set_idx (int): Index of the transmitter set.
        tx_idx (int): Index of the transmitter within its set.
        rx_set_idx (int): Index of the receiver set.

    Returns:
        str: Complete filename with .mat extension.
    """
    str_id = get_txrx_str_id(tx_set_idx, tx_idx, rx_set_idx)
    return f"{key}_{str_id}.mat"

# ============================================================================
# Compression Utilities
# ============================================================================

def zip(folder_path: str) -> str:
    """Create zip archive of folder contents.

    This function creates a zip archive containing all files and subdirectories in the 
    specified folder. The archive is created in the same directory as the folder with
    '.zip' appended to the folder name. The directory structure is preserved in the zip.

    Args:
        folder_path (str): Path to folder to be zipped

    Returns:
        Path to the created zip file
    """
    zip_path = folder_path + ".zip"
    
    # Get all files and folders recursively
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Get full path of file
            file_path = os.path.join(root, file)
            # Get relative path from the base folder for preserving structure
            rel_path = os.path.relpath(file_path, os.path.dirname(folder_path))
            all_files.append((file_path, rel_path))

    # Create a zip file
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file_path, rel_path in tqdm(all_files, desc="Compressing", unit="file"):
            zipf.write(file_path, rel_path)

    return zip_path


def unzip(path_to_zip: str) -> str:
    """Extract a zip file to its parent directory.

    This function extracts the contents of a zip file to the directory
    containing the zip file.

    Args:
        path_to_zip (str): Path to the zip file to extract.

    Raises:
        zipfile.BadZipFile: If zip file is corrupted.
        OSError: If extraction fails due to file system issues.

    Returns:
        Path to the extracted folder
    """
    extracted_path = path_to_zip.replace(".zip", "")
    with zipfile.ZipFile(path_to_zip, "r") as zip_ref:
        files = zip_ref.namelist()
        for file in tqdm(files, desc="Extracting", unit="file"):
            zip_ref.extract(file, extracted_path)

    return extracted_path

# ============================================================================
# Coordinate Utilities
# ============================================================================

def cartesian_to_spherical(cartesian_coords: np.ndarray) -> np.ndarray:
    """Convert Cartesian coordinates to spherical coordinates.
    
    Args:
        cartesian_coords: Array of shape [n_points, 3] containing Cartesian coordinates (x, y, z)
        
    Returns:
        Array of shape [n_points, 3] containing spherical coordinates (r, azimuth, elevation) in radians
        where r is the magnitude (distance from origin)
    """
    spherical_coords = np.zeros((cartesian_coords.shape[0], 3))
    
    # Calculate magnitude (r) - distance from origin
    spherical_coords[:, 0] = np.sqrt(np.sum(cartesian_coords**2, axis=1))
    
    # Calculate azimuth (φ) - angle in xy plane
    spherical_coords[:, 1] = np.arctan2(cartesian_coords[:, 1], cartesian_coords[:, 0])
    
    # Calculate elevation (θ) - angle from xy plane
    r_xy = np.sqrt(cartesian_coords[:, 0]**2 + cartesian_coords[:, 1]**2)
    spherical_coords[:, 2] = np.arctan2(cartesian_coords[:, 2], r_xy)
    
    return spherical_coords

def spherical_to_cartesian(spherical_coords: np.ndarray) -> np.ndarray:
    """Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        spherical_coords: Array containing spherical coordinates (r, azimuth, elevation) in radians
            where r is the magnitude (distance from origin). Can have any number of leading dimensions,
            but the last dimension must be 3.
        
    Returns:
        Array of same shape as input containing Cartesian coordinates (x, y, z)
    """
    # Preserve input shape
    cartesian_coords = np.zeros_like(spherical_coords)
    r = spherical_coords[..., 0]
    azimuth = spherical_coords[..., 1]
    elevation = spherical_coords[..., 2]
    
    cartesian_coords[..., 0] = r * np.cos(elevation) * np.cos(azimuth)  # x
    cartesian_coords[..., 1] = r * np.cos(elevation) * np.sin(azimuth)  # y
    cartesian_coords[..., 2] = r * np.sin(elevation)                    # z
    
    return cartesian_coords

# ============================================================================
# Other Utilities
# ============================================================================

def compare_two_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> bool:
    """Compare two dictionaries for equality.
            
    This function performs a deep comparison of two dictionaries, handling
    nested dictionaries.
    
    Args:
        dict1 (dict): First dictionary to compare
        dict2 (dict): Second dictionary to compare

    Returns:
        set: Set of keys in dict1 that are not in dict2
    """
    additional_keys = dict1.keys() - dict2.keys()
    for key, item in dict1.items():
        if isinstance(item, dict):
            if key in dict2:
                additional_keys = additional_keys | compare_two_dicts(dict1[key], dict2[key])
    return additional_keys


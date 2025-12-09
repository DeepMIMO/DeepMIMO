"""General utility functions and classes for the DeepMIMO dataset generation.

This module provides utility functions and classes for handling printing,
file naming, string ID generation, and dictionary utilities used across
the DeepMIMO toolkit.
"""

import json
import os
import pickle
import zipfile
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from typing import Any, TypeVar

import numpy as np
import scipy.io
from tqdm import tqdm

from . import consts as c
from .config import config

K = TypeVar("K", bound=str)
V = TypeVar("V")
HEADERS = {"User-Agent": "DeepMIMO-Python/1.0", "Accept": "*/*"}


def check_scen_name(scen_name: str) -> None:
    """Check if a scenario name is valid.

    Args:
        scen_name (str): The scenario name to check

    """
    if np.any([char in scen_name for char in c.SCENARIO_NAME_INVALID_CHARS]):
        invalids = c.SCENARIO_NAME_INVALID_CHARS
        msg = (
            f"Invalid scenario name: {scen_name}.\n"
            f"Contains one of these invalid characters: {invalids}"
        )
        raise ValueError(msg)


def get_scenarios_dir() -> str:
    """Get the absolute path to the scenarios directory.

    This directory contains the extracted scenario folders ready for use.

    Returns:
        str: Absolute path to the scenarios directory

    """
    return str(Path.cwd() / config.get("scenarios_folder"))


def get_scenario_folder(scenario_name: str) -> str:
    """Get the absolute path to a specific scenario folder.

    Args:
        scenario_name: Name of the scenario

    Returns:
        str: Absolute path to the scenario folder

    """
    check_scen_name(scenario_name)
    return str(Path(get_scenarios_dir()) / scenario_name)


def get_rt_sources_dir() -> str:
    """Get the absolute path to the ray tracing sources directory.

    This directory contains the downloaded RT source files.

    Returns:
        str: Absolute path to the RT sources directory

    """
    return str(Path.cwd() / config.get("rt_sources_folder"))


def get_rt_source_folder(scenario_name: str) -> str:
    """Get the absolute path to a specific RT source folder.

    Args:
        scenario_name: Name of the scenario

    Returns:
        str: Absolute path to the RT source folder (extracted contents)

    """
    check_scen_name(scenario_name)
    return str(Path(get_rt_sources_dir()) / scenario_name)


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
    scenario_folder_path = Path(scenario_folder)
    if not scenario_folder_path.exists():
        msg = f"Scenario folder not found: {scenario_name}"
        raise FileNotFoundError(msg)
    path = scenario_folder_path / f"{c.PARAMS_FILENAME}.json"
    if not path.exists():
        subdirs = [d for d in scenario_folder_path.iterdir() if d.is_dir()]
        if subdirs:
            path = subdirs[0] / f"{c.PARAMS_FILENAME}.json"
    if not path.exists():
        msg = f"Params file not found for scenario: {scenario_name}"
        raise FileNotFoundError(msg)
    return str(path)


def get_available_scenarios() -> list:
    """Get a list of all available scenarios in the scenarios directory.

    Returns:
        list: List of scenario names (folder names in the scenarios directory)

    """
    scenarios_dir = get_scenarios_dir()
    scenarios_dir_path = Path(scenarios_dir)
    if not scenarios_dir_path.exists():
        return []
    scenarios = [f.name for f in scenarios_dir_path.iterdir() if f.is_dir()]
    return sorted(scenarios)


def get_mat_filename(
    key: str, tx_set_idx: int, tx_idx: int, rx_set_idx: int, fmt: str = c.MAT_FMT
) -> str:
    """Generate a .mat filename for storing DeepMIMO data.

    Args:
        key (str): The key identifier for the data type.
        tx_set_idx (int): Index of the transmitter set.
        tx_idx (int): Index of the transmitter within its set.
        rx_set_idx (int): Index of the receiver set.
        fmt (str): File extension/format. Defaults to `c.MAT_FMT`.

    Returns:
        str: Complete filename with .mat extension.

    """
    str_id = f"t{tx_set_idx:03}_tx{tx_idx:03}_r{rx_set_idx:03}"
    return f"{key}_{str_id}.{fmt}"


def save_mat(data: np.ndarray, data_key: str, file_path: str, fmt: str = c.MAT_FMT) -> None:
    """Save data to a .mat file with standardized naming.

    This function saves data to a .mat file using standardized naming conventions.
    If transmitter/receiver indices are provided, the filename will include those indices.
    Otherwise, it will use just the data_key as the filename.

    For example:
    - With indices: {data_key}_t{tx_set_idx}_{tx_idx}_r{rx_set_idx}.mat
    - Without indices: {data_key}.mat

    Args:
        data: Data array to save
        data_key: Key identifier for the data type
        file_path: Output path
        fmt: File format/extension. Defaults to `c.MAT_FMT`.
        fmt: File extension/format. Defaults to `c.MAT_FMT`.

    """
    if fmt == "mat":
        scipy.io.savemat(file_path, {data_key: data})
    elif fmt == "npz":
        np.savez_compressed(file_path.replace(".mat", ".npz"), **{data_key: data})
    elif fmt == "npy":
        np.save(file_path.replace(".mat", ".npz"), data)
    else:
        msg = 'Format {fmt} not recognized. Choose "mat" (default), "npz" or "npy".'
        raise ValueError(msg)


def load_mat(mat_path: str, key: str | None = None) -> Any:
    """Load a .mat file with supported extensions (mat, npz, npy).

    This function tries to load a .mat file with supported extensions (mat, npz, npy).
    If the file is not found, it raises an exception.

    Args:
        mat_path: Path to the .mat file
        key: Optional key for npz/npy loads

    """
    supported_formats = [".mat", ".npz", ".npy"]
    mat_path_obj = Path(mat_path)
    base_path = mat_path_obj.parent / mat_path_obj.stem
    for fmt in supported_formats:
        try_path = base_path.with_suffix(fmt)
        if try_path.exists():
            if fmt == ".mat":
                return scipy.io.loadmat(try_path)[key]
            if fmt == ".npz":
                return np.load(try_path, allow_pickle=True)[key]
            if fmt == ".npy":
                return np.load(try_path)
    print(f"No supported format found for {mat_path}. Supported formats are: {supported_formats}")
    return None


def save_dict_as_json(output_path: str, data_dict: dict[str, Any]) -> None:
    """Save dictionary as JSON, handling NumPy arrays and other non-JSON types.

    Args:
        output_path: Path to save JSON file
        data_dict: Dictionary to save

    """
    if not output_path.endswith(".json"):
        output_path += ".json"

    def numpy_handler(x: Any) -> list[Any] | str:
        return x.tolist() if isinstance(x, np.ndarray) else str(x)

    with Path(output_path).open("w") as f:
        json.dump(data_dict, f, indent=2, default=numpy_handler)


def load_dict_from_json(file_path: str) -> dict[str, Any]:
    """Load dictionary from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary containing loaded data

    """
    with Path(file_path).open() as f:
        return json.load(f)


def deep_dict_merge(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, preserving values from dict1 for keys not in dict2.

    This function recursively merges two dictionaries, keeping values from dict1
    for keys that are not present in dict2. For keys present in both dictionaries,
    if both values are dictionaries, they are recursively merged. Otherwise, the
    value from dict2 is used.

    Args:
        dict1: Base dictionary to merge into
        dict2: Dictionary with values to override

    Returns:
        Merged dictionary

    Example:
        >>> dict1 = {'a': 1, 'b': {'c': 2, 'd': 3}}
        >>> dict2 = {'b': {'c': 4}}
        >>> deep_dict_merge(dict1, dict2)
        {'a': 1, 'b': {'c': 4, 'd': 3}}

    """
    if hasattr(dict1, "to_dict"):
        dict1 = dict1.to_dict()
    if hasattr(dict2, "to_dict"):
        dict2 = dict2.to_dict()
    result = deepcopy(dict1)
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_dict_merge(result[key], value)
        else:
            result[key] = value
    return result


def compare_two_dicts(dict1: dict[str, Any], dict2: dict[str, Any]) -> bool:
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
        if isinstance(item, dict) and key in dict2:
            additional_keys = additional_keys | compare_two_dicts(item, dict2[key])
    return additional_keys


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

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Initialize DotDict with a dictionary.

        Args:
            data: Dictionary to convert to DotDict

        """
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
        except KeyError as err:
            raise AttributeError(key) from err

    def __setattr__(self, key: str, value: Any) -> None:
        """Enable dot notation assignment with property support.

        This method first checks if the attribute is a property with a setter.
        If it is, it uses the property setter. Otherwise, it falls back to
        storing the value in the internal dictionary.
        """
        if key == "_data":
            super().__setattr__(key, value)
            return
        attr = getattr(type(self), key, None)
        if isinstance(attr, property) and attr.fset is not None:
            attr.fset(self, value)
        else:
            if isinstance(value, dict) and (not isinstance(value, DotDict)):
                value = DotDict(value)
            self._data[key] = value

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Enable dictionary-style assignment."""
        if isinstance(value, dict) and (not isinstance(value, DotDict)):
            value = DotDict(value)
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        """Enable dictionary-style deletion."""
        del self._data[key]

    def update(self, other: dict[str, Any]) -> None:
        """Update the dictionary with elements from another dictionary."""
        processed = {
            k: DotDict(v) if isinstance(v, dict) and (not isinstance(v, DotDict)) else v
            for (k, v) in other.items()
        }
        self._data.update(processed)

    def __len__(self) -> int:
        """Return the length of the underlying data dictionary."""
        return len(self._data)

    def __iter__(self) -> Any:
        """Return an iterator over the data dictionary keys."""
        return iter(self._data)

    def __dir__(self) -> Any:
        """Return list of valid attributes."""
        return list(set(list(super().__dir__()) + list(self._data.keys())))

    def keys(self) -> Any:
        """Return dictionary keys."""
        return self._data.keys()

    def values(self) -> Any:
        """Return dictionary values."""
        return self._data.values()

    def items(self) -> Any:
        """Return dictionary items as (key, value) pairs."""
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Get value for key, returning default if key doesn't exist."""
        return self._data.get(key, default)

    def hasattr(self, key: str) -> bool:
        """Safely check if a key exists in the dictionary.

        This method provides a safe way to check for attribute existence
        without raising KeyError, similar to Python's built-in hasattr().

        Args:
            key: The key to check for

        Returns:
            bool: True if the key exists, False otherwise

        """
        return key in self._data

    def to_dict(self) -> dict:
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

    def deepcopy(self) -> "DotDict":
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
        return type(self)(result)

    def __repr__(self) -> str:
        """Return string representation of dictionary."""
        return pformat(self._data)


class PrintIfVerbose:
    """A callable class that conditionally prints messages based on verbosity setting.

    The only purpose of this class is to avoid repeating "if verbose:" all the time.

    Usage:
        vprint = PrintIfVerbose(verbose);
        vprint(message)

    Args:
        verbose (bool): Flag to control whether messages should be printed.

    """

    def __init__(self, *, verbose: bool) -> None:
        """Store verbosity flag.

        Args:
            verbose: Print messages when True.

        """
        self.verbose = verbose

    def __call__(self, message: str) -> None:
        """Print the message if verbose mode is enabled.

        Args:
            message (str): The message to potentially print.

        """
        if self.verbose:
            print(message)


def zip_folder(folder_path: str) -> str:
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
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(folder_path)
            all_files.append((str(file_path), str(rel_path)))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file_path, rel_path in tqdm(all_files, desc="Compressing", unit="file"):
            zipf.write(file_path, rel_path)
    return zip_path


def zip(folder_path: str) -> str:  # noqa: A001
    """Backward-compatible wrapper around `zip_folder`."""
    return zip_folder(folder_path)


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


def cartesian_to_spherical(cartesian_coords: np.ndarray) -> np.ndarray:
    """Convert Cartesian coordinates to spherical coordinates.

    Args:
        cartesian_coords: Array [n_points, 3] of Cartesian coordinates (x, y, z)

    Returns:
        Array [n_points, 3] of spherical coordinates (r, azimuth, elevation) in radians.
        r is the magnitude (distance from origin).

    """
    spherical_coords = np.zeros((cartesian_coords.shape[0], 3))
    spherical_coords[:, 0] = np.sqrt(np.sum(cartesian_coords**2, axis=1))
    spherical_coords[:, 1] = np.arctan2(cartesian_coords[:, 1], cartesian_coords[:, 0])
    r_xy = np.sqrt(cartesian_coords[:, 0] ** 2 + cartesian_coords[:, 1] ** 2)
    spherical_coords[:, 2] = np.arctan2(cartesian_coords[:, 2], r_xy)
    return spherical_coords


def spherical_to_cartesian(spherical_coords: np.ndarray) -> np.ndarray:
    """Convert spherical coordinates to Cartesian coordinates.

    Args:
        spherical_coords: Array with spherical coordinates (r, elevation, azimuth) in radians.
            r is the magnitude (distance from origin). Leading dimensions allowed; last
            dimension must be 3.
            Reference: https://en.wikipedia.org/wiki/Spherical_coordinate_system
            Note: DeepMIMO uses elevation from the xy plane, while Sionna/Wikipedia use z-axis.

    Returns:
        Array of same shape containing Cartesian coordinates (x, y, z).

    """
    cartesian_coords = np.zeros_like(spherical_coords)
    r = spherical_coords[..., 0]
    elevation = spherical_coords[..., 1]
    azimuth = spherical_coords[..., 2]
    cartesian_coords[..., 0] = r * np.sin(elevation) * np.cos(azimuth)
    cartesian_coords[..., 1] = r * np.sin(elevation) * np.sin(azimuth)
    cartesian_coords[..., 2] = r * np.cos(elevation)
    return cartesian_coords


class DelegatingList(list):
    """A list subclass that delegates method calls to each item in the list.

    When a method is called on this class, it will be called on each item in the list
    and the results will be returned as a list.
    """

    def __getattr__(self, name: Any) -> Any:
        """Delegate attribute access to each item in the list.

        If the attribute is a method, it will be called on each item and results returned as a list.
        If the attribute is a property, a list of property values will be returned.
        If the attribute is a list-like object, it will be wrapped in a DelegatingList.
        """
        if not self:
            msg = f"Empty list has no attribute '{name}'"
            raise AttributeError(msg)
        first_attr = getattr(self[0], name)
        if callable(first_attr):

            def method(*args: Any, **kwargs: Any) -> DelegatingList:
                results = [getattr(item, name)(*args, **kwargs) for item in self]
                return DelegatingList(results)

            return method
        results = [getattr(item, name) for item in self]
        return DelegatingList(results)

    def __setattr__(self, name: Any, value: Any) -> None:
        """Delegate attribute assignment to each item in the list.

        If value is a list/iterable, each item in the list gets the corresponding value.
        Otherwise, all items get the same value.
        """
        if name in self.__dict__:
            super().__setattr__(name, value)
            return
        if not self:
            msg = f"Empty list has no attribute '{name}'"
            raise AttributeError(msg)
        if (
            hasattr(value, "__iter__")
            and (not isinstance(value, (str, bytes)))
            and (len(value) == len(self))
        ):
            for item, val in __builtins__["zip"](self, value):
                setattr(item, name, val)
        else:
            for item in self:
                setattr(item, name, value)


def save_pickle(obj: Any, filename: str) -> None:
    """Save an object to a pickle file.

    Args:
        obj (Any): Object to save
        filename (str): Path to save pickle file

    Raises:
        IOError: If file cannot be written

    """
    with Path(filename).open("wb") as file:
        pickle.dump(obj, file)


def load_pickle(filename: str) -> Any:
    """Load an object from a pickle file.

    Args:
        filename (str): Path to pickle file

    Returns:
        Any: Unpickled object

    Raises:
        FileNotFoundError: If file does not exist
        pickle.UnpicklingError: If file cannot be unpickled

    """
    with Path(filename).open("rb") as file:
        # Trusted internal data only; external inputs should avoid pickle for safety.
        return pickle.load(file)  # noqa: S301

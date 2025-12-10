"""I/O utilities for DeepMIMO.

This module provides functions for reading and writing various file formats
used in DeepMIMO, including MATLAB files, JSON, pickle, and zip archives.
"""

import json
import os
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
from tqdm import tqdm

from deepmimo import consts as c


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

    """
    if fmt == "mat":
        scipy.io.savemat(file_path, {data_key: data})
    elif fmt == "npz":
        np.savez_compressed(file_path.replace(".mat", ".npz"), **{data_key: data})
    elif fmt == "npy":
        np.save(file_path.replace(".mat", ".npz"), data)
    else:
        msg = f'Format {fmt} not recognized. Choose "mat" (default), "npz" or "npy".'
        raise ValueError(msg)


def load_mat(mat_path: str, key: str | None = None) -> np.ndarray | None:
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


def _numpy_handler(x: Any) -> list[Any] | str:
    """Convert numpy arrays to lists for JSON serialization."""
    return x.tolist() if isinstance(x, np.ndarray) else str(x)


def save_dict_as_json(output_path: str, data_dict: dict[str, Any]) -> None:
    """Save dictionary as JSON, handling NumPy arrays and other non-JSON types.

    Args:
        output_path: Path to save JSON file
        data_dict: Dictionary to save

    """
    if not output_path.endswith(".json"):
        output_path += ".json"

    with Path(output_path).open("w") as f:
        json.dump(data_dict, f, indent=2, default=_numpy_handler)


def load_dict_from_json(file_path: str) -> dict[str, Any]:
    """Load dictionary from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary containing loaded data

    """
    with Path(file_path).open() as f:
        return json.load(f)


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


def zip(folder_path: str) -> str:  # noqa: A001
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

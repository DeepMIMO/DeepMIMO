"""DeepMIMO Dataset Loading Module.

This module provides functionality for loading DeepMIMO scenarios from disk.
It handles:
- Loading ray-tracing data from saved files
- Creating Dataset and MacroDataset objects
- Managing scenario parameters and metadata
- TX/RX set validation and loading
"""

# Standard library imports
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np

# Local imports
from deepmimo import consts as c
from deepmimo.core.materials import MaterialList
from deepmimo.core.scene import Scene
from deepmimo.datasets.dataset import Dataset, DynamicDataset, MacroDataset
from deepmimo.utils import (
    DotDict,
    get_mat_filename,
    get_params_path,
    get_scenario_folder,
    load_dict_from_json,
    load_mat,
)


def load(scen_name: str, **load_params: Any) -> Dataset | MacroDataset:
    """Load a DeepMIMO scenario.

    This function loads raytracing data and creates a Dataset or MacroDataset instance.

    Args:
        scen_name (str): Name of the scenario to load
        **load_params: Additional parameters for loading the scenario. Can be passed as a dictionary
            or as keyword arguments. Available parameters are:

            * max_paths (int, optional): Maximum number of paths to load. Defaults to 10.

            * tx_sets (dict or list or str, optional): Transmitter sets to load.
                Defaults to 'all'. Can be:
                - dict: Mapping of set IDs to lists of indices or 'all'
                - list: List of set IDs to load all indices from
                - str: 'all' to load all sets and indices

            * rx_sets (dict or list or str, optional): Receiver sets to load.
                Same format as tx_sets. Defaults to 'all'.

            * matrices (list of str or str, optional): List of matrix names to load.
                Defaults to 'all'. Can be:
                - list: List of matrix names to load
                - str: 'all' to load all available matrices

    Returns:
        Dataset or MacroDataset: Loaded dataset(s)

    Raises:
        ValueError: If scenario files cannot be loaded

    """
    # Convert scenario name to lowercase for robustness
    scen_name = scen_name.lower()

    # Handle absolute paths
    if Path(scen_name).is_absolute():
        scen_folder = scen_name
        scen_name = Path(scen_folder).name
    else:
        scen_folder = get_scenario_folder(scen_name)

    # Download scenario if needed
    if not Path(scen_folder).exists():
        print("Scenario not found. Would you like to download it? [Y/n]")
        response = input().lower()
        if response in ["", "y", "yes"]:
            # Lazy import to avoid circular dependency
            from deepmimo.api import download  # noqa: PLC0415

            download(scen_name)
        else:
            msg = f"Scenario {scen_name} not found"
            raise ValueError(msg)

    # Load parameters file
    params_file = get_params_path(scen_name)
    params = load_dict_from_json(params_file)

    # Load scenario data
    n_snapshots = params[c.SCENE_PARAM_NAME][c.SCENE_PARAM_NUMBER_SCENES]
    if n_snapshots > 1:  # dynamic (multiple scenes)
        dataset_list = []
        scen_folder_path = Path(scen_folder)
        scene_folders = sorted([d.name for d in scen_folder_path.iterdir() if d.is_dir()])
        for snapshot_i in range(n_snapshots):
            snapshot_folder = str(scen_folder_path / scene_folders[snapshot_i])
            print(f"Scene {snapshot_i + 1}/{n_snapshots}")
            dataset_list += [_load_dataset(snapshot_folder, params, load_params)]
        dataset = DynamicDataset(dataset_list, scen_name)
    else:  # static (single scene)
        dataset = _load_dataset(scen_folder, params, load_params)
    return dataset


def _load_dataset(folder: str, params: dict, load_params: dict) -> Dataset | MacroDataset:
    """Load a single dataset from a scenario folder.

    Args:
        folder: Path to the scenario folder
        params: Dictionary containing scenario parameters
        load_params: Dictionary containing parameters for loading the dataset

    Returns:
        Dataset or MacroDataset: Loaded dataset with shared parameters set

    """
    dataset = _load_raytracing_scene(folder, params[c.TXRX_PARAM_NAME], **load_params)

    # Set shared parameters
    dataset[c.NAME_PARAM_NAME] = Path(folder).name

    dataset[c.RT_PARAMS_PARAM_NAME] = params[c.RT_PARAMS_PARAM_NAME]
    dataset[c.SCENE_PARAM_NAME] = Scene.from_data(folder)
    dataset[c.MATERIALS_PARAM_NAME] = MaterialList.from_dict(params[c.MATERIALS_PARAM_NAME])

    return dataset


def _load_raytracing_scene(  # noqa: PLR0913 - all scene loading parameters required
    scene_folder: str,
    txrx_dict: dict,
    max_paths: int = c.MAX_PATHS,
    tx_sets: dict[int, list | str] | list | str = "all",
    rx_sets: dict[int, list | str] | list | str = "rx_only",
    matrices: list[str] | str = "all",
) -> Dataset:
    """Load raytracing data for a scene.

    Args:
        scene_folder (str): Path to the folder containing raytracing data files.
        txrx_dict (dict): Dictionary containing transmitter and receiver sets.
        max_paths (int): Maximum number of paths to load. Defaults to 5.
        tx_sets (dict or list or str): Transmitter sets to load. Defaults to 'all'.
        rx_sets (dict or list or str): Receiver sets to load. Defaults to 'rx_only'.
        matrices (list of str or str): List of matrix names to load. Defaults to 'all'.

    Returns:
        Dataset: Dataset containing the requested matrices for each tx-rx pair

    """
    tx_sets = _validate_txrx_sets(tx_sets, txrx_dict, "tx")
    rx_sets = _validate_txrx_sets(rx_sets, txrx_dict, "rx")
    dataset_list = []
    bs_idx = 0

    for tx_set_id, tx_idxs in tx_sets.items():
        for rx_set_id, rx_idxs in rx_sets.items():
            for tx_idx in tx_idxs:
                dataset_list.append({})
                print(
                    "Loading TXRX PAIR: "
                    f"TXset {tx_set_id} (tx_idx {tx_idx}) & RXset {rx_set_id} "
                    f"(rx_idxs {len(rx_idxs)})",
                )
                dataset_list[bs_idx] = _load_tx_rx_raydata(
                    scene_folder,
                    tx_set_id,
                    rx_set_id,
                    tx_idx,
                    rx_idxs,
                    max_paths,
                    matrices,
                )

                dataset_list[bs_idx]["txrx"] = {
                    "tx_set_id": tx_set_id,
                    "rx_set_id": rx_set_id,
                    "tx_idx": int(tx_idx),
                }
                bs_idx += 1

    # Convert dictionary to Dataset at the end
    if len(dataset_list) > 1:
        final_dataset = MacroDataset([Dataset(d_dict) for d_dict in dataset_list])
    else:
        final_dataset = Dataset(dataset_list[0])

    final_dataset[c.LOAD_PARAMS_PARAM_NAME] = DotDict(
        {
            "max_paths": max_paths,
            "tx_sets": tx_sets,
            "rx_sets": rx_sets,
            "matrices": matrices,
        },
    )
    return final_dataset


def _load_tx_rx_raydata(  # noqa: PLR0913 - all raytracing data loading parameters required
    rayfolder: str,
    tx_set_id: int,
    rx_set_id: int,
    tx_idx: int,
    rx_idxs: np.ndarray | list,
    max_paths: int,
    matrices_to_load: list[str] | str = "all",
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Load raytracing data for a transmitter-receiver pair.

    This function loads raytracing data files containing path information
    between a transmitter and set of receivers.

    Args:
        rayfolder (str): Path to folder containing raytracing data
        tx_set_id (int): Index of transmitter set
        rx_set_id (int): Index of receiver set
        tx_idx (int): Index of transmitter within set
        rx_idxs (numpy.ndarray or list): Indices of receivers to load
        max_paths (int): Maximum number of paths to load
        matrices_to_load (list of str, optional): List of matrix names to load.
        verbose: Whether to print detailed info while loading.

    Returns:
        dict: Dictionary containing loaded raytracing data

    Raises:
        ValueError: If required data files are missing or invalid

    """
    tx_dict = {
        c.AOA_AZ_PARAM_NAME: None,
        c.AOA_EL_PARAM_NAME: None,
        c.AOD_AZ_PARAM_NAME: None,
        c.AOD_EL_PARAM_NAME: None,
        c.POWER_PARAM_NAME: None,
        c.PHASE_PARAM_NAME: None,
        c.DELAY_PARAM_NAME: None,
        c.RX_POS_PARAM_NAME: None,
        c.TX_POS_PARAM_NAME: None,
        c.INTERACTIONS_PARAM_NAME: None,
        c.INTERACTIONS_POS_PARAM_NAME: None,
    }

    if matrices_to_load == "all":
        matrices_to_load = tx_dict.keys()
    else:
        matrices_to_load = [] if matrices_to_load is None else matrices_to_load
        valid_matrices = set(tx_dict.keys())
        invalid = set(matrices_to_load) - valid_matrices
        if invalid:
            msg = f"Invalid matrix names: {invalid}. Valid names are: {valid_matrices}"
            raise ValueError(msg)

    for key, _value in list(tx_dict.items()):
        if key not in matrices_to_load:
            continue

        mat_filename = get_mat_filename(key, tx_set_id, tx_idx, rx_set_id)
        mat_path = str(Path(rayfolder) / mat_filename)

        if verbose:
            print(f"Loading {mat_filename}...", end="")

        tx_dict[key] = load_mat(mat_path, key)

        if tx_dict[key] is None:
            continue

        # Filter by selected rx indices (all but tx positions)
        if key != c.TX_POS_PARAM_NAME:
            tx_dict[key] = tx_dict[key][rx_idxs]

        # Trim by max paths
        if key not in [c.RX_POS_PARAM_NAME, c.TX_POS_PARAM_NAME]:
            tx_dict[key] = tx_dict[key][:, :max_paths, ...]

        if verbose:
            print(f"Done. Shape: {tx_dict[key].shape}")
    return tx_dict


# Helper functions
def _validate_set_id(set_id: int, valid_set_ids: list[int], set_str: str, info_str: str) -> None:
    """Validate that a set ID is in the allowed list.

    Args:
        set_id: Set ID to validate
        valid_set_ids: List of valid set IDs
        set_str: "Tx" or "Rx" for error messages
        info_str: Additional info string for error messages

    Raises:
        ValueError: If set ID is not valid

    """
    if set_id not in valid_set_ids:
        raise ValueError(f"{set_str} set {set_id} not in allowed sets {valid_set_ids}\n" + info_str)


def _process_dict_sets(
    sets: dict[int, list | str | np.ndarray],
    txrx_dict: dict[str, Any],
    valid_set_ids: list[int],
    set_str: str,
    info_str: str,
) -> dict[int, np.ndarray]:
    """Process TX/RX sets specified as a dictionary.

    Args:
        sets: Dictionary mapping set IDs to indices
        txrx_dict: Raytracing parameters
        valid_set_ids: List of valid set IDs
        set_str: "Tx" or "Rx" for error messages
        info_str: Additional info string for error messages

    Returns:
        Processed dictionary with numpy arrays

    Raises:
        ValueError: If any set ID or indices are invalid

    """
    for set_id, idxs in sets.items():
        _validate_set_id(set_id, valid_set_ids, set_str, info_str)

        # Get the txrx_set info for this index
        txrx_set_key = f"txrx_set_{set_id}"
        txrx_set = txrx_dict[txrx_set_key]
        all_idxs_available = np.arange(txrx_set["num_points"])

        # Convert to numpy array if needed
        if isinstance(idxs, np.ndarray):
            pass  # Already correct type
        elif isinstance(idxs, list):
            sets[set_id] = np.array(idxs)
        elif isinstance(idxs, str):
            if idxs == "all":
                sets[set_id] = all_idxs_available
            else:
                msg = f"String '{idxs}' not recognized for tx/rx indices "
                raise ValueError(msg)
        else:
            msg = "Only <list> of <np.ndarray> allowed as tx/rx indices"
            raise TypeError(msg)

        # Validate that indices are within available range
        if not set(sets[set_id]).issubset(set(all_idxs_available.tolist())):
            raise ValueError(f"Some indices of {idxs} are not in {all_idxs_available}. " + info_str)

    return sets


def _process_list_sets(
    sets: list[int],
    txrx_dict: dict[str, Any],
    valid_set_ids: list[int],
    set_str: str,
    info_str: str,
) -> dict[int, np.ndarray]:
    """Process TX/RX sets specified as a list.

    Args:
        sets: List of set IDs
        txrx_dict: Raytracing parameters
        valid_set_ids: List of valid set IDs
        set_str: "Tx" or "Rx" for error messages
        info_str: Additional info string for error messages

    Returns:
        Dictionary mapping set IDs to all available indices

    Raises:
        ValueError: If any set ID is invalid

    """
    sets_dict = {}
    for set_id in sets:
        _validate_set_id(set_id, valid_set_ids, set_str, info_str)
        sets_dict[set_id] = np.arange(txrx_dict[f"txrx_set_{set_id}"]["num_points"])
    return sets_dict


def _process_str_sets(
    sets: str,
    txrx_dict: dict[str, Any],
    valid_set_ids: list[int],
    tx_or_rx: str,
) -> dict[int, np.ndarray]:
    """Process TX/RX sets specified as a string.

    Args:
        sets: String specification ("all" or "rx_only")
        txrx_dict: Raytracing parameters
        valid_set_ids: List of valid set IDs
        tx_or_rx: "tx" or "rx"

    Returns:
        Dictionary mapping set IDs to all available indices

    Raises:
        ValueError: If string is not recognized

    """
    if sets not in ["all", "rx_only"]:
        msg = (
            f"String '{sets}' not understood. Only strings allowed "
            "are 'all' to generate all available sets and indices, "
            "or 'rx_only' to generate all available rx sets and indices"
        )
        raise ValueError(msg)

    sets_dict = {}
    for set_id in valid_set_ids:
        set_dict = txrx_dict[f"txrx_set_{set_id}"]

        # If rx_only, only include sets that are only rx
        if sets == "rx_only" and tx_or_rx == "rx" and set_dict["is_tx"]:
            continue
        sets_dict[set_id] = np.arange(set_dict["num_points"])

    return sets_dict


def _validate_txrx_sets(
    sets: dict[int, list | str] | list | str,
    txrx_dict: dict[str, Any],
    tx_or_rx: str = "tx",
) -> dict[int, list]:
    """Validate and process TX/RX set specifications.

    This function validates and processes transmitter/receiver set specifications,
    ensuring they match the available sets in the raytracing parameters.

    Args:
        sets (dict or list or str): TX/RX set specifications as dict, list, or string
        txrx_dict (dict): Raytracing parameters containing valid set information
        tx_or_rx (str): Whether validating TX or RX sets. Defaults to 'tx'

    Returns:
        dict: Dictionary mapping set indices to lists of valid TX/RX indices

    Raises:
        ValueError: If invalid TX/RX sets are specified

    """
    # Get valid TX/RX sets in a deterministic order
    tx_sets = [txrx_dict[key] for key in sorted(txrx_dict.keys()) if txrx_dict[key]["is_tx"]]
    rx_sets = [txrx_dict[key] for key in sorted(txrx_dict.keys()) if txrx_dict[key]["is_rx"]]

    valid_tx_set_ids = [tx_set["id"] for tx_set in tx_sets]
    valid_rx_set_ids = [rx_set["id"] for rx_set in rx_sets]

    valid_set_ids = valid_tx_set_ids if tx_or_rx == "tx" else valid_rx_set_ids
    set_str = "Tx" if tx_or_rx == "tx" else "Rx"

    info_str = "To see supported TX/RX sets and indices run dm.info(<scenario_name>)"

    # Process based on input type
    if isinstance(sets, dict):
        return _process_dict_sets(sets, txrx_dict, valid_set_ids, set_str, info_str)
    if isinstance(sets, list):
        return _process_list_sets(sets, txrx_dict, valid_set_ids, set_str, info_str)
    if isinstance(sets, str):
        return _process_str_sets(sets, txrx_dict, valid_set_ids, tx_or_rx)

    msg = "Sets must be dict, list, or str"
    raise ValueError(msg)


def validate_txrx_sets(
    sets: dict[int, list | str] | list | str,
    txrx_dict: dict[str, Any],
    tx_or_rx: str = "tx",
) -> dict[int, list]:
    """Public wrapper around `_validate_txrx_sets`."""
    return _validate_txrx_sets(sets, txrx_dict, tx_or_rx)

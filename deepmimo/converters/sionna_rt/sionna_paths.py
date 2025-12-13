"""Sionna Ray Tracing Paths Module.

This module handles loading and converting path data from Sionna's format to DeepMIMO's format.
"""

from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from deepmimo import consts as c
from deepmimo.converters.converter_utils import compress_path_data
from deepmimo.utils import get_mat_filename, load_pickle, save_mat

# Interaction Type Map for Sionna
INTERACTIONS_MAP = {
    0: c.INTERACTION_LOS,  # LoS
    1: c.INTERACTION_REFLECTION,  # Reflection
    2: c.INTERACTION_DIFFRACTION,  # Diffraction
    3: c.INTERACTION_SCATTERING,  # Diffuse Scattering
    4: None,  # Sionna RIS is not supported yet
}

SIONNA_TYPE_LOS = 0
SIONNA_TYPE_REFLECTION = 1
SIONNA_TYPE_DIFFRACTION = 2
SIONNA_TYPE_SCATTERING = 3
SIONNA_TYPE_RIS = 4


def _is_sionna_v1(sionna_version: str) -> bool:
    """Determine if Sionna version is 1.x or higher."""
    return sionna_version.startswith("1.")


def _preallocate_data(n_rx: int) -> dict:
    """Pre-allocate data for path conversion.

    Args:
        n_rx: Number of RXs

    Returns:
        data: Dictionary containing pre-allocated data

    """
    return {
        c.RX_POS_PARAM_NAME: np.zeros((n_rx, 3), dtype=c.FP_TYPE),
        c.TX_POS_PARAM_NAME: np.zeros((1, 3), dtype=c.FP_TYPE),
        c.AOA_AZ_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.AOA_EL_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.AOD_AZ_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.AOD_EL_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.DELAY_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.POWER_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.PHASE_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.INTERACTIONS_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.INTERACTIONS_POS_PARAM_NAME: np.zeros(
            (n_rx, c.MAX_PATHS, c.MAX_INTER_PER_PATH, 3),
            dtype=c.FP_TYPE,
        )
        * np.nan,
    }


SIONNA_TYPE_LOS = 0
SIONNA_TYPE_REFLECTION = 1
SIONNA_TYPE_DIFFRACTION = 2
SIONNA_TYPE_SCATTERING = 3
SIONNA_TYPE_RIS = 4

MULTI_ANT_NDIM = 3
TWO_D = 2
EXPECTED_TXRX_SETS = 2


def _process_paths_batch(  # noqa: PLR0913, PLR0915
    paths_dict: dict,
    data: dict,
    t: int,
    targets: np.ndarray,
    rx_pos: np.ndarray,
    sionna_version: str,
    tx_ant_idx: int = 0,
    rx_ant_idx: int = 0,
) -> int:
    """Process a batch of paths from Sionna format and store in DeepMIMO format.

    Args:
        paths_dict: Dictionary containing Sionna path data
        data: Dictionary to store processed path data
        t: Transmitter index in current paths dictionary
        targets: Array of target positions
        rx_pos: Array of RX positions
        sionna_version: Sionna version string
        tx_ant_idx: Index of TX antenna element to process
        rx_ant_idx: Index of RX antenna element to process

    Returns:
        int: Number of inactive receivers found in this batch

    """
    inactive_count = 0

    a = paths_dict["a"]
    tau = paths_dict["tau"]
    phi_r = paths_dict["phi_r"]
    phi_t = paths_dict["phi_t"]
    theta_r = paths_dict["theta_r"]
    theta_t = paths_dict["theta_t"]
    vertices = paths_dict["vertices"]

    # Sionna 0.x, uses 'types' & Sionna 1.x, uses 'interactions'
    types = _get_path_key(paths_dict, "types", "interactions")

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
    sionna_v1 = _is_sionna_v1(sionna_version)
    if not sionna_v1:
        b = 0  # batch index (assumed always 0, but can be looped over)
        a = a[b, ..., 0]
        tau = tau[b, ...]
        phi_r = phi_r[b, ...]
        phi_t = phi_t[b, ...]
        theta_r = theta_r[b, ...]
        theta_t = theta_t[b, ...]
        types = types[b, ...]

    # Handle multi-antenna arrays
    tx_idx = t

    if theta_r.ndim > MULTI_ANT_NDIM:  # Multi-antenna case
        # Extract data for specific antenna elements
        a = a[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        tau = tau[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        phi_r = phi_r[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        phi_t = phi_t[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        theta_r = theta_r[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        theta_t = theta_t[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        if not sionna_v1:
            # the vertices are always (max_depth, n_rx, n_tx, max_paths, 3)
            # i.e. no antenna dimensions in vertices in sionna 0.x
            vertices = vertices[:, :, tx_idx, ...]
            types = types[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        else:
            types = types[:, :, rx_ant_idx, :, tx_ant_idx]
            vertices = vertices[:, :, rx_ant_idx, tx_idx, tx_ant_idx, :]

    else:  # Single antenna case
        # For single antenna, we need to extract the correct dimensions
        a = a[:, 0, tx_idx, 0, :]  # Remove antenna dimensions
        tau = tau[:, tx_idx, :]
        phi_r = phi_r[:, tx_idx, :]
        phi_t = phi_t[:, tx_idx, :]
        theta_r = theta_r[:, tx_idx, :]
        theta_t = theta_t[:, tx_idx, :]
        vertices = vertices[:, :, tx_idx, ...]

        if not sionna_v1:
            types = types[:, 0, tx_idx, 0, :]

    n_rx = a.shape[0]
    for rel_rx_idx in range(n_rx):
        abs_idx_arr = np.where(np.all(rx_pos == targets[rel_rx_idx], axis=1))[0]
        if len(abs_idx_arr) == 0:
            # RX position not found in global RX list, skip
            continue
        abs_idx = abs_idx_arr[0]

        # Get amplitude and remove any extra dimensions
        amp = a[rel_rx_idx]

        non_zero_path_idxs = np.where(amp != 0)[0][: c.MAX_PATHS]
        n_paths = len(non_zero_path_idxs)
        if n_paths == 0:
            inactive_count += 1
            continue

        # Ensure that the paths are sorted by amplitude
        sorted_path_idxs = np.argsort(np.abs(amp))[::-1]
        path_idxs = sorted_path_idxs[:n_paths]

        data[c.POWER_PARAM_NAME][abs_idx, :n_paths] = 20 * np.log10(np.abs(amp[path_idxs]))
        data[c.PHASE_PARAM_NAME][abs_idx, :n_paths] = np.angle(amp[path_idxs], deg=True)

        data[c.AOA_AZ_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(phi_r[rel_rx_idx, path_idxs])
        data[c.AOD_AZ_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(phi_t[rel_rx_idx, path_idxs])
        data[c.AOA_EL_PARAM_NAME][abs_idx, :n_paths] = 90.0 - np.rad2deg(
            theta_r[rel_rx_idx, path_idxs]
        )
        data[c.AOD_EL_PARAM_NAME][abs_idx, :n_paths] = 90.0 - np.rad2deg(
            theta_t[rel_rx_idx, path_idxs]
        )

        data[c.DELAY_PARAM_NAME][abs_idx, :n_paths] = tau[rel_rx_idx, path_idxs]

        # Interaction positions and types (vertices is (max_depth, n_rx, max_paths, 3))
        inter_pos_rx = vertices[:, rel_rx_idx, path_idxs, :].swapaxes(0, 1)
        n_interactions = inter_pos_rx.shape[1]
        inter_pos_rx[inter_pos_rx == 0] = np.nan

        # NOTE: this is a workaround to handle no interaction positions
        data[c.INTERACTIONS_POS_PARAM_NAME][abs_idx, :n_paths, :n_interactions, :] = inter_pos_rx
        if sionna_v1:
            # For Sionna v1, types is (max_depth, n_rx, n_tx, max_paths)
            # We need to get (n_paths, max_depth) for the current rx/tx pair
            path_types = types[:, rel_rx_idx, tx_idx, path_idxs].swapaxes(0, 1)
            inter_types = _transform_interaction_types(path_types)
        else:
            inter_types = _get_sionna_interaction_types(types[rel_rx_idx, path_idxs], inter_pos_rx)

        data[c.INTERACTIONS_PARAM_NAME][abs_idx, :n_paths] = inter_types

    return inactive_count


def _get_path_key(
    paths_dict: dict[str, Any], key: str, fallback_key: str | None = None, default: Any = None
) -> Any:
    if key in paths_dict:
        return paths_dict[key]
    if fallback_key and fallback_key in paths_dict:
        return paths_dict[fallback_key]
    if default is not None:
        return default
    msg = f"Neither '{key}' nor '{fallback_key}' found in paths_dict."
    raise KeyError(msg)


def _transform_interaction_types(types: np.ndarray) -> np.ndarray:
    """Transform a (n_paths, max_depth) interaction types array into a (n_paths,) array.

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
            interaction_str = "".join(str(int(x)) for x in valid_interactions if x != 0)
            result[i] = float(interaction_str)

    return result


def _get_sionna_interaction_types(  # noqa: C901, PLR0912
    types: np.ndarray, inter_pos: np.ndarray
) -> np.ndarray:
    """Convert Sionna interaction types to DeepMIMO interaction codes.

    Args:
        types: Array of interaction types from Sionna (N_PATHS,)
        inter_pos: Array of interaction positions (N_PATHS x MAX_INTERACTIONS x 3)

    Returns:
        np.ndarray: Array of DeepMIMO interaction codes (N_PATHS,)

    Note: This function is only used for Sionna 0.x.

    """
    # Ensure types is a numpy array
    types = np.asarray(types)
    original_shape = types.shape

    # Flatten if multidimensional to simplify processing
    if types.ndim > 1:
        types_flat = types.flatten()
        n_paths = len(types_flat)
        # inter_pos is assumed to be (..., max_interactions, 3) matching types structure
        # We flatten the batch dimensions of inter_pos to match types_flat
        # Target shape: (n_paths, max_interactions, 3)
        if inter_pos.ndim >= TWO_D:
            max_interactions = inter_pos.shape[-2]
            inter_pos_flat = inter_pos.reshape(n_paths, max_interactions, 3)
        else:
            # Handle unexpected shape gracefully or let downstream error catch it
            inter_pos_flat = inter_pos

        types = types_flat
        inter_pos = inter_pos_flat
    elif types.ndim == 0:
        types = np.array([types])
        original_shape = (1,)

    # Get number of paths
    n_paths = len(types)
    result = np.zeros(n_paths, dtype=np.float32)

    # For each path
    for path_idx in range(n_paths):
        # Skip if no type (nan or 0)
        current_type = types[path_idx]
        if np.any(np.isnan(current_type)) or np.all(current_type == 0):
            continue

        sionna_type = int(current_type) if np.ndim(current_type) == 0 else int(current_type[0])

        # Handle LoS case (type 0)
        if sionna_type == 0:
            result[path_idx] = c.INTERACTION_LOS
            continue

        # Count number of actual interactions by checking non-nan positions
        if inter_pos.ndim == TWO_D:  # Single path case
            n_interactions = np.nansum(~np.isnan(inter_pos[:, 0]))
        else:  # Multiple paths case
            n_interactions = np.nansum(~np.isnan(inter_pos[path_idx, :, 0]))

        if n_interactions == 0:  # Skip if no interactions
            continue

        # Handle different Sionna interaction types
        if sionna_type == SIONNA_TYPE_REFLECTION:  # Pure reflection path
            # Create string of '1's with length = number of reflections
            code = "1" * n_interactions
            result[path_idx] = np.float32(code)

        elif sionna_type == SIONNA_TYPE_DIFFRACTION:  # Single diffraction path
            # Always just '2' since Sionna only allows single diffraction
            result[path_idx] = c.INTERACTION_DIFFRACTION

        elif sionna_type == SIONNA_TYPE_SCATTERING:  # Scattering path with possible reflections
            # Create string of '1's for reflections + '3' at the end for scattering
            code = "1" * (n_interactions - 1) + "3" if n_interactions > 1 else "3"
            result[path_idx] = np.float32(code)

        elif sionna_type == SIONNA_TYPE_RIS:
            msg = "RIS code not supported yet"
            raise NotImplementedError(msg)
        else:
            msg = f"Unknown Sionna interaction type: {sionna_type}"
            raise ValueError(msg)

    return result.reshape(original_shape)


def read_paths(  # noqa: C901, PLR0912, PLR0915
    load_folder: str, save_folder: str, txrx_dict: dict, sionna_version: str
) -> None:
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

    -- Information about the Sionna paths (from
    https://nvlabs.github.io/sionna/api/rt.html#paths) --

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

    - For multi-antenna arrays, each antenna element is treated as a separate transmitter

    """
    path_dict_list = load_pickle(str(Path(load_folder) / "sionna_paths.pkl"))

    # Collect all unique TX positions from all path dictionaries
    all_tx_pos = np.unique(
        np.vstack(
            [
                _get_path_key(paths_dict, "sources", "src_positions")
                for paths_dict in path_dict_list
            ],
        ),
        axis=0,
    )
    n_tx = len(all_tx_pos)

    # Collect all RX positions while maintaining order and removing duplicates
    all_rx_pos = np.vstack(
        [_get_path_key(paths_dict, "targets", "tgt_positions") for paths_dict in path_dict_list],
    )

    _, unique_indices = np.unique(all_rx_pos, axis=0, return_index=True)
    rx_pos = all_rx_pos[np.sort(unique_indices)]  # Sort indices to maintain original order
    n_rx = len(rx_pos)

    # NOTE: sources and targets have unique positions across antenna elements too.
    # This is why we either support multi-antenna or multi-user/BS.
    n_txrx_sets = len(txrx_dict.keys())
    if n_txrx_sets != EXPECTED_TXRX_SETS:
        msg = "Only one pair of TXRX sets supported for now"
        raise ValueError(msg)

    # Get number of TXs, RXs, and respective antenna elements from txrx_dict
    n_tx_ant = txrx_dict["txrx_set_0"]["num_ant"]
    n_rx_ant = txrx_dict["txrx_set_1"]["num_ant"]
    n_txs = n_tx // n_tx_ant
    n_rxs = n_rx // n_rx_ant

    multi_tx_ant = n_tx_ant > 1
    multi_rx_ant = n_rx_ant > 1
    if multi_tx_ant and n_txs > 1:
        msg = "Multi-antenna & multi-TX not supported yet"
        raise ValueError(msg)
    if multi_rx_ant and n_rxs > 1:
        msg = "Multi-antenna & multi-RX not supported yet"
        raise ValueError(msg)

    # Initialize inactive indices list
    rx_inactive_idxs_count = 0
    bs_bs_paths = False

    # Process each TX position and antenna element combination
    for tx_idx, tx_pos_target in enumerate(all_tx_pos):
        # for printing purposes
        idx_of_tx = 0 if multi_tx_ant else tx_idx
        idx_of_tx_ant = tx_idx if multi_tx_ant else 0

        # Pre-allocate matrices
        data = _preallocate_data(n_rx)

        data[c.RX_POS_PARAM_NAME], data[c.TX_POS_PARAM_NAME] = rx_pos, tx_pos_target

        # Create progress bar
        pbar = tqdm(
            total=n_rx,
            desc=f"Processing receivers for TX {idx_of_tx}, Ant {idx_of_tx_ant}",
        )

        # Process each batch of paths
        for path_dict_idx, paths_dict in enumerate(path_dict_list):
            sources = _get_path_key(paths_dict, "sources", "src_positions")
            tx_idx_in_dict = np.where(np.all(sources == tx_pos_target, axis=1))[0]
            if len(tx_idx_in_dict) == 0:
                continue
            if path_dict_idx == 0:
                targets = _get_path_key(paths_dict, "targets", "tgt_positions")
                if np.array_equal(sources, targets):
                    bs_bs_paths = True
                    continue

            tx_ant_idx = tx_idx_in_dict[0] if multi_tx_ant else 0
            t = 0 if multi_tx_ant else tx_idx_in_dict[0]
            batch_size = targets.shape[0]
            targets = _get_path_key(paths_dict, "targets", "tgt_positions")

            # Process each RX antenna element
            for rx_ant_idx in range(n_rx_ant):
                inactive_count = _process_paths_batch(
                    paths_dict,
                    data,
                    t,
                    targets,
                    rx_pos,
                    sionna_version,
                    tx_ant_idx,
                    rx_ant_idx,
                )

            if tx_idx == 0 and tx_ant_idx == 0:
                rx_inactive_idxs_count += inactive_count
            pbar.update(batch_size)

        pbar.close()

        # Compress data before saving
        data = compress_path_data(data)

        # Save each data key with antenna index in filename
        for key in data:
            idx = tx_ant_idx if multi_tx_ant else tx_idx
            mat_file = get_mat_filename(key, 0, idx, 1)  # tx_set=0, tx_idx=tx_ant_idx, rx_set=1
            save_mat(data[key], key, str(Path(save_folder) / mat_file))

        if bs_bs_paths:
            if multi_tx_ant:
                msg = "Multi-antenna BS-BS paths not supported yet"
                raise NotImplementedError(msg)
                # It would just be necessary to loop over the sources like above

            print(f"BS-BS paths found for TX {tx_idx}, Ant {tx_ant_idx}")

            paths_dict = path_dict_list[0]
            all_bs_pos = _get_path_key(paths_dict, "sources", "src_positions")
            num_bs = len(all_bs_pos)
            data_bs_bs = _preallocate_data(num_bs)
            data_bs_bs[c.RX_POS_PARAM_NAME] = all_bs_pos
            data_bs_bs[c.TX_POS_PARAM_NAME] = tx_pos_target

            # Process BS-BS paths using helper function
            for rx_ant_idx in range(n_rx_ant):
                inactive_count = _process_paths_batch(
                    paths_dict,
                    data_bs_bs,
                    t,
                    all_bs_pos,
                    rx_pos,
                    sionna_version,
                    tx_ant_idx,
                    rx_ant_idx,
                )

            # Compress data before saving
            data_bs_bs = compress_path_data(data_bs_bs)

            # Save each data key
            for key in data_bs_bs:
                mat_file = get_mat_filename(
                    key,
                    0,
                    tx_ant_idx,
                    0,
                )  # tx_set=0, tx_idx=tx_ant_idx, rx_set=0
                save_mat(data_bs_bs[key], key, str(Path(save_folder) / mat_file))

    if bs_bs_paths:
        txrx_dict["txrx_set_0"]["is_rx"] = True  # add BS set also as RX

    # Update txrx_dict with tx and rx numbers
    txrx_dict["txrx_set_0"]["num_points"] = n_tx
    txrx_dict["txrx_set_0"]["num_active_points"] = n_tx

    txrx_dict["txrx_set_1"]["num_points"] = n_rx
    txrx_dict["txrx_set_1"]["num_active_points"] = n_rx - rx_inactive_idxs_count

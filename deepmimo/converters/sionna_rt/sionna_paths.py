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

# Sionna 2.0 interaction bitflags
SIONNA_INTERACTION_SPECULAR = 1
SIONNA_INTERACTION_DIFFUSE = 2
SIONNA_INTERACTION_REFRACTION = 4
SIONNA_INTERACTION_DIFFRACTION = 8

# DeepMIMO interaction type map (used for LoS detection)
SIONNA_TYPE_LOS = 0

MULTI_ANT_NDIM = 3
TWO_D = 2
EXPECTED_TXRX_SETS = 2


def _preallocate_data(n_rx: int) -> dict:
    """Pre-allocate data for path conversion."""
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
    """Transform a (n_paths, max_depth) per-depth interaction array into DeepMIMO codes.

    Each element of the result is an integer formed by concatenating the non-zero
    interaction type digits along the depth dimension.

    Args:
        types: Array of shape (n_paths, max_depth) with per-depth interaction types.
               0 = LoS/padding, 1 = Reflection, 2 = Diffraction, 3 = Scattering.

    Returns:
        np.ndarray: Shape (n_paths,) where each element encodes the interaction sequence.

    Example::

        [[0, 0, 0],   ->  [0,    # LoS
         [1, 1, 0],        11,   # Two reflections
         [1, 3, 0],        13,   # Reflection then scattering
         [2, 0, 0]]         2]   # Single diffraction

    """
    n_paths = types.shape[0]
    result = np.zeros(n_paths, dtype=np.float32)

    for i in range(n_paths):
        path = types[i]
        if np.all(path == 0):
            result[i] = c.INTERACTION_LOS
            continue

        non_zero_mask = path != 0
        if np.any(non_zero_mask):
            non_zero_indices = np.where(non_zero_mask)[0]
            valid_interactions = path[: non_zero_indices[-1] + 1]
            interaction_str = "".join(str(int(x)) for x in valid_interactions if x != 0)
            result[i] = float(interaction_str)

    return result


def _process_paths_batch(  # noqa: PLR0913, PLR0915
    paths_dict: dict,
    data: dict,
    t: int,
    targets: np.ndarray,
    rx_pos: np.ndarray,
    tx_ant_idx: int = 0,
    rx_ant_idx: int = 0,
) -> int:
    """Process a batch of paths from Sionna format and store in DeepMIMO format.

    Args:
        paths_dict: Dictionary containing Sionna path data (Sionna 2.0 format).
        data: Dictionary to store processed path data.
        t: Transmitter index in current paths dictionary.
        targets: Array of target positions.
        rx_pos: Array of RX positions.
        tx_ant_idx: Index of TX antenna element to process.
        rx_ant_idx: Index of RX antenna element to process.

    Returns:
        int: Number of inactive receivers found in this batch.

    Notes:
        Sionna 2.0 array shapes (no batch dimension):
        - a:            [num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        - tau:          [num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or
                        [num_rx, num_tx, max_num_paths]
        - phi_r/phi_t:  same as tau
        - interactions: [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, max_paths]
        - vertices:     [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, max_paths, 3]

    """
    inactive_count = 0

    a = paths_dict["a"]
    tau = paths_dict["tau"]
    phi_r = paths_dict["phi_r"]
    phi_t = paths_dict["phi_t"]
    theta_r = paths_dict["theta_r"]
    theta_t = paths_dict["theta_t"]
    vertices = paths_dict["vertices"]
    types = paths_dict["interactions"]

    tx_idx = t

    if theta_r.ndim > MULTI_ANT_NDIM:  # Multi-antenna case
        a = a[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        tau = tau[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        phi_r = phi_r[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        phi_t = phi_t[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        theta_r = theta_r[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        theta_t = theta_t[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        types = types[:, :, rx_ant_idx, :, tx_ant_idx]
        vertices = vertices[:, :, rx_ant_idx, tx_idx, tx_ant_idx, :]

    else:  # Single antenna case
        a = a[:, 0, tx_idx, 0, :]
        tau = tau[:, tx_idx, :]
        phi_r = phi_r[:, tx_idx, :]
        phi_t = phi_t[:, tx_idx, :]
        theta_r = theta_r[:, tx_idx, :]
        theta_t = theta_t[:, tx_idx, :]
        vertices = vertices[:, :, tx_idx, ...]

    n_rx = a.shape[0]
    for rel_rx_idx in range(n_rx):
        abs_idx_arr = np.where(np.all(rx_pos == targets[rel_rx_idx], axis=1))[0]
        if len(abs_idx_arr) == 0:
            continue
        abs_idx = abs_idx_arr[0]

        amp = a[rel_rx_idx]

        non_zero_path_idxs = np.where(amp != 0)[0][: c.MAX_PATHS]
        n_paths = len(non_zero_path_idxs)
        if n_paths == 0:
            inactive_count += 1
            continue

        sorted_path_idxs = np.argsort(np.abs(amp))[::-1]
        path_idxs = sorted_path_idxs[:n_paths]

        data[c.POWER_PARAM_NAME][abs_idx, :n_paths] = 20 * np.log10(np.abs(amp[path_idxs]))
        data[c.PHASE_PARAM_NAME][abs_idx, :n_paths] = np.angle(amp[path_idxs], deg=True)

        data[c.AOA_AZ_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(phi_r[rel_rx_idx, path_idxs])
        data[c.AOD_AZ_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(phi_t[rel_rx_idx, path_idxs])
        data[c.AOA_EL_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(theta_r[rel_rx_idx, path_idxs])
        data[c.AOD_EL_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(theta_t[rel_rx_idx, path_idxs])

        data[c.DELAY_PARAM_NAME][abs_idx, :n_paths] = tau[rel_rx_idx, path_idxs]

        # vertices: (max_depth, n_rx, max_paths, 3) → (n_paths, max_depth, 3)
        inter_pos_rx = vertices[:, rel_rx_idx, path_idxs, :].swapaxes(0, 1)
        n_interactions = inter_pos_rx.shape[1]
        inter_pos_rx[inter_pos_rx == 0] = np.nan
        data[c.INTERACTIONS_POS_PARAM_NAME][abs_idx, :n_paths, :n_interactions, :] = inter_pos_rx

        # interactions: (max_depth, n_rx, n_tx, max_paths) → (n_paths, max_depth)
        path_types = types[:, rel_rx_idx, tx_idx, path_idxs].swapaxes(0, 1)
        inter_types = _transform_interaction_types(path_types)
        data[c.INTERACTIONS_PARAM_NAME][abs_idx, :n_paths] = inter_types

    return inactive_count


def read_paths(  # noqa: C901, PLR0912, PLR0915
    load_folder: str, save_folder: str, txrx_dict: dict
) -> None:
    """Read and convert path data from Sionna format.

    Args:
        load_folder: Path to folder containing Sionna path files.
        save_folder: Path to save converted path data.
        txrx_dict: Dictionary containing TX/RX set information from read_txrx.

    Notes:
        Expects ``sionna_paths.pkl`` produced by the Sionna 2.0 exporter.
        Each path dictionary can contain one or more transmitters.

    """
    path_dict_list = load_pickle(str(Path(load_folder) / "sionna_paths.pkl"))

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

    all_rx_pos = np.vstack(
        [_get_path_key(paths_dict, "targets", "tgt_positions") for paths_dict in path_dict_list],
    )

    _, unique_indices = np.unique(all_rx_pos, axis=0, return_index=True)
    rx_pos = all_rx_pos[np.sort(unique_indices)]
    n_rx = len(rx_pos)

    n_txrx_sets = len(txrx_dict.keys())
    if n_txrx_sets != EXPECTED_TXRX_SETS:
        msg = "Only one pair of TXRX sets supported for now"
        raise ValueError(msg)

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

    rx_inactive_idxs_count = 0
    bs_bs_paths = False

    for tx_idx, tx_pos_target in enumerate(all_tx_pos):
        idx_of_tx = 0 if multi_tx_ant else tx_idx
        idx_of_tx_ant = tx_idx if multi_tx_ant else 0

        data = _preallocate_data(n_rx)
        data[c.RX_POS_PARAM_NAME], data[c.TX_POS_PARAM_NAME] = rx_pos, tx_pos_target

        pbar = tqdm(
            total=n_rx,
            desc=f"Processing receivers for TX {idx_of_tx}, Ant {idx_of_tx_ant}",
        )

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

            for rx_ant_idx in range(n_rx_ant):
                inactive_count = _process_paths_batch(
                    paths_dict,
                    data,
                    t,
                    targets,
                    rx_pos,
                    tx_ant_idx,
                    rx_ant_idx,
                )

            if tx_idx == 0 and tx_ant_idx == 0:
                rx_inactive_idxs_count += inactive_count
            pbar.update(batch_size)

        pbar.close()

        data = compress_path_data(data)

        for key in data:
            idx = tx_ant_idx if multi_tx_ant else tx_idx
            mat_file = get_mat_filename(key, 0, idx, 1)
            save_mat(data[key], key, str(Path(save_folder) / mat_file))

        if bs_bs_paths:
            if multi_tx_ant:
                msg = "Multi-antenna BS-BS paths not supported yet"
                raise NotImplementedError(msg)

            print(f"BS-BS paths found for TX {tx_idx}, Ant {tx_ant_idx}")

            paths_dict = path_dict_list[0]
            all_bs_pos = _get_path_key(paths_dict, "sources", "src_positions")
            num_bs = len(all_bs_pos)
            data_bs_bs = _preallocate_data(num_bs)
            data_bs_bs[c.RX_POS_PARAM_NAME] = all_bs_pos
            data_bs_bs[c.TX_POS_PARAM_NAME] = tx_pos_target

            for rx_ant_idx in range(n_rx_ant):
                inactive_count = _process_paths_batch(
                    paths_dict,
                    data_bs_bs,
                    t,
                    all_bs_pos,
                    rx_pos,
                    tx_ant_idx,
                    rx_ant_idx,
                )

            data_bs_bs = compress_path_data(data_bs_bs)

            for key in data_bs_bs:
                mat_file = get_mat_filename(key, 0, tx_ant_idx, 0)
                save_mat(data_bs_bs[key], key, str(Path(save_folder) / mat_file))

    if bs_bs_paths:
        txrx_dict["txrx_set_0"]["is_rx"] = True

    txrx_dict["txrx_set_0"]["num_points"] = n_tx
    txrx_dict["txrx_set_0"]["num_active_points"] = n_tx

    txrx_dict["txrx_set_1"]["num_points"] = n_rx
    txrx_dict["txrx_set_1"]["num_active_points"] = n_rx - rx_inactive_idxs_count

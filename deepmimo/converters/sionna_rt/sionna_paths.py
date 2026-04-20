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

# Sionna 2.0 InteractionType enum values (sionna.rt.constants.InteractionType).
# These are NOT the same as DeepMIMO codes — remapping is required.
SIONNA_INTERACTION_NONE = 0  # padding slot / LoS (no bounce at this depth)
SIONNA_INTERACTION_SPECULAR = 1  # specular reflection
SIONNA_INTERACTION_DIFFUSE = 2  # diffuse / lambertian scattering
SIONNA_INTERACTION_REFRACTION = 4  # transmission through a surface
SIONNA_INTERACTION_DIFFRACTION = 8  # edge diffraction (Keller cone)

# DeepMIMO interaction codes differ from Sionna's enum values.
# This table maps each Sionna 2.0 type to the corresponding DeepMIMO code so
# that the per-depth digit-concatenation encoding remains consistent with other
# DeepMIMO ray tracers (e.g. Wireless InSite uses the same DeepMIMO codes).
_SIONNA_TO_DEEPMIMO: dict[int, int] = {
    SIONNA_INTERACTION_SPECULAR: c.INTERACTION_REFLECTION,  # 1 → 1 (unchanged)
    SIONNA_INTERACTION_DIFFUSE: c.INTERACTION_SCATTERING,  # 2 → 3
    SIONNA_INTERACTION_REFRACTION: c.INTERACTION_TRANSMISSION,  # 4 → 4 (unchanged)
    SIONNA_INTERACTION_DIFFRACTION: c.INTERACTION_DIFFRACTION,  # 8 → 2
}

# Dimension thresholds used to distinguish single- vs multi-antenna arrays.
# Sionna 2.0 inserts antenna dims only when num_ant > 1.
MULTI_ANT_NDIM = 3
TWO_D = 2
EXPECTED_TXRX_SETS = 2


def _preallocate_data(n_rx: int) -> dict:
    """Pre-allocate path data arrays for n_rx receivers, filled with NaN.

    NaN (not 0) is the sentinel for absent paths so that downstream code can
    distinguish "no path" from "path with zero delay / power".
    """
    nan_2d = (n_rx, c.MAX_PATHS)
    return {
        c.RX_POS_PARAM_NAME: np.zeros((n_rx, 3), dtype=c.FP_TYPE),
        c.TX_POS_PARAM_NAME: np.zeros((1, 3), dtype=c.FP_TYPE),
        c.AOA_AZ_PARAM_NAME: np.full(nan_2d, np.nan, dtype=c.FP_TYPE),
        c.AOA_EL_PARAM_NAME: np.full(nan_2d, np.nan, dtype=c.FP_TYPE),
        c.AOD_AZ_PARAM_NAME: np.full(nan_2d, np.nan, dtype=c.FP_TYPE),
        c.AOD_EL_PARAM_NAME: np.full(nan_2d, np.nan, dtype=c.FP_TYPE),
        c.DELAY_PARAM_NAME: np.full(nan_2d, np.nan, dtype=c.FP_TYPE),
        c.POWER_PARAM_NAME: np.full(nan_2d, np.nan, dtype=c.FP_TYPE),
        c.PHASE_PARAM_NAME: np.full(nan_2d, np.nan, dtype=c.FP_TYPE),
        c.INTERACTIONS_PARAM_NAME: np.full(nan_2d, np.nan, dtype=c.FP_TYPE),
        c.INTERACTIONS_POS_PARAM_NAME: np.full(
            (n_rx, c.MAX_PATHS, c.MAX_INTER_PER_PATH, 3), np.nan, dtype=c.FP_TYPE
        ),
    }


def _get_path_key(
    paths_dict: dict[str, Any], key: str, fallback_key: str | None = None, default: Any = None
) -> Any:
    """Fetch a value from paths_dict with an optional legacy-key fallback.

    Needed because exported field names changed between Sionna versions (e.g.
    'sources' vs 'src_positions').  Raise KeyError only when both keys are
    absent and no default was provided.
    """
    if key in paths_dict:
        return paths_dict[key]
    if fallback_key and fallback_key in paths_dict:
        return paths_dict[fallback_key]
    if default is not None:
        return default
    msg = f"Neither '{key}' nor '{fallback_key}' found in paths_dict."
    raise KeyError(msg)


def transform_interaction_types(types: np.ndarray) -> np.ndarray:
    """Transform per-depth Sionna 2.0 interaction flags into DeepMIMO path codes.

    DeepMIMO encodes the interaction sequence for a path as a single float whose
    decimal digits are the per-bounce DeepMIMO codes in order, e.g.:
        LoS              → 0
        one reflection   → 1
        two reflections  → 11
        refl + scatter   → 13
        diffraction      → 2

    Sionna 2.0 uses different numeric values for its InteractionType enum
    (SPECULAR=1, DIFFUSE=2, REFRACTION=4, DIFFRACTION=8). These are remapped
    through ``_SIONNA_TO_DEEPMIMO`` before concatenation.

    Args:
        types: Array of shape (n_paths, max_depth) with Sionna 2.0
               InteractionType values per depth slot.

    Returns:
        np.ndarray: Shape (n_paths,) where each element is the DeepMIMO path
        interaction code.

    Example::

        Input (Sionna values):          Output (DeepMIMO codes):
        [[0, 0, 0],    # LoS         →  [0,
         [1, 1, 0],    # 2x specular →   11,
         [1, 2, 0],    # refl+diff   →   13,   (DIFFUSE 2 → code 3)
         [8, 0, 0]]    # diffraction →    2]   (DIFFRACTION 8 → code 2)

    """
    n_paths = types.shape[0]
    result = np.zeros(n_paths, dtype=np.float32)

    for i in range(n_paths):
        path = types[i]

        # All-zero depth slots → LoS (no bounces)
        if np.all(path == 0):
            result[i] = c.INTERACTION_LOS
            continue

        # Find the last non-zero depth to avoid trailing padding zeros
        non_zero_indices = np.where(path != 0)[0]
        valid_raw = path[: non_zero_indices[-1] + 1]

        # Remap Sionna enum values → DeepMIMO codes, then concatenate as digits
        remapped = [_SIONNA_TO_DEEPMIMO.get(int(x), int(x)) for x in valid_raw if x != 0]
        result[i] = float("".join(str(v) for v in remapped))

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
    """Process one Sionna batch and write path data into the DeepMIMO data dict.

    Args:
        paths_dict: Exported Sionna path dictionary (Sionna 2.0 format).
        data: Pre-allocated DeepMIMO data dict (from ``_preallocate_data``).
        t: TX index within this paths_dict (column in the TX dimension).
        targets: RX positions for this batch, shape (n_batch, 3).
        rx_pos: All RX positions in the scenario, shape (n_rx, 3).
            Used to map batch-relative indices to global indices.
        tx_ant_idx: TX antenna element index (multi-antenna case only).
        rx_ant_idx: RX antenna element index (multi-antenna case only).

    Returns:
        int: Number of receivers in this batch with zero active paths.

    Notes:
        Sionna 2.0 array layouts (no batch dim compared with 0.x):

        Single-antenna (common case):
          a:            (num_rx, 1, num_tx, 1, max_paths)
          tau/angles:   (num_rx, num_tx, max_paths)
          interactions: (max_depth, num_rx, num_tx, max_paths)
          vertices:     (max_depth, num_rx, num_tx, max_paths, 3)

        Multi-antenna:
          a:            (num_rx, num_rx_ant, num_tx, num_tx_ant, max_paths)
          tau/angles:   same but with antenna dims inserted
          interactions: (max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, max_paths)
          vertices:     (max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, max_paths, 3)

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

    # Slice to the requested TX/RX antenna element.
    # theta_r.ndim == MULTI_ANT_NDIM+1 (4-D) when antenna dims are present.
    if theta_r.ndim > MULTI_ANT_NDIM:
        # Multi-antenna: antenna dims at positions 1 (rx_ant) and 3 (tx_ant)
        a = a[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        tau = tau[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        phi_r = phi_r[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        phi_t = phi_t[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        theta_r = theta_r[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        theta_t = theta_t[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        types = types[:, :, rx_ant_idx, :, tx_ant_idx]
        vertices = vertices[:, :, rx_ant_idx, tx_idx, tx_ant_idx, :]
    else:
        # Single-antenna: squeeze the singleton antenna dims (index 0)
        a = a[:, 0, tx_idx, 0, :]
        tau = tau[:, tx_idx, :]
        phi_r = phi_r[:, tx_idx, :]
        phi_t = phi_t[:, tx_idx, :]
        theta_r = theta_r[:, tx_idx, :]
        theta_t = theta_t[:, tx_idx, :]
        vertices = vertices[:, :, tx_idx, ...]
        # types stays (max_depth, num_rx, num_tx, max_paths) — tx dim resolved later

    n_rx = a.shape[0]
    for rel_rx_idx in range(n_rx):
        # Map batch-local RX index to global RX position index
        abs_idx_arr = np.where(np.all(rx_pos == targets[rel_rx_idx], axis=1))[0]
        if len(abs_idx_arr) == 0:
            continue  # target not in the global rx_pos grid (can happen with floating point)
        abs_idx = abs_idx_arr[0]

        amp = a[rel_rx_idx]

        # Keep only paths with non-zero amplitude, capped at MAX_PATHS
        non_zero_path_idxs = np.where(amp != 0)[0][: c.MAX_PATHS]
        n_paths = len(non_zero_path_idxs)
        if n_paths == 0:
            inactive_count += 1
            continue

        # Sort retained paths by descending power so the strongest path is index 0
        sorted_path_idxs = np.argsort(np.abs(amp[non_zero_path_idxs]))[::-1]
        path_idxs = non_zero_path_idxs[sorted_path_idxs]

        # Power in dB, phase in degrees
        data[c.POWER_PARAM_NAME][abs_idx, :n_paths] = 20 * np.log10(np.abs(amp[path_idxs]))
        data[c.PHASE_PARAM_NAME][abs_idx, :n_paths] = np.angle(amp[path_idxs], deg=True)

        # Angles of arrival / departure in degrees
        data[c.AOA_AZ_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(phi_r[rel_rx_idx, path_idxs])
        data[c.AOD_AZ_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(phi_t[rel_rx_idx, path_idxs])
        data[c.AOA_EL_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(theta_r[rel_rx_idx, path_idxs])
        data[c.AOD_EL_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(theta_t[rel_rx_idx, path_idxs])

        data[c.DELAY_PARAM_NAME][abs_idx, :n_paths] = tau[rel_rx_idx, path_idxs]

        # Interaction type codes: (max_depth, n_rx, n_tx, max_paths) → (n_paths, max_depth)
        path_types = types[:, rel_rx_idx, tx_idx, path_idxs].swapaxes(0, 1)

        # Bounce positions: (max_depth, n_rx, max_paths, 3) → (n_paths, max_depth, 3)
        inter_pos_rx = vertices[:, rel_rx_idx, path_idxs, :].swapaxes(0, 1)
        n_interactions = inter_pos_rx.shape[1]
        # Depth slots with NONE type (0) are empty padding — mark them NaN.
        # Using the type array avoids falsely nulling valid positions that have a
        # coordinate of exactly 0 (e.g. a building face at x=0).
        inter_pos_rx[path_types == SIONNA_INTERACTION_NONE] = np.nan
        data[c.INTERACTIONS_POS_PARAM_NAME][abs_idx, :n_paths, :n_interactions, :] = inter_pos_rx

        data[c.INTERACTIONS_PARAM_NAME][abs_idx, :n_paths] = transform_interaction_types(path_types)

    return inactive_count


def read_paths(  # noqa: C901, PLR0912, PLR0915
    load_folder: str, save_folder: str, txrx_dict: dict
) -> None:
    """Read and convert path data from Sionna format to DeepMIMO .mat files.

    Args:
        load_folder: Directory containing ``sionna_paths.pkl`` (from exporter).
        save_folder: Directory where DeepMIMO ``.mat`` files will be written.
        txrx_dict: TX/RX set info dict returned by ``read_txrx``.

    Notes:
        Expects the 2.0 exporter format: a list of per-batch path dicts, one per
        ``PathSolver`` call.  Each dict may contain multiple TX positions as
        columns in the TX dimension.

    """
    path_dict_list = load_pickle(str(Path(load_folder) / "sionna_paths.pkl"))

    # Collect all TX positions seen across batches (rows in each 'sources' array)
    all_tx_pos = np.unique(
        np.vstack(
            [_get_path_key(paths_dict, "sources", "src_positions") for paths_dict in path_dict_list]
        ),
        axis=0,
    )
    n_tx = len(all_tx_pos)

    # Stack all target positions from every batch to reconstruct the full RX grid
    all_rx_pos = np.vstack(
        [_get_path_key(paths_dict, "targets", "tgt_positions") for paths_dict in path_dict_list]
    )
    # Deduplicate while preserving original order (np.unique reorders; undo that)
    _, unique_indices = np.unique(all_rx_pos, axis=0, return_index=True)
    rx_pos = all_rx_pos[np.sort(unique_indices)]
    n_rx = len(rx_pos)

    n_txrx_sets = len(txrx_dict.keys())
    if n_txrx_sets != EXPECTED_TXRX_SETS:
        msg = "Only one pair of TXRX sets supported for now"
        raise ValueError(msg)

    n_tx_ant = txrx_dict["txrx_set_0"]["num_ant"]
    n_rx_ant = txrx_dict["txrx_set_1"]["num_ant"]
    # When multi-antenna, all antenna positions appear as separate TX/RX entries;
    # divide to get the number of physical device locations.
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
    bs_bs_paths = False  # set to True if the first batch contains BS-BS paths

    for tx_idx, tx_pos_target in enumerate(all_tx_pos):
        # For multi-antenna TX, tx_idx encodes the antenna element; for single
        # antenna, it encodes the physical TX location.
        idx_of_tx = 0 if multi_tx_ant else tx_idx
        idx_of_tx_ant = tx_idx if multi_tx_ant else 0
        # Default fallback; overwritten in the inner loop for multi-antenna TX.
        tx_ant_idx = idx_of_tx_ant

        data = _preallocate_data(n_rx)
        data[c.RX_POS_PARAM_NAME] = rx_pos
        data[c.TX_POS_PARAM_NAME] = tx_pos_target[np.newaxis]  # keep (1, 3) shape

        pbar = tqdm(
            total=n_rx,
            desc=f"Processing receivers for TX {idx_of_tx}, Ant {idx_of_tx_ant}",
        )

        for path_dict_idx, paths_dict in enumerate(path_dict_list):
            sources = _get_path_key(paths_dict, "sources", "src_positions")

            # Skip batches that don't include this TX position
            tx_idx_in_dict = np.where(np.all(sources == tx_pos_target, axis=1))[0]
            if len(tx_idx_in_dict) == 0:
                continue

            # The first batch may be a BS-BS measurement (sources == targets)
            if path_dict_idx == 0:
                targets = _get_path_key(paths_dict, "targets", "tgt_positions")
                if np.array_equal(sources, targets):
                    bs_bs_paths = True
                    continue

            tx_ant_idx = tx_idx_in_dict[0] if multi_tx_ant else 0
            t = 0 if multi_tx_ant else tx_idx_in_dict[0]
            targets = _get_path_key(paths_dict, "targets", "tgt_positions")
            batch_size = targets.shape[0]

            for rx_ant_idx in range(n_rx_ant):
                inactive_count = _process_paths_batch(
                    paths_dict, data, t, targets, rx_pos, tx_ant_idx, rx_ant_idx
                )

            if tx_idx == 0 and tx_ant_idx == 0:
                rx_inactive_idxs_count += inactive_count
            pbar.update(batch_size)

        pbar.close()

        data = compress_path_data(data)

        # Save one .mat file per channel parameter per TX
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
            data_bs_bs[c.TX_POS_PARAM_NAME] = tx_pos_target[np.newaxis]

            for rx_ant_idx in range(n_rx_ant):
                inactive_count = _process_paths_batch(
                    paths_dict, data_bs_bs, t, all_bs_pos, rx_pos, tx_ant_idx, rx_ant_idx
                )

            data_bs_bs = compress_path_data(data_bs_bs)

            for key in data_bs_bs:
                mat_file = get_mat_filename(key, 0, tx_ant_idx, 0)
                save_mat(data_bs_bs[key], key, str(Path(save_folder) / mat_file))

    if bs_bs_paths:
        # Mark TX set as also acting as RX so the converter treats it correctly
        txrx_dict["txrx_set_0"]["is_rx"] = True

    txrx_dict["txrx_set_0"]["num_points"] = n_tx
    txrx_dict["txrx_set_0"]["num_active_points"] = n_tx

    txrx_dict["txrx_set_1"]["num_points"] = n_rx
    txrx_dict["txrx_set_1"]["num_active_points"] = n_rx - rx_inactive_idxs_count

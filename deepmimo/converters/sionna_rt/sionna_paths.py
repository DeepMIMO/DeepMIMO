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
    if types.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    types_int = types.astype(np.int64)

    # Lookup table mirroring ``_SIONNA_TO_DEEPMIMO.get(x, x)``: identity passthrough
    # with the Sionna→DeepMIMO remappings overlaid (2→3 scattering, 8→2 diffraction).
    lut_size = max(int(types_int.max(initial=0)) + 1, max(_SIONNA_TO_DEEPMIMO) + 1)
    lut = np.arange(lut_size, dtype=np.int64)
    for sionna_val, dm_code in _SIONNA_TO_DEEPMIMO.items():
        lut[sionna_val] = dm_code
    digits = lut[types_int]  # DeepMIMO digit per depth slot (0 where NONE)

    # Concatenate the non-zero digits, in depth order, into a decimal code.
    # NONE(0) slots are skipped (including padding zeros between bounces), so the
    # j-th kept slot contributes ``digit * 10 ** (n_kept - rank)`` where ``rank``
    # is its 1-based position among kept slots.  All-zero rows → 0 (LoS).
    nonzero = types_int != 0
    n_kept = nonzero.sum(axis=1, keepdims=True)
    rank = np.cumsum(nonzero, axis=1)
    weights = np.where(nonzero, 10.0 ** (n_kept - rank), 0.0)
    # float64 sum is exact for the digit counts produced by Sionna (max_depth
    # slots ⇒ ≤ max_depth digits); cast to float32 like the original.
    return (digits * weights).sum(axis=1).astype(np.float32)


def _build_rx_pos_index(rx_pos: np.ndarray) -> dict[bytes, int]:
    """Map each ``rx_pos`` row's raw bytes to its global index.

    Replaces the original per-receiver ``np.where(np.all(rx_pos == target))``
    scan (which made the whole batch loop O(n_rx**2)) with an O(1) dict lookup.
    Using ``row.tobytes()`` preserves the exact float-equality and first-match
    semantics of the original scan; ``rx_pos`` rows are unique in practice
    (``read_paths`` dedups them), so first-match never actually differs.
    """
    index: dict[bytes, int] = {}
    for i in range(len(rx_pos)):
        index.setdefault(rx_pos[i].tobytes(), i)
    return index


def _process_paths_batch(  # noqa: PLR0913, PLR0915
    paths_dict: dict,
    data: dict,
    t: int,
    targets: np.ndarray,
    rx_pos_index: dict[bytes, int],
    tx_ant_idx: int = 0,
    rx_ant_idx: int = 0,
) -> int:
    """Process one Sionna batch and write path data into the DeepMIMO data dict.

    Fully vectorized over the batch's receivers: every receiver's active-path
    selection, descending-power sort, scalar conversions and interaction
    encoding are computed in a handful of array ops instead of a per-receiver
    Python loop.  The output is numerically identical to the original loop (and
    byte-identical when per-receiver path magnitudes are distinct, which they
    always are for physical ray-tracing data).

    Args:
        paths_dict: Exported Sionna path dictionary (Sionna 2.0 format).
        data: Pre-allocated DeepMIMO data dict (from ``_preallocate_data``).
        t: TX index within this paths_dict (column in the TX dimension).
        targets: RX positions for this batch, shape (n_batch, 3).
        rx_pos_index: Map from global ``rx_pos`` row bytes to global index, from
            ``_build_rx_pos_index``.  Used to map batch-local indices to global.
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

    n_batch, n_sionna_paths = a.shape

    # Map every batch-local target to its global index in a single pass. Targets
    # missing from the global grid get -1 and are skipped (matches the original).
    abs_idx = np.fromiter(
        (rx_pos_index.get(targets[i].tobytes(), -1) for i in range(n_batch)),
        dtype=np.int64,
        count=n_batch,
    )
    found = abs_idx >= 0

    # Select active paths: non-zero amplitude, keeping the first MAX_PATHS (by
    # Sionna index) then sorting those by descending magnitude. ``cumsum`` of the
    # non-zero mask reproduces the original ``np.where(amp != 0)[0][:MAX_PATHS]``.
    mag = np.abs(a)
    nonzero = a != 0
    keep = nonzero & (np.cumsum(nonzero, axis=1) <= c.MAX_PATHS)
    n_paths = keep.sum(axis=1)

    # Reversing a stable ascending sort places kept paths first in descending
    # magnitude order while reproducing the original ``np.argsort(...)[::-1]``
    # tie ordering: numpy's default sort is insertion sort (stable) for the
    # per-receiver path counts seen here, so equal-magnitude paths keep the exact
    # order the loop produced. Non-kept slots (key -inf) sort to the very end.
    sort_key = np.where(keep, mag, -np.inf)
    order = np.argsort(sort_key, axis=1, kind="stable")[:, ::-1]

    n_take = min(n_sionna_paths, c.MAX_PATHS)
    cols = order[:, :n_take]  # (n_batch, n_take) Sionna path indices, strongest-first
    valid = np.arange(n_take)[None, :] < n_paths[:, None]
    invalid = ~valid

    # Gather the per-path quantities for the whole batch at once.
    amp_s = np.take_along_axis(a, cols, axis=1)
    with np.errstate(divide="ignore"):
        power = 20 * np.log10(np.abs(amp_s))  # -inf only at padded slots (NaN'd below)
    phase = np.angle(amp_s, deg=True)
    aoa_az = np.rad2deg(np.take_along_axis(phi_r, cols, axis=1))
    aod_az = np.rad2deg(np.take_along_axis(phi_t, cols, axis=1))
    aoa_el = np.rad2deg(np.take_along_axis(theta_r, cols, axis=1))
    aod_el = np.rad2deg(np.take_along_axis(theta_t, cols, axis=1))
    delay = np.take_along_axis(tau, cols, axis=1)

    # types: (max_depth, n_batch, n_tx, n_sionna_paths) → (n_batch, n_take, max_depth)
    types_tx = np.moveaxis(types[:, :, tx_idx, :], 0, 2)
    path_types = np.take_along_axis(types_tx, cols[:, :, None], axis=1)

    # vertices: (max_depth, n_batch, n_sionna_paths, 3) → (n_batch, n_take, max_depth, 3)
    vert = np.moveaxis(vertices, 0, 2)
    inter_pos = np.take_along_axis(vert, cols[:, :, None, None], axis=1).copy()
    max_depth = inter_pos.shape[2]

    codes = transform_interaction_types(path_types.reshape(-1, max_depth)).reshape(n_batch, n_take)

    # NONE(0) depth slots are empty padding — mark them NaN. Using the type array
    # avoids falsely nulling valid positions at a coordinate of exactly 0.
    inter_pos[path_types == SIONNA_INTERACTION_NONE] = np.nan

    # Blank out padded path slots (rank >= n_paths) so they stay NaN, exactly as
    # the pre-allocated arrays would after the original per-receiver writes.
    power[invalid] = np.nan
    phase[invalid] = np.nan
    aoa_az[invalid] = np.nan
    aod_az[invalid] = np.nan
    aoa_el[invalid] = np.nan
    aod_el[invalid] = np.nan
    delay[invalid] = np.nan
    codes[invalid] = np.nan
    inter_pos[invalid] = np.nan

    # Scatter into the global data arrays (skip targets absent from rx_pos).
    rows = abs_idx[found]
    cols_slice = slice(None, n_take)
    data[c.POWER_PARAM_NAME][rows, cols_slice] = power[found]
    data[c.PHASE_PARAM_NAME][rows, cols_slice] = phase[found]
    data[c.AOA_AZ_PARAM_NAME][rows, cols_slice] = aoa_az[found]
    data[c.AOD_AZ_PARAM_NAME][rows, cols_slice] = aod_az[found]
    data[c.AOA_EL_PARAM_NAME][rows, cols_slice] = aoa_el[found]
    data[c.AOD_EL_PARAM_NAME][rows, cols_slice] = aod_el[found]
    data[c.DELAY_PARAM_NAME][rows, cols_slice] = delay[found]
    data[c.INTERACTIONS_PARAM_NAME][rows, cols_slice] = codes[found]
    data[c.INTERACTIONS_POS_PARAM_NAME][rows, :n_take, :max_depth, :] = inter_pos[found]

    return int(np.sum(n_paths[found] == 0))


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

    # Build the global position→index map once (O(n_rx)); _process_paths_batch
    # uses it for O(1) per-target lookups instead of an O(n_rx) scan per receiver.
    rx_pos_index = _build_rx_pos_index(rx_pos)

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
                    paths_dict, data, t, targets, rx_pos_index, tx_ant_idx, rx_ant_idx
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
                    paths_dict, data_bs_bs, t, all_bs_pos, rx_pos_index, tx_ant_idx, rx_ant_idx
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

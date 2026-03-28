"""Helpers for explicit merged-grid indexing and v3 compatibility."""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

from deepmimo.datasets.dataset import Dataset
from deepmimo.datasets.sampling import get_grid_idxs


class MergedGridDataset(Dataset):
    """Dataset wrapper that resolves global row/col indexing across merged RX grids."""

    def __init__(self, data: dict[str, Any] | None = None, *, compat_spec: dict[str, Any]) -> None:
        """Initialize a merged dataset with precomputed global grid metadata.

        Args:
            data: Dataset payload dictionary.
            compat_spec: Global row/col metadata used to resolve indices.

        """
        super().__init__(data or {})
        object.__setattr__(self, "_compat_spec", compat_spec)

    def _resolve_global_grid_idxs(
        self,
        axis: str,
        idxs: int | list[int] | np.ndarray,
    ) -> np.ndarray:
        idxs_arr = np.asarray([idxs] if isinstance(idxs, int) else idxs, dtype=int).ravel()
        if idxs_arr.size == 0:
            return np.array([], dtype=int)

        if axis == "row":
            grid_offsets = np.asarray(self._compat_spec["row_offsets"], dtype=int)
            grid_axes = self._compat_spec["row_axes"]
        elif axis == "col":
            grid_offsets = np.asarray(self._compat_spec["col_offsets"], dtype=int)
            grid_axes = self._compat_spec["col_axes"]
        else:
            msg = f"Invalid axis '{axis}', must be 'row' or 'col'"
            raise ValueError(msg)

        if np.any(idxs_arr < 0) or np.any(idxs_arr >= grid_offsets[-1]):
            msg = (
                f"{axis}_idxs must be in range [0, {grid_offsets[-1]}), "
                f"but got min={idxs_arr.min()}, max={idxs_arr.max()}"
            )
            raise IndexError(msg)

        ue_offsets = np.asarray(self._compat_spec["ue_offsets"], dtype=int)
        grid_sizes = [np.asarray(g, dtype=int) for g in self._compat_spec["grid_sizes"]]

        # Resolve each requested row/col in caller-provided order.
        # This mirrors get_grid_idxs behavior for single-grid datasets.
        grid_idxs = np.searchsorted(grid_offsets[1:], idxs_arr, side="right")
        all_ue_idxs = []
        for idx, grid_idx in zip(idxs_arr, grid_idxs, strict=False):
            local_idx = int(idx - grid_offsets[grid_idx])
            grid_axis = grid_axes[grid_idx]
            local_ue_idxs = get_grid_idxs(grid_sizes[grid_idx], grid_axis, np.array([local_idx]))
            all_ue_idxs.append(local_ue_idxs + ue_offsets[grid_idx])

        if not all_ue_idxs:
            return np.array([], dtype=int)
        return np.concatenate(all_ue_idxs).astype(int)

    def _get_row_idxs(self, row_idxs: int | list[int] | np.ndarray) -> np.ndarray:
        """Return indices of users in global merged rows."""
        return self._resolve_global_grid_idxs("row", row_idxs)

    def _get_col_idxs(self, col_idxs: int | list[int] | np.ndarray) -> np.ndarray:
        """Return indices of users in global merged columns."""
        return self._resolve_global_grid_idxs("col", col_idxs)


# Backward-compatible alias for the initial implementation name.
V3CompatDataset = MergedGridDataset

_MERGE_EXCLUDED_KEYS = {"n_ue", "grid_size", "grid_spacing"}


def _pad_concat_users(arrays: list[np.ndarray]) -> np.ndarray:
    """Pad per-user arrays to common non-user dimensions, then concatenate on axis 0."""
    ndim = arrays[0].ndim
    target_shape = [max(arr.shape[d] for arr in arrays) for d in range(1, ndim)]
    padded_arrays = []
    for arr in arrays:
        arr_to_pad = arr
        if np.issubdtype(arr_to_pad.dtype, np.integer) or np.issubdtype(arr_to_pad.dtype, np.bool_):
            arr_to_pad = arr_to_pad.astype(np.float32)

        pad_width = [(0, 0)] + [
            (0, target_shape[d - 1] - arr_to_pad.shape[d]) for d in range(1, ndim)
        ]
        if np.issubdtype(arr_to_pad.dtype, np.complexfloating):
            pad_value = np.nan + 0j
        elif np.issubdtype(arr_to_pad.dtype, np.floating):
            pad_value = np.nan
        else:
            pad_value = 0
        padded_arrays.append(
            np.pad(arr_to_pad, pad_width, mode="constant", constant_values=pad_value)
        )
    return np.concatenate(padded_arrays, axis=0)


def _rx_rank_map(txrx_sets: list[Any] | dict[str, Any]) -> dict[int, int]:
    """Map RX set IDs to their scenario order rank using numeric sorting."""
    values = txrx_sets.values() if isinstance(txrx_sets, dict) else txrx_sets
    rx_set_ids = []
    for txrx_set in values:
        is_rx = txrx_set["is_rx"] if isinstance(txrx_set, dict) else txrx_set.is_rx
        if not is_rx:
            continue
        rx_set_id = txrx_set["id"] if isinstance(txrx_set, dict) else txrx_set.id
        rx_set_ids.append(int(rx_set_id))

    rx_set_ids.sort()
    return {rx_set_id: rank for rank, rx_set_id in enumerate(rx_set_ids)}


def _resolve_rx_rank_map(
    datasets: list[Dataset],
    *,
    rx_rank_map: dict[int, int] | None = None,
    txrx_sets: list[Any] | dict[str, Any] | None = None,
) -> dict[int, int]:
    """Resolve RX rank metadata from explicit input, scenario metadata, or dataset contents."""
    if rx_rank_map is not None:
        return rx_rank_map
    if txrx_sets is not None:
        return _rx_rank_map(txrx_sets)
    if datasets:
        with contextlib.suppress(AttributeError, KeyError, OSError, ValueError):
            return _rx_rank_map(datasets[0].txrx_sets)

    rx_set_ids = sorted(
        {
            int(dataset.get("txrx", {}).get("rx_set_id", -1))
            for dataset in datasets
            if dataset.hasattr("txrx")
        }
    )
    return {rx_set_id: rank for rank, rx_set_id in enumerate(rx_set_ids)}


def _compat_grid_spec(
    datasets: list[Dataset],
    *,
    indexing: str,
    rx_rank_map: dict[int, int],
) -> dict[str, Any]:
    """Build global row/col indexing metadata for merged multi-grid datasets."""
    grid_sizes = [np.asarray(ds.grid_size, dtype=int) for ds in datasets]
    ue_offsets = np.cumsum([0, *[int(ds.n_ue) for ds in datasets[:-1]]], dtype=int)
    rx_set_ids = [int(ds.get("txrx", {}).get("rx_set_id", -1)) for ds in datasets]

    if indexing == "native":
        row_axes = ["row"] * len(datasets)
        col_axes = ["col"] * len(datasets)
    elif indexing == "v3":
        row_axes = [
            "row" if rx_rank_map.get(rx_set_id, 0) == 0 else "col" for rx_set_id in rx_set_ids
        ]
        col_axes = [
            "col" if rx_rank_map.get(rx_set_id, 0) == 0 else "row" for rx_set_id in rx_set_ids
        ]
    else:
        msg = f"Unknown indexing mode '{indexing}'. Expected 'native' or 'v3'."
        raise ValueError(msg)

    n_rows_per_grid = [
        int(grid_size[1]) if row_axis == "row" else int(grid_size[0])
        for grid_size, row_axis in zip(grid_sizes, row_axes, strict=False)
    ]
    n_cols_per_grid = [
        int(grid_size[0]) if col_axis == "col" else int(grid_size[1])
        for grid_size, col_axis in zip(grid_sizes, col_axes, strict=False)
    ]
    row_offsets = np.cumsum([0, *n_rows_per_grid], dtype=int)
    col_offsets = np.cumsum([0, *n_cols_per_grid], dtype=int)

    return {
        "ue_offsets": ue_offsets,
        "grid_sizes": grid_sizes,
        "row_axes": row_axes,
        "col_axes": col_axes,
        "row_offsets": row_offsets,
        "col_offsets": col_offsets,
    }


def merge_datasets(  # noqa: C901, PLR0912
    datasets: list[Dataset],
    *,
    indexing: str = "native",
    rx_rank_map: dict[int, int] | None = None,
    txrx_sets: list[Any] | dict[str, Any] | None = None,
) -> Dataset:
    """Merge datasets that share one transmitter into an explicit merged-grid dataset."""
    if not datasets:
        msg = "Cannot merge an empty dataset list"
        raise ValueError(msg)

    tx_keys = {
        (
            int(dataset.get("txrx", {}).get("tx_set_id", -1)),
            int(dataset.get("txrx", {}).get("tx_idx", -1)),
        )
        for dataset in datasets
    }
    rx_set_ids = {int(dataset.get("txrx", {}).get("rx_set_id", -1)) for dataset in datasets}
    if len(tx_keys) != 1:
        if len(rx_set_ids) == 1:
            msg = (
                "Merging datasets across multiple transmitters is not supported yet because "
                "Dataset operations assume a single transmitter view."
            )
            raise NotImplementedError(msg)
        msg = "Selected datasets must share the same transmitter or the same receiver grid"
        raise ValueError(msg)

    if len(datasets) == 1 and indexing == "native":
        return datasets[0]

    resolved_rx_rank_map = _resolve_rx_rank_map(
        datasets,
        rx_rank_map=rx_rank_map,
        txrx_sets=txrx_sets,
    )

    merged_data: dict[str, Any] = {}
    keys: list[str] = []
    seen: set[str] = set()
    for dataset in datasets:
        for key in dataset:
            if key in _MERGE_EXCLUDED_KEYS:
                continue
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)

    for key in keys:
        present_values = [dataset[key] for dataset in datasets if dataset.hasattr(key)]
        present_datasets = [dataset for dataset in datasets if dataset.hasattr(key)]
        if not present_values:
            continue
        first_value = present_values[0]

        if len(present_values) != len(datasets):
            # Key is asymmetric across RX grids; keep first available value safely.
            merged_data[key] = first_value
            continue

        is_per_user_array = (
            isinstance(first_value, np.ndarray)
            and first_value.ndim > 0
            and all(
                isinstance(value, np.ndarray) and value.ndim == first_value.ndim
                for value in present_values
            )
            and all(
                value.shape[0] == dataset.n_ue
                for value, dataset in zip(present_values, present_datasets, strict=False)
            )
        )

        if is_per_user_array:
            same_tail_shapes = all(
                value.shape[1:] == present_values[0].shape[1:] for value in present_values
            )
            if same_tail_shapes:
                merged_data[key] = np.concatenate(present_values, axis=0)
            else:
                merged_data[key] = _pad_concat_users(present_values)
        else:
            merged_data[key] = first_value

    merged_data["txrx_parts"] = [
        dict(dataset.txrx) for dataset in datasets if dataset.hasattr("txrx")
    ]
    if datasets[0].hasattr("txrx"):
        merged_data["txrx"] = dict(datasets[0].txrx)

    compat_spec = _compat_grid_spec(
        datasets,
        indexing=indexing,
        rx_rank_map=resolved_rx_rank_map,
    )
    return MergedGridDataset(merged_data, compat_spec=compat_spec)

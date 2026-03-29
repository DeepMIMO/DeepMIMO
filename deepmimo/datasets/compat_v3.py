"""Backward-compatibility helpers for v3-style merged-grid loading."""

from __future__ import annotations

from typing import Any

from deepmimo.datasets.dataset import Dataset, MacroDataset, merge_datasets


def _rx_rank_map(txrx_dict: dict[str, Any]) -> dict[int, int]:
    """Map RX set IDs to their scenario order rank using numeric sorting."""
    rx_sets = [txrx_dict[key] for key in sorted(txrx_dict.keys()) if txrx_dict[key]["is_rx"]]
    rx_set_ids = sorted(int(rx_set["id"]) for rx_set in rx_sets)
    return {rx_set_id: rank for rank, rx_set_id in enumerate(rx_set_ids)}


def _v3_grid_axes(
    datasets: list[Dataset],
    rx_rank_map: dict[int, int],
) -> tuple[list[str], list[str]]:
    """Return v3 row/col axis overrides for the provided RX-grid datasets."""
    if not rx_rank_map:
        msg = (
            "V3 compatibility requires RX rank metadata. Load the dataset through "
            "`dm.load(..., compat_v3=True)` or pass a non-empty rank map."
        )
        raise ValueError(msg)

    rx_set_ids = [int(ds.get("txrx", {}).get("rx_set_id", -1)) for ds in datasets]
    row_axes = ["row" if rx_rank_map.get(rx_set_id, 0) == 0 else "col" for rx_set_id in rx_set_ids]
    col_axes = ["col" if rx_rank_map.get(rx_set_id, 0) == 0 else "row" for rx_set_id in rx_set_ids]
    return row_axes, col_axes


def _merge_rx_grids_v3(
    datasets: list[Dataset],
    rx_rank_map: dict[int, int],
) -> Dataset:
    """Merge RX-grid datasets for one TX into a v3-compatible merged view."""
    row_axes, col_axes = _v3_grid_axes(datasets, rx_rank_map)
    return merge_datasets(datasets, row_axes=row_axes, col_axes=col_axes)


def _apply_v3_compat(
    dataset: Dataset | MacroDataset,
    rx_rank_map: dict[int, int],
) -> Dataset | MacroDataset:
    """Return a loader-level v3 compatibility view with RX grids merged per TX."""
    if isinstance(dataset, Dataset):
        return _merge_rx_grids_v3([dataset], rx_rank_map)
    if len(dataset) <= 1:
        return _merge_rx_grids_v3(dataset.datasets, rx_rank_map)

    grouped: dict[tuple[int, int], list[Dataset]] = {}
    ordered_tx_keys: list[tuple[int, int]] = []
    for child in dataset.datasets:
        txrx = child.get("txrx", {})
        tx_key = (int(txrx.get("tx_set_id", -1)), int(txrx.get("tx_idx", -1)))
        if tx_key not in grouped:
            grouped[tx_key] = []
            ordered_tx_keys.append(tx_key)
        grouped[tx_key].append(child)

    merged = [_merge_rx_grids_v3(grouped[tx_key], rx_rank_map) for tx_key in ordered_tx_keys]
    return merged[0] if len(merged) == 1 else MacroDataset(merged)

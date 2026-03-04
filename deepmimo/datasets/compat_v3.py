"""Backward-compatibility helpers for v3-like multi-grid indexing."""

from __future__ import annotations

from typing import Any

import numpy as np

from deepmimo.datasets.dataset import Dataset
from deepmimo.datasets.sampling import get_grid_idxs


class V3CompatDataset(Dataset):
    """Dataset wrapper that applies v3-style global row/col indexing across RX grids."""

    def __init__(self, data: dict[str, Any] | None = None, *, compat_spec: dict[str, Any]) -> None:
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

        all_ue_idxs = []
        for grid_idx, grid_axis in enumerate(grid_axes):
            mask = (idxs_arr >= grid_offsets[grid_idx]) & (idxs_arr < grid_offsets[grid_idx + 1])
            if not np.any(mask):
                continue
            local_idxs = idxs_arr[mask] - grid_offsets[grid_idx]
            local_ue_idxs = get_grid_idxs(grid_sizes[grid_idx], grid_axis, local_idxs)
            all_ue_idxs.append(local_ue_idxs + ue_offsets[grid_idx])

        if not all_ue_idxs:
            return np.array([], dtype=int)
        return np.concatenate(all_ue_idxs).astype(int)

    def _get_row_idxs(self, row_idxs: int | list[int] | np.ndarray) -> np.ndarray:
        """Return indices of users in v3-compatible global rows."""
        return self._resolve_global_grid_idxs("row", row_idxs)

    def _get_col_idxs(self, col_idxs: int | list[int] | np.ndarray) -> np.ndarray:
        """Return indices of users in v3-compatible global columns."""
        return self._resolve_global_grid_idxs("col", col_idxs)

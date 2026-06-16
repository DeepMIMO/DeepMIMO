# ruff: noqa: EM101, EM102, TRY003
"""Shared metadata helpers for MATLAB RT converter domain builders."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Any

from .errors import MatlabRTValidationError
from .matrices import MatlabRTMatrixSet
from .schema import MatlabRTExport

RAYTRACER_NAME_MATLAB_RT = "MATLAB Ray Tracing"
TX_SET_ID = 0
RX_SET_ID = 1
MATERIAL_DEFAULT_NAME = "PEC"
PLACEHOLDER_SCENE_REPRESENTATION = "empty"


def validate_export(export: MatlabRTExport) -> None:
    """Validate metadata builder input type."""
    if not isinstance(export, MatlabRTExport):
        raise TypeError("export must be a MatlabRTExport instance.")


def validate_matrix_sets(
    export: MatlabRTExport,
    matrix_sets: Sequence[MatlabRTMatrixSet],
) -> tuple[MatlabRTMatrixSet, ...]:
    """Validate matrix sets match the parsed export TX/RX dimensions."""
    if isinstance(matrix_sets, (str, bytes)) or not isinstance(matrix_sets, Sequence):
        raise TypeError("matrix_sets must be a sequence of MatlabRTMatrixSet.")

    checked = tuple(matrix_sets)
    if len(checked) != export.num_tx:
        raise MatlabRTValidationError(
            f"matrix set count {len(checked)} does not match export.num_tx={export.num_tx}."
        )

    expected_tx_indices = tuple(transmitter.index for transmitter in export.transmitters)
    observed_tx_indices = tuple(matrix_set.tx_index for matrix_set in checked)
    if observed_tx_indices != expected_tx_indices:
        raise MatlabRTValidationError(
            "matrix set TX order "
            f"{observed_tx_indices} does not match export order {expected_tx_indices}."
        )

    for matrix_set in checked:
        if not isinstance(matrix_set, MatlabRTMatrixSet):
            raise TypeError("matrix_sets must contain MatlabRTMatrixSet instances.")
        if matrix_set.scalar_shape[0] != export.num_rx:
            raise MatlabRTValidationError(
                f"matrix set tx{matrix_set.tx_index} receiver count "
                f"{matrix_set.scalar_shape[0]} does not match export.num_rx={export.num_rx}."
            )
        if len(matrix_set.path_counts) != export.num_rx:
            raise MatlabRTValidationError(
                f"matrix set tx{matrix_set.tx_index} path_counts length does not match "
                "export.num_rx."
            )

    return checked


def nonnegative_int(value: Any, name: str) -> int:
    """Validate a non-negative integer-like metadata value."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise MatlabRTValidationError(f"{name} must be a non-negative integer.")
    if value < 0:
        raise MatlabRTValidationError(f"{name} must be non-negative.")
    return value


def finite_float(value: Any, name: str) -> float:
    """Validate a finite numeric metadata value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MatlabRTValidationError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise MatlabRTValidationError(f"{name} must be finite.")
    return result


def string_or_default(value: Any, default: str) -> str:
    """Return a non-empty string value or an explicit default."""
    if value is None:
        return default
    if not isinstance(value, str):
        raise MatlabRTValidationError("metadata string values must be strings.")
    stripped = value.strip()
    return stripped or default


def max_export_interactions(export: MatlabRTExport) -> int:
    """Return maximum parsed interaction count across all rays."""
    return max((ray.num_interactions for link in export.links for ray in link.rays), default=0)


def max_matrix_interactions(matrix_sets: Sequence[MatlabRTMatrixSet]) -> int:
    """Return maximum assembled interaction depth across matrix sets."""
    return max((matrix_set.inter_pos_shape[2] for matrix_set in matrix_sets), default=0)

# ruff: noqa: EM101, EM102, TRY003
"""In-memory DeepMIMO-style matrix assembly for MATLAB RT path rows."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from .errors import MatlabRTValidationError
from .paths import MatlabRTLinkPathRows, MatlabRTPathRow, MatlabRTTxPathRows

SCALAR_MATRIX_FIELDS = (
    "power",
    "phase",
    "delay",
    "aoa_az",
    "aoa_el",
    "aod_az",
    "aod_el",
    "inter",
)
MATRIX_FIELDS = (*SCALAR_MATRIX_FIELDS, "inter_pos", "rx_pos", "tx_pos")
VECTOR3_LENGTH = 3


@dataclass(frozen=True)
class MatlabRTMatrixSet:
    """In-memory matrices for one MATLAB RT transmitter point."""

    tx_index: int
    tx_row_index: int
    matrices: dict[str, np.ndarray]
    path_counts: tuple[int, ...]

    @property
    def scalar_shape(self) -> tuple[int, int]:
        """Return the shared shape for scalar path matrices."""
        return tuple(self.matrices["power"].shape)

    @property
    def inter_pos_shape(self) -> tuple[int, int, int, int]:
        """Return the shape of the packed interaction-position matrix."""
        return tuple(self.matrices["inter_pos"].shape)

    @property
    def rx_pos_shape(self) -> tuple[int, int]:
        """Return the receiver-position matrix shape."""
        return tuple(self.matrices["rx_pos"].shape)

    @property
    def tx_pos_shape(self) -> tuple[int, int]:
        """Return the transmitter-position matrix shape."""
        return tuple(self.matrices["tx_pos"].shape)


def assemble_all_tx_matrices(
    tx_groups: Sequence[MatlabRTTxPathRows],
) -> tuple[MatlabRTMatrixSet, ...]:
    """Assemble in-memory matrices for every transmitter path-row group."""
    if isinstance(tx_groups, (str, bytes)) or not isinstance(tx_groups, Sequence):
        raise TypeError("tx_groups must be a sequence of MatlabRTTxPathRows.")

    return tuple(assemble_tx_matrices(tx_group) for tx_group in tx_groups)


def assemble_tx_matrices(tx_rows: MatlabRTTxPathRows) -> MatlabRTMatrixSet:
    """Assemble DeepMIMO-style NumPy arrays for one transmitter path-row group."""
    _validate_tx_path_rows(tx_rows)

    num_rx, max_paths = tx_rows.scalar_shape
    max_interactions = tx_rows.max_interactions
    matrices = empty_matrix_dict(
        num_rx=num_rx,
        max_paths=max_paths,
        max_interactions=max_interactions,
    )
    matrices["rx_pos"] = np.asarray(tx_rows.receiver_positions_m, dtype=np.float64)
    matrices["tx_pos"] = np.asarray([tx_rows.tx_position_m], dtype=np.float64)

    for link in tx_rows.links:
        rx_row = link.rx_row_index
        for row in link.rows:
            path_col = row.path_index
            matrices["power"][rx_row, path_col] = row.power_dbw
            matrices["phase"][rx_row, path_col] = row.phase_deg
            matrices["delay"][rx_row, path_col] = row.delay_s
            matrices["aoa_az"][rx_row, path_col] = row.aoa_az_deg
            matrices["aoa_el"][rx_row, path_col] = row.aoa_el_deg
            matrices["aod_az"][rx_row, path_col] = row.aod_az_deg
            matrices["aod_el"][rx_row, path_col] = row.aod_el_deg
            matrices["inter"][rx_row, path_col] = row.interaction_code
            matrices["inter_pos"][rx_row, path_col, :, :] = np.asarray(
                row.interaction_positions_m,
                dtype=np.float64,
            )

    return MatlabRTMatrixSet(
        tx_index=tx_rows.tx_index,
        tx_row_index=tx_rows.tx_row_index,
        matrices=matrices,
        path_counts=tx_rows.path_counts,
    )


def empty_matrix_dict(
    *,
    num_rx: int,
    max_paths: int,
    max_interactions: int,
) -> dict[str, np.ndarray]:
    """Create empty NaN-filled matrix arrays with MVP DeepMIMO shapes."""
    _validate_positive_int(num_rx, "num_rx")
    _validate_positive_int(max_paths, "max_paths")
    _validate_positive_int(max_interactions, "max_interactions")

    scalar_shape = (num_rx, max_paths)
    matrices = {
        field: np.full(scalar_shape, np.nan, dtype=np.float64)
        for field in SCALAR_MATRIX_FIELDS
    }
    matrices["inter_pos"] = np.full(
        (num_rx, max_paths, max_interactions, 3),
        np.nan,
        dtype=np.float64,
    )
    matrices["rx_pos"] = np.full((num_rx, 3), np.nan, dtype=np.float64)
    matrices["tx_pos"] = np.full((1, 3), np.nan, dtype=np.float64)
    return matrices


def _validate_tx_path_rows(tx_rows: MatlabRTTxPathRows) -> None:
    """Validate that path-row data can be assembled without shape ambiguity."""
    if not isinstance(tx_rows, MatlabRTTxPathRows):
        raise TypeError("tx_rows must be a MatlabRTTxPathRows instance.")

    _validate_positive_int(tx_rows.max_paths, "tx_rows.max_paths")
    _validate_positive_int(tx_rows.max_interactions, "tx_rows.max_interactions")
    _validate_vector3(tx_rows.tx_position_m, "tx_rows.tx_position_m", allow_nan=False)
    for rx_index, receiver_position in enumerate(tx_rows.receiver_positions_m):
        _validate_vector3(
            receiver_position,
            f"tx_rows.receiver_positions_m[{rx_index}]",
            allow_nan=False,
        )

    num_rx = len(tx_rows.receiver_positions_m)
    if num_rx == 0:
        raise MatlabRTValidationError("tx_rows must include at least one receiver position.")
    if len(tx_rows.links) != num_rx:
        raise MatlabRTValidationError(
            f"tx_rows.links length {len(tx_rows.links)} does not match num_rx={num_rx}."
        )

    observed_rx_rows: set[int] = set()
    for link in tx_rows.links:
        _validate_link_path_rows(link, tx_rows)
        if link.rx_row_index in observed_rx_rows:
            raise MatlabRTValidationError(f"Duplicate rx_row_index {link.rx_row_index}.")
        observed_rx_rows.add(link.rx_row_index)

    expected_rx_rows = set(range(num_rx))
    if observed_rx_rows != expected_rx_rows:
        raise MatlabRTValidationError(
            "tx_rows links must cover every receiver row exactly once: "
            f"missing={sorted(expected_rx_rows - observed_rx_rows)}, "
            f"extra={sorted(observed_rx_rows - expected_rx_rows)}."
        )


def _validate_link_path_rows(
    link: MatlabRTLinkPathRows,
    tx_rows: MatlabRTTxPathRows,
) -> None:
    """Validate one link path-row block before assigning matrix slots."""
    if not isinstance(link, MatlabRTLinkPathRows):
        raise TypeError("tx_rows.links must contain MatlabRTLinkPathRows.")
    if link.tx_index != tx_rows.tx_index:
        raise MatlabRTValidationError(
            f"Link tx_index {link.tx_index} does not match group tx_index {tx_rows.tx_index}."
        )
    if link.max_paths != tx_rows.max_paths:
        raise MatlabRTValidationError("link.max_paths must match tx_rows.max_paths.")
    if link.max_interactions != tx_rows.max_interactions:
        raise MatlabRTValidationError("link.max_interactions must match tx_rows.max_interactions.")
    if len(link.rows) != tx_rows.max_paths:
        raise MatlabRTValidationError(
            f"Link ({link.tx_index}, {link.rx_index}) row count {len(link.rows)} "
            f"does not match max_paths={tx_rows.max_paths}."
        )
    if not 0 <= link.rx_row_index < len(tx_rows.receiver_positions_m):
        raise MatlabRTValidationError(f"Invalid rx_row_index {link.rx_row_index}.")

    _validate_vector3(link.tx_position_m, "link.tx_position_m", allow_nan=False)
    _validate_vector3(link.rx_position_m, "link.rx_position_m", allow_nan=False)

    valid_count = sum(not row.is_padding for row in link.rows)
    if link.path_count != valid_count:
        raise MatlabRTValidationError(
            f"Link ({link.tx_index}, {link.rx_index}) path_count {link.path_count} "
            f"does not match valid row count {valid_count}."
        )

    observed_path_cols: set[int] = set()
    for expected_path_index, row in enumerate(link.rows):
        _validate_path_row(row, link, expected_path_index)
        observed_path_cols.add(row.path_index)

    expected_path_cols = set(range(tx_rows.max_paths))
    if observed_path_cols != expected_path_cols:
        raise MatlabRTValidationError(
            f"Link ({link.tx_index}, {link.rx_index}) rows must cover all path columns."
        )


def _validate_path_row(
    row: MatlabRTPathRow,
    link: MatlabRTLinkPathRows,
    expected_path_index: int,
) -> None:
    """Validate one row before assigning it to a matrix path slot."""
    if not isinstance(row, MatlabRTPathRow):
        raise TypeError("link.rows must contain MatlabRTPathRow.")
    if row.tx_index != link.tx_index or row.rx_index != link.rx_index:
        raise MatlabRTValidationError("Path row TX/RX indices do not match parent link.")
    if row.path_index != expected_path_index:
        raise MatlabRTValidationError(
            f"Path row index {row.path_index} does not match expected {expected_path_index}."
        )

    if row.is_padding:
        _validate_padding_row(row, link.max_interactions)
        return

    for field_name, value in (
        ("aoa_az_deg", row.aoa_az_deg),
        ("aoa_el_deg", row.aoa_el_deg),
        ("aod_az_deg", row.aod_az_deg),
        ("aod_el_deg", row.aod_el_deg),
        ("power_dbw", row.power_dbw),
        ("phase_deg", row.phase_deg),
        ("delay_s", row.delay_s),
        ("interaction_code", row.interaction_code),
    ):
        _validate_real(value, f"row.{field_name}", allow_nan=False)

    _validate_interaction_positions(row, link.max_interactions, allow_nan=True)


def _validate_padding_row(row: MatlabRTPathRow, max_interactions: int) -> None:
    """Validate that padding rows are explicit NaN-filled slots."""
    for field_name, value in (
        ("aoa_az_deg", row.aoa_az_deg),
        ("aoa_el_deg", row.aoa_el_deg),
        ("aod_az_deg", row.aod_az_deg),
        ("aod_el_deg", row.aod_el_deg),
        ("power_dbw", row.power_dbw),
        ("phase_deg", row.phase_deg),
        ("delay_s", row.delay_s),
        ("interaction_code", row.interaction_code),
    ):
        _validate_real(value, f"padding_row.{field_name}", allow_nan=True)
        if not math.isnan(float(value)):
            raise MatlabRTValidationError(f"padding_row.{field_name} must be NaN.")

    _validate_interaction_positions(row, max_interactions, allow_nan=True)
    for position in row.interaction_positions_m:
        if any(not math.isnan(float(value)) for value in position):
            raise MatlabRTValidationError("padding row interaction positions must be NaN.")


def _validate_interaction_positions(
    row: MatlabRTPathRow,
    max_interactions: int,
    *,
    allow_nan: bool,
) -> None:
    """Validate packed interaction position rows."""
    if len(row.interaction_positions_m) != max_interactions:
        raise MatlabRTValidationError(
            f"row.interaction_positions_m length {len(row.interaction_positions_m)} "
            f"does not match max_interactions={max_interactions}."
        )
    for position_index, position in enumerate(row.interaction_positions_m):
        _validate_vector3(
            position,
            f"row.interaction_positions_m[{position_index}]",
            allow_nan=allow_nan,
        )


def _validate_vector3(value: object, name: str, *, allow_nan: bool) -> None:
    """Validate one vector with three real numeric entries."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MatlabRTValidationError(f"{name} must be a 3D numeric sequence.")
    if len(value) != VECTOR3_LENGTH:
        raise MatlabRTValidationError(f"{name} must contain exactly three values.")
    for item_index, item in enumerate(value):
        _validate_real(item, f"{name}[{item_index}]", allow_nan=allow_nan)


def _validate_real(value: object, name: str, *, allow_nan: bool) -> None:
    """Validate one real scalar, optionally permitting NaN padding."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MatlabRTValidationError(f"{name} must be a real numeric value.")

    numeric = float(value)
    if math.isnan(numeric) and allow_nan:
        return
    if not math.isfinite(numeric):
        raise MatlabRTValidationError(f"{name} must be finite.")


def _validate_positive_int(value: int, name: str) -> None:
    """Validate a positive integer shape parameter."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")

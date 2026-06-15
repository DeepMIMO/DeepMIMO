# ruff: noqa: EM101, EM102, PLR0913, TRY003
"""In-memory path-row helpers for parsed MATLAB RT exports."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import MatlabRTValidationError, UnsupportedMatlabRTFeatureError
from .interactions import DEEPMIMO_LOS_CODE, matlab_interaction_type_to_code
from .schema import MatlabRTExport, MatlabRTInteraction, MatlabRTLink, MatlabRTRay
from .units import (
    matlab_elevation_to_deepmimo_theta,
    matlab_path_loss_to_power_dbw,
    matlab_phase_to_deg,
    normalize_signed_deg,
)

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class MatlabRTPathRow:
    """One future matrix path slot for one parsed MATLAB RT ray."""

    tx_index: int
    rx_index: int
    path_index: int
    ray_index: int | None
    is_padding: bool
    line_of_sight: bool | None
    aoa_az_deg: float
    aoa_el_deg: float
    aod_az_deg: float
    aod_el_deg: float
    power_dbw: float
    phase_deg: float
    delay_s: float
    interaction_code: float
    interaction_positions_m: tuple[Vector3, ...]
    path_coordinates_m: tuple[Vector3, ...]

    @property
    def scalar_values(self) -> tuple[float, float, float, float, float, float, float, float]:
        """Return scalar path values in the future matrix-field order."""
        return (
            self.aoa_az_deg,
            self.aoa_el_deg,
            self.aod_az_deg,
            self.aod_el_deg,
            self.power_dbw,
            self.phase_deg,
            self.delay_s,
            self.interaction_code,
        )

    @property
    def interaction_positions_shape(self) -> tuple[int, int]:
        """Return the shape of this row's packed interaction positions."""
        return (len(self.interaction_positions_m), 3)


@dataclass(frozen=True)
class MatlabRTLinkPathRows:
    """Padded path rows for one explicit TX/RX link."""

    tx_index: int
    rx_index: int
    rx_row_index: int
    tx_position_m: Vector3
    rx_position_m: Vector3
    path_count: int
    max_paths: int
    max_interactions: int
    rows: tuple[MatlabRTPathRow, ...]

    @property
    def scalar_shape(self) -> tuple[int]:
        """Return this link row's future scalar matrix slice shape."""
        return (len(self.rows),)

    @property
    def interaction_positions_shape(self) -> tuple[int, int, int]:
        """Return this link row's future interaction-position slice shape."""
        return (len(self.rows), self.max_interactions, 3)


@dataclass(frozen=True)
class MatlabRTTxPathRows:
    """All padded path rows needed to assemble one future per-TX matrix set."""

    tx_index: int
    tx_row_index: int
    tx_position_m: Vector3
    receiver_positions_m: tuple[Vector3, ...]
    max_paths: int
    max_interactions: int
    links: tuple[MatlabRTLinkPathRows, ...]

    @property
    def scalar_shape(self) -> tuple[int, int]:
        """Return the future scalar matrix shape for this TX group."""
        return (len(self.links), self.max_paths)

    @property
    def interaction_positions_shape(self) -> tuple[int, int, int, int]:
        """Return the future ``inter_pos`` matrix shape for this TX group."""
        return (len(self.links), self.max_paths, self.max_interactions, 3)

    @property
    def path_counts(self) -> tuple[int, ...]:
        """Return valid path counts in receiver-row order."""
        return tuple(link.path_count for link in self.links)


def group_links_by_pair(export: MatlabRTExport) -> dict[tuple[int, int], MatlabRTLink]:
    """Return explicit MATLAB RT links keyed by ``(tx_index, rx_index)``."""
    _validate_export_type(export)

    links_by_pair: dict[tuple[int, int], MatlabRTLink] = {}
    for link in export.links:
        pair = (link.tx_index, link.rx_index)
        if pair in links_by_pair:
            raise MatlabRTValidationError(f"Duplicate MATLAB RT link pair: {pair}.")
        links_by_pair[pair] = link

    return links_by_pair


def source_links_by_tx(export: MatlabRTExport, tx_index: int) -> tuple[MatlabRTLink, ...]:
    """Return deterministic receiver-ordered links for one transmitter index."""
    _validate_export_graph(export)
    _validate_known_tx(export, tx_index)

    return tuple(
        sorted(
            (link for link in export.links if link.tx_index == tx_index),
            key=lambda link: link.rx_index,
        )
    )


def build_path_row_groups(
    export: MatlabRTExport,
    *,
    tx_power_dbw: float = 0.0,
    tx_gain_db: float = 0.0,
    rx_gain_db: float = 0.0,
) -> tuple[MatlabRTTxPathRows, ...]:
    """Build deterministic, padded in-memory path rows for every transmitter."""
    _validate_export_graph(export)

    tx_groups: list[MatlabRTTxPathRows] = []
    receiver_positions = tuple(receiver.antenna_position_m for receiver in export.receivers)

    for tx_row_index, transmitter in enumerate(export.transmitters):
        links = source_links_by_tx(export, transmitter.index)
        max_paths = max((link.num_rays for link in links), default=0)
        max_paths = max(max_paths, 1)
        max_interactions = max(
            (ray.num_interactions for link in links for ray in link.rays),
            default=0,
        )
        max_interactions = max(max_interactions, 1)

        link_rows = tuple(
            build_link_path_rows(
                link,
                rx_row_index=link.rx_index - 1,
                max_paths=max_paths,
                max_interactions=max_interactions,
                tx_power_dbw=tx_power_dbw,
                tx_gain_db=tx_gain_db,
                rx_gain_db=rx_gain_db,
            )
            for link in links
        )

        tx_groups.append(
            MatlabRTTxPathRows(
                tx_index=transmitter.index,
                tx_row_index=tx_row_index,
                tx_position_m=transmitter.antenna_position_m,
                receiver_positions_m=receiver_positions,
                max_paths=max_paths,
                max_interactions=max_interactions,
                links=link_rows,
            )
        )

    return tuple(tx_groups)


def build_link_path_rows(
    link: MatlabRTLink,
    *,
    rx_row_index: int,
    max_paths: int,
    max_interactions: int,
    tx_power_dbw: float = 0.0,
    tx_gain_db: float = 0.0,
    rx_gain_db: float = 0.0,
) -> MatlabRTLinkPathRows:
    """Build padded path rows for one already-validated TX/RX link."""
    _validate_non_negative_int(rx_row_index, "rx_row_index")
    _validate_positive_int(max_paths, "max_paths")
    _validate_positive_int(max_interactions, "max_interactions")
    _validate_unique_ray_indices(link)

    valid_rows = tuple(
        path_row_from_ray(
            ray,
            tx_index=link.tx_index,
            rx_index=link.rx_index,
            path_index=path_index,
            max_interactions=max_interactions,
            tx_power_dbw=tx_power_dbw,
            tx_gain_db=tx_gain_db,
            rx_gain_db=rx_gain_db,
        )
        for path_index, ray in enumerate(
            sorted(
                link.rays,
                key=lambda ray: (
                    -_power_for_sort(
                        ray,
                        tx_power_dbw=tx_power_dbw,
                        tx_gain_db=tx_gain_db,
                        rx_gain_db=rx_gain_db,
                    ),
                    ray.index,
                ),
            )
        )
    )

    if len(valid_rows) > max_paths:
        raise MatlabRTValidationError(
            f"Link ({link.tx_index}, {link.rx_index}) has {len(valid_rows)} rows, "
            f"exceeding max_paths={max_paths}."
        )

    padding_rows = tuple(
        padding_path_row(
            tx_index=link.tx_index,
            rx_index=link.rx_index,
            path_index=path_index,
            max_interactions=max_interactions,
        )
        for path_index in range(len(valid_rows), max_paths)
    )

    return MatlabRTLinkPathRows(
        tx_index=link.tx_index,
        rx_index=link.rx_index,
        rx_row_index=rx_row_index,
        tx_position_m=link.tx_position_m,
        rx_position_m=link.rx_position_m,
        path_count=len(valid_rows),
        max_paths=max_paths,
        max_interactions=max_interactions,
        rows=valid_rows + padding_rows,
    )


def path_row_from_ray(
    ray: MatlabRTRay,
    *,
    tx_index: int,
    rx_index: int,
    path_index: int,
    max_interactions: int,
    tx_power_dbw: float = 0.0,
    tx_gain_db: float = 0.0,
    rx_gain_db: float = 0.0,
) -> MatlabRTPathRow:
    """Convert one parsed MATLAB RT ray into one in-memory path row."""
    _validate_non_negative_int(path_index, "path_index")
    _validate_positive_int(max_interactions, "max_interactions")

    try:
        aoa_az, aoa_el = ray.angle_of_arrival_deg
        aod_az, aod_el = ray.angle_of_departure_deg
        interaction_positions = _pack_interaction_positions(ray.interactions, max_interactions)

        return MatlabRTPathRow(
            tx_index=tx_index,
            rx_index=rx_index,
            path_index=path_index,
            ray_index=ray.index,
            is_padding=False,
            line_of_sight=ray.line_of_sight,
            aoa_az_deg=normalize_signed_deg(aoa_az),
            aoa_el_deg=matlab_elevation_to_deepmimo_theta(aoa_el),
            aod_az_deg=normalize_signed_deg(aod_az),
            aod_el_deg=matlab_elevation_to_deepmimo_theta(aod_el),
            power_dbw=matlab_path_loss_to_power_dbw(
                ray.path_loss_db,
                tx_power_dbw=tx_power_dbw,
                tx_gain_db=tx_gain_db,
                rx_gain_db=rx_gain_db,
            ),
            phase_deg=matlab_phase_to_deg(
                phase_shift_deg=ray.phase_shift_deg,
                phase_shift_rad=ray.phase_shift_rad,
            ),
            delay_s=_finite_float(ray.propagation_delay_s, "ray.propagation_delay_s"),
            interaction_code=float(_interaction_sequence_code(ray.interactions)),
            interaction_positions_m=interaction_positions,
            path_coordinates_m=ray.path_coordinates_m,
        )
    except (MatlabRTValidationError, UnsupportedMatlabRTFeatureError):
        raise
    except Exception as exc:
        raise MatlabRTValidationError(
            f"Invalid ray {ray.index} on link ({tx_index}, {rx_index}): {exc}"
        ) from exc


def padding_path_row(
    *,
    tx_index: int,
    rx_index: int,
    path_index: int,
    max_interactions: int,
) -> MatlabRTPathRow:
    """Create one explicit NaN padding path row."""
    _validate_non_negative_int(path_index, "path_index")
    _validate_positive_int(max_interactions, "max_interactions")

    return MatlabRTPathRow(
        tx_index=tx_index,
        rx_index=rx_index,
        path_index=path_index,
        ray_index=None,
        is_padding=True,
        line_of_sight=None,
        aoa_az_deg=math.nan,
        aoa_el_deg=math.nan,
        aod_az_deg=math.nan,
        aod_el_deg=math.nan,
        power_dbw=math.nan,
        phase_deg=math.nan,
        delay_s=math.nan,
        interaction_code=math.nan,
        interaction_positions_m=_nan_positions(max_interactions),
        path_coordinates_m=(),
    )


def _validate_export_graph(export: MatlabRTExport) -> None:
    """Validate that a parsed export still has complete explicit link coverage."""
    _validate_export_type(export)
    transmitter_indices = tuple(transmitter.index for transmitter in export.transmitters)
    receiver_indices = tuple(receiver.index for receiver in export.receivers)
    if len(transmitter_indices) != len(set(transmitter_indices)):
        raise MatlabRTValidationError("Transmitter indices must be unique.")
    if len(receiver_indices) != len(set(receiver_indices)):
        raise MatlabRTValidationError("Receiver indices must be unique.")

    observed_pairs = set(group_links_by_pair(export))
    expected_pairs = {
        (tx_index, rx_index)
        for tx_index in transmitter_indices
        for rx_index in receiver_indices
    }
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        extra = sorted(observed_pairs - expected_pairs)
        raise MatlabRTValidationError(
            f"Incomplete MATLAB RT path-row coverage: missing={missing}, extra={extra}."
        )


def _validate_export_type(export: MatlabRTExport) -> None:
    """Validate the path-row builder input type."""
    if not isinstance(export, MatlabRTExport):
        raise TypeError("export must be a MatlabRTExport instance.")


def _validate_known_tx(export: MatlabRTExport, tx_index: int) -> None:
    """Validate that a transmitter index exists in an export."""
    if tx_index not in {transmitter.index for transmitter in export.transmitters}:
        raise MatlabRTValidationError(f"Unknown transmitter index: {tx_index}.")


def _validate_unique_ray_indices(link: MatlabRTLink) -> None:
    """Validate that path-row ordering has unique ray index tie-breakers."""
    ray_indices = [ray.index for ray in link.rays]
    if len(ray_indices) != len(set(ray_indices)):
        raise MatlabRTValidationError(
            f"Duplicate ray indices on link ({link.tx_index}, {link.rx_index})."
        )


def _validate_non_negative_int(value: int, name: str) -> None:
    """Validate a non-negative integer helper argument."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_positive_int(value: int, name: str) -> None:
    """Validate a positive integer helper argument."""
    _validate_non_negative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive.")


def _power_for_sort(
    ray: MatlabRTRay,
    *,
    tx_power_dbw: float,
    tx_gain_db: float,
    rx_gain_db: float,
) -> float:
    """Return sortable received power for one ray."""
    try:
        return matlab_path_loss_to_power_dbw(
            ray.path_loss_db,
            tx_power_dbw=tx_power_dbw,
            tx_gain_db=tx_gain_db,
            rx_gain_db=rx_gain_db,
        )
    except Exception as exc:
        raise MatlabRTValidationError(f"Invalid path loss on ray {ray.index}: {exc}") from exc


def _interaction_sequence_code(interactions: tuple[MatlabRTInteraction, ...]) -> int:
    """Encode parsed interaction objects with the MVP interaction convention."""
    if not interactions:
        return DEEPMIMO_LOS_CODE

    return int(
        "".join(
            str(matlab_interaction_type_to_code(interaction.type))
            for interaction in interactions
        )
    )


def _pack_interaction_positions(
    interactions: tuple[MatlabRTInteraction, ...],
    max_interactions: int,
) -> tuple[Vector3, ...]:
    """Pack parsed interaction positions into fixed-depth rows."""
    if len(interactions) > max_interactions:
        raise MatlabRTValidationError(
            f"Interaction count {len(interactions)} exceeds max_interactions={max_interactions}."
        )

    positions = tuple(interaction.location_m for interaction in interactions)
    return positions + _nan_positions(max_interactions - len(positions))


def _nan_positions(count: int) -> tuple[Vector3, ...]:
    """Return ``count`` NaN vector rows."""
    _validate_non_negative_int(count, "count")
    return tuple((math.nan, math.nan, math.nan) for _ in range(count))


def _finite_float(value: float, name: str) -> float:
    """Validate a finite float coming from a parsed schema object."""
    result = float(value)
    if not math.isfinite(result):
        raise MatlabRTValidationError(f"{name} must be finite.")
    return result

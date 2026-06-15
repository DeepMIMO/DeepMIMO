# ruff: noqa: EM101, EM102, TRY003
"""Parser and validation boundary for MATLAB RT JSON exports."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .errors import (
    MatlabRTSchemaError,
    MatlabRTValidationError,
    UnsupportedMatlabRTFeatureError,
)
from .interactions import matlab_interaction_type_to_code
from .schema import (
    MatlabRTExport,
    MatlabRTInteraction,
    MatlabRTLink,
    MatlabRTMetadata,
    MatlabRTRay,
    MatlabRTReceiver,
    MatlabRTScene,
    MatlabRTTransmitter,
)


def parse_matlab_rt_json(path: str | Path) -> MatlabRTExport:
    """Load and parse a supported MATLAB RT JSON export."""
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return parse_matlab_rt_export(data, source_path=source_path)


def parse_matlab_rt_export(
    data: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> MatlabRTExport:
    """Parse a MATLAB RT export mapping into typed converter schema objects."""
    export_data = _as_mapping(data, "export")
    source = Path(source_path) if source_path is not None else None
    raw = deepcopy(dict(export_data))

    metadata = _parse_metadata(_require_mapping(export_data, "metadata", "export"))
    scene = _parse_scene(_require_mapping(export_data, "scene", "export"))
    propagation_model = dict(_require_mapping(export_data, "propagation_model", "export"))
    _require_cartesian(
        propagation_model.get("coordinate_system", scene.coordinate_system),
        "propagation_model.coordinate_system",
    )

    if _is_multi_link_export(export_data):
        transmitters = _parse_transmitters(
            _require_sequence(export_data, "transmitters", "export"),
        )
        receivers = _parse_receivers(_require_sequence(export_data, "receivers", "export"))
        num_tx = _require_int(export_data, "num_tx", "export")
        num_rx = _require_int(export_data, "num_rx", "export")
        if num_tx != len(transmitters):
            raise MatlabRTValidationError(
                f"num_tx mismatch: declared {num_tx}, parsed {len(transmitters)}."
            )
        if num_rx != len(receivers):
            raise MatlabRTValidationError(
                f"num_rx mismatch: declared {num_rx}, parsed {len(receivers)}."
            )
        links = _parse_links(
            _require_sequence(export_data, "links", "export"),
            transmitters,
            receivers,
        )
    elif _is_single_link_export(export_data):
        transmitters = (
            _parse_transmitter(_require_mapping(export_data, "transmitter", "export"), index=1),
        )
        receivers = (_parse_receiver(_require_mapping(export_data, "receiver", "export"), index=1),)
        rays = _parse_rays(_require_sequence(export_data, "rays", "export"))
        declared_num_rays = _require_int(export_data, "num_rays", "export")
        if declared_num_rays != len(rays):
            raise MatlabRTValidationError(
                f"num_rays mismatch: declared {declared_num_rays}, parsed {len(rays)}."
            )
        links = (
            MatlabRTLink(
                index=1,
                tx_index=1,
                rx_index=1,
                tx_name=transmitters[0].name,
                rx_name=receivers[0].name,
                tx_position_m=transmitters[0].antenna_position_m,
                rx_position_m=receivers[0].antenna_position_m,
                rays=rays,
                raw={
                    "index": 1,
                    "tx_index": 1,
                    "rx_index": 1,
                    "num_rays": declared_num_rays,
                    "rays": [ray.raw for ray in rays],
                },
            ),
        )
    else:
        missing_schema = _missing_supported_schema_fields(export_data)
        raise MatlabRTSchemaError(
            "MATLAB RT export does not match supported single-link or multi-link schema. "
            f"Missing fields: {sorted(missing_schema)}."
        )

    return MatlabRTExport(
        source_path=source,
        metadata=metadata,
        scene=scene,
        propagation_model=propagation_model,
        transmitters=transmitters,
        receivers=receivers,
        links=links,
        raw=raw,
    )


def _is_multi_link_export(data: Mapping[str, Any]) -> bool:
    """Return true when data declares a multi-link MATLAB RT export."""
    multi_markers = {"num_tx", "num_rx", "transmitters", "receivers"}
    if multi_markers & set(data):
        missing = (multi_markers | {"links"}) - set(data)
        if missing:
            raise MatlabRTSchemaError(f"Missing required multi-link fields: {sorted(missing)}.")
        return True
    return False


def _is_single_link_export(data: Mapping[str, Any]) -> bool:
    """Return true when data declares a single-link MATLAB RT export."""
    single_fields = {"transmitter", "receiver", "num_rays", "rays"}
    return single_fields.issubset(data)


def _missing_supported_schema_fields(data: Mapping[str, Any]) -> set[str]:
    """Return missing fields for the closest supported schema."""
    single_required = {
        "metadata",
        "scene",
        "propagation_model",
        "transmitter",
        "receiver",
        "num_rays",
        "rays",
    }
    multi_required = {
        "metadata",
        "scene",
        "propagation_model",
        "num_tx",
        "num_rx",
        "transmitters",
        "receivers",
        "links",
    }
    missing_single = single_required - set(data)
    missing_multi = multi_required - set(data)
    return missing_single if len(missing_single) <= len(missing_multi) else missing_multi


def _parse_metadata(data: Mapping[str, Any]) -> MatlabRTMetadata:
    """Parse top-level MATLAB RT metadata."""
    return MatlabRTMetadata(
        experiment=_optional_str(data.get("experiment"), "metadata.experiment"),
        matlab_version=_optional_str(data.get("matlab_version"), "metadata.matlab_version"),
        description=_optional_str(data.get("description"), "metadata.description"),
        raw=deepcopy(dict(data)),
    )


def _parse_scene(data: Mapping[str, Any]) -> MatlabRTScene:
    """Parse scene metadata and validate Cartesian coordinates."""
    coordinate_system = _require_str(data, "coordinate_system", "scene")
    _require_cartesian(coordinate_system, "scene.coordinate_system")

    return MatlabRTScene(
        coordinate_system=coordinate_system,
        frequency_hz=_require_finite_float(data, "frequency_hz", "scene"),
        raw=deepcopy(dict(data)),
    )


def _parse_transmitters(values: Sequence[Any]) -> tuple[MatlabRTTransmitter, ...]:
    """Parse, validate, and deterministically order transmitter records."""
    transmitters = tuple(
        sorted(
            (
                _parse_transmitter(_as_mapping(value, "transmitter"))
                for value in _as_sequence(values, "transmitters")
            ),
            key=lambda transmitter: transmitter.index,
        )
    )
    _validate_contiguous_indices(
        [transmitter.index for transmitter in transmitters],
        "transmitters",
    )
    return transmitters


def _parse_receivers(values: Sequence[Any]) -> tuple[MatlabRTReceiver, ...]:
    """Parse, validate, and deterministically order receiver records."""
    receivers = tuple(
        sorted(
            (
                _parse_receiver(_as_mapping(value, "receiver"))
                for value in _as_sequence(values, "receivers")
            ),
            key=lambda receiver: receiver.index,
        )
    )
    _validate_contiguous_indices([receiver.index for receiver in receivers], "receivers")
    return receivers


def _parse_transmitter(data: Mapping[str, Any], *, index: int | None = None) -> MatlabRTTransmitter:
    """Parse one transmitter record."""
    parsed_index = index if index is not None else _require_int(data, "index", "transmitter")
    name = _optional_str(data.get("name"), "transmitter.name") or f"tx{parsed_index}"

    return MatlabRTTransmitter(
        index=parsed_index,
        name=name,
        antenna_position_m=_require_vector3(data, "antenna_position_m", "transmitter"),
        transmitter_frequency_hz=_optional_finite_float(
            data.get("transmitter_frequency_hz"),
            "transmitter.transmitter_frequency_hz",
        ),
        raw=deepcopy(dict(data)),
    )


def _parse_receiver(data: Mapping[str, Any], *, index: int | None = None) -> MatlabRTReceiver:
    """Parse one receiver record."""
    parsed_index = index if index is not None else _require_int(data, "index", "receiver")
    name = _optional_str(data.get("name"), "receiver.name") or f"rx{parsed_index}"

    return MatlabRTReceiver(
        index=parsed_index,
        name=name,
        antenna_position_m=_require_vector3(data, "antenna_position_m", "receiver"),
        raw=deepcopy(dict(data)),
    )


def _parse_links(
    values: Sequence[Any],
    transmitters: tuple[MatlabRTTransmitter, ...],
    receivers: tuple[MatlabRTReceiver, ...],
) -> tuple[MatlabRTLink, ...]:
    """Parse and validate a complete deterministic TX/RX link grid."""
    transmitter_by_index = {transmitter.index: transmitter for transmitter in transmitters}
    receiver_by_index = {receiver.index: receiver for receiver in receivers}

    links = tuple(
        sorted(
            (
                _parse_link(
                    _as_mapping(value, "link"),
                    transmitter_by_index,
                    receiver_by_index,
                )
                for value in _as_sequence(values, "links")
            ),
            key=lambda link: (link.tx_index, link.rx_index),
        )
    )
    _validate_link_coverage(links, transmitter_by_index, receiver_by_index)
    return links


def _parse_link(
    data: Mapping[str, Any],
    transmitter_by_index: Mapping[int, MatlabRTTransmitter],
    receiver_by_index: Mapping[int, MatlabRTReceiver],
) -> MatlabRTLink:
    """Parse one TX/RX link."""
    tx_index = _require_int(data, "tx_index", "link")
    rx_index = _require_int(data, "rx_index", "link")

    if tx_index not in transmitter_by_index:
        raise MatlabRTValidationError(f"link references unknown tx_index {tx_index}.")
    if rx_index not in receiver_by_index:
        raise MatlabRTValidationError(f"link references unknown rx_index {rx_index}.")

    tx_position_m = _require_vector3(data, "tx_position_m", "link")
    rx_position_m = _require_vector3(data, "rx_position_m", "link")
    transmitter = transmitter_by_index[tx_index]
    receiver = receiver_by_index[rx_index]

    _require_same_vector3(tx_position_m, transmitter.antenna_position_m, "link.tx_position_m")
    _require_same_vector3(rx_position_m, receiver.antenna_position_m, "link.rx_position_m")

    rays = _parse_rays(_require_sequence(data, "rays", "link"))
    declared_num_rays = _require_int(data, "num_rays", "link")
    if declared_num_rays != len(rays):
        raise MatlabRTValidationError(
            f"num_rays mismatch on link ({tx_index}, {rx_index}): "
            f"declared {declared_num_rays}, parsed {len(rays)}."
        )

    return MatlabRTLink(
        index=_require_int(data, "index", "link"),
        tx_index=tx_index,
        rx_index=rx_index,
        tx_name=_optional_str(data.get("tx_name"), "link.tx_name"),
        rx_name=_optional_str(data.get("rx_name"), "link.rx_name"),
        tx_position_m=tx_position_m,
        rx_position_m=rx_position_m,
        rays=rays,
        raw=deepcopy(dict(data)),
    )


def _parse_rays(values: Sequence[Any]) -> tuple[MatlabRTRay, ...]:
    """Parse, validate, and deterministically order ray records."""
    rays = tuple(
        sorted(
            (_parse_ray(_as_mapping(value, "ray")) for value in _as_sequence(values, "rays")),
            key=lambda ray: ray.index,
        )
    )
    _validate_unique_indices([ray.index for ray in rays], "rays")
    return rays


def _parse_ray(data: Mapping[str, Any]) -> MatlabRTRay:
    """Parse one MATLAB ``comm.Ray`` object."""
    coordinate_system = _optional_str(data.get("coordinate_system"), "ray.coordinate_system")
    if coordinate_system is not None:
        _require_cartesian(coordinate_system, "ray.coordinate_system")

    interactions = _parse_interactions(_require_sequence(data, "interactions", "ray"))
    declared_num_interactions = _require_int(data, "num_interactions", "ray")
    if declared_num_interactions != len(interactions):
        raise MatlabRTValidationError(
            "num_interactions mismatch: "
            f"declared {declared_num_interactions}, parsed {len(interactions)}."
        )

    path_coordinates_m = _require_path_coordinates(data, "path_coordinates_m", "ray")
    min_path_points = len(interactions) + 2
    if len(path_coordinates_m) < min_path_points:
        raise MatlabRTValidationError(
            "path_coordinates_m must include TX, interactions, and RX points: "
            f"got {len(path_coordinates_m)}, expected at least {min_path_points}."
        )
    phase_shift_rad = _optional_finite_float(data.get("phase_shift_rad"), "ray.phase_shift_rad")
    phase_shift_deg = _optional_finite_float(data.get("phase_shift_deg"), "ray.phase_shift_deg")
    if phase_shift_rad is None and phase_shift_deg is None:
        raise MatlabRTSchemaError("ray requires phase_shift_deg or phase_shift_rad.")

    return MatlabRTRay(
        index=_require_int(data, "index", "ray"),
        line_of_sight=_require_bool(data, "line_of_sight", "ray"),
        transmitter_location_m=_require_vector3(data, "transmitter_location_m", "ray"),
        receiver_location_m=_require_vector3(data, "receiver_location_m", "ray"),
        frequency_hz=_optional_finite_float(data.get("frequency_hz"), "ray.frequency_hz"),
        path_loss_db=_require_finite_float(data, "path_loss_db", "ray"),
        phase_shift_rad=phase_shift_rad,
        phase_shift_deg=phase_shift_deg,
        propagation_delay_s=_require_finite_float(data, "propagation_delay_s", "ray"),
        propagation_distance_m=_require_finite_float(data, "propagation_distance_m", "ray"),
        angle_of_departure_deg=_require_vector2(data, "angle_of_departure_deg", "ray"),
        angle_of_arrival_deg=_require_vector2(data, "angle_of_arrival_deg", "ray"),
        interactions=interactions,
        path_coordinates_m=path_coordinates_m,
        raw=deepcopy(dict(data)),
    )


def _parse_interactions(values: Sequence[Any]) -> tuple[MatlabRTInteraction, ...]:
    """Parse supported MATLAB RT interactions."""
    interactions = tuple(
        sorted(
            (
                _parse_interaction(_as_mapping(value, "interaction"))
                for value in _as_sequence(values, "interactions")
            ),
            key=lambda interaction: interaction.index if interaction.index is not None else 0,
        )
    )
    _validate_unique_indices(
        [interaction.index for interaction in interactions if interaction.index is not None],
        "interactions",
    )
    return interactions


def _parse_interaction(data: Mapping[str, Any]) -> MatlabRTInteraction:
    """Parse one supported MATLAB RT interaction."""
    interaction_type = _require_str(data, "Type", "interaction")
    matlab_interaction_type_to_code(interaction_type)

    return MatlabRTInteraction(
        index=_optional_int(data.get("index"), "interaction.index"),
        type=interaction_type,
        location_m=_require_vector3(data, "Location", "interaction"),
        material_name=_optional_material_name(data.get("MaterialName")),
        raw=deepcopy(dict(data)),
    )


def _validate_link_coverage(
    links: tuple[MatlabRTLink, ...],
    transmitter_by_index: Mapping[int, MatlabRTTransmitter],
    receiver_by_index: Mapping[int, MatlabRTReceiver],
) -> None:
    """Validate complete, unique ``(tx_index, rx_index)`` link coverage."""
    observed_pairs = [(link.tx_index, link.rx_index) for link in links]
    if len(observed_pairs) != len(set(observed_pairs)):
        raise MatlabRTValidationError("duplicate MATLAB RT link pairs are not allowed.")

    expected_pairs = {
        (tx_index, rx_index) for tx_index in transmitter_by_index for rx_index in receiver_by_index
    }
    observed_pair_set = set(observed_pairs)
    if observed_pair_set != expected_pairs:
        missing = sorted(expected_pairs - observed_pair_set)
        extra = sorted(observed_pair_set - expected_pairs)
        raise MatlabRTValidationError(
            f"incomplete MATLAB RT link coverage: missing={missing}, extra={extra}."
        )


def _validate_contiguous_indices(indices: list[int], context: str) -> None:
    """Require one-based contiguous indices for site records."""
    _validate_unique_indices(indices, context)
    expected = list(range(1, len(indices) + 1))
    if sorted(indices) != expected:
        raise MatlabRTValidationError(f"{context} indices must be contiguous and one-based.")


def _validate_unique_indices(indices: list[int], context: str) -> None:
    """Require unique integer indices."""
    if len(indices) != len(set(indices)):
        raise MatlabRTValidationError(f"{context} indices must be unique.")


def _require_cartesian(value: Any, context: str) -> None:
    """Reject unsupported non-Cartesian coordinate systems."""
    coordinate_system = _as_str(value, context)
    if coordinate_system.lower() != "cartesian":
        raise UnsupportedMatlabRTFeatureError(
            f"{context} must be Cartesian for MATLAB RT MVP, got {coordinate_system!r}."
        )


def _require_same_vector3(
    observed: tuple[float, float, float],
    expected: tuple[float, float, float],
    context: str,
) -> None:
    """Require two 3D position vectors to match within numeric tolerance."""
    if any(
        not math.isclose(obs, exp, rel_tol=0.0, abs_tol=1e-9)
        for obs, exp in zip(observed, expected, strict=True)
    ):
        raise MatlabRTValidationError(f"{context} is inconsistent with site position.")


def _require_mapping(data: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    """Fetch a required mapping field."""
    value = _require_key(data, key, context)
    return _as_mapping(value, f"{context}.{key}")


def _require_sequence(data: Mapping[str, Any], key: str, context: str) -> Sequence[Any]:
    """Fetch a required sequence field."""
    value = _require_key(data, key, context)
    return _as_sequence(value, f"{context}.{key}")


def _require_vector3(data: Mapping[str, Any], key: str, context: str) -> tuple[float, float, float]:
    """Fetch a required finite 3D vector."""
    return _as_float_tuple(_require_key(data, key, context), 3, f"{context}.{key}")


def _require_vector2(data: Mapping[str, Any], key: str, context: str) -> tuple[float, float]:
    """Fetch a required finite 2D vector."""
    return _as_float_tuple(_require_key(data, key, context), 2, f"{context}.{key}")


def _require_path_coordinates(
    data: Mapping[str, Any],
    key: str,
    context: str,
) -> tuple[tuple[float, float, float], ...]:
    """Fetch required finite path coordinate rows."""
    value = _require_sequence(data, key, context)
    return tuple(_as_float_tuple(row, 3, f"{context}.{key}") for row in value)


def _require_finite_float(data: Mapping[str, Any], key: str, context: str) -> float:
    """Fetch a required finite float."""
    return _as_finite_float(_require_key(data, key, context), f"{context}.{key}")


def _optional_finite_float(value: Any, context: str) -> float | None:
    """Parse an optional finite float."""
    if value is None:
        return None
    return _as_finite_float(value, context)


def _require_int(data: Mapping[str, Any], key: str, context: str) -> int:
    """Fetch a required integer."""
    return _as_int(_require_key(data, key, context), f"{context}.{key}")


def _optional_int(value: Any, context: str) -> int | None:
    """Parse an optional integer."""
    if value is None:
        return None
    return _as_int(value, context)


def _require_bool(data: Mapping[str, Any], key: str, context: str) -> bool:
    """Fetch a required boolean."""
    value = _require_key(data, key, context)
    if not isinstance(value, bool):
        raise MatlabRTSchemaError(f"{context}.{key} must be a boolean.")
    return value


def _require_str(data: Mapping[str, Any], key: str, context: str) -> str:
    """Fetch a required non-empty string."""
    return _as_str(_require_key(data, key, context), f"{context}.{key}")


def _optional_str(value: Any, context: str) -> str | None:
    """Parse an optional non-empty string."""
    if value is None:
        return None
    return _as_str(value, context)


def _optional_material_name(value: Any) -> str | None:
    """Parse an optional MATLAB material name, treating empty names as unknown."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise MatlabRTSchemaError("interaction.MaterialName must be a string when present.")
    stripped = value.strip()
    return stripped or None


def _require_key(data: Mapping[str, Any], key: str, context: str) -> Any:
    """Fetch a required key from a mapping."""
    if key not in data:
        raise MatlabRTSchemaError(f"Missing required field: {context}.{key}.")
    return data[key]


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    """Validate a mapping value."""
    if not isinstance(value, Mapping):
        raise MatlabRTSchemaError(f"{context} must be a mapping.")
    return value


def _as_sequence(value: Any, context: str) -> Sequence[Any]:
    """Validate a non-string sequence value."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MatlabRTSchemaError(f"{context} must be a sequence.")
    return value


def _as_str(value: Any, context: str) -> str:
    """Validate a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise MatlabRTSchemaError(f"{context} must be a non-empty string.")
    return value


def _as_int(value: Any, context: str) -> int:
    """Validate an integer value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise MatlabRTSchemaError(f"{context} must be an integer.")
    if value < 0:
        raise MatlabRTValidationError(f"{context} must be non-negative.")
    return value


def _as_finite_float(value: Any, context: str) -> float:
    """Validate a finite real numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MatlabRTSchemaError(f"{context} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise MatlabRTValidationError(f"{context} must be finite.")
    return result


def _as_float_tuple(value: Any, length: int, context: str) -> tuple[float, ...]:
    """Validate a finite numeric vector and return it as a tuple."""
    sequence = _as_sequence(value, context)
    if len(sequence) != length:
        raise MatlabRTValidationError(f"{context} must contain exactly {length} values.")
    return tuple(_as_finite_float(item, context) for item in sequence)

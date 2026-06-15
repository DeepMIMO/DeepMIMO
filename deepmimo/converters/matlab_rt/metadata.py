"""In-memory metadata builders for MATLAB RT DeepMIMO scenarios."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from numbers import Real
from typing import Any

from deepmimo import consts as c
from deepmimo.core.materials import Material
from deepmimo.core.txrx import TxRxSet

from .errors import MatlabRTValidationError
from .matrices import MatlabRTMatrixSet
from .schema import MatlabRTExport

RAYTRACER_NAME_MATLAB_RT = "MATLAB Ray Tracing"
TX_SET_ID = 0
RX_SET_ID = 1
MATERIAL_DEFAULT_NAME = "PEC"
MATLAB_RT_METADATA_KEY = "matlab_rt"
PLACEHOLDER_SCENE_REPRESENTATION = "empty"
MVP_EXCLUDED_FEATURES = (
    "MPLM",
    "path_hash",
    "inter_obj",
    "velocity-derived Doppler",
    "advanced scene metadata",
)
REQUIRED_MATERIAL_KEYS = frozenset(
    {
        "id",
        c.MATERIALS_PARAM_NAME_FIELD,
        c.MATERIALS_PARAM_PERMITTIVITY,
        c.MATERIALS_PARAM_CONDUCTIVITY,
        c.MATERIALS_PARAM_SCATTERING_MODEL,
        c.MATERIALS_PARAM_SCATTERING_COEF,
        c.MATERIALS_PARAM_CROSS_POL_COEF,
        "alpha_r",
        "alpha_i",
        "lambda_param",
        "roughness",
        "thickness",
        "vertical_attenuation",
        "horizontal_attenuation",
        "itu_a",
        "itu_b",
        "itu_c",
        "itu_d",
    }
)


def build_params(
    export: MatlabRTExport,
    matrix_sets: Sequence[MatlabRTMatrixSet],
    *,
    scenario_name: str = "",
    tx_power_dbw: float = 0.0,
    tx_gain_db: float = 0.0,
    rx_gain_db: float = 0.0,
) -> dict[str, Any]:
    """Build a complete in-memory ``params.json`` dictionary for the MVP."""
    _validate_export(export)
    checked_matrix_sets = _validate_matrix_sets(export, matrix_sets)

    return {
        c.VERSION_PARAM_NAME: c.VERSION,
        c.RT_PARAMS_PARAM_NAME: build_rt_params(
            export,
            checked_matrix_sets,
            tx_power_dbw=tx_power_dbw,
            tx_gain_db=tx_gain_db,
            rx_gain_db=rx_gain_db,
        ),
        c.SCENE_PARAM_NAME: build_scene_metadata(export),
        c.MATERIALS_PARAM_NAME: build_material_metadata(export),
        c.TXRX_PARAM_NAME: build_txrx_metadata(export),
        MATLAB_RT_METADATA_KEY: build_scenario_metadata(
            export,
            checked_matrix_sets,
            scenario_name=scenario_name,
        ),
    }


def build_rt_params(
    export: MatlabRTExport,
    matrix_sets: Sequence[MatlabRTMatrixSet],
    *,
    tx_power_dbw: float = 0.0,
    tx_gain_db: float = 0.0,
    rx_gain_db: float = 0.0,
) -> dict[str, Any]:
    """Build DeepMIMO ray-tracing params from a parsed MATLAB RT export."""
    _validate_export(export)
    checked_matrix_sets = _validate_matrix_sets(export, matrix_sets)
    tx_power = _finite_float(tx_power_dbw, "tx_power_dbw")
    tx_gain = _finite_float(tx_gain_db, "tx_gain_db")
    rx_gain = _finite_float(rx_gain_db, "rx_gain_db")

    max_reflections = _nonnegative_int(
        export.propagation_model.get("max_num_reflections", _max_export_interactions(export)),
        "propagation_model.max_num_reflections",
    )
    max_diffractions = _nonnegative_int(
        export.propagation_model.get("max_num_diffractions", 0),
        "propagation_model.max_num_diffractions",
    )
    max_scattering = _nonnegative_int(
        export.propagation_model.get("max_num_scattering", 0),
        "max_num_scattering",
    )
    max_transmissions = _nonnegative_int(
        export.propagation_model.get("max_num_transmissions", 0),
        "max_num_transmissions",
    )
    max_path_depth = max(
        max_reflections,
        max_diffractions,
        max_scattering,
        max_transmissions,
        _max_export_interactions(export),
        _max_matrix_interactions(checked_matrix_sets),
    )

    return {
        c.RT_PARAM_RAYTRACER: RAYTRACER_NAME_MATLAB_RT,
        c.RT_PARAM_RAYTRACER_VERSION: export.metadata.matlab_version or "unknown",
        c.RT_PARAM_FREQUENCY: _finite_float(export.scene.frequency_hz, "scene.frequency_hz"),
        c.RT_PARAM_PATH_DEPTH: max_path_depth,
        c.RT_PARAM_MAX_REFLECTIONS: max_reflections,
        c.RT_PARAM_MAX_DIFFRACTIONS: max_diffractions,
        c.RT_PARAM_MAX_SCATTERING: max_scattering,
        c.RT_PARAM_MAX_TRANSMISSIONS: max_transmissions,
        c.RT_PARAM_DIFFUSE_REFLECTIONS: 0,
        c.RT_PARAM_DIFFUSE_DIFFRACTIONS: 0,
        c.RT_PARAM_DIFFUSE_TRANSMISSIONS: 0,
        c.RT_PARAM_DIFFUSE_FINAL_ONLY: False,
        c.RT_PARAM_DIFFUSE_RANDOM_PHASES: False,
        c.RT_PARAM_TERRAIN_REFLECTION: False,
        c.RT_PARAM_TERRAIN_DIFFRACTION: False,
        c.RT_PARAM_TERRAIN_SCATTERING: False,
        c.RT_PARAM_NUM_RAYS: sum(link.num_rays for link in export.links),
        c.RT_PARAM_RAY_CASTING_METHOD: _string_or_default(
            export.propagation_model.get("method"),
            "matlab_rt",
        ),
        c.RT_PARAM_SYNTHETIC_ARRAY: True,
        c.RT_PARAM_RAY_CASTING_RANGE_AZ: 360.0,
        c.RT_PARAM_RAY_CASTING_RANGE_EL: 180.0,
        c.RT_PARAM_GPS_BBOX: [0.0, 0.0, 0.0, 0.0],
        "tx_power_dbw": tx_power,
        "tx_gain_db": tx_gain,
        "rx_gain_db": rx_gain,
        "power_policy": "power_dbw = tx_power_dbw + tx_gain_db + rx_gain_db - path_loss_db",
        "angle_policy": "MATLAB elevation maps to DeepMIMO theta as theta = 90 - elevation",
    }


def build_scene_metadata(export: MatlabRTExport) -> dict[str, Any]:
    """Build an explicit empty scene metadata block for the MVP."""
    _validate_export(export)
    return {
        c.SCENE_PARAM_NUMBER_SCENES: 1,
        c.SCENE_PARAM_N_OBJECTS: 0,
        c.SCENE_PARAM_N_VERTICES: 0,
        c.SCENE_PARAM_N_FACES: 0,
        c.SCENE_PARAM_N_TRIANGULAR_FACES: 0,
        c.SCENE_PARAM_REPRESENTATION: PLACEHOLDER_SCENE_REPRESENTATION,
        "source_coordinate_system": export.scene.coordinate_system,
        "is_placeholder": True,
        "object_rich_scene_supported": False,
    }


def build_txrx_metadata(export: MatlabRTExport) -> dict[str, dict[str, Any]]:
    """Build DeepMIMO TX/RX set metadata for one MATLAB TX set and RX set."""
    _validate_export(export)
    tx_set = TxRxSet(
        name="matlab_tx",
        id_orig=TX_SET_ID,
        id=TX_SET_ID,
        is_tx=True,
        is_rx=False,
        num_points=export.num_tx,
        num_active_points=export.num_tx,
        num_ant=1,
        dual_pol=False,
    )
    rx_set = TxRxSet(
        name="matlab_rx",
        id_orig=RX_SET_ID,
        id=RX_SET_ID,
        is_tx=False,
        is_rx=True,
        num_points=export.num_rx,
        num_active_points=export.num_rx,
        num_ant=1,
        dual_pol=False,
    )

    return {
        f"txrx_set_{TX_SET_ID}": tx_set.to_dict(),
        f"txrx_set_{RX_SET_ID}": rx_set.to_dict(),
    }


def build_material_metadata(export: MatlabRTExport) -> dict[str, dict[str, Any]]:
    """Build placeholder material records with all DeepMIMO summary keys."""
    _validate_export(export)
    names = _material_names_from_export(export)
    materials = {
        f"material_{index}": default_material_record(index=index, name=name)
        for index, name in enumerate(names)
    }
    validate_material_metadata(materials)
    return materials


def default_material_record(*, index: int = 0, name: str = MATERIAL_DEFAULT_NAME) -> dict[str, Any]:
    """Return a complete placeholder material record for MATLAB RT MVP output."""
    material_index = _nonnegative_int(index, "material index")
    material_name = _string_or_default(name, MATERIAL_DEFAULT_NAME)
    return asdict(
        Material(
            id=material_index,
            name=material_name,
            permittivity=1.0,
            conductivity=0.0,
        )
    )


def validate_material_metadata(materials: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate material records include keys needed by loading and summary."""
    if not isinstance(materials, Mapping) or not materials:
        raise MatlabRTValidationError("materials must be a non-empty mapping.")

    for material_key, material in materials.items():
        if not isinstance(material_key, str) or not material_key:
            raise MatlabRTValidationError("material keys must be non-empty strings.")
        if not isinstance(material, Mapping):
            raise MatlabRTValidationError(f"{material_key} must be a mapping.")

        missing = REQUIRED_MATERIAL_KEYS - set(material)
        if missing:
            raise MatlabRTValidationError(
                f"{material_key} is missing required material keys: {sorted(missing)}."
            )


def build_scenario_metadata(
    export: MatlabRTExport,
    matrix_sets: Sequence[MatlabRTMatrixSet],
    *,
    scenario_name: str = "",
) -> dict[str, Any]:
    """Build MATLAB-RT-specific metadata ignored by DeepMIMO core loaders."""
    _validate_export(export)
    checked_matrix_sets = _validate_matrix_sets(export, matrix_sets)

    return {
        "scenario_name": scenario_name,
        "source": "matlab_rt_json",
        "experiment": export.metadata.experiment,
        "description": export.metadata.description,
        "coordinate_system": export.scene.coordinate_system,
        "source_path": str(export.source_path) if export.source_path is not None else None,
        "mvp_scope": "file-based MATLAB RT JSON conversion",
        "excluded_features": list(MVP_EXCLUDED_FEATURES),
        "path_counts": {
            f"tx{matrix_set.tx_index}": list(matrix_set.path_counts)
            for matrix_set in checked_matrix_sets
        },
        "matrix_shapes": {
            f"tx{matrix_set.tx_index}": {
                key: list(value.shape)
                for key, value in matrix_set.matrices.items()
            }
            for matrix_set in checked_matrix_sets
        },
    }


def _validate_export(export: MatlabRTExport) -> None:
    """Validate metadata builder input type."""
    if not isinstance(export, MatlabRTExport):
        raise TypeError("export must be a MatlabRTExport instance.")


def _validate_matrix_sets(
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
                f"matrix set tx{matrix_set.tx_index} receiver count {matrix_set.scalar_shape[0]} "
                f"does not match export.num_rx={export.num_rx}."
            )
        if len(matrix_set.path_counts) != export.num_rx:
            raise MatlabRTValidationError(
                f"matrix set tx{matrix_set.tx_index} path_counts length does not match "
                "export.num_rx."
            )

    return checked


def _material_names_from_export(export: MatlabRTExport) -> tuple[str, ...]:
    """Return deterministic material names referenced by MATLAB interactions."""
    names = sorted(
        {
            interaction.material_name.strip()
            for link in export.links
            for ray in link.rays
            for interaction in ray.interactions
            if interaction.material_name and interaction.material_name.strip()
        }
    )
    if not names:
        return (MATERIAL_DEFAULT_NAME,)
    return tuple(names)


def _max_export_interactions(export: MatlabRTExport) -> int:
    """Return maximum parsed interaction count across all rays."""
    return max((ray.num_interactions for link in export.links for ray in link.rays), default=0)


def _max_matrix_interactions(matrix_sets: Sequence[MatlabRTMatrixSet]) -> int:
    """Return maximum assembled interaction depth across matrix sets."""
    return max((matrix_set.inter_pos_shape[2] for matrix_set in matrix_sets), default=0)


def _nonnegative_int(value: Any, name: str) -> int:
    """Validate a non-negative integer-like metadata value."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise MatlabRTValidationError(f"{name} must be a non-negative integer.")
    if value < 0:
        raise MatlabRTValidationError(f"{name} must be non-negative.")
    return value


def _finite_float(value: Any, name: str) -> float:
    """Validate a finite numeric metadata value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MatlabRTValidationError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise MatlabRTValidationError(f"{name} must be finite.")
    return result


def _string_or_default(value: Any, default: str) -> str:
    """Return a non-empty string value or an explicit default."""
    if value is None:
        return default
    if not isinstance(value, str):
        raise MatlabRTValidationError("metadata string values must be strings.")
    stripped = value.strip()
    return stripped if stripped else default

"""Ray-tracing parameter extraction for MATLAB RT JSON exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepmimo import consts as c

from ._metadata_common import (
    RAYTRACER_NAME_MATLAB_RT,
    finite_float,
    max_export_interactions,
    max_matrix_interactions,
    nonnegative_int,
    string_or_default,
    validate_export,
    validate_matrix_sets,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .matrices import MatlabRTMatrixSet
    from .schema import MatlabRTExport


def build_rt_params(
    export: MatlabRTExport,
    matrix_sets: Sequence[MatlabRTMatrixSet],
    *,
    tx_power_dbw: float = 0.0,
    tx_gain_db: float = 0.0,
    rx_gain_db: float = 0.0,
) -> dict[str, Any]:
    """Build DeepMIMO ray-tracing params from a parsed MATLAB RT export."""
    validate_export(export)
    checked_matrix_sets = validate_matrix_sets(export, matrix_sets)
    tx_power = finite_float(tx_power_dbw, "tx_power_dbw")
    tx_gain = finite_float(tx_gain_db, "tx_gain_db")
    rx_gain = finite_float(rx_gain_db, "rx_gain_db")

    max_reflections = nonnegative_int(
        export.propagation_model.get("max_num_reflections", max_export_interactions(export)),
        "propagation_model.max_num_reflections",
    )
    max_diffractions = nonnegative_int(
        export.propagation_model.get("max_num_diffractions", 0),
        "propagation_model.max_num_diffractions",
    )
    max_scattering = nonnegative_int(
        export.propagation_model.get("max_num_scattering", 0),
        "max_num_scattering",
    )
    max_transmissions = nonnegative_int(
        export.propagation_model.get("max_num_transmissions", 0),
        "max_num_transmissions",
    )
    max_path_depth = max(
        max_reflections,
        max_diffractions,
        max_scattering,
        max_transmissions,
        max_export_interactions(export),
        max_matrix_interactions(checked_matrix_sets),
    )

    return {
        c.RT_PARAM_RAYTRACER: RAYTRACER_NAME_MATLAB_RT,
        c.RT_PARAM_RAYTRACER_VERSION: export.metadata.matlab_version or "unknown",
        c.RT_PARAM_FREQUENCY: finite_float(export.scene.frequency_hz, "scene.frequency_hz"),
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
        c.RT_PARAM_RAY_CASTING_METHOD: string_or_default(
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

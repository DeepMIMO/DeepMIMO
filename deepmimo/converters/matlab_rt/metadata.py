# ruff: noqa: PLR0913
"""In-memory metadata orchestration for MATLAB RT DeepMIMO scenarios."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepmimo import consts as c

from ._metadata_common import (
    MATERIAL_DEFAULT_NAME,
    PLACEHOLDER_SCENE_REPRESENTATION,
    RAYTRACER_NAME_MATLAB_RT,
    RX_SET_ID,
    TX_SET_ID,
    validate_export,
    validate_matrix_sets,
)
from .matlab_rt_materials import (
    REQUIRED_MATERIAL_KEYS,
    build_material_metadata,
    default_material_record,
    validate_material_metadata,
)
from .matlab_rt_rt_params import build_rt_params
from .matlab_rt_scene import build_scene_metadata
from .matlab_rt_txrx import build_txrx_metadata

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .matrices import MatlabRTMatrixSet
    from .schema import MatlabRTExport

MATLAB_RT_METADATA_KEY = "matlab_rt"
MVP_EXCLUDED_FEATURES = (
    "MPLM",
    "path_hash",
    "inter_obj",
    "velocity-derived Doppler",
    "advanced scene object/material metadata",
)

__all__ = [
    "MATERIAL_DEFAULT_NAME",
    "MATLAB_RT_METADATA_KEY",
    "MVP_EXCLUDED_FEATURES",
    "PLACEHOLDER_SCENE_REPRESENTATION",
    "RAYTRACER_NAME_MATLAB_RT",
    "REQUIRED_MATERIAL_KEYS",
    "RX_SET_ID",
    "TX_SET_ID",
    "build_material_metadata",
    "build_params",
    "build_rt_params",
    "build_scenario_metadata",
    "build_scene_metadata",
    "build_txrx_metadata",
    "default_material_record",
    "validate_material_metadata",
]


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
    validate_export(export)
    checked_matrix_sets = validate_matrix_sets(export, matrix_sets)

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


def build_scenario_metadata(
    export: MatlabRTExport,
    matrix_sets: Sequence[MatlabRTMatrixSet],
    *,
    scenario_name: str = "",
) -> dict[str, Any]:
    """Build MATLAB-RT-specific metadata ignored by DeepMIMO core loaders."""
    validate_export(export)
    checked_matrix_sets = validate_matrix_sets(export, matrix_sets)

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
                key: list(value.shape) for key, value in matrix_set.matrices.items()
            }
            for matrix_set in checked_matrix_sets
        },
    }

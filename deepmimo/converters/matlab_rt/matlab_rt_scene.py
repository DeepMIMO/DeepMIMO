"""Scene metadata extraction for MATLAB RT JSON exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepmimo import consts as c

from ._metadata_common import PLACEHOLDER_SCENE_REPRESENTATION, validate_export

if TYPE_CHECKING:
    from .schema import MatlabRTExport


def build_scene_metadata(export: MatlabRTExport) -> dict[str, Any]:
    """Build an explicit empty scene metadata block for the MVP."""
    validate_export(export)
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

# ruff: noqa: EM101, EM102, TRY003
"""Material metadata extraction for MATLAB RT JSON exports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from deepmimo import consts as c
from deepmimo.core.materials import Material

from ._metadata_common import (
    MATERIAL_DEFAULT_NAME,
    nonnegative_int,
    string_or_default,
    validate_export,
)
from .errors import MatlabRTValidationError

if TYPE_CHECKING:
    from .schema import MatlabRTExport

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


def build_material_metadata(export: MatlabRTExport) -> dict[str, dict[str, Any]]:
    """Build placeholder material records with all DeepMIMO summary keys."""
    validate_export(export)
    names = _material_names_from_export(export)
    materials = {
        f"material_{index}": default_material_record(index=index, name=name)
        for index, name in enumerate(names)
    }
    validate_material_metadata(materials)
    return materials


def default_material_record(*, index: int = 0, name: str = MATERIAL_DEFAULT_NAME) -> dict[str, Any]:
    """Return a complete placeholder material record for MATLAB RT MVP output."""
    material_index = nonnegative_int(index, "material index")
    material_name = string_or_default(name, MATERIAL_DEFAULT_NAME)
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

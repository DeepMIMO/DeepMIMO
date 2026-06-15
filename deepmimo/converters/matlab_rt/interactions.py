"""Pure interaction helpers for MATLAB RT exports."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from .errors import UnsupportedMatlabRTFeatureError


DEEPMIMO_LOS_CODE = 0
DEEPMIMO_REFLECTION_CODE = 1
MATLAB_REFLECTION_TYPE = "reflection"


def _as_interaction_sequence(interactions: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    """Validate that a value is a sequence of interaction mappings."""
    if isinstance(interactions, (str, bytes)) or not isinstance(interactions, Sequence):
        raise TypeError("interactions must be a sequence of mappings.")

    for interaction in interactions:
        if not isinstance(interaction, Mapping):
            raise TypeError("each interaction must be a mapping.")

    return interactions


def _as_finite_float(value: Any, name: str) -> float:
    """Convert a real numeric value to ``float`` and reject invalid values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric value.")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")

    return result


def _validate_max_interactions(max_interactions: int) -> int:
    """Validate a requested interaction packing depth."""
    if isinstance(max_interactions, bool) or not isinstance(max_interactions, Integral):
        raise TypeError("max_interactions must be an integer.")
    if max_interactions < 0:
        raise ValueError("max_interactions must be non-negative.")
    return int(max_interactions)


def matlab_interaction_type_to_code(interaction_type: str) -> int:
    """Map a MATLAB RT interaction type to a DeepMIMO interaction code."""
    if not isinstance(interaction_type, str) or not interaction_type.strip():
        raise TypeError("interaction_type must be a non-empty string.")

    normalized_type = interaction_type.strip().lower()
    if normalized_type == MATLAB_REFLECTION_TYPE:
        return DEEPMIMO_REFLECTION_CODE

    raise UnsupportedMatlabRTFeatureError(
        f"Unsupported MATLAB RT interaction type for MVP: {interaction_type!r}."
    )


def matlab_interaction_code(interaction: Mapping[str, Any]) -> int:
    """Return the DeepMIMO code for one MATLAB RT interaction mapping."""
    if not isinstance(interaction, Mapping):
        raise TypeError("interaction must be a mapping.")

    return matlab_interaction_type_to_code(interaction.get("Type"))


def matlab_interaction_location(interaction: Mapping[str, Any]) -> np.ndarray:
    """Extract one MATLAB RT interaction location as a finite ``[3]`` array."""
    if not isinstance(interaction, Mapping):
        raise TypeError("interaction must be a mapping.")

    matlab_interaction_code(interaction)

    location = interaction.get("Location")
    if isinstance(location, (str, bytes)) or not isinstance(location, Sequence):
        raise TypeError("interaction Location must be a numeric sequence with three values.")
    if len(location) != 3:
        raise ValueError("interaction Location must contain exactly three values.")

    return np.array(
        [_as_finite_float(value, "interaction Location") for value in location],
        dtype=float,
    )


def matlab_interaction_sequence_code(interactions: Sequence[Mapping[str, Any]]) -> int:
    """Encode a MATLAB RT interaction sequence as a DeepMIMO decimal path code."""
    checked_interactions = _as_interaction_sequence(interactions)
    if not checked_interactions:
        return DEEPMIMO_LOS_CODE

    digits = [str(matlab_interaction_code(interaction)) for interaction in checked_interactions]
    return int("".join(digits))


def matlab_ray_interaction_code(ray: Mapping[str, Any]) -> int:
    """Return the DeepMIMO interaction code for a MATLAB ray-like mapping."""
    if not isinstance(ray, Mapping):
        raise TypeError("ray must be a mapping.")

    interactions = ray.get("interactions", [])
    if bool(ray.get("line_of_sight", False)) and not interactions:
        return DEEPMIMO_LOS_CODE

    return matlab_interaction_sequence_code(interactions)


def extract_interaction_positions(interactions: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Extract supported MATLAB interaction positions with shape ``[n, 3]``."""
    checked_interactions = _as_interaction_sequence(interactions)
    if not checked_interactions:
        return np.empty((0, 3), dtype=float)

    return np.vstack(
        [matlab_interaction_location(interaction) for interaction in checked_interactions]
    )


def pack_interaction_positions(
    interactions: Sequence[Mapping[str, Any]],
    *,
    max_interactions: int | None = None,
) -> np.ndarray:
    """Pack interaction positions for a future DeepMIMO ``inter_pos`` path slot."""
    positions = extract_interaction_positions(interactions)
    if max_interactions is None:
        return positions

    packing_depth = _validate_max_interactions(max_interactions)
    if positions.shape[0] > packing_depth:
        raise ValueError(
            "interaction count exceeds max_interactions: "
            f"{positions.shape[0]} > {packing_depth}."
        )

    packed = np.full((packing_depth, 3), np.nan, dtype=float)
    if positions.size:
        packed[: positions.shape[0], :] = positions

    return packed

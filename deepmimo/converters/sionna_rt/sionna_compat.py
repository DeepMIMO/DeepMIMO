"""Compatibility helpers for reading Sionna RT exports.

Sionna 2.x holds scalars as Dr.Jit arrays, so ``Scene.frequency`` and every
material property reach the exporter with shape ``(1,)`` and are pickled that
way. NumPy 2 removed the implicit conversion of a one-element array to a Python
number, so ``float(...)`` on those values raises. These helpers accept either
form, which keeps folders exported by earlier versions readable.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def as_scalar(value: Any, default: float | None = None) -> float:
    """Coerce a Sionna-exported value to a Python float.

    Args:
        value: Scalar, one-element sequence, or None.
        default: Value to return when ``value`` is None. If None, a None input
            raises instead.

    Returns:
        The value as a float.

    Raises:
        TypeError: If ``value`` is None and no default was given.

    """
    if value is None:
        if default is None:
            msg = "Expected a numeric value, got None."
            raise TypeError(msg)
        return float(default)
    array = np.asarray(value).reshape(-1)
    return float(array[0])

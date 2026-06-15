"""Pure numeric conversion helpers for MATLAB RT exports."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any


def _as_finite_float(value: Any, name: str) -> float:
    """Convert a real numeric value to ``float`` and reject invalid values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric value.")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")

    return result


def normalize_signed_deg(angle_deg: float) -> float:
    """Normalize an angle in degrees to the signed ``[-180, 180]`` interval."""
    angle = _as_finite_float(angle_deg, "angle_deg")
    normalized = (angle + 180.0) % 360.0 - 180.0

    if math.isclose(normalized, -180.0, abs_tol=1e-12) and angle > 0.0:
        return 180.0
    if math.isclose(normalized, 0.0, abs_tol=1e-12):
        return 0.0

    return normalized


def matlab_elevation_to_deepmimo_theta(elevation_deg: float) -> float:
    """Convert MATLAB elevation from the xy-plane to DeepMIMO polar theta."""
    elevation = _as_finite_float(elevation_deg, "elevation_deg")
    return 90.0 - elevation


def matlab_path_loss_to_power_dbw(
    path_loss_db: float,
    *,
    tx_power_dbw: float = 0.0,
    tx_gain_db: float = 0.0,
    rx_gain_db: float = 0.0,
) -> float:
    """Convert MATLAB path loss in dB to DeepMIMO received power in dBW."""
    path_loss = _as_finite_float(path_loss_db, "path_loss_db")
    tx_power = _as_finite_float(tx_power_dbw, "tx_power_dbw")
    tx_gain = _as_finite_float(tx_gain_db, "tx_gain_db")
    rx_gain = _as_finite_float(rx_gain_db, "rx_gain_db")

    return tx_power + tx_gain + rx_gain - path_loss


def matlab_phase_rad_to_deg(phase_shift_rad: float) -> float:
    """Convert MATLAB phase shift in radians to degrees."""
    phase_rad = _as_finite_float(phase_shift_rad, "phase_shift_rad")
    return math.degrees(phase_rad)


def matlab_phase_to_deg(
    *,
    phase_shift_deg: float | None = None,
    phase_shift_rad: float | None = None,
) -> float:
    """Return MATLAB phase in degrees, preferring degree input when present."""
    if phase_shift_deg is not None:
        return _as_finite_float(phase_shift_deg, "phase_shift_deg")
    if phase_shift_rad is not None:
        return matlab_phase_rad_to_deg(phase_shift_rad)

    raise ValueError("MATLAB phase requires phase_shift_deg or phase_shift_rad.")


def matlab_ray_phase_to_deg(ray: Mapping[str, Any]) -> float:
    """Extract phase from a MATLAB ray-like mapping and return degrees."""
    if not isinstance(ray, Mapping):
        raise TypeError("ray must be a mapping.")

    if "phase_shift_deg" in ray and ray["phase_shift_deg"] is not None:
        return matlab_phase_to_deg(phase_shift_deg=ray["phase_shift_deg"])
    if "phase_shift_rad" in ray and ray["phase_shift_rad"] is not None:
        return matlab_phase_to_deg(phase_shift_rad=ray["phase_shift_rad"])

    raise ValueError("MATLAB ray requires phase_shift_deg or phase_shift_rad.")

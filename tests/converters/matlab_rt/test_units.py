"""Tests for MATLAB RT pure unit conversion helpers."""

import math
import unittest

from deepmimo.converters.matlab_rt.units import (
    matlab_elevation_to_deepmimo_theta,
    matlab_path_loss_to_power_dbw,
    matlab_phase_rad_to_deg,
    matlab_phase_to_deg,
    matlab_ray_phase_to_deg,
    normalize_signed_deg,
)
from tests.converters.matlab_rt import expect_raises


class TestMatlabRTUnits(unittest.TestCase):
    """Validate deterministic, side-effect-free unit conversions."""

    def test_normalize_signed_deg_required_values(self) -> None:
        """Normalize representative MATLAB azimuth values."""
        assert normalize_signed_deg(0) == 0.0
        assert normalize_signed_deg(180) == 180.0
        assert normalize_signed_deg(-180) == -180.0
        assert normalize_signed_deg(360) == 0.0
        assert normalize_signed_deg(-45) == -45.0

    def test_normalize_signed_deg_rejects_invalid_values(self) -> None:
        """Reject non-finite or non-numeric angles."""
        with expect_raises(ValueError):
            normalize_signed_deg(float("nan"))
        with expect_raises(ValueError):
            normalize_signed_deg(float("inf"))
        with expect_raises(TypeError):
            normalize_signed_deg("0")

    def test_matlab_elevation_to_deepmimo_theta_required_values(self) -> None:
        """Convert MATLAB elevation to DeepMIMO theta."""
        assert matlab_elevation_to_deepmimo_theta(0) == 90.0
        assert matlab_elevation_to_deepmimo_theta(90) == 0.0
        assert matlab_elevation_to_deepmimo_theta(-90) == 180.0

    def test_matlab_path_loss_to_power_dbw_required_values(self) -> None:
        """Convert path loss to received power under explicit gain policies."""
        assert matlab_path_loss_to_power_dbw(63.0) == -63.0
        assert matlab_path_loss_to_power_dbw(63.0, tx_gain_db=3.0) == -60.0
        assert matlab_path_loss_to_power_dbw(63.0, rx_gain_db=2.0) == -61.0
        assert (
            matlab_path_loss_to_power_dbw(
                63.0,
                tx_power_dbw=1.0,
                tx_gain_db=3.0,
                rx_gain_db=2.0,
            )
            == -57.0
        )

    def test_matlab_path_loss_to_power_dbw_rejects_invalid_values(self) -> None:
        """Reject invalid path-loss and gain values."""
        with expect_raises(ValueError):
            matlab_path_loss_to_power_dbw(float("nan"))
        with expect_raises(TypeError):
            matlab_path_loss_to_power_dbw(63.0, tx_gain_db="3")

    def test_matlab_phase_to_deg_uses_degree_input(self) -> None:
        """Use MATLAB degree phase directly when available."""
        assert matlab_phase_to_deg(phase_shift_deg=269.5) == 269.5
        assert matlab_phase_to_deg(phase_shift_deg=10.0, phase_shift_rad=math.pi) == 10.0

    def test_matlab_phase_rad_to_deg_and_fallback(self) -> None:
        """Convert radian phase when degree phase is unavailable."""
        assert math.isclose(matlab_phase_rad_to_deg(math.pi / 2), 90.0)
        assert math.isclose(matlab_phase_to_deg(phase_shift_rad=math.pi), 180.0)
        assert math.isclose(matlab_ray_phase_to_deg({"phase_shift_rad": math.pi / 4}), 45.0)

    def test_matlab_ray_phase_prefers_degree_input(self) -> None:
        """Extract phase from ray-like mappings with degree priority."""
        assert (
            matlab_ray_phase_to_deg({"phase_shift_deg": 38.0, "phase_shift_rad": math.pi}) == 38.0
        )

    def test_matlab_phase_invalid_input_behavior(self) -> None:
        """Reject missing, non-finite, or non-mapping phase inputs."""
        with expect_raises(ValueError):
            matlab_phase_to_deg()
        with expect_raises(ValueError):
            matlab_phase_to_deg(phase_shift_deg=float("nan"))
        with expect_raises(TypeError):
            matlab_phase_to_deg(phase_shift_deg="90")
        with expect_raises(ValueError):
            matlab_ray_phase_to_deg({})
        with expect_raises(TypeError):
            matlab_ray_phase_to_deg(["phase_shift_deg", 90.0])


if __name__ == "__main__":
    unittest.main()

"""Antenna patterns module for DeepMIMO.

This module provides antenna radiation pattern modeling capabilities for the DeepMIMO
dataset generator. It implements various antenna types and their radiation patterns,
including:
- Omnidirectional patterns
- Half-wave dipole patterns

The module provides a unified interface for applying antenna patterns to signal power
calculations in MIMO channel generation.
"""

# Third-party imports
import numpy as np

# Local imports
from deepmimo import consts as c


def _pattern_isotropic(_theta: np.ndarray, _phi: np.ndarray) -> np.ndarray:
    """Compute isotropic antenna pattern.

    Args:
        theta (np.ndarray): Theta angles in radians.
        phi (np.ndarray): Phi angles in radians.

    Returns:
        float: Unit gain (1.0) regardless of input angles.

    """
    return 1.0


EPS_ANGLE = 1e-10


def _pattern_halfwave_dipole(theta: np.ndarray, _phi: np.ndarray) -> np.ndarray:
    """Compute half-wave dipole antenna pattern.

    This function implements the theoretical radiation pattern of a half-wave
    dipole antenna, including its characteristic figure-8 shape.
    The pattern follows the formula: G(θ) = 1.643 * [cos(π/2 * cos(θ))]²/sin(θ)
    where θ is measured from the dipole axis.

    Reference: Balanis, C.A. "Antenna Theory: Analysis and Design", 4th Edition

    Args:
        theta (np.ndarray): Theta angles in radians.
        phi (np.ndarray): Phi angles in radians.

    Returns:
        np.ndarray: Antenna gain pattern for given angles.

    """
    max_gain = 1.643  # Half-wave dipole maximum directivity

    # Convert to numpy array if not already
    theta = np.asarray(theta)

    # Initialize pattern array
    pattern = np.zeros_like(theta, dtype=np.float64)

    # Handle valid angles (not near 0 or π)
    valid_angles = np.abs(np.sin(theta)) > EPS_ANGLE

    # Calculate the pattern using the standard dipole formula
    # Pre-compute terms for better performance
    theta_valid = theta[valid_angles]
    sin_theta = np.sin(theta_valid)
    cos_term = np.cos(np.pi / 2 * np.cos(theta_valid))

    # Apply the formula: G(θ) = max_gain * [cos(π/2 * cos(θ))]²/sin²(θ)
    pattern[valid_angles] = max_gain * (cos_term**2 / sin_theta)

    return pattern


# Pattern registry mapping pattern names to their functions
PATTERN_REGISTRY = {
    "isotropic": _pattern_isotropic,
    "halfwave-dipole": _pattern_halfwave_dipole,
}


class AntennaPattern:
    """Class for handling antenna radiation patterns.

    This class manages the radiation patterns for both TX and RX antennas,
    providing a unified interface for pattern application in signal power
    calculations.

    Attributes:
        tx_pattern_fn (Optional[Callable]): Function implementing TX antenna pattern.
        rx_pattern_fn (Optional[Callable]): Function implementing RX antenna pattern.

    """

    def __init__(self, tx_pattern: str, rx_pattern: str) -> None:
        """Initialize antenna patterns for transmitter and receiver.

        Args:
            tx_pattern (str): Transmitter antenna pattern type from PARAMSET_ANT_RAD_PAT_VALS.
            rx_pattern (str): Receiver antenna pattern type from PARAMSET_ANT_RAD_PAT_VALS.

        Raises:
            NotImplementedError: If specified pattern type is not supported.

        """
        self.tx_pattern_fn = self._load_pattern(tx_pattern, "TX")
        self.rx_pattern_fn = self._load_pattern(rx_pattern, "RX")

    def _load_pattern(self, pattern: str, pattern_type: str) -> callable:
        """Load antenna pattern function from registry.

        Args:
            pattern (str): Pattern type from PARAMSET_ANT_RAD_PAT_VALS.
            pattern_type (str): Either "TX" or "RX" for error messages.

        Returns:
            callable: Pattern function to use.

        Raises:
            NotImplementedError: If pattern is not supported or not implemented.

        """
        if pattern not in c.PARAMSET_ANT_RAD_PAT_VALS:
            msg = (
                f"The '{pattern}' antenna radiation pattern is not applicable for "
                f"{pattern_type}."
            )
            raise NotImplementedError(msg)
        if pattern not in PATTERN_REGISTRY:
            msg = f"The pattern '{pattern}' is defined but not implemented for {pattern_type}."
            raise NotImplementedError(
                msg,
            )
        return PATTERN_REGISTRY[pattern]

    def apply(
        self,
        power: np.ndarray,
        aoa_theta: np.ndarray,
        aoa_phi: np.ndarray,
        aod_theta: np.ndarray,
        aod_phi: np.ndarray,
    ) -> np.ndarray:
        """Apply antenna patterns to input power.

        This function applies both TX and RX antenna patterns to modify the input
        power values based on arrival and departure angles.

        Args:
            power (np.ndarray): Input power values.
            aoa_theta (np.ndarray): Angle of arrival theta angles in radians.
            aoa_phi (np.ndarray): Angle of arrival phi angles in radians.
            aod_theta (np.ndarray): Angle of departure theta angles in radians.
            aod_phi (np.ndarray): Angle of departure phi angles in radians.

        Returns:
            np.ndarray: Modified power values after applying antenna patterns.

        """
        pattern = self.tx_pattern_fn(aod_theta, aod_phi) * self.rx_pattern_fn(aoa_theta, aoa_phi)
        return power * pattern

    def apply_batch(
        self,
        power: np.ndarray,
        aoa_theta: np.ndarray,
        aoa_phi: np.ndarray,
        aod_theta: np.ndarray,
        aod_phi: np.ndarray,
    ) -> np.ndarray:
        """Apply antenna patterns to powers in batch.

        Args:
            power (np.ndarray): Powers array with shape (n_users, n_paths)
            aoa_theta (np.ndarray): Angle of arrival elevation angles (n_users, n_paths)
            aoa_phi (np.ndarray): Angle of arrival azimuth angles (n_users, n_paths)
            aod_theta (np.ndarray): Angle of departure elevation angles (n_users, n_paths)
            aod_phi (np.ndarray): Angle of departure azimuth angles (n_users, n_paths)

        Returns:
            np.ndarray: Modified powers with antenna patterns applied (n_users, n_paths)

        """
        # Reshape inputs to 2D if they're 1D
        if power.ndim == 1:
            power = power.reshape(1, -1)
            aoa_theta = aoa_theta.reshape(1, -1)
            aoa_phi = aoa_phi.reshape(1, -1)
            aod_theta = aod_theta.reshape(1, -1)
            aod_phi = aod_phi.reshape(1, -1)

        pattern = self.tx_pattern_fn(aod_theta, aod_phi) * self.rx_pattern_fn(aoa_theta, aoa_phi)
        return power * pattern

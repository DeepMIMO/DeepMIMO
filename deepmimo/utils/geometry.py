"""Geometry utilities for DeepMIMO.

This module provides functions for coordinate transformations and geometric calculations.
"""

import numpy as np


def cartesian_to_spherical(cartesian_coords: np.ndarray) -> np.ndarray:
    """Convert Cartesian coordinates to spherical coordinates.

    Args:
        cartesian_coords: Array [n_points, 3] of Cartesian coordinates (x, y, z)

    Returns:
        Array [n_points, 3] of spherical coordinates (r, azimuth, elevation) in radians.
        r is the magnitude (distance from origin).

    """
    spherical_coords = np.zeros((cartesian_coords.shape[0], 3))
    spherical_coords[:, 0] = np.sqrt(np.sum(cartesian_coords**2, axis=1))
    spherical_coords[:, 1] = np.arctan2(cartesian_coords[:, 1], cartesian_coords[:, 0])
    r_xy = np.sqrt(cartesian_coords[:, 0] ** 2 + cartesian_coords[:, 1] ** 2)
    spherical_coords[:, 2] = np.arctan2(cartesian_coords[:, 2], r_xy)
    return spherical_coords


def spherical_to_cartesian(spherical_coords: np.ndarray) -> np.ndarray:
    """Convert spherical coordinates to Cartesian coordinates.

    Args:
        spherical_coords: Array with spherical coordinates (r, elevation, azimuth) in radians.
            r is the magnitude (distance from origin). Leading dimensions allowed; last
            dimension must be 3.
            Reference: https://en.wikipedia.org/wiki/Spherical_coordinate_system
            Note: DeepMIMO uses elevation from the xy plane, while Sionna/Wikipedia use z-axis.

    Returns:
        Array of same shape containing Cartesian coordinates (x, y, z).

    """
    cartesian_coords = np.zeros_like(spherical_coords)
    r = spherical_coords[..., 0]
    elevation = spherical_coords[..., 1]
    azimuth = spherical_coords[..., 2]
    cartesian_coords[..., 0] = r * np.sin(elevation) * np.cos(azimuth)
    cartesian_coords[..., 1] = r * np.sin(elevation) * np.sin(azimuth)
    cartesian_coords[..., 2] = r * np.cos(elevation)
    return cartesian_coords


"""Geometry utilities for DeepMIMO.

This module provides functions for coordinate transformations and geometric calculations.

Angle convention
----------------
Both conversions below use the DeepMIMO spherical convention documented in
``docs/resources/conventions.md``, matching the dataset angle arrays
(``aoa_el``/``aod_el`` and ``aoa_az``/``aod_az``) and the array-response code in
``deepmimo.generator.geometry``:

- ``theta`` is the polar (inclination) angle measured from +z: 0 at +z, pi/2 in the
  xy-plane, pi at -z. Despite the historical ``_el`` suffix on the dataset arrays, this
  is *not* an elevation measured up from the horizon.
- ``phi`` is the azimuth measured counter-clockwise from +x in the xy-plane.

Both functions order their triples as ``(r, theta, phi)`` so that they are exact
inverses of one another.
"""

import numpy as np


def cartesian_to_spherical(cartesian_coords: np.ndarray) -> np.ndarray:
    """Convert Cartesian coordinates to spherical coordinates.

    Args:
        cartesian_coords: Array of Cartesian coordinates (x, y, z). Leading dimensions
            are allowed; the last dimension must be 3.

    Returns:
        Array of the same shape containing spherical coordinates (r, theta, phi), with
        the angles in radians. ``r`` is the distance from the origin, ``theta`` is the
        polar angle from +z in [0, pi] and ``phi`` is the azimuth from +x in (-pi, pi].
        This is the exact inverse of :func:`spherical_to_cartesian`.

    Note:
        This is a behavior change: the returned triple is now ordered ``(r, theta, phi)``.
        It was previously ``(r, phi, elevation_from_xy_plane)``, which was neither the
        library's documented angle convention nor the inverse of
        :func:`spherical_to_cartesian` (see issue #63).

    """
    cartesian_coords = np.asarray(cartesian_coords, dtype=float)
    x = cartesian_coords[..., 0]
    y = cartesian_coords[..., 1]
    z = cartesian_coords[..., 2]

    r_xy = np.hypot(x, y)
    spherical_coords = np.zeros_like(cartesian_coords)
    spherical_coords[..., 0] = np.hypot(r_xy, z)
    spherical_coords[..., 1] = np.arctan2(r_xy, z)
    spherical_coords[..., 2] = np.arctan2(y, x)
    return spherical_coords


def spherical_to_cartesian(spherical_coords: np.ndarray) -> np.ndarray:
    """Convert spherical coordinates to Cartesian coordinates.

    Args:
        spherical_coords: Array with spherical coordinates (r, theta, phi), with the
            angles in radians. ``r`` is the distance from the origin, ``theta`` is the
            polar angle from +z and ``phi`` is the azimuth from +x. Leading dimensions
            are allowed; the last dimension must be 3.
            Reference: https://en.wikipedia.org/wiki/Spherical_coordinate_system

    Returns:
        Array of the same shape containing Cartesian coordinates (x, y, z).

    """
    spherical_coords = np.asarray(spherical_coords, dtype=float)
    r = spherical_coords[..., 0]
    theta = spherical_coords[..., 1]
    phi = spherical_coords[..., 2]

    cartesian_coords = np.zeros_like(spherical_coords)
    cartesian_coords[..., 0] = r * np.sin(theta) * np.cos(phi)
    cartesian_coords[..., 1] = r * np.sin(theta) * np.sin(phi)
    cartesian_coords[..., 2] = r * np.cos(theta)
    return cartesian_coords

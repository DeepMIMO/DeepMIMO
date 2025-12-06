"""Utilities Module for DeepMIMO Dataset Processing.

This module provides utility functions and classes for processing DeepMIMO datasets,
including:
- Unit conversions (dBW to Watts)
- Array steering vector calculations
- Path analysis and feature extraction
- Position sampling and filtering utilities

The module serves as a collection of helper functions used throughout the DeepMIMO
dataset generation process.
"""

# Standard library imports

# Third-party imports
import numpy as np

################################## For User ###################################


def get_linear_idxs(
    rx_pos: np.ndarray,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    n_steps: int,
    filter_repeated: bool = True,
) -> np.ndarray:
    """Return indices of users along a linear path between two points.

    Args:
        rx_pos (np.ndarray): Positions of dataset points
        start_pos (np.ndarray): Starting position coordinates (2D or 3D)
        end_pos (np.ndarray): Ending position coordinates (2D or 3D)
        n_steps (int): Number of steps along the path (default: 100)
        filter_repeated (bool): Whether to filter repeated positions (default: True)

    Returns:
        Array of indices of users along the linear path

    """
    start_pos = np.array(start_pos)
    end_pos = np.array(end_pos)

    # Ensure 3D coordinates
    if start_pos.shape[0] == 2:
        start_pos = np.concatenate((start_pos, [0]))
    if end_pos.shape[0] == 2:
        end_pos = np.concatenate((end_pos, [0]))

    xs = np.linspace(start_pos[0], end_pos[0], n_steps).reshape((-1, 1))
    ys = np.linspace(start_pos[1], end_pos[1], n_steps).reshape((-1, 1))
    zs = np.linspace(start_pos[2], end_pos[2], n_steps).reshape((-1, 1))
    interpolated_pos = np.hstack((xs, ys, zs))
    idxs = np.array([np.argmin(np.linalg.norm(rx_pos - pos, axis=1)) for pos in interpolated_pos])
    # Filters only adjacent repeated points
    if filter_repeated:
        idxs = np.concatenate(([idxs[0]], idxs[1:][(idxs[1:] - idxs[:-1]) != 0]))
    return idxs


def dbw2watt(val: float | np.ndarray) -> float | np.ndarray:
    """Convert power from dBW to Watts.

    This function performs the standard conversion from decibel-watts (dBW)
    to linear power in Watts.

    Args:
        val: Power value(s) in dBW

    Returns:
        Power value(s) in Watts

    """
    return 10 ** (val / 10)


def get_uniform_idxs(n_ue: int, grid_size: np.ndarray, steps: list[int]) -> np.ndarray:
    """Return indices of users at uniform intervals.

    Args:
        n_ue: Number of users
        grid_size: Grid size [x_size, y_size]
        steps: List of sampling steps for each dimension [x_step, y_step]

    Returns:
        Array of indices for uniformly sampled users

    Raises:
        ValueError: If dataset does not have a valid grid structure

    """
    # Check if dataset has valid grid structure
    if steps == [1, 1]:
        return np.arange(n_ue)

    if np.prod(grid_size) != n_ue:
        print(
            f"Warning. Grid_size: {grid_size} = {np.prod(grid_size)} users != {n_ue} users in rx_pos",
        )
        print("Computing pseudo-uniform indices.")

        _grid_size = grid_size
        while np.prod(_grid_size) > n_ue:
            _grid_size -= 1  # Decrease grid size by 1 until product <= n_ue
    else:
        # Get indices of users at uniform intervals
        _grid_size = grid_size

    cols = np.arange(_grid_size[0], step=steps[0])
    rows = np.arange(_grid_size[1], step=steps[1])
    idxs = np.array([j + i * _grid_size[0] for i in rows for j in cols])

    return idxs


def get_grid_idxs(
    grid_size: np.ndarray,
    axis: str,
    idxs: int | list[int] | np.ndarray,
) -> np.ndarray:
    """Return indices of users in the specified rows or columns, assuming a grid structure.

    Args:
        grid_size: Grid size as [x_size, y_size] where x_size is number of columns and y_size is number of rows
        axis: Either 'row' or 'col' to specify which indices to get
        idxs: Array of row or column indices to include

    Returns:
        Array of indices of receivers in the specified rows or columns

    Raises:
        ValueError: If axis is not 'row' or 'col'

    """
    if axis not in ["row", "col"]:
        raise ValueError("axis must be either 'row' or 'col'")

    if isinstance(idxs, int):
        idxs = [idxs]

    indices = []
    if axis == "row":
        # Each row contains grid_size[0] elements (number of columns)
        for row in idxs:
            row_start = row * grid_size[0]
            row_indices = np.arange(row_start, row_start + grid_size[0])
            indices.extend(row_indices)
    else:  # axis == 'col'
        # Each column contains grid_size[1] elements
        for col in idxs:
            col_indices = col + np.arange(grid_size[1]) * grid_size[0]
            indices.extend(col_indices)

    return np.array(indices)


def get_idxs_with_limits(data_pos: np.ndarray, **limits: float) -> np.ndarray:
    """Return indices of users within specified coordinate limits.

    Args:
        data_pos: User positions array [n_users, 3]
        **limits: Coordinate limits as keyword arguments:
            x_min, x_max: X coordinate limits
            y_min, y_max: Y coordinate limits
            z_min, z_max: Z coordinate limits

    Returns:
        Array of indices for users within limits

    """
    valid_limits = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
    if not all(key in valid_limits for key in limits):
        raise ValueError(f"Invalid limit key. Supported limits are: {valid_limits}")

    # Start with all indices as valid
    valid_idxs = np.arange(len(data_pos))

    # Apply each limit sequentially
    coord_map = {"x": 0, "y": 1, "z": 2}
    for limit_name, limit_value in limits.items():
        coord = limit_name.split("_")[0]  # Extract 'x', 'y', or 'z'
        is_min = limit_name.endswith("min")

        if coord_map[coord] >= data_pos.shape[1]:
            raise ValueError(f"Cannot apply {coord} limit to {data_pos.shape[1]}D positions")

        if is_min:
            mask = data_pos[valid_idxs, coord_map[coord]] >= limit_value
        else:  # is_max
            mask = data_pos[valid_idxs, coord_map[coord]] <= limit_value

        valid_idxs = valid_idxs[mask]

    return valid_idxs

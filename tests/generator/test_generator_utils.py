"""Tests for DeepMIMO Generator Utilities."""

import numpy as np

from deepmimo.generator import generator_utils as gu


def test_get_linear_idxs() -> None:
    # 2D case
    rx_pos = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
    start = [0, 0]
    end = [3, 3]
    # Expecting [0, 1, 2, 3]

    # rx_pos in test should be 3D if function expects 3D or logic handles it.
    rx_pos_3d = np.column_stack([rx_pos, np.zeros(4)])
    idxs = gu.get_linear_idxs(rx_pos_3d, start, end, n_steps=4)
    np.testing.assert_array_equal(idxs, [0, 1, 2, 3])


def test_dbw2watt() -> None:
    assert gu.dbw2watt(0) == 1.0
    assert gu.dbw2watt(10) == 10.0
    assert gu.dbw2watt(20) == 100.0


def test_get_uniform_idxs() -> None:
    # Grid 10x10 = 100 users.
    n_ue = 100
    grid_size = np.array([10, 10])
    steps = [2, 2]
    # Columns: 0, 2, 4, 6, 8 (5)
    # Rows: 0, 2, 4, 6, 8 (5)
    # Total 25 users.
    idxs = gu.get_uniform_idxs(n_ue, grid_size, steps)
    assert len(idxs) == 25
    assert idxs[0] == 0

    # Test warning/pseudo-uniform
    # 99 users != 10*10=100.
    # If steps=[1, 1], it short-circuits to arange(n_ue).
    idxs_all = gu.get_uniform_idxs(99, grid_size, [1, 1])
    assert len(idxs_all) == 99

    # To test grid reduction logic, use steps != [1, 1]
    # [10, 10] -> [9, 9] (prod 81).
    # steps=[2, 2] -> 5x5 = 25 users.
    idxs_pseudo = gu.get_uniform_idxs(99, grid_size, [2, 2])
    assert len(idxs_pseudo) == 25


def test_get_grid_idxs() -> None:
    grid_size = np.array([10, 5])  # 10 cols, 5 rows
    # Row 0: 0..9
    idxs = gu.get_grid_idxs(grid_size, "row", 0)
    np.testing.assert_array_equal(idxs, np.arange(10))

    # Col 0: 0, 10, 20, 30, 40
    idxs = gu.get_grid_idxs(grid_size, "col", 0)
    np.testing.assert_array_equal(idxs, np.arange(0, 50, 10))


def test_get_idxs_with_limits() -> None:
    pos = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    # x < 1.5 -> indices 0, 1
    idxs = gu.get_idxs_with_limits(pos, x_max=1.5)
    np.testing.assert_array_equal(idxs, [0, 1])

    # y > 0.5 -> indices 1, 2
    idxs = gu.get_idxs_with_limits(pos, y_min=0.5)
    np.testing.assert_array_equal(idxs, [1, 2])

    # z in [0.5, 1.5] -> index 1
    idxs = gu.get_idxs_with_limits(pos, z_min=0.5, z_max=1.5)
    np.testing.assert_array_equal(idxs, [1])

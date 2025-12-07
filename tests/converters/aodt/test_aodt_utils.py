"""Tests for AODT Utils."""

import numpy as np
from deepmimo.converters.aodt import aodt_utils

def test_dict_to_array():
    pt = {"1": 1.0, "2": 2.0, "3": 3.0}
    arr = aodt_utils.dict_to_array(pt)
    np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

def test_process_points():
    pts = [{"1": 1.0, "2": 2.0, "3": 3.0}, np.array([4.0, 5.0, 6.0])]
    arr = aodt_utils.process_points(pts)
    assert arr.shape == (2, 3)
    np.testing.assert_array_equal(arr[0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(arr[1], [4.0, 5.0, 6.0])


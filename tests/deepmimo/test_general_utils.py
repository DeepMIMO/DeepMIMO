"""Tests for DeepMIMO general utilities."""

import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from deepmimo import general_utils
from deepmimo.config import config

@pytest.fixture
def temp_dir(tmp_path):
    """Fixture to provide a temporary directory."""
    return tmp_path

def test_check_scen_name():
    """Test scenario name validation."""
    general_utils.check_scen_name("valid_name")
    
    with pytest.raises(ValueError):
        general_utils.check_scen_name("invalid/name")
        
    with pytest.raises(ValueError):
        general_utils.check_scen_name("invalid\\name")

def test_get_dirs(temp_dir):
    """Test directory getter functions."""
    # Mock config to return a path inside temp_dir
    # Patch deepmimo.general_utils.config
    with patch("deepmimo.general_utils.config.get", return_value="scenarios"):
        with patch("deepmimo.general_utils.os.getcwd", return_value=str(temp_dir)):
            assert general_utils.get_scenarios_dir() == os.path.join(str(temp_dir), "scenarios")
            
            # Test get_scenario_folder
            assert general_utils.get_scenario_folder("my_scen") == os.path.join(str(temp_dir), "scenarios", "my_scen")

def test_get_mat_filename():
    """Test MAT filename generation."""
    fname = general_utils.get_mat_filename("channel", 1, 2, 3)
    assert fname == "channel_t001_tx002_r003.npz"
    
    fname_npz = general_utils.get_mat_filename("channel", 1, 2, 3, fmt="npz")
    assert fname_npz == "channel_t001_tx002_r003.npz"

def test_mat_save_load(temp_dir):
    """Test saving and loading MAT files."""
    data = np.array([[1, 2], [3, 4]])
    key = "test_data"
    
    # Test MAT format
    mat_path = os.path.join(temp_dir, "test_mat.mat")
    general_utils.save_mat(data, key, mat_path, fmt="mat")
    loaded = general_utils.load_mat(mat_path, key)
    np.testing.assert_array_equal(data, loaded)
    
    # Test NPZ format
    npz_path = os.path.join(temp_dir, "test_npz.npz")
    general_utils.save_mat(data, key, npz_path, fmt="npz")
    # load_mat tries extension replacement, so pass .mat or .npz
    loaded_npz = general_utils.load_mat(npz_path, key)
    np.testing.assert_array_equal(data, loaded_npz)
    
    # Test NPY format
    npy_path = os.path.join(temp_dir, "test_npy.npy")
    general_utils.save_mat(data, key, npy_path, fmt="npy")
    loaded_npy = general_utils.load_mat(npy_path) # NPY doesn't need key
    np.testing.assert_array_equal(data, loaded_npy)

def test_json_save_load(temp_dir):
    """Test saving and loading JSON files."""
    data = {"a": 1, "b": [1, 2, 3], "c": np.array([4, 5, 6])}
    json_path = os.path.join(temp_dir, "test.json")
    
    general_utils.save_dict_as_json(json_path, data)
    loaded = general_utils.load_dict_from_json(json_path)
    
    assert loaded["a"] == 1
    assert loaded["b"] == [1, 2, 3]
    assert loaded["c"] == [4, 5, 6]  # Converted to list

def test_deep_dict_merge():
    """Test deep merging of dictionaries."""
    d1 = {"a": 1, "b": {"c": 2, "d": 3}}
    d2 = {"b": {"c": 4}, "e": 5}
    
    merged = general_utils.deep_dict_merge(d1, d2)
    
    assert merged["a"] == 1
    assert merged["b"]["c"] == 4  # Overwritten
    assert merged["b"]["d"] == 3  # Preserved
    assert merged["e"] == 5       # Added

def test_dot_dict():
    """Test DotDict functionality."""
    d = general_utils.DotDict({"a": 1, "b": {"c": 2}})
    
    # Access
    assert d.a == 1
    assert d.b.c == 2
    assert d['b']['c'] == 2
    
    # Assignment
    d.a = 10
    assert d.a == 10
    assert d['a'] == 10
    
    # Nested assignment
    d.b.d = 3
    assert d.b.d == 3
    
    # New attribute
    d.new_attr = "test"
    assert d.new_attr == "test"
    
    # Dictionary methods
    assert "a" in d.keys()
    assert 10 in d.values()
    
    # To dict
    reg_dict = d.to_dict()
    assert isinstance(reg_dict, dict)
    assert not isinstance(reg_dict, general_utils.DotDict)
    assert isinstance(reg_dict['b'], dict)
    assert not isinstance(reg_dict['b'], general_utils.DotDict)

def test_coordinate_conversion():
    """Test coordinate conversion functions."""
    # Test cartesian to spherical
    cart = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sph = general_utils.cartesian_to_spherical(cart)
    
    # Point (1,0,0): r=1, az=0, el=0
    assert np.allclose(sph[0], [1, 0, 0])
    
    # Point (0,1,0): r=1, az=pi/2, el=0
    assert np.allclose(sph[1], [1, np.pi/2, 0])
    
    # Point (0,0,1): r=1, az=0 (undefined), el=pi/2
    assert np.isclose(sph[2, 0], 1)
    assert np.isclose(sph[2, 2], np.pi/2)
    
    # Test spherical to cartesian (inverse)
    # NOTE: spherical_to_cartesian expects inclination (from z-axis) but cartesian_to_spherical
    # returns elevation (from xy-plane). We need to convert before testing inverse.
    sph_for_inverse = sph.copy()
    sph_for_inverse[..., 1] = np.pi/2 - sph[..., 2] # Inclination = pi/2 - Elevation (if Elevation is from horizon)
    # Wait, cartesian_to_spherical returns (r, az, el).
    # spherical_to_cartesian takes (r, inc, az). Note order change too!
    # Let's check spherical_to_cartesian signature: (r, elevation, azimuth) where elevation is FROM Z AXIS?
    # Actually, let's look at the implementation of spherical_to_cartesian in general_utils:
    # cartesian_coords[..., 0] = r * np.sin(elevation) * np.cos(azimuth)  # x
    # This implies 'elevation' param IS inclination (theta).
    # And the order is (..., 1) -> elevation/inc, (..., 2) -> azimuth.
    # cartesian_to_spherical returns: (..., 1) -> az, (..., 2) -> el.
    # So we need to swap indices 1 and 2, AND convert el to inc.
    
    sph_input = np.zeros_like(sph)
    sph_input[..., 0] = sph[..., 0] # r
    sph_input[..., 1] = np.pi/2 - sph[..., 2] # inclination from z = pi/2 - elevation from xy
    sph_input[..., 2] = sph[..., 1] # azimuth
    
    back_cart = general_utils.spherical_to_cartesian(sph_input)
    assert np.allclose(back_cart, cart, atol=1e-7)

def test_delegating_list():
    """Test DelegatingList functionality."""
    class Item:
        def __init__(self, val):
            self.val = val
        def double(self):
            return self.val * 2
            
    items = [Item(1), Item(2), Item(3)]
    dlist = general_utils.DelegatingList(items)
    
    # Property access delegation
    assert dlist.val == [1, 2, 3]
    
    # Method call delegation
    assert dlist.double() == [2, 4, 6]
    
    # Attribute assignment delegation
    dlist.val = [10, 20, 30]
    assert items[0].val == 10
    assert items[1].val == 20
    assert items[2].val == 30
    
    # Scalar assignment
    dlist.val = 0
    assert items[0].val == 0

def test_comp_next_pwr_10():
    """Test comp_next_pwr_10 logic."""
    # Re-implement locally as it's not in the package
    def comp_next_pwr_10(arr):
        # def compute_number_paths(arr):
        # Handle zero separately
        result = np.zeros_like(arr, dtype=int)

        # For non-zero values, calculate order
        non_zero = arr > 0
        result[non_zero] = np.floor(np.log10(arr[non_zero])).astype(int) + 1

        return result

    arr = np.array([0, 1, 9, 10, 99, 100, 999, 1000])
    expected = np.array([0, 1, 1, 2, 2, 3, 3, 4])
    
    custom_orders = comp_next_pwr_10(arr)
    np.testing.assert_array_equal(custom_orders, expected)

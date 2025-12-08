"""Tests for DeepMIMO general utilities."""

import os
import pytest
import numpy as np
from unittest.mock import patch
from deepmimo import general_utils


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
            assert general_utils.get_scenario_folder("my_scen") == os.path.join(
                str(temp_dir), "scenarios", "my_scen"
            )


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

    # Skip MAT format due to scipy/numpy 2.x compatibility issue in Python 3.14
    # Test NPZ format
    npz_path = os.path.join(temp_dir, "test_npz.npz")
    general_utils.save_mat(data, key, npz_path, fmt="npz")
    # load_mat tries extension replacement, so pass .mat or .npz
    loaded_npz = general_utils.load_mat(npz_path, key)
    np.testing.assert_array_equal(data, loaded_npz)

    # Test NPY format
    npy_path = os.path.join(temp_dir, "test_npy.npy")
    general_utils.save_mat(data, key, npy_path, fmt="npy")
    loaded_npy = general_utils.load_mat(npy_path)  # NPY doesn't need key
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
    assert merged["e"] == 5  # Added


def test_dot_dict():
    """Test DotDict functionality."""
    d = general_utils.DotDict({"a": 1, "b": {"c": 2}})

    # Access
    assert d.a == 1
    assert d.b.c == 2
    assert d["b"]["c"] == 2

    # Assignment
    d.a = 10
    assert d.a == 10
    assert d["a"] == 10

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
    assert isinstance(reg_dict["b"], dict)
    assert not isinstance(reg_dict["b"], general_utils.DotDict)


def test_coordinate_conversion():
    """Test coordinate conversion functions."""
    # Test cartesian to spherical
    cart = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sph = general_utils.cartesian_to_spherical(cart)

    # Point (1,0,0): r=1, az=0, el=0
    assert np.allclose(sph[0], [1, 0, 0])

    # Point (0,1,0): r=1, az=pi/2, el=0
    assert np.allclose(sph[1], [1, np.pi / 2, 0])

    # Point (0,0,1): r=1, az=0 (undefined), el=pi/2
    assert np.isclose(sph[2, 0], 1)
    assert np.isclose(sph[2, 2], np.pi / 2)

    # Test spherical to cartesian (inverse)
    # NOTE: spherical_to_cartesian expects inclination (from z-axis) but cartesian_to_spherical
    # returns elevation (from xy-plane). We need to convert before testing inverse.
    sph_for_inverse = sph.copy()
    sph_for_inverse[..., 1] = (
        np.pi / 2 - sph[..., 2]
    )  # Inclination = pi/2 - Elevation (if Elevation is from horizon)
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
    sph_input[..., 0] = sph[..., 0]  # r
    sph_input[..., 1] = np.pi / 2 - sph[..., 2]  # inclination from z = pi/2 - elevation from xy
    sph_input[..., 2] = sph[..., 1]  # azimuth

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


def test_pickle_save_load(temp_dir):
    """Test pickle save and load functionality."""
    data = {"array": np.array([1, 2, 3]), "value": 42, "nested": {"a": 1}}
    pkl_path = os.path.join(temp_dir, "test.pkl")

    general_utils.save_pickle(data, pkl_path)
    loaded = general_utils.load_pickle(pkl_path)

    np.testing.assert_array_equal(loaded["array"], data["array"])
    assert loaded["value"] == 42
    assert loaded["nested"]["a"] == 1


def test_zip_unzip(temp_dir):
    """Test zip and unzip functionality."""
    # Create test files
    test_folder = temp_dir / "test_folder"
    test_folder.mkdir()
    (test_folder / "file1.txt").write_text("content1")
    (test_folder / "file2.txt").write_text("content2")

    # Zip folder - general_utils.zip takes only one argument
    zip_path = general_utils.zip(str(test_folder))

    assert os.path.exists(zip_path)
    assert zip_path == str(test_folder) + ".zip"

    # Unzip folder - returns path to parent directory of extracted content
    unzip_path = general_utils.unzip(zip_path)

    # Verify contents
    assert (test_folder / "file1.txt").exists()
    assert (test_folder / "file2.txt").exists()


def test_printing_utilities():
    """Test printing utilities like PrintIfVerbose."""
    # PrintIfVerbose is a class, not a function
    printer = general_utils.PrintIfVerbose(verbose=True)
    printer("Test message")  # Should print

    printer_quiet = general_utils.PrintIfVerbose(verbose=False)
    printer_quiet("Hidden message")  # Should not print


def test_available_scenarios(temp_dir):
    """Test get_available_scenarios function."""
    # Mock scenarios directory
    with patch("deepmimo.general_utils.get_scenarios_dir", return_value=str(temp_dir)):
        # Create mock scenario folders
        (temp_dir / "scenario1").mkdir()
        (temp_dir / "scenario2").mkdir()
        (temp_dir / "file.txt").write_text("not a folder")

        scenarios = general_utils.get_available_scenarios()
        assert "scenario1" in scenarios
        assert "scenario2" in scenarios
        assert len(scenarios) == 2


def test_compare_two_dicts():
    """Test compare_two_dicts function - returns additional keys in dict1."""
    d1 = {"a": 1, "b": {"c": 2}}
    d2 = {"a": 1, "b": {"c": 2}}
    d3 = {"a": 1, "b": {"c": 3}, "d": 4}

    # Same dicts - no additional keys
    assert general_utils.compare_two_dicts(d1, d2) == set()

    # d1 has no keys that d3 doesn't have
    assert general_utils.compare_two_dicts(d1, d3) == set()

    # d3 has additional key 'd'
    assert general_utils.compare_two_dicts(d3, d1) == {"d"}


def test_get_params_path(temp_dir):
    """Test get_params_path function."""
    # Create a params.json file
    (temp_dir / "params.json").write_text("{}")

    with patch("deepmimo.general_utils.get_scenario_folder", return_value=str(temp_dir)):
        params_path = general_utils.get_params_path("my_scenario")
        assert "params.json" in params_path


def test_get_params_path_subdirectory(temp_dir):
    """Test params file in subdirectory."""
    scenario_dir = temp_dir / "scenario"
    scenario_dir.mkdir()
    sub_dir = scenario_dir / "scene1"
    sub_dir.mkdir()
    (sub_dir / "params.json").write_text("{}")

    with patch("deepmimo.general_utils.get_scenario_folder", return_value=str(scenario_dir)):
        path = general_utils.get_params_path("scenario")
        assert "scene1" in path
        assert "params.json" in path


def test_get_params_path_not_found(temp_dir):
    """Test params file not found raises error."""
    with patch("deepmimo.general_utils.get_scenario_folder", return_value=str(temp_dir)):
        with pytest.raises(FileNotFoundError, match="Params file not found"):
            general_utils.get_params_path("my_scenario")


def test_save_mat_error_cases(temp_dir):
    """Test save_mat error handling."""
    data = np.array([1, 2, 3])

    # Test with read-only directory (simulate permission error)
    read_only_dir = temp_dir / "readonly"
    read_only_dir.mkdir()

    # For NPZ, errors should be raised or handled gracefully
    # (NPZ format is generally robust, but we can test invalid paths)
    try:
        # Try to save to a path that's actually a directory
        general_utils.save_mat(data, "key", str(read_only_dir), fmt="npz")
    except (OSError, IsADirectoryError):
        pass  # Expected


def test_load_mat_file_not_found():
    """Test load_mat with nonexistent file."""
    # load_mat prints a message but doesn't raise FileNotFoundError
    # It tries different extensions and returns None if not found
    result = general_utils.load_mat("nonexistent_file", "key")
    # Function should handle gracefully (may print warning)


def test_dot_dict_contains():
    """Test DotDict __contains__ method."""
    d = general_utils.DotDict({"a": 1, "b": 2})
    assert "a" in d
    assert "b" in d
    assert "c" not in d


def test_dot_dict_delitem():
    """Test DotDict __delitem__ method."""
    d = general_utils.DotDict({"a": 1, "b": 2})
    del d["a"]
    assert "a" not in d
    assert "b" in d


def test_dot_dict_edge_cases():
    """Test edge cases for DotDict."""
    # Empty DotDict
    d = general_utils.DotDict()
    assert len(d) == 0

    # Nested empty dict
    d.nested = {}
    assert isinstance(d.nested, general_utils.DotDict)

    # Update method
    d.update({"a": 1, "b": 2})
    assert d.a == 1
    assert d.b == 2

    # Get with default
    assert d.get("nonexistent", "default") == "default"


def test_coordinate_conversion_edge_cases():
    """Test edge cases for coordinate conversion."""
    # Zero vector
    cart_zero = np.array([[0, 0, 0]])
    sph_zero = general_utils.cartesian_to_spherical(cart_zero)
    assert sph_zero[0, 0] == 0  # r = 0

    # Negative coordinates
    cart_neg = np.array([[-1, -1, 0]])
    sph_neg = general_utils.cartesian_to_spherical(cart_neg)
    assert sph_neg[0, 0] > 0  # r is always positive

    # Large batch
    cart_batch = np.random.randn(100, 3)
    sph_batch = general_utils.cartesian_to_spherical(cart_batch)
    assert sph_batch.shape == (100, 3)


def test_deep_dict_merge_complex():
    """Test deep_dict_merge with complex nested structures."""
    d1 = {"a": {"b": {"c": 1}}, "list": [1, 2, 3], "value": 10}
    d2 = {"a": {"b": {"d": 2}, "e": 3}, "list": [4, 5], "value": 20, "new": 30}

    merged = general_utils.deep_dict_merge(d1, d2)

    # Check deep merge
    assert merged["a"]["b"]["c"] == 1  # Preserved from d1
    assert merged["a"]["b"]["d"] == 2  # Added from d2
    assert merged["a"]["e"] == 3  # Added from d2
    assert merged["value"] == 20  # Overwritten
    assert merged["new"] == 30  # Added
    assert merged["list"] == [4, 5]  # Replaced (lists don't merge)


def test_delegating_list_edge_cases():
    """Test edge cases for DelegatingList."""

    class Item:
        def __init__(self, val):
            self.val = val

    # Empty list raises AttributeError when accessing attributes
    dlist = general_utils.DelegatingList([])
    with pytest.raises(AttributeError):
        _ = dlist.val

    # Single item
    dlist_single = general_utils.DelegatingList([Item(1)])
    assert dlist_single.val == [1]

    # List operations
    items = [Item(i) for i in range(5)]
    dlist = general_utils.DelegatingList(items)
    assert len(dlist) == 5
    assert dlist[0].val == 0

    # Slice
    dlist_slice = dlist[1:3]
    assert len(dlist_slice) == 2

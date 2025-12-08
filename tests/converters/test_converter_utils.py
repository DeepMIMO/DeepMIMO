"""Tests for DeepMIMO Converter Utilities."""

from unittest.mock import patch

import numpy as np

from deepmimo import consts as c
from deepmimo.converters import converter_utils as cu


def test_ext_in_list():
    files = ["a.txt", "b.py", "c.txt"]
    assert cu.ext_in_list(".txt", files) == ["a.txt", "c.txt"]
    assert cu.ext_in_list(".py", files) == ["b.py"]


def test_check_scenario_exists(tmp_path):
    scen_dir = tmp_path / "scenarios"
    scen_dir.mkdir()

    # Does not exist
    assert cu.check_scenario_exists(str(scen_dir), "new_scen") is True

    # Exists, overwrite=True
    (scen_dir / "old_scen").mkdir()
    assert cu.check_scenario_exists(str(scen_dir), "old_scen", overwrite=True) is True

    # Exists, overwrite=False
    assert cu.check_scenario_exists(str(scen_dir), "old_scen", overwrite=False) is False

    # Exists, prompt yes
    with patch("builtins.input", return_value="y"):
        assert cu.check_scenario_exists(str(scen_dir), "old_scen") is True


def test_comp_next_pwr_10():
    arr = np.array([0, 1, 10, 99, 100])
    res = cu.comp_next_pwr_10(arr)
    np.testing.assert_array_equal(res, [0, 1, 2, 2, 3])


def test_get_max_paths():
    data = {c.AOA_AZ_PARAM_NAME: np.array([[1, 2, np.nan], [3, np.nan, np.nan]])}
    # Col 0: [1, 3] -> valid
    # Col 1: [2, nan] -> valid (not all nan)
    # Col 2: [nan, nan] -> all nan -> stop
    assert cu.get_max_paths(data) == 2


def test_compress_path_data():
    # Mock data: 2 users, 5 paths allocated, but only 2 used.
    # And interactions codes up to 2 digits (e.g. 11).

    n_users = 2
    n_paths = 5
    n_bounces = 10

    data = {
        c.AOA_AZ_PARAM_NAME: np.full((n_users, n_paths), np.nan),
        c.INTERACTIONS_PARAM_NAME: np.zeros((n_users, n_paths)),
        c.INTERACTIONS_POS_PARAM_NAME: np.zeros((n_users, n_paths, n_bounces, 3)),
        c.RX_POS_PARAM_NAME: np.zeros((n_users, 3)),  # Should be ignored
    }

    # Set valid paths
    data[c.AOA_AZ_PARAM_NAME][:, 0:2] = 1.0  # 2 paths valid
    data[c.INTERACTIONS_PARAM_NAME][:, 0] = 11  # 2 interactions (digits)

    compressed = cu.compress_path_data(data)

    # Check path trimming
    assert compressed[c.AOA_AZ_PARAM_NAME].shape == (n_users, 2)

    # Check bounce trimming
    # Max bounces = 2 (from 11)
    assert compressed[c.INTERACTIONS_POS_PARAM_NAME].shape == (n_users, 2, 2, 3)

    # Check ignored
    assert compressed[c.RX_POS_PARAM_NAME].shape == (n_users, 3)

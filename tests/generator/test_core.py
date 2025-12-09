"""Tests for DeepMIMO Core Generation Module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deepmimo import consts as c
from deepmimo.generator import core


@pytest.fixture
def mock_dataset_cls():
    """Fixture mocking Dataset class within generator.core."""
    with patch("deepmimo.generator.core.Dataset") as mock:
        yield mock


@pytest.fixture
def mock_utils():
    """Fixture providing patched core helpers and filesystem checks."""
    with (
        patch("deepmimo.generator.core.load_dict_from_json") as load_json,
        patch("deepmimo.generator.core.load_mat") as load_mat,
        patch("deepmimo.generator.core.get_scenario_folder") as get_folder,
        patch("deepmimo.generator.core.get_params_path") as get_params,
        patch("deepmimo.generator.core.Scene") as mock_scene,
        patch("deepmimo.generator.core.MaterialList") as mock_mat_list,
        patch.object(Path, "exists") as mock_exists,
    ):
        mock_exists.return_value = True
        yield {
            "load_json": load_json,
            "load_mat": load_mat,
            "get_folder": get_folder,
            "get_params": get_params,
            "scene": mock_scene,
            "mat_list": mock_mat_list,
            "exists": mock_exists,
        }


def test_load_single_scene(mock_utils, mock_dataset_cls) -> None:
    """Test loading a single scene."""
    # Setup mocks
    mock_utils["exists"].return_value = True
    mock_utils["get_folder"].return_value = "/path/to/scen"
    mock_utils["get_params"].return_value = "/path/to/params.json"

    params = {
        c.SCENE_PARAM_NAME: {c.SCENE_PARAM_NUMBER_SCENES: 1},
        c.TXRX_PARAM_NAME: {
            "txrx_set_0": {"id": 0, "is_tx": True, "is_rx": False, "num_points": 1},
            "txrx_set_1": {"id": 1, "is_tx": False, "is_rx": True, "num_points": 10},
        },
        c.RT_PARAMS_PARAM_NAME: {},
        c.MATERIALS_PARAM_NAME: {},
    }
    mock_utils["load_json"].return_value = params

    # Mock load_mat to return some dummy data
    mock_utils["load_mat"].return_value = np.zeros((10, 5))  # 10 RX, 5 paths

    # Call load
    core.load("test_scen")

    # Verify calls
    mock_utils["get_folder"].assert_called_with("test_scen")
    mock_dataset_cls.assert_called()  # Dataset constructor called

    # Verify dataset properties set (on the mock instance)
    # ds_instance = dataset


def test_generate(mock_utils, mock_dataset_cls) -> None:
    """Test generate function."""
    # Setup mocks similar to load
    mock_utils["exists"].return_value = True
    params = {
        c.SCENE_PARAM_NAME: {c.SCENE_PARAM_NUMBER_SCENES: 1},
        c.TXRX_PARAM_NAME: {
            "txrx_set_0": {"id": 0, "is_tx": True, "is_rx": False, "num_points": 1},
            "txrx_set_1": {"id": 1, "is_tx": False, "is_rx": True, "num_points": 10},
        },
        c.RT_PARAMS_PARAM_NAME: {},
        c.MATERIALS_PARAM_NAME: {},
    }
    mock_utils["load_json"].return_value = params

    # Mock dataset instance
    ds_mock = MagicMock()
    mock_dataset_cls.return_value = ds_mock

    # Call generate
    result = core.generate("test_scen")

    assert result == ds_mock
    ds_mock.compute_channels.assert_called()


def test_validate_txrx_sets() -> None:
    """Test _validate_txrx_sets logic."""
    txrx_dict = {
        "txrx_set_0": {"id": 0, "is_tx": True, "is_rx": False, "num_points": 5},
        "txrx_set_1": {"id": 1, "is_tx": False, "is_rx": True, "num_points": 10},
    }

    # Test "all"
    sets = core.validate_txrx_sets("all", txrx_dict, "tx")
    assert 0 in sets
    assert len(sets[0]) == 5

    sets = core.validate_txrx_sets("all", txrx_dict, "rx")
    assert 1 in sets
    assert len(sets[1]) == 10

    # Test list of IDs
    sets = core.validate_txrx_sets([0], txrx_dict, "tx")
    assert 0 in sets
    assert len(sets[0]) == 5

    # Test dict
    sets = core.validate_txrx_sets({0: [0, 1]}, txrx_dict, "tx")
    assert 0 in sets
    assert len(sets[0]) == 2
    assert sets[0][0] == 0

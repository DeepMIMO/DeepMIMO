"""Tests for DeepMIMO Core Generation Module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deepmimo import consts as c
from deepmimo.datasets import generate as generate_func
from deepmimo.datasets import load as load_func
from deepmimo.datasets.load import (
    _process_dict_sets,
    _process_list_sets,
    _process_str_sets,
    validate_txrx_sets,
)


@pytest.fixture
def mock_dataset_cls():
    """Fixture mocking Dataset class within datasets.load."""
    with patch("deepmimo.datasets.load.Dataset") as mock:
        yield mock


@pytest.fixture
def mock_utils():
    """Fixture providing patched core helpers and filesystem checks."""
    with (
        patch("deepmimo.datasets.load.load_dict_from_json") as load_json,
        patch("deepmimo.datasets.load.load_mat") as load_mat,
        patch("deepmimo.datasets.load.get_scenario_folder") as get_folder,
        patch("deepmimo.datasets.load.get_params_path") as get_params,
        patch("deepmimo.datasets.load.Scene") as mock_scene,
        patch("deepmimo.datasets.load.MaterialList") as mock_mat_list,
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
    load_func("test_scen")

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
    result = generate_func("test_scen")

    assert result == ds_mock
    ds_mock.compute_channels.assert_called()


def test_validate_txrx_sets() -> None:
    """Test _validate_txrx_sets logic."""
    txrx_dict = {
        "txrx_set_0": {"id": 0, "is_tx": True, "is_rx": False, "num_points": 5},
        "txrx_set_1": {"id": 1, "is_tx": False, "is_rx": True, "num_points": 10},
    }

    # Test "all"
    sets = validate_txrx_sets("all", txrx_dict, "tx")
    assert 0 in sets
    assert len(sets[0]) == 5

    sets = validate_txrx_sets("all", txrx_dict, "rx")
    assert 1 in sets
    assert len(sets[1]) == 10

    # Test list of IDs
    sets = validate_txrx_sets([0], txrx_dict, "tx")
    assert 0 in sets
    assert len(sets[0]) == 5

    # Test dict
    sets = validate_txrx_sets({0: [0, 1]}, txrx_dict, "tx")
    assert 0 in sets
    assert len(sets[0]) == 2
    assert sets[0][0] == 0


_TXRX = {
    "txrx_set_0": {"id": 0, "is_tx": True, "is_rx": False, "num_points": 3},
    "txrx_set_1": {"id": 1, "is_tx": False, "is_rx": True, "num_points": 5},
}
_VALID_TX = [0]
_VALID_RX = [1]
_INFO = "run dm.info()"


def test_process_dict_sets_list_indices() -> None:
    """List of indices should be converted to a numpy array."""
    result = _process_dict_sets({0: [0, 2]}, _TXRX, _VALID_TX, "Tx", _INFO)
    np.testing.assert_array_equal(result[0], [0, 2])


def test_process_dict_sets_all_string() -> None:
    """'all' string should expand to all available indices."""
    result = _process_dict_sets({0: "all"}, _TXRX, _VALID_TX, "Tx", _INFO)
    np.testing.assert_array_equal(result[0], np.arange(3))


def test_process_dict_sets_ndarray_passthrough() -> None:
    """Pre-built numpy array indices should pass through unchanged."""
    idxs = np.array([1, 2])
    result = _process_dict_sets({0: idxs}, _TXRX, _VALID_TX, "Tx", _INFO)
    np.testing.assert_array_equal(result[0], idxs)


def test_process_dict_sets_out_of_range_raises() -> None:
    """Indices outside valid range should raise ValueError."""
    with pytest.raises(ValueError, match=r"."):
        _process_dict_sets({0: [10]}, _TXRX, _VALID_TX, "Tx", _INFO)


def test_process_dict_sets_invalid_string_raises() -> None:
    """Unrecognized string index should raise ValueError."""
    with pytest.raises(ValueError, match="not recognized"):
        _process_dict_sets({0: "first"}, _TXRX, _VALID_TX, "Tx", _INFO)


def test_process_dict_sets_invalid_type_raises() -> None:
    """Non-list/array/string indices should raise TypeError."""
    with pytest.raises(TypeError):
        _process_dict_sets({0: 42}, _TXRX, _VALID_TX, "Tx", _INFO)


def test_process_list_sets_returns_all_indices() -> None:
    """List of set IDs should expand each to all available indices."""
    result = _process_list_sets([1], _TXRX, _VALID_RX, "Rx", _INFO)
    np.testing.assert_array_equal(result[1], np.arange(5))


def test_process_list_sets_invalid_id_raises() -> None:
    """Unknown set ID should raise ValueError."""
    with pytest.raises(ValueError, match=r"."):
        _process_list_sets([99], _TXRX, _VALID_TX, "Tx", _INFO)


def test_process_str_sets_all_tx() -> None:
    """'all' for TX should include both TX-only and TX/RX sets."""
    result = _process_str_sets("all", _TXRX, _VALID_TX, "tx")
    assert 0 in result
    np.testing.assert_array_equal(result[0], np.arange(3))


def test_process_str_sets_rx_only_excludes_tx() -> None:
    """'rx_only' for RX should skip sets that are also TX."""
    mixed_txrx = {
        "txrx_set_0": {"id": 0, "is_tx": True, "is_rx": True, "num_points": 2},
        "txrx_set_1": {"id": 1, "is_tx": False, "is_rx": True, "num_points": 4},
    }
    result = _process_str_sets("rx_only", mixed_txrx, [0, 1], "rx")
    assert 0 not in result
    assert 1 in result


def test_process_str_sets_invalid_raises() -> None:
    """Unrecognized string should raise ValueError."""
    with pytest.raises(ValueError, match="not understood"):
        _process_str_sets("none", _TXRX, _VALID_TX, "tx")


# ---------------------------------------------------------------------------
# Additional coverage tests for load.py
# ---------------------------------------------------------------------------


from deepmimo.datasets.load import (  # noqa: E402
    DynamicDataset,
    MacroDataset,
    _load_raytracing_scene,
    _load_tx_rx_raydata,
    _validate_txrx_sets,
)

_PARAMS_SINGLE = {
    c.SCENE_PARAM_NAME: {c.SCENE_PARAM_NUMBER_SCENES: 1},
    c.TXRX_PARAM_NAME: {
        "txrx_set_0": {"id": 0, "is_tx": True, "is_rx": False, "num_points": 1},
        "txrx_set_1": {"id": 1, "is_tx": False, "is_rx": True, "num_points": 3},
    },
    c.RT_PARAMS_PARAM_NAME: {},
    c.MATERIALS_PARAM_NAME: {},
}

_PARAMS_DYNAMIC = {
    c.SCENE_PARAM_NAME: {c.SCENE_PARAM_NUMBER_SCENES: 2},
    c.TXRX_PARAM_NAME: {
        "txrx_set_0": {"id": 0, "is_tx": True, "is_rx": False, "num_points": 1},
        "txrx_set_1": {"id": 1, "is_tx": False, "is_rx": True, "num_points": 3},
    },
    c.RT_PARAMS_PARAM_NAME: {},
    c.MATERIALS_PARAM_NAME: {},
}


@pytest.fixture
def mock_load_utils():
    """Patch all external I/O helpers used by load() and _load_dataset()."""
    with (
        patch("deepmimo.datasets.load.load_dict_from_json") as load_json,
        patch("deepmimo.datasets.load.load_mat") as load_mat,
        patch("deepmimo.datasets.load.get_scenario_folder") as get_folder,
        patch("deepmimo.datasets.load.get_params_path") as get_params,
        patch("deepmimo.datasets.load.Scene") as mock_scene,
        patch("deepmimo.datasets.load.MaterialList") as mock_mat_list,
        patch.object(Path, "exists", return_value=True),
    ):
        get_folder.return_value = "/fake/folder"
        get_params.return_value = "/fake/params.json"
        load_mat.return_value = np.zeros((3, 5))
        yield {
            "load_json": load_json,
            "load_mat": load_mat,
            "get_folder": get_folder,
            "get_params": get_params,
            "scene": mock_scene,
            "mat_list": mock_mat_list,
        }


# ── Absolute path branch (lines 83-84) ─────────────────────────────────────


def test_load_absolute_path_sets_scen_folder(mock_load_utils) -> None:
    """When scen_name is an absolute path, scen_folder=scen_name and name=basename."""
    mock_load_utils["load_json"].return_value = _PARAMS_SINGLE

    with patch.object(Path, "is_absolute", return_value=True):
        result = load_func("/tmp/my_scen")  # noqa: S108

    # The scenario name extracted from the absolute path should be "my_scen"
    assert result[c.NAME_PARAM_NAME] == "my_scen"
    # get_scenario_folder should NOT have been called (absolute path used directly)
    mock_load_utils["get_folder"].assert_not_called()


# ── Download prompt branch (lines 90-99) ───────────────────────────────────


def test_load_download_prompt_n_raises(mock_load_utils) -> None:
    """Responding 'n' to the download prompt raises ValueError."""
    mock_load_utils["load_json"].return_value = _PARAMS_SINGLE

    with (
        patch.object(Path, "exists", return_value=False),
        patch.object(Path, "is_absolute", return_value=False),
        patch("builtins.input", return_value="n"),
        pytest.raises(ValueError, match="not found"),
    ):
        load_func("nonexistent_scen")


def test_load_download_prompt_y_calls_download(mock_load_utils) -> None:
    """Responding 'y' to the download prompt triggers download() and completes loading."""
    mock_load_utils["load_json"].return_value = _PARAMS_SINGLE
    mock_download = MagicMock()

    # First Path.exists() call returns False (triggers prompt); subsequent calls return True
    exists_results = iter([False, True, True, True, True])

    with (
        patch.object(Path, "exists", side_effect=lambda: next(exists_results, True)),
        patch.object(Path, "is_absolute", return_value=False),
        patch("builtins.input", return_value="y"),
        patch("deepmimo.api.download", mock_download),
    ):
        result = load_func("test_scen")

    mock_download.assert_called_once_with("test_scen")
    assert result is not None


# ── Multi-snapshot / DynamicDataset branch (lines 109-119) ─────────────────


def test_load_multi_snapshot_returns_dynamic_dataset(mock_load_utils) -> None:
    """When n_snapshots > 1, load() should return a DynamicDataset."""
    mock_load_utils["load_json"].return_value = _PARAMS_DYNAMIC

    mock_dir1 = MagicMock()
    mock_dir1.name = "scene_0001"
    mock_dir1.is_dir.return_value = True
    mock_dir2 = MagicMock()
    mock_dir2.name = "scene_0002"
    mock_dir2.is_dir.return_value = True

    with patch.object(Path, "iterdir", return_value=[mock_dir1, mock_dir2]):
        result = load_func("test_scen")

    assert isinstance(result, DynamicDataset)


# ── compat_v3 + dynamic raises ValueError (lines 109-111) ──────────────────


def test_load_compat_v3_with_dynamic_raises(mock_load_utils) -> None:
    """`compat_v3=True` must raise ValueError for dynamic (multi-snapshot) scenarios."""
    mock_load_utils["load_json"].return_value = _PARAMS_DYNAMIC

    with pytest.raises(ValueError, match="compat_v3"):
        load_func("test_scen", compat_v3=True)


# ── _load_raytracing_scene multiple tx → MacroDataset (line 210) ───────────


def test_load_raytracing_scene_multiple_tx_returns_macro() -> None:
    """Two TX indices for a single TX set should produce a MacroDataset."""
    txrx_dict = {
        "txrx_set_0": {"id": 0, "is_tx": True, "is_rx": False, "num_points": 2},
        "txrx_set_1": {"id": 1, "is_tx": False, "is_rx": True, "num_points": 3},
    }
    with patch("deepmimo.datasets.load.load_mat", return_value=np.zeros((3, 5))):
        result = _load_raytracing_scene("/fake/folder", txrx_dict)

    assert isinstance(result, MacroDataset)


def test_load_raytracing_scene_single_tx_returns_dataset() -> None:
    """A single TX index should return a plain Dataset (not MacroDataset)."""
    from deepmimo.datasets.dataset import Dataset  # noqa: PLC0415

    txrx_dict = {
        "txrx_set_0": {"id": 0, "is_tx": True, "is_rx": False, "num_points": 1},
        "txrx_set_1": {"id": 1, "is_tx": False, "is_rx": True, "num_points": 3},
    }
    with patch("deepmimo.datasets.load.load_mat", return_value=np.zeros((3, 5))):
        result = _load_raytracing_scene("/fake/folder", txrx_dict)

    assert isinstance(result, Dataset)
    assert not isinstance(result, MacroDataset)


# ── _load_tx_rx_raydata verbose=True (lines 289-306) ───────────────────────


def test_load_tx_rx_raydata_verbose_prints_loading(capsys) -> None:
    """verbose=True should print 'Loading ...' and 'Done.' for each matrix."""
    with patch("deepmimo.datasets.load.load_mat", return_value=np.zeros((5, 10))):
        _load_tx_rx_raydata("/fake", 0, 1, 0, np.arange(5), 10, verbose=True)

    captured = capsys.readouterr()
    assert "Loading" in captured.out
    assert "Done" in captured.out


def test_load_tx_rx_raydata_verbose_none_mat(capsys) -> None:
    """verbose=True with load_mat returning None should not raise."""
    with patch("deepmimo.datasets.load.load_mat", return_value=None):
        result = _load_tx_rx_raydata("/fake", 0, 1, 0, np.arange(5), 10, verbose=True)

    captured = capsys.readouterr()
    assert "Loading" in captured.out
    # All values should be None when load_mat returns None
    assert all(v is None for v in result.values())


# ── _load_tx_rx_raydata with specific matrix list (lines 275-280) ──────────


def test_load_tx_rx_raydata_specific_matrix_list() -> None:
    """Only matrices in matrices_to_load should be populated."""
    with patch("deepmimo.datasets.load.load_mat", return_value=np.zeros((5, 10))):
        result = _load_tx_rx_raydata(
            "/fake", 0, 1, 0, np.arange(5), 10, matrices_to_load=[c.AOA_AZ_PARAM_NAME]
        )

    # Only aoa_az should be non-None
    assert result[c.AOA_AZ_PARAM_NAME] is not None
    # All other keys should remain None
    for key in [c.AOD_AZ_PARAM_NAME, c.POWER_PARAM_NAME, c.DELAY_PARAM_NAME]:
        assert result[key] is None, f"Expected {key} to be None"


def test_load_tx_rx_raydata_none_matrices_loads_nothing() -> None:
    """matrices_to_load=None should result in all keys being None."""
    with patch("deepmimo.datasets.load.load_mat", return_value=np.zeros((5, 10))):
        result = _load_tx_rx_raydata("/fake", 0, 1, 0, np.arange(5), 10, matrices_to_load=None)

    assert all(v is None for v in result.values())


# ── _load_tx_rx_raydata invalid matrix name raises (lines 277-280) ─────────


def test_load_tx_rx_raydata_invalid_matrix_name_raises() -> None:
    """Invalid matrix name in matrices_to_load should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid matrix"):
        _load_tx_rx_raydata(
            "/fake", 0, 1, 0, np.arange(5), 10, matrices_to_load=["not_a_real_matrix"]
        )


# ── _validate_txrx_sets non-dict/list/str raises (lines 494-495) ───────────


def test_validate_txrx_sets_invalid_type_raises() -> None:
    """Passing an integer (non-dict/list/str) should raise ValueError."""
    txrx_dict = {
        "txrx_set_0": {"id": 0, "is_tx": True, "is_rx": False, "num_points": 3},
    }
    with pytest.raises(ValueError, match="Sets must be"):
        _validate_txrx_sets(42, txrx_dict, "tx")


def test_validate_txrx_sets_tuple_raises() -> None:
    """Passing a tuple (non-dict/list/str) should raise ValueError."""
    txrx_dict = {
        "txrx_set_0": {"id": 0, "is_tx": True, "is_rx": False, "num_points": 3},
    }
    with pytest.raises(ValueError, match="Sets must be"):
        _validate_txrx_sets((0,), txrx_dict, "tx")

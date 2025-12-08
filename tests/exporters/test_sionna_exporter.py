"""Tests for Sionna Exporter."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock
import sys

# Mock sionna before importing exporter
mock_sionna = MagicMock()
mock_sionna.rt.Paths = MagicMock
mock_sionna.rt.Scene = MagicMock
sys.modules["sionna"] = mock_sionna
sys.modules["sionna.rt"] = mock_sionna.rt

import deepmimo.exporters.sionna_exporter as sionna_exporter


@pytest.fixture
def mock_scene():
    scene = MagicMock()
    scene._scene_objects = {}

    # Mock arrays
    rx_array = MagicMock()
    rx_array.positions.numpy.return_value = np.zeros((2, 3))
    rx_array.array_size = 2
    rx_array.num_ant = 2

    tx_array = MagicMock()
    tx_array.positions.numpy.return_value = np.zeros((1, 3))
    tx_array.array_size = 1
    tx_array.num_ant = 1

    scene.rx_array = rx_array
    scene.tx_array = tx_array
    scene.bandwidth.numpy.return_value = 1e9
    scene.frequency.numpy.return_value = 28e9

    return scene


@pytest.fixture
def mock_paths():
    paths = MagicMock()
    # Mock attributes expected by _paths_to_dict
    paths.tau.numpy.return_value = np.array([1.0])
    paths.phi_r.numpy.return_value = np.array([0.0])
    paths.phi_t.numpy.return_value = np.array([0.0])
    paths.theta_r.numpy.return_value = np.array([0.0])
    paths.theta_t.numpy.return_value = np.array([0.0])
    paths.a.numpy.return_value = np.array([1.0 + 0j])
    paths.types.numpy.return_value = np.array([0])  # Sionna 0.x style

    # Ensure dir(paths) returns these
    paths.__dir__ = Mock(return_value=["tau", "phi_r", "phi_t", "theta_r", "theta_t", "a", "types"])

    # Need to mock getattr behavior or ensure _paths_to_dict uses getattr correctly on Mock
    # MagicMock handles getattr by returning Mocks, so we need to set values
    # _paths_to_dict iterates dir(paths) and gets attrs.
    # We should patch _paths_to_dict or control dir() better.
    return paths


@patch("deepmimo.exporters.sionna_exporter.is_sionna_v1")
@patch("deepmimo.exporters.sionna_exporter._paths_to_dict")
def test_export_paths(mock_p2d, mock_v1, mock_paths):
    mock_v1.return_value = False
    # Setup _paths_to_dict return
    mock_p2d.return_value = {
        "tau": MagicMock(numpy=lambda: np.array([1.0])),
        "phi_r": MagicMock(numpy=lambda: np.array([0.0])),
        "phi_t": MagicMock(numpy=lambda: np.array([0.0])),
        "theta_r": MagicMock(numpy=lambda: np.array([0.0])),
        "theta_t": MagicMock(numpy=lambda: np.array([0.0])),
        "a": MagicMock(numpy=lambda: np.array([1.0 + 0j])),
        "types": MagicMock(numpy=lambda: np.array([0])),
        "sources": MagicMock(numpy=lambda: np.array([0])),
        "targets": MagicMock(numpy=lambda: np.array([0])),
        "vertices": MagicMock(numpy=lambda: np.array([0])),
    }

    dicts = sionna_exporter.export_paths(mock_paths)
    assert len(dicts) == 1
    assert dicts[0]["tau"][0] == 1.0
    assert "types" in dicts[0]


@patch("deepmimo.exporters.sionna_exporter.is_sionna_v1")
def test_export_scene_materials(mock_v1, mock_scene):
    mock_v1.return_value = False
    # Create mock material
    mat = MagicMock()
    mat.name = "Mat1"
    mat.conductivity.numpy.return_value = 1.0
    mat.relative_permittivity.numpy.return_value = 2.0
    mat.scattering_coefficient.numpy.return_value = 0.5
    mat.xpd_coefficient.numpy.return_value = 0.0
    mat.scattering_pattern = MagicMock()

    obj = MagicMock()
    obj.radio_material = mat
    mock_scene._scene_objects = {"obj1": obj}

    mats_list, indices = sionna_exporter.export_scene_materials(mock_scene)
    assert len(mats_list) == 1
    assert mats_list[0]["name"] == "Mat1"
    assert len(indices) == 1


@patch("deepmimo.exporters.sionna_exporter.is_sionna_v1")
@patch("deepmimo.exporters.sionna_exporter.get_sionna_version")
def test_export_scene_rt_params(mock_ver, mock_v1, mock_scene):
    mock_v1.return_value = False
    mock_ver.return_value = "0.19.1"
    # Mock _scene_to_dict via patch or setting attributes on mock_scene
    with patch("deepmimo.exporters.sionna_exporter._scene_to_dict") as mock_s2d:
        mock_s2d.return_value = {
            "rx_array": mock_scene.rx_array,
            "tx_array": mock_scene.tx_array,
            "bandwidth": mock_scene.bandwidth,
            "frequency": mock_scene.frequency,
        }

        params = sionna_exporter.export_scene_rt_params(mock_scene, samples_per_src=100)
        assert params["frequency"] == 28e9
        assert params["raytracer_version"] == "0.19.1"
        assert params["samples_per_src"] == 100


@patch("deepmimo.exporters.sionna_exporter.get_sionna_version")
@patch("deepmimo.exporters.sionna_exporter.save_pickle")
@patch("os.makedirs")
@patch("deepmimo.exporters.sionna_exporter.is_sionna_v1")
def test_sionna_exporter_flow(mock_v1, mock_makedirs, mock_save, mock_ver, mock_scene, mock_paths):
    mock_v1.return_value = False
    mock_ver.return_value = "0.19.1"
    with (
        patch("deepmimo.exporters.sionna_exporter.export_paths") as mock_ep,
        patch("deepmimo.exporters.sionna_exporter.export_scene_materials") as mock_esm,
        patch("deepmimo.exporters.sionna_exporter.export_scene_rt_params") as mock_esrp,
        patch("deepmimo.exporters.sionna_exporter.export_scene_buildings") as mock_esb,
    ):
        mock_ep.return_value = [{"path": 1}]
        mock_esm.return_value = ([{"mat": 1}], [0])
        mock_esrp.return_value = {"param": 1}
        mock_esb.return_value = (np.zeros((1, 3)), {})

        sionna_exporter.sionna_exporter(mock_scene, mock_paths, {}, "out_dir")

        assert mock_save.call_count == 6

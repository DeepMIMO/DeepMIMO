"""Tests for Sionna Exporter (Sionna 2.0)."""

import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Mock sionna before importing exporter
mock_sionna = MagicMock()
mock_sionna.rt.Paths = MagicMock
mock_sionna.rt.Scene = MagicMock
sys.modules["sionna"] = mock_sionna
sys.modules["sionna.rt"] = mock_sionna.rt

from deepmimo.exporters import sionna_exporter  # noqa: E402


@pytest.fixture
def mock_scene():
    """Create a mocked Sionna 2.0 scene object."""
    scene = MagicMock()
    scene.scene_objects = {}

    rx_array = MagicMock()
    rx_array.positions.return_value.numpy.return_value = np.zeros((2, 3))
    rx_array.array_size = 2
    rx_array.num_ant = 2

    tx_array = MagicMock()
    tx_array.positions.return_value.numpy.return_value = np.zeros((1, 3))
    tx_array.array_size = 1
    tx_array.num_ant = 1

    scene.rx_array = rx_array
    scene.tx_array = tx_array
    scene.bandwidth.numpy.return_value = 1e9
    scene.frequency.numpy.return_value = 28e9
    scene.wavelength = 0.0107  # ~28 GHz

    return scene


@pytest.fixture
def mock_paths():
    """Create mocked Sionna 2.0 path attributes."""
    paths = MagicMock()
    paths.tau.numpy.return_value = np.array([1.0])
    paths.phi_r.numpy.return_value = np.array([0.0])
    paths.phi_t.numpy.return_value = np.array([0.0])
    paths.theta_r.numpy.return_value = np.array([0.0])
    paths.theta_t.numpy.return_value = np.array([0.0])
    # a is a (real, imag) tensor pair in Sionna 2.0
    paths.a = (MagicMock(numpy=lambda: np.array([1.0])), MagicMock(numpy=lambda: np.array([0.0])))
    paths.interactions.numpy.return_value = np.array([0])

    paths.__dir__ = Mock(
        return_value=["tau", "phi_r", "phi_t", "theta_r", "theta_t", "a", "interactions"]
    )
    return paths


@patch("deepmimo.exporters.sionna_exporter._paths_to_dict")
def test_export_paths(mock_p2d, mock_paths) -> None:
    """Export Sionna 2.0 paths to dictionaries."""
    mock_p2d.return_value = {
        "tau": MagicMock(numpy=lambda: np.array([1.0])),
        "phi_r": MagicMock(numpy=lambda: np.array([0.0])),
        "phi_t": MagicMock(numpy=lambda: np.array([0.0])),
        "theta_r": MagicMock(numpy=lambda: np.array([0.0])),
        "theta_t": MagicMock(numpy=lambda: np.array([0.0])),
        "interactions": MagicMock(numpy=lambda: np.array([0])),
        "sources": MagicMock(numpy=lambda: np.zeros((3, 1))),
        "targets": MagicMock(numpy=lambda: np.zeros((3, 5))),
        "vertices": MagicMock(numpy=lambda: np.zeros((5, 1, 10, 3))),
        "a": (MagicMock(numpy=lambda: np.array([1.0])), MagicMock(numpy=lambda: np.array([0.0]))),
    }

    dicts = sionna_exporter.export_paths(mock_paths)
    assert len(dicts) == 1
    assert dicts[0]["tau"][0] == 1.0
    assert "interactions" in dicts[0]
    assert "types" not in dicts[0]
    # a should be complex
    assert np.iscomplexobj(dicts[0]["a"])


def test_export_scene_materials(mock_scene) -> None:
    """Export scene materials and indices."""
    mat = MagicMock()
    mat.name = "Mat1"
    mat.conductivity.numpy.return_value = 1.0
    mat.relative_permittivity.numpy.return_value = 2.0
    mat.scattering_coefficient.numpy.return_value = 0.5
    mat.xpd_coefficient.numpy.return_value = 0.0
    mat.scattering_pattern = MagicMock()

    obj = MagicMock()
    obj.radio_material = mat
    mock_scene.scene_objects = {"obj1": obj}

    mats_list, indices = sionna_exporter.export_scene_materials(mock_scene)
    assert len(mats_list) == 1
    assert mats_list[0]["name"] == "Mat1"
    assert len(indices) == 1


@patch("deepmimo.exporters.sionna_exporter.get_sionna_version")
def test_export_scene_rt_params(mock_ver, mock_scene) -> None:
    """Export Sionna 2.0 scene parameters."""
    mock_ver.return_value = "2.0.1"
    with patch("deepmimo.exporters.sionna_exporter._scene_to_dict") as mock_s2d:
        mock_s2d.return_value = {
            "rx_array": mock_scene.rx_array,
            "tx_array": mock_scene.tx_array,
            "bandwidth": mock_scene.bandwidth,
            "frequency": mock_scene.frequency,
        }

        params = sionna_exporter.export_scene_rt_params(mock_scene, samples_per_src=100)
        assert params["frequency"] == 28e9
        assert params["raytracer_version"] == "2.0.1"
        assert params["samples_per_src"] == 100
        # Aliases for converter compatibility
        assert "num_samples" in params
        assert params["num_samples"] == 100
        assert "reflection" in params
        assert "scattering" in params


@patch("deepmimo.exporters.sionna_exporter.save_pickle")
def test_sionna_exporter_flow(mock_save, mock_scene, mock_paths) -> None:
    """End-to-end export flow for Sionna 2.0 exporter."""
    with (
        patch("deepmimo.exporters.sionna_exporter.get_sionna_version") as mock_ver,
        patch("os.makedirs"),
        patch("deepmimo.exporters.sionna_exporter.export_paths") as mock_ep,
        patch("deepmimo.exporters.sionna_exporter.export_scene_materials") as mock_esm,
        patch("deepmimo.exporters.sionna_exporter.export_scene_rt_params") as mock_esrp,
        patch("deepmimo.exporters.sionna_exporter.export_scene_buildings") as mock_esb,
    ):
        mock_ver.return_value = "2.0.1"
        mock_ep.return_value = [{"path": 1}]
        mock_esm.return_value = ([{"mat": 1}], [0])
        mock_esrp.return_value = {"param": 1}
        mock_esb.return_value = (np.zeros((1, 3)), {})

        sionna_exporter.sionna_exporter(mock_scene, mock_paths, {}, "out_dir")

        assert mock_save.call_count == 6

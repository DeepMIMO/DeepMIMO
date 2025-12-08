"""Tests for DeepMIMO Web Export."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

from deepmimo import web_export


class TestWebExport(unittest.TestCase):
    @patch("deepmimo.web_export.Path")
    @patch("deepmimo.web_export.json.dump")
    @patch("deepmimo.web_export._process_single_dataset_to_binary")
    def test_export_dataset_to_binary_single(self, mock_process, mock_json, mock_path) -> None:
        dataset = MagicMock()
        # Simulate single dataset by removing 'datasets' attribute
        if hasattr(dataset, "datasets"):
            del dataset.datasets

        mock_process.return_value = {"info": "test"}

        web_export.export_dataset_to_binary(dataset, "MyScenario")

        mock_process.assert_called_once()
        mock_json.assert_called_once()

    @patch("deepmimo.web_export.Path")
    @patch("deepmimo.web_export.json.dump")
    @patch("deepmimo.web_export._process_macro_dataset")
    def test_export_dataset_to_binary_macro(self, mock_process, mock_json, mock_path) -> None:
        dataset = MagicMock()
        dataset.datasets = [MagicMock()]

        mock_process.return_value = [{"info": "test"}]

        web_export.export_dataset_to_binary(dataset, "MyScenario")

        mock_process.assert_called_once()
        mock_json.assert_called_once()

    @patch("deepmimo.web_export._save_binary_array")
    @patch("deepmimo.web_export._process_scene_to_binary")
    def test_process_single_dataset(self, mock_scene_proc, mock_save) -> None:
        dataset = MagicMock()
        dataset.rx_pos = np.zeros((10, 3))
        dataset.tx_pos = np.zeros((1, 3))
        dataset.inter = np.zeros((10, 2))  # 2 paths
        dataset.inter_pos = np.zeros((10, 2, 1, 3))

        # Test processing
        res = web_export._process_single_dataset_to_binary(dataset, Path(), 1, 1)

        assert res["totalUsers"] == 10
        mock_save.assert_called()  # Should save rx_pos, tx_pos, etc.
        mock_scene_proc.assert_called()

    @patch("deepmimo.web_export._save_binary_array")
    def test_process_scene(self, mock_save) -> None:
        scene = MagicMock()

        # Mock objects
        mock_obj = MagicMock()
        mock_face = MagicMock()
        mock_face.vertices = [np.zeros(3), np.zeros(3), np.zeros(3)]  # List of vertices
        mock_obj.faces = [mock_face]

        scene.get_objects.return_value = [mock_obj]

        web_export._process_scene_to_binary(scene, Path())

        # Should call save for buildings, terrain, vegetation
        # Since get_objects returns list for all calls (mock default), it saves all.
        assert mock_save.call_count >= 1

    def test_save_binary_array(self) -> None:
        arr = np.array([1.0, 2.0], dtype=np.float32)
        with patch.object(Path, "open", mock_open()) as mock_file:
            web_export._save_binary_array(arr, "test.bin")
            mock_file.assert_called_with("wb")
            # Check writes: dtype, shape dims, shape values, data
            handle = mock_file()
            assert handle.write.called

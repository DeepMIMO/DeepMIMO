"""Tests for Sionna RT converter orchestrator."""

from unittest.mock import MagicMock, patch

from deepmimo.converters.sionna_rt.sionna_converter import sionna_rt_converter


def test_sionna_rt_converter_flow() -> None:
    """Verify the full orchestration: all sub-readers called, result returned."""
    base = "deepmimo.converters.sionna_rt.sionna_converter"
    with (
        patch("pathlib.Path.mkdir"),
        patch("shutil.rmtree"),
        patch(f"{base}.cu") as mock_cu,
        patch(f"{base}.read_scene") as mock_read_scene,
        patch(f"{base}.read_materials") as mock_read_materials,
        patch(f"{base}.read_paths") as mock_read_paths,
        patch(f"{base}.read_txrx") as mock_read_txrx,
        patch(f"{base}.read_rt_params") as mock_read_rt_params,
    ):
        mock_cu.check_scenario_exists.return_value = True
        mock_read_rt_params.return_value = {"raytracer_version": "2.0.1"}
        mock_read_txrx.return_value = {}
        mock_read_materials.return_value = ({}, {})

        mock_scene_obj = MagicMock()
        mock_scene_obj.export_data.return_value = {}
        mock_read_scene.return_value = mock_scene_obj

        rt_folder = "/path/to/rt_folder"
        result = sionna_rt_converter(rt_folder, scenario_name="test_scen")

        assert result == "test_scen"
        mock_read_rt_params.assert_called_once_with(rt_folder)
        mock_read_txrx.assert_called_once()
        mock_read_paths.assert_called_once()
        mock_read_materials.assert_called_once()
        mock_read_scene.assert_called_once()
        mock_cu.save_params.assert_called_once()
        mock_cu.save_scenario.assert_called_once()


def test_sionna_rt_converter_returns_none_when_scenario_exists() -> None:
    """Converter returns None when check_scenario_exists returns False."""
    base = "deepmimo.converters.sionna_rt.sionna_converter"
    with (
        patch(f"{base}.cu") as mock_cu,
        patch(f"{base}.read_rt_params"),
    ):
        mock_cu.check_scenario_exists.return_value = False
        result = sionna_rt_converter("/path/to/rt_folder")
        assert result is None

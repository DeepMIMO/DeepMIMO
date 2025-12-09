"""Tests for AODT Converter."""

from unittest.mock import patch

from deepmimo.converters.aodt import aodt_converter


def test_aodt_rt_converter() -> None:
    """Execute the AODT RT converter with mocked dependencies."""
    base = "deepmimo.converters.aodt.aodt_converter"
    with (
        patch("pathlib.Path.mkdir") as mock_mkdir,
        patch(f"{base}.shutil.rmtree") as mock_rmtree,
        patch(f"{base}.cu") as mock_cu,
        patch(f"{base}.read_materials") as mock_read_mat,
        patch(f"{base}.read_paths") as mock_read_paths,
        patch(f"{base}.read_txrx") as mock_read_txrx,
        patch(f"{base}.read_rt_params") as mock_read_params,
    ):
        mock_cu.check_scenario_exists.return_value = True
        mock_read_params.return_value = {}
        mock_read_txrx.return_value = {}
        mock_read_mat.return_value = {}

        res = aodt_converter.aodt_rt_converter("/dummy/path", vis_scene=False)

        assert res == "path"  # basename of /dummy/path
        mock_read_paths.assert_called()
        mock_cu.save_params.assert_called()
        mock_cu.save_scenario.assert_called()
        mock_mkdir.assert_called()
        mock_rmtree.assert_not_called()

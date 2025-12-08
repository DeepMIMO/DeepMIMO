"""Tests for AODT Converter."""

from unittest.mock import patch
from deepmimo.converters.aodt import aodt_converter


@patch("deepmimo.converters.aodt.aodt_converter.read_rt_params")
@patch("deepmimo.converters.aodt.aodt_converter.read_txrx")
@patch("deepmimo.converters.aodt.aodt_converter.read_paths")
@patch("deepmimo.converters.aodt.aodt_converter.read_materials")
@patch("deepmimo.converters.aodt.aodt_converter.cu")
@patch("deepmimo.converters.aodt.aodt_converter.os.makedirs")
@patch("deepmimo.converters.aodt.aodt_converter.shutil.rmtree")
def test_aodt_rt_converter(
    mock_rmtree,
    mock_makedirs,
    mock_cu,
    mock_read_mat,
    mock_read_paths,
    mock_read_txrx,
    mock_read_params,
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

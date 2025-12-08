"""Tests for Wireless Insite Converter."""

from unittest.mock import MagicMock, patch

from deepmimo.converters.wireless_insite import insite_converter
from deepmimo.scene import Scene


@patch("deepmimo.converters.wireless_insite.insite_converter.read_rt_params")
@patch("deepmimo.converters.wireless_insite.insite_converter.read_txrx")
@patch("deepmimo.converters.wireless_insite.insite_converter.read_paths")
@patch("deepmimo.converters.wireless_insite.insite_converter.read_materials")
@patch("deepmimo.converters.wireless_insite.insite_converter.read_scene")
@patch("deepmimo.converters.wireless_insite.insite_converter.cu")
@patch("shutil.rmtree")
@patch("os.makedirs")
def test_insite_rt_converter(
    mock_makedirs,
    mock_rmtree,
    mock_cu,
    mock_read_scene,
    mock_read_mats,
    mock_read_paths,
    mock_read_txrx,
    mock_read_params,
    tmp_path,
):
    # Setup mocks
    mock_cu.check_scenario_exists.return_value = True
    mock_read_params.return_value = {"freq": 28e9}
    mock_read_txrx.return_value = {}
    mock_read_mats.return_value = {}

    mock_scene = MagicMock(spec=Scene)
    mock_scene.export_data.return_value = {}
    mock_read_scene.return_value = mock_scene

    # Call converter
    res = insite_converter.insite_rt_converter(str(tmp_path), vis_scene=False, print_params=False)

    assert res == tmp_path.name.lower()
    mock_read_paths.assert_called()
    mock_cu.save_params.assert_called()
    mock_cu.save_scenario.assert_called()

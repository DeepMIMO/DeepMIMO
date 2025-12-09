"""Tests for Wireless Insite Converter."""

from unittest.mock import MagicMock, patch

from deepmimo.converters.wireless_insite import insite_converter
from deepmimo.scene import Scene


def test_insite_rt_converter(tmp_path) -> None:
    """Run the full InSite RT converter with mocked dependencies."""
    base = "deepmimo.converters.wireless_insite.insite_converter"
    with (
        patch("os.makedirs") as mock_makedirs,
        patch("shutil.rmtree") as mock_rmtree,
        patch(f"{base}.cu") as mock_cu,
        patch(f"{base}.read_scene") as mock_read_scene,
        patch(f"{base}.read_materials") as mock_read_mats,
        patch(f"{base}.read_paths") as mock_read_paths,
        patch(f"{base}.read_txrx") as mock_read_txrx,
        patch(f"{base}.read_rt_params") as mock_read_params,
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
        res = insite_converter.insite_rt_converter(
            str(tmp_path), vis_scene=False, print_params=False
        )

        assert res == tmp_path.name.lower()
        mock_read_paths.assert_called()
        mock_cu.save_params.assert_called()
        mock_cu.save_scenario.assert_called()
        mock_makedirs.assert_called()
        mock_rmtree.assert_not_called()

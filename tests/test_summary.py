"""Tests for DeepMIMO summary module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from deepmimo import consts as c

# Correctly import functions from the module
from deepmimo.summary import plot_summary, summary


class TestSummary(unittest.TestCase):
    @patch("deepmimo.summary.load_dict_from_json")
    @patch("deepmimo.summary.get_params_path")
    def test_summary(self, mock_get_path, mock_load_json) -> None:
        mock_get_path.return_value = "params.json"

        # Mock params dictionary
        mock_load_json.return_value = {
            c.RT_PARAMS_PARAM_NAME: {
                c.RT_PARAM_RAYTRACER: "InSite",
                c.RT_PARAM_RAYTRACER_VERSION: "3.3",
                c.RT_PARAM_FREQUENCY: 28e9,
                c.RT_PARAM_PATH_DEPTH: 1,
                c.RT_PARAM_MAX_REFLECTIONS: 1,
                c.RT_PARAM_MAX_DIFFRACTIONS: 0,
                c.RT_PARAM_MAX_SCATTERING: 0,
                c.RT_PARAM_MAX_TRANSMISSIONS: 0,
                c.RT_PARAM_DIFFUSE_REFLECTIONS: 0,
                c.RT_PARAM_DIFFUSE_DIFFRACTIONS: 0,
                c.RT_PARAM_DIFFUSE_TRANSMISSIONS: 0,
                c.RT_PARAM_DIFFUSE_FINAL_ONLY: False,
                c.RT_PARAM_DIFFUSE_RANDOM_PHASES: False,
                c.RT_PARAM_TERRAIN_REFLECTION: False,
                c.RT_PARAM_TERRAIN_DIFFRACTION: False,
                c.RT_PARAM_TERRAIN_SCATTERING: False,
                c.RT_PARAM_NUM_RAYS: 1000,
                c.RT_PARAM_RAY_CASTING_METHOD: "Shoot & Bounce",
                c.RT_PARAM_RAY_CASTING_RANGE_AZ: 360,
                c.RT_PARAM_RAY_CASTING_RANGE_EL: 180,
                c.RT_PARAM_SYNTHETIC_ARRAY: False,
                c.RT_PARAM_GPS_BBOX: (0, 0, 1, 1),
            },
            c.SCENE_PARAM_NAME: {
                c.SCENE_PARAM_NUMBER_SCENES: 1,
                c.SCENE_PARAM_N_OBJECTS: 10,
                c.SCENE_PARAM_N_VERTICES: 100,
                c.SCENE_PARAM_N_FACES: 50,
                c.SCENE_PARAM_N_TRIANGULAR_FACES: 50,
            },
            c.MATERIALS_PARAM_NAME: {
                "mat1": {
                    c.MATERIALS_PARAM_NAME_FIELD: "Material 1",
                    c.MATERIALS_PARAM_PERMITTIVITY: 5.0,
                    c.MATERIALS_PARAM_CONDUCTIVITY: 0.01,
                    c.MATERIALS_PARAM_SCATTERING_MODEL: "None",
                    c.MATERIALS_PARAM_SCATTERING_COEF: 0.1,
                    c.MATERIALS_PARAM_CROSS_POL_COEF: 0.1,
                }
            },
            c.TXRX_PARAM_NAME: {
                "tx_set": {
                    c.TXRX_PARAM_NAME_FIELD: "TX1",
                    c.TXRX_PARAM_IS_TX: True,
                    c.TXRX_PARAM_IS_RX: False,
                    c.TXRX_PARAM_NUM_POINTS: 1,
                    c.TXRX_PARAM_NUM_ACTIVE_POINTS: 1,
                    c.TXRX_PARAM_NUM_ANT: 1,
                    c.TXRX_PARAM_DUAL_POL: False,
                },
                "rx_set": {
                    c.TXRX_PARAM_NAME_FIELD: "RX1",
                    c.TXRX_PARAM_IS_TX: False,
                    c.TXRX_PARAM_IS_RX: True,
                    c.TXRX_PARAM_NUM_POINTS: 10,
                    c.TXRX_PARAM_NUM_ACTIVE_POINTS: 10,
                    c.TXRX_PARAM_NUM_ANT: 1,
                    c.TXRX_PARAM_DUAL_POL: False,
                },
            },
        }

        # Test string output
        res = summary("MyScenario", print_summary=False)
        assert "DeepMIMO MyScenario Scenario Summary" in res
        assert "Frequency: 28.0 GHz" in res
        assert "Total number of receivers: 10" in res

    @patch("deepmimo.summary.plt")
    @patch("pathlib.Path.mkdir")
    def test_plot_summary(self, mock_mkdir, mock_plt) -> None:
        # Mock dataset
        mock_ds = MagicMock()
        mock_ds.scene.plot = MagicMock()
        mock_ds.txrx_sets = [
            MagicMock(is_tx=True, num_points=1, id=0, is_rx=False),
            MagicMock(is_tx=False, is_rx=True, id=1, num_points=10),
        ]
        mock_ds.txrx = [{"rx_set_id": 1}]  # Map to rx set
        mock_ds.bs_pos = np.array([[0, 0, 10]])
        mock_ds.rx_pos = np.zeros((10, 3))
        mock_ds.n_ue = 10
        mock_ds.has_valid_grid.return_value = False

        # Configure returned dataset from indexing (mock_ds[0])
        sub_ds = MagicMock()
        sub_ds.n_ue = 10
        sub_ds.rx_pos = np.zeros((10, 3))
        sub_ds.has_valid_grid.return_value = False
        mock_ds.__getitem__.return_value = sub_ds

        # Test plotting
        res = plot_summary("MyScenario", dataset=mock_ds, save_imgs=True)
        assert len(res) == 2  # 2 images
        mock_ds.scene.plot.assert_called()
        mock_plt.savefig.assert_called()

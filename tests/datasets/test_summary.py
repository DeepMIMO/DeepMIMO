"""Tests for DeepMIMO summary module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deepmimo import consts as c
from deepmimo.datasets.dataset import Dataset, DynamicDataset, MacroDataset
from deepmimo.datasets.stats import (
    _STATS_REQUIRED_MATRICES,
    _compute_delay_stats,
    _compute_path_stats,
)
from deepmimo.datasets.summary import plot_summary, stats, summary


class TestSummary(unittest.TestCase):
    """Unit tests for dataset summary reporting."""

    @patch("deepmimo.datasets.summary.load_dict_from_json")
    @patch("deepmimo.datasets.summary.get_params_path")
    def test_summary(self, mock_get_path, mock_load_json) -> None:
        """Render textual summary from mocked params."""
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

    @patch("deepmimo.datasets.summary.plt")
    @patch("pathlib.Path.mkdir")
    def test_plot_summary(self, mock_mkdir, mock_plt) -> None:
        """Render summary plots and ensure images are saved."""
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
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch(
        "deepmimo.datasets.stats._compute_scene_stats",
        return_value={"scene": {}, "objects": {}, "buildings": None, "terrain": None},
    )
    @patch("deepmimo.datasets.stats._format_stats", return_value="formatted stats")
    @patch("deepmimo.datasets.stats._compute_stats", return_value={"path": {}})
    @patch("deepmimo.datasets.load.load")
    def test_stats_calls_targeted_load(
        self, mock_load, mock_compute, mock_format, mock_scene_stats
    ) -> None:
        """Stats should load only the requested TX/RX pair."""
        mock_load.return_value = Dataset(
            {"txrx": {"tx_set_id": 4, "tx_idx": 0, "rx_set_id": 1}, "scene": object()}
        )

        out = stats("o1_3p5", tx_sets=[4], rx_sets=[1], print_summary=False)

        assert out == "formatted stats"
        mock_load.assert_called_once_with(
            "o1_3p5",
            tx_sets=[4],
            rx_sets=[1],
            matrices=list(_STATS_REQUIRED_MATRICES),
        )
        mock_compute.assert_called_once()
        mock_format.assert_called_once()
        mock_scene_stats.assert_called_once()

    @patch(
        "deepmimo.datasets.stats._compute_scene_stats",
        return_value={"scene": {}, "objects": {}, "buildings": None, "terrain": None},
    )
    @patch("deepmimo.datasets.stats._format_stats", return_value="formatted stats")
    @patch("deepmimo.datasets.stats._compute_stats", return_value={"path": {}})
    @patch("deepmimo.datasets.load.load")
    def test_stats_uses_first_dynamic_snapshot(
        self, mock_load, mock_compute, mock_format, mock_scene_stats
    ) -> None:
        """Stats should use snapshot 1 for DynamicDataset inputs."""
        selected = Dataset(
            {"txrx": {"tx_set_id": 4, "tx_idx": 0, "rx_set_id": 1}, "scene": object()}
        )
        snapshot = MacroDataset([selected])
        snapshot.name = "snap_0"
        dynamic = DynamicDataset([snapshot], name="o1_3p5")
        mock_load.return_value = dynamic

        stats("o1_3p5", tx_sets=[4], rx_sets=[1], print_summary=False)

        compute_arg = mock_compute.call_args[0][0]
        assert compute_arg is selected
        mock_format.assert_called_once()
        mock_scene_stats.assert_called_once()

    @patch(
        "deepmimo.datasets.stats._compute_scene_stats",
        return_value={"scene": {}, "objects": {}, "buildings": None, "terrain": None},
    )
    @patch("deepmimo.datasets.stats._format_stats", return_value="formatted stats")
    @patch("deepmimo.datasets.stats._compute_stats", return_value={"path": {}})
    @patch("deepmimo.datasets.load.load")
    def test_stats_raises_when_no_dataset_matches_selectors(
        self, mock_load, mock_compute, mock_format, mock_scene_stats
    ) -> None:
        """Stats should raise clear error when no pair matches requested selectors."""
        mock_load.return_value = Dataset(
            {"txrx": {"tx_set_id": 9, "tx_idx": 1, "rx_set_id": 2}, "scene": object()}
        )

        with pytest.raises(ValueError, match="has no datasets matching") as exc:
            stats("o1_3p5", tx_sets=[4], rx_sets=[1], print_summary=False)
        assert "has no datasets matching" in str(exc.value)
        mock_compute.assert_not_called()
        mock_format.assert_not_called()
        mock_scene_stats.assert_not_called()

    @patch(
        "deepmimo.datasets.stats._compute_scene_stats",
        return_value={"scene": {}, "objects": {}, "buildings": None, "terrain": None},
    )
    @patch("deepmimo.datasets.stats._format_stats", return_value="formatted stats")
    @patch("deepmimo.datasets.stats._compute_stats", return_value={"path": {}})
    @patch("deepmimo.datasets.load.load")
    def test_stats_uses_load_default_rx_sets_when_omitted(
        self, mock_load, mock_compute, mock_format, mock_scene_stats
    ) -> None:
        """Stats should preserve the loader default RX selection when omitted."""
        mock_load.return_value = Dataset(
            {"txrx": {"tx_set_id": 4, "tx_idx": 0, "rx_set_id": 3}, "scene": object()}
        )

        out = stats("o1_3p5", tx_sets=[4], print_summary=False)

        assert out == "formatted stats"
        mock_load.assert_called_once_with(
            "o1_3p5",
            tx_sets=[4],
            matrices=list(_STATS_REQUIRED_MATRICES),
        )
        mock_compute.assert_called_once()
        mock_format.assert_called_once()
        mock_scene_stats.assert_called_once()

    @patch(
        "deepmimo.datasets.stats._compute_scene_stats",
        return_value={"scene": {}, "objects": {}, "buildings": None, "terrain": None},
    )
    @patch("deepmimo.datasets.stats._format_stats", return_value="formatted stats")
    @patch("deepmimo.datasets.stats._compute_stats", return_value={"path": {}})
    @patch("deepmimo.datasets.load.load")
    def test_stats_supports_multiple_tx_sets(
        self, mock_load, mock_compute, mock_format, mock_scene_stats
    ) -> None:
        """Stats should accept multiple TX sets and report each pair separately."""
        shared_scene = object()
        d1 = Dataset(
            {"txrx": {"tx_set_id": 4, "tx_idx": 0, "rx_set_id": 0}, "scene": shared_scene}
        )
        d2 = Dataset(
            {"txrx": {"tx_set_id": 5, "tx_idx": 0, "rx_set_id": 0}, "scene": shared_scene}
        )
        mock_load.return_value = MacroDataset([d1, d2])

        out = stats("o1_3p5", tx_sets=[4, 5], rx_sets=[0], print_summary=False)

        assert "[TXset 4 (tx_idx 0) | RXset 0]" in out
        assert "[TXset 5 (tx_idx 0) | RXset 0]" in out
        mock_load.assert_called_once_with(
            "o1_3p5",
            tx_sets=[4, 5],
            rx_sets=[0],
            matrices=list(_STATS_REQUIRED_MATRICES),
        )
        assert mock_compute.call_count == 2
        assert mock_compute.call_args_list[0][0][0] is d1
        assert mock_compute.call_args_list[1][0][0] is d2
        assert mock_format.call_count == 2
        mock_scene_stats.assert_called_once()

    def test_compute_path_stats_uses_lazy_num_interactions(self) -> None:
        """Path stats should trigger lazy num_interactions computation."""
        ds = Dataset(
            {
                c.AOA_AZ_PARAM_NAME: np.array([[0.0, 1.0, np.nan]]),
                c.INTERACTIONS_PARAM_NAME: np.array([[0.0, 12.0, np.nan]]),
                c.RX_POS_PARAM_NAME: np.array([[0.0, 0.0, 0.0]]),
                c.TX_POS_PARAM_NAME: np.array([0.0, 0.0, 1.0]),
            }
        )

        path_stats, _context = _compute_path_stats(ds)

        assert path_stats["avg_interactions_per_path"] == pytest.approx(1.0)
        assert path_stats["max_interactions"] == 2

    def test_compute_delay_stats_uses_lazy_power_linear(self) -> None:
        """Delay stats should trigger lazy power_linear computation for RMS values."""
        ds = Dataset(
            {
                c.DELAY_PARAM_NAME: np.array([[1e-9, 2e-9, np.nan], [5e-9, np.nan, np.nan]]),
                c.POWER_PARAM_NAME: np.array([[0.0, -3.0, np.nan], [0.0, np.nan, np.nan]]),
                c.RX_POS_PARAM_NAME: np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            }
        )

        delay_stats = _compute_delay_stats(ds)

        assert delay_stats["avg_rms_delay_ns"] == pytest.approx(0.4715905974)
        assert delay_stats["max_rms_delay_ns"] == pytest.approx(0.4715905974)

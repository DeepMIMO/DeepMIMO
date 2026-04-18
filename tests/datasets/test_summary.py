"""Tests for DeepMIMO summary module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.spatial import QhullError

from deepmimo import consts as c
from deepmimo.datasets.dataset import Dataset, DynamicDataset, MacroDataset
from deepmimo.datasets.stats import (
    _POWER_STATS_KEYS,
    _SPATIAL_STATS_KEYS,
    _STATS_REQUIRED_MATRICES,
    _building_volume,
    _compute_building_stats,
    _compute_channel_stats,
    _compute_coverage_stats,
    _compute_delay_stats,
    _compute_descriptive_stats,
    _compute_path_stats,
    _compute_power_stats,
    _compute_rms_delays,
    _compute_scene_stats,
    _compute_stats,
    _compute_terrain_stats,
    _empty_descriptive_stats,
    _empty_path_stats_and_context,
    _footprint_polygon_and_area,
    _format_object_distribution,
    _format_optional_line,
    _format_optional_pair,
    _format_section_lines,
    _format_stats,
    _resolve_stats_datasets,
    _robust_stats,
    _selector_matches_id,
    stats,
)
from deepmimo.datasets.summary import plot_summary, summary


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
    @patch("deepmimo.datasets.stats.load")
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
    @patch("deepmimo.datasets.stats.load")
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
    @patch("deepmimo.datasets.stats.load")
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
    @patch("deepmimo.datasets.stats.load")
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
    @patch("deepmimo.datasets.stats.load")
    def test_stats_supports_multiple_tx_sets(
        self, mock_load, mock_compute, mock_format, mock_scene_stats
    ) -> None:
        """Stats should accept multiple TX sets and report each pair separately."""
        shared_scene = object()
        d1 = Dataset({"txrx": {"tx_set_id": 4, "tx_idx": 0, "rx_set_id": 0}, "scene": shared_scene})
        d2 = Dataset({"txrx": {"tx_set_id": 5, "tx_idx": 0, "rx_set_id": 0}, "scene": shared_scene})
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


def test_robust_stats_symmetric() -> None:
    """Symmetric distribution should return zero IQR and equal p10/p90 distance."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    median, iqr, p10, p90 = _robust_stats(values)
    assert median == pytest.approx(3.0)
    assert iqr == pytest.approx(2.0)
    assert p10 == pytest.approx(1.4)
    assert p90 == pytest.approx(4.6)


def test_robust_stats_single_value() -> None:
    """Single-element array should return zero IQR and equal percentiles."""
    values = np.array([7.0])
    median, iqr, p10, p90 = _robust_stats(values)
    assert median == pytest.approx(7.0)
    assert iqr == pytest.approx(0.0)
    assert p10 == pytest.approx(7.0)
    assert p90 == pytest.approx(7.0)


def test_compute_rms_delays_single_path_per_user_returns_empty() -> None:
    """Users with only one valid path have zero delay spread — excluded from result."""
    delays = np.array([[1e-9, np.nan], [2e-9, np.nan]])
    powers = np.array([[1.0, np.nan], [1.0, np.nan]])
    result = _compute_rms_delays(delays, powers, n_ue=2)
    assert result == []


def test_compute_rms_delays_equal_weight_two_paths() -> None:
    """Equal-power paths symmetric around mean should produce expected RMS spread."""
    delays = np.array([[1e-9, 3e-9]])
    powers = np.array([[1.0, 1.0]])
    result = _compute_rms_delays(delays, powers, n_ue=1)
    assert len(result) == 1
    assert result[0] == pytest.approx(1e-9, rel=1e-5)


def test_compute_rms_delays_zero_sum_power_excluded() -> None:
    """Users where all paths have zero linear power should be excluded."""
    delays = np.array([[1e-9, 2e-9], [1e-9, 2e-9]])
    powers = np.array([[0.0, 0.0], [1.0, 1.0]])
    result = _compute_rms_delays(delays, powers, n_ue=2)
    assert len(result) == 1  # Only second user included


def test_format_optional_line_with_value() -> None:
    """Present values should be formatted with label, value, unit."""
    line = _format_optional_line("Avg delay", 12.34, unit=" ns", precision=1)
    assert line == "- Avg delay: 12.3 ns\n"


def test_format_optional_line_none() -> None:
    """Missing values should render as N/A."""
    line = _format_optional_line("Avg delay", None, unit=" ns")
    assert line == "- Avg delay: N/A\n"


def test_format_optional_pair_both_present() -> None:
    """Both values present should produce '<first>/<second><unit>'."""
    result = _format_optional_pair(1.0, 9.0, unit=" dB", precision=1)
    assert result == "1.0/9.0 dB"


def test_format_optional_pair_one_none() -> None:
    """Either value being None should produce N/A."""
    assert _format_optional_pair(None, 9.0, unit=" dB") == "N/A"
    assert _format_optional_pair(1.0, None, unit=" dB") == "N/A"


# ---------------------------------------------------------------------------
# _empty_descriptive_stats (line 189)
# ---------------------------------------------------------------------------


def test_empty_descriptive_stats_returns_all_none() -> None:
    """All output keys should be None for empty descriptive stats."""
    keys = {"avg": "avg_pathloss", "min": "min_pathloss", "max": "max_pathloss"}
    result = _empty_descriptive_stats(keys)
    assert result == {"avg_pathloss": None, "min_pathloss": None, "max_pathloss": None}


def test_empty_descriptive_stats_uses_power_keys() -> None:
    """_empty_descriptive_stats with _POWER_STATS_KEYS should have all None values."""
    result = _empty_descriptive_stats(_POWER_STATS_KEYS)
    assert all(v is None for v in result.values())
    assert set(result.keys()) == set(_POWER_STATS_KEYS.values())


# ---------------------------------------------------------------------------
# _compute_descriptive_stats with actual values (lines 199-214)
# ---------------------------------------------------------------------------


def test_compute_descriptive_stats_with_values() -> None:
    """_compute_descriptive_stats should return correct statistics for numeric input."""
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    result = _compute_descriptive_stats(values, keys=_POWER_STATS_KEYS)
    assert result["avg_pathloss"] == pytest.approx(30.0)
    assert result["min_pathloss"] == pytest.approx(10.0)
    assert result["max_pathloss"] == pytest.approx(50.0)
    assert result["median_pathloss"] == pytest.approx(30.0)
    assert result["pathloss_p10"] == pytest.approx(14.0)
    assert result["pathloss_p90"] == pytest.approx(46.0)


def test_compute_descriptive_stats_with_scale() -> None:
    """Scale factor should be applied to all returned statistics."""
    values = np.array([1.0, 2.0, 3.0])
    result = _compute_descriptive_stats(values, keys=_SPATIAL_STATS_KEYS, scale=10.0)
    assert result["avg_distance_bs"] == pytest.approx(20.0)
    assert result["min_distance_bs"] == pytest.approx(10.0)
    assert result["max_distance_bs"] == pytest.approx(30.0)


def test_compute_descriptive_stats_empty_returns_none_values() -> None:
    """All-NaN input should return None for every key."""
    values = np.array([np.nan, np.nan])
    result = _compute_descriptive_stats(values, keys=_POWER_STATS_KEYS)
    assert all(v is None for v in result.values())


# ---------------------------------------------------------------------------
# _empty_path_stats_and_context (line 219)
# ---------------------------------------------------------------------------


def test_empty_path_stats_and_context_structure() -> None:
    """_empty_path_stats_and_context should return zero-filled stats and empty context."""
    path_stats, context = _empty_path_stats_and_context()
    assert path_stats["avg_paths_per_user"] == pytest.approx(0.0)
    assert path_stats["max_paths_per_user"] == 0
    assert path_stats["min_paths_per_user"] == 0
    assert path_stats["los_percentage"] == pytest.approx(0.0)
    assert path_stats["nlos_percentage"] == pytest.approx(0.0)
    assert path_stats["no_paths_percentage"] == pytest.approx(0.0)
    assert path_stats["avg_interactions_per_path"] == pytest.approx(0.0)
    assert path_stats["max_interactions"] == 0

    assert context["total_users"] == 0
    assert context["los_users"] == 0
    assert context["num_paths"].size == 0
    assert context["valid_users"].size == 0


# ---------------------------------------------------------------------------
# _compute_path_stats with mismatched sizes (lines 244, 246-247)
# ---------------------------------------------------------------------------


def test_compute_path_stats_empty_num_paths_returns_empty() -> None:
    """Empty num_paths should trigger _empty_path_stats_and_context."""
    ds = MagicMock()
    ds.num_paths = np.array([])
    ds.los = np.array([1, 0])
    path_stats, context = _compute_path_stats(ds)
    assert path_stats["avg_paths_per_user"] == pytest.approx(0.0)
    assert context["total_users"] == 0


def test_compute_path_stats_empty_los_returns_empty() -> None:
    """Empty los array should trigger _empty_path_stats_and_context."""
    ds = MagicMock()
    ds.num_paths = np.array([1, 2])
    ds.los = np.array([])
    path_stats, context = _compute_path_stats(ds)
    assert path_stats["avg_paths_per_user"] == pytest.approx(0.0)
    assert context["total_users"] == 0


def test_compute_path_stats_mismatched_sizes_raises() -> None:
    """Mismatched num_paths and los sizes should raise ValueError."""
    ds = MagicMock()
    ds.num_paths = np.array([1, 2, 3])
    ds.los = np.array([1, 0])
    with pytest.raises(ValueError, match="inconsistent lengths"):
        _compute_path_stats(ds)


# ---------------------------------------------------------------------------
# _compute_power_stats (line 280)
# ---------------------------------------------------------------------------


def test_compute_power_stats_returns_descriptive_stats() -> None:
    """_compute_power_stats should compute correct descriptive stats from pathloss."""
    ds = MagicMock()
    ds.pathloss = np.array([80.0, 90.0, 100.0, np.nan])
    result = _compute_power_stats(ds)
    assert result["avg_pathloss"] == pytest.approx(90.0)
    assert result["min_pathloss"] == pytest.approx(80.0)
    assert result["max_pathloss"] == pytest.approx(100.0)


def test_compute_power_stats_all_nan_returns_none_values() -> None:
    """All-NaN pathloss should return None for all power stats."""
    ds = MagicMock()
    ds.pathloss = np.array([np.nan, np.nan])
    result = _compute_power_stats(ds)
    assert all(v is None for v in result.values())


# ---------------------------------------------------------------------------
# _compute_delay_stats with empty delays (lines 327, 349-357)
# ---------------------------------------------------------------------------


def test_compute_delay_stats_empty_delays_returns_none_values() -> None:
    """All-NaN delays should return all None stats."""
    ds = MagicMock()
    ds.delay = np.array([np.nan, np.nan])
    result = _compute_delay_stats(ds)
    assert result["min_delay_ns"] is None
    assert result["max_delay_ns"] is None
    assert result["avg_delay_ns"] is None
    assert result["avg_rms_delay_ns"] is None
    assert result["max_rms_delay_ns"] is None


def test_compute_delay_stats_valid_delays_no_rms() -> None:
    """Valid delays but single-path users (no RMS) should have None rms stats."""
    ds = MagicMock()
    # Two users, each with one valid path only (no RMS delay spread possible)
    ds.delay = np.array([[1e-9, np.nan], [2e-9, np.nan]])
    ds.power_linear = np.array([[1.0, np.nan], [1.0, np.nan]])
    ds.n_ue = 2
    result = _compute_delay_stats(ds)
    # min/max/avg delay should be present
    assert result["min_delay_ns"] == pytest.approx(1.0)
    assert result["max_delay_ns"] == pytest.approx(2.0)
    assert result["avg_delay_ns"] == pytest.approx(1.5)
    # rms stats should be None (no users with >1 valid path)
    assert result["avg_rms_delay_ns"] is None
    assert result["max_rms_delay_ns"] is None


# ---------------------------------------------------------------------------
# _compute_coverage_stats with zero total_users (lines 388-395)
# ---------------------------------------------------------------------------


def test_compute_coverage_stats_zero_total_users() -> None:
    """Zero total_users should return all-zero coverage stats."""
    result = _compute_coverage_stats(
        num_paths=np.array([], dtype=float),
        valid_users=np.array([], dtype=bool),
        los_users=0,
        total_users=0,
    )
    assert result["coverage_percentage"] == pytest.approx(0.0)
    assert result["los_coverage_percentage"] == pytest.approx(0.0)
    assert result["avg_paths_per_covered_user"] == pytest.approx(0.0)


def test_compute_coverage_stats_with_covered_users() -> None:
    """Compute correct percentages when some users are covered."""
    num_paths = np.array([3.0, 0.0, 2.0, 1.0])
    valid_users = num_paths > 0
    result = _compute_coverage_stats(
        num_paths=num_paths,
        valid_users=valid_users,
        los_users=1,
        total_users=4,
    )
    assert result["coverage_percentage"] == pytest.approx(75.0)
    assert result["los_coverage_percentage"] == pytest.approx(25.0)
    # Average paths among covered: (3 + 2 + 1) / 3 = 2.0
    assert result["avg_paths_per_covered_user"] == pytest.approx(2.0)


def test_compute_coverage_stats_no_covered_users() -> None:
    """No covered users: avg_paths_per_covered_user should be 0.0."""
    num_paths = np.array([0.0, 0.0, 0.0])
    valid_users = num_paths > 0
    result = _compute_coverage_stats(
        num_paths=num_paths,
        valid_users=valid_users,
        los_users=0,
        total_users=3,
    )
    assert result["coverage_percentage"] == pytest.approx(0.0)
    assert result["avg_paths_per_covered_user"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _compute_channel_stats (lines 413-414)
# ---------------------------------------------------------------------------


def test_compute_channel_stats_returns_all_sections() -> None:
    """_compute_channel_stats should return path/power/delay/coverage/spatial keys."""
    ds = MagicMock()
    ds.num_paths = np.array([2, 0, 1])
    ds.los = np.array([1, -1, 0])
    ds.num_interactions = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    ds.pathloss = np.array([np.nan])
    ds.delay = np.array([np.nan])
    ds.power_linear = np.array([np.nan])
    ds.n_ue = 3
    ds.distance = np.array([np.nan])
    result = _compute_channel_stats(ds)
    assert "path" in result
    assert "power" in result
    assert "delay" in result
    assert "coverage" in result
    assert "spatial" in result


# ---------------------------------------------------------------------------
# _footprint_polygon_and_area (lines 425-428)
# ---------------------------------------------------------------------------


def test_footprint_polygon_and_area_too_few_points_returns_empty() -> None:
    """Fewer than 3 unique 2D points should return empty polygon and zero area."""
    obj = MagicMock()
    # Only 2 unique 2D points (colinear / degenerate)
    obj.vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 5.0]])
    polygon, area = _footprint_polygon_and_area(obj)
    assert polygon.shape == (0, 2)
    assert area == pytest.approx(0.0)


def test_footprint_polygon_and_area_valid_polygon() -> None:
    """Valid non-collinear 2D points should return a non-zero area convex hull."""
    obj = MagicMock()
    # Square footprint: 4 corners at (0,0), (1,0), (1,1), (0,1), z=anything
    obj.vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    polygon, area = _footprint_polygon_and_area(obj)
    assert polygon.shape[0] >= 3
    assert area == pytest.approx(1.0, rel=1e-3)


def test_footprint_polygon_and_area_collinear_returns_empty() -> None:
    """Collinear 2D points (even if 3+) should return empty polygon and zero area."""
    obj = MagicMock()
    # All points lie on a single line in XY, z varies
    obj.vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 2.0],
            [3.0, 0.0, 3.0],
        ]
    )
    _, area = _footprint_polygon_and_area(obj)
    assert area == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _building_volume (lines 436-448)
# ---------------------------------------------------------------------------


def test_building_volume_too_few_unique_points_returns_zero() -> None:
    """Fewer than 4 unique 3D points should return 0."""
    obj = MagicMock()
    obj.vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert _building_volume(obj) == pytest.approx(0.0)


def test_building_volume_coplanar_points_returns_zero() -> None:
    """4+ coplanar points (rank < 3) should return 0."""
    obj = MagicMock()
    # All points in the XY plane — rank of (pts - pts[0]) is 2
    obj.vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )
    assert _building_volume(obj) == pytest.approx(0.0)


def test_building_volume_valid_3d_hull_returns_volume() -> None:
    """A tetrahedron should have the correct convex hull volume."""
    obj = MagicMock()
    # Regular tetrahedron-like points: volume = 1/3 * base_area * height
    # Use a unit cube subset for a simple check: just verify > 0
    obj.vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    volume = _building_volume(obj)
    assert volume == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# _compute_building_stats (lines 466-474)
# ---------------------------------------------------------------------------


def _make_building_mock(height: float, vertices: np.ndarray) -> MagicMock:
    """Create a mock building object with height, vertices and label."""
    obj = MagicMock()
    obj.height = height
    obj.vertices = vertices
    obj.label = "buildings"
    return obj


def test_compute_building_stats_empty_list_returns_none() -> None:
    """Empty building list should return None."""
    assert _compute_building_stats([], scene_total_area=1000.0) is None


def test_compute_building_stats_with_buildings() -> None:
    """Building stats should compute height/volume/footprint metrics correctly."""
    # Two buildings with simple unit-cube geometry
    cube_verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    b1 = _make_building_mock(height=10.0, vertices=cube_verts)
    b2 = _make_building_mock(height=20.0, vertices=cube_verts)
    result = _compute_building_stats([b1, b2], scene_total_area=100.0)
    assert result is not None
    assert result["avg_height"] == pytest.approx(15.0)
    assert result["min_height"] == pytest.approx(10.0)
    assert result["max_height"] == pytest.approx(20.0)
    # Each building has footprint ~1.0 m² => total 2.0, density = 2%
    assert result["total_footprint"] == pytest.approx(2.0, rel=1e-2)
    assert result["building_density"] == pytest.approx(2.0, rel=1e-2)


def test_compute_building_stats_zero_scene_area_density_zero() -> None:
    """Zero scene area should produce building_density of 0.0 (no div-by-zero)."""
    cube_verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    b = _make_building_mock(height=5.0, vertices=cube_verts)
    result = _compute_building_stats([b], scene_total_area=0.0)
    assert result is not None
    assert result["building_density"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _compute_terrain_stats (lines 485-487)
# ---------------------------------------------------------------------------


def _make_terrain_mock(height: float) -> MagicMock:
    """Create a mock terrain object with height and label."""
    obj = MagicMock()
    obj.height = height
    obj.label = "terrain"
    return obj


def test_compute_terrain_stats_empty_list_returns_none() -> None:
    """Empty terrain list should return None."""
    assert _compute_terrain_stats([]) is None


def test_compute_terrain_stats_with_terrain() -> None:
    """Terrain stats should compute height range, avg, std and elevation change."""
    t1 = _make_terrain_mock(0.0)
    t2 = _make_terrain_mock(10.0)
    t3 = _make_terrain_mock(5.0)
    result = _compute_terrain_stats([t1, t2, t3])
    assert result is not None
    assert result["min_height"] == pytest.approx(0.0)
    assert result["max_height"] == pytest.approx(10.0)
    assert result["avg_height"] == pytest.approx(5.0)
    assert result["total_elevation_change"] == pytest.approx(10.0)
    assert result["height_std"] == pytest.approx(np.std([0.0, 10.0, 5.0]))


# ---------------------------------------------------------------------------
# _compute_scene_stats via mock (lines 543, 548)
# ---------------------------------------------------------------------------


def _make_scene_dataset_mock(
    width: float = 100.0,
    length: float = 200.0,
    height: float = 50.0,
    objects: list | None = None,
) -> MagicMock:
    """Create a mock dataset with scene bounding_box and objects."""
    ds = MagicMock()
    ds.scene.bounding_box.width = width
    ds.scene.bounding_box.length = length
    ds.scene.bounding_box.height = height
    ds.scene.objects = objects if objects is not None else []
    return ds


def test_compute_scene_stats_no_objects() -> None:
    """Scene with no objects should return scene dims and None for buildings/terrain."""
    ds = _make_scene_dataset_mock()
    result = _compute_scene_stats(ds)
    assert result["scene"]["width"] == pytest.approx(100.0)
    assert result["scene"]["length"] == pytest.approx(200.0)
    assert result["scene"]["height"] == pytest.approx(50.0)
    assert result["scene"]["total_area"] == pytest.approx(20000.0)
    assert result["scene"]["total_volume"] == pytest.approx(1000000.0)
    assert result["buildings"] is None
    assert result["terrain"] is None
    assert result["objects"] == {}


def test_compute_scene_stats_with_buildings_and_terrain() -> None:
    """Scene with building and terrain objects should produce non-None stats for both."""
    cube_verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    b = _make_building_mock(height=10.0, vertices=cube_verts)
    t = _make_terrain_mock(height=2.0)
    ds = _make_scene_dataset_mock(objects=[b, t])
    result = _compute_scene_stats(ds)
    assert result["buildings"] is not None
    assert result["terrain"] is not None
    assert result["objects"] == {"buildings": 1, "terrain": 1}


# ---------------------------------------------------------------------------
# _format_section_lines with "pair" kind (lines 556-578)
# ---------------------------------------------------------------------------


def test_format_section_lines_raw_kind() -> None:
    """'raw' kind specs should be formatted with template substitution."""
    specs = (("raw", "Coverage", "{coverage_percentage:.1f}%"),)
    lines = _format_section_lines({"coverage_percentage": 75.0}, specs)
    assert lines == ["- Coverage: 75.0%\n"]


def test_format_section_lines_optional_kind_present() -> None:
    """'optional' kind with present value should render formatted number."""
    specs = (("optional", "Avg delay", "avg_delay_ns", " ns", 1),)
    lines = _format_section_lines({"avg_delay_ns": 12.3}, specs)
    assert lines == ["- Avg delay: 12.3 ns\n"]


def test_format_section_lines_optional_kind_none() -> None:
    """'optional' kind with None value should render N/A."""
    specs = (("optional", "Avg delay", "avg_delay_ns", " ns", 1),)
    lines = _format_section_lines({"avg_delay_ns": None}, specs)
    assert lines == ["- Avg delay: N/A\n"]


def test_format_section_lines_pair_kind_both_present() -> None:
    """'pair' kind with both values should render 'first/second unit'."""
    specs = (("pair", "Pathloss p10/p90", "pathloss_p10", "pathloss_p90", " dB", 1),)
    lines = _format_section_lines({"pathloss_p10": 70.0, "pathloss_p90": 110.0}, specs)
    assert lines == ["- Pathloss p10/p90: 70.0/110.0 dB\n"]


def test_format_section_lines_pair_kind_one_none() -> None:
    """'pair' kind with a None value should render N/A."""
    specs = (("pair", "Distance p10/p90", "distance_p10", "distance_p90", " m", 1),)
    lines = _format_section_lines({"distance_p10": None, "distance_p90": 500.0}, specs)
    assert lines == ["- Distance p10/p90: N/A\n"]


def test_format_section_lines_multiple_kinds() -> None:
    """Multiple spec kinds in one call should each produce a correctly formatted line."""
    specs = (
        ("raw", "LOS pct", "{los_percentage:.1f}%"),
        ("optional", "Avg PL", "avg_pathloss", " dB", 1),
        ("pair", "PL p10/p90", "pathloss_p10", "pathloss_p90", " dB", 1),
    )
    stats_dict = {
        "los_percentage": 60.0,
        "avg_pathloss": 95.0,
        "pathloss_p10": 80.0,
        "pathloss_p90": 110.0,
    }
    lines = _format_section_lines(stats_dict, specs)
    assert len(lines) == 3
    assert "60.0%" in lines[0]
    assert "95.0" in lines[1]
    assert "80.0/110.0" in lines[2]


# ---------------------------------------------------------------------------
# _format_object_distribution (lines 583-584)
# ---------------------------------------------------------------------------


def test_format_object_distribution_nonempty() -> None:
    """Object distribution should list each label with its count."""
    result = _format_object_distribution({"buildings": 5, "terrain": 2})
    assert "Object Distribution" in result
    assert "Buildings: 5" in result
    assert "Terrain: 2" in result


def test_format_object_distribution_empty() -> None:
    """Empty object counts should return section header with no bullet lines."""
    result = _format_object_distribution({})
    assert "Object Distribution" in result
    # No bullet lines should be present
    assert "- " not in result


# ---------------------------------------------------------------------------
# _format_stats with buildings and terrain sections (lines 589-608)
# ---------------------------------------------------------------------------


def _make_full_stats_dict(
    *,
    include_buildings: bool = False,
    include_terrain: bool = False,
) -> dict:
    """Build a minimal stats dict that can be passed to _format_stats."""
    path_stats = {
        "avg_paths_per_user": 2.5,
        "max_paths_per_user": 5,
        "min_paths_per_user": 0,
        "los_percentage": 40.0,
        "nlos_percentage": 50.0,
        "no_paths_percentage": 10.0,
        "avg_interactions_per_path": 1.2,
        "max_interactions": 3,
    }
    coverage_stats = {
        "coverage_percentage": 90.0,
        "los_coverage_percentage": 40.0,
        "avg_paths_per_covered_user": 2.8,
    }
    scene_stats = {
        "width": 500.0,
        "length": 500.0,
        "height": 100.0,
        "total_area": 250000.0,
        "total_volume": 25000000.0,
    }
    power_stats = dict.fromkeys(_POWER_STATS_KEYS.values())
    delay_stats = {
        "min_delay_ns": None,
        "max_delay_ns": None,
        "avg_delay_ns": None,
        "avg_rms_delay_ns": None,
        "max_rms_delay_ns": None,
    }
    spatial_stats = dict.fromkeys(_SPATIAL_STATS_KEYS.values())

    buildings = None
    if include_buildings:
        buildings = {
            "avg_height": 15.0,
            "min_height": 10.0,
            "max_height": 20.0,
            "median_height": 15.0,
            "height_iqr": 5.0,
            "height_p10": 10.5,
            "height_p90": 19.5,
            "avg_volume": 500.0,
            "total_volume": 1000.0,
            "avg_footprint": 50.0,
            "total_footprint": 100.0,
            "building_density": 0.04,
        }

    terrain = None
    if include_terrain:
        terrain = {
            "min_height": 0.0,
            "max_height": 10.0,
            "avg_height": 5.0,
            "height_std": 3.0,
            "total_elevation_change": 10.0,
        }

    return {
        "path": path_stats,
        "power": power_stats,
        "delay": delay_stats,
        "coverage": coverage_stats,
        "spatial": spatial_stats,
        "scene": scene_stats,
        "objects": {"buildings": 2} if include_buildings else {},
        "buildings": buildings,
        "terrain": terrain,
    }


def test_format_stats_no_buildings_no_terrain() -> None:
    """Format stats without optional sections should omit them."""
    result = _format_stats(_make_full_stats_dict())
    assert "Path Statistics" in result
    assert "Coverage Statistics" in result
    assert "Scene Dimensions" in result
    assert "Building Characteristics" not in result
    assert "Terrain Characteristics" not in result


def test_format_stats_with_buildings() -> None:
    """Format stats with buildings section should include Building Characteristics."""
    result = _format_stats(_make_full_stats_dict(include_buildings=True))
    assert "Building Characteristics" in result
    assert "Average height" in result
    assert "Terrain Characteristics" not in result


def test_format_stats_with_terrain() -> None:
    """Format stats with terrain section should include Terrain Characteristics."""
    result = _format_stats(_make_full_stats_dict(include_terrain=True))
    assert "Terrain Characteristics" in result
    assert "Height range" in result
    assert "Building Characteristics" not in result


def test_format_stats_with_buildings_and_terrain() -> None:
    """Format stats with both sections should include both characteristic sections."""
    result = _format_stats(_make_full_stats_dict(include_buildings=True, include_terrain=True))
    assert "Building Characteristics" in result
    assert "Terrain Characteristics" in result


# ---------------------------------------------------------------------------
# stats() with empty tx_sets / rx_sets raises ValueError (lines 765-769)
# ---------------------------------------------------------------------------


def test_stats_empty_tx_sets_list_raises() -> None:
    """Empty tx_sets list should raise ValueError before loading."""
    with pytest.raises(ValueError, match="tx_sets must be non-empty"):
        stats("any_scenario", tx_sets=[], print_summary=False)


def test_stats_empty_tx_sets_dict_raises() -> None:
    """Empty tx_sets dict should raise ValueError before loading."""
    with pytest.raises(ValueError, match="tx_sets must be non-empty"):
        stats("any_scenario", tx_sets={}, print_summary=False)


def test_stats_empty_rx_sets_list_raises() -> None:
    """Empty rx_sets list should raise ValueError before loading."""
    with pytest.raises(ValueError, match="rx_sets must be non-empty"):
        stats("any_scenario", rx_sets=[], print_summary=False)


def test_stats_empty_rx_sets_dict_raises() -> None:
    """Empty rx_sets dict should raise ValueError before loading."""
    with pytest.raises(ValueError, match="rx_sets must be non-empty"):
        stats("any_scenario", rx_sets={}, print_summary=False)


# ---------------------------------------------------------------------------
# stats() when load raises ValueError (lines 781-786)
# ---------------------------------------------------------------------------


@patch("deepmimo.datasets.stats.load")
def test_stats_load_raises_value_error_is_reraised(mock_load) -> None:
    """ValueError from load() should be caught and re-raised with context info."""
    mock_load.side_effect = ValueError("scenario not found")
    with pytest.raises(ValueError, match="Failed to load scenario"):
        stats("missing_scenario", print_summary=False)


@patch("deepmimo.datasets.stats.load")
def test_stats_load_raises_includes_original_message(mock_load) -> None:
    """Reraised ValueError should contain the original load error message."""
    mock_load.side_effect = ValueError("disk read error")
    with pytest.raises(ValueError, match="disk read error"):
        stats("missing_scenario", print_summary=False)


# ---------------------------------------------------------------------------
# stats() with print_summary=True (lines 797, 802-803) — use capsys
# ---------------------------------------------------------------------------


@patch(
    "deepmimo.datasets.stats._compute_scene_stats",
    return_value={"scene": {}, "objects": {}, "buildings": None, "terrain": None},
)
@patch("deepmimo.datasets.stats._format_stats", return_value="STATS OUTPUT")
@patch("deepmimo.datasets.stats._compute_stats", return_value={"path": {}})
@patch("deepmimo.datasets.stats.load")
def test_stats_print_summary_true_prints_and_returns_none(
    mock_load,
    _mock_compute,  # noqa: PT019
    _mock_format,  # noqa: PT019
    _mock_scene_stats,  # noqa: PT019
    capsys,
) -> None:
    """print_summary=True should print the report and return None."""
    mock_load.return_value = Dataset(
        {"txrx": {"tx_set_id": 1, "tx_idx": 0, "rx_set_id": 0}, "scene": object()}
    )
    result = stats("o1_3p5", print_summary=True)
    assert result is None
    captured = capsys.readouterr()
    assert "Calculating scenario statistics" in captured.out
    assert "STATS OUTPUT" in captured.out


@patch(
    "deepmimo.datasets.stats._compute_scene_stats",
    return_value={"scene": {}, "objects": {}, "buildings": None, "terrain": None},
)
@patch("deepmimo.datasets.stats._format_stats", return_value="STATS OUTPUT")
@patch("deepmimo.datasets.stats._compute_stats", return_value={"path": {}})
@patch("deepmimo.datasets.stats.load")
def test_stats_print_summary_false_returns_string_no_print(
    mock_load,
    _mock_compute,  # noqa: PT019
    _mock_format,  # noqa: PT019
    _mock_scene_stats,  # noqa: PT019
    capsys,
) -> None:
    """print_summary=False should return the formatted string without printing."""
    mock_load.return_value = Dataset(
        {"txrx": {"tx_set_id": 1, "tx_idx": 0, "rx_set_id": 0}, "scene": object()}
    )
    result = stats("o1_3p5", print_summary=False)
    assert result == "STATS OUTPUT"
    captured = capsys.readouterr()
    assert captured.out == ""


@patch(
    "deepmimo.datasets.stats._compute_scene_stats",
    return_value={"scene": {}, "objects": {}, "buildings": None, "terrain": None},
)
@patch("deepmimo.datasets.stats._format_stats", return_value="formatted stats")
@patch("deepmimo.datasets.stats._compute_stats", return_value={"path": {}})
@patch("deepmimo.datasets.stats.load")
def test_stats_print_summary_prints_dynamic_snapshot_message(
    mock_load,
    _mock_compute,  # noqa: PT019
    _mock_format,  # noqa: PT019
    _mock_scene_stats,  # noqa: PT019
    capsys,
) -> None:
    """print_summary=True should print the dynamic snapshot message when applicable."""
    selected = Dataset({"txrx": {"tx_set_id": 1, "tx_idx": 0, "rx_set_id": 0}, "scene": object()})
    snapshot = MacroDataset([selected])
    snapshot.name = "snap_0"
    dynamic = DynamicDataset([snapshot], name="dyn_scenario")
    mock_load.return_value = dynamic

    result = stats("dyn_scenario", print_summary=True)
    assert result is None
    captured = capsys.readouterr()
    assert "Dynamic scenario detected" in captured.out


# ---------------------------------------------------------------------------
# Additional coverage: _resolve_stats_datasets error paths (lines 629-630, 635-637)
# ---------------------------------------------------------------------------


def test_resolve_stats_datasets_empty_dynamic_raises() -> None:
    """Empty DynamicDataset should raise ValueError with a clear message."""
    empty_dynamic = DynamicDataset([], name="empty_dyn")
    with pytest.raises(ValueError, match="contains no snapshots"):
        _resolve_stats_datasets(empty_dynamic, "empty_dyn")


def test_resolve_stats_datasets_empty_macro_raises() -> None:
    """Empty MacroDataset should raise ValueError about no TX/RX pairs."""
    empty_macro = MacroDataset([])
    with pytest.raises(ValueError, match="contains no TX/RX dataset pairs"):
        _resolve_stats_datasets(empty_macro, "empty_macro")


def test_resolve_stats_datasets_single_dataset_no_messages() -> None:
    """Single Dataset should return it wrapped in a list with no messages."""
    ds = Dataset({"txrx": {"tx_set_id": 1, "tx_idx": 0, "rx_set_id": 0}, "scene": object()})
    datasets, messages = _resolve_stats_datasets(ds, "test")
    assert datasets == [ds]
    assert messages == []


def test_resolve_stats_datasets_macro_returns_message() -> None:
    """MacroDataset should return datasets list plus a count message."""
    ds1 = Dataset({"txrx": {"tx_set_id": 1, "tx_idx": 0, "rx_set_id": 0}, "scene": object()})
    ds2 = Dataset({"txrx": {"tx_set_id": 2, "tx_idx": 0, "rx_set_id": 0}, "scene": object()})
    macro = MacroDataset([ds1, ds2])
    datasets, messages = _resolve_stats_datasets(macro, "test_macro")
    assert len(datasets) == 2
    assert any("2 TX/RX pair(s)" in m for m in messages)


# ---------------------------------------------------------------------------
# Additional coverage: _selector_matches_id branches (lines 652, 654)
# ---------------------------------------------------------------------------


def test_selector_matches_id_none_selector_returns_true() -> None:
    """None selector should match any set_id."""
    assert _selector_matches_id(None, 5) is True
    assert _selector_matches_id(None, None) is True


def test_selector_matches_id_string_selector_returns_true() -> None:
    """String selector ('all') should match any set_id."""
    assert _selector_matches_id("all", 5) is True
    assert _selector_matches_id("all", None) is True


def test_selector_matches_id_none_set_id_returns_false() -> None:
    """Non-None selector with None set_id should return False."""
    assert _selector_matches_id([1, 2, 3], None) is False
    assert _selector_matches_id({1: "all", 2: "all"}, None) is False


def test_selector_matches_id_dict_selector_match() -> None:
    """Dict selector should return True when set_id is a key."""
    selector = {1: "all", 3: [0, 1]}
    assert _selector_matches_id(selector, 1) is True
    assert _selector_matches_id(selector, 3) is True
    assert _selector_matches_id(selector, 2) is False


def test_selector_matches_id_list_selector_match() -> None:
    """List selector should return True when set_id is in the list."""
    assert _selector_matches_id([1, 2, 3], 2) is True
    assert _selector_matches_id([1, 2, 3], 4) is False


# ---------------------------------------------------------------------------
# Additional coverage: _compute_stats with cached_scene_stats=None (lines 691-695)
# ---------------------------------------------------------------------------


@patch("deepmimo.datasets.stats._compute_scene_stats")
@patch("deepmimo.datasets.stats._compute_channel_stats")
def test_compute_stats_computes_scene_when_cache_is_none(mock_channel, mock_scene) -> None:
    """_compute_stats should call _compute_scene_stats when cache is None."""
    mock_channel.return_value = {"path": {}}
    mock_scene.return_value = {"scene": {}, "objects": {}, "buildings": None, "terrain": None}
    ds = MagicMock()
    result = _compute_stats(ds, cached_scene_stats=None)
    mock_scene.assert_called_once_with(ds)
    assert "scene" in result
    assert "path" in result


@patch("deepmimo.datasets.stats._compute_scene_stats")
@patch("deepmimo.datasets.stats._compute_channel_stats")
def test_compute_stats_does_not_recompute_scene_when_cached(mock_channel, mock_scene) -> None:
    """_compute_stats should not call _compute_scene_stats when cache is provided."""
    mock_channel.return_value = {"path": {}}
    cached = {"scene": {"width": 100.0}, "objects": {}, "buildings": None, "terrain": None}
    ds = MagicMock()
    result = _compute_stats(ds, cached_scene_stats=cached)
    mock_scene.assert_not_called()
    assert result["scene"] == {"width": 100.0}


# ---------------------------------------------------------------------------
# Additional coverage: _building_volume QhullError path (lines 407-408)
# ---------------------------------------------------------------------------


def test_building_volume_qhull_error_returns_zero() -> None:
    """When ConvexHull raises QhullError, _building_volume should return 0.0."""
    obj = MagicMock()
    # Provide enough non-coplanar unique points so we pass the rank check,
    # but patch ConvexHull to raise QhullError anyway.
    obj.vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    with patch("deepmimo.datasets.stats.ConvexHull", side_effect=QhullError("forced")):
        result = _building_volume(obj)
    assert result == pytest.approx(0.0)

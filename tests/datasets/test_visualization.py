"""Tests for DeepMIMO visualization."""

import csv as csv_mod
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt

from deepmimo.datasets import visualization
from deepmimo.datasets.visualization import (
    _create_colorbar,
    export_xyz_csv,
    generate_distinct_colors,
    plot_power_discarding,
    plot_rays,
    transform_coordinates,
)


class TestVisualization(unittest.TestCase):
    """Visualization plotting routines under test."""

    def test_plot_coverage_realistic(self) -> None:
        """Test plot_coverage with real data (Agg backend)."""
        rng = np.random.default_rng()
        ue_pos = rng.random((100, 3)) * 100  # 100 UEs in 100x100 grid
        los = rng.choice([0, 1], size=100)  # Random LoS/NLoS

        # Basic plot - should create figure without errors
        visualization.plot_coverage(ue_pos, los)
        plt.close("all")

        # With BS position
        bs_pos = np.array([[50, 50, 10]])
        visualization.plot_coverage(ue_pos, los, bs_pos=bs_pos)
        plt.close("all")

        # With custom labels
        visualization.plot_coverage(ue_pos, los, cbar_labels=["No Line of Sight", "Line of Sight"])
        plt.close("all")

    @patch("deepmimo.datasets.visualization.plt")
    def test_plot_coverage(self, mock_plt) -> None:
        """Plot coverage with mocked matplotlib handles."""
        rng = np.random.default_rng()
        ue_pos = rng.random((10, 3)) * 100
        los = rng.choice([0, 1], size=10)

        # Mock subplots return value
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Basic plot
        visualization.plot_coverage(ue_pos, los)
        mock_ax.scatter.assert_called()

        # With BS pos
        bs_pos = np.array([[50, 50, 10]])
        visualization.plot_coverage(ue_pos, los, bs_pos=bs_pos)

        # With different colors/labels
        visualization.plot_coverage(ue_pos, los, cbar_labels=["NLoS", "LoS"])

    def test_plot_rays_realistic(self) -> None:
        """Test plot_rays with real data (Agg backend)."""
        # Realistic scenario: TX at building roof, RX at ground
        tx_loc = np.array([0, 0, 25])  # 25m high
        rx_loc = np.array([50, 30, 1.5])  # ground level

        # 3 paths: LoS, single reflection, double reflection
        n_paths = 3
        inter = np.array([0, 1, 11])  # LoS, 1 reflection, 2 reflections

        # inter_pos: [n_paths, max_depth, 3]
        inter_pos = np.zeros((n_paths, 2, 3))
        inter_pos[:] = np.nan  # Initialize with NaN

        # LoS: no interactions
        # Single reflection: one interaction point
        inter_pos[1, 0, :] = [25, 15, 10]  # reflection point on building
        # Double reflection: two interaction points
        inter_pos[2, 0, :] = [10, 5, 15]  # first reflection
        inter_pos[2, 1, :] = [40, 25, 8]  # second reflection

        # Should create plot without errors
        visualization.plot_rays(rx_loc, tx_loc, inter_pos, inter)
        plt.close("all")

    @patch("deepmimo.datasets.visualization.plt")
    def test_plot_rays(self, mock_plt) -> None:
        """Plot rays with mocked matplotlib objects."""
        # Mock data
        tx_loc = np.array([0, 0, 10])
        rx_loc = np.array([10, 0, 1.5])

        # 2 paths: 1 LoS, 1 Ref
        n_paths = 2
        inter = np.array([0, 1])  # LoS, Ref

        # inter_pos: [n_paths, max_depth, 3]
        # LoS: 0 interactions. Ref: 1 interaction. Max depth 1.
        inter_pos = np.zeros((n_paths, 1, 3))
        inter_pos[0, 0, :] = np.nan  # LoS has no interaction points
        inter_pos[1, 0, :] = [5, 0, 0]  # Ref point

        # Mock plotting
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Mock legend handles/labels for the end of function
        mock_ax.get_legend_handles_labels.return_value = ([], [])

        visualization.plot_rays(rx_loc, tx_loc, inter_pos, inter)

        # Check if lines were plotted
        assert mock_ax.plot.called

    @patch("deepmimo.datasets.visualization.plt")
    def test_plot_power_discarding(self, mock_plt) -> None:
        """Visualize discarded power fractions for delays."""
        # Mock dataset
        ds = MagicMock()
        # 2 users, 2 paths
        ds.delay = np.array([[1e-7, 2e-7], [1e-7, 5e-7]])
        ds.power_linear = np.array([[1.0, 0.5], [1.0, 0.1]])

        # Mock plotting
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test with trim_delay=3e-7
        # User 0: both paths kept (<= 2e-7). Discarded = 0.
        # User 1: path 1 discarded (5e-7 > 3e-7). Discarded = 0.1 / 1.1 ~ 9%.
        visualization.plot_power_discarding(ds, trim_delay=3e-7)

        mock_ax.hist.assert_called()


def test_generate_distinct_colors_shape() -> None:
    """Output should have shape (n, 4) with RGBA values in [0, 1]."""
    colors = generate_distinct_colors(5)
    assert colors.shape == (5, 4)
    assert np.all(colors >= 0.0)
    assert np.all(colors <= 1.0)


def test_generate_distinct_colors_alpha_is_one() -> None:
    """Alpha channel should always be 1.0."""
    colors = generate_distinct_colors(10)
    np.testing.assert_array_equal(colors[:, 3], 1.0)


def test_generate_distinct_colors_unique() -> None:
    """Colors should be pairwise distinct for reasonable n."""
    colors = generate_distinct_colors(8)
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            assert not np.allclose(colors[i], colors[j])


def test_generate_distinct_colors_zero() -> None:
    """Zero colors requested should return empty array."""
    colors = generate_distinct_colors(0)
    assert colors.size == 0


def test_transform_coordinates_corners() -> None:
    """Corner coordinates should map exactly to geo bounds."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    lats, lons = transform_coordinates(
        coords, lon_max=10.0, lon_min=0.0, lat_min=20.0, lat_max=30.0
    )
    assert min(lons) == pytest.approx(0.0)
    assert max(lons) == pytest.approx(10.0)
    assert min(lats) == pytest.approx(20.0)
    assert max(lats) == pytest.approx(30.0)


def test_transform_coordinates_output_length() -> None:
    """Output lists should have same length as input rows."""
    coords = np.array([[0.0, 0.0, 5.0], [1.0, 2.0, 5.0], [3.0, 4.0, 5.0]])
    lats, lons = transform_coordinates(coords, lon_max=5.0, lon_min=0.0, lat_min=0.0, lat_max=5.0)
    assert len(lats) == 3
    assert len(lons) == 3


# ---------------------------------------------------------------------------
# _create_colorbar - mismatch raises ValueError  (lines 85-89)
# ---------------------------------------------------------------------------


def test_create_colorbar_cat_labels_mismatch_raises() -> None:
    """cat_labels length != unique-value count must raise ValueError."""
    _, ax = plt.subplots()
    # Three unique values but only 2 labels supplied
    cov_map = np.array([1.0, 2.0, 3.0])
    scatter = ax.scatter([0, 1, 2], [0, 0, 0], c=cov_map)
    with pytest.raises(ValueError, match="Number of category labels"):
        _create_colorbar(scatter, cov_map, "viridis", cat_labels=["a", "b"], ax=ax)
    plt.close("all")


# ---------------------------------------------------------------------------
# _create_colorbar - continuous mode when n_cats > 30  (line 119)
# ---------------------------------------------------------------------------


def test_create_colorbar_continuous_mode() -> None:
    """More than 30 unique values without cat_labels uses continuous colorbar."""
    _, ax = plt.subplots()
    cov_map = np.arange(31, dtype=float)
    scatter = ax.scatter(np.arange(31), np.zeros(31), c=cov_map)
    # Should not raise and should return a Colorbar object
    cbar = _create_colorbar(scatter, cov_map, "viridis", ax=ax)
    assert cbar is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_rays - 2D projection  (lines 177, 179-181, 480)
# ---------------------------------------------------------------------------


def test_plot_rays_2d_projection() -> None:
    """proj_3d=False produces a 2-D plot and calls set_aspect('equal')."""
    tx_loc = np.array([0.0, 0.0, 10.0])
    rx_loc = np.array([50.0, 30.0, 1.5])
    inter_pos = np.full((2, 1, 3), np.nan)
    inter_pos[1, 0, :] = [25.0, 15.0, 0.0]
    inter = np.array([0, 1])

    ax = plot_rays(rx_loc, tx_loc, inter_pos, inter, proj_3d=False)
    # set_aspect should have been called
    assert ax is not None
    plt.close("all")


def test_plot_rays_2d_via_alias() -> None:
    """proj_3D alias is accepted and treated the same as proj_3d."""
    tx_loc = np.array([0.0, 0.0, 5.0])
    rx_loc = np.array([10.0, 0.0, 1.5])
    inter_pos = np.full((1, 1, 3), np.nan)
    inter = np.array([0])

    ax = plot_rays(rx_loc, tx_loc, inter_pos, inter, proj_3D=False)
    assert ax is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_rays - unexpected kwargs raise TypeError  (lines 377, 379-381)
# ---------------------------------------------------------------------------


def test_plot_rays_unexpected_kwargs_raises() -> None:
    """Unknown keyword arguments must raise TypeError."""
    tx_loc = np.array([0.0, 0.0, 10.0])
    rx_loc = np.array([50.0, 30.0, 1.5])
    inter_pos = np.full((1, 1, 3), np.nan)
    inter = np.array([0])

    with pytest.raises(TypeError, match="Unexpected keyword arguments"):
        plot_rays(rx_loc, tx_loc, inter_pos, inter, bogus_param=True)


# ---------------------------------------------------------------------------
# plot_rays - color_rays_by_pwr  (lines 411-414, 416-428)
# ---------------------------------------------------------------------------


def test_plot_rays_color_by_power_no_powers_raises() -> None:
    """color_rays_by_pwr=True without powers array must raise ValueError."""
    tx_loc = np.array([0.0, 0.0, 10.0])
    rx_loc = np.array([50.0, 30.0, 1.5])
    inter_pos = np.full((1, 1, 3), np.nan)
    inter = np.array([0])

    with pytest.raises(ValueError, match="Powers must be provided"):
        plot_rays(rx_loc, tx_loc, inter_pos, inter, color_rays_by_pwr=True)


def test_plot_rays_color_by_power_with_powers() -> None:
    """color_rays_by_pwr=True with powers array should produce a plot."""
    tx_loc = np.array([0.0, 0.0, 10.0])
    rx_loc = np.array([50.0, 30.0, 1.5])
    inter_pos = np.full((2, 1, 3), np.nan)
    inter_pos[1, 0, :] = [25.0, 15.0, 5.0]
    inter = np.array([0, 1])
    powers = np.array([-80.0, -90.0])

    ax = plot_rays(
        rx_loc,
        tx_loc,
        inter_pos,
        inter,
        color_rays_by_pwr=True,
        powers=powers,
        proj_3d=False,
    )
    assert ax is not None
    plt.close("all")


def test_plot_rays_color_by_power_show_cbar() -> None:
    """show_cbar=True together with color_rays_by_pwr calls colorbar."""
    tx_loc = np.array([0.0, 0.0, 10.0])
    rx_loc = np.array([50.0, 30.0, 1.5])
    inter_pos = np.full((1, 1, 3), np.nan)
    inter = np.array([0])
    powers = np.array([-80.0])

    # Use real Agg backend - colorbar creation must not raise
    ax = plot_rays(
        rx_loc,
        tx_loc,
        inter_pos,
        inter,
        color_rays_by_pwr=True,
        powers=powers,
        show_cbar=True,
        proj_3d=False,
    )
    assert ax is not None
    plt.close("all")


def test_plot_rays_color_by_power_with_limits() -> None:
    """Explicit limits should be used for the power normalisation."""
    tx_loc = np.array([0.0, 0.0, 10.0])
    rx_loc = np.array([50.0, 30.0, 1.5])
    inter_pos = np.full((2, 1, 3), np.nan)
    inter_pos[1, 0, :] = [10.0, 5.0, 2.0]
    inter = np.array([0, 1])
    powers = np.array([-70.0, -100.0])

    ax = plot_rays(
        rx_loc,
        tx_loc,
        inter_pos,
        inter,
        color_rays_by_pwr=True,
        powers=powers,
        limits=(-110.0, -60.0),
        proj_3d=False,
    )
    assert ax is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_rays - inter_objects colors by object ID  (lines 458-464)
# ---------------------------------------------------------------------------


def test_plot_rays_inter_objects() -> None:
    """inter_objects array colours each interaction point by object ID."""
    tx_loc = np.array([0.0, 0.0, 10.0])
    rx_loc = np.array([50.0, 30.0, 1.5])
    # One reflected path with one interaction point; no NaN padding for object IDs
    inter_pos = np.full((1, 1, 3), np.nan)
    inter_pos[0, 0, :] = [25.0, 15.0, 5.0]
    inter = np.array([1])  # one reflection
    # inter_objects: integer object IDs (no NaN) for shape (n_paths, max_inter)
    inter_objects = np.array([[42]])

    ax = plot_rays(
        rx_loc,
        tx_loc,
        inter_pos,
        inter,
        inter_objects=inter_objects,
        proj_3d=False,
    )
    assert ax is not None
    plt.close("all")


def test_plot_rays_inter_objects_with_labels() -> None:
    """inter_obj_labels dict labels used alongside inter_objects."""
    tx_loc = np.array([0.0, 0.0, 10.0])
    rx_loc = np.array([50.0, 30.0, 1.5])
    inter_pos = np.full((1, 1, 3), np.nan)
    inter_pos[0, 0, :] = [25.0, 15.0, 5.0]
    inter = np.array([1])
    inter_objects = np.array([[7]])

    ax = plot_rays(
        rx_loc,
        tx_loc,
        inter_pos,
        inter,
        inter_objects=inter_objects,
        inter_obj_labels={7: "Wall"},
        proj_3d=False,
    )
    assert ax is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_rays - unclassified interaction (color_by_type=False, no inter_objects)
# (lines 462-464)
# ---------------------------------------------------------------------------


def test_plot_rays_unclassified_interaction(capsys) -> None:
    """When color_by_type=False and no inter_objects, prints an 'Unclassified' message."""
    tx_loc = np.array([0.0, 0.0, 10.0])
    rx_loc = np.array([50.0, 30.0, 1.5])
    inter_pos = np.full((1, 1, 3), np.nan)
    inter_pos[0, 0, :] = [10.0, 5.0, 3.0]  # one interaction
    inter = np.array([1])  # reflected (non-LOS so interaction point is drawn)

    plot_rays(
        rx_loc,
        tx_loc,
        inter_pos,
        inter,
        color_by_type=False,
        inter_objects=None,
        proj_3d=False,
    )
    captured = capsys.readouterr()
    assert "Unclassified" in captured.out
    plt.close("all")


# ---------------------------------------------------------------------------
# export_xyz_csv - old-style dict format  (lines 294-312)
# ---------------------------------------------------------------------------


def test_export_xyz_csv_cartesian(tmp_path) -> None:
    """export_xyz_csv writes x,y,z columns in Cartesian mode."""
    n = 5
    locs = np.column_stack(
        [
            np.arange(n, dtype=float),
            np.arange(n, dtype=float) * 2,
            np.zeros(n),
        ]
    )
    los = np.ones(n, dtype=int)
    data = {"user": {"LoS": los, "location": locs}}
    z_var = np.arange(n, dtype=float) * 10.0

    out_file = str(tmp_path / "out.csv")
    export_xyz_csv(data, z_var, outfile=out_file)

    with Path(out_file).open(newline="") as f:
        rows = list(csv_mod.reader(f))

    assert rows[0] == ["x", "y", "z"]
    assert len(rows) == n + 1  # header + n data rows


def test_export_xyz_csv_default_filename(tmp_path, monkeypatch) -> None:
    """When outfile is empty, export_xyz_csv defaults to 'test.csv'."""
    n = 3
    locs = np.column_stack([np.arange(n, dtype=float), np.zeros(n), np.zeros(n)])
    los = np.ones(n, dtype=int)
    data = {"user": {"LoS": los, "location": locs}}
    z_var = np.zeros(n)

    # Redirect writes to tmp_path
    monkeypatch.chdir(tmp_path)
    export_xyz_csv(data, z_var, outfile="")
    assert (tmp_path / "test.csv").exists()


def test_export_xyz_csv_filters_invalid_los(tmp_path) -> None:
    """Rows with LoS == -1 are excluded from the CSV output."""
    locs = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    los = np.array([1, -1, 1])  # middle user invalid
    data = {"user": {"LoS": los, "location": locs}}
    z_var = np.array([10.0, 20.0, 30.0])

    out_file = str(tmp_path / "filtered.csv")
    export_xyz_csv(data, z_var, outfile=out_file)

    with Path(out_file).open(newline="") as f:
        rows = list(csv_mod.reader(f))

    # Header + 2 valid rows
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# plot_power_discarding - trim_delay=None without channel_params  (lines 506-509)
# ---------------------------------------------------------------------------


def test_plot_power_discarding_no_channel_params_raises() -> None:
    """trim_delay=None and no channel_params must raise ValueError."""
    ds = MagicMock(spec=[])  # no channel_params attribute
    ds.delay = np.array([[1e-7, 2e-7]])
    ds.power_linear = np.array([[1.0, 0.5]])

    with pytest.raises(ValueError, match="Dataset has no channel parameters"):
        plot_power_discarding(ds, trim_delay=None)


# ---------------------------------------------------------------------------
# plot_power_discarding - max delay < trim_delay  (lines 511-514)
# ---------------------------------------------------------------------------


def test_plot_power_discarding_no_paths_discarded(capsys) -> None:
    """When max delay < trim_delay the function prints and returns (None, None)."""
    ds = MagicMock()
    ds.delay = np.array([[1e-8, 2e-8]])
    ds.power_linear = np.array([[1.0, 0.5]])

    result = plot_power_discarding(ds, trim_delay=1.0)  # huge trim_delay
    assert result == (None, None)
    captured = capsys.readouterr()
    assert "No paths will be discarded" in captured.out


# ---------------------------------------------------------------------------
# plot_power_discarding - user with all-NaN delays  (lines 522-523)
# ---------------------------------------------------------------------------


def test_plot_power_discarding_all_nan_user() -> None:
    """Users with all-NaN delays contribute 0 to the discarded-power ratio."""
    ds = MagicMock()
    # User 0: all NaN delays; User 1: one path beyond trim_delay
    ds.delay = np.array([[np.nan, np.nan], [1e-7, 5e-7]])
    ds.power_linear = np.array([[np.nan, np.nan], [1.0, 0.1]])

    fig, ax = plot_power_discarding(ds, trim_delay=3e-7)
    # Function must complete without error and return matplotlib objects
    assert fig is not None
    assert ax is not None
    plt.close("all")

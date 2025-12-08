"""Tests for DeepMIMO visualization."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt
from deepmimo.generator import visualization


class TestVisualization(unittest.TestCase):
    def test_plot_coverage_realistic(self):
        """Test plot_coverage with real data (Agg backend)."""
        ue_pos = np.random.rand(100, 3) * 100  # 100 UEs in 100x100 grid
        los = np.random.choice([0, 1], size=(100,))  # Random LoS/NLoS

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

    @patch("deepmimo.generator.visualization.plt")
    def test_plot_coverage(self, mock_plt):
        ue_pos = np.random.rand(10, 3) * 100
        los = np.random.choice([0, 1], size=(10,))

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

    def test_plot_rays_realistic(self):
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

    @patch("deepmimo.generator.visualization.plt")
    def test_plot_rays(self, mock_plt):
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

    @patch("deepmimo.generator.visualization.plt")
    def test_plot_power_discarding(self, mock_plt):
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

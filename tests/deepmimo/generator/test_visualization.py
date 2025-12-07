"""Tests for DeepMIMO Visualization Module."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from deepmimo.generator import visualization as viz

def test_transform_coordinates():
    coords = np.array([[0, 0], [100, 100]])
    # lat_min, lat_max, lon_min, lon_max
    # function signature: coords, lon_max, lon_min, lat_min, lat_max
    lats, lons = viz.transform_coordinates(coords, -100, -101, 30, 31)
    # x=0 (min) -> lon_min (-101). x=100 (max) -> lon_max (-100).
    # y=0 (min) -> lat_min (30). y=100 (max) -> lat_max (31).
    assert np.isclose(lons[0], -101)
    assert np.isclose(lons[1], -100)
    assert np.isclose(lats[0], 30)
    assert np.isclose(lats[1], 31)

@patch("matplotlib.pyplot.subplots")
def test_plot_coverage(mock_subplots):
    fig_mock = MagicMock()
    ax_mock = MagicMock()
    mock_subplots.return_value = (fig_mock, ax_mock)
    
    rxs = np.random.rand(10, 3)
    cov_map = np.random.rand(10)
    
    viz.plot_coverage(rxs, cov_map)
    
    ax_mock.scatter.assert_called()
    
@patch("matplotlib.pyplot.subplots")
def test_plot_rays(mock_subplots):
    fig_mock = MagicMock()
    ax_mock = MagicMock()
    mock_subplots.return_value = (fig_mock, ax_mock)
    
    rx_loc = np.zeros(3)
    tx_loc = np.array([10, 10, 10])
    inter_pos = np.random.rand(2, 2, 3) # 2 paths, 2 interactions
    inter = np.array([11, 22]) # codes
    
    # Mock get_legend_handles_labels to return dummy values so it doesn't fail
    ax_mock.get_legend_handles_labels.return_value = (['h1'], ['l1'])
    
    viz.plot_rays(rx_loc, tx_loc, inter_pos, inter, ax=ax_mock)
    
    # We expect multiple plot/scatter calls
    assert ax_mock.plot.called or ax_mock.scatter.called

def test_export_xyz_csv(tmp_path):
    data = {
        "user": {
            "LoS": np.array([1, 0, -1]),
            "location": np.array([[0,0,0], [1,1,1], [2,2,2]])
        }
    }
    z_var = np.array([10, 20, 30])
    outfile = tmp_path / "test.csv"
    
    viz.export_xyz_csv(data, z_var, str(outfile))
    
    assert outfile.exists()
    # Check content
    with open(outfile) as f:
        lines = f.readlines()
        # Header + 2 valid users (LoS != -1) = 3 lines
        assert len(lines) == 3 


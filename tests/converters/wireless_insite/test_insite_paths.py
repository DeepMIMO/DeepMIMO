"""Tests for Wireless Insite Paths."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from deepmimo.converters.wireless_insite import insite_paths


def test_update_txrx_points():
    txrx_dict = {"txrx_set_1": {"id_orig": 10, "num_points": 0, "num_active_points": 0}}
    rx_pos = np.zeros((5, 3))
    path_loss = np.array([100, 250, 100, 250, 100])  # 2 inactive (250 dB)

    insite_paths.update_txrx_points(txrx_dict, 1, rx_pos, path_loss)

    assert txrx_dict["txrx_set_1"]["num_points"] == 5
    assert txrx_dict["txrx_set_1"]["num_active_points"] == 3


@patch("deepmimo.converters.wireless_insite.insite_paths.paths_parser")
@patch("deepmimo.converters.wireless_insite.insite_paths.read_pl_p2m_file")
@patch("deepmimo.converters.wireless_insite.insite_paths.save_mat")
@patch("deepmimo.converters.wireless_insite.insite_paths.extract_tx_pos")
def test_read_paths(mock_extract, mock_save, mock_read_pl, mock_parser, tmp_path):
    # Setup mocks
    p2m_dir = tmp_path / "p2m"
    p2m_dir.mkdir()
    (p2m_dir / "proj.paths.t001_01.r002.p2m").touch()

    txrx_dict = {
        "txrx_set_0": {"id": 0, "id_orig": 1, "is_tx": True, "num_points": 1, "is_rx": False},
        "txrx_set_1": {"id": 1, "id_orig": 2, "is_tx": False, "num_points": 1, "is_rx": True},
    }

    mock_parser.return_value = {"power": np.array([1.0])}
    mock_read_pl.return_value = (np.zeros((1, 3)), None, np.array([100.0]))
    mock_extract.return_value = np.array([0, 0, 0])

    insite_paths.read_paths(str(tmp_path), "out_dir", txrx_dict)

    # Check save_mat called
    assert mock_save.called
    # Check parser called
    mock_parser.assert_called()

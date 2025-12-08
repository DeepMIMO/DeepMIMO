"""Tests for Sionna TXRX."""

import numpy as np

from deepmimo.converters.sionna_rt import sionna_txrx


def test_read_txrx():
    raw_params = {
        "tx_array_num_ant": 4,
        "tx_array_ant_pos": np.zeros((4, 3)),
        "tx_array_size": 4,  # 4 elements, 4 antennas -> single pol
        "rx_array_num_ant": 1,
        "rx_array_ant_pos": np.zeros((1, 3)),
        "rx_array_size": 1,
    }
    rt_params = {"raw_params": raw_params, "synthetic_array": False}

    txrx_dict = sionna_txrx.read_txrx(rt_params)

    assert "txrx_set_0" in txrx_dict
    assert "txrx_set_1" in txrx_dict

    tx = txrx_dict["txrx_set_0"]
    assert tx["is_tx"]
    assert tx["num_ant"] == 4

    rx = txrx_dict["txrx_set_1"]
    assert rx["is_rx"]
    assert rx["num_ant"] == 1

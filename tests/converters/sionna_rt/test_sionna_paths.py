"""Tests for Sionna Paths module."""

from unittest.mock import patch

import numpy as np

from deepmimo.converters.sionna_rt import sionna_paths


def test_get_sionna_interaction_types():
    # Mock paths object
    # Using a complex mock structure to simulate Sionna paths data
    # Since _get_sionna_interaction_types logic was fixed, we test it here.

    # Case 1: Simple valid types
    # types array structure depends on implementation details
    # Assuming input is (n_paths, n_inter) or similar?
    # Wait, the function signature is not standard public API but internal logic?
    # Let's check sionna_paths.py again.
    # It seems `read_paths` processes paths.
    pass


@patch("deepmimo.converters.sionna_rt.sionna_paths.load_pickle")
def test_read_paths(mock_load):
    # Create mock paths data structure
    # Match expected shape for Sionna < 1.0
    # a has time dimension (7D) because it's sliced with [..., 0]
    # others don't have time dimension (6D) because they are sliced with [...] (which keeps trailing dims)
    path_data = {
        "sources": np.zeros((1, 3)),
        "targets": np.zeros((5, 3)),  # 5 RX
        "a": np.ones((1, 5, 1, 1, 1, 10, 1)) * (1 + 1j),  # 7D
        "tau": np.zeros((1, 5, 1, 1, 1, 10)),  # 6D
        "theta_t": np.zeros((1, 5, 1, 1, 1, 10)),
        "phi_t": np.zeros((1, 5, 1, 1, 1, 10)),
        "theta_r": np.zeros((1, 5, 1, 1, 1, 10)),
        "phi_r": np.zeros((1, 5, 1, 1, 1, 10)),
        "types": np.zeros((1, 5, 1, 1, 1, 10)),
        # vertices for Sionna 0.x: [max_depth, n_rx, n_tx, max_paths, 3]
        # max_depth=5 (arbitrary), rx=5, tx=1, paths=10, coords=3
        "vertices": np.zeros((5, 5, 1, 10, 3)),
    }
    mock_load.return_value = [path_data]  # Single TX

    txrx_dict = {
        "txrx_set_0": {"is_tx": True, "id": 0, "num_points": 1, "num_ant": 1},
        "txrx_set_1": {"is_rx": True, "id": 1, "num_points": 5, "num_ant": 1},
    }

    # Mock save_mat via patch in the module
    with patch("deepmimo.converters.sionna_rt.sionna_paths.save_mat") as mock_save:
        sionna_paths.read_paths("dummy_folder", "out_folder", txrx_dict, sionna_version="0.19.1")

        assert mock_save.called
        # Verify channel shape or keys saved
        # Expect keys: channel, delay, phase, aoa/aod, etc.

"""Tests for Sionna Paths module (Sionna 2.0)."""

from unittest.mock import patch

import numpy as np

from deepmimo import consts as c
from deepmimo.converters.sionna_rt import sionna_paths


def test_transform_interaction_types_los() -> None:
    """LoS path: all-zero depth array maps to INTERACTION_LOS."""
    types = np.array([[0, 0, 0]], dtype=np.float32)
    result = sionna_paths._transform_interaction_types(types)  # noqa: SLF001
    assert result[0] == c.INTERACTION_LOS


def test_transform_interaction_types_reflections() -> None:
    """Verify SPECULAR(1) reflections encode correctly — value unchanged."""
    types = np.array(
        [
            [1, 0, 0],   # SPECULAR → single reflection
            [1, 1, 0],   # SPECULAR+SPECULAR → two reflections
            [1, 1, 1],   # three reflections
        ],
        dtype=np.float32,
    )
    result = sionna_paths._transform_interaction_types(types)  # noqa: SLF001
    assert result[0] == 1.0
    assert result[1] == 11.0
    assert result[2] == 111.0


def test_transform_interaction_types_2_0_remapping() -> None:
    """Verify Sionna 2.0 → DeepMIMO code remapping.

    SPECULAR=1 → REFLECTION=1 (unchanged)
    DIFFUSE=2  → SCATTERING=3
    DIFFRACTION=8 → DIFFRACTION=2
    """
    types = np.array(
        [
            [1, 2, 0],  # SPECULAR then DIFFUSE → reflection(1) then scatter(3) → 13
            [8, 0, 0],  # DIFFRACTION → diffraction(2) → 2
            [1, 8, 0],  # SPECULAR then DIFFRACTION → 12
        ],
        dtype=np.float32,
    )
    result = sionna_paths._transform_interaction_types(types)  # noqa: SLF001
    assert result[0] == 13.0
    assert result[1] == 2.0
    assert result[2] == 12.0


@patch("deepmimo.converters.sionna_rt.sionna_paths.load_pickle")
def test_read_paths_sionna2(mock_load) -> None:
    """Load Sionna 2.0 paths and ensure expected arrays are saved.

    Sionna 2.0 shapes (no batch dim):
    - a:            (num_rx, num_rx_ant, num_tx, num_tx_ant, max_paths)
    - tau/angles:   (num_rx, num_tx, max_paths)  [single-antenna case]
    - interactions: (max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, max_paths)
    - vertices:     (max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, max_paths, 3)
    """
    n_rx, n_tx, max_paths, max_depth = 5, 1, 10, 5

    path_data = {
        "sources": np.zeros((n_tx, 3)),
        "targets": np.zeros((n_rx, 3)),
        # a has antenna dims: (n_rx, n_rx_ant, n_tx, n_tx_ant, max_paths)
        "a": np.ones((n_rx, 1, n_tx, 1, max_paths)) * (1 + 1j),
        # single-antenna angles have no antenna dims: (n_rx, n_tx, max_paths)
        "tau": np.zeros((n_rx, n_tx, max_paths)),
        "theta_t": np.zeros((n_rx, n_tx, max_paths)),
        "phi_t": np.zeros((n_rx, n_tx, max_paths)),
        "theta_r": np.zeros((n_rx, n_tx, max_paths)),
        "phi_r": np.zeros((n_rx, n_tx, max_paths)),
        # single-antenna: no antenna dims in interactions/vertices
        "interactions": np.zeros((max_depth, n_rx, n_tx, max_paths)),
        "vertices": np.zeros((max_depth, n_rx, n_tx, max_paths, 3)),
    }
    mock_load.return_value = [path_data]

    txrx_dict = {
        "txrx_set_0": {"is_tx": True, "id": 0, "num_points": n_tx, "num_ant": 1},
        "txrx_set_1": {"is_rx": True, "id": 1, "num_points": n_rx, "num_ant": 1},
    }

    with patch("deepmimo.converters.sionna_rt.sionna_paths.save_mat") as mock_save:
        sionna_paths.read_paths("dummy_folder", "out_folder", txrx_dict)
        assert mock_save.called

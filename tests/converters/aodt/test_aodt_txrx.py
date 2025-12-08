"""Tests for AODT TXRX."""

from unittest.mock import MagicMock, patch

from deepmimo.converters.aodt import aodt_txrx


@patch("deepmimo.converters.aodt.aodt_txrx.pd")
def test_read_txrx(mock_pd) -> None:
    # Mock scenario dataframe for tx/rx info
    df = MagicMock()
    # Iterate over rows
    # Row 1: TX
    row1 = {
        "device_type": "transmitter",
        "device_id": 0,
        "x": 0,
        "y": 0,
        "z": 10,
        "rot_x": 0,
        "rot_y": 0,
        "rot_z": 0,
    }
    # Row 2: RX
    row2 = {
        "device_type": "receiver",
        "device_id": 1,
        "x": 10,
        "y": 10,
        "z": 1.5,
        "rot_x": 0,
        "rot_y": 0,
        "rot_z": 0,
    }

    df.iterrows.return_value = [(0, row1), (1, row2)]
    df.__len__.return_value = 2  # Fix empty check

    # Mock return for patterns.parquet check
    mock_patterns = MagicMock()
    mock_patterns.__len__.return_value = 1
    mock_patterns.__getitem__.return_value = mock_patterns  # Support filtering
    # Support accessing type: pattern.iloc[0]["type"]
    mock_patterns.iloc.__getitem__.return_value.__getitem__.return_value = 0

    # Panel mock with necessary columns
    mock_panels = MagicMock()
    mock_panels.__len__.return_value = 1
    panel_row = {
        "panel_id": 0,
        "panel_name": "panel1",
        "antenna_names": ["a1"],
        "antenna_pattern_indices": [0],
        "frequencies": [28e9],
        "thetas": [0],
        "phis": [0],
        "reference_freq": 28e9,
        "dual_polarized": False,
        "num_loc_antenna_horz": 1,
        "num_loc_antenna_vert": 1,
        "antenna_spacing_horz": 0.5,
        "antenna_spacing_vert": 0.5,
        "antenna_roll_angle_first_polz": 0,
        "antenna_roll_angle_second_polz": 0,
    }
    mock_panels.iterrows.return_value = [(0, panel_row)]

    # Mock RUs (transmitters)
    mock_rus = MagicMock()
    mock_rus.__len__.return_value = 1
    ru_row = {
        "ID": 0,
        "position": [0, 0, 10],
        "radiated_power": 30.0,
        "mech_tilt": 0.0,
        "mech_azimuth": 0.0,
        "subcarrier_spacing": 15,
        "fft_size": 1024,
        "panel": [0],  # references panel_id 0
        "du_id": None,
        "du_manual_assign": False,
    }
    mock_rus.iterrows.return_value = [(0, ru_row)]

    # Mock UEs (receivers)
    mock_ues = MagicMock()
    mock_ues.__len__.return_value = 1
    ue_row = {
        "ID": 1,
        "is_manual": False,
        "is_manual_mobility": False,
        "radiated_power": 10.0,
        "height": 1.5,
        "mech_tilt": 0.0,
        "panel": [0],
        "is_indoor_mobility": False,
        "bler_target": 0.1,
        "batch_indices": [0],
        "waypoint_ids": [0],
        "waypoint_points": [[0, 0, 0]],
        "waypoint_stops": [0],
        "waypoint_speeds": [0],
        "trajectory_ids": [0],
        "trajectory_points": [[0, 0, 0]],
        "trajectory_stops": [0],
        "trajectory_speeds": [0],
        "route_positions": [[[10, 10, 1.5]]],  # list of lists of points?
        "route_orientations": [[[0, 0, 0]]],
        "route_speeds": [0],
        "route_times": [0],
    }
    mock_ues.iterrows.return_value = [(0, ue_row)]

    def side_effect(path):
        if "patterns" in str(path):
            return mock_patterns
        if "panels" in str(path):
            return mock_panels
        if "rus" in str(path):
            return mock_rus
        if "ues" in str(path):
            return mock_ues
        return df

    mock_pd.read_parquet.side_effect = side_effect

    # Mock rt_params
    rt_params = {
        "gps_bbox": (0, 0, 0, 0),
        "raw_params": {"duration": 1, "interval": 1},  # Avoid ZeroDivisionError
    }

    with patch("os.path.exists", return_value=True):
        txrx = aodt_txrx.read_txrx("dummy_folder", rt_params)
    assert len(txrx) == 2
    assert txrx["txrx_set_0"]["is_tx"]
    assert txrx["txrx_set_1"]["is_rx"]

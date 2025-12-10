"""Tests for AODT Paths."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from deepmimo import consts as c
from deepmimo.converters.aodt import aodt_paths


def test_transform_interaction_types() -> None:
    """Transform AODT interaction labels to DeepMIMO codes."""
    # LoS: only emission and reception
    types_los = np.array(["emission", "reception"], dtype=object)
    assert aodt_paths.transform_interaction_types(types_los) == c.INTERACTION_LOS

    # Reflection
    types_ref = np.array(["emission", "reflection", "reception"], dtype=object)
    assert aodt_paths.transform_interaction_types(types_ref) == 1.0

    # Ref + Diff
    types_mixed = np.array(["emission", "reflection", "diffraction", "reception"], dtype=object)
    assert aodt_paths.transform_interaction_types(types_mixed) == 12.0


@patch("deepmimo.converters.aodt.aodt_paths.pd")
@patch.object(Path, "exists")
@patch("pathlib.Path.mkdir")
@patch("deepmimo.converters.aodt.aodt_paths.gu.save_mat")
def test_read_paths(mock_save_mat, mock_mkdir, mock_exists, mock_pd) -> None:
    """Test read_paths with mocked parquet data."""
    mock_exists.return_value = True

    # Mock dataframes
    # Paths DF: ue_id, ru_id, time_idx, points, interaction_types
    # CIRs DF: ue_id, ru_id, time_idx, ue_ant_el, cir_re, cir_im, cir_delay

    # Mock row class for paths
    class MockPathRow:
        def __init__(self) -> None:
            self.points = np.array([[0, 0, 0], [100, 100, 100], [200, 200, 200]])  # cm
            self.interaction_types = np.array(["emission", "reflection", "reception"], dtype=object)

    mock_paths_df = MagicMock()
    mock_paths_df.__len__.return_value = 1
    mock_paths_df.__getitem__.return_value = mock_paths_df  # for column access returning series
    mock_paths_df.unique.side_effect = lambda: np.array([0])  # for ue_id, time_idx, ru_id

    # Mock filtering
    # paths_df[col == val] -> returns self (mock)
    mock_paths_df.__getitem__.return_value = mock_paths_df

    # Setup itertuples to return our mock row
    mock_paths_df.itertuples.return_value = [MockPathRow()]

    mock_cirs_df = MagicMock()
    mock_cirs_df.__len__.return_value = 1
    mock_cirs_df.__getitem__.return_value = mock_cirs_df
    mock_cirs_df.unique.return_value = np.array([0])

    # CIR data
    mock_cirs_df.iloc.__getitem__.return_value = mock_cirs_df  # slice returns mock
    mock_cirs_df.__getitem__.side_effect = None  # Reset side effect

    # Mock column access on cirs_df slice
    def cir_col_access(key):
        m = MagicMock()
        # Only check string keys to avoid truth value ambiguity with arrays
        if isinstance(key, str) and key in ["cir_re", "cir_im", "cir_delay"]:
            m.to_numpy.return_value = np.array([[1.0]])  # wrap in array for [0] indexing
        elif not isinstance(key, str):
            # For boolean indexing (e.g., df[df["col"] == val]), return mock_cirs_df
            return mock_cirs_df
        return m

    mock_cirs_df.__getitem__.side_effect = cir_col_access

    # Setup read_parquet to return these DFs
    def read_parquet_side_effect(path):
        if "raypaths" in path:
            return mock_paths_df
        if "cirs" in path:
            return mock_cirs_df
        return MagicMock()

    mock_pd.read_parquet.side_effect = read_parquet_side_effect

    txrx_dict = {
        "bs1": {
            "is_tx": True,
            "is_rx": False,  # Added missing key
            "id_orig": 0,
            "id": 1,
            "num_ant": 1,
        },
        "ue1": {"is_rx": True, "is_tx": False, "id_orig": 1, "id": 2, "num_ant": 1},
    }

    aodt_paths.read_paths("rt_folder", "out_folder", txrx_dict)

    # Verify save_mat called
    assert mock_save_mat.called
    mock_mkdir.assert_called()

"""Tests for AODT Scene."""

import pytest
from unittest.mock import patch, MagicMock
from deepmimo.converters.aodt import aodt_scene

@patch("deepmimo.converters.aodt.aodt_scene.pd")
def test_read_scene(mock_pd):
    # Mock panels dataframe
    df = MagicMock()
    df.__len__.return_value = 1 # Fix empty check
    # Columns matches AODTScene expectation
    row = {
        "prim_path": "path/to/prim",
        "material": 1,
        "is_rf_active": True,
        "is_rf_diffuse": False,
        "is_rf_diffraction": False,
        "is_rf_transmission": False
    }
    df.iterrows.return_value = [(0, row)]
    mock_pd.read_parquet.return_value = df
    
    with patch("os.path.exists", return_value=True):
        scene = aodt_scene.read_scene("dummy_folder")
        assert len(scene.primitives) == 1
        # Check if object parsed correctly
        assert scene.materials[0] == 1


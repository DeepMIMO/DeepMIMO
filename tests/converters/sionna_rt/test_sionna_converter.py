"""Test module for Sionna Ray Tracing converter functionality.

This module contains tests for the Sionna RT converter, particularly focusing on
the interaction type conversion between Sionna and DeepMIMO formats.
"""

# %%
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deepmimo import consts as c
from deepmimo.converters.sionna_rt.sionna_converter import sionna_rt_converter
from deepmimo.converters.sionna_rt.sionna_paths import (
    _get_sionna_interaction_types as get_sionna_interaction_types,
)


def test_get_sionna_interaction_types() -> None:
    """Test conversion of Sionna interaction types to DeepMIMO codes."""
    print("\n=== Testing Sionna to DeepMIMO Interaction Type Mapping ===")

    # Test data dimensions
    n_users = 3
    max_paths = 4
    max_interactions = 5

    # Create test types array (N_USERS x MAX_PATHS)
    types = np.array(
        [
            # User 1: LoS, 1 reflection, 2 reflections, diffraction
            [0, 1, 1, 2],
            # User 2: scattering, 1 refl + scattering, RIS (4), NaN
            [3, 3, 4, np.nan],
            # User 3: 3 reflections, LoS, 2 refl + scattering, NaN
            [1, 0, 3, np.nan],
        ],
        dtype=np.float32,
    )

    # Create a copy without RIS for main test
    types_no_ris = types.copy()
    types_no_ris[1, 2] = np.nan  # Replace RIS with NaN

    # Create test interaction positions (N_USERS x MAX_PATHS x MAX_INTERACTIONS x 3)
    inter_pos = np.zeros((n_users, max_paths, max_interactions, 3)) * np.nan

    # User 1
    inter_pos[0, 1, 0] = [1, 1, 1]
    inter_pos[0, 2, 0:2] = [[1, 1, 1], [2, 2, 2]]
    inter_pos[0, 3, 0] = [3, 3, 3]

    # User 2
    inter_pos[1, 0, 0] = [4, 4, 4]
    inter_pos[1, 1, 0:2] = [[5, 5, 5], [6, 6, 6]]
    inter_pos[1, 2, 0] = [7, 7, 7]

    # User 3
    inter_pos[2, 0, 0:3] = [[8, 8, 8], [9, 9, 9], [10, 10, 10]]
    inter_pos[2, 2, 0:3] = [[11, 11, 11], [12, 12, 12], [13, 13, 13]]

    # Expected output
    expected = np.zeros((n_users, max_paths), dtype=np.float32)
    # User 1
    expected[0, 0] = c.INTERACTION_LOS  # LoS
    expected[0, 1] = 1  # Single reflection
    expected[0, 2] = 11  # Two reflections
    expected[0, 3] = c.INTERACTION_DIFFRACTION  # Single diffraction

    # User 2
    expected[1, 0] = 3  # Single scattering
    expected[1, 1] = 13  # One reflection + scattering
    expected[1, 2] = 0  # RIS (not supported)
    expected[1, 3] = 0  # NaN path

    # User 3
    expected[2, 0] = 111  # Three reflections
    expected[2, 1] = c.INTERACTION_LOS  # LoS
    expected[2, 2] = 113  # Two reflections + scattering
    expected[2, 3] = 0  # NaN path

    # Test main functionality (without RIS)
    result = get_sionna_interaction_types(types_no_ris, inter_pos)

    np.testing.assert_array_almost_equal(result, expected)

    # Test RIS error separately
    with pytest.raises(NotImplementedError):
        get_sionna_interaction_types(types, inter_pos)


def test_edge_cases() -> None:
    """Test edge cases for interaction type conversion."""
    # Test empty arrays
    types = np.zeros((0, 5), dtype=np.float32)
    inter_pos = np.zeros((0, 5, 3, 3), dtype=np.float32)
    result = get_sionna_interaction_types(types, inter_pos)
    assert result.shape == (0, 5), "Empty array test failed"

    # Test all NaN
    types = np.full((2, 3), np.nan, dtype=np.float32)
    inter_pos = np.full((2, 3, 4, 3), np.nan, dtype=np.float32)
    result = get_sionna_interaction_types(types, inter_pos)
    assert np.all(result == 0), "All NaN test failed"

    # Test all zeros
    types = np.zeros((2, 3), dtype=np.float32)
    inter_pos = np.zeros((2, 3, 4, 3), dtype=np.float32)
    result = get_sionna_interaction_types(types, inter_pos)
    assert np.all(result == 0), "All zeros test failed"


@patch("deepmimo.converters.sionna_rt.sionna_converter.read_rt_params")
@patch("deepmimo.converters.sionna_rt.sionna_converter.read_txrx")
@patch("deepmimo.converters.sionna_rt.sionna_converter.read_paths")
@patch("deepmimo.converters.sionna_rt.sionna_converter.read_materials")
@patch("deepmimo.converters.sionna_rt.sionna_converter.read_scene")
@patch("deepmimo.converters.sionna_rt.sionna_converter.cu")
@patch("shutil.rmtree")
@patch("pathlib.Path.mkdir")
def test_sionna_rt_converter_flow(
    mock_mkdir,
    mock_rmtree,
    mock_cu,
    mock_read_scene,
    mock_read_materials,
    mock_read_paths,
    mock_read_txrx,
    mock_read_rt_params,
) -> None:
    """Test the full flow of sionna_rt_converter orchestrator."""
    # Setup mocks
    mock_cu.check_scenario_exists.return_value = True
    mock_read_rt_params.return_value = {"raytracer_version": "0.19.0"}
    mock_read_txrx.return_value = {}
    mock_read_materials.return_value = ({}, {})

    mock_scene_obj = MagicMock()
    mock_scene_obj.export_data.return_value = {}
    mock_read_scene.return_value = mock_scene_obj

    # Run converter
    rt_folder = "/path/to/rt_folder"
    result = sionna_rt_converter(rt_folder, scenario_name="test_scen")

    # Verify calls
    assert result == "test_scen"
    mock_read_rt_params.assert_called_once_with(rt_folder)
    mock_read_txrx.assert_called_once()
    mock_read_paths.assert_called_once()
    mock_read_materials.assert_called_once()
    mock_read_scene.assert_called_once()
    mock_cu.save_params.assert_called_once()
    mock_cu.save_scenario.assert_called_once()

    # Test failure if scenario check fails
    mock_cu.check_scenario_exists.return_value = False
    result_fail = sionna_rt_converter(rt_folder)
    assert result_fail is None

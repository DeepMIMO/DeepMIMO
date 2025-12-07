"""Tests for DeepMIMO Summary module."""

import pytest
from unittest.mock import patch, MagicMock
from deepmimo.summary import summary
from deepmimo import consts as c

@patch("deepmimo.summary.load_dict_from_json")
@patch("deepmimo.summary.get_params_path")
def test_summary_generation(mock_get_path, mock_load_json):
    """Test summary string generation."""
    # Mock params data
    mock_params = {
        c.RT_PARAMS_PARAM_NAME: {
            c.RT_PARAM_RAYTRACER: "TestRT",
            c.RT_PARAM_RAYTRACER_VERSION: "1.0",
            c.RT_PARAM_FREQUENCY: 28e9,
            c.RT_PARAM_PATH_DEPTH: 5,
            c.RT_PARAM_MAX_REFLECTIONS: 3,
            c.RT_PARAM_MAX_DIFFRACTIONS: 1,
            c.RT_PARAM_MAX_SCATTERING: 0,
            c.RT_PARAM_MAX_TRANSMISSIONS: 0,
            c.RT_PARAM_NUM_RAYS: 1000,
            c.RT_PARAM_RAY_CASTING_METHOD: "uniform",
            c.RT_PARAM_RAY_CASTING_RANGE_AZ: 360,
            c.RT_PARAM_RAY_CASTING_RANGE_EL: 180,
            c.RT_PARAM_SYNTHETIC_ARRAY: True,
            # Add missing keys to prevent KeyError
            c.RT_PARAM_DIFFUSE_REFLECTIONS: 0,
            c.RT_PARAM_DIFFUSE_DIFFRACTIONS: 0,
            c.RT_PARAM_DIFFUSE_TRANSMISSIONS: 0,
            c.RT_PARAM_DIFFUSE_FINAL_ONLY: False,
            c.RT_PARAM_DIFFUSE_RANDOM_PHASES: False,
            c.RT_PARAM_TERRAIN_REFLECTION: False,
            c.RT_PARAM_TERRAIN_DIFFRACTION: False,
            c.RT_PARAM_TERRAIN_SCATTERING: False,
        },
        c.SCENE_PARAM_NAME: {
            c.SCENE_PARAM_NUMBER_SCENES: 1,
            c.SCENE_PARAM_N_OBJECTS: 10,
            c.SCENE_PARAM_N_VERTICES: 100,
            c.SCENE_PARAM_N_FACES: 50,
            c.SCENE_PARAM_N_TRIANGULAR_FACES: 100,
        },
        c.MATERIALS_PARAM_NAME: {
            "mat1": {
                c.MATERIALS_PARAM_NAME_FIELD: "Concrete",
                c.MATERIALS_PARAM_PERMITTIVITY: 5.0,
                c.MATERIALS_PARAM_CONDUCTIVITY: 0.1,
                c.MATERIALS_PARAM_SCATTERING_MODEL: "none",
                c.MATERIALS_PARAM_SCATTERING_COEF: 0.0,
                c.MATERIALS_PARAM_CROSS_POL_COEF: 0.0,
            }
        },
        c.TXRX_PARAM_NAME: {
            "set1": {
                c.TXRX_PARAM_NAME_FIELD: "BS",
                c.TXRX_PARAM_IS_TX: True,
                c.TXRX_PARAM_IS_RX: False,
                c.TXRX_PARAM_NUM_POINTS: 1,
                c.TXRX_PARAM_NUM_ACTIVE_POINTS: 1,
                c.TXRX_PARAM_NUM_ANT: 64,
                c.TXRX_PARAM_DUAL_POL: False,
            },
            "set2": {
                c.TXRX_PARAM_NAME_FIELD: "UE",
                c.TXRX_PARAM_IS_TX: False,
                c.TXRX_PARAM_IS_RX: True,
                c.TXRX_PARAM_NUM_POINTS: 10,
                c.TXRX_PARAM_NUM_ACTIVE_POINTS: 10,
                c.TXRX_PARAM_NUM_ANT: 1,
                c.TXRX_PARAM_DUAL_POL: False,
            }
        }
    }
    
    mock_load_json.return_value = mock_params
    mock_get_path.return_value = "dummy/path/params.json"
    
    # Test string generation (print_summary=False)
    s = summary("test_scenario", print_summary=False)
    
    assert "DeepMIMO test_scenario Scenario Summary" in s
    assert "Ray-tracer: TestRT v1.0" in s
    assert "Frequency: 28.0 GHz" in s
    assert "Total number of receivers: 10" in s
    assert "Total number of transmitters: 1" in s


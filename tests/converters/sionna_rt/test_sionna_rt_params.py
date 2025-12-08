"""Tests for Sionna RT Params."""

from unittest.mock import patch
from deepmimo.converters.sionna_rt import sionna_rt_params
from deepmimo import consts as c


@patch("deepmimo.converters.sionna_rt.sionna_rt_params.load_pickle")
def test_read_rt_params(mock_load):
    raw_params = {
        "los": True,
        "tx_array_size": 1,
        "tx_array_num_ant": 1,
        "num_samples": 100,
        "frequency": 3.5e9,
        "max_depth": 5,
        "reflection": True,
        "diffraction": True,
        "scattering": True,
        "scat_random_phases": True,
        "synthetic_array": True,
        "method": "fibonacci",
        "min_lat": 0,
        "min_lon": 0,
        "max_lat": 0,
        "max_lon": 0,
    }
    mock_load.return_value = raw_params

    params = sionna_rt_params.read_rt_params("/dummy/path")
    assert params["raytracer_name"] == c.RAYTRACER_NAME_SIONNA
    assert params["max_path_depth"] == 5
    assert params["max_reflections"] == 5  # if reflection=True, max_reflections=max_depth

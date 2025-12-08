"""Tests for DeepMIMO RT Parameters module."""


import pytest

from deepmimo.rt_params import RayTracingParameters


def test_rt_params_initialization():
    """Test RayTracingParameters initialization with required fields."""
    params = RayTracingParameters(
        raytracer_name="TestRT",
        raytracer_version="1.0",
        frequency=28e9,
        max_path_depth=5,
        max_reflections=3,
        max_diffractions=1,
        max_scattering=0,
        max_transmissions=0,
    )

    assert params.raytracer_name == "TestRT"
    assert params.frequency == 28e9
    assert params.num_rays == 1000000  # Default value
    assert params.synthetic_array is True  # Default value


def test_rt_params_to_dict():
    """Test converting params to dictionary."""
    params = RayTracingParameters(
        raytracer_name="TestRT",
        raytracer_version="1.0",
        frequency=28e9,
        max_path_depth=5,
        max_reflections=3,
        max_diffractions=1,
        max_scattering=0,
        max_transmissions=0,
    )

    d = params.to_dict()
    assert isinstance(d, dict)
    assert d["raytracer_name"] == "TestRT"
    assert d["max_reflections"] == 3
    assert d["num_rays"] == 1000000


def test_rt_params_from_dict():
    """Test creating params from dictionary."""
    data = {
        "raytracer_name": "TestRT",
        "raytracer_version": "1.0",
        "frequency": 28e9,
        "max_path_depth": 5,
        "max_reflections": 3,
        "max_diffractions": 1,
        "max_scattering": 0,
        "max_transmissions": 0,
        "num_rays": 500,
    }

    params = RayTracingParameters.from_dict(data)
    assert params.raytracer_name == "TestRT"
    assert params.num_rays == 500
    assert params.raw_params == {}


def test_rt_params_from_dict_with_raw():
    """Test creating params from dictionary with raw params."""
    data = {
        "raytracer_name": "TestRT",
        "raytracer_version": "1.0",
        "frequency": 28e9,
        "max_path_depth": 5,
        "max_reflections": 3,
        "max_diffractions": 1,
        "max_scattering": 0,
        "max_transmissions": 0,
    }
    raw = {"some_internal_key": 123}

    params = RayTracingParameters.from_dict(data, raw_params=raw)
    assert params.raw_params == raw


def test_rt_params_read_parameters_abstract():
    """Test that read_parameters raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        RayTracingParameters.read_parameters("dummy/path")

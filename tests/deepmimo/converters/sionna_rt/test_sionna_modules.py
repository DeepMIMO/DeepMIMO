import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from deepmimo.converters.sionna_rt import sionna_materials, sionna_rt_params, sionna_scene, sionna_txrx
from deepmimo import consts as c

@patch("deepmimo.converters.sionna_rt.sionna_materials.load_pickle")
def test_read_materials(mock_load):
    # Mock materials
    mock_materials = [
        {
            "scattering_pattern": "LambertianPattern",
            "scattering_coefficient": 0.5,
            "relative_permittivity": 5.0,
            "conductivity": 0.1,
            "xpd_coefficient": 0.0,
            "alpha_r": 4.0,
            "alpha_i": 4.0,
            "lambda_": 0.5
        }
    ]
    mock_indices = {"obj1": 0}
    
    def side_effect(path):
        if "sionna_materials.pkl" in path:
            return mock_materials
        if "sionna_material_indices.pkl" in path:
            return mock_indices
        return None
    mock_load.side_effect = side_effect
    
    mats, idxs = sionna_materials.read_materials("/dummy/path")
    assert len(mats) == 1
    assert "material_0" in mats
    assert mats["material_0"]["permittivity"] == 5.0
    assert idxs["obj1"] == 0

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
        "min_lat": 0, "min_lon": 0, "max_lat": 0, "max_lon": 0
    }
    mock_load.return_value = raw_params
    
    params = sionna_rt_params.read_rt_params("/dummy/path")
    assert params["raytracer_name"] == c.RAYTRACER_NAME_SIONNA
    assert params["max_path_depth"] == 5
    assert params["max_reflections"] == 5 # if reflection=True, max_reflections=max_depth

@patch("deepmimo.converters.sionna_rt.sionna_scene.load_pickle")
@patch("deepmimo.converters.sionna_rt.sionna_scene.get_object_faces")
def test_read_scene(mock_get_faces, mock_load):
    # Mock vertices: 4 vertices forming a quad
    mock_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
    ])
    mock_objects = {"floor": (0, 4)} # name -> range
    
    def side_effect(path):
        if "sionna_vertices.pkl" in path:
            return mock_vertices
        if "sionna_objects.pkl" in path:
            return mock_objects
        return None
    mock_load.side_effect = side_effect
    
    # Mock face generation
    mock_get_faces.return_value = [
        [(0,0,0), (1,0,0), (1,1,0), (0,1,0)] # One face
    ]
    
    material_indices = [0] # one object
    
    scene = sionna_scene.read_scene("/dummy/path", material_indices)
    
    assert len(scene.objects) == 1
    obj = scene.objects[0]
    assert obj.name == "floor"
    assert obj.label == "terrain" # "floor" keyword

def test_read_txrx():
    raw_params = {
        "tx_array_num_ant": 4,
        "tx_array_ant_pos": np.zeros((4, 3)),
        "tx_array_size": 4, # 4 elements, 4 antennas -> single pol
        
        "rx_array_num_ant": 1,
        "rx_array_ant_pos": np.zeros((1, 3)),
        "rx_array_size": 1,
    }
    rt_params = {
        "raw_params": raw_params,
        "synthetic_array": False
    }
    
    txrx_dict = sionna_txrx.read_txrx(rt_params)
    
    assert "txrx_set_0" in txrx_dict
    assert "txrx_set_1" in txrx_dict
    
    tx = txrx_dict["txrx_set_0"]
    assert tx["is_tx"]
    assert tx["num_ant"] == 4
    
    rx = txrx_dict["txrx_set_1"]
    assert rx["is_rx"]
    assert rx["num_ant"] == 1


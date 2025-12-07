import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from deepmimo.converters.aodt import aodt_paths, aodt_utils, aodt_converter
from deepmimo import consts as c

# --- Test aodt_utils ---
def test_dict_to_array():
    pt = {"1": 1.0, "2": 2.0, "3": 3.0}
    arr = aodt_utils.dict_to_array(pt)
    np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

def test_process_points():
    pts = [{"1": 1.0, "2": 2.0, "3": 3.0}, np.array([4.0, 5.0, 6.0])]
    arr = aodt_utils.process_points(pts)
    assert arr.shape == (2, 3)
    np.testing.assert_array_equal(arr[0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(arr[1], [4.0, 5.0, 6.0])

# --- Test aodt_paths ---
def test_transform_interaction_types():
    # LoS: only emission and reception
    types_los = np.array(["emission", "reception"], dtype=object)
    assert aodt_paths._transform_interaction_types(types_los) == c.INTERACTION_LOS
    
    # Reflection
    types_ref = np.array(["emission", "reflection", "reception"], dtype=object)
    assert aodt_paths._transform_interaction_types(types_ref) == 1.0
    
    # Ref + Diff
    types_mixed = np.array(["emission", "reflection", "diffraction", "reception"], dtype=object)
    assert aodt_paths._transform_interaction_types(types_mixed) == 12.0

# --- Test aodt_converter ---
@patch("deepmimo.converters.aodt.aodt_converter.read_rt_params")
@patch("deepmimo.converters.aodt.aodt_converter.read_txrx")
@patch("deepmimo.converters.aodt.aodt_converter.read_paths")
@patch("deepmimo.converters.aodt.aodt_converter.read_materials")
@patch("deepmimo.converters.aodt.aodt_converter.cu")
@patch("deepmimo.converters.aodt.aodt_converter.os.makedirs")
@patch("deepmimo.converters.aodt.aodt_converter.shutil.rmtree")
def test_aodt_rt_converter(mock_rmtree, mock_makedirs, mock_cu, mock_read_mat, mock_read_paths, mock_read_txrx, mock_read_params):
    mock_cu.check_scenario_exists.return_value = True
    mock_read_params.return_value = {}
    mock_read_txrx.return_value = {}
    mock_read_mat.return_value = {}
    
    res = aodt_converter.aodt_rt_converter("/dummy/path", vis_scene=False)
    
    assert res == "path" # basename of /dummy/path
    mock_read_paths.assert_called()
    mock_cu.save_params.assert_called()
    mock_cu.save_scenario.assert_called()


"""Tests for DeepMIMO core structure modules."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from deepmimo import rt_params, scene, txrx
from deepmimo.materials import Material, MaterialList

# --- rt_params.py ---
def test_ray_tracing_parameters():
    params = rt_params.RayTracingParameters(
        raytracer_name="test",
        raytracer_version="1.0",
        frequency=28e9,
        max_path_depth=5,
        max_reflections=2,
        max_diffractions=1,
        max_scattering=0,
        max_transmissions=0
    )
    assert params.frequency == 28e9
    d = params.to_dict()
    assert d["raytracer_name"] == "test"
    
    # Test from_dict
    p2 = rt_params.RayTracingParameters.from_dict(d)
    assert p2.frequency == 28e9

# --- txrx.py ---
def test_txrx_set():
    ts = txrx.TxRxSet(name="BS1", id=0, is_tx=True, num_points=1)
    assert ts.is_tx
    assert not ts.is_rx
    assert str(ts) == "TXSet(name='BS1', id=0, points=1)"
    
    ts_dict = ts.to_dict()
    assert ts_dict["name"] == "BS1"

def test_txrx_pair():
    tx = txrx.TxRxSet(name="BS", id=0, is_tx=True, num_points=2)
    rx = txrx.TxRxSet(name="UE", id=1, is_rx=True, num_points=5)
    pair = txrx.TxRxPair(tx=tx, rx=rx, tx_idx=0)
    assert pair.get_ids() == (0, 1)
    assert "BS[0]" in str(pair)

def test_get_txrx_pairs():
    tx = txrx.TxRxSet(name="BS", id=0, is_tx=True, num_points=2)
    rx = txrx.TxRxSet(name="UE", id=1, is_rx=True, num_points=1)
    sets = [tx, rx]
    pairs = txrx.get_txrx_pairs(sets)
    # Should have 2 pairs: (BS[0], UE) and (BS[1], UE)
    assert len(pairs) == 2
    assert pairs[0].tx_idx == 0
    assert pairs[1].tx_idx == 1

# --- scene.py ---
def test_bounding_box():
    bb = scene.BoundingBox(0, 10, 0, 20, 0, 5)
    assert bb.width == 10
    assert bb.length == 20
    assert bb.height == 5
    np.testing.assert_array_equal(bb.center, [5, 10, 2.5])

def test_face():
    vertices = [[0,0,0], [1,0,0], [0,1,0]]
    face = scene.Face(vertices)
    assert face.area == 0.5
    np.testing.assert_array_equal(face.normal, [0, 0, 1])
    assert face.num_triangular_faces == 1

def test_physical_element():
    vertices = [[0,0,0], [1,0,0], [0,1,0]]
    face = scene.Face(vertices, material_idx=1)
    elem = scene.PhysicalElement(faces=[face], name="Obj1")
    assert elem.name == "Obj1"
    assert 1 in elem.materials
    
    # Test bounding box computation
    assert elem.bounding_box.x_min == 0
    assert elem.bounding_box.x_max == 1

def test_scene():
    sc = scene.Scene()
    vertices = [[0,0,0], [1,0,0], [0,1,0]]
    face = scene.Face(vertices, material_idx=1)
    elem = scene.PhysicalElement(faces=[face], name="Obj1")
    
    sc.add_object(elem)
    assert len(sc.objects) == 1
    
    group = sc.get_objects()
    assert len(group) == 1
    
    # Test export/load flow (mocked)
    with patch("deepmimo.scene.save_mat") as mock_save_mat, \
         patch("deepmimo.scene.save_dict_as_json") as mock_save_json, \
         patch("pathlib.Path.mkdir"):
        
        meta = sc.export_data("test_folder")
        assert meta["n_objects"] == 1
        mock_save_mat.assert_called()
        mock_save_json.assert_called()

def test_get_object_faces():
    # Test fast mode (convex hull)
    # Cube vertices
    vertices = [
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,1], [1,0,1], [1,1,1], [0,1,1]
    ]
    faces = scene.get_object_faces(vertices, fast=True)
    assert len(faces) >= 6 # Cube has 6 faces, but convex hull generator might return more or less depending on collinearity
    # For simple cube, it should return top, bottom + 4 sides = 6.
    assert len(faces) == 6


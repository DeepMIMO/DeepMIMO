"""Tests for Sionna Scene."""

from unittest.mock import patch

import numpy as np

from deepmimo.converters.sionna_rt import sionna_scene


@patch("deepmimo.converters.sionna_rt.sionna_scene.load_pickle")
@patch("deepmimo.converters.sionna_rt.sionna_scene.get_object_faces")
def test_read_scene(mock_get_faces, mock_load) -> None:
    # Mock vertices: 4 vertices forming a quad
    mock_vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    mock_objects = {"floor": (0, 4)}  # name -> range

    def side_effect(path):
        if "sionna_vertices.pkl" in path:
            return mock_vertices
        if "sionna_objects.pkl" in path:
            return mock_objects
        return None

    mock_load.side_effect = side_effect

    # Mock face generation
    mock_get_faces.return_value = [
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]  # One face
    ]

    material_indices = [0]  # one object

    scene = sionna_scene.read_scene("/dummy/path", material_indices)

    assert len(scene.objects) == 1
    obj = scene.objects[0]
    assert obj.name == "floor"
    assert obj.label == "terrain"  # "floor" keyword

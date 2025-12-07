"""Tests for DeepMIMO Scene module."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from deepmimo.scene import (
    BoundingBox, Face, PhysicalElement, PhysicalElementGroup, Scene,
    CAT_BUILDINGS, CAT_TERRAIN, CAT_OBJECTS
)

# --- BoundingBox Tests ---
def test_bounding_box():
    bb = BoundingBox(0, 10, 0, 20, 0, 5)
    assert bb.x_min == 0
    assert bb.x_max == 10
    assert bb.width == 10
    assert bb.length == 20
    assert bb.height == 5
    np.testing.assert_array_equal(bb.center, [5, 10, 2.5])

# --- Face Tests ---
def test_face_properties():
    # Defined counter-clockwise in xy plane
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    face = Face(vertices)
    
    # Normal should be (0, 0, 1)
    np.testing.assert_array_almost_equal(face.normal, [0, 0, 1])
    
    # Area should be 1
    assert face.area == 1.0
    
    # Centroid
    np.testing.assert_array_almost_equal(face.centroid, [0.5, 0.5, 0])
    
    # Triangular faces (fan triangulation)
    # [0, 1, 2] and [0, 2, 3]
    assert face.num_triangular_faces == 2
    tris = face.triangular_faces
    assert len(tris) == 2

# --- PhysicalElement Tests ---
def test_physical_element():
    # Create a simple cube
    # Bottom face
    f1 = Face([[0,0,0], [1,0,0], [1,1,0], [0,1,0]])
    # Top face
    f2 = Face([[0,0,1], [1,0,1], [1,1,1], [0,1,1]])
    
    obj = PhysicalElement(faces=[f1, f2], name="Cube", label=CAT_BUILDINGS)
    
    assert obj.name == "Cube"
    assert obj.label == CAT_BUILDINGS
    assert len(obj.faces) == 2
    
    # Bounding box
    assert obj.bounding_box.z_max == 1
    assert obj.bounding_box.height == 1
    
    # Position
    np.testing.assert_array_almost_equal(obj.position, [0.5, 0.5, 0.5])
    
    # Velocity setter
    obj.vel = [1, 2, 3]
    np.testing.assert_array_equal(obj.vel, [1, 2, 3])
    
    with pytest.raises(ValueError):
        obj.vel = [1, 2] # Wrong shape

# --- PhysicalElementGroup Tests ---
def test_physical_element_group():
    obj1 = PhysicalElement([Face([[0,0,0], [1,0,0], [0,1,0]])], label=CAT_BUILDINGS)
    obj2 = PhysicalElement([Face([[2,0,0], [3,0,0], [2,1,0]])], label=CAT_TERRAIN)
    
    group = PhysicalElementGroup([obj1, obj2])
    assert len(group) == 2
    assert group[0] == obj1
    
    # Filter
    buildings = group.get_objects(label=CAT_BUILDINGS)
    assert len(buildings) == 1
    assert buildings[0] == obj1
    
    # Bounding box of group
    bb = group.bounding_box
    assert bb.x_min == 0
    assert bb.x_max == 3

# --- Scene Tests ---
def test_scene_management():
    scene = Scene()
    obj = PhysicalElement([Face([[0,0,0], [1,0,0], [0,1,0]])], label=CAT_OBJECTS)
    
    scene.add_object(obj)
    assert len(scene.objects) == 1
    assert len(scene.get_objects(CAT_OBJECTS)) == 1
    
    # Counts
    counts = scene.count_objects_by_label()
    assert counts[CAT_OBJECTS] == 1
    
    # Bounding box
    assert scene.bounding_box is not None

def test_scene_export_import(tmp_path):
    scene = Scene()
    # Add a simple object
    obj = PhysicalElement([Face([[0,0,0], [1,0,0], [0,1,0]])], name="Tri", label=CAT_OBJECTS)
    scene.add_object(obj)
    
    base_folder = str(tmp_path / "scene_data")
    
    # Export
    metadata = scene.export_data(base_folder)
    assert metadata["n_objects"] == 1
    
    # Import
    scene2 = Scene.from_data(base_folder)
    assert len(scene2.objects) == 1
    assert scene2.objects[0].name == "Tri"
    # Check vertices roughly match
    np.testing.assert_array_almost_equal(
        scene2.objects[0].faces[0].vertices,
        obj.faces[0].vertices
    )

@patch("matplotlib.pyplot.subplots")
def test_scene_plot(mock_subplots):
    """Test plotting calls."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    
    scene = Scene()
    obj = PhysicalElement([Face([[0,0,0], [1,0,0], [0,1,0]])], label=CAT_OBJECTS)
    scene.add_object(obj)
    
    # Plot 3D
    scene.plot(proj_3D=True)
    # Check if add_collection3d was called
    assert mock_ax.add_collection3d.called
    
    # Plot 2D
    scene.plot(proj_3D=False)
    # Check if fill was called
    assert mock_ax.fill.called

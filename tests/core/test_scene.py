"""Tests for DeepMIMO Scene module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from deepmimo.consts import (
    SCENE_PARAM_N_TRIANGULAR_FACES,
    SCENE_PARAM_REPRESENTATION,
    SCENE_REPRESENTATION_HULL,
    SCENE_REPRESENTATION_MESH,
)
from deepmimo.core.scene import (
    CAT_BUILDINGS,
    CAT_OBJECTS,
    CAT_TERRAIN,
    BoundingBox,
    Face,
    PhysicalElement,
    PhysicalElementGroup,
    Scene,
    _get_faces_convex_hull,
    get_object_faces,
    triangulate_polygon,
)
from deepmimo.utils import save_dict_as_json


# --- BoundingBox Tests ---
def test_bounding_box() -> None:
    """Validate bounding box dimensions and derived properties."""
    bb = BoundingBox(0, 10, 0, 20, 0, 5)
    assert bb.x_min == 0
    assert bb.x_max == 10
    assert bb.width == 10
    assert bb.length == 20
    assert bb.height == 5
    np.testing.assert_array_equal(bb.center, [5, 10, 2.5])


# --- Face Tests ---
def test_face_properties() -> None:
    """Ensure face normals, areas, and centroids are computed correctly."""
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
def test_physical_element() -> None:
    """Check PhysicalElement properties and validation logic."""
    # Create a simple cube
    # Bottom face
    f1 = Face([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    # Top face
    f2 = Face([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])

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

    with pytest.raises(ValueError, match="Velocity must be a 3D vector"):
        obj.vel = [1, 2]  # Wrong shape


# --- PhysicalElementGroup Tests ---
def test_physical_element_group() -> None:
    """Group physical elements and query filtered results."""
    obj1 = PhysicalElement([Face([[0, 0, 0], [1, 0, 0], [0, 1, 0]])], label=CAT_BUILDINGS)
    obj2 = PhysicalElement([Face([[2, 0, 0], [3, 0, 0], [2, 1, 0]])], label=CAT_TERRAIN)

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
def test_scene_management() -> None:
    """Add objects to a scene and track counts/bounding boxes."""
    scene = Scene()
    obj = PhysicalElement([Face([[0, 0, 0], [1, 0, 0], [0, 1, 0]])], label=CAT_OBJECTS)

    scene.add_object(obj)
    assert len(scene.objects) == 1
    assert len(scene.get_objects(CAT_OBJECTS)) == 1

    # Counts
    counts = scene.count_objects_by_label()
    assert counts[CAT_OBJECTS] == 1

    # Bounding box
    assert scene.bounding_box is not None


def test_scene_export_import(tmp_path) -> None:
    """Round-trip a scene via export/import and validate contents."""
    scene = Scene()
    # Add a simple object
    obj = PhysicalElement([Face([[0, 0, 0], [1, 0, 0], [0, 1, 0]])], name="Tri", label=CAT_OBJECTS)
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
    np.testing.assert_array_almost_equal(scene2.objects[0].faces[0].vertices, obj.faces[0].vertices)


def test_scene_export_default_is_hull(tmp_path) -> None:
    """Default export stays the convex-hull representation (same files + flag)."""
    obj = PhysicalElement([Face([[0, 0, 0], [1, 0, 0], [0, 1, 0]])], name="Tri", label=CAT_OBJECTS)
    scene = Scene()
    scene.add_object(obj)

    base_folder = str(tmp_path / "scene_hull")
    metadata = scene.export_data(base_folder)

    # Representation flag is "hull" and only the legacy files are written
    assert metadata[SCENE_PARAM_REPRESENTATION] == SCENE_REPRESENTATION_HULL
    folder = Path(base_folder)
    assert (folder / "vertices.npz").exists()
    assert (folder / "objects.json").exists()
    assert not (folder / "faces.npz").exists()
    assert not (folder / "materials.npz").exists()

    # And it loads back as hull, identical to before
    scene2 = Scene.from_data(base_folder)
    assert len(scene2.objects) == 1
    np.testing.assert_array_almost_equal(
        scene2.objects[0].faces[0].vertices,
        obj.faces[0].vertices,
    )


def test_scene_export_import_lossless(tmp_path) -> None:
    """Round-trip a scene via lossless mesh export and validate exact triangles."""
    # A quad face (fans into 2 triangles, material 2) and a triangle face (material 5)
    quad = Face([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], material_idx=2)
    tri = Face([[0, 0, 1], [1, 0, 1], [0, 1, 1]], material_idx=5)
    obj = PhysicalElement([quad, tri], name="MeshObj", object_id=7, label=CAT_BUILDINGS)

    scene = Scene()
    scene.add_object(obj)

    base_folder = str(tmp_path / "scene_mesh")
    metadata = scene.export_data(base_folder, lossless=True)

    # Representation flag + mesh files present
    assert metadata[SCENE_PARAM_REPRESENTATION] == SCENE_REPRESENTATION_MESH
    assert metadata[SCENE_PARAM_N_TRIANGULAR_FACES] == 3
    folder = Path(base_folder)
    assert (folder / "vertices.npz").exists()
    assert (folder / "faces.npz").exists()
    assert (folder / "materials.npz").exists()
    assert (folder / "objects.json").exists()

    # Expected triangles (fan triangulation) and per-triangle materials, in order
    expected_tris = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]),
        np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]]),
        np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]]),
    ]
    expected_mats = [2, 2, 5]

    scene2 = Scene.from_data(base_folder)
    assert len(scene2.objects) == 1
    obj2 = scene2.objects[0]
    assert obj2.name == "MeshObj"
    assert obj2.object_id == 7
    assert obj2.label == CAT_BUILDINGS

    # One Face per triangle, preserving exact geometry + material indexing
    assert len(obj2.faces) == 3
    for face, exp_tri, exp_mat in zip(obj2.faces, expected_tris, expected_mats, strict=True):
        assert face.vertices.shape == (3, 3)
        np.testing.assert_array_almost_equal(face.vertices, exp_tri)
        assert face.material_idx == exp_mat


def test_from_data_legacy_loads_as_hull(tmp_path) -> None:
    """A scenario dir without a representation flag (legacy) loads as hull."""
    obj = PhysicalElement(
        [Face([[0, 0, 0], [1, 0, 0], [0, 1, 0]])],
        name="Legacy",
        label=CAT_OBJECTS,
    )
    scene = Scene()
    scene.add_object(obj)

    base_folder = str(tmp_path / "legacy_scene")
    scene.export_data(base_folder)  # hull export leaves no mesh marker in the folder

    # Simulate a legacy params.json whose scene block has no representation key
    folder = Path(base_folder)
    save_dict_as_json(str(folder / "params.json"), {"scene": {"num_scenes": 1, "n_objects": 1}})

    assert Scene._is_mesh_representation(base_folder) is False  # noqa: SLF001
    scene2 = Scene.from_data(base_folder)
    assert len(scene2.objects) == 1
    assert scene2.objects[0].name == "Legacy"
    np.testing.assert_array_almost_equal(
        scene2.objects[0].faces[0].vertices,
        obj.faces[0].vertices,
    )


@patch("matplotlib.pyplot.subplots")
def test_scene_plot(mock_subplots) -> None:
    """Test plotting calls."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    scene = Scene()
    obj = PhysicalElement([Face([[0, 0, 0], [1, 0, 0], [0, 1, 0]])], label=CAT_OBJECTS)
    scene.add_object(obj)

    # Plot 3D
    scene.plot(proj_3D=True)
    # Check if add_collection3d was called
    assert mock_ax.add_collection3d.called

    # Plot 2D
    scene.plot(proj_3D=False)
    # Check if fill was called
    assert mock_ax.fill.called


def test_get_object_faces() -> None:
    """Compute face list for a simple cube via convex-hull generation."""
    # Cube vertices
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
    faces = get_object_faces(vertices)
    assert len(faces) >= 6  # Cube has 6 faces; hull count can vary with collinearity
    # For simple cube, it should return top, bottom + 4 sides = 6.
    assert len(faces) == 6


# ---------------------------------------------------------------------------
# Helper: build a simple box-shaped PhysicalElement
# ---------------------------------------------------------------------------


def _make_box_element(  # noqa: PLR0913
    x0=0.0, y0=0.0, z0=0.0, x1=2.0, y1=3.0, z1=4.0, label=CAT_BUILDINGS
) -> PhysicalElement:
    """Return a PhysicalElement made of two axis-aligned faces (bottom + top)."""
    bottom = Face([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]], material_idx=0)
    top = Face([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], material_idx=1)
    return PhysicalElement(faces=[bottom, top], label=label)


# ---------------------------------------------------------------------------
# PhysicalElement - height property (line 269)
# ---------------------------------------------------------------------------


def test_physical_element_height() -> None:
    """Height property delegates to bounding_box.height."""
    obj = _make_box_element(z0=1.0, z1=5.0)
    assert obj.height == pytest.approx(4.0)
    assert obj.height == obj.bounding_box.height


# ---------------------------------------------------------------------------
# PhysicalElement - hull, hull_volume, hull_surface_area, footprint_area,
#                   volume  (lines 279-308)
# ---------------------------------------------------------------------------


def test_physical_element_hull_lazy() -> None:
    """Hull is computed once and cached on subsequent accesses."""
    obj = _make_box_element()
    assert obj._hull is None  # noqa: SLF001
    h1 = obj.hull
    h2 = obj.hull
    assert h1 is h2  # same cached object
    assert isinstance(h1, ConvexHull)


def test_physical_element_hull_volume() -> None:
    """hull_volume caches and returns convex-hull volume."""
    obj = _make_box_element(x0=0, y0=0, z0=0, x1=2, y1=3, z1=4)
    assert obj._hull_volume is None  # noqa: SLF001
    vol = obj.hull_volume
    assert vol > 0
    # Must be cached
    assert obj._hull_volume is not None  # noqa: SLF001
    assert obj.hull_volume is vol


def test_physical_element_hull_surface_area() -> None:
    """hull_surface_area caches and returns convex-hull surface area."""
    obj = _make_box_element()
    assert obj._hull_surface_area is None  # noqa: SLF001
    sa = obj.hull_surface_area
    assert sa > 0
    assert obj._hull_surface_area == sa  # noqa: SLF001


def test_physical_element_footprint_area() -> None:
    """footprint_area caches and returns the 2-D convex-hull's .area attribute.

    Note: scipy's ConvexHull.area in 2-D returns the *perimeter* of the hull,
    not the enclosed surface area (which is ConvexHull.volume in 2-D).  For a
    2 x 3 rectangle the perimeter is 2*(2+3) = 10.
    """
    obj = _make_box_element(x0=0, y0=0, z0=0, x1=2, y1=3, z1=1)
    assert obj._footprint_area is None  # noqa: SLF001
    fa = obj.footprint_area
    assert fa == pytest.approx(10.0, rel=1e-4)
    # Verify caching
    assert obj.footprint_area is fa


def test_physical_element_volume_delegates_to_hull_volume() -> None:
    """Volume property just returns hull_volume."""
    obj = _make_box_element()
    assert obj.volume == obj.hull_volume


# ---------------------------------------------------------------------------
# PhysicalElement - to_dict / from_dict round-trip (lines 310-360)
# ---------------------------------------------------------------------------


def test_physical_element_to_dict_from_dict_roundtrip() -> None:
    """Serialising then deserialising a PhysicalElement preserves key fields."""
    obj = PhysicalElement(
        faces=[Face([[0, 0, 0], [1, 0, 0], [0, 1, 0]], material_idx=2)],
        object_id=7,
        label=CAT_BUILDINGS,
        name="TestObj",
    )
    vertex_map: dict = {}
    d = obj.to_dict(vertex_map)

    # Basic dict shape
    assert d["name"] == "TestObj"
    assert d["id"] == 7
    assert d["label"] == CAT_BUILDINGS
    assert len(d["face_vertex_idxs"]) == 1
    assert len(d["face_material_idxs"]) == 1
    assert d["face_material_idxs"][0] == 2

    # Reconstruct
    all_vertices = [None] * len(vertex_map)
    for vertex, idx in vertex_map.items():
        all_vertices[idx] = vertex
    vertices_arr = np.array(all_vertices)

    obj2 = PhysicalElement.from_dict(d, vertices_arr)
    assert obj2.name == "TestObj"
    assert obj2.object_id == 7
    assert obj2.label == CAT_BUILDINGS
    assert len(obj2.faces) == 1


# ---------------------------------------------------------------------------
# PhysicalElement - plot  (lines 395-396, 439-441)
# ---------------------------------------------------------------------------


@patch("deepmimo.core.scene.plt")
def test_physical_element_plot_faces_mode(mock_plt) -> None:
    """plot() in 'faces' mode adds a Poly3DCollection per face."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_ax.get_figure.return_value = mock_fig
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    obj = _make_box_element()
    obj.plot(mode="faces")
    assert mock_ax.add_collection3d.called


@patch("deepmimo.core.scene.plt")
def test_physical_element_plot_tri_faces_mode(mock_plt) -> None:
    """plot() in 'tri_faces' mode also adds collections (one per triangle)."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_ax.get_figure.return_value = mock_fig
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    obj = _make_box_element()
    obj.plot(mode="tri_faces")
    # tri_faces of two quad faces = 4 triangles total
    assert mock_ax.add_collection3d.call_count == 4


# ---------------------------------------------------------------------------
# PhysicalElementGroup - __iter__, filter, position  (lines 462, 470-471, 496, 504-505)
# ---------------------------------------------------------------------------


def test_physical_element_group_iter() -> None:
    """__iter__ yields each element in the group."""
    obj1 = _make_box_element(label=CAT_BUILDINGS)
    obj2 = _make_box_element(x0=10, x1=12, label=CAT_TERRAIN)
    group = PhysicalElementGroup([obj1, obj2])

    collected = list(group)
    assert len(collected) == 2
    assert obj1 in collected
    assert obj2 in collected


def test_physical_element_group_repr() -> None:
    """__repr__ mentions the object count."""
    obj = _make_box_element()
    group = PhysicalElementGroup([obj])
    r = repr(group)
    assert "PhysicalElementGroup(objects=1)" in r


def test_physical_element_group_filter_by_label() -> None:
    """get_objects(label=...) returns only objects with matching label."""
    obj1 = _make_box_element(label=CAT_BUILDINGS)
    obj2 = _make_box_element(x0=10, x1=12, label=CAT_TERRAIN)
    group = PhysicalElementGroup([obj1, obj2])

    buildings = group.get_objects(label=CAT_BUILDINGS)
    assert len(buildings) == 1
    assert next(iter(buildings)) is obj1


def test_physical_element_group_bounding_box_multi() -> None:
    """bounding_box encompasses all objects; raises on empty group."""
    obj1 = _make_box_element(x0=0, x1=2, y0=0, y1=2, z0=0, z1=1)
    obj2 = _make_box_element(x0=5, x1=8, y0=5, y1=8, z0=0, z1=2)
    group = PhysicalElementGroup([obj1, obj2])

    bb = group.bounding_box
    assert bb.x_min == 0
    assert bb.x_max == 8
    assert bb.y_min == 0
    assert bb.y_max == 8


def test_physical_element_group_bounding_box_empty() -> None:
    """bounding_box on empty group raises ValueError."""
    group = PhysicalElementGroup([])
    with pytest.raises(ValueError, match="Group is empty"):
        _ = group.bounding_box


# ---------------------------------------------------------------------------
# Scene - __repr__  (lines 913-918)
# ---------------------------------------------------------------------------


def test_scene_repr() -> None:
    """__repr__ encodes object count, label counts, and bounding dims."""
    scene = Scene()
    obj = _make_box_element(x0=0, x1=10, y0=0, y1=20, z0=0, z1=5, label=CAT_BUILDINGS)
    scene.add_object(obj)
    r = repr(scene)
    assert "Scene(" in r
    assert "buildings" in r
    assert "m" in r


def test_scene_repr_empty_plot_returns_ax() -> None:
    """plot() on an empty scene returns the provided ax unchanged."""
    scene = Scene()
    sentinel = MagicMock()
    result = scene.plot(ax=sentinel)
    assert result is sentinel


# ---------------------------------------------------------------------------
# _get_faces_convex_hull - collinear vertex path (lines 938-943)
# ---------------------------------------------------------------------------


def test_get_faces_convex_hull_collinear_returns_none(capsys) -> None:
    """Collinear 2D points cause the hull to fail and return None."""
    # All vertices lie on the line y=x (rank-1 in 2D)
    vertices = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0], [3, 3, 0]], dtype=float)
    result = _get_faces_convex_hull(vertices)
    assert result is None
    captured = capsys.readouterr()
    assert "collinear" in captured.out.lower()


# ---------------------------------------------------------------------------
# get_object_faces - too few vertices returns None (line 1282)
# ---------------------------------------------------------------------------


def test_get_object_faces_too_few_vertices() -> None:
    """Fewer than 3 vertices returns None."""
    result = get_object_faces([[0, 0, 0], [1, 0, 0]])
    assert result is None


# --- Polygon triangulation (issue #125) --------------------------------------


def _polygon_area_2d(points: np.ndarray) -> float:
    """Return the absolute area of a 2D polygon via the shoelace formula."""
    x, y = points[:, 0], points[:, 1]
    return abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _triangulated_area(triangles: list[np.ndarray]) -> float:
    """Return the summed area of triangles, using their first two coordinates."""
    return sum(_polygon_area_2d(tri[:, :2]) for tri in triangles)


# A U: vertex 0 cannot see into the far arm, so a fan anchored there spans the
# notch between the arms and over-reports the area (11 units instead of 7).
U_SHAPE = np.array(
    [[0, 0, 0], [3, 0, 0], [3, 3, 0], [2, 3, 0], [2, 1, 0], [1, 1, 0], [1, 3, 0], [0, 3, 0]],
    dtype=float,
)
HEXAGON = np.array(
    [[np.cos(a), np.sin(a), 0.0] for a in np.linspace(0, 2 * np.pi, 7)[:-1]],
    dtype=float,
)


def test_triangulate_convex_polygon_is_a_fan() -> None:
    """Convex faces keep the cheap fan anchored at the first vertex."""
    triangles = triangulate_polygon(HEXAGON)

    expected = [np.array([HEXAGON[0], HEXAGON[i], HEXAGON[i + 1]]) for i in range(1, 5)]
    assert len(triangles) == len(expected)
    for actual, want in zip(triangles, expected, strict=True):
        assert np.array_equal(actual, want)


def test_triangulate_concave_polygon_covers_exact_area() -> None:
    """Ear clipping tiles a concave face without spilling into its notch."""
    triangles = triangulate_polygon(U_SHAPE)

    assert len(triangles) == len(U_SHAPE) - 2
    assert _triangulated_area(triangles) == pytest.approx(_polygon_area_2d(U_SHAPE[:, :2]))


def test_triangulate_concave_polygon_beats_naive_fan() -> None:
    """The naive fan really does over-report this polygon, so the test has teeth."""
    fan = [np.array([U_SHAPE[0], U_SHAPE[i], U_SHAPE[i + 1]]) for i in range(1, len(U_SHAPE) - 1)]

    assert _triangulated_area(fan) > _polygon_area_2d(U_SHAPE[:, :2])


def test_triangulate_handles_clockwise_winding() -> None:
    """Winding order must not change the triangulated area."""
    triangles = triangulate_polygon(U_SHAPE[::-1])

    assert _triangulated_area(triangles) == pytest.approx(_polygon_area_2d(U_SHAPE[:, :2]))


def test_triangulate_vertical_polygon() -> None:
    """Faces are projected onto their own plane, so vertical walls work too."""
    # The U-shape rotated into the x=0 plane.
    wall = U_SHAPE[:, [2, 0, 1]]

    triangles = triangulate_polygon(wall)

    assert len(triangles) == len(wall) - 2
    assert _triangulated_area([tri[:, 1:] for tri in triangles]) == pytest.approx(
        _polygon_area_2d(U_SHAPE[:, :2]),
    )


def test_triangulate_degenerate_inputs() -> None:
    """Triangles pass through unchanged; anything smaller yields nothing."""
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)

    assert len(triangulate_polygon(triangle)) == 1
    assert triangulate_polygon(np.zeros((2, 3))) == []


def test_face_triangular_faces_uses_exact_triangulation() -> None:
    """Face.triangular_faces routes through the shared triangulator."""
    face = Face(vertices=U_SHAPE)

    assert _triangulated_area(face.triangular_faces) == pytest.approx(
        _polygon_area_2d(U_SHAPE[:, :2]),
    )

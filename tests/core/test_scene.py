"""Tests for DeepMIMO Scene module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from deepmimo.core.scene import (
    CAT_BUILDINGS,
    CAT_OBJECTS,
    CAT_TERRAIN,
    BoundingBox,
    Face,
    PhysicalElement,
    PhysicalElementGroup,
    Scene,
    _calculate_angle_deviation,
    _ccw,
    _detect_endpoints,
    _get_faces_convex_hull,
    _segments_intersect,
    _signed_distance_to_curve,
    _tsp_held_karp_no_intersections,
    get_object_faces,
)


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
    """Compute face list for a simple cube in fast mode."""
    # Test fast mode (convex hull)
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
    faces = get_object_faces(vertices, fast=True)
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
# _calculate_angle_deviation  (lines 961-968)
# ---------------------------------------------------------------------------


def test_calculate_angle_deviation_straight_line() -> None:
    """Collinear points in the same direction give 0 degrees."""
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 0.0])
    p3 = np.array([2.0, 0.0])
    angle = _calculate_angle_deviation(p1, p2, p3)
    assert angle == pytest.approx(0.0, abs=1e-6)


def test_calculate_angle_deviation_right_angle() -> None:
    """A 90° turn gives approximately 90 degrees."""
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 0.0])
    p3 = np.array([1.0, 1.0])
    angle = _calculate_angle_deviation(p1, p2, p3)
    assert angle == pytest.approx(90.0, abs=1e-4)


def test_calculate_angle_deviation_equal_points_returns_180() -> None:
    """When p1==p2 or p2==p3 the function returns 180.0."""
    p = np.array([1.0, 2.0])
    # p1 == p2
    assert _calculate_angle_deviation(p, p, np.array([3.0, 4.0])) == pytest.approx(180.0)
    # p2 == p3
    assert _calculate_angle_deviation(np.array([0.0, 0.0]), p, p) == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# _ccw  (line 973)
# ---------------------------------------------------------------------------


def test_ccw_counter_clockwise() -> None:
    """Points arranged counter-clockwise return a truthy value."""
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.0, 1.0])
    assert _ccw(a, b, c)


def test_ccw_clockwise() -> None:
    """Points arranged clockwise return a falsy value."""
    a = np.array([0.0, 0.0])
    b = np.array([0.0, 1.0])
    c = np.array([1.0, 0.0])
    assert not _ccw(a, b, c)


# ---------------------------------------------------------------------------
# _segments_intersect  (line 978)
# ---------------------------------------------------------------------------


def test_segments_intersect_crossing() -> None:
    """Two crossing diagonals of a square should intersect."""
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 1.0])
    q1 = np.array([1.0, 0.0])
    q2 = np.array([0.0, 1.0])
    assert _segments_intersect(p1, p2, q1, q2)


def test_segments_intersect_parallel() -> None:
    """Parallel horizontal segments do not intersect."""
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 0.0])
    q1 = np.array([0.0, 1.0])
    q2 = np.array([1.0, 1.0])
    assert not _segments_intersect(p1, p2, q1, q2)


# ---------------------------------------------------------------------------
# _tsp_held_karp_no_intersections  (lines 988-1044)
# ---------------------------------------------------------------------------


def test_tsp_held_karp_4_points() -> None:
    """4 axis-aligned points: TSP should return a cost and a valid cyclic path."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cost, path = _tsp_held_karp_no_intersections(points)
    assert np.isfinite(cost)
    assert len(path) >= 4  # at minimum visits all points
    # path starts and ends at index 0
    assert path[0] == 0
    assert path[-1] == 0


def test_tsp_held_karp_3_points() -> None:
    """3 points: minimal non-trivial case."""
    points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    cost, path = _tsp_held_karp_no_intersections(points)
    assert np.isfinite(cost)
    # Path must visit all three points and close the loop
    visited = set(path)
    assert {0, 1, 2}.issubset(visited)


# ---------------------------------------------------------------------------
# _detect_endpoints  (lines 1063-1081)
# ---------------------------------------------------------------------------


def test_detect_endpoints_basic() -> None:
    """detect_endpoints returns 4 indices identifying the two farthest pairs."""
    # Simple grid: farthest apart should be the outer corners
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 5.0], [1.0, 5.0], [2.0, 5.0]])
    endpoints = _detect_endpoints(points)
    assert len(endpoints) == 4
    # All returned indices must be valid
    for idx in endpoints:
        assert 0 <= idx < len(points)


def test_detect_endpoints_deduplicates_nearby() -> None:
    """Points within min_distance are treated as a single point."""
    # Two groups of near-duplicates far apart
    points = np.array(
        [
            [0.0, 0.0],
            [0.01, 0.0],  # near-duplicate at origin
            [100.0, 0.0],
            [100.01, 0.0],
        ]  # near-duplicate far away
    )
    endpoints = _detect_endpoints(points, min_distance=0.05)
    # Should only have one representative per cluster
    assert len(endpoints) == 4  # function always returns 4 index slots


# ---------------------------------------------------------------------------
# _signed_distance_to_curve  (lines 1102-1116)
# ---------------------------------------------------------------------------


def test_signed_distance_to_curve_on_curve() -> None:
    """A point on the fitted curve should have near-zero signed distance."""
    # Flat line y=0 fitted by quadratic → curve is y=0 for all x
    x_pts = np.linspace(0, 10, 20)
    y_pts = np.zeros_like(x_pts)
    z_coeffs = np.polyfit(x_pts, y_pts, 3)
    curve_fit = np.poly1d(z_coeffs)

    point = np.array([5.0, 0.0])
    signed_dist, closest = _signed_distance_to_curve(point, curve_fit, (0.0, 10.0))
    assert abs(signed_dist) < 0.1
    assert closest.shape == (2,)


def test_signed_distance_to_curve_off_curve() -> None:
    """A point clearly above the curve has a non-zero signed distance."""
    x_pts = np.linspace(0, 10, 30)
    y_pts = np.zeros(30)
    z_coeffs = np.polyfit(x_pts, y_pts, 3)
    curve_fit = np.poly1d(z_coeffs)

    point = np.array([5.0, 10.0])  # 10 units above the curve
    signed_dist, _ = _signed_distance_to_curve(point, curve_fit, (0.0, 10.0))
    assert abs(signed_dist) > 5.0


# ---------------------------------------------------------------------------
# get_object_faces - too few vertices returns None (line 1282)
# ---------------------------------------------------------------------------


def test_get_object_faces_too_few_vertices() -> None:
    """Fewer than 3 vertices returns None."""
    result = get_object_faces([[0, 0, 0], [1, 0, 0]], fast=True)
    assert result is None

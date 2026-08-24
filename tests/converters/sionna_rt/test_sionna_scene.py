"""Tests for Sionna Scene."""

from unittest.mock import patch

import numpy as np

from deepmimo.converters.sionna_rt import sionna_scene
from deepmimo.converters.sionna_rt.sionna_scene import _cluster_by_aabb
from deepmimo.core.scene import CAT_BUILDINGS, CAT_TERRAIN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_comp(
    mn: tuple[float, float, float],
    mx: tuple[float, float, float],
    label: str = CAT_BUILDINGS,
    name: str = "obj",
) -> sionna_scene._Component:
    """Return a minimal component tuple whose AABB is [mn, mx]."""
    verts = np.array([mn, mx], dtype=float)
    return (verts, 0, label, name, True)


# ---------------------------------------------------------------------------
# _cluster_by_aabb unit tests
# ---------------------------------------------------------------------------


def test_cluster_empty() -> None:
    """Return empty list for empty input."""
    assert _cluster_by_aabb([]) == []


def test_cluster_single() -> None:
    """Single component returns one singleton cluster."""
    comp = _make_comp((0, 0, 0), (1, 1, 1))
    clusters = _cluster_by_aabb([comp])
    assert clusters == [[0]]


def test_cluster_identical_aabbs() -> None:
    """Two components with the same AABB should merge."""
    a = _make_comp((0, 0, 0), (10, 10, 20))
    b = _make_comp((0, 0, 0), (10, 10, 20))
    clusters = _cluster_by_aabb([a, b])
    assert len(clusters) == 1
    assert set(clusters[0]) == {0, 1}


def test_cluster_small_inside_large() -> None:
    """Small component fully inside large → one cluster."""
    large = _make_comp((0, 0, 0), (10, 10, 20))  # vol = 2000
    small = _make_comp((2, 2, 5), (8, 8, 15))  # vol = 360, fully inside
    # overlap ratio = 360/360 = 1.0 >= 0.8
    clusters = _cluster_by_aabb([large, small])
    assert len(clusters) == 1
    assert set(clusters[0]) == {0, 1}


def test_cluster_no_overlap() -> None:
    """Separated components → two clusters."""
    a = _make_comp((0, 0, 0), (5, 5, 5))
    b = _make_comp((10, 10, 10), (15, 15, 15))
    clusters = _cluster_by_aabb([a, b])
    assert len(clusters) == 2


def test_cluster_partial_overlap_below_threshold() -> None:
    """Overlap below 0.8 threshold → two clusters."""
    # A: [0-10]^3, vol=1000; B: [5-20]^3, vol=3375
    # Intersection: [5-10]^3, vol=125; ratio=125/1000=0.125 < 0.8
    a = _make_comp((0, 0, 0), (10, 10, 10))
    b = _make_comp((5, 5, 5), (20, 20, 20))
    clusters = _cluster_by_aabb([a, b])
    assert len(clusters) == 2


def test_cluster_partial_overlap_above_threshold() -> None:
    """Overlap above threshold → one cluster."""
    # A: [0-10]^3, vol=1000; B: [1-9]^3, vol=512
    # Intersection: [1-9]^3, vol=512; ratio=512/512=1.0 >= 0.8
    a = _make_comp((0, 0, 0), (10, 10, 10))
    b = _make_comp((1, 1, 1), (9, 9, 9))
    clusters = _cluster_by_aabb([a, b])
    assert len(clusters) == 1


def test_cluster_different_labels_not_merged() -> None:
    """Building and terrain with same AABB → two clusters (labels differ)."""
    building = _make_comp((0, 0, 0), (10, 10, 20), label=CAT_BUILDINGS)
    terrain = _make_comp((0, 0, 0), (10, 10, 20), label=CAT_TERRAIN)
    clusters = _cluster_by_aabb([building, terrain])
    assert len(clusters) == 2


def test_cluster_three_components_two_same_building() -> None:
    """Large building + small window (same building) + separate building → 2 clusters."""
    large_building = _make_comp((0, 0, 0), (10, 10, 20))  # building A concrete
    window = _make_comp((2, 2, 5), (8, 8, 15))  # building A glass (inside)
    separate = _make_comp((20, 0, 0), (30, 10, 20))  # building B
    clusters = _cluster_by_aabb([large_building, window, separate])
    assert len(clusters) == 2
    # large_building and window must be in the same cluster
    merged = next(cl for cl in clusters if len(cl) == 2)
    assert set(merged) == {0, 1}
    solo = next(cl for cl in clusters if len(cl) == 1)
    assert solo == [2]


def test_cluster_custom_threshold() -> None:
    """Higher threshold prevents merge that default allows."""
    # A: [0-10]^2 x [0-10] vol=1000; B: [0-10]^2 x [-1-8.5] → B vol=950
    # Intersection: [0-10]^2 x [0-8.5] vol=850; ratio=850/950≈0.895
    # With threshold=0.8 → merge; with threshold=0.95 → no merge
    a = _make_comp((0, 0, 0), (10, 10, 10))
    b = _make_comp((0, 0, -1), (10, 10, 8.5))
    # intersection vol = 10*10*8.5 = 850; min vol = min(1000,950)=950; ratio≈0.895
    clusters_default = _cluster_by_aabb([a, b], overlap_threshold=0.8)
    clusters_strict = _cluster_by_aabb([a, b], overlap_threshold=0.95)
    assert len(clusters_default) == 1
    assert len(clusters_strict) == 2


# ---------------------------------------------------------------------------
# read_scene integration tests (mocked I/O)
# ---------------------------------------------------------------------------

# Shared mock geometry: 8 large-building vertices + 8 small-window vertices
# Large building AABB: [0-10, 0-10, 0-20], window AABB: [2-8, 2-8, 5-15]
_LARGE_VERTS = np.array(
    [
        [0, 0, 0],
        [10, 0, 0],
        [10, 10, 0],
        [0, 10, 0],
        [0, 0, 20],
        [10, 0, 20],
        [10, 10, 20],
        [0, 10, 20],
    ],
    dtype=float,
)

_WINDOW_VERTS = np.array(
    [
        [2, 2, 5],
        [8, 2, 5],
        [8, 8, 5],
        [2, 8, 5],
        [2, 2, 15],
        [8, 2, 15],
        [8, 8, 15],
        [2, 8, 15],
    ],
    dtype=float,
)

_MOCK_VERTICES = np.vstack([_LARGE_VERTS, _WINDOW_VERTS])
_MOCK_OBJECTS_MULTI_MATERIAL = {
    "concrete_building": (0, 8),  # 8 verts → large building
    "glass_building": (8, 16),  # 8 verts → small window inside
}
_MOCK_FACES = [
    [(0, 0, 0), (1, 0, 0), (1, 1, 0)],
]


def _make_load_side_effect(objects):
    def side_effect(path):
        if "sionna_vertices.pkl" in path:
            return _MOCK_VERTICES
        if "sionna_objects.pkl" in path:
            return objects
        raise FileNotFoundError(path)

    return side_effect


@patch("deepmimo.converters.sionna_rt.sionna_scene.load_pickle")
@patch("deepmimo.converters.sionna_rt.sionna_scene.get_object_faces")
def test_read_scene_basic(mock_get_faces, mock_load) -> None:
    """Single-material floor object → one terrain object."""
    mock_load.side_effect = _make_load_side_effect({"floor": (0, 4)})
    mock_get_faces.return_value = [_MOCK_FACES[0]]
    scene = sionna_scene.read_scene("/dummy", [0])
    assert len(scene.objects) == 1
    assert scene.objects[0].label == "terrain"


@patch("deepmimo.converters.sionna_rt.sionna_scene.load_pickle")
@patch("deepmimo.converters.sionna_rt.sionna_scene.get_object_faces")
def test_read_scene_deduplicate_true_merges_overlapping(mock_get_faces, mock_load) -> None:
    """Two material groups for the same building → one object with deduplicate=True."""
    mock_load.side_effect = _make_load_side_effect(_MOCK_OBJECTS_MULTI_MATERIAL)
    mock_get_faces.return_value = [_MOCK_FACES[0]]
    scene = sionna_scene.read_scene("/dummy", [0, 1], deduplicate=True)
    assert len(scene.objects) == 1


@patch("deepmimo.converters.sionna_rt.sionna_scene.load_pickle")
@patch("deepmimo.converters.sionna_rt.sionna_scene.get_object_faces")
def test_read_scene_deduplicate_false_keeps_separate(mock_get_faces, mock_load) -> None:
    """Two material groups → two objects with deduplicate=False."""
    mock_load.side_effect = _make_load_side_effect(_MOCK_OBJECTS_MULTI_MATERIAL)
    mock_get_faces.return_value = [_MOCK_FACES[0]]
    scene = sionna_scene.read_scene("/dummy", [0, 1], deduplicate=False)
    assert len(scene.objects) == 2


@patch("deepmimo.converters.sionna_rt.sionna_scene.load_pickle")
@patch("deepmimo.converters.sionna_rt.sionna_scene.get_object_faces")
def test_read_scene_default_deduplicates(mock_get_faces, mock_load) -> None:
    """Default call with overlapping multi-material building → one object."""
    mock_load.side_effect = _make_load_side_effect(_MOCK_OBJECTS_MULTI_MATERIAL)
    mock_get_faces.return_value = [_MOCK_FACES[0]]
    scene = sionna_scene.read_scene("/dummy", [0, 1])
    assert len(scene.objects) == 1


@patch("deepmimo.converters.sionna_rt.sionna_scene.load_pickle")
@patch("deepmimo.converters.sionna_rt.sionna_scene.get_object_faces")
def test_read_scene_dedup_does_not_merge_terrain_with_buildings(mock_get_faces, mock_load) -> None:
    """A terrain plane and a building whose AABBs overlap are NOT merged."""
    objects = {
        "ground": (0, 8),  # label=terrain (contains "ground")
        "glass_building": (8, 16),  # label=buildings
    }
    mock_load.side_effect = _make_load_side_effect(objects)
    mock_get_faces.return_value = [_MOCK_FACES[0]]
    scene = sionna_scene.read_scene("/dummy", [0, 1], deduplicate=True)
    assert len(scene.objects) == 2
    labels = {obj.label for obj in scene.objects}
    assert "terrain" in labels
    assert "buildings" in labels


@patch("deepmimo.converters.sionna_rt.sionna_scene.load_pickle")
@patch("deepmimo.converters.sionna_rt.sionna_scene.get_object_faces")
def test_read_scene_non_overlapping_buildings_stay_separate(mock_get_faces, mock_load) -> None:
    """Two separate buildings (no AABB overlap) stay as two objects."""
    # Build non-overlapping vertex sets
    verts_a = np.array(
        [[0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0], [0, 0, 10], [5, 5, 10]], dtype=float
    )
    verts_b = np.array(
        [[20, 0, 0], [25, 0, 0], [25, 5, 0], [20, 5, 0], [20, 0, 10], [25, 5, 10]], dtype=float
    )
    combined = np.vstack([verts_a, verts_b])

    def side_effect(path):
        if "sionna_vertices.pkl" in path:
            return combined
        if "sionna_objects.pkl" in path:
            return {"building_a": (0, 6), "building_b": (6, 12)}
        raise FileNotFoundError(path)

    mock_load.side_effect = side_effect
    mock_get_faces.return_value = [_MOCK_FACES[0]]
    scene = sionna_scene.read_scene("/dummy", [0, 1], deduplicate=True)
    assert len(scene.objects) == 2


@patch("deepmimo.converters.sionna_rt.sionna_scene.load_pickle")
@patch("deepmimo.converters.sionna_rt.sionna_scene.get_object_faces")
def test_read_scene_deduplicate_uses_first_component_material(mock_get_faces, mock_load) -> None:
    """The merged object inherits the material_idx of the first component (index 0)."""
    mock_load.side_effect = _make_load_side_effect(_MOCK_OBJECTS_MULTI_MATERIAL)
    mock_get_faces.return_value = [_MOCK_FACES[0]]
    scene = sionna_scene.read_scene("/dummy", [3, 7], deduplicate=True)
    assert len(scene.objects) == 1
    # material_idx of first component (concrete_building) is 3
    assert scene.objects[0].faces[0].material_idx == 3


@patch("deepmimo.converters.sionna_rt.sionna_scene.load_pickle")
@patch("deepmimo.converters.sionna_rt.sionna_scene.get_object_faces")
def test_read_scene_overlap_threshold_controls_merging(mock_get_faces, mock_load) -> None:
    """Lower threshold merges; higher threshold keeps separate."""
    # Large: [0-10, 0-10, 0-10]; Partial: [0-10, 0-10, -1-8.5]
    # Intersection: [0-10, 0-10, 0-8.5] vol=850; partial vol=950; ratio≈0.895
    partial_verts = np.array(
        [
            [0, 0, -1],
            [10, 0, -1],
            [10, 10, -1],
            [0, 10, -1],
            [0, 0, 8.5],
            [10, 0, 8.5],
            [10, 10, 8.5],
            [0, 10, 8.5],
        ],
        dtype=float,
    )
    combined = np.vstack([_LARGE_VERTS, partial_verts])

    def side_effect(path):
        if "sionna_vertices.pkl" in path:
            return combined
        if "sionna_objects.pkl" in path:
            return {"concrete_building": (0, 8), "partial_material": (8, 16)}
        raise FileNotFoundError(path)

    mock_load.side_effect = side_effect
    mock_get_faces.return_value = [_MOCK_FACES[0]]

    # threshold=0.8 → merges (ratio≈0.895 >= 0.8)
    scene_merged = sionna_scene.read_scene(
        "/dummy", [0, 1], deduplicate=True, overlap_threshold=0.8
    )
    assert len(scene_merged.objects) == 1

    # threshold=0.95 → does not merge (ratio≈0.895 < 0.95)
    mock_load.side_effect = side_effect  # reset
    scene_split = sionna_scene.read_scene(
        "/dummy", [0, 1], deduplicate=True, overlap_threshold=0.95
    )
    assert len(scene_split.objects) == 2


# ---------------------------------------------------------------------------
# Lossless conversion
# ---------------------------------------------------------------------------

# A vertical wall panel plus a horizontal floor, as two Sionna objects. The wall
# is the interesting one: its 2D footprint is a line, which the convex-hull path
# rejects as collinear.
_WALL_TRIS = np.array([[0, 1, 2], [0, 2, 3]])
_LOSSLESS_VERTICES = np.array(
    [
        # wall at x = 0, spanning y and z
        [0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 4.0, 3.0],
        [0.0, 0.0, 3.0],
        # floor slab
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [4.0, 4.0, 0.0],
        [0.0, 4.0, 0.0],
    ],
)
_LOSSLESS_OBJECTS = {"wall_panel": (0, 4), "ground_plane": (4, 8)}
_LOSSLESS_FACES = np.vstack([_WALL_TRIS, _WALL_TRIS + 4])


def _patched_pickles(monkeypatch) -> None:
    """Serve a small synthetic Sionna export to read_scene."""
    payloads = {
        "sionna_vertices.pkl": _LOSSLESS_VERTICES,
        "sionna_objects.pkl": _LOSSLESS_OBJECTS,
        "sionna_faces.pkl": _LOSSLESS_FACES,
    }
    monkeypatch.setattr(
        sionna_scene,
        "load_pickle",
        lambda path: payloads[path.rsplit("/", 1)[-1]],
    )


def test_lossless_keeps_every_declared_triangle(monkeypatch) -> None:
    """Lossless conversion emits one face per declared triangle."""
    _patched_pickles(monkeypatch)

    scene = sionna_scene.read_scene("ignored", [0, 1], lossless=True)

    assert len(scene.objects) == len(_LOSSLESS_OBJECTS)
    assert sum(len(obj.faces) for obj in scene.objects) == len(_LOSSLESS_FACES)


def test_lossless_keeps_vertical_walls(monkeypatch) -> None:
    """A wall the hull path discards as collinear survives losslessly.

    This is the defect that left converted indoor scenes with a floor, a ceiling
    and no walls at all.
    """
    _patched_pickles(monkeypatch)

    scene = sionna_scene.read_scene("ignored", [0, 1], lossless=True)
    wall = next(obj for obj in scene.objects if obj.name == "wall_panel")
    vertices = np.concatenate([face.vertices for face in wall.faces])

    assert len(wall.faces) == len(_WALL_TRIS)
    assert np.ptp(vertices[:, 0]) == 0.0  # perfectly vertical
    assert np.ptp(vertices[:, 2]) == 3.0  # full height retained


def test_lossless_preserves_object_names_and_labels(monkeypatch) -> None:
    """Object identity survives, and ground surfaces are labelled as terrain."""
    _patched_pickles(monkeypatch)

    scene = sionna_scene.read_scene("ignored", [0, 1], lossless=True)
    labels = {obj.name: obj.label for obj in scene.objects}

    assert labels == {"wall_panel": CAT_BUILDINGS, "ground_plane": CAT_TERRAIN}


def test_lossless_does_not_run_the_component_split(monkeypatch) -> None:
    """The connected-component pass is skipped, which is what makes it fast.

    That pass exists to reassemble a merged mesh before hulling it; with no hull
    it is pure cost, and it dominates conversion time on a detailed interior.
    """
    _patched_pickles(monkeypatch)
    calls = []

    def _record(*args) -> list:
        calls.append(args)
        return []

    monkeypatch.setattr(sionna_scene, "_split_into_component_meshes", _record)

    sionna_scene.read_scene("ignored", [0, 1], lossless=True)

    assert calls == []


def test_lossless_assigns_material_per_object(monkeypatch) -> None:
    """Each object's faces carry that object's material index."""
    _patched_pickles(monkeypatch)

    scene = sionna_scene.read_scene("ignored", [0, 1], lossless=True)
    by_name = {obj.name: obj for obj in scene.objects}

    assert {f.material_idx for f in by_name["wall_panel"].faces} == {0}
    assert {f.material_idx for f in by_name["ground_plane"].faces} == {1}

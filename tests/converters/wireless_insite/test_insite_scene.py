"""Tests for Wireless Insite Scene conversion."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deepmimo.converters.wireless_insite import insite_scene
from deepmimo.core.scene import CAT_BUILDINGS, CAT_OBJECTS, get_object_faces


def test_extract_objects() -> None:
    """Extract connected faces into physical objects."""
    content = """
    begin_<face>
      0.0 0.0 0.0
      1.0 0.0 0.0
      0.0 1.0 0.0
    end_<face>
    begin_<face>
      0.0 0.0 0.0
      0.0 1.0 0.0
      0.0 0.0 1.0
    end_<face>
    """
    objects = insite_scene.extract_objects(content)
    assert len(objects) == 1  # Connected faces should form 1 object
    assert len(objects[0]) == 4  # 4 unique vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1)


@patch("pathlib.Path.open")
@patch("deepmimo.converters.wireless_insite.insite_scene.get_object_faces")
def test_physical_object_parser(mock_get_faces, mock_path_open) -> None:
    """Parse a single CITY file into physical objects."""
    mock_file = MagicMock()
    mock_file.read.return_value = "dummy content"
    mock_path_open.return_value.__enter__.return_value = mock_file

    # Mock extract_objects to return dummy vertices
    with patch(
        "deepmimo.converters.wireless_insite.insite_scene.extract_objects",
        return_value=[[[0, 0, 0], [1, 0, 0], [0, 1, 0]]],
    ):
        mock_get_faces.return_value = [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]]  # Single face

        parser = insite_scene.PhysicalObjectParser("test.city")
        objects = parser.parse()

        assert len(objects) == 1
        assert objects[0].label == CAT_BUILDINGS


@patch("deepmimo.converters.wireless_insite.insite_scene.PhysicalObjectParser")
def test_read_scene(mock_parser_cls, tmp_path) -> None:
    """Read a scene directory and aggregate parsed objects."""
    # Create dummy files
    (tmp_path / "test.city").touch()

    mock_parser = MagicMock()
    mock_obj = MagicMock()
    mock_parser.parse.return_value = [mock_obj]
    mock_parser_cls.return_value = mock_parser

    scene = insite_scene.read_scene(tmp_path)
    assert len(scene.objects) == 1


# --- Indoor geometry preservation (issue #126) -------------------------------

WALL_AND_DESK_FILE = """
begin_<structure>
begin_<sub_structure>
begin_<face>
double_sided
Material 3
nVertices 4
-15.9704995849 19.8690896930 0.0000000000
-15.9704995849 16.4511458519 0.0000000000
-15.9704995849 16.4511458519 5.0000000000
-15.9704995849 19.8690896930 5.0000000000
end_<face>
end_<sub_structure>
end_<structure>
begin_<structure>
begin_<sub_structure>
begin_<face>
Material 0
nVertices 3
-19.1732536776 16.7591643727 1.4428974628
-19.1732536776 17.0385361111 1.2199573755
-19.1732536776 17.0385361111 1.1795883417
end_<face>
end_<sub_structure>
end_<structure>
"""


def test_object_extension_is_recognized() -> None:
    """Wireless InSite writes standalone geometry as .object, not .obj."""
    assert insite_scene.OBJECT_LABELS[".object"] == CAT_OBJECTS


def test_extract_object_faces_groups_disconnected_geometry() -> None:
    """Faces that share no vertices become separate objects, keeping their faces."""
    objects = insite_scene.extract_object_faces(WALL_AND_DESK_FILE)

    assert len(objects) == 2
    assert [len(faces) for faces in objects] == [1, 1]
    # (material index, vertex count) of the single face in each object
    assert [(faces[0][0], len(faces[0][1])) for faces in objects] == [(3, 4), (0, 3)]


def test_extract_object_faces_keeps_vertical_walls() -> None:
    """A vertical wall projects to a line in xy and must not be dropped.

    The convex-hull path builds objects from their 2D footprint, so a flat wall
    is discarded as "collinear". Preserving the declared faces keeps it.
    """
    wall_vertices = insite_scene.extract_object_faces(WALL_AND_DESK_FILE)[0][0][1]
    xs = {round(x, 6) for x, _, _ in wall_vertices}

    assert len(xs) == 1  # perfectly vertical
    assert get_object_faces(np.array(wall_vertices)) is None  # hull path loses it


def test_parse_lossless_preserves_geometry(tmp_path) -> None:
    """Lossless parsing keeps every declared face and maps materials."""
    file = tmp_path / "room.flp"
    file.write_text(WALL_AND_DESK_FILE)

    parser = insite_scene.PhysicalObjectParser(
        str(file),
        lossless=True,
        material_map={0: 7, 3: 4},
    )
    objects = parser.parse()

    assert len(objects) == 2
    assert [len(obj.faces) for obj in objects] == [1, 1]
    assert [obj.faces[0].material_idx for obj in objects] == [4, 7]
    # exact coordinates, not a hull approximation
    assert objects[0].faces[0].vertices[0] == pytest.approx(
        [-15.9704995849, 19.8690896930, 0.0],
        abs=1e-4,
    )


def test_parse_does_not_accumulate_object_names(tmp_path) -> None:
    """Object names are derived per object, not appended to the previous one."""
    file = tmp_path / "room.flp"
    file.write_text(WALL_AND_DESK_FILE)

    objects = insite_scene.PhysicalObjectParser(str(file), lossless=True).parse()

    assert [obj.name for obj in objects] == ["room_0", "room_1"]


def test_extract_object_faces_separates_touching_free_objects() -> None:
    """Two solids that share no vertices stay separate, as in a .city file.

    Outdoor files declare every building inside a single <structure> block, so
    grouping by structure would collapse a whole city into one object.
    """
    two_triangles = """
    begin_<face>
    Material 0
    nVertices 3
    0.0000000000 0.0000000000 0.0000000000
    1.0000000000 0.0000000000 0.0000000000
    0.0000000000 1.0000000000 0.0000000000
    end_<face>
    begin_<face>
    Material 0
    nVertices 3
    50.0000000000 50.0000000000 0.0000000000
    51.0000000000 50.0000000000 0.0000000000
    50.0000000000 51.0000000000 0.0000000000
    end_<face>
    """
    assert len(insite_scene.extract_object_faces(two_triangles)) == 2

"""Tests for Wireless Insite Scene conversion."""

from unittest.mock import MagicMock, patch

from deepmimo.converters.wireless_insite import insite_scene
from deepmimo.scene import CAT_BUILDINGS


def test_extract_objects():
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


@patch("builtins.open")
@patch("deepmimo.converters.wireless_insite.insite_scene.get_object_faces")
def test_physical_object_parser(mock_get_faces, mock_open):
    mock_file = MagicMock()
    mock_file.read.return_value = "dummy content"
    mock_open.return_value.__enter__.return_value = mock_file

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
def test_read_scene(mock_parser_cls, tmp_path):
    # Create dummy files
    (tmp_path / "test.city").touch()

    mock_parser = MagicMock()
    mock_obj = MagicMock()
    mock_parser.parse.return_value = [mock_obj]
    mock_parser_cls.return_value = mock_parser

    scene = insite_scene.read_scene(tmp_path)
    assert len(scene.objects) == 1

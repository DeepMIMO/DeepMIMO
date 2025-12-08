"""Tests for DeepMIMO Converter module."""

from unittest.mock import patch, MagicMock
from deepmimo.converters import converter


def test_find_converter_from_dir(tmp_path):
    # Create dummy files
    (tmp_path / "test.parquet").touch()
    conv = converter._find_converter_from_dir(str(tmp_path))
    assert conv.__name__ == "aodt_rt_converter"

    (tmp_path / "test.parquet").unlink()
    (tmp_path / "test.pkl").touch()
    conv = converter._find_converter_from_dir(str(tmp_path))
    assert conv.__name__ == "sionna_rt_converter"

    (tmp_path / "test.pkl").unlink()
    (tmp_path / "test.setup").touch()
    conv = converter._find_converter_from_dir(str(tmp_path))
    assert conv.__name__ == "insite_rt_converter"

    (tmp_path / "test.setup").unlink()
    conv = converter._find_converter_from_dir(str(tmp_path))
    assert conv is None


@patch("deepmimo.converters.converter._find_converter_from_dir")
def test_convert_root(mock_find, tmp_path):
    mock_converter = MagicMock()
    mock_find.return_value = mock_converter

    converter.convert(str(tmp_path))

    mock_find.assert_called_with(str(tmp_path))
    mock_converter.assert_called_with(str(tmp_path))


@patch("deepmimo.converters.converter._find_converter_from_dir")
def test_convert_subdir(mock_find, tmp_path):
    # Mock finding only in subdir
    def side_effect(path):
        if path == str(tmp_path):
            return None
        return MagicMock()

    mock_find.side_effect = side_effect

    subdir = tmp_path / "scene1"
    subdir.mkdir()

    res = converter.convert(str(tmp_path))

    # Should have called find on subdir
    mock_find.assert_any_call(str(subdir))

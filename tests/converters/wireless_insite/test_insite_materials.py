"""Tests for Wireless Insite Materials."""

from unittest.mock import MagicMock, patch

from deepmimo.converters.wireless_insite import insite_materials
from deepmimo.core.materials import Material


def test_insite_material_conversion() -> None:
    """Convert a basic InSite material into DeepMIMO material fields."""
    imat = insite_materials.InsiteMaterial(
        id=1,
        name="Concrete",
        conductivity=2.0,
        permittivity=4.0,
        roughness=0.1,
        thickness=0.2,
        diffuse_scattering_model="lambertian",
        fields_diffusively_scattered=0.5,
    )
    mat = imat.to_material()
    assert mat.name == "Concrete"
    assert mat.conductivity == 2.0
    assert mat.permittivity == 4.0
    assert mat.scattering_model == Material.SCATTERING_LAMBERTIAN
    assert mat.scattering_coefficient == 0.5


def test_insite_foliage_conversion() -> None:
    """Convert an InSite foliage entry to material with scattering defaults."""
    ifol = insite_materials.InsiteFoliage(
        id=2, name="Tree", thickness=5.0, vertical_attenuation=1.0, permittivity_vr=1.2
    )
    mat = ifol.to_material()
    assert mat.name == "Tree"
    assert mat.thickness == 5.0
    assert mat.vertical_attenuation == 1.0
    # Foliage uses SCATTERING_NONE in to_material
    assert mat.scattering_model == Material.SCATTERING_NONE


@patch("deepmimo.converters.wireless_insite.insite_materials.parse_file")
def test_read_materials(mock_parse) -> None:
    """Read and normalize materials from mocked Wireless Insite files."""
    # Mock document structure
    layer = MagicMock()
    layer.values = {
        "conductivity": 1.0,
        "permittivity": 2.0,
        "roughness": 0.0,
        "thickness": 0.1,
    }
    mat_node = MagicMock()
    mat_node.name = "Mat1"
    mat_node.labels = []
    mat_node.values = {"diffuse_scattering_model": "lambertian", "DielectricLayer": layer}
    mat_node.all_values = {"DielectricLayer": [layer]}

    root_node = MagicMock()
    # Node keeps the last occurrence in `values` and every occurrence in `all_values`
    root_node.values = {"Material": mat_node}
    root_node.all_values = {"Material": [mat_node]}

    mock_parse.return_value = {"Primitive": root_node}

    with patch("pathlib.Path.glob") as mock_glob, patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = True
        mock_glob.return_value = [MagicMock()]  # Found 1 file

        mats = insite_materials.read_materials("dummy_dir")
        # MaterialList.to_dict() returns keys "material_{id}"
        # Since we have 1 material and IDs are assigned from 0
        assert len(mats) == 1
        assert mats["material_0"]["permittivity"] == 2.0

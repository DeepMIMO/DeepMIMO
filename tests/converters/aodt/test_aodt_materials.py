"""Tests for AODT material handling."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from deepmimo.converters.aodt import aodt_materials
from deepmimo.core.materials import Material


def test_aodt_material_conversion() -> None:
    """Convert AODTMaterial to DeepMIMO Material."""
    aodt_mat = aodt_materials.AODTMaterial(
        id=0,
        label="Brick",
        itu_r_p2040_a=1.0,
        itu_r_p2040_b=0.0,
        itu_r_p2040_c=0.0,
        itu_r_p2040_d=0.0,
        scattering_coeff=0.5,
        scattering_xpd=0.1,
        rms_roughness=0.01,
        exponent_alpha_r=4.0,
        exponent_alpha_i=4.0,
        lambda_r=0.5,
        thickness_m=0.2,
    )

    dm_mat = aodt_mat.to_material(freq_ghz=28.0)
    assert isinstance(dm_mat, Material)
    assert dm_mat.permittivity == 1.0  # a + b*f^c, b=0 -> a=1.0
    assert dm_mat.scattering_coefficient == 0.5
    assert dm_mat.thickness == 0.2


@patch("deepmimo.converters.aodt.aodt_materials.pd")
@patch.object(Path, "exists")
def test_read_materials(mock_exists, mock_pd) -> None:
    """Read AODT material parquet and convert rows to dict."""
    mock_exists.return_value = True

    # Mock DataFrame
    mock_df = MagicMock()
    mock_df.__len__.return_value = 1

    # Mock row
    row_data = {
        "label": "Brick",
        "itu_r_p2040_a": 1.0,
        "itu_r_p2040_b": 0.0,
        "itu_r_p2040_c": 0.0,
        "itu_r_p2040_d": 0.0,
        "scattering_coeff": 0.5,
        "scattering_xpd": 0.1,
        "rms_roughness": 0.01,
        "exponent_alpha_r": 4.0,
        "exponent_alpha_i": 4.0,
        "lambda_r": 0.5,
        "thickness_m": 0.2,
    }

    # Iterate yields index and series-like object
    series_mock = MagicMock()
    series_mock.to_dict.return_value = row_data
    series_mock.__getitem__.side_effect = row_data.get

    mock_df.iterrows.return_value = [(0, series_mock)]
    mock_pd.read_parquet.return_value = mock_df

    materials_dict = aodt_materials.read_materials("/dummy/path")

    assert "material_0" in materials_dict
    assert materials_dict["material_0"]["name"] == "Brick"

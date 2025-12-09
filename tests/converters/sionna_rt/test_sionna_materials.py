"""Tests for Sionna Materials."""

from unittest.mock import patch

from deepmimo.converters.sionna_rt import sionna_materials


@patch("deepmimo.converters.sionna_rt.sionna_materials.load_pickle")
def test_read_materials(mock_load) -> None:
    """Read Sionna material pickles and map to DeepMIMO structures."""
    # Mock materials
    mock_materials = [
        {
            "scattering_pattern": "LambertianPattern",
            "scattering_coefficient": 0.5,
            "relative_permittivity": 5.0,
            "conductivity": 0.1,
            "xpd_coefficient": 0.0,
            "alpha_r": 4.0,
            "alpha_i": 4.0,
            "lambda_": 0.5,
        }
    ]
    mock_indices = {"obj1": 0}

    def side_effect(path):
        if "sionna_materials.pkl" in path:
            return mock_materials
        if "sionna_material_indices.pkl" in path:
            return mock_indices
        return None

    mock_load.side_effect = side_effect

    mats, idxs = sionna_materials.read_materials("/dummy/path")
    assert len(mats) == 1
    assert "material_0" in mats
    assert mats["material_0"]["permittivity"] == 5.0
    assert idxs["obj1"] == 0

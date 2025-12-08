"""Tests for Wireless Insite RT Parameters."""

from unittest.mock import MagicMock, patch

import numpy as np

from deepmimo.consts import RAYTRACER_NAME_WIRELESS_INSITE
from deepmimo.converters.wireless_insite import insite_rt_params


def test_get_gps_bbox():
    # Test simple case
    origin_lat = 40.0
    origin_lon = -74.0
    vertices = np.array([[0, 0, 0], [100, 100, 0]])

    bbox = insite_rt_params._get_gps_bbox(origin_lat, origin_lon, vertices)

    assert len(bbox) == 4
    # Check bounds are somewhat reasonable (near origin)
    assert np.isclose(bbox[0], 40.0, atol=0.01)  # min_lat
    assert np.isclose(bbox[1], -74.0, atol=0.01)  # min_lon

    # Test default case (0,0 origin)
    bbox_zero = insite_rt_params._get_gps_bbox(0, 0, vertices)
    assert bbox_zero == (0, 0, 0, 0)


@patch("deepmimo.converters.wireless_insite.insite_rt_params.parse_file")
def test_read_rt_params(mock_parse, tmp_path):
    # Mock .setup file parsing
    mock_setup_folder = tmp_path / "sim_folder"
    mock_setup_folder.mkdir()
    (mock_setup_folder / "sim.setup").touch()

    # Mock document structure
    # Structure: Node -> values -> Node

    model_node = MagicMock()
    model_node.values = {
        "ray_spacing": 0.25,
        "max_reflections": 3,
        "max_transmissions": 0,
        "terrain_diffractions": "No",
    }

    apg_node = MagicMock()
    apg_node.values = {"path_depth": 5}

    diffuse_node = MagicMock()
    diffuse_node.values = {
        "enabled": True,
        "diffuse_reflections": 1,
        "diffuse_diffractions": 0,
        "diffuse_transmissions": 0,
        "final_interaction_only": False,
    }

    ref_node = MagicMock()
    ref_node.values = {"latitude": 40.0, "longitude": -74.0}

    boundary_node = MagicMock()
    boundary_node.values = {"reference": ref_node}
    boundary_node.data = [[0, 0, 0], [10, 10, 0]]

    study_node = MagicMock()
    study_node.values = {
        "model": model_node,
        "apg_acceleration": apg_node,
        "diffuse_scattering": diffuse_node,
        "boundary": boundary_node,
    }

    waveform_node = MagicMock()
    waveform_node.values = {"CarrierFrequency": 28e9}

    antenna_node = MagicMock()
    antenna_node.values = {}

    prim_node = MagicMock()
    prim_node.values = {"studyarea": study_node, "Waveform": waveform_node, "antenna": antenna_node}

    mock_doc = {"Prim": prim_node}
    mock_parse.return_value = mock_doc

    params = insite_rt_params.InsiteRayTracingParameters.read_rt_params(mock_setup_folder)

    assert params.raytracer_name == RAYTRACER_NAME_WIRELESS_INSITE
    assert params.frequency == 28e9
    assert params.max_reflections == 3
    assert params.max_scattering == 1
    assert params.num_rays > 0  # calculated from ray_spacing

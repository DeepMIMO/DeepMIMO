"""Tests for MATLAB RT JSON parser and schema validation."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from deepmimo.converters.matlab_rt.errors import (
    MatlabRTSchemaError,
    MatlabRTValidationError,
    UnsupportedMatlabRTFeatureError,
)
from deepmimo.converters.matlab_rt.parser import parse_matlab_rt_export, parse_matlab_rt_json
from deepmimo.converters.matlab_rt.schema import (
    MatlabRTExport,
    MatlabRTInteraction,
    MatlabRTLink,
    MatlabRTRay,
)
from tests.converters.matlab_rt import expect_raises


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Load one MATLAB RT JSON fixture as a mutable dictionary."""
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


class TestMatlabRTParser(unittest.TestCase):
    """Validate typed parsing and MVP schema checks."""

    def test_los_fixture_parses(self) -> None:
        """The one-link LoS fixture normalizes into one typed link."""
        export = parse_matlab_rt_json(FIXTURE_DIR / "matlab_rt_los.json")

        assert isinstance(export, MatlabRTExport)
        assert export.num_tx == 1
        assert export.num_rx == 1
        assert len(export.links) == 1
        assert export.scene.coordinate_system == "cartesian"

        link = export.links[0]
        assert isinstance(link, MatlabRTLink)
        assert (link.tx_index, link.rx_index) == (1, 1)
        assert link.num_rays == 1

        ray = link.rays[0]
        assert isinstance(ray, MatlabRTRay)
        assert ray.line_of_sight
        assert ray.num_interactions == 0
        assert ray.angle_of_departure_deg == (0.0, -2.8624052261117479)
        assert ray.angle_of_arrival_deg == (180.0, 2.8624052261117479)

    def test_nlos_fixture_parses(self) -> None:
        """The one-link NLoS fixture preserves reflection metadata."""
        export = parse_matlab_rt_json(FIXTURE_DIR / "matlab_rt_nlos_reflection.json")

        assert export.num_tx == 1
        assert export.num_rx == 1
        assert export.links[0].num_rays == 2

        reflected_ray = export.links[0].rays[1]
        assert not reflected_ray.line_of_sight
        assert reflected_ray.num_interactions == 1
        assert isinstance(reflected_ray.interactions[0], MatlabRTInteraction)
        assert reflected_ray.interactions[0].type == "Reflection"
        assert reflected_ray.interactions[0].location_m == (5.0, 5.0, 1.0)
        assert reflected_ray.interactions[0].material_name == "PEC"

    def test_multilink_fixture_parses(self) -> None:
        """The multi-link fixture parses with deterministic TX-major coverage."""
        export = parse_matlab_rt_json(FIXTURE_DIR / "matlab_rt_multilink.json")

        assert export.num_tx == 2
        assert export.num_rx == 2
        assert [tx.index for tx in export.transmitters] == [1, 2]
        assert [rx.index for rx in export.receivers] == [1, 2]
        assert [(link.tx_index, link.rx_index) for link in export.links] == [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
        ]
        assert [link.num_rays for link in export.links] == [2, 0, 0, 2]
        assert export.links[1].rays == ()
        assert export.links[2].rays == ()

    def test_link_order_is_deterministic(self) -> None:
        """Parser returns deterministic link order independent of JSON order."""
        data = load_fixture("matlab_rt_multilink.json")
        data["links"] = list(reversed(data["links"]))

        export = parse_matlab_rt_export(data)

        assert [(link.tx_index, link.rx_index) for link in export.links] == [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
        ]

    def test_missing_links_fails_cleanly_for_multilink_schema(self) -> None:
        """A multi-link export without links raises a typed schema error."""
        data = load_fixture("matlab_rt_multilink.json")
        data.pop("links")

        with expect_raises(MatlabRTSchemaError):
            parse_matlab_rt_export(data)

    def test_incomplete_link_coverage_fails_cleanly(self) -> None:
        """A multi-link export must include every TX/RX link pair explicitly."""
        data = load_fixture("matlab_rt_multilink.json")
        data["links"] = data["links"][:-1]

        with expect_raises(MatlabRTValidationError):
            parse_matlab_rt_export(data)

    def test_num_rays_mismatch_fails_cleanly(self) -> None:
        """Declared ray counts must match the parsed ray list length."""
        data = load_fixture("matlab_rt_multilink.json")
        data["links"][0]["num_rays"] = 99

        with expect_raises(MatlabRTValidationError):
            parse_matlab_rt_export(data)

    def test_single_link_num_rays_mismatch_fails_cleanly(self) -> None:
        """Single-link exports also enforce top-level ray counts."""
        data = load_fixture("matlab_rt_los.json")
        data["num_rays"] = 2

        with expect_raises(MatlabRTValidationError):
            parse_matlab_rt_export(data)

    def test_unsupported_coordinate_system_fails_cleanly(self) -> None:
        """Non-Cartesian coordinates are outside the MATLAB RT MVP."""
        data = load_fixture("matlab_rt_multilink.json")
        data["scene"]["coordinate_system"] = "geographic"

        with expect_raises(UnsupportedMatlabRTFeatureError):
            parse_matlab_rt_export(data)

    def test_invalid_interaction_type_fails_cleanly(self) -> None:
        """Unsupported MATLAB interaction types raise the typed feature error."""
        data = load_fixture("matlab_rt_nlos_reflection.json")
        data["rays"][1]["interactions"][0]["Type"] = "Diffraction"

        with expect_raises(UnsupportedMatlabRTFeatureError):
            parse_matlab_rt_export(data)

    def test_empty_material_name_normalizes_to_none(self) -> None:
        """Empty MATLAB material names are treated as unknown optional metadata."""
        data = load_fixture("matlab_rt_nlos_reflection.json")
        data["rays"][1]["interactions"][0]["MaterialName"] = "   "

        export = parse_matlab_rt_export(data)

        assert export.links[0].rays[1].interactions[0].material_name is None

    def test_inconsistent_link_indices_fail_cleanly(self) -> None:
        """Link indices must refer to known TX/RX site indices."""
        data = load_fixture("matlab_rt_multilink.json")
        data["links"][0]["tx_index"] = 99

        with expect_raises(MatlabRTValidationError):
            parse_matlab_rt_export(data)

    def test_mutating_input_after_parse_does_not_change_raw_snapshot(self) -> None:
        """Parsed objects hold a defensive raw snapshot."""
        data = load_fixture("matlab_rt_los.json")
        parsed = parse_matlab_rt_export(data)
        original_name = parsed.transmitters[0].raw["name"]

        mutated = deepcopy(data)
        mutated["transmitter"]["name"] = "changed"

        assert original_name == "tx1"


if __name__ == "__main__":
    unittest.main()

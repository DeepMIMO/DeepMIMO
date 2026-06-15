"""Tests for MATLAB RT in-memory NumPy matrix assembly."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from deepmimo.converters.matlab_rt.errors import MatlabRTValidationError
from deepmimo.converters.matlab_rt.matrices import (
    MATRIX_FIELDS,
    SCALAR_MATRIX_FIELDS,
    assemble_all_tx_matrices,
    assemble_tx_matrices,
)
from deepmimo.converters.matlab_rt.parser import parse_matlab_rt_json
from deepmimo.converters.matlab_rt.paths import build_path_row_groups


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def load_matrix_sets() -> tuple:
    """Assemble all TX matrices from the multi-link fixture."""
    export = parse_matlab_rt_json(FIXTURE_DIR / "matlab_rt_multilink.json")
    return assemble_all_tx_matrices(build_path_row_groups(export))


class TestMatlabRTMatrixAssembly(unittest.TestCase):
    """Validate pure in-memory DeepMIMO-style matrix assembly."""

    def test_shapes_and_tx_separation(self) -> None:
        """Multi-link fixture creates one matrix set per TX with DeepMIMO shapes."""
        tx1_set, tx2_set = load_matrix_sets()

        for matrix_set in (tx1_set, tx2_set):
            self.assertEqual(set(matrix_set.matrices), set(MATRIX_FIELDS))
            self.assertEqual(matrix_set.scalar_shape, (2, 2))
            self.assertEqual(matrix_set.inter_pos_shape, (2, 2, 1, 3))
            self.assertEqual(matrix_set.rx_pos_shape, (2, 3))
            self.assertEqual(matrix_set.tx_pos_shape, (1, 3))
            for field in SCALAR_MATRIX_FIELDS:
                self.assertEqual(matrix_set.matrices[field].shape, (2, 2))

        np.testing.assert_allclose(tx1_set.matrices["tx_pos"], np.array([[0.0, 0.0, 1.0]]))
        np.testing.assert_allclose(tx2_set.matrices["tx_pos"], np.array([[0.0, 10.0, 1.0]]))
        self.assertEqual(tx1_set.path_counts, (2, 0))
        self.assertEqual(tx2_set.path_counts, (0, 2))

    def test_values_padding_and_interactions(self) -> None:
        """Power, angles, interaction codes, and NaN padding match the fixture."""
        tx1_set, tx2_set = load_matrix_sets()

        self.assertAlmostEqual(tx1_set.matrices["power"][0, 0], -63.32914410888889)
        self.assertAlmostEqual(tx1_set.matrices["power"][0, 1], -66.33944406441894)
        self.assertEqual(tx1_set.matrices["aoa_az"][0, 0], 180.0)
        self.assertEqual(tx1_set.matrices["aoa_el"][0, 0], 90.0)
        self.assertAlmostEqual(tx1_set.matrices["aod_az"][0, 1], 44.99999998535911)
        self.assertEqual(tx1_set.matrices["inter"][0, 0], 0.0)
        self.assertEqual(tx1_set.matrices["inter"][0, 1], 1.0)
        np.testing.assert_allclose(
            tx1_set.matrices["inter_pos"][0, 1, 0, :],
            np.array([5.000000001277659, 4.999999998722342, 1.0]),
        )

        self.assertTrue(np.isnan(tx1_set.matrices["power"][1]).all())
        self.assertTrue(np.isnan(tx1_set.matrices["inter_pos"][1]).all())
        self.assertTrue(np.isnan(tx2_set.matrices["power"][0]).all())
        self.assertAlmostEqual(tx2_set.matrices["aod_az"][1, 1], -44.99999998535911)

    def test_malformed_groups_fail_cleanly(self) -> None:
        """Matrix assembly rejects inconsistent path-row groups."""
        export = parse_matlab_rt_json(FIXTURE_DIR / "matlab_rt_multilink.json")
        group = build_path_row_groups(export)[0]

        with self.assertRaises(MatlabRTValidationError):
            assemble_tx_matrices(replace(group, links=group.links[:-1]))

        bad_link = replace(group.links[0], rows=group.links[0].rows[:-1])
        with self.assertRaises(MatlabRTValidationError):
            assemble_tx_matrices(replace(group, links=(bad_link,) + group.links[1:]))


if __name__ == "__main__":
    unittest.main()

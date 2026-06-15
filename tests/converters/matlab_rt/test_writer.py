"""Tests for MATLAB RT filesystem scenario writer."""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import numpy as np

from deepmimo import consts as c
from deepmimo.converters.matlab_rt.errors import MatlabRTWriterError
from deepmimo.converters.matlab_rt.matrices import MATRIX_FIELDS, assemble_all_tx_matrices
from deepmimo.converters.matlab_rt.metadata import build_params
from deepmimo.converters.matlab_rt.parser import parse_matlab_rt_json
from deepmimo.converters.matlab_rt.paths import build_path_row_groups
from deepmimo.converters.matlab_rt.writer import expected_matrix_filenames, write_scenario_folder
from tests.converters.matlab_rt import expect_raises


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
SCENARIO_NAME = "matlab_rt_writer_test"


def build_writer_inputs() -> tuple[dict, tuple]:
    """Build deterministic params and matrix sets from the multi-link fixture."""
    export = parse_matlab_rt_json(FIXTURE_DIR / "matlab_rt_multilink.json")
    matrix_sets = assemble_all_tx_matrices(build_path_row_groups(export))
    params = build_params(export, matrix_sets, scenario_name=SCENARIO_NAME)
    return params, matrix_sets


@contextmanager
def writer_tempdir() -> Iterator[str]:
    """Create a temporary directory inside the writable workspace."""
    temp_root = Path.cwd() / f".matlab_rt_writer_test_tmp_{uuid.uuid4().hex}"
    temp_root.mkdir()
    try:
        yield str(temp_root)
    finally:
        shutil.rmtree(temp_root)


class TestMatlabRTWriter(unittest.TestCase):
    """Validate writing DeepMIMO scenario files without orchestration."""

    def test_writes_required_files_and_arrays(self) -> None:
        """Writer emits params, scene placeholders, and all expected matrix files."""
        params, matrix_sets = build_writer_inputs()

        with writer_tempdir() as tmpdir:
            result = write_scenario_folder(
                scenario_root=tmpdir,
                scenario_name=SCENARIO_NAME,
                params=params,
                matrix_sets=matrix_sets,
            )

            expected_files = {
                "params.json",
                "objects.json",
                "vertices.npz",
                *expected_matrix_filenames(matrix_sets),
            }
            assert {path.name for path in result.scenario_path.iterdir()} == expected_files
            assert "power_t000_tx000_r001.npz" in expected_files
            assert "tx_pos_t000_tx001_r001.npz" in expected_files

            with (result.scenario_path / "params.json").open("r", encoding="utf-8") as file:
                saved_params = json.load(file)
            with (result.scenario_path / "objects.json").open("r", encoding="utf-8") as file:
                assert json.load(file) == []
            with np.load(result.scenario_path / "vertices.npz") as vertices_npz:
                assert vertices_npz["vertices"].shape == (0, 3)

            assert saved_params[c.RT_PARAMS_PARAM_NAME][c.RT_PARAM_FREQUENCY] == 3.5e9
            with np.load(result.scenario_path / "power_t000_tx000_r001.npz") as power_npz:
                np.testing.assert_allclose(
                    power_npz["power"],
                    matrix_sets[0].matrices["power"],
                    equal_nan=True,
                )
            assert len(expected_files) == 3 + len(MATRIX_FIELDS) * 2

    def test_overwrite_policy(self) -> None:
        """Existing output is refused unless overwrite is explicitly enabled."""
        params, matrix_sets = build_writer_inputs()

        with writer_tempdir() as tmpdir:
            first_result = write_scenario_folder(
                scenario_root=tmpdir,
                scenario_name=SCENARIO_NAME,
                params=params,
                matrix_sets=matrix_sets,
            )
            sentinel = first_result.scenario_path / "sentinel.txt"
            sentinel.write_text("stale", encoding="utf-8")

            with expect_raises(MatlabRTWriterError):
                write_scenario_folder(
                    scenario_root=tmpdir,
                    scenario_name=SCENARIO_NAME,
                    params=params,
                    matrix_sets=matrix_sets,
                )

            assert sentinel.exists()
            second_result = write_scenario_folder(
                scenario_root=tmpdir,
                scenario_name=SCENARIO_NAME,
                params=params,
                matrix_sets=matrix_sets,
                overwrite=True,
            )
            assert not (second_result.scenario_path / "sentinel.txt").exists()

    def test_rejects_invalid_inputs(self) -> None:
        """Writer rejects unsafe names, malformed params, and incomplete matrices."""
        params, matrix_sets = build_writer_inputs()
        malformed_matrices = dict(matrix_sets[0].matrices)
        malformed_matrices.pop("power")
        malformed_sets = (replace(matrix_sets[0], matrices=malformed_matrices), *matrix_sets[1:])

        with writer_tempdir() as tmpdir:
            with expect_raises(MatlabRTWriterError):
                write_scenario_folder(
                    scenario_root=tmpdir,
                    scenario_name="../bad",
                    params=params,
                    matrix_sets=matrix_sets,
                )
            with expect_raises(MatlabRTWriterError):
                write_scenario_folder(
                    scenario_root=tmpdir,
                    scenario_name=SCENARIO_NAME,
                    params={},
                    matrix_sets=matrix_sets,
                )
            with expect_raises(MatlabRTWriterError):
                write_scenario_folder(
                    scenario_root=tmpdir,
                    scenario_name=SCENARIO_NAME,
                    params=params,
                    matrix_sets=malformed_sets,
                )


if __name__ == "__main__":
    unittest.main()

# ruff: noqa: EM101, EM102, TRY003
"""Filesystem writer for MATLAB RT DeepMIMO scenario folders."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from deepmimo import consts as c
from deepmimo.utils.scenarios import check_scen_name, get_mat_filename

from .errors import MatlabRTWriterError
from .matrices import MATRIX_FIELDS, MatlabRTMatrixSet
from .metadata import RX_SET_ID, TX_SET_ID


@dataclass(frozen=True)
class MatlabRTWriteResult:
    """Summary of a MATLAB RT scenario folder write."""

    scenario_name: str
    scenario_root: Path
    scenario_path: Path
    files_written: tuple[Path, ...]
    matrix_files: dict[str, dict[str, Path]]
    matrix_shapes: dict[str, dict[str, list[int]]]


def write_scenario_folder(
    *,
    scenario_root: str | Path,
    scenario_name: str,
    params: Mapping[str, Any],
    matrix_sets: Sequence[MatlabRTMatrixSet],
    overwrite: bool = False,
) -> MatlabRTWriteResult:
    """Write one DeepMIMO-compatible scenario folder from in-memory data."""
    normalized_name = _normalize_scenario_name(scenario_name)
    root_path = Path(scenario_root)
    scenario_path = _safe_scenario_path(root_path, normalized_name)
    checked_params = _validate_params(params)
    checked_matrix_sets = _validate_matrix_sets(matrix_sets)

    if scenario_path.exists():
        if not overwrite:
            raise MatlabRTWriterError(f"Scenario output already exists: {scenario_path}")
        _remove_existing_scenario(root_path, scenario_path)

    scenario_path.mkdir(parents=True, exist_ok=False)

    files_written: list[Path] = []
    matrix_files: dict[str, dict[str, Path]] = {}
    matrix_shapes: dict[str, dict[str, list[int]]] = {}

    files_written.append(write_params_json(scenario_path, checked_params))
    files_written.append(write_objects_json(scenario_path))
    files_written.append(write_vertices_npz(scenario_path))

    for matrix_set in checked_matrix_sets:
        tx_key = f"tx{matrix_set.tx_index}"
        matrix_files[tx_key] = {}
        matrix_shapes[tx_key] = {}
        for matrix_name in MATRIX_FIELDS:
            output_path = write_matrix_npz(scenario_path, matrix_set, matrix_name)
            files_written.append(output_path)
            matrix_files[tx_key][matrix_name] = output_path
            matrix_shapes[tx_key][matrix_name] = list(matrix_set.matrices[matrix_name].shape)

    return MatlabRTWriteResult(
        scenario_name=normalized_name,
        scenario_root=root_path,
        scenario_path=scenario_path,
        files_written=tuple(files_written),
        matrix_files=matrix_files,
        matrix_shapes=matrix_shapes,
    )


def write_params_json(scenario_path: str | Path, params: Mapping[str, Any]) -> Path:
    """Write ``params.json`` into an existing scenario folder."""
    checked_params = _validate_params(params)
    output_path = Path(scenario_path) / f"{c.PARAMS_FILENAME}.json"
    _write_json(output_path, checked_params)
    return output_path


def write_objects_json(scenario_path: str | Path) -> Path:
    """Write an empty ``objects.json`` placeholder for the MVP scene."""
    output_path = Path(scenario_path) / "objects.json"
    _write_json(output_path, [])
    return output_path


def write_vertices_npz(scenario_path: str | Path) -> Path:
    """Write an empty ``vertices.npz`` placeholder for the MVP scene."""
    output_path = Path(scenario_path) / "vertices.npz"
    np.savez_compressed(output_path, vertices=np.zeros((0, 3), dtype=np.float64))
    return output_path


def write_matrix_npz(
    scenario_path: str | Path,
    matrix_set: MatlabRTMatrixSet,
    matrix_name: str,
) -> Path:
    """Write one per-TX matrix file using the DeepMIMO filename convention."""
    _validate_matrix_set(matrix_set)
    if matrix_name not in MATRIX_FIELDS:
        raise MatlabRTWriterError(f"Unsupported matrix field: {matrix_name!r}")

    array = matrix_set.matrices[matrix_name]
    filename = get_mat_filename(
        matrix_name,
        TX_SET_ID,
        matrix_set.tx_row_index,
        RX_SET_ID,
    )
    output_path = Path(scenario_path) / filename
    np.savez_compressed(output_path, **{matrix_name: array})
    return output_path


def expected_matrix_filenames(matrix_sets: Sequence[MatlabRTMatrixSet]) -> tuple[str, ...]:
    """Return deterministic matrix filenames for the supplied matrix sets."""
    checked_matrix_sets = _validate_matrix_sets(matrix_sets)
    return tuple(
        get_mat_filename(matrix_name, TX_SET_ID, matrix_set.tx_row_index, RX_SET_ID)
        for matrix_set in checked_matrix_sets
        for matrix_name in MATRIX_FIELDS
    )


def _normalize_scenario_name(scenario_name: str) -> str:
    """Validate and normalize a scenario name for DeepMIMO loading."""
    if not isinstance(scenario_name, str) or not scenario_name.strip():
        raise MatlabRTWriterError("scenario_name must be a non-empty string.")
    normalized = scenario_name.strip().lower()
    try:
        check_scen_name(normalized)
    except ValueError as exc:
        raise MatlabRTWriterError(str(exc)) from exc
    return normalized


def _safe_scenario_path(root_path: Path, scenario_name: str) -> Path:
    """Return a scenario path verified to stay inside the requested root."""
    root_resolved = root_path.resolve()
    scenario_path = root_resolved / scenario_name
    scenario_resolved = scenario_path.resolve()
    try:
        common_path = os.path.commonpath([str(root_resolved), str(scenario_resolved)])
    except ValueError as exc:
        message = "scenario_root and scenario output must be on the same drive."
        raise MatlabRTWriterError(message) from exc
    if common_path != str(root_resolved):
        raise MatlabRTWriterError(f"Refusing to write outside scenario_root: {scenario_path}")
    return scenario_path


def _remove_existing_scenario(root_path: Path, scenario_path: Path) -> None:
    """Remove an existing scenario only after root containment checks."""
    safe_path = _safe_scenario_path(root_path, scenario_path.name)
    if safe_path.resolve() != scenario_path.resolve():
        raise MatlabRTWriterError(f"Refusing to overwrite unexpected path: {scenario_path}")
    if scenario_path.is_file():
        raise MatlabRTWriterError(f"Scenario output path is a file: {scenario_path}")
    shutil.rmtree(scenario_path)


def _validate_params(params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the top-level params object accepted by the writer."""
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping.")
    required = {
        c.VERSION_PARAM_NAME,
        c.RT_PARAMS_PARAM_NAME,
        c.SCENE_PARAM_NAME,
        c.MATERIALS_PARAM_NAME,
        c.TXRX_PARAM_NAME,
    }
    missing = required - set(params)
    if missing:
        raise MatlabRTWriterError(f"params missing required sections: {sorted(missing)}")
    return params


def _validate_matrix_sets(
    matrix_sets: Sequence[MatlabRTMatrixSet],
) -> tuple[MatlabRTMatrixSet, ...]:
    """Validate writer matrix-set inputs."""
    if isinstance(matrix_sets, (str, bytes)) or not isinstance(matrix_sets, Sequence):
        raise TypeError("matrix_sets must be a sequence of MatlabRTMatrixSet.")
    checked = tuple(matrix_sets)
    if not checked:
        raise MatlabRTWriterError("matrix_sets must not be empty.")
    for matrix_set in checked:
        _validate_matrix_set(matrix_set)
    return checked


def _validate_matrix_set(matrix_set: MatlabRTMatrixSet) -> None:
    """Validate one in-memory matrix set before writing."""
    if not isinstance(matrix_set, MatlabRTMatrixSet):
        raise TypeError("matrix_sets must contain MatlabRTMatrixSet instances.")

    missing = set(MATRIX_FIELDS) - set(matrix_set.matrices)
    if missing:
        raise MatlabRTWriterError(
            f"matrix_set tx{matrix_set.tx_index} missing matrices: {sorted(missing)}"
        )

    for matrix_name in MATRIX_FIELDS:
        array = matrix_set.matrices[matrix_name]
        if not isinstance(array, np.ndarray):
            raise MatlabRTWriterError(f"matrix {matrix_name!r} must be a NumPy array.")


def _write_json(output_path: Path, data: Any) -> None:
    """Write deterministic UTF-8 JSON."""
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, indent=2, allow_nan=False)
        file.write("\n")

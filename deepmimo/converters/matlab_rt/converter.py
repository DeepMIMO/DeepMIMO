"""Public orchestration API for MATLAB RT JSON conversion."""

from __future__ import annotations

from pathlib import Path

from .matrices import assemble_all_tx_matrices
from .metadata import build_params
from .parser import parse_matlab_rt_json
from .paths import build_path_row_groups
from .schema import MatlabRTExport
from .writer import MatlabRTWriteResult, write_scenario_folder


def convert_matlab_rt_json(
    source: str | Path | MatlabRTExport,
    *,
    scenario_root: str | Path,
    scenario_name: str,
    overwrite: bool = False,
    tx_power_dbw: float = 0.0,
    tx_gain_db: float = 0.0,
    rx_gain_db: float = 0.0,
) -> MatlabRTWriteResult:
    """Convert one MVP MATLAB RT JSON export into a DeepMIMO scenario folder.

    Args:
        source: JSON export path or a parsed ``MatlabRTExport``.
        scenario_root: Directory that will contain the generated scenario folder.
        scenario_name: Name of the generated DeepMIMO scenario.
        overwrite: If true, replace an existing scenario folder with the same name.
        tx_power_dbw: Transmit power used by the received-power policy.
        tx_gain_db: Optional TX gain used by the received-power policy.
        rx_gain_db: Optional RX gain used by the received-power policy.

    Returns:
        ``MatlabRTWriteResult`` with output paths, written files, and matrix shapes.
    """
    export = _load_or_validate_source(source)
    path_groups = build_path_row_groups(
        export,
        tx_power_dbw=tx_power_dbw,
        tx_gain_db=tx_gain_db,
        rx_gain_db=rx_gain_db,
    )
    matrix_sets = assemble_all_tx_matrices(path_groups)
    params = build_params(
        export,
        matrix_sets,
        scenario_name=scenario_name,
        tx_power_dbw=tx_power_dbw,
        tx_gain_db=tx_gain_db,
        rx_gain_db=rx_gain_db,
    )
    return write_scenario_folder(
        scenario_root=scenario_root,
        scenario_name=scenario_name,
        params=params,
        matrix_sets=matrix_sets,
        overwrite=overwrite,
    )


def _load_or_validate_source(source: str | Path | MatlabRTExport) -> MatlabRTExport:
    """Return a parsed MATLAB RT export from a path or parsed object."""
    if isinstance(source, MatlabRTExport):
        return source
    if isinstance(source, (str, Path)):
        return parse_matlab_rt_json(source)
    raise TypeError("source must be a JSON path or MatlabRTExport instance.")

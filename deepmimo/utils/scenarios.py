"""Scenario and path management utilities for DeepMIMO.

This module provides functions for managing scenario paths, RT source paths,
and related file operations.
"""

from pathlib import Path

import numpy as np

from deepmimo import consts as c
from deepmimo.config import config


def check_scen_name(scen_name: str) -> None:
    """Check if a scenario name is valid.

    Args:
        scen_name (str): The scenario name to check

    """
    if np.any([char in scen_name for char in c.SCENARIO_NAME_INVALID_CHARS]):
        invalids = c.SCENARIO_NAME_INVALID_CHARS
        msg = (
            f"Invalid scenario name: {scen_name}.\n"
            f"Contains one of these invalid characters: {invalids}"
        )
        raise ValueError(msg)


def get_scenarios_dir() -> str:
    """Get the absolute path to the scenarios directory.

    This directory contains the extracted scenario folders ready for use.

    Returns:
        str: Absolute path to the scenarios directory

    """
    return str(Path.cwd() / config.get("scenarios_folder"))


def get_scenario_folder(scenario_name: str) -> str:
    """Get the absolute path to a specific scenario folder.

    Args:
        scenario_name: Name of the scenario

    Returns:
        str: Absolute path to the scenario folder

    """
    check_scen_name(scenario_name)
    return str(Path(get_scenarios_dir()) / scenario_name)


def get_rt_sources_dir() -> str:
    """Get the absolute path to the ray tracing sources directory.

    This directory contains the downloaded RT source files.

    Returns:
        str: Absolute path to the RT sources directory

    """
    return str(Path.cwd() / config.get("rt_sources_folder"))


def get_rt_source_folder(scenario_name: str) -> str:
    """Get the absolute path to a specific RT source folder.

    Args:
        scenario_name: Name of the scenario

    Returns:
        str: Absolute path to the RT source folder (extracted contents)

    """
    check_scen_name(scenario_name)
    return str(Path(get_rt_sources_dir()) / scenario_name)


def get_params_path(scenario_name: str) -> str:
    """Get the absolute path to a scenario's params file.

    Args:
        scenario_name: Name of the scenario

    Returns:
        str: Absolute path to the scenario's params file

    Raises:
        FileNotFoundError: If the scenario folder or params file is not found

    """
    check_scen_name(scenario_name)
    scenario_folder = get_scenario_folder(scenario_name)
    scenario_folder_path = Path(scenario_folder)
    if not scenario_folder_path.exists():
        msg = f"Scenario folder not found: {scenario_name}"
        raise FileNotFoundError(msg)
    path = scenario_folder_path / f"{c.PARAMS_FILENAME}.json"
    if not path.exists():
        subdirs = [d for d in scenario_folder_path.iterdir() if d.is_dir()]
        if subdirs:
            path = subdirs[0] / f"{c.PARAMS_FILENAME}.json"
    if not path.exists():
        msg = f"Params file not found for scenario: {scenario_name}"
        raise FileNotFoundError(msg)
    return str(path)


def get_available_scenarios() -> list:
    """Get list of available scenarios in the scenarios directory.

    Returns:
        list: List of scenario folder names

    """
    scenarios_dir = Path(get_scenarios_dir())
    if not scenarios_dir.exists():
        return []

    return [f.name for f in scenarios_dir.iterdir() if f.is_dir() and not f.name.startswith(".")]


def get_mat_filename(
    key: str, tx_set_idx: int, tx_idx: int, rx_set_idx: int, fmt: str = c.MAT_FMT
) -> str:
    """Generate a .mat filename for storing DeepMIMO data.

    Args:
        key (str): The key identifier for the data type.
        tx_set_idx (int): Index of the transmitter set.
        tx_idx (int): Index of the transmitter within its set.
        rx_set_idx (int): Index of the receiver set.
        fmt (str): File extension/format. Defaults to `c.MAT_FMT`.

    Returns:
        str: Complete filename with .mat extension.

    """
    str_id = f"t{tx_set_idx:03}_tx{tx_idx:03}_r{rx_set_idx:03}"
    return f"{key}_{str_id}.{fmt}"


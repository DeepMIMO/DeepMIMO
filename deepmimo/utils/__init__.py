"""Utility modules for DeepMIMO.

This package provides various utility functions and classes used throughout DeepMIMO:
- scenarios: Scenario and path management
- io: File I/O operations (MATLAB, JSON, pickle, zip)
- geometry: Coordinate transformations
- data_structures: Custom data structures (DotDict, DelegatingList)
- dict_utils: Dictionary manipulation utilities
- info: Package information and version
"""

# Scenario management
# Data structures
from .data_structures import DelegatingList, DotDict, PrintIfVerbose

# Dictionary utilities
from .dict_utils import compare_two_dicts, deep_dict_merge

# Geometry
from .geometry import cartesian_to_spherical, spherical_to_cartesian

# Info
from .info import info

# I/O operations
from .io import (
    load_dict_from_json,
    load_mat,
    load_pickle,
    save_dict_as_json,
    save_mat,
    save_pickle,
    unzip,
    zip,  # noqa: A004
)
from .scenarios import (
    check_scen_name,
    get_available_scenarios,
    get_mat_filename,
    get_params_path,
    get_rt_source_folder,
    get_rt_sources_dir,
    get_scenario_folder,
    get_scenarios_dir,
)

__all__ = [
    # Data structures
    "DelegatingList",
    "DotDict",
    "PrintIfVerbose",
    # Geometry
    "cartesian_to_spherical",
    # Scenarios
    "check_scen_name",
    # Dict utils
    "compare_two_dicts",
    "deep_dict_merge",
    "get_available_scenarios",
    "get_mat_filename",
    "get_params_path",
    "get_rt_source_folder",
    "get_rt_sources_dir",
    "get_scenario_folder",
    "get_scenarios_dir",
    # Info
    "info",
    # I/O
    "load_dict_from_json",
    "load_mat",
    "load_pickle",
    "save_dict_as_json",
    "save_mat",
    "save_pickle",
    "spherical_to_cartesian",
    "unzip",
    "zip",
]


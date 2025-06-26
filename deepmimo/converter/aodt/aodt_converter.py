"""
AODT (Aerial Optical Digital Twin) to DeepMIMO Scenario Converter.

This module serves as the main entry point for converting AODT raytracing simulation
outputs into DeepMIMO-compatible scenario files. It processes parquet files containing:

1. Scenario parameters (scenario.parquet)
2. Ray paths and interactions (raypaths.parquet)
3. Channel information (cfrs.parquet, cirs.parquet)
4. TX/RX configurations (rus.parquet, ues.parquet)
5. Material properties (materials.parquet)
6. Antenna patterns (patterns.parquet)
7. Time information (time_info.parquet)

The adapter assumes RUs are transmitters and UEs are receivers.

Main Entry Point:
    aodt_rt_converter(): Converts a complete AODT scenario to DeepMIMO format
"""

import os
import shutil
from typing import Optional
from pprint import pprint

from ... import consts as c
from .. import converter_utils as cu

from .aodt_rt_params import read_rt_params
from .aodt_txrx import read_txrx
from .aodt_paths import read_paths
from .aodt_materials import read_materials
from .aodt_scene import read_scene

def aodt_rt_converter(rt_folder: str, copy_source: bool = False,
                     overwrite: Optional[bool] = None, vis_scene: bool = True, 
                     scenario_name: str = '', print_params: bool = True,
                     parent_folder: str = '', num_scenes: int = 1) -> str:
    """Convert AODT ray-tracing data to DeepMIMO format.

    This function handles the conversion of AODT ray-tracing simulation 
    data into the DeepMIMO dataset format. It processes parquet files containing
    path data, scenario parameters, and transmitter/receiver configurations.

    Args:
        rt_folder (str): Path to folder containing AODT parquet files.
        copy_source (bool): Whether to copy ray-tracing source files to output.
        overwrite (Optional[bool]): Whether to overwrite existing files. Prompts if None.
        vis_scene (bool): Whether to visualize the scene layout. Defaults to True.
        scenario_name (str): Custom name for output folder. Uses folder name if empty.
        print_params (bool): Whether to print parameters to console. Defaults to True.
        parent_folder (str): Parent folder for time-varying scenarios. Defaults to empty.
        num_scenes (int): Number of scenes in time-varying scenario. Defaults to 1.

    Returns:
        str: Path to output folder containing converted DeepMIMO dataset.
        
    Raises:
        FileNotFoundError: If required parquet files are missing.
        ValueError: If transmitter or receiver IDs are invalid.
    """
    print('Converting from AODT')

    # Get scenario name from folder if not provided
    scen_name = scenario_name if scenario_name else os.path.basename(rt_folder)
    
    # Setup temporary output folder
    temp_folder = os.path.join(rt_folder, scen_name + c.DEEPMIMO_CONVERSION_SUFFIX)
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder)

    # Read ray tracing parameters from scenario.parquet
    rt_params = read_rt_params(rt_folder)

    # Read TXRX configurations from rus.parquet and ues.parquet
    txrx_dict = read_txrx(rt_folder, rt_params)

    # Read Paths from raypaths.parquet
    read_paths(rt_folder, temp_folder, txrx_dict)

    # Read Materials from materials.parquet
    materials_dict = read_materials(rt_folder)

    # Read Scene data from world.parquet
    if False:
        scene = read_scene(rt_folder)
        scene_dict = scene.export_data(temp_folder) if scene else {}
    else:
        scene, scene_dict = None, {}
    scene_dict[c.SCENE_PARAM_NUMBER_SCENES] = num_scenes

    # Visualize if requested
    if vis_scene and scene:
        scene.plot()
    
    # Save parameters to params.json
    params = {
        c.VERSION_PARAM_NAME: c.VERSION,
        c.RT_PARAMS_PARAM_NAME: rt_params,
        c.TXRX_PARAM_NAME: txrx_dict,
        c.MATERIALS_PARAM_NAME: materials_dict,
        c.SCENE_PARAM_NAME: scene_dict
    }
    cu.save_params(params, temp_folder)
    if print_params:
        pprint(params)

    # Save scenario to deepmimo scenarios folder
    scenarios_folder = os.path.join(c.SCENARIOS_FOLDER, parent_folder)
    scen_name = cu.save_scenario(temp_folder, scenarios_folder, overwrite=overwrite)
    
    # Copy source files if requested
    if copy_source:
        cu.save_rt_source_files(rt_folder, ['.parquet'])
    
    return scen_name
    
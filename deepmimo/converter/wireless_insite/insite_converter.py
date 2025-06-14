"""Wireless Insite to DeepMIMO Scenario Converter.

This module serves as the main entry point for converting Wireless Insite raytracing simulation
outputs into DeepMIMO-compatible scenario files. It orchestrates the conversion process by:

1. Reading and processing setup files (.setup)
2. Reading TX/RX configurations (.txrx)
3. Converting path data from .p2m files
4. Processing material properties (.city, .ter, .veg)
5. Converting scene geometry
6. Saving all data in DeepMIMO format

Module Dependencies:
    insite_rt_params.py: Ray tracing parameters parsing
    insite_materials.py: Material property handling
    insite_scene.py: Scene geometry conversion
    insite_txrx.py: TX/RX configuration handling
    insite_paths.py: Path data processing
    p2m_parser.py: Low-level .p2m file parsing

The adapter assumes BSs are transmitters and users are receivers. Uplink channels
can be generated using (transpose) reciprocity.

Main Entry Point:
    insite_rt_converter(): Converts a complete Wireless Insite scenario to DeepMIMO format
"""

# Standard library imports
import os
import shutil
from typing import Optional
from pprint import pprint

# Local imports
from .. import converter_utils as cu
from ... import consts as c

from .insite_rt_params import read_rt_params
from .insite_txrx import read_txrx
from .insite_paths import read_paths
from .insite_materials import read_materials
from .insite_scene import read_scene


# Constants
MATERIAL_FILES = ['.city', '.ter', '.veg']
SETUP_FILES = ['.setup', '.txrx'] + MATERIAL_FILES
SOURCE_EXTS = SETUP_FILES + ['.kmz']  # Files to copy to ray tracing source zip

def insite_rt_converter(rt_folder: str, copy_source: bool = False,
                        overwrite: Optional[bool] = None, vis_scene: bool = True, 
                        scenario_name: str = '', print_params: bool = True,
                        parent_folder: str = '', num_scenes: int = 1) -> str:
    """Convert Wireless InSite ray-tracing data to DeepMIMO format.

    This function handles the conversion of Wireless InSite ray-tracing simulation 
    data into the DeepMIMO dataset format. It processes path files (.p2m), setup files,
    and transmitter/receiver configurations to generate channel matrices and metadata.

    Args:
        rt_folder (str): Path to folder containing .setup, .txrx, and material files.
        copy_source (bool): Whether to copy ray-tracing source files to output.
        overwrite (Optional[bool]): Whether to overwrite existing files. Prompts if None. Defaults to None.
        vis_scene (bool): Whether to visualize the scene layout. Defaults to False.
        scenario_name (str): Custom name for output folder. Uses p2m folder name if empty.
        print_params (bool): Whether to print the parameters to the console. Defaults to False.
        parent_folder (str): Name of parent folder containing the scenario. Defaults to empty string.
                             If empty, the scenario is saved in the DeepMIMO scenarios folder.
                             This parameter is only used if the scenario is time-varying.
        num_scenes (int): Number of scenes in the scenario. Defaults to 1.
                          This parameter is only used if the scenario is time-varying.
    Returns:
        str: Path to output folder containing converted DeepMIMO dataset.
        
    Raises:
        FileNotFoundError: If required input files are missing.
        ValueError: If transmitter or receiver IDs are invalid.
    """

    # Get scenario name from folder if not provided
    scen_name = scenario_name if scenario_name else os.path.basename(rt_folder)
    
    # Get paths for input and output folders
    temp_folder = os.path.join(os.path.dirname(rt_folder), scen_name + c.DEEPMIMO_CONVERSION_SUFFIX)
    
    # Create output folder
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder)
    
    # Read ray tracing parameters
    rt_params = read_rt_params(rt_folder)

    # Read TXRX (.txrx)
    txrx_dict = read_txrx(rt_folder)
    
    # Read Paths (.p2m)
    read_paths(rt_folder, temp_folder, txrx_dict)
    
    # Read Materials of all objects (.city, .ter, .veg)
    materials_dict = read_materials(rt_folder)
    
    # Read scene objects
    scene = read_scene(rt_folder)
    scene_dict = scene.export_data(temp_folder)
    scene_dict[c.SCENE_PARAM_NUMBER_SCENES] = num_scenes

    # Visualize if requested
    if vis_scene:
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

    # Save (move) scenario to deepmimo scenarios folder
    scenarios_folder = os.path.join(c.SCENARIOS_FOLDER, parent_folder)
    scen_name = cu.save_scenario(temp_folder, scenarios_folder, overwrite=overwrite)
    
    # Copy and zip ray tracing source files as well
    if copy_source:
        cu.save_rt_source_files(rt_folder, SOURCE_EXTS)
    
    return scen_name
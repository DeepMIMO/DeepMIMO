"""
Sionna Ray Tracing Converter Module.

This module provides functionality for converting Sionna Ray Tracing output files
into the DeepMIMO format. It handles reading and processing ray tracing data including:
- Path information (angles, delays, powers, interactions, ...)
- TX/RX locations and parameters 
- Scene geometry and materials
"""

import os
import shutil
from pprint import pprint

from ... import consts as c
from .. import converter_utils as cu

from .sionna_rt_params import read_rt_params
from .sionna_txrx import read_txrx
from .sionna_paths import read_paths
from .sionna_materials import read_materials
from .sionna_scene import read_scene

def sionna_rt_converter(rt_folder: str, copy_source: bool = False,
                        overwrite: bool = None, vis_scene: bool = True, 
                        scenario_name: str = '', print_params: bool = False) -> str:
    """Convert Sionna ray-tracing data to DeepMIMO format.

    This function handles the conversion of Sionna ray-tracing simulation 
    data into the DeepMIMO dataset format. It processes path data, setup files,
    and transmitter/receiver configurations to generate channel matrices and metadata.

    Args:
        rt_folder (str): Path to folder containing Sionna ray-tracing data.
        copy_source (bool): Whether to copy ray-tracing source files to output.
        overwrite (bool): Whether to overwrite existing files. Prompts if None. Defaults to None.
        vis_scene (bool): Whether to visualize the scene layout. Defaults to False.
        scenario_name (str): Custom name for output folder. Uses rt folder name if empty.

    Returns:
        str: Path to output folder containing converted DeepMIMO dataset.
        
    Raises:
        FileNotFoundError: If required input files are missing.
        ValueError: If transmitter or receiver IDs are invalid.
    """
    print('converting from sionna RT')

    # Get scenario name from folder if not provided
    scen_name = scenario_name if scenario_name else os.path.basename(rt_folder)
    
    # Setup output folder
    output_folder = os.path.join(rt_folder, scen_name + '_deepmimo')
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # Read ray tracing parameters
    rt_params = read_rt_params(rt_folder)

    # Read TXRX
    txrx_dict = read_txrx(rt_params)

    # Read Paths (.paths)
    read_paths(rt_folder, output_folder, txrx_dict, rt_params['raytracer_version'])

    # Read Materials (.materials)
    materials_dict, material_indices = read_materials(rt_folder, output_folder)

    # Read Scene data
    scene = read_scene(rt_folder, material_indices)
    scene_dict = scene.export_data(output_folder) if scene else {}
    
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
    cu.save_params(params, output_folder)
    if print_params:
        pprint(params)

    # Save scenario to deepmimo scenarios folder
    scen_name = cu.save_scenario(output_folder, scen_name=scenario_name, overwrite=overwrite)
    
    # Copy and zip ray tracing source files as well
    if copy_source:
        cu.save_rt_source_files(rt_folder, ['.pkl'])
    
    return scen_name


if __name__ == '__main__':
    rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/' + \
                'all_runs/run_02-02-2025_15H45M26S/scen_0/DeepMIMO_folder'
    output_folder = os.path.join(rt_folder, 'test_deepmimo')

    rt_params = read_rt_params(rt_folder)
    txrx_dict = read_txrx(rt_params)
    read_paths(rt_folder, output_folder)
    read_materials(rt_folder, output_folder)


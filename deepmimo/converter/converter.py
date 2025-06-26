"""
Main converter module for processing raytracing data from different sources.

This module provides functionality to automatically detect and convert raytracing
data from various supported formats (AODT, Sionna RT, Wireless Insite) into
a standardized scenario format.
"""

# Standard library imports
import os
from typing import Dict, Any, Optional, Callable

# Local imports
from .. import consts as c
from . import converter_utils as cu
from .aodt.aodt_converter import aodt_rt_converter
from .sionna_rt.sionna_converter import sionna_rt_converter
from .wireless_insite.insite_converter import insite_rt_converter

def _find_converter_from_dir(directory: str) -> Optional[Callable]:
    """Helper function to find the appropriate converter for a given directory.
    
    Args:
        directory (str): Path to the directory to search for raytracing data
        
    Returns:
        Optional[Callable]: The converter function if a converter is found, or None if no converter is found
    """
    files_in_dir = os.listdir(directory)
    if cu.ext_in_list('.parquet', files_in_dir):
        print("Using AODT converter")
        return aodt_rt_converter
    elif cu.ext_in_list('.pkl', files_in_dir):
        print("Using Sionna RT converter")
        return sionna_rt_converter
    elif cu.ext_in_list('.setup', files_in_dir):
        print("Using Wireless Insite converter")
        return insite_rt_converter
    return None
    
def convert(path_to_rt_folder: str, **conversion_params: Dict[str, Any]) -> Optional[Any]:
    """Create a standardized scenario from raytracing data.
    
    This function automatically detects the raytracing data format based on file 
    extensions and uses the appropriate converter to generate a standardized scenario.
    It supports AODT, Sionna RT, and Wireless Insite formats. It will check both the
    root directory and immediate subdirectories for the required files.

    Args:
        path_to_rt_folder (str): Path to the folder containing raytracing data. 
                                 If the folder contains multiple scenes, the function will
                                 sort them with sorted() and convert each folder to a time snapshot.
        **conversion_params (Dict[str, Any]): Additional parameters for the conversion process

    Returns:
        Optional[Any]: Scenario object if conversion is successful, None otherwise
    """
    print('Determining converter...')

    # First try the root directory
    rt_converter = _find_converter_from_dir(path_to_rt_folder)
    
    # If not found in root, try immediate subdirectories
    if rt_converter is not None: 
        scenario = rt_converter(path_to_rt_folder, **conversion_params)
    else: # Possibly a time-varying scenario
        print(f'No converter match found for root directory: {path_to_rt_folder}')
        print('Checking subdirectories...')
        subdirs = sorted([os.path.join(path_to_rt_folder, d) for d in os.listdir(path_to_rt_folder)
                          if os.path.isdir(os.path.join(path_to_rt_folder, d))])
        if len(subdirs) > 0:
            rt_converter = _find_converter_from_dir(subdirs[0])
        else:
            print('No subdirectories found')
            return None
        
        if rt_converter is None:
            print('No converter match found in subdirectories.')
            print('Make sure the folder contains ray tracing output files (.pkl, .parquet, .setup, etc.)')
            print(f'Supported ray tracers: {c.SUPPORTED_RAYTRACERS}')
            return None

        # The scenario is time-varying if the converter is found

        if 'scenario_name' in conversion_params:
            scenario = conversion_params.pop('scenario_name')
        else:
            scenario = os.path.basename(path_to_rt_folder)

        # Replace the scenario_name string in the conversion_params by parent_folder
        conversion_params['parent_folder'] = scenario
        conversion_params['num_scenes'] = len(subdirs)
        
        # Convert each subdirectory to a time snapshot
        for subdir in subdirs:
            _ = rt_converter(subdir, **conversion_params)
    
    if rt_converter is None:
        print("Unknown ray tracer type")
        return None
    
    return scenario

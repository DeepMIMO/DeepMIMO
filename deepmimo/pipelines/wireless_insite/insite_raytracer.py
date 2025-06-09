"""
Wireless InSite Ray Tracing Pipeline.

This module provides functionality for running electromagnetic simulations using Wireless InSite.
It handles:
- Setting up the simulation environment from OpenStreetMap data:
    - Configuring transmitter and receiver positions 
    - Generating terrain, building and road models
    - Creating simulation configuration files (.setup, .txrx, .ter, .city)
- Running the ray tracing simulation

The module creates both human-readable text files and an XML file used by Wireless InSite.
The text files are useful for debugging and verification in the Wireless InSite UI.
"""

#%% Imports

# Standard library imports
import os
from dataclasses import fields
from typing import Dict, Tuple, Any

# Third-party imports
import numpy as np

# Local application imports
from .WI_interface.XmlGenerator import XmlGenerator
from .WI_interface.SetupEditor import SetupEditor, RayTracingParam
from .WI_interface.TxRxEditor import TxRxEditor
from .WI_interface.TerrainEditor import TerrainEditor
from .convert_ply2city import convert_to_city_file

# Project-specific imports
from ..utils.geo_utils import convert_GpsBBox2CartesianBBox
from ..utils.pipeline_utils import run_command

from ...consts import BBOX_PAD
from ...config import config

TERRAIN_TEMPLATE = "newTerrain.ter"

def create_directory_structure(osm_folder: str, rt_params: Dict[str, Any]) -> Tuple[str, str]:
    """Create folders for the scenario generations with a names based on parameters.
    
    Args:
        base_path (str): Base path for the scenario
        rt_params (Dict[str, Any]): Ray tracing parameters
        
    Returns:
        Tuple[str, str]: Paths to the insite directory and study area directory
    """
    
    # Format folder name with key parameters
    folder_name = (f"insite_{rt_params['carrier_freq']/1e9:.1f}GHz_"
                   f"{rt_params['max_reflections']}R_{rt_params['max_diffractions']}D_"
                   f"{1 if rt_params['ds_enable'] else 0}S")
    insite_path = os.path.join(osm_folder, folder_name)
    os.makedirs(insite_path, exist_ok=True)

    insite_path = os.path.join(osm_folder, folder_name)
    study_area_path = os.path.join(insite_path, "study_area")

    # Create directories
    for path in [insite_path, study_area_path]:
        os.makedirs(path, exist_ok=True)

    return insite_path, study_area_path


def raytrace_insite(osm_folder: str, tx_pos: np.ndarray, rx_pos: np.ndarray, **rt_params: Any) -> str:
    """Run Wireless InSite ray tracing simulation.
    
    This function sets up the simulation environment, generates the necessary files,
    and runs the ray tracing simulation. It creates both human-readable text files
    (.setup, .txrx, .ter, .city) and the XML file that is actually used by Wireless InSite.
    
    Required Parameters:
        osm_folder (str): Path to the OSM folder
        tx_pos (np.ndarray): Transmitter positions (M x 3 array)
        rx_pos (np.ndarray): Receiver positions (N x 3 array)
        rt_params (dict): Dictionary containing:
            Required Paths:
                - wi_exe (str): Path to Wireless InSite executable
                - wi_lic (str): Path to Wireless InSite license
                - building_material (str): Path to building material file
                - road_material (str): Path to road material file
                - terrain_material (str): Path to terrain material file
            
            Required Parameters:
                - carrier_freq (float): Carrier frequency in Hz
                - bandwidth (float): Bandwidth in Hz
                - grid_spacing (float): Grid spacing in meters
                - ue_height (float): UE height in meters
                - conform_to_terrain (bool): Whether to conform to terrain
                - min_lat, min_lon, max_lat, max_lon (float): GPS coordinates of the area
                - origin_lat, origin_lon (float): Origin GPS coordinates
            
            Ray Tracing Parameters:
                - max_paths (int): Maximum number of paths to render
                - ray_spacing (float): Spacing between rays
                - max_reflections (int): Maximum number of reflections
                - max_transmissions (int): Maximum number of transmissions
                - max_diffractions (int): Maximum number of diffractions
                - ds_enable (bool): Whether diffuse scattering is enabled
                - ds_max_reflections (int): Maximum number of diffuse reflections
                - ds_max_transmissions (int): Maximum number of diffuse transmissions
                - ds_max_diffractions (int): Maximum number of diffuse diffractions
                - ds_final_interaction_only (bool): Whether to only apply diffuse scattering at final interaction
    
    Returns:
        str: Path to the insite directory containing all generated files
    """
    insite_path, study_area_path = create_directory_structure(osm_folder, rt_params)
    
    # Create buildings.city & roads.city files
    bldgs_city = convert_to_city_file(osm_folder, insite_path, "buildings", rt_params['building_material'])
    roads_city = convert_to_city_file(osm_folder, insite_path, "roads", rt_params['road_material'])

    xmin_pad, ymin_pad, xmax_pad, ymax_pad = convert_GpsBBox2CartesianBBox(
        rt_params['min_lat'], rt_params['min_lon'], rt_params['max_lat'], rt_params['max_lon'],
        rt_params['origin_lat'], rt_params['origin_lon'], pad=BBOX_PAD
    ) # pad makes the box larger -> only for terrain placement and study area boundary purposes

    # Create terrain file (.ter)
    terrain_editor = TerrainEditor()
    terrain_editor.set_vertex(xmin=xmin_pad, ymin=ymin_pad, xmax=xmax_pad, ymax=ymax_pad)
    terrain_editor.set_material(rt_params['terrain_material'])
    terrain_editor.save(os.path.join(insite_path, TERRAIN_TEMPLATE))

    # Configure Tx/Rx (.txrx)
    txrx_editor = TxRxEditor()

    #   TX (BS)
    for b_idx, pos in enumerate(tx_pos):
        txrx_editor.add_txrx(
            txrx_type="points",
            is_transmitter=True,
            is_receiver=rt_params['bs2bs'],
            pos=pos,
            name=f"BS{b_idx+1}",
            conform_to_terrain=False)

    #   RX (UEs) - adding ues as points is much slower than as a grid
    if False:
        txrx_editor.add_txrx(
                txrx_type="points",
                is_transmitter=False,
                is_receiver=True,
                pos=rx_pos,
                name="user_grid",
                conform_to_terrain=rt_params['conform_to_terrain'])
    
    # The user grid should cover the bounding box area that was fetched from OSM
    grid_side = [xmax_pad - xmin_pad - 2 * BBOX_PAD + rt_params['grid_spacing'], 
                 ymax_pad - ymin_pad - 2 * BBOX_PAD + rt_params['grid_spacing']]
    txrx_editor.add_txrx(
        txrx_type="grid",
        is_transmitter=False,
        is_receiver=True,
        pos=[xmin_pad + BBOX_PAD + 1e-3, ymin_pad + BBOX_PAD, rt_params['ue_height']],
        name="UE_grid",
        grid_side=grid_side,
        grid_spacing=rt_params['grid_spacing'],
        conform_to_terrain=rt_params['conform_to_terrain']
    )
    txrx_editor.save(os.path.join(insite_path, "insite.txrx"))

    # Get ray tracing parameter names from the dataclass
    rt_param_names = {field.name for field in fields(RayTracingParam)}
    rt_params_filtered = {k: v for k, v in rt_params.items() if k in rt_param_names}

    # Define study area bbox in Cartesian coordinates
    study_area_vertex = np.array([[xmin_pad, ymin_pad, 0],
                                  [xmax_pad, ymin_pad, 0],
                                  [xmax_pad, ymax_pad, 0],
                                  [xmin_pad, ymax_pad, 0]])

    # Create setup file (.setup)
    scenario = SetupEditor(insite_path)
    scenario.set_carrierFreq(rt_params['carrier_freq'])
    scenario.set_bandwidth(rt_params['bandwidth'])
    scenario.set_study_area(zmin=-3, zmax=20, all_vertex=study_area_vertex)
    mean_lat = (rt_params['min_lat'] + rt_params['max_lat']) / 2
    mean_lon = (rt_params['min_lon'] + rt_params['max_lon']) / 2
    scenario.set_origin(mean_lat, mean_lon)
    scenario.set_ray_tracing_param(rt_params_filtered)
    scenario.set_txrx("insite.txrx")
    scenario.add_feature(TERRAIN_TEMPLATE, "terrain")
    if bldgs_city:
        scenario.add_feature(bldgs_city, "city")
    else:
        raise Exception('No buildings found. Check Blender Export to ply.')
    if roads_city:
        scenario.add_feature(roads_city, "road")
    scenario.save("insite") # insite.setup

    # Generate XML file (.xml) - What Wireless InSite executable actually uses
    wi_major_version = int(config.get('wireless_insite_version')[0])
    xml_generator = XmlGenerator(insite_path, scenario, txrx_editor, version=wi_major_version)
    xml_generator.update()
    xml_path = os.path.join(insite_path, "insite.study_area.xml")
    xml_generator.save(xml_path)

    license_info = ["-set_licenses", rt_params['wi_lic']] if wi_major_version >= 4 else []
    
    # Run Wireless InSite using the XML file
    command = [rt_params['wi_exe'], "-f", xml_path, "-out", study_area_path, "-p", "insite"] + license_info
    run_command(command, "RAY TRACING: Wireless InSite")
    
    return insite_path

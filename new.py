import sys
import utm
# Import existing modules
from pathlib import Path
from generate_city.generate_city import generate_city
from WI_interface.XmlGenerator import XmlGenerator
from WI_interface.SetupEditor import SetupEditor
from WI_interface.TxRxEditor import TxRxEditor
from WI_interface.TerrainEditor import TerrainEditor
import pandas as pd
import deepmimo as dm
import subprocess
import os
from utils.geo_utils import convert_GpsBBox2CartesianBBox, convert_Gps2RelativeCartesian
from constants import PROJ_ROOT, GRID_SPACING, UE_HEIGHT, BS_HEIGHT, BLENDER_PATH
import numpy as np
from datetime import datetime as dt 

df = pd.read_csv('params.csv')

def create_directory_structure(base_path, rt_params):
    """Create necessary directories for the scenario with a professional folder name based on parameters."""
    
    # Format folder name with key parameters
    folder_name = (f"insite_{rt_params['carrier_freq']/1e9:.1f}GHz_{rt_params['bandwidth']/1e6:.0f}MHz_"
                   f"{rt_params['max_paths']}paths_{rt_params['max_reflections']}ref_{rt_params['max_transmissions']}trans_{rt_params['max_diffractions']}diff")
    
    insite_path = base_path / folder_name
    intermediate_path = insite_path / "intermediate_files"
    mat_path = insite_path / "study_area_mat"
    study_area_path = insite_path / "study_area"

    # Create directories
    for path in [insite_path, intermediate_path, mat_path, study_area_path]:
        path.mkdir(parents=True, exist_ok=True)

    return insite_path, intermediate_path, mat_path, study_area_path

def get_grid_info(xmin, ymin, xmax, ymax, grid_spacing):
    """Calculate the grid layout and extract available rows and users per row."""
    # Create grid
    x_coords = np.arange(xmin, xmax + grid_spacing, grid_spacing)
    y_coords = np.arange(ymin, ymax + grid_spacing, grid_spacing)
    # Indices of rows and number of users per row
    row_indices = np.arange(len(y_coords) - 1)
    users_per_row = len(x_coords) - 1  # Each row has the same number of users
    return row_indices, users_per_row


def run_command(command, description):
    """Run a shell command and stream output in real-time."""
    print(f"\n🚀 Starting: {description}...\n")
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")

    # Stream the output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end="")  # Print each line as it arrives

    process.stdout.close()
    process.wait()

    print(f"\n✅ {description} completed!\n")

def read_rt_configs(row):
    scene_name = row['scenario_name']
    min_lat = row['min_lat']
    min_lon = row['min_lon']
    max_lat = row['max_lat']
    max_lon = row['max_lon']
    bs_lats = np.array(row['bs_lat'].split(',')).astype(np.float32)
    bs_lons = np.array(row['bs_lon'].split(',')).astype(np.float32)
    carrier_freq = row['freq (ghz)'] * 1e9
    n_reflections = row['n_reflections']
    diffraction = bool(row['ds_enable'])
    scattering = bool(row['ds_enable'])
    
    max_paths = row['max_paths']
    ray_spacing = row['ray_spacing']
    max_transmissions = row['max_transmissions']
    max_diffractions = row['max_diffractions']

    ds_enable = row['ds_enable']
    ds_max_reflections = row['ds_max_reflections']
    ds_max_transmissions = row['ds_max_transmissions']
    ds_max_diffractions = row['ds_max_diffractions']
    ds_final_interaction_only = row['ds_final_interaction_only']
    

    rt_params = {
        'scene_name': scene_name,
        'min_lat': min_lat,
        'min_lon': min_lon,
        'max_lat': max_lat,
        'max_lon': max_lon,
        'bs_lats': bs_lats,
        'bs_lons': bs_lons,
        'carrier_freq': carrier_freq,
        ## changed
        'max_reflections': n_reflections,
        'diffraction': diffraction,
        'scattering': scattering,

        'max_paths': max_paths,
        'ray_spacing': ray_spacing,
        'max_transmissions': max_transmissions,
        'max_diffractions': max_diffractions,

        'ds_enable': ds_enable,
        'ds_max_reflections': ds_max_reflections,
        'ds_max_transmissions': ds_max_transmissions,
        'ds_max_diffractions': ds_max_diffractions,
        'ds_final_interaction_only': ds_final_interaction_only
    }
    return rt_params


def gen_tx_pos(rt_params):
    num_bs = len(rt_params['bs_lats'])
    print(f"Number of BSs: {num_bs}")
    bs_pos = [[convert_Gps2RelativeCartesian(rt_params['bs_lats'][i], rt_params['bs_lons'][i], rt_params['origin_lat'], rt_params['origin_lon'])[0],
                convert_Gps2RelativeCartesian(rt_params['bs_lats'][i], rt_params['bs_lons'][i], rt_params['origin_lat'], rt_params['origin_lon'])[1], 
                BS_HEIGHT]
                for i in range(num_bs)]
    return bs_pos

def gen_rx_pos(row, osm_folder):
    with open(os.path.join('osm_exports', osm_folder, 'osm_gps_origin.txt'), "r") as f:
        origin_lat, origin_lon = map(float, f.read().split())
    print(f"origin_lat: {origin_lat}, origin_lon: {origin_lon}")

    user_grid = generate_user_grid(row, origin_lat, origin_lon)
    print(f"User grid shape: {user_grid.shape}")
    return user_grid

def generate_user_grid(row, origin_lat, origin_lon):
    """Generate user grid in Cartesian coordinates."""
    min_lat, min_lon = row['min_lat'], row['min_lon']
    max_lat, max_lon = row['max_lat'], row['max_lon']
    xmin, ymin, xmax, ymax = convert_GpsBBox2CartesianBBox(
        min_lat, min_lon, 
        max_lat, max_lon, 
        origin_lat, origin_lon)
    grid_x = np.arange(xmin, xmax + GRID_SPACING, GRID_SPACING)
    grid_y = np.arange(ymin, ymax + GRID_SPACING, GRID_SPACING)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = np.zeros_like(grid_x) + UE_HEIGHT
    return np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=-1) 


def call_blender1(rt_params):
    osm_command = [
        "python", "run_osm_extraction.py",
        "--minlat", str(rt_params['min_lat']), "--minlon", str(rt_params['min_lon']),
        "--maxlat", str(rt_params['max_lat']), "--maxlon", str(rt_params['max_lon'])
    ]
    run_command(osm_command, "OSM Extraction")

def call_blender2():
    pass

def insite_raytrace(osm_folder, tx_pos, rx_pos, **rt_params):
    

    insite_path, intermediate_path, mat_path, study_area_path = create_directory_structure(osm_folder, rt_params)
    root_dir = Path("C:/Users/namhyunk/Desktop/osm2dt")
    # Generate city features
    city_feature_list = generate_city(
        str(osm_folder) + os.sep,  # Add trailing separator explicitly
        str(insite_path) + os.sep,
        minlat=rt_params['min_lat'],
        minlon=rt_params['min_lon'],
        maxlat=rt_params['max_lat'],
        maxlon=rt_params['max_lon'],
        building_mtl_path=str(root_dir / "resource/material/ITU Concrete 3.5 GHz.mtl"),
        road_mtl_path=str(root_dir / "resource/material/Asphalt_1GHz.mtl"),
    )

    xmin, ymin, xmax, ymax = convert_GpsBBox2CartesianBBox(
        rt_params['min_lat'], rt_params['min_lon'], rt_params['max_lat'], rt_params['max_lon'],
        rt_params['origin_lat'], rt_params['origin_lon'], pad=0
    )
    xmin_pad, ymin_pad, xmax_pad, ymax_pad = convert_GpsBBox2CartesianBBox(
        rt_params['min_lat'], rt_params['min_lon'], rt_params['max_lat'], rt_params['max_lon'],
        rt_params['origin_lat'], rt_params['origin_lon'], pad=30
    )

    folder_name = (f"insite_{rt_params['carrier_freq']/1e9:.1f}GHz_{rt_params['bandwidth']/1e6:.0f}MHz_"
                   f"{rt_params['max_paths']}paths_{rt_params['max_reflections']}ref_{rt_params['max_transmissions']}trans_{rt_params['max_diffractions']}diff")
    insite_path = osm_folder / folder_name

    terrain_editor = TerrainEditor()
    terrain_editor.set_vertex(xmin=xmin_pad, ymin=ymin_pad, xmax=xmax_pad, ymax=ymax_pad)
    terrain_editor.set_material(str(root_dir / "resource/material/ITU Wet earth 3.5 GHz.mtl"))
    terrain_editor.save(str(insite_path / "newTerrain.ter"))

    # Configure Tx/Rx
    txrx_editor = TxRxEditor()
    for b_idx, pos in enumerate(tx_pos):
        txrx_editor.add_txrx(
            txrx_type="points",
            is_transmitter=True,
            is_receiver=True,
            pos=pos,
            name=f"BS{b_idx+1}"
        )

    grid_side = [xmax_pad - xmin_pad, ymax_pad - ymin_pad]
    grid_spacing = rt_params['ray_spacing'] 
    txrx_editor.add_txrx(
        txrx_type="grid",
        is_transmitter=False,
        is_receiver=True,
        pos=[xmin_pad, ymin_pad, rt_params['ue_height']],
        name="UE_grid",
        grid_side=grid_side,
        grid_spacing=grid_spacing
    )
    txrx_editor.save(str(insite_path / "insite.txrx"))

    # Calculate grid info
    row_indices, users_per_row = get_grid_info(xmin, ymin, xmax, ymax, grid_spacing)
    
    # Create setup file
    scenario = SetupEditor(str(insite_path))
    scenario.set_carrierFreq_and_bandwidth(carrier_frequency=rt_params['carrier_freq'], bandwidth=rt_params['bandwidth'])
    scenario.set_study_area(
        zmin=-3,
        zmax=17.5,
        all_vertex=np.array([
            [xmin_pad, ymin_pad, 0],
            [xmax_pad, ymin_pad, 0],
            [xmax_pad, ymax_pad, 0],
            [xmin_pad, ymax_pad, 0]
        ])
    )
    scenario.set_ray_tracing_param(
        rt_params['max_paths'],
        rt_params['ray_spacing'],
        rt_params['max_reflections'],
        rt_params['max_transmissions'],
        rt_params['max_diffractions'],
        rt_params['ds_enable'],
        rt_params['ds_max_reflections'],
        rt_params['ds_max_transmissions'],
        rt_params['ds_max_diffractions'],
        rt_params['ds_final_interaction_only']
    )
    scenario.set_txrx("/insite.txrx")
    scenario.add_feature("newTerrain.ter", "terrain")
    for city_feature in city_feature_list:
        scenario.add_feature(city_feature, "city")
    scenario.save("/insite") # insite

    # Generate XML and run simulation
    xml_generator = XmlGenerator(str(insite_path), "\\insite.setup") # insite.setup
    xml_generator.update()
    xml_path = insite_path / "insite.study_area.xml"
    xml_generator.save(str(xml_path))


## main
for index, row in df.iterrows():
	# TODO1: read_rt_configs()
    rt_params = read_rt_configs(row) # dict(n_reflections, diffraction, scattering, ...)
    rt_params['ue_height'] = 2
    rt_params['bandwidth'] = 10e6
    
    # TODO2: call_blender1()
    call_blender1(rt_params)

    # TODO5: call_blender2()
    # call_blender2(rt_params)

    # Identify the paths --
    # osm_folder = os.path.join(PROJ_ROOT, f"bbox_{rt_params['min_lat']}_{rt_params['min_lon']}_{rt_params['max_lat']}_{rt_params['max_lon']}".replace(".", "-"))
    root_dir = Path("C:/Users/namhyunk/Desktop/osm2dt")
    bbox_folder = f"bbox_{rt_params['min_lat']}_{rt_params['min_lon']}_{rt_params['max_lat']}_{rt_params['max_lon']}".replace('.', '-')
    osm_folder = root_dir / "osm_exports" / bbox_folder
    csv_path = os.path.join(PROJ_ROOT, 'params.csv')

    with open(os.path.join('osm_exports', osm_folder, 'osm_gps_origin.txt'), "r") as f:
        rt_params['origin_lat'], rt_params['origin_lon'] = map(float, f.read().split())
    print(f"origin_lat: {rt_params['origin_lat']}, origin_lon: {rt_params['origin_lon']}")

    user_grid = generate_user_grid(row, rt_params['origin_lat'], rt_params['origin_lon'])
    print(f"User grid shape: {user_grid.shape}")

	# TODO3: gen_positins()
	# Generate XY user grid and BS positions
    rx_pos = gen_rx_pos(row, osm_folder)  # N x 3 (N ~ 20k)
    tx_pos = gen_tx_pos(rt_params)  # M x 3 (M ~ 3)

    # TODO4: insite_raytrace()
	# Ray Tracing
    insite_rt_path = insite_raytrace(osm_folder, tx_pos, rx_pos, **rt_params)
    
    print('end of a scenario!!')
    print('--------------------')

	# Convert to DeepMIMO
    scen_insite = dm.convert(insite_rt_path) #-- supposed to be deepmimo

	# Test Conversion
    dataset_insite = dm.load(scen_insite)
    



for index, row in df.iterrows():

    # STEP 1: replace the CSV logic
    # python scenario_generator.py --minlat 64.11029 --minlon -21.90496 --maxlat 64.11197 --maxlon -21.90077 --bs 64.11118,-21.90184 --ue_height 1.5 --bs_height 15 --max_path 25 --max_reflections 3 --max_diffractions 2

    command = ['python', 'scenario_generator.py',
            '--minlat', row['min_lat'], 
            '--minlon', row['min_lon'],
            '--maxlat', row['max_lat'],
            '--maxlon', row['max_lon'],
            '--bs', row['bs_lat'], row['bs_lon'],
            '--ue_height', '1.5', 
            '--bs_height', '15', 
            '--max_path', '25', 
            '--max_reflections', '3', 
            '--max_diffractions', '2']

    subprocess.run(command, capture_output=True, text=True, check=True)


    # STEP 2: replace the CSV logic
    rt_params = read_rt_configs(row)
    run_insite(**rt_params)

    # STEP 3: replace the CSV logic
    blender_path = call_blender1()
    run_insite(blender_path, **rt_params)
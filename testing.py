#%% Imports

import os
import numpy as np
import deepmimo as dm
import matplotlib.pyplot as plt

from api_keys import DEEPMIMO_API_KEY

#%% V4 Conversion

# Example usage
rt_folder = './RT_SOURCES/asu_campus'
# rt_folder = './P2Ms/simple_street_canyon_test'
# rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/all_runs/run_02-02-2025_15H45M26S/scen_0/DeepMIMO_folder'
# rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/all_runs/run_02-02-2025_15H45M26S/scen_0/sionna_test'
# rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/all_runs/run_02-02-2025_15H45M26S/scen_0/sionna_export_test2'
# rt_folder = r'C:\Users\jmora\Downloads\DeepMIMOv4-hao-test\all_runs\run_03-08-2025_15H38M57S\NewYork\sionna_export_full'
# rt_folder = r'C:\Users\jmora\Documents\GitHub\AutoRayTracing\all_runs\run_03-09-2025_18H18M51S\NewYork\sionna_export_RX'
# rt_folder = 'C:/Users/jmora/Documents/GitHub/DeepMIMO/P2Ms/sionna_test_scen'

scen_name = os.path.basename(rt_folder)
dm.convert(rt_folder, overwrite=True, scenario_name=scen_name, vis_scene=True)

#%% Trimming by path type

dataset = dm.load('asu_campus_3p5')

# dataset_t = dataset.trim_by_path_depth(1)
dataset_t = dataset.trim_by_path_type(['LoS', 'R'])

dataset.plot_coverage(dataset.los, title='Full dataset')
dataset_t.plot_coverage(dataset_t.los, title='Trimmed dataset')

#%% num interactions
dataset.plot_coverage(dataset.num_interactions[:,0], title='Number of interactions')
dataset_t.plot_coverage(dataset_t.num_interactions[:,0], title='Number of interactions')

#%% num paths
dataset.plot_coverage(dataset.num_paths, title='Number of paths')
dataset_t.plot_coverage(dataset_t.num_paths, title='Number of paths')

#%% interaction type
dataset_t.plot_coverage(dataset_t.inter[:,0], title='Interaction type')

#%% Plot rays
dataset_t.plot_rays(9)


#%% Sionna verification

import pickle
def load_pickle(filename: str):
    with open(filename, 'rb') as file:
        return pickle.load(file)
    
from pprint import pprint

p  = load_pickle(os.path.join(rt_folder, 'sionna_paths.pkl'))
m  = load_pickle(os.path.join(rt_folder, 'sionna_materials.pkl'))
mi = load_pickle(os.path.join(rt_folder, 'sionna_material_indices.pkl'))
rt = load_pickle(os.path.join(rt_folder, 'sionna_rt_params.pkl'))
v  = load_pickle(os.path.join(rt_folder, 'sionna_vertices.pkl'))
o  = load_pickle(os.path.join(rt_folder, 'sionna_objects.pkl'))

dataset = dm.load(scen_name)

#%% Plotting

main_keys = ['aoa_az', 'aoa_el', 'aod_az', 'aod_el', 'delay', 'power', 'phase',
             'los', 'distance', 'num_paths', 'inter_int']

for key in main_keys:
    plt_var = dataset[key][:,0] if dataset[key].ndim == 2 else dataset[key]
    dataset.plot_coverage(plt_var, title=key, scat_sz=20)

# Q1: Is phase correct? (should be degrees, but it's probably radians)
# Q2: los is not correct. Most are multiples reflections.
# Q3: inter_int is not correct - not accusing multiple reflections.

#%% Test channel generation

import deepmimo as dm
dataset = dm.load('asu_campus_3p5')

# Create channel parameters with all options
ch_params = dm.ChannelParameters()

# Antenna parameters

# Base station antenna parameters (lists are automatically converted to numpy arrays)
ch_params.bs_antenna.rotation = [0, 0, 0]  # [az, el, pol] in degrees
ch_params.bs_antenna.shape = [8, 1]        # [horizontal, vertical] elements
ch_params.bs_antenna.spacing = 0.5         # Element spacing in wavelengths

# User equipment antenna parameters (lists are automatically converted to numpy arrays)
ch_params.ue_antenna.rotation = [0, 0, 0]  # [az, el, pol] in degrees
ch_params.ue_antenna.shape = [1, 1]        # [horizontal, vertical] elements
ch_params.ue_antenna.spacing = 0.5         # Element spacing in wavelengths

# Channel parameters
ch_params.freq_domain = True  # Whether to compute frequency domain channels
ch_params.num_paths = 10      # Number of paths

# OFDM parameters
ch_params.ofdm.bandwidth = 10e6                      # Bandwidth in Hz
ch_params.ofdm.subcarriers = 512                     # Number of subcarriers
ch_params.ofdm.selected_subcarriers = [0]            # Which subcarriers to generate (list automatically converted to array)
ch_params.ofdm.rx_filter = 0                         # Receive Low Pass / ADC Filter

dataset.set_channel_params(ch_params)

# Generate channels
# dataset.compute_channels(ch_params)
dataset.channel.shape


#%% PLOT RAYS

dm.plot_rays(dataset['rx_pos'][10], dataset['tx_pos'][0],
             dataset['inter_pos'][10], dataset['inter'][10],
             proj_3D=True, color_by_type=True)

# Next: determine which buildings interact with each ray. 
#       make a set of those buildings for all the rays in the user.
#       plot the buildings that matter to that user along with the rays.
#       (based on the building bounding boxes)
#       Use the PhysicalObjects class to plot a group of buildings.
#### NEXT: Make a plot of just SOME of the buildings
# buildings_scene = dm.Scene()
# for obj in buildings[:3]:
#     buildings_scene.add_object(obj)
    
# buildings_scene.plot()
#####

#%% LOOP for zip (no upload)

# Get all available scenarios using function
scenarios = dm.get_available_scenarios()

metadata_dict = {
    'bbCoords': {
        "minLat": 40.68503298,
        "minLon": -73.84682129, 
        "maxLat": 40.68597435,
        "maxLon": -73.84336302
    },
    'digitalTwin': True,
    'environment': 'indoor',
    "city": "New York"
}
metadata_dict = {}

# Zip the filtered scenarios
for scenario in scenarios:
    scen_path = dm.get_scenario_folder(scenario)
    # dm.zip(scen_path)
    if not ('boston' in scenario):
        continue
    print(f"\nProcessing: {scenario}")
    # continue
    dm.upload(scenario, DEEPMIMO_API_KEY, skip_zip=False, extra_metadata=metadata_dict)

#%% LOOP all scenarios (summary, load, plot)

import deepmimo as dm
import matplotlib.pyplot as plt
base_path = 'F:/deepmimo_loop_ready'
subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]

# Load all scenarios
for subfolder in subfolders[:-5]:
    scen_name = os.path.basename(subfolder)
    
    # dm.summary(scen_name)

    # dm.load(scen_name, 'matrices': None)

    try:
        d = dm.load(scen_name, tx_sets={1: []}, rx_sets={2: []}, matrices=None)
    except Exception as e:
        d = dm.load(scen_name, tx_sets={1: []}, rx_sets={4: []}, matrices=None)
    _, ax = d.scene.plot(title=False)
    ax.set_title(scen_name + ': ' + ax.get_title())
    plt.show()

#%%

from clickhouse_driver import Client
client = Client('localhost')
client.execute('SHOW DATABASES')

# Export: Load the files into the database and make the table requests from there.
dm_aodt_rt_path = dm.aodt.exporter(db_name='aerial_2025_6_22_16_10_16', 
                                   clickhouse_client_obj=client)

# Convert: Read the parquet files into a deepmimo dataset
# aodt_scen_name = dm.convert(dm_aodt_rt_path)
# aodt_dataset = dm.load(aodt_scen_name)

#%%
import deepmimo as dm
import pandas as pd

aodt_scen_name = 'aerial_2025_6_22_16_10_16'
folder = f'aodt_scripts/{aodt_scen_name}'
# df = pd.read_parquet(os.path.join(folder, 'db_info.parquet'))

# df.head()
dm.convert(folder, overwrite=True)

aodt_scen = dm.load(aodt_scen_name)
# TODO: save a single file for all ues (aodt_paths)


#%%

for file in os.listdir(folder):
    if file.endswith('.parquet'):
        df = pd.read_parquet(os.path.join(folder, file))
        # print(df.head())
        print(f'file: {file}, columns: {df.columns}')
        if df.empty:
            print(f'WARNING: {file} is empty')


#%%









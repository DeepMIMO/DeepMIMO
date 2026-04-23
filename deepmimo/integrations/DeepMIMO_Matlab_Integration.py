import deepmimo as dm
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

scene_name = 'asu_campus_3p5'

# dm.download(scene_name) # Uncomment this to download the dataset

dataset = dm.load(scene_name)
aoa_az = dataset.aoa_az
aoa_el = dataset.aoa_el
aod_az = dataset.aod_az
aod_el = dataset.aod_el
delay = dataset.delay
power = dataset.power
phase = dataset.phase
los = dataset.los
# carrier_frequency = dataset.rt_params.raw_params.waveform.CarrierFrequency
carrier_frequency = dataset.rt_params.frequency
bandwidth = dataset.rt_params.raw_params.waveform.bandwidth

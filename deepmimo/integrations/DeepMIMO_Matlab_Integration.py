import deepmimo as dm
import numpy as np
import matplotlib.pyplot as plt
import os

scene_name = 'asu_campus_3p5'

dm.download(scene_name) # Uncomment this to download the dataset

dataset = dm.load(scene_name)
aoa_az = np.array(dataset.aoa_az)
aoa_el = np.array(dataset.aoa_el)
aod_az = np.array(dataset.aod_az)
aod_el = np.array(dataset.aod_el)
delay = np.array(dataset.delay)
power = np.array(dataset.power)
phase = np.array(dataset.phase)
los = np.array(dataset.los)
carrier_frequency = dataset.rt_params.raw_params.waveform.CarrierFrequency
bandwidth = dataset.rt_params.raw_params.waveform.bandwidth


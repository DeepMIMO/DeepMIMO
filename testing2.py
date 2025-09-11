
#%%
import os
import numpy as np
import deepmimo as dm
import matplotlib.pyplot as plt

from tqdm import tqdm
# from api_keys import DEEPMIMO_API_KEY

# d_10cm = dm.load('asu_campus_3p5_10cm', filter_matrices=['inter_pos'])

matrices = ['rx_pos', 'tx_pos', 'aoa_az', 'aod_az', 'aoa_el', 'aod_el', 
            'delay', 'power', 'phase', 'inter']
# dataset = dm.load('asu_campus_3p5_10cm', matrices=matrices)

dataset = dm.load('asu_campus_3p5')

dataset_t = dataset.subset(dataset.get_active_idxs())

#%%

ch_params = dm.ChannelParameters()
ch_params.bs_antenna.shape = [3, 1]
ch_params.ue_antenna.shape = [1, 1]
ch_params.doppler = True

dataset_t.set_doppler(1)
H = dataset_t.compute_channels(ch_params)
H2 = dataset_t.compute_channels(ch_params, times=0.0)
H3 = dataset_t.compute_channels(ch_params, times=0.001)
H4 = dataset_t.compute_channels(ch_params, times=np.array([0.0, 0.001]))

#%%
rtol = 1e-15
print(f'np.allclose(H, H2, rtol={rtol}): {np.allclose(H, H2, rtol=rtol)}')

print(f'np.allclose(H, H3, rtol={rtol}): {np.allclose(H, H3, rtol=rtol)}')

print(f'np.allclose(H2, H3, rtol={rtol}): {np.allclose(H2, H3, rtol=rtol)}')

print(f'np.allclose(H2, H4[..., 0], rtol={rtol}): {np.allclose(H2, H4[..., 0], rtol=rtol)}')

print(f'np.allclose(H3, H4[..., 1], rtol={rtol}): {np.allclose(H3, H4[..., 1], rtol=rtol)}')





#%%
# Getting rows and columns indices

import matplotlib.pyplot as plt
import numpy as np

import deepmimo as dm

dataset = dm.load("asu_campus_3p5", matrices=["rx_pos", "tx_pos", "inter", "aoa_az"])

row_idxs = dataset.get_row_idxs(np.arange(40, 60))
col_idxs = dataset.get_col_idxs(np.arange(40, 60))

dataset_sub1 = dataset.subset(row_idxs)
dataset_sub2 = dataset.subset(col_idxs)

dataset.plot_coverage(dataset.los, title="Full dataset")
x_lim, y_lim = plt.xlim(), plt.ylim()

dataset_sub1.plot_coverage(dataset_sub1.los, title="Row subset")
plt.xlim(x_lim)
plt.ylim(y_lim)

dataset_sub2.plot_coverage(dataset_sub2.los, title="Column subset")
plt.xlim(x_lim)
plt.ylim(y_lim)

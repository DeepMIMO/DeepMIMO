# Generator

The generator module is the core of DeepMIMO. This module takes ray tracing scenarios saved in the DeepMIMO format, and generates channels. 

Below is an ascii diagram of how the simulations from the ray tracers are converted into DeepMIMO scenarios (by the converter module, following the DeepMIMO SPEC), and then loaded and used to generate channels (with the generator module).
```c++

+-----------------+     +-------------------+    +-------------------+
| WIRELESS INSITE |     |     SIONNA_RT     |    |       AODT        |
+--------+--------+     +---------+---------+    +---------+---------+
         |                        |                        |
         +------------------------+------------------------+
                                  |
                                  v
                         +------------------+
                         |   dm.convert()   |
                         +--------+---------+
                                  v
                         +------------------+
                         |    DEEPMIMO      |
                         |    SCENARIOS     |
                         +--------+---------+
                                  v
                      +-------------------------+
                      |   dataset = dm.load()   |
                      +-----------+-------------+
                                  v
                    +-----------------------------+
                    | dataset.compute_channels()  |
                    +-------------+---------------+
                                  v
                         +------------------+
                         |  dataset.plot()  |
                         +------------------+
```

Dependencies of the Generator Module:

```
generator/
  ├── core.py (Main generation functions)
  ├── channel.py (Channel computation)
  ├── dataset.py (Dataset classes)
  |    ├── geometry.py (Antenna array functions)
  |    └── ant_patterns.py (Antenna patterns)
  ├── visualization.py (Plotting functions)
  └── utils.py (Helper functions)
```

Additionally, the generator module depends on:
- `scene.py` for physical world representation
- `materials.py` for material properties
- `general_utils.py` for utility functions
- `api.py` for scenario management


## Load Dataset

```python
import deepmimo as dm

# Load a scenario
dataset = dm.load('asu_campus_3p5')
```

```{tip}
For detailed examples of loading, see the <a href="../manual_full.html#detailed-load">Detailed Load</a> Section of the DeepMIMO Mannual.
```

```{eval-rst}
.. autofunction:: deepmimo.generator.core.load
```


## Generate Channels

The `ChannelGenParameters` class manages parameters for MIMO channel generation.

```python
import deepmimo as dm

# Load a scenario
dataset = dm.load('asu_campus_3p5')

# Instantiate channel parameters
params = dm.ChannelGenParameters()

# Configure BS antenna array
params.bs_antenna.shape = np.array([8, 1])  # 8x1 array
params.bs_antenna.spacing = 0.5  # Half-wavelength spacing
params.bs_antenna.rotation = np.array([0, 0, 0])  # No rotation

# Configure UE antenna array
params.ue_antenna.shape = np.array([1, 1])  # Single antenna
params.ue_antenna.spacing = 0.5
params.ue_antenna.rotation = np.array([0, 0, 0])

# Configure OFDM parameters
params.ofdm.subcarriers = 512  # Number of subcarriers
params.ofdm.bandwidth = 10e6  # 10 MHz bandwidth

# Generate frequency-domain channels
params.freq_domain = True
channels = dataset.compute_channels(params)
```

```{tip}
For detailed examples of generating channels, see the <a href="../manual_full.html#channel-generation">Channel Generation</a> Section of the DeepMIMO Mannual.
```

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `bs_antenna.shape` | [8, 1] | BS antenna array dimensions |
| `bs_antenna.spacing` | 0.5 | BS antenna spacing (wavelengths) |
| `bs_antenna.rotation` | [0, 0, 0] | BS rotation angles (degrees) |
| `ue_antenna.shape` | [1, 1] | UE antenna array dimensions |
| `ue_antenna.spacing` | 0.5 | UE antenna spacing (wavelengths) |
| `ue_antenna.rotation` | [0, 0, 0] | UE rotation angles (degrees) |
| `ofdm.subcarriers` | 512 | Number of OFDM subcarriers |
| `ofdm.bandwidth` | 10e6 | OFDM bandwidth (Hz) |

```{eval-rst}
.. autoclass:: deepmimo.generator.channel.ChannelGenParameters
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autofunction:: deepmimo.generator.dataset.Dataset.compute_channels

```

## Dataset

The `Dataset` class represents a single dataset within DeepMIMO, containing transmitter, receiver, and channel information for a specific scenario configuration.

```python
import deepmimo as dm

# Load a dataset
dataset = dm.load('scenario_name')

# Access transmitter data
tx_locations = dataset.tx_locations
n_tx = len(dataset.tx_locations)

# Access receiver data
rx_locations = dataset.rx_locations
n_rx = len(dataset.rx_locations)

# Access channel data
channels = dataset.channels  # If already computed
```

### Core Properties

| Property       | Description                             | Dimensions    |
|----------------|-----------------------------------------|---------------|
| `rx_pos`       | Receiver locations                      | N x 3         |
| `tx_pos`       | Transmitter locations                   | 1 x 3         |
| `power`        | Path powers in dBm                      | N x P         |
| `phase`        | Path phases in degrees                  | N x P         |
| `delay`        | Path delays in seconds                  | N x P         |
| `aoa_az/aoa_el`| Angles of arrival (azimuth/elevation)   | N x P         |
| `aod_az/aod_el`| Angles of departure (azimuth/elevation) | N x P         |
| `inter`        | Path interaction indicators             | N x P         |
| `inter_pos`    | Path interaction positions              | N x P x I x 3 |

- N: number of receivers in the receiver set
- P: maximum number of paths
- I: maximum number of interactions along any path

The maximum number of paths and interactions are either configured by the load function or hardcoded to a absolute maximum value. 

### Computed Properties
| `channels` | ndarray | Channel matrices |
| `parameters` | dict | Dataset-specific parameters |
| `num_paths` | int | Number of paths generated for each user |
| `pathloss` | ndarray | Path loss values for each path |
| `aod_theta_rot` | ndarray | Rotated angles of departure in elevation |
| `aod_phi_rot` | ndarray | Rotated angles of departure in azimuth |
| `aoa_theta_rot` | ndarray | Rotated angles of arrival in elevation |
| `aoa_phi_rot` | ndarray | Rotated angles of arrival in azimuth |
| `fov` | dict | Field of view parameters |
| `grid_size` | tuple | Size of the grid for the dataset |
| `grid_spacing` | float | Spacing of the grid for the dataset |

### Sampling & Trimming
```python
# Get uniform indices
uniform_idxs = dataset.get_uniform_idxs([2,2])

# Trim dataset to have 1 every 2 samples, along x and y
dataset2 = dataset.subset(uniform_idxs)

# Example of dataset trimming
active_idxs = dataset2.get_active_idxs()

# Further trim the dataset down to include only users with channels 
# (typically outside buildings)
dataset2 = dataset.subset(uniform_idxs)
```

```{tip}
For detailed examples of sampling users from a dataset and creating subsets of a dataset, see the <a href="../manual_full.html#user-sampling">User Sampling</a> Section of the DeepMIMO Mannual.
```

### Plotting

```python
# Plot coverage
plot_coverage = dataset.plot_coverage()

# Plot rays
plot_rays = dataset.plot_rays()
```

```{tip}
For more details on the visualization functions, see the <a href="../manual_full.html#visualization">Visualization</a> Section of the DeepMIMO Mannual, and the <a href="visualization.html">Visualization API</a> section of this noteoobk.
```

### Dataset Class
```{eval-rst}
.. autoclass:: deepmimo.generator.dataset.Dataset
   :members:
   :undoc-members:
   :show-inheritance:
```


## MacroDataset

The `MacroDataset` class is a container for managing multiple datasets, providing unified access to their data. This is the default output of the dm.load() if there are multiple txrx pairs.

```python
# Access individual datasets
dataset = macro_dataset[0]  # First dataset
datasets = macro_dataset[1:3]  # Slice of datasets

# Iterate over datasets
for dataset in macro_dataset:
    print(f"Dataset has {len(dataset)} users")

# Batch operations
channels = macro_dataset.compute_channels()
```

```{eval-rst}
.. autoclass:: deepmimo.generator.dataset.MacroDataset
   :members:
   :undoc-members:
   :show-inheritance:
```

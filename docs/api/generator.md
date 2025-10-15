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
For detailed examples of loading, see the <a href="../manual_full.html#detailed-load">Detailed Load</a> Section of the DeepMIMO Manual.
```

```{eval-rst}
.. autofunction:: deepmimo.generator.core.load
```


## Generate Channels

The `ChannelParameters` class manages parameters for MIMO channel generation.

```python
import deepmimo as dm

# Load a scenario
dataset = dm.load('asu_campus_3p5')

# Instantiate channel parameters
params = dm.ChannelParameters()

# Configure BS antenna array
params.bs_antenna.shape = [8, 1]  # 8x1 array
params.bs_antenna.spacing = 0.5  # Half-wavelength spacing
params.bs_antenna.rotation = [0, 0, 0]  # No rotation

# Configure UE antenna array
params.ue_antenna.shape = [1, 1]  # Single antenna
params.ue_antenna.spacing = 0.5
params.ue_antenna.rotation = [0, 0, 0]

# Configure OFDM parameters
params.ofdm.subcarriers = 512  # Number of subcarriers
params.ofdm.bandwidth = 10e6  # 10 MHz bandwidth
params.ofdm.selected_subcarriers = [0]  # Which subcarriers to generate

# Generate frequency-domain channels
params.doppler = False
params.freq_domain = True
channels = dataset.compute_channels(params)
```

```{tip}
For detailed examples of generating channels, see the <a href="../manual_full.html#channel-generation">Channel Generation</a> Section of the DeepMIMO Manual.
```

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `bs_antenna.shape` | [8, 1] | BS antenna array dimensions (horizontal, vertical)|
| `bs_antenna.spacing` | 0.5 | BS antenna spacing (wavelengths) |
| `bs_antenna.rotation` | [0, 0, 0] | BS rotation angles (degrees around x,y,z) |
| `ue_antenna.shape` | [1, 1] | UE antenna array dimensions (horizontal, vertical)|
| `ue_antenna.spacing` | 0.5 | UE antenna spacing (wavelengths) |
| `ue_antenna.rotation` | [0, 0, 0] | UE rotation angles (degrees around x,y,z) |
| `ofdm.subcarriers` | 512 | Number of OFDM subcarriers |
| `ofdm.selected_subcarriers` | 512 | Indices of selected OFDM subcarriers |
| `ofdm.bandwidth` | 10e6 | OFDM bandwidth (Hz) |
| `freq_domain` | True | Boolean for generating the channel in frequency (OFDM) |
| `doppler` | False | Boolean for adding Doppler frequency shifts to the channel |

Note 1: Rotation angles follow the right-hand rule.
Note 2: The default orientation of an antenna panel is along the +X axis.

```{eval-rst}
.. autoclass:: deepmimo.generator.channel.ChannelParameters
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autofunction:: deepmimo.generator.dataset.Dataset.compute_channels

```

## Doppler

Doppler effects can be added to the generated channels (in time or frequency domain) in three different ways:
- Set Doppler directly: Manually set the Doppler frequencies per user (and optionally, per path)
- Set Speeds directly: Manually set the TX, RX or object speeds, which automatically computes Doppler frequencies
- Set Time Reference: Automatically compute TX, RX and object speeds across scenes (only works with Dynamic Datasets)

```{note}
For Doppler to be added to the channel, the parameter `doppler` must be set to `True` in the channel parameters.
```

For more details about working with datasets and its methods, see the [Dataset](#dataset) section below.

### Set Doppler

You can directly specify Doppler shifts in three ways:

```python
# Same Doppler shift for all users
dopplers1 = 10  # [Hz]
dataset.set_doppler(dopplers1)
dataset.compute_channels(dm.ChannelParameters(doppler=True))

# Different Doppler shift for different users
dopplers2 = np.random.randint(20, 51, size=(dataset.n_ue,))
dataset.set_doppler(dopplers2)
dataset.compute_channels(dm.ChannelParameters(doppler=True))

# Different Doppler shift for different users and paths
dopplers3 = np.random.randint(20, 51, size=(dataset.n_ue, dataset.max_paths))
dataset.set_doppler(dopplers3)
dataset.compute_channels(dm.ChannelParameters(doppler=True))
```

### Set Velocities

You can set velocities for receivers, transmitters, and objects in the scene. This will in turn add doppler to the paths that interact with those entities:

```python
# Set rx velocities manually (same for all users)
dataset.rx_vel = [5, 0, 0]  # (x, y, z) [m/s]

# Set rx velocities manually (different per users)
min_speed, max_speed = 0, 10
random_velocities = np.zeros((dataset.n_ue, 3))
random_velocities[:, :2] = np.random.uniform(min_speed, max_speed, size=(dataset.n_ue, 2))
dataset.rx_vel = random_velocities  # Note: z = 0 assumes users at ground level

# Set tx velocities manually
dataset.tx_vel = [0, 0, 0]

# Set object velocities manually
dataset.set_obj_vel(obj_idx=[1, 3, 6], vel=[[0, 5, 0], [0, 5, 6], [0, 0, 3]])
# Note: object indices should match the indices/ids in dataset.scene.objects

dataset.compute_channels(dm.ChannelParameters(doppler=True))
```

### Set Timestamps

For Dynamic Datasets (i.e. multi-scene datasets), setting timestamps will automatically compute velocities for the receivers, transmitters or objects that move across scenes:

```python
# Uniform snapshots
dataset.set_timestamps(10)  # seconds between scenes

# Non-uniform snapshots
times = [0, 1.5, 2.3, 4.4, 5.8, 7.1, 8.9, 10.2, 11.7, 13.0]
dataset.set_timestamps(times)  # timestamps of each scene
```

After setting timestamps, you can access the computed velocities:
```python
print(f'timestamps: {dataset.timestamps}')
print(f'rx_vel: {dataset.rx_vel}')
print(f'tx_vel: {dataset.tx_vel}')
print(f'obj_vel: {[obj.vel for obj in dataset.scene.objects]}')
```

```{note}
Setting timestamps requires a Dynamic Dataset. The Dynamic dataset is the exact same as 
a normal dataset, but instead of providing a folder with the ray tracing results directly inside, we provide a folder with many of such folders inside, one for each scene. See more in the [Dataset](#dataset) and DynamicDataset sections below.
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
For detailed examples of sampling users from a dataset and creating subsets of a dataset, see the <a href="../manual_full.html#user-sampling">User Sampling</a> Section of the DeepMIMO Manual.
```

### Plotting

```python
# Plot coverage
plot_coverage = dataset.plot_coverage()

# Plot rays
plot_rays = dataset.plot_rays()
```

```{tip}
For more details on the visualization functions, see the <a href="../manual_full.html#visualization">Visualization</a> Section of the DeepMIMO Manual, and the <a href="visualization.html">Visualization API</a> section of this noteoobk.
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

## DynamicDataset

The `DynamicDataset` class extends `MacroDataset` to handle multiple time snapshots of a scenario. Each snapshot is represented by a `MacroDataset` instance, allowing you to track changes in the environment over time.

```python
# Convert a dynamic dataset
dm.convert(rt_folder) # rt_folder must contain individual folders of ray tracing results

# Load a dynamic dataset
dynamic_dataset = dm.load('scenario_name')  # Returns DynamicDataset if multiple time snapshots exist

# Access individual time snapshots
snapshot = dynamic_dataset[0]  # First time snapshot
snapshots = dynamic_dataset[1:3]  # Slice of time snapshots

# Access basic properties
print(f"Number of scenes: {len(dynamic_dataset)}")  # or dynamic_dataset.n_scenes
print(f"Scene names: {dynamic_dataset.names}")
```

```{eval-rst}
.. autoclass:: deepmimo.generator.dataset.DynamicDataset
   :members:
   :undoc-members:
   :show-inheritance:
```




# Generator

The generator module handles MIMO channel generation from ray tracing data. This module computes channel matrices from geometric ray paths and antenna array configurations.

Below is an ascii diagram of how the simulations from the ray tracers are converted into DeepMIMO scenarios (by the converter module, following the DeepMIMO SPEC), and then loaded and used to generate channels.

```text
┌───────────────────┐     ┌────────────────┐     ┌────────────────┐
│ WIRELESS INSITE   │ ──▶ │   SIONNA_RT    │ ──▶ │      AODT      │
└─────────┬─────────┘     └───────┬────────┘     └───────┬────────┘
          │                       │                      │
          └──────────────┬────────┴──────────────┬───────┘
                         ▼                       ▼
                   ┌───────────────┐       ┌───────────────┐
                   │  dm.convert() │       │  dm.convert() │
                   └──────┬────────┘       └──────┬────────┘
                          │                       │
                          └──────────────┬────────┘
                                         ▼
                               ┌────────────────┐
                               │   DEEPMIMO     │
                               │   SCENARIOS    │
                               └──────┬─────────┘
                                      ▼
                        ┌────────────────────────┐
                        │  dataset = dm.load()   │
                        └──────────┬─────────────┘
                                   ▼
                   ┌──────────────────────────────┐
                   │ dataset.compute_channels()   │
                   └──────────────┬───────────────┘
                                  ▼
                         ┌────────────────┐
                         │  dataset.plot()│
                         └────────────────┘
```

DeepMIMO Package Structure:

```
deepmimo/
  ├── core/ (Data models: Scene, Materials, RayTracingParameters, TxRxSet)
  ├── datasets/ (Dataset classes, load/generate, visualization, sampling)
  ├── generator/ (Channel computation)
  │    ├── channel.py (MIMO channel generation)
  │    ├── geometry.py (Antenna array functions)
  │    └── ant_patterns.py (Antenna patterns)
  ├── api/ (Database operations: download, upload, search)
  ├── utils/ (Utilities: I/O, scenarios, geometry)
  └── integrations/ (Export to other simulators)
```


## Channel Parameters

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

!!! tip "Channel generation examples"
    See the <a href="../manual/#channel-generation">Channel Generation</a> section of the DeepMIMO Manual for more examples.

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

::: deepmimo.generator.channel.ChannelParameters

::: deepmimo.datasets.dataset.Dataset.compute_channels

## Compute Channels

Once a dataset is loaded, channels can be computed using the `compute_channels()` method:

```python
import deepmimo as dm

# Load dataset
dataset = dm.load('asu_campus_3p5')

# Configure channel parameters
params = dm.ChannelParameters()
params.bs_antenna.shape = [8, 1]
params.ue_antenna.shape = [1, 1]

# Compute channels
channels = dataset.compute_channels(params)
```

::: deepmimo.datasets.dataset.Dataset.compute_channels


## Doppler

Doppler effects can be added to the generated channels (in time or frequency domain) in three different ways:
- Set Doppler directly: Manually set the Doppler frequencies per user (and optionally, per path)
- Set Speeds directly: Manually set the TX, RX or object speeds, which automatically computes Doppler frequencies
- Set Time Reference: Automatically compute TX, RX and object speeds across scenes (only works with Dynamic Datasets)

!!! note
    To add Doppler to the channel, set `doppler=True` in the channel parameters.

For more details about working with datasets and its methods, see the [Datasets API](datasets.md).

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

!!! note
    Setting timestamps requires a Dynamic Dataset. Provide a folder containing one subfolder per scene (ray-tracing results per snapshot). See the [Datasets API](datasets.md) for more information.


## Antenna Patterns

The generator module supports custom antenna patterns for more realistic channel modeling.

::: deepmimo.generator.ant_patterns.AntennaPattern


## Geometry Functions

Geometric functions for beamforming and array processing:

```python
import deepmimo as dm

# Generate steering vector
steering_vector = dm.steering_vec(
    array_shape=[8, 1],
    angles=[45, 30],  # [azimuth, elevation]
    spacing=0.5
)
```

::: deepmimo.generator.geometry.steering_vec

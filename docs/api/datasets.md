# Datasets

The datasets module handles dataset loading, generation, manipulation, and visualization.

```
deepmimo/datasets/
  ├── dataset.py (Dataset classes: Dataset, MacroDataset, DynamicDataset)
  ├── load.py (Load datasets from scenarios)
  ├── generate.py (Generate datasets with channel computation)
  ├── array_wrapper.py (DeepMIMOArray for array manipulation)
  ├── visualization.py (Plotting functions)
  ├── summary.py (Dataset summary functions)
  └── sampling.py (User sampling utilities)
```

## Load Dataset

```python
import deepmimo as dm

# Load a scenario
dataset = dm.load('asu_campus_3p5')
```

!!! tip "Detailed load examples"
    See the <a href="../manual/#detailed-load">Detailed Load</a> section of the DeepMIMO Manual for more examples.

::: deepmimo.datasets.load.load


## Generate Dataset

The `generate()` function combines loading and channel computation in a single step.

```python
import deepmimo as dm

# Generate dataset with custom parameters
dataset = dm.generate(
    'asu_campus_3p5',
    load_params={'tx_sets': [0], 'rx_sets': [0]},
    ch_params={'bs_antenna': {'shape': [8, 1]}}
)
```

::: deepmimo.datasets.generate.generate


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

### Sampling & Trimming

Unified index selection (dispatcher):
```python
# Uniform sampling on the grid
uniform_idxs = dataset.get_idxs('uniform', steps=[2, 2])

# Active users (paths > 0)
active_idxs = dataset.get_idxs('active')

# Users along a line
linear_idxs = dataset.get_idxs('linear', start_pos=[0,0,0], end_pos=[100,0,0], n_steps=50)
```

Create trimmed datasets (physical trimming):
```python
# Subset by indices
dataset2 = dataset.trim(idxs=uniform_idxs)

# Apply FoV trimming (uses rotated angles)
dataset_fov = dataset.trim(bs_fov=[90, 90])  # optional: ue_fov=[...]

# Combine trims efficiently (order: idxs -> FoV -> path depth -> path type)
dataset_t = dataset.trim(
    idxs=active_idxs,
    bs_fov=[90, 90],
    path_depth=1,
    path_types=['LoS', 'R']
)
```

!!! tip "User sampling and subsets"
    See the <a href="../manual/#user-sampling">User Sampling</a> section of the DeepMIMO Manual for examples of sampling users and creating subsets.

### Plotting

```python
# Plot coverage
plot_coverage = dataset.plot_coverage()

# Plot rays
plot_rays = dataset.plot_rays()
```

!!! tip "Visualization details"
    For visualization examples, see the <a href="../manual/#visualization">Visualization</a> section of the manual and the <a href="visualization.html">Visualization API</a>.

### Dataset Class
::: deepmimo.datasets.dataset.Dataset


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

::: deepmimo.datasets.dataset.MacroDataset

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

::: deepmimo.datasets.dataset.DynamicDataset


## DeepMIMOArray

The `DeepMIMOArray` class provides convenient array manipulation for dataset properties.

::: deepmimo.datasets.array_wrapper.DeepMIMOArray


## Summary Functions

Get dataset summary statistics:

```python
import deepmimo as dm

# Get summary
summary_dict = dm.summary('scenario_name')

# Plot summary
dm.plot_summary('scenario_name')
```

::: deepmimo.datasets.summary.summary

::: deepmimo.datasets.summary.plot_summary


## Sampling Functions

Utilities for selecting user indices:

```python
# Uniform sampling
idxs = dm.get_uniform_idxs(n_ue=1000, grid_size=[10, 10], steps=[2, 2])

# Linear sampling
idxs = dm.get_linear_idxs(rx_pos, start_pos=[0,0,0], end_pos=[100,0,0], n_steps=50)

# Grid sampling
idxs = dm.get_grid_idxs(grid_size=[10, 10], rows=[0, 1, 2], cols=[0, 1, 2])

# Position limits
idxs = dm.get_idxs_with_limits(rx_pos, x_min=0, x_max=100, y_min=0, y_max=100)
```

::: deepmimo.datasets.sampling.get_uniform_idxs

::: deepmimo.datasets.sampling.get_linear_idxs

::: deepmimo.datasets.sampling.get_grid_idxs

::: deepmimo.datasets.sampling.get_idxs_with_limits

::: deepmimo.datasets.sampling.dbw2watt


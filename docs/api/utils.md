# Utilities

DeepMIMO provides utility modules organized by function:
```
deepmimo/utils/
  ├── scenarios.py (Scenario management)
  ├── io.py (File I/O operations)
  ├── geometry.py (Geometric calculations)
  ├── data_structures.py (Custom data structures)
  └── dict_utils.py (Dictionary utilities)

deepmimo/datasets/sampling.py
  ├── Unit Conversion (dbw2watt)
  └── Position Sampling (get_uniform_idxs, get_linear_idxs, get_grid_idxs)

deepmimo/generator/geometry.py
  └── Beamforming (steering_vec)
```

## Scenario Management

```python
import deepmimo as dm

# Get available scenarios
scenarios = dm.get_available_scenarios()

# Get scenario paths
folder = dm.get_scenario_folder('scenario_name')
params = dm.get_params_path('scenario_name')
```

::: deepmimo.utils.scenarios.get_scenario_folder

::: deepmimo.utils.scenarios.get_params_path

::: deepmimo.utils.scenarios.get_available_scenarios

## User Sampling

```python
# Get uniform sampling indices
idxs = dm.get_uniform_idxs(
    n_ue=1000,           # Number of users
    grid_size=[10, 10],  # Grid dimensions
    steps=[2, 2]         # Sampling steps
)

# Get indices for specific rows in a grid dataset (if it's a grid)
idxs = dataset.get_row_idxs(np.arange(40, 60))  # Get rows 40-59

# Get indices for specific columns in a grid dataset (if it's a grid)
idxs = dataset.get_col_idxs(np.arange(40, 60))  # Get columns 40-59

# Get positions within limits
idxs = dm.get_idxs_with_limits(
    data_pos,
    x_min=0, x_max=100,
    y_min=0, y_max=100,
    z_min=0, z_max=50
)

# Create linear sampling along a path
idxs = dm.get_linear_idxs(
    rx_pos,               # Receiver positions [N, 3]
    start_pos=[0, 0, 0],  # Start position
    end_pos=[100, 0, 0],  # End position
    n_steps=100           # Number of samples
)
```

!!! tip "User sampling examples"
    See the <a href="../manual/#user-sampling">User Sampling</a> section of the DeepMIMO Manual for examples.

::: deepmimo.datasets.sampling.get_uniform_idxs

::: deepmimo.datasets.sampling.get_idxs_with_limits

::: deepmimo.datasets.sampling.get_grid_idxs

::: deepmimo.datasets.sampling.get_linear_idxs


!!! tip "User sampling examples"
    See the <a href="../manual/#user-sampling">User Sampling</a> section of the DeepMIMO Manual for examples.

## Beamforming

!!! tip "Beamforming examples"
    See the <a href="../manual/#beamforming">Beamforming</a> section of the DeepMIMO Manual for examples.

::: deepmimo.generator.geometry.steering_vec

## Unit Conversions

```python
# Convert dBW to Watts
power_w = dm.dbw2watt(power_dbw)
```

::: deepmimo.datasets.sampling.dbw2watt

## Zip & Unzip
```python
# File compression
dm.zip('path/to/folder')
dm.unzip('path/to/file.zip')
```

::: deepmimo.utils.io.zip

::: deepmimo.utils.io.unzip

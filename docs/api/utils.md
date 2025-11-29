# Utilities

DeepMIMO provides two utility modules:
```
deepmimo/general_utils.py
  ├── Scenario Management (get_available_scenarios, get_params_path, get_scenario_folder)
  └── Zip & Unzip (zip, unzip)

deepmimo/generator/generator_utils.py
  ├── Unit Conversion (dbw2watt)
  └── Position Sampling (get_uniform_idxs, LinearPath)

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

```{eval-rst}

.. autofunction:: deepmimo.general_utils.get_scenario_folder

.. autofunction:: deepmimo.general_utils.get_params_path

.. autofunction:: deepmimo.general_utils.get_available_scenarios
```

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

# Create linear path through dataset
path = dm.LinearPath(
    rx_pos,               # Receiver positions
    first_pos=[0, 0, 0],  # Start position
    last_pos=[100, 0, 0], # End position
    res=1.0,              # Spatial resolution
    n_steps=100           # Number of steps
)

# Access path data
positions = path.pos
indices = path.idxs
```

```{tip}
See the <a href="../manual_full.html#user-sampling">User Sampling Section</a> of the DeepMIMO Manual for examples.
```

```{eval-rst}

.. autofunction:: deepmimo.generator.generator_utils.get_uniform_idxs

.. autofunction:: deepmimo.generator.generator_utils.get_idxs_with_limits

.. autofunction:: deepmimo.generator.generator_utils.get_grid_idxs

.. autoclass:: deepmimo.generator.generator_utils.LinearPath
   :members:
   :undoc-members:
   :show-inheritance:

```


```{tip}
See the User Sampling Section of the DeepMIMO Manual for examples.
```

## Beamforming

```{tip}
See the <a href="../manual_full.html#beamforming">Beamforming Section</a> of the DeepMIMO Manual for examples.
```

```{eval-rst}

.. autofunction:: deepmimo.generator.geometry.steering_vec
```

## Unit Conversions

```python
# Convert dBW to Watts
power_w = dm.dbw2watt(power_dbw)
```

```{eval-rst}
.. autofunction:: deepmimo.generator.generator_utils.dbw2watt
```

## Zip & Unzip
```python
# File compression
dm.zip('path/to/folder')
dm.unzip('path/to/file.zip')
```

```{eval-rst}
.. autofunction:: deepmimo.general_utils.zip

.. autofunction:: deepmimo.general_utils.unzip
```

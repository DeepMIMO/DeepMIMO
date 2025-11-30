# Visualization

The visualization module provides tools for visualizing DeepMIMO datasets, including coverage maps, ray paths, and channel characteristics.

```{tip}
Many more visualization examples are available in the <a href="../manual_full.html#visualization">Visualization Section</a> of the DeepMIMO Manual.
```

## Coverage Maps

Create coverage map visualization for user positions.

```python
import deepmimo as dm

# Load a scenario and select the txrx pair with a receiver grid
dataset = dm.load('asu_campus_3p5')[0]

# Example plot LoS status with default settings
dm.plot_coverage(dataset.rx_pos, dataset.los)

# Customize visualization
ax, cbar = dm.plot_coverage(
    rxs,                   # User positions (N×3)
    cov_map,               # Coverage values
    dpi=100,               # Plot resolution
    figsize=(6, 4),        # Figure size
    cbar_title='Power',    # Colorbar title
    title=True,            # Show title
    scat_sz=0.5,           # Marker size
    bs_pos=bs_position,    # Base station position
    bs_ori=bs_orientation, # Base station orientation
    legend=True,           # Show legend
    proj_3D=False,         # 2D/3D projection
    cmap='viridis'         # Color map
)

# Plot multiple features
features = ['aoa_az', 'aoa_el', 'aod_az', 'aod_el', 
            'delay', 'power', 'phase', 'los', 'num_paths']

for key in features:
    plt_var = dataset[key][:,0] if dataset[key].ndim == 2 else dataset[key]
    dataset.plot_coverage(plt_var, title=key)  # wrapper to plot_coverage(dataset.rx_pos)
```

::: deepmimo.generator.visualization.plot_coverage

## Rays
Plot ray paths between transmitter and receiver with interaction points.

```python
import deepmimo as dm

# Load a scenario and select the txrx pair with a receiver grid
dataset = dm.load('asu_campus_3p5')[0]

# Plot ray paths
fig, ax = dm.plot_rays(
    dataset.rx_pos,      # Receiver location
    dataset.tx_pos,      # Transmitter location
    dataset.inter_pos,   # Interaction positions
    dataset.inter,       # Interaction types
    figsize=(10, 8),     # Figure size
    dpi=100,             # Plot resolution
    proj_3D=True,        # 3D projection
    color_by_type=True   # Color by interaction type
)

# Plot ray paths with wrapper
dataset.plot_rays(10)  # user index
```

::: deepmimo.generator.visualization.plot_rays

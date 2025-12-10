# Core Models

The core module provides fundamental data model classes for representing wireless scenarios, including physical scenes, materials, ray tracing parameters, and transmitter/receiver configurations.

```
deepmimo/core/
  ├── scene.py (Physical environment representation)
  ├── materials.py (Material properties)
  ├── rt_params.py (Ray tracing parameters)
  └── txrx.py (Transmitter/receiver configurations)
```

## Scene

The `Scene` class represents a complete physical environment containing multiple objects.

```python
import deepmimo as dm

# Create a new scene
scene = dm.Scene()

# Add objects
scene.add_object(building)
scene.add_objects([tree1, tree2])

# Get objects by category
buildings = scene.get_objects(label='buildings')
metal_objects = scene.get_objects(material=1)  # material_id = 1

# 3D visualization
scene.plot(mode='faces')
```

For detailed information about Scene and related classes (BoundingBox, Face, PhysicalElement, PhysicalElementGroup), see the [Scene API Reference](scene.md).

::: deepmimo.core.scene.Scene


## Materials

The `Material` and `MaterialList` classes manage material properties for electromagnetic simulations.

```python
import deepmimo as dm

# Create a material
material = dm.Material(
    id=1,
    name='Concrete',
    permittivity=4.5,
    conductivity=0.02,
    scattering_coefficient=0.2
)

# Create material list
materials = dm.MaterialList()
materials.add_materials([concrete, glass, metal])
```

For detailed information about Materials, see the [Materials API Reference](materials.md).

::: deepmimo.core.materials.Material

::: deepmimo.core.materials.MaterialList


## Ray Tracing Parameters

The `RayTracingParameters` class configures ray tracing simulation settings.

```python
import deepmimo as dm

# Create ray tracing parameters
rt_params = dm.RayTracingParameters(
    max_num_paths=25,
    max_num_interactions=5,
    min_path_gain=-160.0
)
```

::: deepmimo.core.rt_params.RayTracingParameters


## Transmitter/Receiver Configuration

Classes for managing transmitter and receiver configurations in scenarios.

```python
import deepmimo as dm

# Get available TX/RX sets for a scenario
txrx_sets = dm.get_txrx_sets('scenario_name')

# Get available TX/RX pairs
pairs = dm.get_txrx_pairs('scenario_name')

# Print available pairs
dm.print_available_txrx_pair_ids('scenario_name')
```

::: deepmimo.core.txrx.TxRxSet

::: deepmimo.core.txrx.TxRxPair

::: deepmimo.core.txrx.get_txrx_sets

::: deepmimo.core.txrx.get_txrx_pairs

::: deepmimo.core.txrx.print_available_txrx_pair_ids


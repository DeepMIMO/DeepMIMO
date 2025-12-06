# Materials

The materials module provides classes for representing and managing material properties in wireless environments, including electromagnetic and scattering characteristics.

## Material
The `Material` class represents a single material with its electromagnetic and scattering properties.

```python
import deepmimo as dm

# Create a material
material = dm.Material(
    id=1,
    name='Concrete',
    permittivity=4.5,
    conductivity=0.02,
    scattering_model='directive',
    scattering_coefficient=0.2,
    cross_polarization_coefficient=0.1,
    alpha_r=4.0,
    alpha_i=4.0,
    lambda_param=0.5,
    roughness=0.001,
    thickness=0.3,
    vertical_attenuation=0.5,
    horizontal_attenuation=0.3
)
```


<!-- ```{eval-rst}
.. autoclass:: deepmimo.materials.Material
   :members:
   :undoc-members:
   :show-inheritance:
``` -->

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | int | Unique identifier |
| `name` | str | Material name |
| `permittivity` | float | Relative permittivity |
| `conductivity` | float | Conductivity (S/m) |
| `scattering_coefficient` | float | Fraction of incident fields scattered (0-1) |
| `cross_polarization_coefficient` | float | Fraction of scattered field cross-polarized (0-1) |
| `roughness` | float | Surface roughness (m) |
| `thickness` | float | Material thickness (m) |
| `vertical_attenuation` | float | Vertical attenuation (dB/m) |
| `horizontal_attenuation` | float | Horizontal attenuation (dB/m) |

The Material class supports different scattering models:

| Model | Description | Parameters |
|-------|-------------|------------|
| `SCATTERING_NONE` | No scattering | None |
| `SCATTERING_LAMBERTIAN` | Lambertian scattering | `scattering_coefficient` |
| `SCATTERING_DIRECTIVE` | Directive scattering | `alpha_r`, `alpha_i`, `lambda_param` |

```python
# Configure directive scattering
material.scattering_model = Material.SCATTERING_DIRECTIVE
material.alpha_r = 4.0  # Forward lobe width
material.alpha_i = 4.0  # Backward lobe width
material.lambda_param = 0.5  # Forward/backward ratio
```

The field `vertical_attenuation` and `horizontal_attenuation` describe the attenuation properties of the material, and are mainly used in folliage. 

## MaterialList
The `MaterialList` class manages collections of materials and provides database functionality.

```python
# Create material list
materials = dm.MaterialList()

# Add materials
materials.add_materials([concrete, glass, metal])

# Access materials
concrete = materials[0]  # By index
metal_objects = materials[1:3]  # Slice

# Export/import
materials_dict = materials.to_dict()
materials = dm.MaterialList.from_dict(materials_dict)

# Material management

dataset = dm.load('asu_campus_3p5')[0]

materials = dataset.materials

# Get materials used by buildings
buildings = scene.get_objects(label='buildings')
building_materials = buildings.get_materials()

# Get objects with a certain material
objects_with_material = dataset.scene.get_objects(material=building_materials[0])
```

::: deepmimo.materials.MaterialList

!!! tip "Materials examples"
    See the <a href="../manual/#scene-materials">Materials</a> section of the DeepMIMO Manual for examples.

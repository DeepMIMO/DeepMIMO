# Scene

The `scene` module provides classes for representing and managing physical objects in a wireless environment, including buildings, terrain, vegetation, and other structures that affect wireless propagation.

```
scene.py
  ├── BoundingBox (3D bounds)
  ├── Face (Surface representation)
  ├── PhysicalElement (Individual objects)
  ├── PhysicalElementGroup (Object collections)
  └── Scene (Complete environment)
```

The `Scene` class acts as a container for multiple `PhysicalElement` objects, 
each representing a distinct object in the environment. Each `PhysicalElement` is 
composed of `Face` objects, which define the surfaces of the element and are associated 
with materials. The `BoundingBox` class provides spatial boundaries for these elements. 
Together, these components allow for the representation and manipulation of complex environments, 
with functionalities for plotting and material management integrated into the scene.


The scene module depends on:
- `materials.py` for material properties
- `general_utils.py` for utility functions
- NumPy for geometric computations
- Matplotlib for visualization


## BoundingBox

Dataclass for bounding boxes.

```python
# Create a bounding box
bbox = dm.BoundingBox(x_min=0, x_max=10, y_min=0, y_max=5, z_min=0, z_max=3)
```

| Property | Description |
|----------|-------------|
| `x_min`  | Minimum x-coordinate |
| `x_max`  | Maximum x-coordinate |
| `y_min`  | Minimum y-coordinate |
| `y_max`  | Maximum y-coordinate |
| `z_min`  | Minimum z-coordinate |
| `z_max`  | Maximum z-coordinate |
| `width`  | Width (X dimension) of the bounding box |
| `length` | Length (Y dimension) of the bounding box |
| `height` | Height (Z dimension) of the bounding box |

<!-- ```{eval-rst}
.. autoclass:: deepmimo.scene.BoundingBox
   :members:
   :undoc-members:
   :show-inheritance:
``` -->

## Face

The `Face` class represents a single face (surface) of a physical object.

```python
# Create a face
face = dm.Face(
    vertices=vertices,  # Array of vertex coordinates
    material_idx=1  # Material index
)
```

::: deepmimo.scene.Face

## PhysicalElement

The `PhysicalElement` class represents individual physical objects in the scene.

There are standard categories for physical elements, used only to organize them:

| Category | Description | Example Objects |
|----------|-------------|-----------------|
| `buildings` | Building structures | Houses, offices |
| `terrain` | Ground surfaces | Ground, hills |
| `vegetation` | Plant life | Trees, bushes |
| `floorplans` | Indoor layouts | Walls, rooms |
| `objects` | Other items | Cars, signs |

```python
# Create a physical object
element = dm.PhysicalElement(
    faces=faces,  # List of Face objects
    object_id=1,  # Unique identifier
    label='buildings',  # Object category
    name='Building 1'  # Optional name
)

# Access properties
height = element.height
volume = element.volume
position = element.position
```

::: deepmimo.scene.PhysicalElement

## PhysicalElementGroup

The `PhysicalElementGroup` class manages collections of physical objects.

```python
# Create a group
group = dm.PhysicalElementGroup(objects)

# Filter objects
buildings = group.get_objects(label='buildings')
metal = group.get_objects(material=1)

# Access objects
first = group[0]
subset = group[1:3]
```

::: deepmimo.scene.PhysicalElementGroup

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

# Export scene data to dictionary
metadata_dict = scene.export_data()

# Load scene from dictionary
scene = dm.Scene.from_data(metadata_dict)

# 3D visualization
scene.plot(mode='faces')  # Use convex hull representation
scene.plot(mode='tri_faces')  # Use triangular representation

# 2D top-down view
scene.plot(proj_3D=False)
```

::: deepmimo.scene.Scene


# Core Module Capabilities

This document details the capabilities of the DeepMIMO core module, which provides fundamental data structures for representing physical scenes, materials, ray-tracing parameters, and transmitter/receiver configurations.

## Overview

The core module provides four main components:
- **Scene**: Physical geometry representation (buildings, terrain, objects)
- **Materials**: Electromagnetic and scattering properties
- **Ray Tracing Parameters**: Simulation configuration
- **TX/RX Sets**: Transmitter and receiver configurations

---

## Scene Classes

### Scene

Top-level container for physical environment geometry.

| Feature | Support | Notes |
|---|---|---|
| **Object Management** | | |
| Add single object | ✓ | `add_object()` |
| Add multiple objects | ✓ | `add_objects()` |
| Get object by ID | ✓ | Index by ID |
| Get object by name | ✓ | `get_object_by_name()` |
| Remove objects | ✓ | By ID or name |
| Object iteration | ✓ | Iterable container |
| Object count | ✓ | `len(scene)` |
| | | |
| **Object Categories** | | |
| Buildings | ✓ | `CAT_BUILDINGS` |
| Terrain | ✓ | `CAT_TERRAIN` |
| Vegetation | ✓ | `CAT_VEGETATION` |
| Floorplans | ✓ | `CAT_FLOORPLANS` |
| Generic objects | ✓ | `CAT_OBJECTS` |
| Custom categories | ✓ | String labels |
| | | |
| **Data Export/Import** | | |
| Export to files | ✓ | `export_data()` - saves vertices/objects |
| Load from files | ✓ | `from_data()` - loads scene |
| NumPy arrays | ✓ | Vertices stored as arrays |
| Pickle format | ✓ | Serialization support |
| | | |
| **Bounding Box** | | |
| GPS coordinates | ✓ | (min_lat, min_lon, max_lat, max_lon) |
| Cartesian coordinates | ✓ | (min_x, min_y, max_x, max_y) |
| Z-axis bounds | ✓ | (min_z, max_z) |
| Automatic calculation | ✓ | From object vertices |
| | | |
| **Visualization** | | |
| 3D plotting | ✓ | `plot()` - matplotlib 3D |
| Color by category | ✓ | Different colors per category |
| TX/RX markers | ✓ | Optional position overlays |
| Custom color schemes | ✓ | Per-category colors |
| Interactive controls | ✓ | Matplotlib interactive mode |
| | | |
| **File Operations** | | |
| Scene vertices file | ✓ | `scene_vertices.pkl` |
| Scene objects file | ✓ | `scene_objects.pkl` |
| Material indices | ✓ | Per-object material mapping |

### PhysicalElement

Represents a single physical object (building, terrain patch, etc.).

| Feature | Support | Notes |
|---|---|---|
| **Properties** | | |
| Object ID | ✓ | Unique identifier |
| Object name | ✓ | String label |
| Category label | ✓ | Building/terrain/vegetation/etc |
| Face list | ✓ | Collection of Face objects |
| | | |
| **Face Management** | | |
| Add single face | ✓ | `add_face()` |
| Add multiple faces | ✓ | `add_faces()` |
| Get face by index | ✓ | Indexed access |
| Face iteration | ✓ | Iterable container |
| Face count | ✓ | `len(element)` |
| | | |
| **Geometric Operations** | | |
| Get all vertices | ✓ | `get_vertices()` |
| Vertex deduplication | ✓ | Unique vertex extraction |
| Bounding box | ✓ | Min/max coordinates |
| | | |
| **Serialization** | | |
| To dictionary | ✓ | `to_dict()` |
| From dictionary | ✓ | `from_dict()` |

### Face

Represents a single face (polygon) of a physical object.

| Feature | Support | Notes |
|---|---|---|
| **Geometry** | | |
| Vertex list | ✓ | List of (x, y, z) tuples |
| Arbitrary polygon | ✓ | Not limited to triangles |
| Convex faces | ✓ | Preferred format |
| Triangular faces | ✓ | Also supported |
| | | |
| **Material Association** | | |
| Material index | ✓ | Integer index into material list |
| Per-face materials | ✓ | Different materials per face |
| | | |
| **Operations** | | |
| Vertex count | ✓ | `len(face.vertices)` |
| Vertex access | ✓ | Indexed/iterable |
| | | |
| **Utilities** | | |
| Convex hull generation | ✓ | `get_object_faces()` with `fast=True` |
| Triangle mesh support | ✓ | `get_object_faces()` with `fast=False` |

### BoundingBox

Represents spatial bounds of scenes or objects.

| Feature | Support | Notes |
|---|---|---|
| **Coordinate Systems** | | |
| GPS coordinates | ✓ | (min_lat, min_lon, max_lat, max_lon) |
| Cartesian 2D | ✓ | (min_x, min_y, max_x, max_y) |
| Cartesian 3D | ✓ | Includes (min_z, max_z) |
| | | |
| **Operations** | | |
| Point containment | ✓ | Check if point inside box |
| Box intersection | ✓ | Check box overlap |
| Volume calculation | ✓ | Compute box volume |

---

## Material Classes

### Material

Represents electromagnetic and scattering properties of materials.

| Feature | Support | Notes |
|---|---|---|
| **Basic Properties** | | |
| Material ID | ✓ | Unique identifier |
| Material name | ✓ | String label |
| Relative permittivity | ✓ | Dielectric constant |
| Conductivity | ✓ | In Siemens/meter |
| | | |
| **Scattering Properties** | | |
| Scattering model | ✓ | None/Lambertian/Directive |
| Scattering coefficient | ✓ | 0-1 range |
| Cross-polarization | ✓ | Coefficient |
| Directive alpha_r | ✓ | Real scattering exponent |
| Directive alpha_i | ✓ | Imaginary scattering exponent |
| Directive lambda | ✓ | Forward/backward ratio |
| | | |
| **Physical Properties** | | |
| Surface roughness | ✓ | In meters |
| Material thickness | ✓ | In meters |
| | | |
| **Foliage Properties** | | |
| Vertical attenuation | ✓ | In dB/m |
| Horizontal attenuation | ✓ | In dB/m |
| | | |
| **ITU-R P.2040 Model** | | |
| Frequency-dependent ε | ✓ | ε = a + b·f^c + j·d·f^c |
| Parameter a | ✓ | Real constant term |
| Parameter b | ✓ | Real frequency coefficient |
| Parameter c | ✓ | Frequency exponent |
| Parameter d | ✓ | Imaginary coefficient |
| | | |
| **Serialization** | | |
| To dictionary | ✓ | `asdict()` via dataclass |
| From dictionary | ✓ | Constructor from dict |
| To tuple | ✓ | `astuple()` via dataclass |

### MaterialList

Container for managing collections of materials.

| Feature | Support | Notes |
|---|---|---|
| **Container Operations** | | |
| Add single material | ✓ | `add_material()` |
| Add multiple materials | ✓ | `add_materials()` |
| Get by index | ✓ | `material_list[idx]` |
| Get by list of indices | ✓ | Returns new MaterialList |
| Length | ✓ | `len(material_list)` |
| Iteration | ✓ | Iterate over materials |
| | | |
| **Property Access** | | |
| Attribute propagation | ✓ | `material_list.permittivity` returns list |
| Batch property access | ✓ | Access property for all materials |
| | | |
| **Search and Filter** | | |
| Find by name | ✓ | `find_by_name()` |
| Find by ID | ✓ | `find_by_id()` |
| Filter by property | ✓ [a] | Via list comprehension |
| | | |
| **Serialization** | | |
| To dictionary | ✓ | `to_dict()` |
| From dictionary | ✓ | `from_dict()` |
| JSON compatible | ✓ | Via dict serialization |
| | | |
| **Display** | | |
| String representation | ✓ | Shows count and names |
| Pretty print | ✓ [b] | Via `repr()` |

---

## Ray Tracing Parameters

### RayTracingParameters

Configuration parameters for ray-tracing simulations.

| Feature | Support | Notes |
|---|---|---|
| **Engine Information** | | |
| Raytracer name | ✓ | Engine identifier |
| Raytracer version | ✓ | Version string |
| | | |
| **Frequency** | | |
| Center frequency | ✓ | In Hz |
| | | |
| **Interaction Limits** | | |
| Max path depth | ✓ | Total interactions (R+D+S+T) |
| Max reflections | ✓ | Reflection limit |
| Max diffractions | ✓ | Diffraction limit |
| Max scattering | ✓ | Diffuse scattering limit |
| Max transmissions | ✓ | Transmission limit |
| | | |
| **Diffuse Scattering Details** | | |
| Diffuse reflections | ✓ | Allowed with scattering |
| Diffuse diffractions | ✓ | Allowed with scattering |
| Diffuse transmissions | ✓ | Allowed with scattering |
| Final interaction only | ✓ | Scattering at last bounce only |
| Random phases | ✓ | Randomize scattered phases |
| | | |
| **Terrain Interactions** | | |
| Terrain reflections | ✓ | Enable/disable |
| Terrain diffractions | ✓ | Enable/disable |
| Terrain scattering | ✓ | Enable/disable |
| | | |
| **Ray Casting** | | |
| Number of rays | ✓ | Rays per antenna |
| Casting method | ✓ | Uniform/other |
| Synthetic array | ✓ | Enable/disable |
| Azimuth range | ✓ | In degrees |
| Elevation range | ✓ | In degrees |
| | | |
| **GPS Bounding Box** | | |
| Min latitude | ✓ | GPS coordinates |
| Min longitude | ✓ | GPS coordinates |
| Max latitude | ✓ | GPS coordinates |
| Max longitude | ✓ | GPS coordinates |
| | | |
| **Raw Parameters** | | |
| Original engine params | ✓ | Stored as dict |
| Parameter preservation | ✓ | No information loss |
| | | |
| **Serialization** | | |
| To dictionary | ✓ | `to_dict()` |
| From dictionary | ✓ | `from_dict()` |
| JSON compatible | ✓ | Via dict serialization |

---

## TX/RX Classes

### TxRxSet

Configuration for a set of transmitters or receivers.

| Feature | Support | Notes |
|---|---|---|
| **Identification** | | |
| Set name | ✓ | String identifier |
| Original ID | ✓ | From ray tracer |
| DeepMIMO ID | ✓ | Converted index |
| TX/RX role flags | ✓ | `is_tx`, `is_rx` |
| | | |
| **Point Configuration** | | |
| Total points | ✓ | `num_points` |
| Active points | ✓ | Points with paths |
| Inactive tracking | ✓ | Computed during conversion |
| | | |
| **Antenna Configuration** | | |
| Number of elements | ✓ | `num_ant` |
| Dual polarization | ✓ | `dual_pol` flag |
| Element positions | ✓ | Relative to center |
| Array orientation | ✓ | [azimuth, elevation, roll] |
| | | |
| **Serialization** | | |
| To dictionary | ✓ | `to_dict()` |
| From dictionary | ✓ | Constructor from dict |
| String representation | ✓ | `repr()` with role, name, ID |

### TxRxPair

Represents a transmitter-receiver pair in simulation.

| Feature | Support | Notes |
|---|---|---|
| **Configuration** | | |
| TX set | ✓ | TxRxSet object |
| RX set | ✓ | TxRxSet object |
| TX index | ✓ | Index within TX set |
| | | |
| **Operations** | | |
| Get IDs | ✓ | Returns (tx_id, rx_id) tuple |
| String representation | ✓ | Shows TX and RX names |

### Utility Functions

| Function | Purpose | Support |
|---|---|---|
| `get_txrx_sets()` | Load all TX/RX sets from scenario | ✓ |
| `get_txrx_pairs()` | Create all TX-RX combinations | ✓ |
| `print_available_txrx_pair_ids()` | Display pair IDs table | ✓ |

---

## Notes

- **[a]** Filtering by property value requires list comprehension: `[m for m in mat_list if m.permittivity > 3]`.
- **[b]** MaterialList `repr()` shows count and material names for quick overview.

---

## Common Patterns

### Creating a Scene

```python
from deepmimo.core import Scene, PhysicalElement, Face

# Create scene
scene = Scene()

# Create a building
faces = [
    Face(vertices=[(0,0,0), (1,0,0), (1,1,0), (0,1,0)], material_idx=0),
    # ... more faces
]
building = PhysicalElement(
    faces=faces,
    object_id=0,
    label='building',
    name='Building_1'
)
scene.add_object(building)
```

### Working with Materials

```python
from deepmimo.core import Material, MaterialList

# Create material
concrete = Material(
    id=0,
    name='concrete',
    permittivity=5.24,
    conductivity=0.07,
    scattering_model='lambertian',
    scattering_coefficient=0.2
)

# Create list
materials = MaterialList()
materials.add_material(concrete)

# Access properties
print(materials.permittivity)  # [5.24]
```

---

## Related Documentation

- [Scene API Reference](../../api/core.md#scene) - Detailed Scene API
- [Materials API Reference](../../api/core.md#materials) - Material properties
- [Converters Capabilities](converters.md) - How converters populate core classes

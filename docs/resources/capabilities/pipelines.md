# Pipelines Module Capabilities

This document details the capabilities of the DeepMIMO pipelines module, which provides end-to-end workflows for ray-tracing simulation and scene generation.

## Overview

The pipelines module supports:
- **Sionna RT Pipeline**: Execute Sionna ray-tracing simulations
- **Wireless InSite Pipeline**: Execute Wireless InSite ray-tracing simulations
- **Blender OSM**: Generate 3D scenes from OpenStreetMap data
- **TX/RX Placement**: Utilities for positioning transmitters and receivers

---

## Sionna RT Pipeline

Executes Sionna ray-tracing simulations with support for both Sionna 0.19.x and 1.x versions.

| Feature | Support | Notes |
|---|---|---|
| **Execution** | | |
| Ray tracing computation | ✓ | Compute paths from scene |
| Batch processing | ✓ | DataLoader for RX batches |
| GPU acceleration | ✓ | Leverages TensorFlow/Mitsuba GPU |
| CPU offload | ✓ | Move results to CPU after computation |
| Progress tracking | ✓ | tqdm progress bars |
| | | |
| **Version Support** | | |
| Sionna 0.19.x | ✓ | Legacy version |
| Sionna 1.x | ✓ | Current version (1.0.0-1.0.2) |
| Automatic version detection | ✓ | Handles API differences |
| Version validation | ✓ | Warns on untested versions |
| | | |
| **Scene Input** | | |
| XML scene files | ✓ | Load from Mitsuba XML |
| Mitsuba scene objects | ✓ | Direct scene import |
| Material assignment | ✓ | ITU/custom materials |
| Custom geometries | ✓ | Any Mitsuba-compatible scene |
| | | |
| **TX/RX Configuration** | | |
| Single TX | ✓ | Single transmitter position |
| Multiple TXs | ✓ | List of TX positions |
| TX grid | ✓ | Automated TX placement |
| RX grid | ✓ | Automated RX placement |
| Custom positions | ✓ | User-defined coordinates |
| Antenna patterns | ✓ [a] | Via Sionna antenna objects |
| Array configurations | ✓ | Multi-element arrays |
| | | |
| **Ray Tracing Parameters** | | |
| Max path depth | ✓ | Total interactions |
| Reflections | ✓ | Specular reflections |
| Diffractions | ✓ | Edge diffractions |
| Scattering | ✓ | Diffuse scattering [b] |
| Transmissions | ✗ [c] | Not supported by Sionna |
| Number of rays | ✓ | Rays per antenna |
| Synthetic array | ✓ | Enable/disable |
| | | |
| **Output Options** | | |
| Export to pickle | ✓ | Via sionna_exporter |
| Path inspection | ✓ | Custom callback function |
| Auto-conversion | ✓ [d] | Optional direct conversion |
| Multi-scene export | ✓ | Time-varying scenarios |
| | | |
| **Advanced Features** | | |
| Path solver configuration | ✓ | Sionna 1.x PathSolver |
| Scene materialization | ✓ | Sionna 0.19.x |
| Custom compute params | ✓ | Pass-through parameters |
| TensorFlow GPU config | ✓ [e] | Sionna 0.19.x only |

---

## Wireless InSite Pipeline

Executes Wireless InSite ray-tracing simulations via command-line interface.

| Feature | Support | Notes |
|---|---|---|
| **Execution** | | |
| Ray tracing computation | ✓ | Via calcprop_server |
| Batch processing | ✓ | Multiple scenarios |
| Multi-core support | ✓ [f] | InSite internal parallelization |
| Progress monitoring | ✗ [g] | External process |
| | | |
| **Version Support** | | |
| Wireless InSite 3.x | ✓ | Tested |
| Wireless InSite 4.x | ✓ | Tested |
| Version detection | ✓ | From config |
| | | |
| **Scene Input** | | |
| PLY building files | ✓ | From Blender OSM |
| PLY terrain files | ✓ | From Blender OSM |
| CITY format | ✓ | Native InSite format |
| TER format | ✓ | Terrain definition |
| VEG format | ✓ | Vegetation (foliage) |
| Custom geometries | ✓ | Any InSite-compatible |
| | | |
| **TX/RX Configuration** | | |
| Single TX | ✓ | Single transmitter |
| Multiple TXs | ✓ | TX grid/list |
| RX grid | ✓ | Grid with spacing |
| Custom TX positions | ✓ | GPS or Cartesian |
| Custom RX positions | ✓ | GPS or Cartesian |
| Antenna patterns | ✓ [h] | InSite antenna library |
| Polarization | ✓ | V/H/Both |
| | | |
| **File Generation** | | |
| Setup file (.setup) | ✓ | XML format |
| TXRX file (.txrx) | ✓ | XML format |
| Terrain file (.ter) | ✓ | Text format |
| City file (.city) | ✓ | From PLY |
| Study area XML | ✓ | For InSite UI |
| | | |
| **Ray Tracing Parameters** | | |
| Carrier frequency | ✓ | In Hz |
| Reflections | ✓ | Max count |
| Diffractions | ✓ | Max count |
| Scattering | ✓ | Enable/disable |
| Transmissions | ✓ | Max count |
| Terrain interactions | ✓ | Enable/disable |
| Waveform properties | ✓ | CW/pulse/OFDM |
| | | |
| **Material Assignment** | | |
| Building materials | ✓ | From config |
| Road materials | ✓ | From config |
| Terrain materials | ✓ | From config |
| ITU-R materials | ✓ | Built-in library |
| Custom materials | ✓ | User-defined |
| | | |
| **Output Options** | | |
| P2M path files | ✓ | Path data |
| Study info files | ✓ | Simulation metadata |
| Auto-conversion | ✓ [d] | Optional direct conversion |
| | | |
| **Advanced Features** | | |
| Multiple study areas | ✓ | Separate simulations |
| Timestamp naming | ✓ | Avoid overwrite |
| License management | ✓ | License file path |
| Executable path | ✓ | Custom InSite location |

---

## Blender OSM Pipeline

Generates 3D scenes from OpenStreetMap data using Blender.

| Feature | Support | Notes |
|---|---|---|
| **Scene Generation** | | |
| OSM data import | ✓ | Via blosm addon |
| Bounding box input | ✓ | GPS coordinates |
| Auto-fetch from OSM | ✓ | Direct download |
| | | |
| **Output Formats** | | |
| InSite format (PLY) | ✓ | Buildings and roads |
| Sionna format (XML) | ✓ | Mitsuba scene |
| Simultaneous export | ✓ | Both formats at once |
| | | |
| **Scene Elements** | | |
| Buildings | ✓ | 3D extruded |
| Roads | ✓ | Ground-level paths |
| Terrain | ✓ | Ground plane |
| Vegetation | ✗ [i] | Not currently extracted |
| | | |
| **Processing** | | |
| Road trimming | ✓ | To bounding box |
| Road filtering | ✓ | Remove invalid |
| Material assignment | ✓ | Buildings and roads |
| Mesh conversion | ✓ | All to mesh objects |
| Object organization | ✓ | By category |
| | | |
| **Coordinate Systems** | | |
| GPS coordinates | ✓ | Input bounding box |
| Cartesian conversion | ✓ | For simulation |
| Origin storage | ✓ | GPS reference point |
| Bounds metadata | ✓ | Saved to file |
| | | |
| **Visualization** | | |
| Camera setup | ✓ | Automatic positioning |
| Scene rendering | ✓ | Before/after processing |
| Image export | ✓ | PNG format |
| Lighting setup | ✓ | World lighting |
| | | |
| **Blender Integration** | | |
| Addon installation | ✓ | Auto-install blosm |
| Headless operation | ✓ | Command-line execution |
| Scene clearing | ✓ | Clean slate for each run |
| Error logging | ✓ | File and console logs |
| | | |
| **Output Organization** | | |
| Folder per scenario | ✓ | Named by bbox |
| Metadata files | ✓ | Origin, bbox, logs |
| Figure output | ✓ | Separate figs folder |
| Skip existing | ✓ | Avoid re-processing |

---

## TX/RX Placement Utilities

Helper functions for positioning transmitters and receivers.

| Feature | Support | Notes |
|---|---|---|
| **TX Placement** | | |
| GPS coordinates | ✓ | Lat/lon/height |
| Multiple TXs | ✓ | List of positions |
| GPS to Cartesian | ✓ | Automatic conversion |
| Origin reference | ✓ | Relative positioning |
| | | |
| **RX Grid Generation** | | |
| Rectangular grid | ✓ | 2D plane |
| Custom spacing | ✓ | Grid resolution |
| Height specification | ✓ | Z-coordinate |
| GPS bounding box | ✓ | Grid area from GPS |
| Cartesian bounds | ✓ | Direct bounds input |
| | | |
| **Plane Grid Generation** | | |
| XY plane (Z-normal) | ✓ | Horizontal plane |
| XZ plane (Y-normal) | ✓ | Vertical plane |
| YZ plane (X-normal) | ✓ | Vertical plane |
| Fixed coordinate | ✓ | Plane position |
| Min/max coordinates | ✓ | Plane extent |
| Custom spacing | ✓ | Grid density |
| | | |
| **Coordinate Conversions** | | |
| GPS to relative Cartesian | ✓ | From origin |
| GPS bbox to Cartesian | ✓ | Bounding box conversion |
| Origin tracking | ✓ | Reference point storage |

---

## Usage Examples

### Sionna RT Pipeline

```python
from deepmimo.pipelines.sionna_rt import raytrace_sionna

# Run Sionna ray tracing
raytrace_sionna(
    scene_path='./my_scene.xml',
    tx_pos=[[0, 0, 10]],
    rx_pos=rx_grid,
    save_folder='./sionna_output',
    max_depth=6,
    num_samples=1e6
)
```

### Wireless InSite Pipeline

```python
from deepmimo.pipelines.wireless_insite import raytrace_insite

# Run InSite ray tracing
raytrace_insite(
    osm_folder='./my_osm_scene',
    tx_pos=tx_positions,
    rx_pos=rx_grid,
    carrier_freq=28e9,
    max_reflections=5,
    max_diffractions=1,
    wi_exe='/path/to/calcprop_server',
    wi_lic='/path/to/license.lic'
)
```

### Blender OSM

```python
from deepmimo.pipelines.blender_osm import fetch_osm_scene

# Generate scene from OSM
fetch_osm_scene(
    minlat=40.7580,
    minlon=-73.9855,
    maxlat=40.7620,
    maxlon=-73.9800,
    output_folder='./times_square',
    output_formats=['insite', 'sionna']
)
```

### TX/RX Placement

```python
from deepmimo.pipelines.txrx_placement import gen_tx_pos, gen_rx_grid

rt_params = {
    'bs_lats': [40.7589],
    'bs_lons': [-73.9851],
    'bs_heights': [25],
    'origin_lat': 40.7580,
    'origin_lon': -73.9855,
    'min_lat': 40.7580,
    'min_lon': -73.9855,
    'max_lat': 40.7620,
    'max_lon': -73.9800,
    'grid_spacing': 5,
    'ue_height': 1.5
}

tx_pos = gen_tx_pos(rt_params)
rx_pos = gen_rx_grid(rt_params)
```

---

## Notes

- **[a]** Antenna patterns configured via Sionna's `Antenna` objects; see Sionna documentation.
- **[b]** Scattering in Sionna limited to final interaction only (diffuse_final_interaction_only).
- **[c]** Transmissions not supported by Sionna RT engine; use AODT or InSite for transmission support.
- **[d]** Auto-conversion requires calling converter after pipeline execution.
- **[e]** TensorFlow GPU configuration only relevant for Sionna 0.19.x (uses TensorFlow); Sionna 1.x uses dr.jit.
- **[f]** Wireless InSite parallelization controlled by InSite settings, not DeepMIMO.
- **[g]** Progress monitoring requires external process monitoring; InSite runs as separate process.
- **[h]** InSite antenna patterns from InSite library; custom patterns require InSite antenna files.
- **[i]** Vegetation extraction from OSM not currently implemented in Blender pipeline.

---

## Common Limitations

These limitations apply across pipeline operations:

- **External dependencies**: Each pipeline requires specific software (Sionna, InSite, Blender).
- **Version compatibility**: Tested versions documented; others may not work.
- **Sequential execution**: Pipelines run sequentially; no built-in parallelization across scenarios.
- **Memory requirements**: Large scenes may require significant RAM for geometry processing.
- **Disk space**: Ray-tracing outputs can be large (GB per scenario).

---

## Related Documentation

- [Sionna RT README](../../pipelines/sionna_rt/README.md) - Detailed Sionna pipeline guide
- [Wireless InSite README](../../pipelines/wireless_insite/README.md) - Detailed InSite pipeline guide
- [Converters Capabilities](converters.md) - Converting pipeline outputs to DeepMIMO
- [Pipeline Tutorial](../../tutorials/manual.py#pipelines) - End-to-end pipeline examples

# Pipelines

DeepMIMO pipelines are end-to-end workflows that take a geographic location (GPS bounding box) all the way through ray tracing to a finished DeepMIMO dataset — with no manual steps and no proprietary GUI tools required.

```
deepmimo/pipelines/
  ├── osm_to_mitsuba.py      ← OSM → Mitsuba scene (core of the no-Blender pipeline)
  ├── txrx_placement.py      ← TX / RX grid generation
  ├── sionna_rt/             ← Sionna RT pipeline driver
  │   ├── sionna_raytracer.py
  │   └── sionna_utils.py
  └── utils/
      ├── geo_utils.py       ← Coordinate conversion, satellite imagery
      ├── osm_utils.py       ← Building queries and placement validation
      └── pipeline_utils.py  ← BS positioning, scenario management
```

---

## What are pipelines?

A pipeline chains together several independent steps so a single function call can produce a complete ray tracing dataset from nothing but a set of GPS coordinates:

1. **Scene generation** — download building footprints from OpenStreetMap and extrude them into 3D meshes (PLY + Mitsuba XML), or use an existing scene from Blender/Wireless InSite.
2. **TX / RX placement** — convert GPS BS coordinates and UE grids into the local Cartesian system expected by the ray tracer.
3. **Ray tracing** — run Sionna RT (or Wireless InSite) on the scene with the configured antennas.
4. **Export & convert** — serialize the path data and convert it to the standardized DeepMIMO format.

The OSM pipeline (documented below) is the primary no-dependency workflow: it requires only `pip install 'deepmimo[sionna]'` and an internet connection.

---

## OSM → Mitsuba Scene

The `osm_to_mitsuba` module converts a GPS bounding box into a Mitsuba 2 scene that Sionna RT can load directly.  It queries the [Overpass API](https://overpass-api.de) for building footprints, extrudes each polygon into a closed 3D mesh, and writes the result as a `scene.xml` + PLY mesh collection.

### Typical usage

```python
from deepmimo.pipelines.osm_to_mitsuba import generate_scene

scene_folder = generate_scene(
    minlat=48.1355, minlon=11.5735,
    maxlat=48.1395, maxlon=11.5795,
    scene_folder="my_munich_scene",
)
# scene_folder/
#   scene.xml              — Mitsuba scene description
#   meshes/ground.ply      — flat ground plane
#   meshes/building_N.ply  — one PLY mesh per OSM building
#   osm_gps_origin.txt     — origin lat/lon for coordinate recovery

import sionna.rt as sionna_rt
scene = sionna_rt.load_scene(scene_folder + "/scene.xml", merge_shapes=False)
```

### How it works

```
GPS bbox
  │
  ▼
Overpass API  →  list of (footprint_xy, height) per building
  │
  ▼
_extrude_footprint()  →  3D vertices + triangle indices per building
  │                       (bottom ring + top ring + side walls + roof fan)
  ▼
_write_ply()  →  building_N.ply (ASCII PLY, one file per building)
  │
  ▼
_write_mitsuba_xml()  →  scene.xml (ITU concrete material + shape refs)
```

Heights come from OSM tags in order of priority:

| Tag | Behaviour |
|-----|-----------|
| `height` | Parse float, strip " m" suffix |
| `building:height` | Same |
| `building:levels` | Multiply by 3 m per floor |
| *(none)* | Default 10 m |

The coordinate origin is the centre of the GPS bounding box, converted to UTM via `utm.from_latlon`.  All PLY vertices are in local metres relative to this origin.  The origin is saved to `osm_gps_origin.txt` so downstream code can recover GPS coordinates.

### API reference

::: deepmimo.pipelines.osm_to_mitsuba.generate_scene

::: deepmimo.pipelines.osm_to_mitsuba.query_osm_buildings

---

## TX / RX Placement

`txrx_placement` converts pipeline configuration dicts into numpy position arrays.

### Typical usage

```python
from deepmimo.pipelines.txrx_placement import gen_plane_grid

# Generate a UE grid covering the scene
rx_pos = gen_plane_grid(
    min_coord1=xmin + 10, max_coord1=xmax - 10,
    min_coord2=ymin + 10, max_coord2=ymax - 10,
    spacing=30.0,
    fixed_coord=1.5,      # UE height in metres
    normal="z",           # grid lies in XY plane
)
# rx_pos: (N, 3) array of [x, y, z] positions
```

### API reference

::: deepmimo.pipelines.txrx_placement.gen_plane_grid

::: deepmimo.pipelines.txrx_placement.gen_rx_grid

::: deepmimo.pipelines.txrx_placement.gen_tx_pos

---

## Geographic Utilities

`geo_utils` handles all coordinate system conversions between GPS (lat/lon), UTM
metres, and the local Cartesian system used inside the pipeline.

### Coordinate systems

| System | Unit | Used for |
|--------|------|----------|
| GPS (WGS 84) | degrees | Overpass API queries, scenario naming |
| UTM | metres (absolute) | intermediate conversion via `utm` package |
| Local Cartesian | metres (relative) | scene XML, Sionna positions, DeepMIMO dataset |

The local Cartesian origin is always the **centre of the GPS bounding box**.

### Satellite imagery

If you have a Google Maps Static API key, `fetch_satellite_view` downloads a
640 × 640 satellite image of the scene area:

```python
from deepmimo.pipelines.utils.geo_utils import fetch_satellite_view

img_path = fetch_satellite_view(
    minlat=48.1355, minlon=11.5735,
    maxlat=48.1395, maxlon=11.5795,
    api_key="YOUR_GOOGLE_MAPS_API_KEY",
    save_dir="./satellite/",
)
```

### API reference

::: deepmimo.pipelines.utils.geo_utils.xy_from_latlong

::: deepmimo.pipelines.utils.geo_utils.convert_GpsBBox2CartesianBBox

::: deepmimo.pipelines.utils.geo_utils.convert_Gps2RelativeCartesian

::: deepmimo.pipelines.utils.geo_utils.haversine_distance

::: deepmimo.pipelines.utils.geo_utils.fetch_satellite_view

---

## Pipeline Utilities

`pipeline_utils` provides helpers for reading pipeline outputs and placing
base stations.

### Typical usage

```python
from deepmimo.pipelines.utils.pipeline_utils import get_origin_coords

# After generate_scene(), recover the coordinate origin
origin_lat, origin_lon = get_origin_coords(osm_scene_folder)
```

### API reference

::: deepmimo.pipelines.utils.pipeline_utils.get_origin_coords

---

## OSM Building Utilities

`osm_utils` provides helpers for querying and validating building data — useful
when you need to place a base station at a specific GPS location and ensure it
does not overlap with any building.

### Typical usage

```python
from deepmimo.pipelines.utils.osm_utils import get_buildings, find_nearest_clear_location

# Find a clear outdoor BS location near a given GPS point
buildings = get_buildings(lat=48.137, lon=11.576)
clear_lat, clear_lon = find_nearest_clear_location(
    original_lat=48.137, original_lon=11.576, buildings=buildings
)
```

### API reference

::: deepmimo.pipelines.utils.osm_utils.get_buildings

::: deepmimo.pipelines.utils.osm_utils.find_nearest_clear_location

::: deepmimo.pipelines.utils.osm_utils.is_point_clear_of_buildings

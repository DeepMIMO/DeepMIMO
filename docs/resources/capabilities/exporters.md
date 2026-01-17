# Exporters Module Capabilities

This document details the capabilities of the DeepMIMO exporters module, which provides functionality for exporting ray-tracing data from external sources into DeepMIMO-compatible formats.

## Overview

The exporters module supports:
- **AODT Exporter**: Export data from AODT ClickHouse database to parquet files
- **Sionna Exporter**: Export Sionna ray-tracing results to pickle files

---

## AODT Exporter

Exports AODT simulation data from ClickHouse database to parquet format for conversion to DeepMIMO.

| Feature | Support | Notes |
|---|---|---|
| **Source Format** | | |
| ClickHouse database | ✓ | Direct database connection |
| Table selection | ✓ | Exports specific tables |
| Table filtering | ✓ | Can ignore specified tables |
| Multiple databases | ✓ | Supports database selection |
| | | |
| **Target Format** | | |
| Parquet files | ✓ | One file per table |
| Column preservation | ✓ | All columns exported |
| Data type preservation | ✓ | Native pandas/parquet types |
| Compression | ✓ [a] | Parquet default compression |
| | | |
| **Exported Tables** | | |
| Scenario parameters | ✓ | `scenario.parquet` |
| Ray paths | ✓ | `raypaths.parquet` |
| Channel responses (CIR) | ✓ | `cirs.parquet` |
| TX configurations (RUs) | ✓ | `rus.parquet` |
| RX configurations (UEs) | ✓ | `ues.parquet` |
| Material properties | ✓ | `materials.parquet` |
| Antenna patterns | ✓ | `patterns.parquet` |
| Time information | ✓ | `time_info.parquet` |
| Antenna panels | ✓ | `panels.parquet` |
| Database info | ✓ | `db_info.parquet` |
| Run configurations | ✓ | `runs.parquet` |
| | | |
| **Ignored Tables** | | |
| CFR data | ✓ [b] | Excluded by default |
| Training results | ✓ [b] | Excluded by default |
| World geometry | ✓ [b] | Excluded by default |
| CSI reports | ✓ [b] | Excluded by default |
| Telemetry | ✓ [b] | Excluded by default |
| DUs | ✓ [b] | Excluded by default |
| RAN config | ✓ [b] | Excluded by default |
| | | |
| **Time-Varying Support** | | |
| Single time slice | ✓ | Exports to single folder |
| Multiple time slices | ✓ | Creates `scene_XXXX` folders |
| Time index filtering | ✓ | Filters by `time_idx` column |
| Time info table | ✓ | Determines number of time slices |
| Per-time-slice folders | ✓ | Organized by time index |
| | | |
| **Database Operations** | | |
| List databases | ✓ | Query available databases |
| List tables | ✓ | Query tables in database |
| Get table columns | ✓ | Query column names |
| Execute queries | ✓ | Generic SQL queries |
| Auto database selection | ✓ [c] | Selects first non-system database |
| | | |
| **Output Organization** | | |
| Named output directory | ✓ | Uses database name |
| Subfolder structure | ✓ | For time-varying scenarios |
| Metadata preservation | ✓ | All table metadata exported |
| File naming convention | ✓ | `{table_name}.parquet` |
| | | |
| **Error Handling** | | |
| Connection errors | ✓ | Graceful error messages |
| Export errors | ✓ | Per-table error handling |
| Empty simulations | ✓ | Validation and error reporting |
| Missing tables | ✓ | Continues with available tables |

---

## Sionna Exporter

Exports Sionna ray-tracing simulation results to pickle format for later conversion to DeepMIMO.

| Feature | Support | Notes |
|---|---|---|
| **Source Format** | | |
| Sionna Paths object | ✓ | From `scene.compute_paths()` |
| Sionna Scene object | ✓ | Scene geometry |
| Single TX Paths | ✓ | Single Paths object |
| Multi-TX Paths | ✓ | List of Paths objects |
| | | |
| **Target Format** | | |
| Pickle files | ✓ | Python pickle serialization |
| Dictionary format | ✓ | Structured nested dicts |
| NumPy arrays | ✓ | Preserves array types |
| | | |
| **Sionna Version Support** | | |
| Sionna 0.19.x | ✓ | Legacy version support |
| Sionna 1.x | ✓ | Current version support |
| Version detection | ✓ | Automatic version handling |
| | | |
| **Exported Path Data** | | |
| Path delays (tau) | ✓ | Time delays |
| Path amplitudes (a) | ✓ | Complex or (real, imag) tuple |
| Receive angles (phi_r, theta_r) | ✓ | AoA azimuth and elevation |
| Transmit angles (phi_t, theta_t) | ✓ | AoD azimuth and elevation |
| Interaction vertices | ✓ | 3D positions of interactions |
| Interaction types | ✓ | `types` (0.x) or `interactions` (1.x) |
| Source positions | ✓ | TX positions |
| Target positions | ✓ | RX positions |
| TX array info | ✓ | Transmit array configuration |
| RX array info | ✓ | Receive array configuration |
| | | |
| **Exported Scene Data** | | |
| Scene objects | ✓ | Object names and properties |
| Object vertices | ✓ | Per-object vertex arrays |
| Vertex coordinates | ✓ | 3D (x, y, z) coordinates |
| Material indices | ✓ | Per-object material mapping |
| Scene bounding box | ✓ [d] | When available from scene |
| GPS coordinates | ✓ [d] | When available from scene |
| | | |
| **Format Differences Handling** | | |
| Sionna 0.x complex amplitudes | ✓ | Exports complex numbers |
| Sionna 1.x tuple amplitudes | ✓ | Exports (real, imag) tuples |
| Sionna 0.x `types` field | ✓ | Interaction type encoding |
| Sionna 1.x `interactions` field | ✓ | Interaction type encoding |
| Position transpose (v1.x) | ✓ | Handles transposed sources/targets |
| | | |
| **Filtering and Processing** | | |
| Relevant fields only | ✓ | Excludes internal/private attrs |
| Non-callable filtering | ✓ | Excludes methods |
| Scene reference removal | ✓ | Avoids circular references |
| Private attribute removal | ✓ | Excludes `_` and `__` attrs |
| | | |
| **Output Files** | | |
| Paths data | ✓ | `sionna_paths.pkl` |
| Scene vertices | ✓ | `sionna_vertices.pkl` |
| Scene objects | ✓ | `sionna_objects.pkl` |
| Material map | ✓ | `sionna_material_map.pkl` |
| RT parameters | ✓ | `sionna_rt_params.pkl` |
| | | |
| **Multi-TX Handling** | | |
| Single Paths object | ✓ | Direct export |
| List of Paths objects | ✓ | Exports list of dicts |
| TX index tracking | ✓ | Preserves TX ordering |
| | | |
| **Error Handling** | | |
| Import errors | ✓ | Graceful Sionna dependency check |
| Version mismatch warnings | ✓ | Warns on untested versions |
| Export failures | ✓ | Exception propagation |

---

## Usage Examples

### AODT Exporter

```python
from clickhouse_connect import get_client
from deepmimo.exporters.aodt_exporter import aodt_exporter

# Connect to ClickHouse
client = get_client(host='localhost', port=8123)

# Export database to parquet files
output_path = aodt_exporter(
    client,
    database='my_simulation',
    output_dir='./aodt_exports',
    ignore_tables=['cfrs', 'training_result']
)

# Convert to DeepMIMO format
import deepmimo as dm
dm.aodt_rt_converter(output_path)
```

### Sionna Exporter

```python
from deepmimo.exporters.sionna_exporter import sionna_exporter
import sionna.rt as rt

# Run Sionna ray tracing
scene = rt.load_scene('scene.xml')
paths = scene.compute_paths()

# Export to pickle
sionna_exporter(
    scene=scene,
    path_list=paths,
    compute_path_params={},  # Parameters used
    save_folder='./sionna_exports'
)

# Convert to DeepMIMO format
import deepmimo as dm
dm.sionna_rt_converter('./sionna_exports')
```

---

## Notes

- **[a]** Parquet uses default compression (typically Snappy). Compression method can be configured via pandas.
- **[b]** Some tables excluded by default to reduce export size. Customize via `ignore_tables` parameter.
- **[c]** If no database specified, selects second database in list (first is usually 'system').
- **[d]** GPS bounding box and scene metadata exported when present in Sionna scene object.

---

## Common Limitations

These limitations apply to both exporters:

- **Python dependencies**: AODT requires pandas, Sionna exporter requires sionna package.
- **Memory constraints**: Large simulations may require chunked processing (not currently implemented).
- **Single simulation**: Each export handles one simulation/scenario at a time.
- **No incremental export**: Full export each time; no incremental/delta updates.

---

## Related Documentation

- [Converters Capabilities](converters.md) - Converting exported data to DeepMIMO format
- [AODT Converter](../../api/converter.md#aodt) - AODT to DeepMIMO conversion
- [Sionna Converter](../../api/converter.md#sionna-rt) - Sionna to DeepMIMO conversion

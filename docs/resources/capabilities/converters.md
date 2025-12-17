# Converter Capability Comparison

This document compares the capabilities of DeepMIMO's ray-tracing converters. For information about the ray tracers themselves (not the converters), see [Comparing Ray Tracers](../comparing_raytracers.md).

## Converter Comparison Table

| Feature | AODT | Sionna RT | Wireless InSite |
|---|---|---|---|
| Multiple BS (TX points) | ✓ | ✓ [a] | ✓ |
| Multiple RX sets | ✗ [b] | ✗ [b] | ✓ |
| TX multi-antenna arrays | ✓ | ✓ [a] | ✗ [l] |
| RX multi-antenna arrays | ✓ | ✓ [a] | ✗ [l] |
| Dual polarization | ✓ [q] | ✓ [q] | ✓ [q] |
| Per-element positions (arrays) | ✓ | ✓ | ✗ [l] |
| Antenna pattern support | Isotropic-only [c] | N/A | Isotropic (via polarization) |
| BS–BS paths | ✗ | ✓ [k], [s] | ✗ |
| Reflections | ✓ | ✓ | ✓ |
| Diffractions | ✓ (≤1) | ✓ (≤1) | ✓ |
| Diffuse scattering | ✓ | ✓ (final only) | ✓ |
| Transmissions | ✓ | ✗ [e] | ✓ |
| Interaction positions export | ✓ | ✓ | ✓ [n] |
| Angles export (AoA/AoD) | ✓ [r] | ✓ | ✓ |
| Delay export | ✓ | ✓ | ✓ |
| Power export | ✓ | ✓ | ✓ |
| Phase export | ✓ | ✓ | ✓ |
| Path sorting by amplitude | ✗ | ✓ [t] | ✗ |
| Inactive RX detection | ✗ [g] | ✓ [g] | ✓ [g] |
| Dynamics/time-varying scenes | ✓ [f] | ✓ [f] | ✓ (API) [f] |
| Scene export | ✓ (disabled by default) [h] | ✓ | ✓ |
| Materials richness | ✓ (ITU-R P.2040) [i] | ✓ (scat. patterns) [i] | ✓ (incl. foliage) [i] |
| GPS bounding box | ✗ [j] | ✓ (when present) [j] | ✓ [j] |
| Terrain interaction flags | ✓ [o] | ✓ [o] | ✓ [o] |
| Uniform ray casting | ✓ | ✓ (mapped) | ✓ |
| Copy RT source files | ✓ [p] | ✓ [p] | ✓ [p] |
| Max path/interactions handled | ✓ | ✓ | ✓ |

---

## Notes

- **[a]** Sionna supports multiple TX points or multi-antenna, but not both simultaneously in one run; same constraint for RX. Single TX and RX sets in metadata.
- **[b]** AODT aggregates all UEs into one RX set; Sionna uses one RX set; InSite supports multiple RX sets.
- **[c]** AODT validates patterns: isotropic required; warns on halfwave dipole, errors on non-isotropic/custom.
- **[e]** Sionna does not support transmissions; AODT and InSite map transmission-like events.
- **[f]** AODT picks a time index for paths and RX positions from routes; Sionna exports one time slice of amplitudes; InSite runs are static per export but converter supports multi-scene foldering.
- **[g]** Inactive RXs: Sionna counts zero-amplitude paths; InSite uses 250 dB path-loss sentinel; AODT does not update active counts.
- **[h]** AODT scene reader exists (USD-like) but is disabled in the main converter flow; Sionna and InSite export scenes.
- **[i]** AODT computes permittivity/conductivity via ITU-R P.2040; Sionna maps RT scattering patterns; InSite includes EM, roughness/thickness, and foliage attenuation.
- **[j]** GPS bbox: Sionna includes when min/max lat/lon provided; InSite computes from study area + origin; AODT sets (0,0,0,0).
- **[k]** Sionna detects sources==targets and exports BS–BS paths.
- **[l]** InSite infers element count from polarization (Vertical/Horizontal=1, Both=2) but doesn't export per-element geometry.
- **[n]** InSite interaction positions parsed from .p2m; Sionna from vertices; AODT from interaction point lists.
- **[o]** Terrain flags: AODT reflection/scattering true, diffraction false; Sionna per-RT flags; InSite per setup.
- **[p]** Copy sources: AODT (.parquet), Sionna (.pkl), InSite (.setup/.txrx/.city/.ter/.veg/.kmz).
- **[q]** Dual-pol: AODT via panel flag; Sionna via num_ant vs size; InSite via "Both" polarization.
- **[r]** AODT computes angles from geometry; Sionna provides angles; InSite from file fields.
- **[s]** Note: AODT doesn't have scene geometry export enabled.
- **[t]** Sionna sorts by path amplitude before truncation to MAX_PATHS.

---

## Common Limitations

The following limitations apply to **all converters** (not specific to any ray tracer):

### Antenna Patterns
All converters currently support only **isotropic antenna patterns**. This is a converter limitation, not a ray-tracer limitation. The ray tracers themselves support various antenna patterns, but the converters don't yet import this information.

### Path Truncation
All converters limit paths to `MAX_PATHS` (default: 10) per TX-RX link for memory efficiency. Additional paths beyond this limit are discarded.

### Single Frequency
All converters handle single-frequency simulations. Multi-band scenarios require separate conversions per frequency band.

### Position Precision
All converters store positions in float32 precision for memory efficiency, which provides ~7 decimal digits of precision.

---

## Important Distinctions

### Converter Limitations vs Ray-Tracer Limitations

Some features marked with ✗ are **ray-tracer limitations** (the ray-tracing software itself doesn't support the feature), while others are **converter limitations** (the converter doesn't import that information yet, even though the ray tracer supports it).

For detailed information about the ray tracers themselves, see [Comparing Ray Tracers](../comparing_raytracers.md).

---

## Converter-Specific Details

### AODT Converter

**Input Format**: Parquet files from AODT ClickHouse database
- `scenario.parquet`, `raypaths.parquet`, `cirs.parquet`, `rus.parquet`, `ues.parquet`, `materials.parquet`, `patterns.parquet`, `panels.parquet`, `time_info.parquet`

**Key Features**:
- ITU-R P.2040 material model support
- Time-indexed scenarios (picks first time index)
- Panel-based antenna configuration
- Rich interaction position data

**Known Limitations**:
- Scene geometry export disabled by default
- Single RX set (aggregates all UEs)
- Inactive RX count not updated

### Sionna RT Converter

**Input Format**: Pickle files from Sionna ray tracer
- `sionna_paths.pkl`, `sionna_vertices.pkl`, `sionna_objects.pkl`, `sionna_material_map.pkl`, `sionna_rt_params.pkl`

**Key Features**:
- Automatic version detection (0.19.x vs 1.x)
- BS-BS path support
- Path amplitude sorting
- Complete scene geometry export

**Known Limitations**:
- No transmission support (ray-tracer limitation)
- Single TX and RX set in metadata
- Multi-antenna OR multiple TXs, not both simultaneously

### Wireless InSite Converter

**Input Format**: XML and text files from Wireless InSite
- `.setup`, `.txrx`, `.city`, `.ter`, `.veg`, `.p2m` files

**Key Features**:
- Multiple RX set support
- Complete scene geometry (buildings, terrain, vegetation)
- Foliage attenuation in materials
- Multiple TX/RX configuration support

**Known Limitations**:
- No per-element antenna positions (infers count from polarization)
- Path sorting not implemented in converter

---

## Usage Examples

### AODT Converter

```python
import deepmimo as dm

# Convert AODT scenario
scenario_name = dm.aodt_rt_converter(
    rt_folder='/path/to/aodt/parquet/files',
    copy_source=True,
    vis_scene=False
)

# Load converted dataset
dataset = dm.load(scenario_name)
```

### Sionna RT Converter

```python
import deepmimo as dm

# Convert Sionna scenario
scenario_name = dm.sionna_rt_converter(
    rt_folder='/path/to/sionna/pickle/files',
    copy_source=True,
    vis_scene=True
)

# Load converted dataset
dataset = dm.load(scenario_name)
```

### Wireless InSite Converter

```python
import deepmimo as dm

# Convert InSite scenario
scenario_name = dm.insite_rt_converter(
    rt_folder='/path/to/insite/output/folder',
    copy_source=True,
    vis_scene=True
)

# Load converted dataset
dataset = dm.load(scenario_name)
```

---

## Related Documentation

- [Comparing Ray Tracers](../comparing_raytracers.md) - Ray-tracer feature comparison
- [Ray-Tracing Guidelines](../raytracing_guidelines.md) - Best practices
- [Exporters Capabilities](exporters.md) - Exporting data before conversion
- [Converter Tutorial](../../tutorials/7_converters.py) - Converter usage examples

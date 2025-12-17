# Dataset Module Capabilities

This document details the capabilities of the DeepMIMO dataset module, which handles loading, manipulating, and generating datasets from ray-tracing scenarios.

## Core Classes

| Feature | Dataset | MacroDataset | DynamicDataset |
|---|---|---|---|
| Single TX/RX pair | ✓ | ✗ | ✗ |
| Multiple TX/RX pairs | ✗ | ✓ | ✓ |
| Time-varying scenarios | ✗ | ✗ | ✓ |
| Dictionary-like access | ✓ | ✓ [a] | ✓ [a] |
| Dot notation access | ✓ | ✓ [a] | ✓ [a] |
| Indexing by UE/TX | ✓ | ✓ | ✓ |

## Loading Capabilities

| Feature | Dataset | MacroDataset | DynamicDataset |
|---|---|---|---|
| Load from scenario name | ✓ | ✓ | ✓ |
| Load from absolute path | ✓ | ✓ | ✓ |
| Selective TX set loading | ✓ | ✓ | ✓ |
| Selective RX set loading | ✓ | ✓ | ✓ |
| Selective matrix loading | ✓ | ✓ | ✓ |
| Max paths limiting | ✓ | ✓ | ✓ |
| Auto-download if missing | ✓ | ✓ | ✓ |
| Scene geometry loading | ✓ | ✓ | ✓ |
| Material properties loading | ✓ | ✓ | ✓ |
| RT parameters loading | ✓ | ✓ | ✓ |

## Channel Generation

| Feature | Dataset | MacroDataset | DynamicDataset |
|---|---|---|---|
| Time domain channels | ✓ | ✓ | ✓ |
| Frequency domain channels | ✓ | ✓ | ✓ |
| MIMO channel matrices | ✓ | ✓ | ✓ |
| Array response computation | ✓ | ✓ | ✓ |
| Custom TX antenna patterns | ✓ [b] | ✓ [b] | ✓ [b] |
| Custom RX antenna patterns | ✓ [b] | ✓ [b] | ✓ [b] |
| Element-wise array patterns | ✓ | ✓ | ✓ |
| Uniform planar arrays | ✓ | ✓ | ✓ |
| Custom array geometries | ✓ | ✓ | ✓ |
| Synthetic array mode | ✓ | ✓ | ✓ |
| Subarray selection | ✓ | ✓ | ✓ |
| Bandwidth configuration | ✓ | ✓ | ✓ |
| OFDM subcarrier mapping | ✓ [c] | ✓ [c] | ✓ [c] |

## Path and Power Operations

| Feature | Dataset | MacroDataset | DynamicDataset |
|---|---|---|---|
| Path power (dBW) | ✓ | ✓ | ✓ |
| Path phase (degrees) | ✓ | ✓ | ✓ |
| Path delays | ✓ | ✓ | ✓ |
| AoA (azimuth/elevation) | ✓ | ✓ | ✓ |
| AoD (azimuth/elevation) | ✓ | ✓ | ✓ |
| Interaction types | ✓ [d] | ✓ [d] | ✓ [d] |
| Interaction positions | ✓ | ✓ | ✓ |
| Total power computation | ✓ | ✓ | ✓ |
| Path loss computation | ✓ | ✓ | ✓ |
| Max power path extraction | ✓ | ✓ | ✓ |
| LoS path identification | ✓ | ✓ | ✓ |
| NLoS path filtering | ✓ | ✓ | ✓ |
| Path depth filtering | ✓ | ✓ | ✓ |
| Path type filtering | ✓ [d] | ✓ [d] | ✓ [d] |
| Path sorting | ✓ | ✓ | ✓ |
| Active UE detection | ✓ | ✓ | ✓ |

## Geometric Operations

| Feature | Dataset | MacroDataset | DynamicDataset |
|---|---|---|---|
| TX positions | ✓ | ✓ | ✓ |
| RX positions | ✓ | ✓ | ✓ |
| TX-RX distances | ✓ | ✓ | ✓ |
| Angle rotations | ✓ [e] | ✓ [e] | ✓ [e] |
| Coordinate transformations | ✓ | ✓ | ✓ |
| Velocity vectors | ✓ [f] | ✓ [f] | ✓ [f] |
| Doppler computation | ✓ [f] | ✓ [f] | ✓ [f] |

## Field of View (FoV)

| Feature | Dataset | MacroDataset | DynamicDataset |
|---|---|---|---|
| TX FoV filtering | ✓ | ✓ | ✓ |
| RX FoV filtering | ✓ | ✓ | ✓ |
| Azimuth-only FoV | ✓ | ✓ | ✓ |
| Elevation-only FoV | ✓ | ✓ | ✓ |
| Combined AzEl FoV | ✓ | ✓ | ✓ |
| FoV caching | ✓ | ✓ | ✓ |
| FoV cache clearing | ✓ | ✓ | ✓ |

## Sampling and Subsetting

| Feature | Dataset | MacroDataset | DynamicDataset |
|---|---|---|---|
| Index-based selection | ✓ | ✓ | ✓ |
| Linear path sampling | ✓ | ✓ | ✓ |
| Uniform grid sampling | ✓ | ✓ | ✓ |
| Grid row/column selection | ✓ | ✓ | ✓ |
| Coordinate limit filtering | ✓ | ✓ | ✓ |
| Active UE filtering | ✓ | ✓ | ✓ |
| Power-based filtering | ✓ | ✓ | ✓ |
| Grid information | ✓ | ✓ | ✓ |
| Dataset trimming | ✓ | ✓ | ✓ |
| Combined trim operations | ✓ [g] | ✓ [g] | ✓ [g] |

## Visualization

| Feature | Dataset | MacroDataset | DynamicDataset |
|---|---|---|---|
| Ray path plotting | ✓ | ✓ [h] | ✓ [h] |
| Coverage map plotting | ✓ | ✓ [h] | ✓ [h] |
| Power discarding plots | ✓ | ✓ [h] | ✓ [h] |
| Summary plots | ✓ | ✓ | ✓ |
| Interactive path selection | ✓ | ✓ [h] | ✓ [h] |
| 3D scene visualization | ✓ | ✓ | ✓ |
| TX/RX position markers | ✓ | ✓ | ✓ |
| Path interaction markers | ✓ | ✓ | ✓ |
| Custom color schemes | ✓ | ✓ | ✓ |

## Export Capabilities

| Feature | Dataset | MacroDataset | DynamicDataset |
|---|---|---|---|
| Binary format export | ✓ | ✓ | ✓ |
| Web visualizer export | ✓ | ✓ | ✓ |
| Sionna format export | ✓ | ✓ | ✓ [i] |
| MATLAB/MAT format | ✓ [j] | ✓ [j] | ✓ [j] |
| NumPy arrays | ✓ | ✓ | ✓ |

## Summary and Reporting

| Feature | Dataset | MacroDataset | DynamicDataset |
|---|---|---|---|
| Text summary | ✓ | ✓ | ✓ |
| Parameter summary | ✓ | ✓ | ✓ |
| Coverage statistics | ✓ | ✓ | ✓ |
| Path statistics | ✓ | ✓ | ✓ |
| Power distribution | ✓ | ✓ | ✓ |
| Formatted display | ✓ | ✓ | ✓ |

## Utility Features

| Feature | Dataset | MacroDataset | DynamicDataset |
|---|---|---|---|
| DeepMIMOArray wrapper | ✓ | ✓ | ✓ |
| Lazy evaluation | ✓ | ✓ | ✓ |
| Memory optimization | ✓ | ✓ | ✓ |
| Key resolution | ✓ | ✓ | ✓ |
| Parameter validation | ✓ | ✓ | ✓ |
| Unit conversions | ✓ [k] | ✓ [k] | ✓ [k] |

---

## Notes

- **[a]** MacroDataset and DynamicDataset delegate to underlying Dataset objects.
- **[b]** Custom antenna patterns supported via `AntennaPattern` class (isotropic, halfwave dipole, custom).
- **[c]** When `freq_domain=True` in channel parameters, supports OFDM subcarrier generation.
- **[d]** Interaction types: LoS (0), Reflection (1), Diffraction (2), Scattering (3), Transmission (5), combinations (e.g., 11=two reflections).
- **[e]** Supports rotation around TX or RX by specifying rotation axes (azimuth, elevation, roll).
- **[f]** Doppler computation requires velocity vectors to be set via `set_velocities()` method.
- **[g]** Trimming supports combined filters: indices, FoV (TX/RX), path depth, path types.
- **[h]** For MacroDataset/DynamicDataset, visualization operates on individual datasets (must specify which).
- **[i]** DynamicDataset export to Sionna requires time-step specification.
- **[j]** Internal storage uses MAT files; direct access via `_data` attribute or matrix keys.
- **[k]** Unit conversions: dBW ↔ Watts, degrees ↔ radians, spherical ↔ Cartesian coordinates.

---

## Common Limitations

These limitations apply to all dataset types:

- **Max paths**: Limited to `MAX_PATHS` (default 10) per TX-RX link for memory efficiency.
- **Path truncation**: Paths beyond `MAX_PATHS` are truncated during loading.
- **Antenna patterns**: Complex 3D patterns must be discretized into azimuth/elevation grids.
- **Memory usage**: Large multi-antenna MIMO channels can require significant memory.
- **Single frequency**: Channel generation is single-frequency per call (multi-band requires multiple generations).

---

## Related Documentation

- [Converters Capabilities](converters.md) - Ray-tracing data conversion
- [Generation Tutorial](../../tutorials/3_channel_generation.py) - Channel generation examples
- [Visualization Tutorial](../../tutorials/2_visualization.py) - Plotting examples
- [Dataset Manipulation](../../tutorials/4_dataset_manipulation.py) - Sampling and filtering

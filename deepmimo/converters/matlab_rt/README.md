# MATLAB RT Converter MVP

This package contains the minimal, file-based MATLAB Ray Tracing converter.

## Scope

The MVP target is:

```text
MATLAB RT JSON export -> DeepMIMO path matrices -> deepmimo.load() -> compute_channels()
```

The first implementation should support the JSON schema validated in the experiment fixtures under:

```text
tests/converters/matlab_rt/fixtures/
```

The converter assumes the JSON was exported from valid MATLAB Ray Tracing output.
It does not repair inconsistent geometry or infer missing links.

## Public API

```python
from deepmimo.converters.matlab_rt import convert_matlab_rt_json

result = convert_matlab_rt_json(
    "matlab_rt_export.json",
    scenario_root="deepmimo_scenarios",
    scenario_name="matlab_rt_example",
    overwrite=False,
)
```

`scenario_root` should be the DeepMIMO scenarios directory that `deepmimo.load()`
can discover, normally `deepmimo_scenarios` from the current working directory.
`scenario_name` is stripped and lowercased before writing.

`convert_matlab_rt_json(...)` accepts either:

- a path to a validated MATLAB RT JSON export
- a parsed `MatlabRTExport` object

It returns a `MatlabRTWriteResult` containing:

- normalized `scenario_name`
- `scenario_root`
- generated `scenario_path`
- all written file paths
- per-TX matrix file paths
- per-TX matrix shapes

## Conversion Pipeline

The public converter performs this fixed MVP pipeline:

```text
MATLAB RT JSON
-> parser/schema validation
-> path-row grouping
-> in-memory matrix assembly
-> in-memory params/metadata construction
-> DeepMIMO scenario folder writer
```

The writer produces:

- `params.json`
- `objects.json`
- `vertices.npz`
- `power`, `phase`, `delay`
- `aoa_az`, `aoa_el`, `aod_az`, `aod_el`
- `inter`, `inter_pos`
- `rx_pos`, `tx_pos`

## Input JSON Contract

Required top-level fields are:

- Common: `metadata`, `scene`, `propagation_model`
- Single-link: `transmitter`, `receiver`, `num_rays`, `rays`
- Multi-link: `num_tx`, `num_rx`, `transmitters`, `receivers`, `links`

Each multi-link export must include every TX/RX pair explicitly. Empty links are
valid, but must be present with `num_rays: 0` and `rays: []`.

Required ray fields are:

- `index`, `line_of_sight`, `transmitter_location_m`, `receiver_location_m`
- `path_loss_db`, `propagation_delay_s`, `propagation_distance_m`
- `angle_of_departure_deg`, `angle_of_arrival_deg`
- `num_interactions`, `interactions`, `path_coordinates_m`
- either `phase_shift_deg` or `phase_shift_rad`

Units and conventions:

- Positions and path coordinates are Cartesian meters.
- Frequencies are Hz, delays are seconds, angles are degrees.
- MATLAB elevation is converted to DeepMIMO theta with `theta = 90 - elevation`.
- Path power is written as dBW using `tx_power_dbw + tx_gain_db + rx_gain_db - path_loss_db`.
- TX, RX, link, ray, and interaction indices in JSON are one-based and deterministic.
- Interaction type support is reflection-only. Multiple reflection interactions are encoded in order; non-reflection interactions are rejected.

## In Scope

- File-based MATLAB RT JSON conversion.
- Cartesian coordinates in meters.
- One TX set and one RX set.
- Multiple TX and RX points.
- LoS paths.
- Reflection-only paths.
- `NaN` padding for empty links.
- Compatibility with `load()`, `compute_channels()`, OFDM channel generation, beamforming, and ray plotting.

## Out of Scope

- MATLAB Engine startup or execution.
- MATLAB server/REST integrations.
- MPLM.
- `inter_obj`.
- `path_hash`.
- velocity-derived Doppler.
- advanced scene geometry and object metadata.

The excluded features should remain documented limitations until object/material identity mapping is designed.

## Metadata Policy

- `objects.json` and `vertices.npz` are intentional empty-scene placeholders.
- Material records are loader/summary placeholders unless object-aware material mapping is added later.
- Empty MATLAB `MaterialName` values are normalized to unknown material metadata and do not create a named material entry.

## Safety

- Existing output is refused unless `overwrite=True`.
- Overwrite is limited to the requested scenario folder under `scenario_root`.
- The converter does not import MATLAB Engine.
- The converter does not run MATLAB.
- The converter does not access the network.

## Future Work / Known Limitations

- Geometry consistency validation is intentionally limited in this MVP.
- The converter does not verify that path coordinates start/end at the link TX/RX positions.
- The converter does not map interactions to scene objects or material identities.

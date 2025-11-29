# DeepMIMO Specs
This specifies how each DeepMIMO version organizes files.

## DeepMIMOv4 Spec

Note: V4 introduces matrix-based storage which significantly improves data loading performance. Instead of loading data user by user, entire matrices can be loaded at once, making operations orders of magnitude faster.

The scenario name is determined solely by the folder name containing the dataset (e.g., `O1_28`, `I2_60`, etc.).

### Primary Matrices

Matrix files follow the pattern: `{matrix_name}_{code}.mat` where code is `t{set_id}_tx{tx_id}_r{rx_set_id}`.
Example with 3 BSs and 1 user grid:
- BS-UE matrices (rx_set_id = 000 for user grid):
  ```
  delay_t001_tx000_r000.mat  # TX Set 1, BS 0 to RX Set 1 (users)
  delay_t001_tx001_r000.mat  # TX Set 1, BS 1 to RX Set 1 (users)
  delay_t001_tx002_r000.mat  # TX Set 1, BS 2 to RX Set 1 (users)
  ```
- BS-BS matrices (rx_set_id = 001 for BS set):
  ```
  delay_t001_tx000_r001.mat  # TX Set 1, BS 0 to RX Set 1 (BSs)
  delay_t001_tx001_r001.mat  # TX Set 1, BS 2 to RX Set 1 (BSs)
  delay_t001_tx002_r001.mat  # TX Set 1, BS 1 to RX Set 1 (BSs)
  ```

Access pattern:
```python
dataset = dm.load(<scen_name>)
```

To simplify the notation, the table below considers `d = dataset[scene][tx]`
Note 1: both d.aoa_az and d['aoa_az'] syntax work
Note 2: static scenarios (single scene) do not need scene indexing. Similarly for single-tx scenarios.

| File Name | Dataset Attribute | Dimensions | Type | Description |
|-----------|------------------|------------|------|-------------|
| rx_loc_{code}.mat    | d.rx_loc | N × 3 | float32 | XYZ coordinates of receiver locations |
| tx_loc_{code}.mat    | d.tx_loc | 1 × 3 | float32 | XYZ coordinates of transmitter location |
| aoa_az_{code}.mat    | d.aoa_az | N × MAX_PATHS | float32 | Azimuth angle of arrival |
| aod_az_{code}.mat    | d.aod_az | N × MAX_PATHS | float32 | Azimuth angle of departure |
| aoa_el_{code}.mat    | d.aoa_el | N × MAX_PATHS | float32 | Elevation angle of arrival |
| aod_el_{code}.mat    | d.aod_el | N × MAX_PATHS | float32 | Elevation angle of departure |
| toa_{code}.mat       | d.toa | N × MAX_PATHS | float32 | Time of arrival |
| phase_{code}.mat     | d.phase | N × MAX_PATHS | float32 | Phase of each path |
| power_{code}.mat     | d.power | N × MAX_PATHS | float32 | Power of each path |
| inter_{code}.mat     | d.inter | N × MAX_PATHS | int32 | Interaction type codes* |
| inter_loc_{code}.mat | d.inter_loc | N × MAX_PATHS × MAX_INTER × 3 | float32 | Interaction point locations |
| vertices_{code}.mat  | d.vertices | N_vert × 3 | float32 | XYZ coordinates of vertices |

*Interaction codes:
- 0: Line of Sight (LoS)
- 1: Single reflection
- 2: Diffraction
- 3: Scattering
- 4: Transmission
- Example: 21 = Tx - Diffraction - Reflection - Rx

The dimensions are:
- `N` - number of users
- `MAX_PATHS` - maximum number of paths DeepMIMO extracts from the ray tracing simulation, varies per scenario
- `MAX_INTER` - maximum number of interactions along one path. The main limitation on this parameter is ray tracing software, which usually stops at 5 or less. 

### JSON Files

| File Name | Description |
|-----------|-------------|
| objects.json | Contains scene object metadata including: name, label, id, face_vertex_idxs (indices into vertices.mat), face_material_idxs (indices of the materials in each face) |
| params.json | Ray tracing parameters including: raytracer info, frequency, max_path_depth, interaction settings (reflections, diffractions, scattering, transmissions), ray casting settings, GPS bounding box, materials, and many raw parameters from the ray tracer that should be sufficient to reproduce the simulation. |

### Secondary (Computed) Matrices

| Dataset Attribute | Dimensions | Type | Description |
|------------------|------------|------|-------------|
| dataset[scene][tx]['chs'] | N_RX_ant × N_TX_ant × N_subcarriers | complex64 | Channel matrices |
| dataset[scene][tx]['distances'] | N × 1 | float32 | TX-RX distances |
| dataset[scene][tx]['pathloss'] | N × 1 | float32 | Path loss values |

Computed matrices are continuously added for convenience of operations. Check `dm.info()` for a complete list.

### Design Principles

1. **Data Efficiency**
   - Eliminate redundant information storage
   - Use efficient data structures and formats
   - Implement lazy evaluation and caching where possible
   - Pre-allocate memory to avoid unnecessary copies

2. **Consistency**
   - Maintain constant antenna configurations across TX/RX
   - Preserve fundamental information (angles, phases, etc.)
   - Flatten single-element lists to reduce nesting

3. **User Experience**
   - Provide transparent access regardless of ray tracer backend
   - Ensure consistent API across different dataset versions
   - Low code access to fundamental matrices
   - Support efficient batch operations

4. **Performance Considerations**
   - GPU acceleration support (optional, via cuPy - yet to implement)
   - Efficient batch processing for channel computations

5. **Transparency and Accessibility**
   - Direct matrix storage with intuitive dimensions for immediate use
   - Human-readable JSON files for parameters and metadata
   - No proprietary parsing required unlike previous versions
   - Open format that simplifies data conversion and interpretation
   - Standardized matrix organization that's self-documenting

## DeepMIMOv3 Spec

### File Structure
N+1 files total, where N = number of RX-TX pairs enabled:
- N data files: One per RX-TX pair
- 1 params file: `params.mat`

Example file names:
- `BS1_BS.mat` - For base station to base station paths (BS-BS communication)
- `BS3_UE_0-1024.mat` - For base station 3 to users 0-1024 (BS-UE communication)

### Primary Matrices

Access patterns:
```python
paths_user_u = dataset[scene][tx_id]['user']['paths'][u]  # For BS-UE communication
paths_bs_b = dataset[scene][tx_id]['bs']['paths'][b]      # For BS-BS communication
```

<table>
<tr>
<th>File Name</th>
<th>Dataset Attribute</th>
<th>Dimensions</th>
<th>Type</th>
<th>Description</th>
</tr>
<tr>
<td rowspan="8">BS{i}_BS.mat<br>or<br>BS{i}_UE_{range}.mat</td>
<td>paths_user_u['AoA_phi']</td>
<td>num_paths × 1</td>
<td>float32</td>
<td>Azimuth angle of arrival</td>
</tr>
<tr>
<td>paths_user_u['AoA_theta']</td>
<td>num_paths × 1</td>
<td>float32</td>
<td>Elevation angle of arrival</td>
</tr>
<tr>
<td>paths_user_u['DoD_phi']</td>
<td>num_paths × 1</td>
<td>float32</td>
<td>Azimuth angle of departure</td>
</tr>
<tr>
<td>paths_user_u['DoD_theta']</td>
<td>num_paths × 1</td>
<td>float32</td>
<td>Elevation angle of departure</td>
</tr>
<tr>
<td>paths_user_u['ToA']</td>
<td>num_paths × 1</td>
<td>float32</td>
<td>Time of arrival</td>
</tr>
<tr>
<td>paths_user_u['phase']</td>
<td>num_paths × 1</td>
<td>float32</td>
<td>Phase of each path</td>
</tr>
<tr>
<td>paths_user_u['power']</td>
<td>num_paths × 1</td>
<td>float32</td>
<td>Power of each path</td>
</tr>
<tr>
<td>paths_user_u['num_paths']</td>
<td>1</td>
<td>int32</td>
<td>Number of paths</td>
</tr>
</table>

*Note: x,y,z coordinates can be lat,long,alt when scenarios come from OSM

## DeepMIMOv2 Spec

Code format: {code} = {scenario}.{bs} where scenario is the environment name (e.g., O1) and bs is the base station ID (e.g., 3p4.1)

### Primary Matrices

Access patterns:
```python
users_dataset = dataset[scene][tx_id]['user']  # For BS-UE communication
bs_dataset = dataset[scene][tx_id]['bs']       # For BS-BS communication
```

Note: Matrix dimensions are specified in {scenario}.params.mat, which contains carrier frequency, number of BSs, transmit power, and user grid information. While there are as many angles as max paths (usually 10) and users in the grid, the matrix is flattened and needs to be parsed per user based on path loss information.

Example file names:
- `{code}.CIR.mat` - Channel impulse response for user data
- `{code}.CIR.BSBS.mat` - Channel impulse response for BS-BS data
- `{code}.DoA.mat` - Direction of arrival for user data

<table>
<tr>
<th>File Name</th>
<th>Dataset Attribute</th>
<th>Type</th>
<th>Description</th>
</tr>
<tr>
<td>{code}.CIR.mat</td>
<td>users_dataset['channel']</td>
<td>complex64</td>
<td>Channel impulse response matrices</td>
</tr>
<tr>
<td>{code}.CIR.BSBS.mat</td>
<td>bs_dataset['channel']</td>
<td>complex64</td>
<td>BS-BS channel impulse response matrices</td>
</tr>
<tr>
<td>{code}.DoA.mat</td>
<td>users_dataset['paths'][u]['DoA']</td>
<td>float32</td>
<td>Angle of arrival information</td>
</tr>
<tr>
<td>{code}.DoA.BSBS.mat</td>
<td>bs_dataset['DoA']</td>
<td>float32</td>
<td>BS-BS angle of arrival information</td>
</tr>
<tr>
<td>{code}.DoD.mat</td>
<td>users_dataset['paths'][u]['DoD']</td>
<td>float32</td>
<td>Angle of departure information</td>
</tr>
<tr>
<td>{code}.DoD.BSBS.mat</td>
<td>bs_dataset['DoD']</td>
<td>float32</td>
<td>BS-BS angle of departure information</td>
</tr>
<tr>
<td>{code}.LoS.mat</td>
<td>paths_user_u['LoS']</td>
<td>int32</td>
<td>Line of sight information</td>
</tr>
<tr>
<td>{code}.LoS.BSBS.mat</td>
<td>bs_dataset['LoS']</td>
<td>int32</td>
<td>BS-BS line of sight information</td>
</tr>
<tr>
<td>{code}.PL.mat</td>
<td>paths_user_u['PL']</td>
<td>float32</td>
<td>Path loss values</td>
</tr>
<tr>
<td>{code}.PL.BSBS.mat</td>
<td>bs_dataset['PL']</td>
<td>float32</td>
<td>BS-BS path loss values</td>
</tr>
</table>

### Design Principles

1. **Data Efficiency**
   - Eliminate redundant information storage
   - Use efficient data structures and formats
   - Implement lazy evaluation and caching where possible
   - Pre-allocate memory to avoid unnecessary copies

2. **Consistency**
   - Maintain constant antenna configurations across TX/RX
   - Preserve fundamental information (angles, phases, etc.)
   - Flatten single-element lists to reduce nesting

3. **User Experience**
   - Provide transparent access regardless of ray tracer backend
   - Ensure consistent API across different dataset versions
   - Low code access to fundamental matrices
   - Support efficient batch operations

4. **Performance Considerations**
   - GPU acceleration support (optional, via cuPy - yet to implement)
   - Efficient batch processing for channel computations

5. **Transparency and Accessibility**
   - Direct matrix storage with intuitive dimensions for immediate use
   - Human-readable JSON files for parameters and metadata
   - No proprietary parsing required unlike previous versions
   - Open format that simplifies data conversion and interpretation
   - Standardized matrix organization that's self-documenting
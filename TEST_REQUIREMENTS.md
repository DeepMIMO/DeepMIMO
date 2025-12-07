# Test Requirements for Mock Reduction

This document lists external files and resources that, if provided, would enable more realistic testing and reduce the need for mocking in the DeepMIMO test suite.

## Current Mock Usage Summary

The test suite currently uses mocking primarily in the following areas:
1. **Converter modules**: Wireless InSite, Sionna RT, AODT converters
2. **Exporter modules**: AODT and Sionna exporters
3. **API module**: File download/upload operations

## Required Files by Module

### 1. Wireless InSite Converter (`deepmimo/converters/wireless_insite/`)

**Purpose**: To test realistic parsing of Wireless InSite project files without mocking XML, setup, and path data.

**Required Files**:
- **Sample Project Folder** containing:
  - `*.setup` file - Project configuration
  - `*.p2m` file - Path data (ray tracing results)
  - `*.xml` file - Material definitions and geometry
  - Scenario XML file with TX/RX set definitions

**Benefit**: Would enable end-to-end tests from reading raw InSite files to producing DeepMIMO datasets, eliminating ~90% of mocks in:
  - `test_xml_parser.py`
  - `test_setup_parser.py`
  - `test_p2m_parser.py`
  - `test_insite_txrx.py`
  - `test_insite_materials.py`
  - `test_insite_scene.py`
  - `test_insite_rt_params.py`
  - `test_insite_converter.py`

**Example Minimal Requirements**:
```
wireless_insite_sample/
├── project.setup         # Scene configuration
├── paths.p2m            # Ray paths (at least 10 RXs, 1 TX, 5 paths)
├── materials.xml        # At least 3 materials (concrete, glass, metal)
└── project.xml          # TX/RX set definitions
```

---

### 2. Sionna RT Converter (`deepmimo/converters/sionna_rt/`)

**Purpose**: To test realistic conversion of Sionna ray tracer output.

**Required Files**:
- **Scene Files**:
  - `.xml` scene description file (for Mitsuba-based scenes)
  - `.ply` mesh file (optional, for complex scenes)
  
- **Path Data**:
  - `.csv` or `.h5` path output file from Sionna RT
  - Path data should include: a, tau, theta_r, theta_t, phi_r, phi_t, types, vertices

**Benefit**: Would eliminate mocking of `h5py`, `xml` parsing, and enable realistic validation of:
  - `test_sionna_paths.py` - Full path processing pipeline
  - `test_sionna_converter.py` - End-to-end conversion
  - `test_sionna_scene.py` - Scene object extraction
  - `test_sionna_materials.py` - Material mapping

**Example Minimal Requirements**:
```
sionna_sample/
├── scene.xml            # Mitsuba scene (or .ply mesh)
├── paths.h5             # Path data (10 UEs, 1 BS, multi-antenna)
└── materials.xml        # Material definitions
```

---

### 3. AODT Converter (`deepmimo/converters/aodt/`)

**Purpose**: To test realistic conversion of Aerial Omniverse Digital Twin parquet files.

**Required Files**:
- **Parquet Files** (small scenario):
  - `scenario.parquet` - Scenario metadata
  - `raypaths.parquet` - Ray path data
  - `cirs.parquet` - Channel impulse responses
  - `ues.parquet` - UE positions and properties
  - `rus.parquet` - RU (Radio Unit / BS) positions and properties
  - `panels.parquet` - Antenna panel configurations
  - `patterns.parquet` - Antenna radiation patterns
  - `objects.parquet` - Scene objects/buildings (optional)

**Benefit**: Would eliminate all `pandas.DataFrame` mocking and enable full integration tests for:
  - `test_aodt_paths.py` - Path and CIR processing
  - `test_aodt_txrx.py` - TX/RX configuration parsing
  - `test_aodt_scene.py` - Scene object loading
  - `test_aodt_rt_params.py` - Parameter extraction
  - `test_aodt_converter.py` - Full conversion pipeline

**Example Minimal Requirements**:
```
aodt_sample/
├── scenario.parquet     # Metadata (1 row)
├── raypaths.parquet     # At least 50 paths
├── cirs.parquet         # Matching CIR data
├── ues.parquet          # At least 10 UEs
├── rus.parquet          # 1-2 RUs
├── panels.parquet       # 1-2 panel configs
└── patterns.parquet     # Isotropic or simple pattern
```

---

### 4. API Module (`deepmimo/api.py`)

**Purpose**: To test realistic file download, upload, and scenario management.

**Required Files**:
- **Sample Scenario Package** (`.zip` file):
  - Contains a valid DeepMIMO scenario folder structure
  - Includes `params.json` and generated dataset files
  - Small size (~1-5 MB) for fast testing

**Benefit**: Would eliminate network request mocking and enable:
  - Real file download/upload testing (using local test server)
  - Hash verification
  - Zip/unzip operations
  - File integrity checks

**Example**:
```
test_scenario.zip
└── TestScenario/
    ├── params.json
    ├── power_t001_tx001_r001.npz
    ├── delay_t001_tx001_r001.npz
    └── ...
```

---

### 5. Exporter Modules (`deepmimo/exporters/`)

**Purpose**: To validate export functionality against known good outputs.

**Required Reference Files**:

#### AODT Exporter:
- Sample `.parquet` export files with known structure
- Example queries for validation

#### Sionna Exporter:
- Sample Sionna scenario XML that can be imported back into Sionna RT
- Reference channel realizations for comparison

**Benefit**: Would enable round-trip testing (export → import → compare) without mocking `pandas`, `pyarrow`, or `sionna`.

---

## Summary of Mock Reduction Potential

| Module | Current Mocks | Files Needed | Estimated Mock Reduction |
|--------|---------------|--------------|--------------------------|
| Wireless InSite | DataFrame, XML, file I/O | Sample project (4-5 files) | ~90% |
| Sionna RT | h5py, XML, arrays | Scene + paths (2-3 files) | ~85% |
| AODT | pandas, parquet | 7 parquet files | ~95% |
| API | requests, network | Sample .zip scenario | ~70% |
| Exporters | pandas, sionna | Reference exports | ~60% |

---

## Recommended Minimal Test Dataset

A single comprehensive test dataset covering all converters:

```
test_datasets/
├── wireless_insite/
│   └── small_urban/
│       ├── project.setup
│       ├── paths.p2m
│       ├── materials.xml
│       └── project.xml
├── sionna/
│   └── simple_street/
│       ├── scene.xml
│       └── paths.h5
├── aodt/
│   └── micro_scenario/
│       ├── scenario.parquet
│       ├── raypaths.parquet
│       ├── cirs.parquet
│       ├── ues.parquet
│       ├── rus.parquet
│       ├── panels.parquet
│       └── patterns.parquet
└── scenarios/
    └── test_scenario.zip
```

**Total Size Estimate**: 10-20 MB (compressed)

---

## Current Testing Strategy

Until the above files are available, the test suite uses:
1. **Realistic data structures** where possible (e.g., numpy arrays with proper shapes)
2. **Minimal mocking** for I/O operations only
3. **Integration tests** for core logic without mocking internal functions
4. **Mock verification** to ensure mocks match real API signatures

This approach provides good coverage (currently **74%**) while maintaining test reliability and speed.


# Testing Status and Plan

## Coverage Overview
Current total coverage is approximately 73%. Key modules with significant test coverage improvements include:
- `deepmimo.api`: 69% (up from 45%)
- `deepmimo.generator.dataset`: 59% (up from 50%)
- `deepmimo.summary`: 81% (up from ~5%)
- `deepmimo.web_export`: 85% (up from ~0%)
- `deepmimo.generator.visualization`: 70% (up from ~0%)

## Mock Usage vs. Realistic Tests

### Modules using Heavy Mocking
These modules interact with external systems (API, File System, Plotting libraries) and rely heavily on mocks to ensure isolation and speed.
- **`deepmimo.api`**: Mocks `requests`, `zipfile`, `shutil`, `os` operations.
- **`deepmimo.summary`**: Mocks `matplotlib`, `load_dict_from_json`, and `Dataset` objects.
- **`deepmimo.generator.visualization`**: Mocks `matplotlib.pyplot` and `Axes` to verify plotting calls without rendering.
- **`deepmimo.web_export`**: Mocks file system operations and internal processing steps.
- **Converters (`wireless_insite`, `sionna_rt`, `aodt`)**: Mock input file reading (XML, Parquet) to test parsing logic without needing large data files.

### Modules using Realistic/Semi-Realistic Tests
These modules test core logic and algorithms using real objects or minimal mocking.
- **`deepmimo.generator.dataset`**: Tests use real `Dataset` instances (populated with synthetic data) to verify trimming, coordinate transformation, and array manipulation logic. `test_dataset_advanced.py` is a good example.
- **`deepmimo.generator.geometry`**: Tests mathematical functions (`_apply_FoV_batch`, `_rotate_angles_batch`) with real numpy arrays.
- **`deepmimo.txrx`**: Tests `TxRxSet` and `TxRxPair` behavior with real objects.
- **`deepmimo.scene`**: Tests scene graph construction and object management with real `Scene` and `PhysicalElement` objects.

## E2E Testing Plan

To further improve reliability, we recommend the following End-to-End (E2E) test scenarios:

1. **Full Conversion Pipeline (Mocked Input)**
   - **Goal**: Verify that `convert()` correctly orchestrates parsing, processing, and saving.
   - **Strategy**: Create a small, synthetic set of ray-tracing output files (e.g., a minimal Wireless InSite project folder). Run `dm.convert()` and verify the output `Dataset` structure and saved files.
   
2. **Dataset Loading and Manipulation**
   - **Goal**: Verify that `load()` correctly restores a saved dataset and that subsequent operations (trimming, combining) work as expected.
   - **Strategy**: Save a synthetic `Dataset` to disk. Load it back using `dm.load()`. Apply `trim()`, `combine()`, and checking consistency.

3. **API Integration (Sandbox)**
   - **Goal**: Verify actual interaction with the DeepMIMO API.
   - **Strategy**: This requires a sandbox/staging environment for the API to avoid polluting production data. Tests would perform real `upload` and `download` operations. (Currently mocked).

4. **Visualization Output**
   - **Goal**: Verify that visualization functions produce valid image files.
   - **Strategy**: Run `plot_summary` or `plot_rays` with `save_imgs=True` and check if valid image files are created (e.g., verify file headers), instead of just checking if `savefig` was called.


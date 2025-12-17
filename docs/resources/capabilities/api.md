# API Module Capabilities

This document details the capabilities of the DeepMIMO API module, which provides functionality for interacting with the DeepMIMO database including downloading scenarios, searching the database, and uploading new scenarios.

## Overview

The API module supports three main operations:
- **Download**: Retrieve scenarios and ray-tracing source files
- **Search**: Query the database for scenarios matching criteria  
- **Upload**: Submit new scenarios to the database

---

## Download Functionality

Downloads scenarios and ray-tracing source files from the DeepMIMO database.

| Feature | Support | Notes |
|---|---|---|
| **Download Types** | | |
| Scenario download | ✓ | Complete DeepMIMO scenarios |
| RT source download | ✓ | Original ray-tracing files |
| | | |
| **File Handling** | | |
| ZIP file download | ✓ | Compressed archives |
| Automatic extraction | ✓ | Unzips to appropriate directory |
| Progress bar | ✓ | tqdm-based progress indicator |
| Resume capability | ✗ [a] | Downloads restart from beginning |
| | | |
| **Storage Locations** | | |
| Default scenarios folder | ✓ | `~/.deepmimo/scenarios/` |
| Default RT sources folder | ✓ | `~/.deepmimo/rt_sources/` |
| Custom output directory | ✓ | User-specified path |
| | | |
| **Validation** | | |
| Existing file check | ✓ | Skips if already downloaded |
| File size verification | ✗ [b] | No checksum validation |
| Extraction verification | ✓ | Checks for successful unzip |
| | | |
| **Security** | | |
| Secure download URLs | ✓ | Token-based authentication |
| HTTPS only | ✓ | Encrypted transfers |
| Temporary URL generation | ✓ | Time-limited access |
| | | |
| **Error Handling** | | |
| Network errors | ✓ | Graceful failure with messages |
| Server errors | ✓ | HTTP status code handling |
| Timeout handling | ✓ | Configurable timeout |
| Disk space errors | ✓ | Exception propagation |
| | | |
| **Integration** | | |
| Auto-download in load() | ✓ | Prompts user if scenario missing |
| Direct URL access | ✓ | `download_url()` for external use |
| Batch download | ✗ [c] | Single scenario per call |

---

## Search Functionality

Searches the DeepMIMO database for scenarios matching specified criteria.

| Feature | Support | Notes |
|---|---|---|
| **Search Criteria** | | |
| Frequency bands | ✓ | sub6, mmW, subTHz |
| Raytracer name | ✓ | AODT, Sionna, Wireless InSite |
| Environment type | ✓ | Indoor, outdoor, or all |
| Number of TXs | ✓ | Min/max range filter |
| Number of RXs | ✓ | Min/max range filter |
| Path depth | ✓ | Min/max range filter |
| Max reflections | ✓ | Min/max range filter |
| Number of rays | ✓ | Min/max range filter |
| Multi-RX antenna | ✓ | Boolean filter |
| Multi-TX antenna | ✓ | Boolean filter |
| Dual polarization | ✓ | Boolean filter |
| BS-to-BS paths | ✓ | Boolean filter |
| Dynamic scenarios | ✓ | Boolean filter |
| Diffraction support | ✓ | Boolean filter |
| Scattering support | ✓ | Boolean filter |
| Transmission support | ✓ | Boolean filter |
| Digital twin | ✓ | Boolean filter |
| City name | ✓ | Text filter |
| GPS bounding box | ✓ | Lat/lon coordinates |
| Has RT source | ✓ | Boolean filter |
| | | |
| **Query Types** | | |
| Empty query | ✓ | Returns all scenarios |
| Single criterion | ✓ | Filter by one field |
| Multiple criteria | ✓ | Combined filters (AND logic) |
| Range filters | ✓ | Min/max numeric ranges |
| Boolean filters | ✓ | True/false/all options |
| | | |
| **Results** | | |
| Scenario list | ✓ | Array of scenario names |
| Result count | ✓ | Number of matches |
| JSON format | ✓ | Structured response |
| | | |
| **Error Handling** | | |
| HTTP errors | ✓ | Detailed error messages |
| Connection errors | ✓ | Network failure handling |
| Timeout errors | ✓ | Request timeout handling |
| Invalid queries | ✓ | Server validation |
| | | |
| **Integration** | | |
| Direct download | ✓ [d] | Use results with `download()` |
| Load pipeline | ✓ [d] | Use results with `load()` |

---

## Upload Functionality

Uploads new scenarios and ray-tracing source files to the DeepMIMO database.

| Feature | Support | Notes |
|---|---|---|
| **Upload Types** | | |
| Scenario upload | ✓ | Complete DeepMIMO datasets |
| RT source upload | ✓ | Original ray-tracing files |
| Image upload | ✓ | Visualization images |
| | | |
| **Authentication** | | |
| API key required | ✓ | User authentication |
| Key validation | ✓ | Server-side verification |
| Secure uploads | ✓ | HTTPS/presigned URLs |
| | | |
| **File Processing** | | |
| Automatic zipping | ✓ | Compresses before upload |
| SHA1 hash calculation | ✓ | File integrity verification |
| File size limits | ✓ | 5GB scenario, 20GB RT source |
| Progress tracking | ✓ | tqdm-based progress bar |
| Chunked upload | ✓ | Large file support |
| | | |
| **Metadata Processing** | | |
| Parameter extraction | ✓ | From params.json |
| Key component generation | ✓ | Summary statistics |
| Frequency band detection | ✓ | sub6/mmW/subTHz |
| Feature flag extraction | ✓ | Multi-ant, dual-pol, etc. |
| GPS bbox extraction | ✓ | Geographic bounds |
| | | |
| **Image Processing** | | |
| Auto-generate images | ✓ | Using `plot_summary()` |
| Custom images | ✓ | User-provided images |
| Format validation | ✓ | PNG, JPG, etc. |
| Size limits | ✓ | Per-image size limit |
| Count limits | ✓ | Max images per upload |
| Image naming | ✓ | Structured naming convention |
| | | |
| **Submission Options** | | |
| Full upload | ✓ | Data + metadata |
| Submission only | ✓ | Metadata without files [e] |
| Include images | ✓ | Optional image upload |
| Custom metadata | ✓ | Extra fields |
| Scenario details | ✓ | Description and notes |
| | | |
| **Validation** | | |
| File existence | ✓ | Checks before upload |
| Size validation | ✓ | Enforces limits |
| Format validation | ✓ | JSON/ZIP checks |
| Parameter validation | ✓ | Required fields |
| Filename sanitization | ✓ | Server-side processing |
| | | |
| **Error Handling** | | |
| File not found | ✓ | Clear error messages |
| Size limit exceeded | ✓ | Pre-upload validation |
| Network errors | ✓ | Graceful failure |
| Server errors | ✓ | HTTP status handling |
| Authentication errors | ✓ | Invalid key detection |
| | | |
| **Progress Reporting** | | |
| File upload progress | ✓ | Byte-level progress |
| Stage indicators | ✓ | Shows current operation |
| Time estimates | ✓ | From tqdm |
| | | |
| **Integration** | | |
| Post-conversion upload | ✓ | Upload after converter |
| Batch uploads | ✗ [f] | Single scenario per call |

---

## Usage Examples

### Download

```python
import deepmimo as dm

# Download scenario
dm.download('O1_60')

# Download RT source
dm.download('O1_60', rt_source=True)

# Custom output directory
dm.download('O1_60', output_dir='./my_scenarios')
```

### Search

```python
import deepmimo as dm

# Search for mmWave scenarios
results = dm.search({
    'bands': ['mmW'],
    'environment': 'outdoor',
    'multiTxAnt': True
})

# Download search results
for scenario_name in results:
    dm.download(scenario_name)
```

### Upload

```python
import deepmimo as dm

# Upload scenario with API key
dm.upload(
    scenario_name='my_scenario',
    key='your_api_key',
    details='Custom urban scenario',
    include_images=True
)

# Upload RT source separately
dm.upload_rt_source(
    scenario_name='my_scenario',
    rt_source_path='/path/to/rt/files',
    key='your_api_key'
)
```

---

## Notes

- **[a]** No resume capability; interrupted downloads must restart. Consider external download manager for large files.
- **[b]** File integrity verified during extraction, but no pre-extraction checksum validation.
- **[c]** Batch downloads require looping; consider `search()` + loop for multiple scenarios.
- **[d]** Search returns scenario names that can be directly passed to `download()` or `load()`.
- **[e]** Submission-only mode useful when files already uploaded; updates metadata only.
- **[f]** Batch uploads not supported; use loop for multiple scenarios.

---

## API Endpoints

Internal endpoints (for reference):

| Endpoint | Purpose | Method |
|---|---|---|
| `/api/download/secure` | Get scenario download URL | GET |
| `/api/search/scenarios` | Search scenarios | POST |
| `/api/b2/authorize-upload` | Get upload authorization | GET |
| `/api/submit` | Create scenario submission | POST |

---

## Configuration

Default values (can be customized in source):

| Parameter | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://deepmimo.net` | API server URL |
| `REQUEST_TIMEOUT` | 300 seconds | HTTP timeout |
| `FILE_SIZE_LIMIT` | 5 GB | Scenario size limit |
| `RT_FILE_SIZE_LIMIT` | 20 GB | RT source size limit |
| `IMAGE_SIZE_LIMIT` | 10 MB | Per-image size limit |
| `MAX_IMAGES_PER_UPLOAD` | 10 | Max images per submission |

---

## Common Limitations

These limitations apply across API operations:

- **Network dependency**: All operations require internet connection.
- **Single scenario**: Operations handle one scenario at a time.
- **No caching**: Search results not cached; queries sent to server each time.
- **Rate limiting**: Server may enforce rate limits (not currently documented).
- **File size limits**: Large scenarios/sources may exceed limits.

---

## Related Documentation

- [Download Tutorial](../../tutorials/1_getting_started.py) - Using download in workflows
- [Search API Reference](../../api/api.md#search) - Search parameter details
- [Upload API Reference](../../api/api.md#upload) - Upload parameter details

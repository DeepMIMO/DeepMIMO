# Integrations Module Capabilities

This document details the capabilities of the DeepMIMO integrations module, which provides compatibility with external tools and platforms.

## Overview

The integrations module currently supports:
- **Sionna Adapter**: Convert DeepMIMO datasets to Sionna format for use with NVIDIA Sionna
- **Web Export**: Export datasets to binary format for the DeepMIMO web visualizer

---

## Sionna Adapter

Converts DeepMIMO datasets to the format expected by Sionna's neural network training pipelines.

| Feature | Support | Notes |
|---|---|---|
| **Input Formats** | | |
| Single Dataset | ✓ | Single TX-RX pair |
| MacroDataset | ✓ | Multiple TX-RX pairs |
| DynamicDataset | ✗ [a] | Time-varying not supported |
| | | |
| **Channel Types** | | |
| Time domain channels | ✓ | Required format |
| Frequency domain channels | ✗ [b] | Not supported by adapter |
| | | |
| **Data Extraction** | | |
| Channel coefficients | ✓ | Returns `(num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps)` |
| Path delays | ✓ | Returns `(num_rx, num_tx, num_paths)` |
| Path powers | ✓ | Extracted from channel coefficients |
| AoA/AoD angles | ✗ [c] | Not included in Sionna format |
| | | |
| **Configuration** | | |
| Multi-BS support | ✓ | All BSs in dataset included |
| Multi-UE support | ✓ | All UEs in dataset included |
| Subset selection | ✓ [d] | Subset dataset before passing to adapter |
| Antenna configuration | ✓ | Extracts num_rx_ant, num_tx_ant from dataset |
| | | |
| **Output Format** | | |
| Sionna channel shape | ✓ | Matches Sionna's expected dimensions |
| Batch iteration | ✓ | `__getitem__` for batch access |
| Length property | ✓ | Total number of samples |
| NumPy arrays | ✓ | Direct NumPy output |
| | | |
| **Validation** | | |
| Shape validation | ✓ | Checks channel dimensions |
| Antenna count validation | ✓ | Ensures consistent antenna counts |
| Channel type validation | ✓ | Raises error if frequency domain |

---

## Web Export

Exports DeepMIMO datasets to binary format for the web-based 3D visualizer.

| Feature | Support | Notes |
|---|---|---|
| **Input Formats** | | |
| Single Dataset | ✓ | Single TX-RX pair |
| MacroDataset | ✓ | Multiple TX-RX pairs with metadata |
| DynamicDataset | ✗ [e] | Requires snapshot selection |
| | | |
| **Exported Data** | | |
| Path power | ✓ | Per-path power values |
| Path delays | ✓ | Per-path delay values |
| AoA azimuth | ✓ | Angle of arrival (azimuth) |
| AoA elevation | ✓ | Angle of arrival (elevation) |
| AoD azimuth | ✓ | Angle of departure (azimuth) |
| AoD elevation | ✓ | Angle of departure (elevation) |
| TX positions | ✓ | 3D coordinates |
| RX positions | ✓ | 3D coordinates |
| Channel matrices | ✗ [f] | Not included (size constraint) |
| Interaction positions | ✗ [f] | Not included (size constraint) |
| | | |
| **Binary Format** | | |
| Custom binary encoding | ✓ | Optimized format with dtype, shape, data |
| Float32 precision | ✓ | Default data type |
| Float64 precision | ✓ | Optional data type |
| Shape preservation | ✓ | Array dimensions preserved |
| Metadata JSON | ✓ | TX/RX set information |
| | | |
| **TX/RX Set Handling** | | |
| Single TX/RX set | ✓ | Simple export |
| Multiple TX/RX sets | ✓ | Separate files per set |
| TX/RX set IDs | ✓ | Included in metadata |
| Set count tracking | ✓ | Total sets in metadata |
| File naming convention | ✓ | `{dataset_name}_tx{tx_id}_rx{rx_id}.bin` |
| | | |
| **Output Structure** | | |
| Binary data files | ✓ | One per TX-RX pair |
| Metadata file | ✓ | JSON with set information |
| Directory organization | ✓ | Organized by dataset name |
| File size optimization | ✓ | Efficient binary encoding |
| | | |
| **Validation** | | |
| Array validation | ✓ | Checks for valid NumPy arrays |
| Path existence | ✓ | Creates output directories |
| File write errors | ✓ | Exception handling |

---

## Usage Examples

### Sionna Adapter

```python
import deepmimo as dm
from deepmimo.integrations.sionna_adapter import SionnaAdapter

# Load dataset
dataset = dm.generate('O1_60')

# Create adapter
adapter = SionnaAdapter(dataset)

# Get batch of samples
for batch_idx in range(len(adapter)):
    channels, delays = adapter[batch_idx]
    # Use with Sionna neural network training
```

### Web Export

```python
import deepmimo as dm

# Load dataset
dataset = dm.generate('O1_60')

# Export for web visualizer
dm.export_dataset_to_binary(
    dataset,
    dataset_name='O1_60',
    output_dir='./web_datasets'
)
```

---

## Notes

- **[a]** DynamicDataset support requires selecting specific time snapshot before conversion.
- **[b]** Sionna adapter requires time-domain channels. Set `freq_domain=False` in `ch_params` when generating.
- **[c]** Angle information not exported to Sionna format as it's not used in standard Sionna workflows.
- **[d]** To export subset, trim dataset before passing to adapter: `dataset.trim(idxs=[...])`.
- **[e]** DynamicDataset can export individual snapshots: `dynamic_dataset[snapshot_idx]`.
- **[f]** Channel matrices and interaction positions omitted from web export due to file size constraints.

---

## Common Limitations

These limitations apply to both integrations:

- **Single frequency**: Both integrations work with single-frequency datasets only.
- **Memory constraints**: Large datasets may require subsetting before export.
- **Data precision**: Web export uses float32 by default for file size optimization.
- **Static snapshots**: Dynamic/time-varying datasets require manual snapshot selection.

---

## Related Documentation

- [Datasets Capabilities](datasets.md) - Dataset manipulation and features
- [Sionna Integration Tutorial](../../tutorials/7_converters.py#sionna-integration) - Sionna adapter usage
- [Web Visualization](../../api/integrations.md) - Web export API reference

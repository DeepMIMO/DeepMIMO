# DeepMIMO
DeepMIMO Toolchain: Bridging RayTracers and 5G Simulators

**[Goal]** Enabling large-scale AI benchmarking using site-specific wireless raytracing datasets.

**[How]** Converting the outputs of the best wireless propagation ray tracers to a distributable format readable by the best simulation toolboxes. 

**[Why]** To revolutionize AI-driven wireless research by making ray tracing data easy to access, share, and benchmark.

## Project Structure
```
deepmimo/
├── api.py                  # API interface with DeepMIMO database
├── scene.py                # Scene (3D environment) management
├── consts.py               # Constants and configurations
├── info.py                 # Information on matrices and parameters
├── materials.py            # Material properties
├── txrx.py                 # Transmitter and receiver
├── rt_params.py            # Ray tracing parameters
├── general_utils.py        # Utility functions
├── converters/             # Ray tracer output converters
│   ├── aodt/               # AODT converter
│   ├── sionna_rt/          # Sionna RT converter
│   ├── wireless_insite/    # Wireless Insite converter
│   ├── converter.py        # Base converter class
│   └── converter_utils.py  # Converter utilities
├── exporters/              # Data exporters
│   ├── aodt_exporter.py    # AODT format exporter
│   └── sionna_exporter.py  # Sionna format exporter
├── dataset/                # Dataset management
│   ├── dataset.py          # Dataset class and management
│   ├── load.py             # Dataset loading functions
│   ├── sampling.py         # User sampling utilities
│   └── array_wrapper.py    # Array management utilities
├── generator/              # Channel generation
│   ├── core.py             # Core generation functionality
│   ├── channel.py          # Channel generation
│   ├── geometry.py         # Geometric calculations
│   └── ant_patterns.py     # Antenna pattern definitions
├── integrations/           # Integrations with 5G simulation tools
│   ├── sionna_adapter.py   # Sionna integration
│   └── matlab/             # Matlab 5GNR integration
└── pipelines/              # Automatic raytracing pipelines
    ├── sionna_rt/          # Sionna raytracer pipeline
    ├── wireless_insite/    # Wireless Insite pipeline
    ├── blender_osm.py      # Blender OSM export utilities
    ├── TxRxPlacement.py    # Transmitter/Receiver placement
    └── utils/              # Pipeline utilities

Additional directories:
├── deepmimo_v3/            # V3 Version for OFDM generation checks
├── scripts/                # Utility scripts and pipelines
├── docs/                   # Documentation
└── test/                   # Test suite
```

## Installation

### Basic Installation
```bash
pip install deepmimo
```

### Development Installation
```bash
git clone https://github.com/DeepMIMO/DeepMIMO.git
cd DeepMIMO
pip install -e .
```

## Usage Examples

### Basic Dataset Generation
```python
import deepmimo as dm

# Load a dataset
dataset = dm.load('asu_campus_3p5')

# Generate channel
channel = dataset.compute_channels()
```

### Convert Ray Tracing Simulations to DeepMIMO
```python
import deepmimo as dm

# Convert Wireless Insite, Sionna, or AODT to DeepMIMO
converter = dm.convert('path_to_ray_tracing_output')
```

### Download and Upload Datasets
```python
import deepmimo as dm

# Download a dataset
dm.download('asu_campus_3p5')

# Upload a dataset to the DeepMIMO (after local conversion)
dm.upload('my_scenario', 'your-api-key')
# get key in "contribute" in deepmimo.net
```

## Building Documentation

| Step    | Command                                           | Description                       |
|---------|---------------------------------------------------|-----------------------------------|
| Install | `pip install .[doc]`                              | Install docs dependencies         |
| Build   | `cd docs`<br>`sphinx-build -b html . _build/html` | Generate HTML documentation       |
| Serve   | `cd docs/_build/html`<br>`python -m http.server`  | View docs at http://localhost:8000|

## Contributing

We welcome contributions to DeepMIMO! To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

We aim to respond to pull requests within 24 hours.

## Citation

If you use this software, please cite it as:

```bibtex
@misc{alkhateeb2019deepmimo,
      title={DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications}, 
      author={Ahmed Alkhateeb},
      year={2019},
      eprint={1902.06435},
      archivePrefix={arXiv},
      primaryClass={cs.IT},
      url={https://arxiv.org/abs/1902.06435}, 
}
```

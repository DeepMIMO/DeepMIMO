<div align="center">
  <h1>DeepMIMO</h1>
  <p><i>Bridging ray tracers and 5G/6G simulators with shareable, site-specific datasets</i></p>
  <p>
    <a href="https://pypi.org/project/deepmimo/"><img alt="PyPI 4.0.0b" src="https://img.shields.io/badge/PyPI-4.0.0b-blue"></a>
    <a href="https://www.python.org/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-blue"></a>
    <a href="https://deepmimo.net"><img alt="Docs" src="https://img.shields.io/badge/docs-deepmimo.net-brightgreen"></a>
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/DeepMIMO/DeepMIMO.svg"></a>
    <a href="https://github.com/astral-sh/uv"><img alt="uv" src="https://img.shields.io/badge/uv-speedy%20installs-ff69b4"></a>
    <a href="https://docs.astral.sh/ruff/"><img alt="ruff" src="https://img.shields.io/badge/lint-ruff-F74C00"></a>
  </p>
  <p>
    Convert world‑class ray tracers (Sionna RT, Wireless InSite, AODT) into portable datasets that plug directly into leading simulators (Sionna, MATLAB 5G, and more).
  </p>
  <img src="docs/assets/dm.gif" alt="DeepMIMO animated showcase" width="800"/>
</div>

<p align="center">
  <a href="#installation">Install</a> •
  <a href="#usage-examples">Quickstart</a> •
  <a href="https://deepmimo.net">Docs</a> •
  <a href="docs/resources">Resources</a> •
  <a href="#faq">FAQ</a> •
  <a href="#citation">Cite</a>
</p>

<table style="width:100%; text-align:center; border-collapse:separate; border-spacing:0 6px;">
  <thead>
    <tr>
      <th>GOAL</th>
      <th style="border-left:1px solid #eaeaea; border-right:1px solid #eaeaea;">HOW</th>
      <th>WHY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Enable large‑scale AI benchmarking using site‑specific wireless ray‑tracing datasets.</td>
      <td style="border-left:1px solid #eaeaea; border-right:1px solid #eaeaea;">Convert outputs from top propagation ray tracers into a clean, distributable format readable by modern simulation toolboxes.</td>
      <td>Make ray‑tracing data easy to access, share, and benchmark—accelerating AI‑driven wireless research.</td>
    </tr>
  </tbody>
  </table>

## Features

- 🚀 Plug-and-play datasets for simulators: **Sionna**, **MATLAB 5G**, and more
- 🔄 Converters for major ray tracers: **Sionna RT**, **Wireless InSite**, **AODT**
- 🧱 Modular pipelines for conversion, validation, export, and distribution
- 📦 Exporters for clean, reproducible, shareable formats
- 📚 Rich documentation, examples, and scenarios to get you started fast
- 🐍 **Python 3.11+**, supports both `pip` and `uv`

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
├── generator/              # Dataset generator
│   ├── core.py             # Core generation functionality
│   ├── dataset.py          # Dataset class and management
│   ├── channel.py          # Channel generation
│   ├── geometry.py         # Geometric calculations
│   ├── ant_patterns.py     # Antenna pattern definitions
│   ├── array_wrapper.py    # Array management utilities
│   ├── visualization.py    # Visualization tools
│   └── generator_utils.py  # Generator utilities
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
pip install --pre deepmimo
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

| Step    | Command                              | Description                        |
|---------|--------------------------------------|------------------------------------|
| Install | `pip install .[doc]`                 | Install docs dependencies          |
| Build   | `mkdocs build`                       | Generate static site into `site/`  |
| Serve   | `mkdocs serve -a 0.0.0.0:8000`       | Preview at http://localhost:8000   |

## Contributing

We welcome contributions to DeepMIMO! To contribute:
1. Fork
2. Change
3. Pull Request

We aim to respond to pull requests within 24 hours.

## FAQ

- Q: Is the package in beta?  
  A: Yes. Install with `pip install --pre deepmimo`.

- Q: What Python versions are supported?  
  A: **Python 3.11+**.

- Q: Which ray tracers are supported?  
  A: **Sionna RT**, **Wireless InSite**, and **AODT** via dedicated converters.

- Q: Which simulators can consume DeepMIMO datasets?  
  A: **Sionna**, **MATLAB 5G**, and other toolboxes via our exporters/integrations.

- Q: Any tips for large datasets?  
  A: Use batching, prefer SSDs, and export intermediate artifacts to avoid recomputation.

## Citation

If you use this software, please cite it:

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

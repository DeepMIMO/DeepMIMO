# Installation

DeepMIMO requires Python 3.11 or later.

## Quick Install

```bash
pip install deepmimo
```

## Source Install

From source:
```bash
git clone https://github.com/DeepMIMO/DeepMIMO.git
cd DeepMIMO
pip install .
```

## Development Install

Install dependencies from the table below based on your needs.

| Method | Command | Description |
|--------|---------|-------------|
| Base | `pip install -e .` | Basic install with core dependencies |
| Development | `pip install -e .[dev]` | Full development environment (includes docs) |
| Pipelines (Sionna 2.0) | `pip install -e .[sionna]` | Ray tracing pipeline with Sionna RT 2.0+ |
| All | `pip install -e .[all]` | Complete installation |

*Note: The `-e` flag makes it so changes in the code are automatically reflected without reinstalling.*

💡 **TIP**: For faster installation, use `uv`:
```bash
pip install uv
uv pip install .[sionna]
```

## Sionna RT Requirements

The `[sionna]` extra requires:
- Python 3.11+
- `sionna-rt>=2.0.1`
- `mitsuba==3.8.0`
- `drjit==1.3.1`
- Native Linux for GPU ray tracing (Windows/WSL2 falls back to CPU — upstream OptiX limitation)

## Previous versions

As a commitment to support reproducible research, we try to always support all versions.

Previous versions are (or will be) available via:
```bash
pip install deepmimo==2.0.0
pip install deepmimo==3.0.0
```

However, if actively working with DeepMIMO, it is advised to migrate the code to v4.
The datasets are exactly the same, the results and parameters are the same too. But there are small code changes that are necessary.

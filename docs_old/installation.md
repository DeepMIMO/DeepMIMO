# Installation

DeepMIMO requires Python 3.10 or later (Python 3.11 recommended for development).

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

And install dependencies from the table below based on your needs.

| Method | Command | Python Version | Description |
|--------|---------|---------------|-------------|
| Base | `pip install -e .` | ≥3.10 | Basic install with core dependencies |
| Documentation | `pip install .[doc]` | ≥3.10 | Install with documentation dependencies |
| Development | `pip install .[dev]` | ≥3.11 | Full development environment |
| Pipelines (Sionna 1.0.x) | `pip install .[sionna1]` | 3.10 or 3.11 | Ray tracing pipeline with Sionna 1.0 |
| Pipelines (Sionna 0.19.x) | `pip install .[sionna019]` | 3.10 or 3.11 | Ray tracing pipeline with Sionna 0.19 |
| All | `pip install .[all]` | 3.10 or 3.11 | Complete installation |

*Note: This is a source installation only installs dependencies. The package itself will be imported directly from the source folder, allowing for immediate testing of code changes.*

💡 **TIP**: The `-e` flag in `pip install -e .` to make it so the changes in the code are automatically reflected on the package, and no need for reinstalling. 

💡 **TIP**: For faster installation, use `uv`:
```bash
pip install uv
uv pip install <...>
```

## Previous versions

As a commitment to support reproducible research, we try to always support all versions. 

Previous versions are (or will be) available via:
```bash
pip install deepmimo==2.0.0
pip install deepmimo==3.0.0
```

However, if actively working with DeepMIMO, it is advised to migrate the code to v4. 
The datasets are exactly the same, the results and parameters are the same too. But there are small code changes that are necessary. 

# Contributing

Thank you for your interest in contributing to DeepMIMO! This guide will help you get started.

## Development Setup

1. Fork and clone in one step using the [GitHub CLI](https://cli.github.com/):
   ```bash
   gh repo fork DeepMIMO/DeepMIMO --clone
   cd DeepMIMO
   ```

2. Install development dependencies with [uv](https://docs.astral.sh/uv/):
   ```bash
   uv sync --extra dev
   ```

   That's it — `uv` creates an isolated `.venv` automatically and installs
   the package in editable mode.  No separate virtual-environment creation
   or `pip install` step is needed.

## Code Style

We follow PEP 8 with some modifications:
- Line length: 100 characters
- Use Google-style docstrings

## Versioning
<global_format_rules>.<converter_version>.<generator_version>

## Documentation Guidelines

### 1. Module-Level Docstrings
```python
"""
Module Name.

Brief description of the module's purpose.

This module provides:
- Feature/responsibility 1
- Feature/responsibility 2
- Feature/responsibility 3

The module serves as [main role/purpose].
"""
```

### 2. Function Docstrings
```python
def function_name(param1: type, param2: type = default) -> return_type:
    """Brief description of function purpose.
    
    Detailed explanation if needed.

    Args:
        param1 (type): Description of param1
        param2 (type, optional): Description of param2. Defaults to default.

    Returns:
        return_type: Description of return value

    Raises:
        ErrorType: Description of when this error is raised
    """
```

### 3. Class Docstrings
```python
class ClassName:
    """Brief description of class purpose.
    
    Detailed explanation of class functionality and usage.

    Attributes:
        attr1 (type): Description of attr1
        attr2 (type): Description of attr2

    Example:
        >>> example usage code
        >>> more example code
    """
```

### 4. Code Organization

Here's an example of how to organize your code:

```python
"""Module docstring."""

# Standard library imports
import os
import sys

# Third-party imports
import numpy as np
import scipy

# Local imports
from . import utils
from .core import Core

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

CONSTANT_1 = value1
CONSTANT_2 = value2

#------------------------------------------------------------------------------
# Helper Functions
#------------------------------------------------------------------------------

def helper_function():
    """Helper function docstring."""
    pass

#------------------------------------------------------------------------------
# Main Classes
#------------------------------------------------------------------------------

class MainClass:
    """Main class docstring."""
    pass
```

## Testing

Currently tests are executed using the DeepMIMO manual. 
Future work will create specific tests in /tests, mainly for CI/CD purposes.

## Documentation

Build documentation:

| Step    | Command                                           | Description                       |
|---------|---------------------------------------------------|-----------------------------------|
| Install | `pip install .[dev]`                             | Install development dependencies  |
| Build   | `mkdocs build`                                   | Generate HTML documentation       |
| Serve   | `mkdocs serve`                                   | View docs at http://localhost:8000|

### Pre-rendering Notebooks

Notebooks live as percent-format `.py` files (edited by humans) alongside
`.ipynb` files (stored outputs for the docs).  The docs site is built with
`execute: false`, so it only displays pre-stored outputs — **you must
re-render any notebook you change** before committing.

```bash
# Activate the environment first (needs jupytext + jupyter + deepmimo[sionna])
source .venv/bin/activate          # or: conda activate dm_env

# Re-render a single notebook
python scripts/pre_render_notebooks.py docs/applications/4_osm_pipeline.py

# Re-render all Sionna application notebooks
python scripts/pre_render_notebooks.py --sionna

# Re-render all tutorial notebooks (requires downloaded scenario data)
python scripts/pre_render_notebooks.py --tutorials

# Convert to .ipynb stub without executing (no outputs — useful for syntax checks)
python scripts/pre_render_notebooks.py --no-execute docs/applications/2_sionna_rt_downstream.py
```

After rendering, commit both the `.py` source and the `.ipynb` output.
`.ipynb` files are gitignored by default (to prevent accidental large commits),
so you must force-add them:

```bash
git add -f docs/applications/4_osm_pipeline.ipynb
git commit -m "Pre-render OSM pipeline notebook"
```

#### Approximate render times

Times measured on an NVIDIA GPU workstation.  CPU-only machines will be
significantly slower for the Sionna notebooks.  Tutorials require downloaded
DeepMIMO scenarios (see [Getting Started](../quickstart.md)).

| Notebook | Requires | Approx. time |
|----------|----------|--------------|
| `tutorials/1_getting_started.py` | Scenario data | ~1 min |
| `tutorials/2_visualization.py` | Scenario data | ~1 min |
| `tutorials/3_channel_generation.py` | Scenario data | ~1 min |
| `tutorials/4_dataset_manipulation.py` | Scenario data | ~1 min |
| `tutorials/5_doppler_mobility.py` | Scenario data | ~2 min |
| `tutorials/6_beamforming.py` | Scenario data | ~2 min |
| `tutorials/7_converters.py` | Scenario data + Sionna | ~3 min |
| `tutorials/8_migration_guide.py` | Scenario data | ~1 min |
| `applications/1_channel_prediction.py` | Scenario data | ~2 min |
| `applications/2_sionna_rt_downstream.py` | `deepmimo[sionna]`, GPU | ~5–10 min |
| `applications/3_sionna_upstream.py` | `deepmimo[sionna]`, GPU | ~5–10 min |
| `applications/4_osm_pipeline.py` | `deepmimo[sionna]`, GPU, internet | ~10–15 min |

!!! note "Skipped notebooks"
    `tutorials/manual.py` is intentionally **not** pre-rendered by this
    script — it is too large and is maintained separately on Colab.


## Pull Request Process

1. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature-name
   ```

4. Open a Pull Request with:
   - Clear description of changes
   - Any related issues
   - Test coverage
   - Documentation updates

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms. 
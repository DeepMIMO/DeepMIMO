# Contributing

Thank you for your interest in contributing to DeepMIMO! This guide will help you get started.

## Development Setup

1. Fork and clone your forked the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/DeepMIMO.git
   cd DeepMIMO
   ```

2. Create a virtual environment (we recommend [miniforge](https://github.com/conda-forge/miniforge)):
   ```bash
   mamba create -n dm_env python=3.11
   mamba activate dm_env
   ```

3. Install development dependencies:
   ```bash
   pip install uv
   uv pip install -e ".[dev]"
   ```

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
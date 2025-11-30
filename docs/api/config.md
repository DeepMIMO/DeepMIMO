# Configuration

This module provides configuration management for DeepMIMO through a singleton configuration instance. 

The configuration interface is provided through a singleton instance `deepmimo.config`:

::: deepmimo.config.DeepMIMOConfig

## Basic Usage

```python
import deepmimo as dm

# Set with parameter names
dm.config('sionna_version', '0.19.1')

# Set with keywords
dm.config(use_gpu=True, gpu_device_id=1)

# Get current value
dm.config('ray_tracer_version')

# Print current config
dm.config()

# Reset to defaults
dm.config.reset()

```


## Default Configurations

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `wireless_insite_version` | Default Wireless InSite version | `3.3.0` |
| `sionna_version` | Default Sionna version | `0.19.1` |
| `aodt_version` | Default AODT version | `1.x` |
| `use_gpu` | Whether to use GPU acceleration | `False` |
| `gpu_device_id` | GPU device ID to use | `0` |
| `scenarios_folder` | Folder containing scenarios files | `'deepmimo_scenarios'` |

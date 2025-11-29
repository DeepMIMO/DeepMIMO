# Converter

The converter module provides functionality to automatically detect and convert raytracing data from various supported formats into a standardized DeepMIMO format.

```
converters/
  ├── converter.py (Main converter)
  ├── converter_utils.py (Helper functions)
  ├── aodt/ (AODT format)
  │   └── aodt_converter.py
  ├── sionna_rt/ (Sionna RT format)
  │   └── sionna_converter.py
  └── wireless_insite/ (Wireless InSite format)
      └── insite_converter.py
```

```{eval-rst}

.. autofunction:: deepmimo.converters.convert
   :noindex:
```

```python
import deepmimo as dm

# Convert raytracing data
scenario = dm.convert(
    path_to_rt_folder,  # Path to raytracing data
    **conversion_params  # Additional parameters
)
```

**Conversion Process**:

1. Format Detection
   ```python
   # Automatic format detection based on file extensions (internal logic)
   if '.aodt' in files:
       converter = aodt_rt_converter
   elif '.pkl' in files:
       converter = sionna_rt_converter
   elif '.setup' in files:
       converter = insite_rt_converter
   ```

2. Read Ray Tracing Parameters
3. Read TXRX Configuration
4. Read Paths 
   - Save DeepMIMO Core Matrices
5. Read Materials
6. Read Scene Objects
   - Save `vertices.mat` and `objects.json`
7. Save `params.json`

```{tip}
*Always* the auto converter that detects which converter should be used based on the contents of the folder. The conversion parameters should match one of the converters below. 
```


## Wireless InSite
```python
from deepmimo.converters import insite_rt_converter

# Convert Wireless InSite project
scenario = insite_rt_converter('path/to/insite_project')

# Required files:
# - <scen_name>.setup (Project setup)
# - <scen_name>.txrx (Project setup)
# - <scen_name>.xml (Project setup)
# - <p2m_folder>/*.paths (Ray paths)
# - <p2m_folder>/*.pl (Pathloss files)
```

```{tip}
Check <a href="../manual_full.html#from-wireless-insite">Conversion From Wireless InSite</a> section in DeepMIMO manual for a full end-to-end example of conversion from a Wireless InSite simulation.
```

```{eval-rst}

.. autofunction:: deepmimo.converters.wireless_insite.insite_converter.insite_rt_converter

```

## Sionna RT

Conversion from Sionna involves 2 steps: 
1. Exporting: Since Sionna RT does not save files, we must save them.
2. Converting: Use the saved files to create a DeepMIMO scenario.

### Exporting
```python
from deepmimo.exporters import sionna_exporter

sionna_save_folder = 'sionna_test_scen/'

sionna_exporter(
   scene,
   path_list,
   my_compute_path_params,
   save_folder=sionna_save_folder
)
```

```{eval-rst}

.. autofunction:: deepmimo.exporters.sionna_exporter.sionna_exporter

```

### Converting

```python
from deepmimo.converters import sionna_rt_converter
# Convert Sionna RT data
scenario = sionna_rt_converter(sionna_save_folder)

# Required files:
# - *.pkl (Scene data)
```

```{tip}
Check <a href="../manual_full.html#from-sionna-rt">Conversion From Sionna RT</a> section in DeepMIMO manual for a conversion example using Sionna 0.19. Sionna 1.0 support is coming in May 2025.
```

```{eval-rst}

.. autofunction:: deepmimo.converters.sionna_rt.sionna_converter.sionna_rt_converter

```

## AODT

Conversion from AODT (Aerial Optical Digital Twin) involves 2 steps:
1. Exporting: Export data from the AODT database to parquet files. This must be executed in a place with access to the clickhouse database used to store simulation results. In case of AODT On the Cloud, this should run in the jupyter notebook tab. 
2. Converting: Convert the exported parquet files to DeepMIMO format. This can be executed either along side the export code, or any place with access to the exported parquet files. For example, we can export the files on the cloud, zip them, download them, and convert them locally.

### Dependencies

AODT support requires additional dependencies. Install them using:
```bash
pip install --pre deepmimo[aodt]
```

### Exporting

```python
from clickhouse_driver import Client
from deepmimo.exporters import aodt_exporter

# Connect to AODT database
db_client = Client('clickhouse')

# Export database to parquet files
aodt_rt_folder = aodt_exporter(
    db_client,
    database='',  # Uses first available database if empty
    output_dir='.'  # Directory to save parquet files
)
```

```{eval-rst}

.. autofunction:: deepmimo.exporters.aodt_exporter.aodt_exporter

```

### Converting

```python
from deepmimo.converters import aodt_rt_converter

# Convert AODT data
scenario = aodt_rt_converter(aodt_rt_folder)

# Required files:
# - scenario.parquet (Scenario parameters)
# - raypaths.parquet (Ray paths and interactions)
# - cirs.parquet (Channel information)
# - rus.parquet, ues.parquet (TX/RX configurations)
# - materials.parquet (Material properties)
# - patterns.parquet (Antenna patterns)
# - time_info.parquet (Time information)
```

```{eval-rst}

.. autofunction:: deepmimo.converters.aodt.aodt_converter.aodt_rt_converter

```



# Converter

The converter module provides functionality to automatically detect and convert raytracing data from various supported formats into a standardized DeepMIMO format.

```
converter/
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

.. autofunction:: deepmimo.converter.convert
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
from deepmimo.converter.wireless_insite.insite_converter import insite_rt_converter

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

.. autofunction:: deepmimo.converter.wireless_insite.insite_converter.insite_rt_converter

```

## Sionna RT

Conversion from Sionna involves 2 steps: 
1. Exporting: Since Sionna RT does not save files, we must save them.
2. Converting: From the saved files

### Exporting
```python
from deepmimo.converter.sionna_rt import sionna_exporter

sionna_exporter.export_to_deepmimo(
   scene,
   path_list,
   my_compute_path_params,
   save_folder='sionna_folder/'
)
```

```{eval-rst}

.. autofunction:: deepmimo.converter.sionna_rt.sionna_exporter.export_to_deepmimo

```

### Converting

```python
from deepmimo.converter.sionna_rt.sionna_converter import sionna_rt_converter
# Convert Sionna RT data
scenario = sionna_rt_converter('path/to/sionna_data')

# Required files:
# - *.pkl (Scene data)
```

```{tip}
Check <a href="../manual_full.html#from-sionna-rt">Conversion From Sionna RT</a> section in DeepMIMO manual for a conversion example using Sionna 0.19. Sionna 1.0 support is coming in May 2025.
```

```{eval-rst}

.. autofunction:: deepmimo.converter.sionna_rt.sionna_converter.sionna_rt_converter

```

## AODT

### (under construction)

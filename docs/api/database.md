# Database

DeepMIMO API leverages a database which is globally available and provides an excellent distribution network for both uploads and downloads worldwide.

## Search

The search functionality allows you to find scenarios in the DeepMIMO database based on various parameters.

```python
import deepmimo as dm

# Define search parameters
query = {
    'bands': ['sub6', 'mmW'],
    'environment': 'outdoor',
    'numRx': {'min': 10e3, 'max': 10e5}
}

# Perform search
scenarios = dm.search(query)  # returns ['scenario1', 'scenario2', ...]

# Can be used later to download scenarios systematically
for scenario in scenarios:
    dm.download(scenario)
    
```

```{eval-rst}

.. autofunction:: deepmimo.api.search

```

## Download

Download scenarios from the DeepMIMO database to your local environment. 

```python
import deepmimo as dm

# Download a specific scenario
scenario_name = 'asu_campus_3p5'
download_path = dm.download(scenario_name)
print(f"Downloaded to: {download_path}")
```

```{eval-rst}

.. autofunction:: deepmimo.api.download

```

## Upload

The upload functionality allows you to contribute scenarios, images, and ray tracing sources to the DeepMIMO database.

### Get API Key

Some operations in the DeepMIMO Database API require an API key, which can be obtained from the [Contribute dashboard](https://deepmimo.net/dashboard) on the DeepMIMO website. You will need to create a Google account to access this dashboard. 

Currently only the upload functions require an API key. 

### Scenario

Upload a scenario to the DeepMIMO database.

```python
import deepmimo as dm

# Upload a scenario
scenario_name = 'my_scenario'
key = 'your_api_key'
dm.upload(scenario_name, key=key)
```

```{eval-rst}

.. autofunction:: deepmimo.api.upload

```

### Images

Upload additional images for a scenario.

```python
import deepmimo as dm

# Upload images
scenario_name = 'my_scenario'
key = 'your_api_key'
img_paths = ['image1.png', 'image2.png']
dm.upload_images(scenario_name, img_paths, key)
```

```{eval-rst}

.. autofunction:: deepmimo.api.upload_images

```

### RT Source

Upload ray tracing source files for a scenario.

```python
import deepmimo as dm

# Upload RT source
scenario_name = 'my_scenario'
rt_zip_path = 'path/to/rt_source.zip'
key = 'your_api_key'
dm.upload_rt_source(scenario_name, rt_zip_path, key)
```

```{eval-rst}

.. autofunction:: deepmimo.api.upload_rt_source

```


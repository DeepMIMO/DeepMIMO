"""DeepMIMO API Search Module.

This module provides functionality for searching scenarios in the DeepMIMO database.

Search flow:
1. Call search() with query dictionary containing search parameters
2. Send request to /api/search/scenarios endpoint
3. Return list of matching scenario names if successful
4. Use returned scenario names to download and load scenarios:
   for scenario_name in search(query):
       dm.download(scenario_name)
       dataset = dm.load(scenario_name)
"""

import json

import requests

from ._common import API_BASE_URL, REQUEST_TIMEOUT


def search(query: dict | None = None) -> list[str] | None:
    """Search for scenarios in the DeepMIMO database.

    Args:
        query: Dictionary containing search parameters from the following list:
            * bands (list[str]): Frequency bands ['sub6', 'mmW', 'subTHz']
            * raytracerName (str): Raytracer name or 'all'
            * environment (str): 'indoor', 'outdoor', or 'all'
            * numTx (dict): Numeric range filter {'min': number, 'max': number}
            * numRx (dict): Numeric range filter {'min': number, 'max': number}
            * pathDepth (dict): Numeric range filter {'min': number, 'max': number}
            * maxReflections (dict): Numeric range filter {'min': number, 'max': number}
            * numRays (dict): Numeric range filter {'min': number, 'max': number}
            * multiRxAnt (bool): Boolean filter or 'all' to ignore
            * multiTxAnt (bool): Boolean filter or 'all' to ignore
            * dualPolarization (bool): Boolean filter or 'all' to ignore
            * BS2BS (bool): Boolean filter or 'all' to ignore
            * dynamic (bool): Boolean filter or 'all' to ignore
            * diffraction (bool): Boolean filter or 'all' to ignore
            * scattering (bool): Boolean filter or 'all' to ignore
            * transmission (bool): Boolean filter or 'all' to ignore
            * digitalTwin (bool): Boolean filter or 'all' to ignore
            * city (str): City name text filter
            * bbCoords (dict): Bounding box coordinates
              {'minLat': float, 'minLon': float, 'maxLat': float, 'maxLon': float}
            * hasRtSource (bool): Boolean filter or 'all' to ignore.
              Note: Unlike other flags which are derived in the package during conversion,
              hasRtSource is set server-side when RT source is uploaded with upload_rt_source()

    Returns:
        Dict containing count and list of matching scenario names if successful, None otherwise

    """
    if query is None:
        query = {}
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/search/scenarios", json=query, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        return data["scenarios"]
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e!s}")
        if hasattr(e.response, "text"):
            try:
                error_data = e.response.json()
                print(f"Server error details: {error_data.get('error', e.response.text)}")
            except (ValueError, json.JSONDecodeError):
                print(f"Server response: {e.response.text}")
    except requests.exceptions.ConnectionError:
        print("Error: Connection failed. Please check your internet connection and try again.")
    except requests.exceptions.Timeout:
        print("Error: Request timed out. Please try again later.")
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e!s}")
    except ValueError as e:
        print(f"Error parsing response: {e!s}")
    except Exception as e:  # noqa: BLE001
        print(f"Unexpected error: {e!s}")

    return None


"""Geographic utilities for coordinate conversion and manipulation.

This module provides comprehensive utilities for working with geographic coordinates,
including conversions between different coordinate systems and geographic calculations.
It includes functionality for:

- Converting between GPS (lat/lon) and UTM coordinates
- Converting between meters and degrees
- Calculating distances between geographic points
- Converting between GPS and Cartesian coordinate systems
- Handling bounding box transformations
- Interacting with Google Maps APIs for city data and satellite imagery

The module is organized into three main sections:
1. Core geographic calculations and conversions
2. Coordinate system transformations
3. Google Maps API utilities

Variables:
    EARTH_RADIUS (float): Earth's radius in meters
    DEGREE_TO_METER (float): Conversion factor from degrees to meters at equator
"""

import os
from math import atan2, cos, radians, sin, sqrt

import numpy as np
import requests
import utm

# Constants
EARTH_RADIUS = 6371000  # meters
DEGREE_TO_METER = 111320  # approx. meters per degree at equator


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in meters using Haversine formula.

    The Haversine formula determines the great-circle distance between two points
    on a sphere given their latitudes and longitudes. This implementation assumes
    a spherical Earth with radius EARTH_RADIUS.

    Args:
        lat1 (float): First point latitude in degrees
        lon1 (float): First point longitude in degrees
        lat2 (float): Second point latitude in degrees
        lon2 (float): Second point longitude in degrees

    Returns:
        float: Distance between points in meters

    Example:
        >>> dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        >>> print(f"Distance between NY and LA: {dist/1000:.1f} km")

    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS * atan2(sqrt(a), sqrt(1 - a))


def meter_to_degree(meters: float, latitude: float) -> float:
    """Convert meters to approximate degrees at given latitude.

    This function provides an approximation of the degree equivalent of a given
    distance in meters at a specific latitude. The conversion accounts for the
    Earth's curvature, which causes the meter-to-degree ratio to vary with latitude.

    Args:
        meters (float): Distance in meters to convert
        latitude (float): Latitude at which to perform the conversion

    Returns:
        float: Approximate degrees corresponding to the given meters

    Note:
        The approximation becomes less accurate at extreme latitudes (near poles)
        and for very large distances.

    """
    return meters / (DEGREE_TO_METER * cos(radians(latitude)))


def xy_from_latlong(
    lat: float | np.ndarray,
    long: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert latitude and longitude to UTM coordinates.

    Transforms GPS coordinates to Universal Transverse Mercator (UTM) coordinates,
    which provide a flat representation of the Earth's surface in meters.

    Args:
        lat (float | np.ndarray): Latitude in degrees
        long (float | np.ndarray): Longitude in degrees

    Returns:
        Tuple[float | np.ndarray, float | np.ndarray]: Tuple containing:
            - x (easting): Distance in meters east from zone origin
            - y (northing): Distance in meters north from equator

    Note:
        UTM zones and hemisphere information are discarded. For high-precision
        applications where zone transitions matter, additional handling may be needed.

    """
    # utm.from_latlon() returns: (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
    x, y, *_ = utm.from_latlon(lat, long)
    return x, y


def convert_GpsBBox2CartesianBBox(
    minlat: float,
    minlon: float,
    maxlat: float,
    maxlon: float,
    origin_lat: float,
    origin_lon: float,
    pad: float = 0,
) -> tuple[float, float, float, float]:
    """Convert a GPS bounding box to a Cartesian bounding box.

    Transforms a geographic bounding box defined by min/max latitude and longitude
    into a local Cartesian coordinate system centered at the specified origin.

    Args:
        minlat (float): Minimum latitude in degrees
        minlon (float): Minimum longitude in degrees
        maxlat (float): Maximum latitude in degrees
        maxlon (float): Maximum longitude in degrees
        origin_lat (float): Origin latitude in degrees
        origin_lon (float): Origin longitude in degrees
        pad (float, optional): Padding to add to the bounding box in meters

    Returns:
        Tuple[float, float, float, float]: Tuple containing:
            - xmin: Minimum x coordinate in meters from origin
            - ymin: Minimum y coordinate in meters from origin
            - xmax: Maximum x coordinate in meters from origin
            - ymax: Maximum y coordinate in meters from origin

    Note:
        The resulting coordinates are in meters relative to the origin point,
        making them suitable for local area calculations and visualizations.

    """
    xmin, ymin = xy_from_latlong(minlat, minlon)
    xmax, ymax = xy_from_latlong(maxlat, maxlon)
    x_origin, y_origin = xy_from_latlong(origin_lat, origin_lon)

    xmin = xmin - x_origin
    xmax = xmax - x_origin
    ymin = ymin - y_origin
    ymax = ymax - y_origin

    return xmin - pad, ymin - pad, xmax + pad, ymax + pad


def convert_Gps2RelativeCartesian(
    lat: float | np.ndarray,
    lon: float | np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert GPS coordinates to relative Cartesian coordinates.

    Transforms GPS coordinates into a local Cartesian coordinate system
    centered at the specified origin point. Useful for local area calculations
    where a flat Earth approximation is acceptable.

    Args:
        lat (float | np.ndarray): Latitude in degrees
        lon (float | np.ndarray): Longitude in degrees
        origin_lat (float): Origin latitude in degrees
        origin_lon (float): Origin longitude in degrees

    Returns:
        Tuple[float | np.ndarray, float | np.ndarray]: Tuple containing:
            - x: Distance in meters east from origin
            - y: Distance in meters north from origin

    Note:
        For areas spanning more than a few kilometers, consider using a proper
        projection system to account for Earth's curvature.

    """
    x_origin, y_origin = xy_from_latlong(origin_lat, origin_lon)
    x, y = xy_from_latlong(lat, lon)

    return x - x_origin, y - y_origin


#############################################
# Google Maps API Utilities
#############################################


def get_city_name(lat: float, lon: float, api_key: str) -> str:
    """Fetch the city name from coordinates using Google Maps Geocoding API.

    Uses reverse geocoding to find the city name for given coordinates.
    Handles API errors gracefully and returns "unknown" if city cannot be determined.

    Args:
        lat (float): Latitude coordinate in degrees
        lon (float): Longitude coordinate in degrees
        api_key (str): Google Maps API key for authentication

    Returns:
        str: City name if found, "unknown" otherwise

    Note:
        Requires a valid Google Maps API key with Geocoding API enabled.
        API calls may incur charges depending on your Google Maps API plan.

    """
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lon}",
        "key": api_key,
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data["status"] == "OK":
            # Look for the city in the address components
            for result in data["results"]:
                for component in result["address_components"]:
                    if "locality" in component["types"]:  # 'locality' typically means city
                        return component["long_name"]
            return "unknown"  # Fallback if no city is found
        print(f"Geocoding error: {data['status']}")
        return "unknown"
    print(f"Geocoding request failed: {response.status_code}")
    return "unknown"


def fetch_satellite_view(
    minlat: float,
    minlon: float,
    maxlat: float,
    maxlon: float,
    api_key: str,
    save_dir: str,
) -> str | None:
    """Fetch a satellite view image of a bounding box using Google Maps Static API.

    Downloads and saves a satellite image for the specified geographic area.
    The image is centered on the bounding box with a fixed zoom level.

    Args:
        minlat (float): Minimum latitude in degrees
        minlon (float): Minimum longitude in degrees
        maxlat (float): Maximum latitude in degrees
        maxlon (float): Maximum longitude in degrees
        api_key (str): Google Maps API key for authentication
        save_dir (str): Directory to save the satellite view image

    Returns:
        Optional[str]: Path to the saved image file, or None if the request fails

    Note:
        - Image size is fixed at 640x640 pixels (maximum for free tier)
        - Zoom level is fixed at 18 (detailed view)
        - Requires a valid Google Maps API key with Static Maps API enabled
        - API calls may incur charges depending on your Google Maps API plan

    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Calculate the center of the bounding box
    center_lat = (minlat + maxlat) / 2
    center_lon = (minlon + maxlon) / 2

    # Parameters for the Static Maps API
    params = {
        "center": f"{center_lat},{center_lon}",
        "zoom": 18,  # Adjust zoom level (higher = more detailed)
        "size": "640x640",  # Image size in pixels (max 640x640 for free tier)
        "maptype": "satellite",  # Options: roadmap, satellite, hybrid, terrain
        "key": api_key,
    }

    # API endpoint
    STATIC_MAP_URL = "https://maps.googleapis.com/maps/api/staticmap"

    # Make the request
    response = requests.get(STATIC_MAP_URL, params=params)

    # Save the image in the specified directory
    if response.status_code == 200:
        image_path = os.path.join(save_dir, "satellite_view.png")
        with open(image_path, "wb") as f:
            f.write(response.content)
        print(f"Satellite view saved as '{image_path}'")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        image_path = None

    return image_path

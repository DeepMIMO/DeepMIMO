# AutoRayTracing

Provides 3 scripts for automating Ray Tracing simulations: 

- param_generator: turns the raw_params.csv into params.csv to facilitate large-scale simulations
    1. reads the map bounding boxes (4 coordinates: min/max lon/lat) from raw_params
    2. generates bboxes of smaller cells of size 500 x 500 meters (default)
    3. saves them to params.csv
- scene_creation: creates scenes for ray tracing (to be executed inside Blender)
    1. reads coordinates from params.csv
    2. fetches maps from OSM using those coordinates
    3. imports (only!) the buildings into Blender
    4. adds ground plane in accordance with coordinate range
    5. configures building materials
    6. exports scene in mitsuba format (.xml)
    7. writes scenes folder to scene_folder.txt
- ray_tracing_base: computes the ray tracing on the generated scenes (to be executed in an environment with Sionna)
    1. reads scene_folder.txt and params.csv
    2. loads a scene
    2. distributes users uniformely across the scene
    3. places the transmitter in a building close to the center of the scene
    4. compute ray tracing simulations with the configured parameters
    5. saves the results (paths)

# Requirements

## Blender (+ OSM plug-in & Mitsuba-blender plug-in)

- Download [Blender 3.3](https://download.blender.org/release/) (no install needed, it's ready to use!)
- Download the zip files of [OSM plug-in](https://prochitecture.gumroad.com/l/blender-osm) and [mitsuba-blender plug-in](https://github.com/mitsuba-renderer/mitsuba-blender/releases/tag/latest)
- Add (and enable after adding!) these zipfiles to Blender via Edit -> Preferences -> Addons -> Install
- In the mitsuba plug-in, make sure the dependencies are installed.
- Restart Blender
- Go into the Scripting Tab of Blender
- Run in the console:
    - `import pip`
    - `pip.main(['install', 'pandas', '--user'])`
    - `pip.main(['install', 'geopy', '--user'])`

## Python Environment with Sionna

For the time being, refer to the [install instructions from Sionna](https://nvlabs.github.io/sionna/installation.html)

If installing on Ubuntu, ask João for the latest install steps.

Other packages necessary: tqdm, geopy, scipy

# Execution: Watch the [step-by-step video](https://youtu.be/usxQ6gtEekY)


# Improvents

1. Install packages in Blender's Python

Current manual method:

```
import pip
pip.main(['install', 'pandas', '--user'])
```

2. Get Blender's python path with 

```
import sys
print(sys.executable)
```

Note: this can be used to automatically install packages in the python script using

```
import subprocess

python_path = "/path/to/your/python"
command = [python_path, "-m", "pip", "install", "some_package", "--user"]

try:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print("Output:", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error:", e.stderr)
```

3. To stop Blender execution, use: raise Exception("Stop here!")



## Need for more accurate geodesic distances

If more accurate distance functions are required in the future, here is an implementation of the Vincent formula that gives 0.01 mm errors at 10k km distances.

```
import math

# Constants for WGS-84 Ellipsoid (used by GPS systems)
WGS84_A = 6378137.0        # Semi-major axis, in meters
WGS84_B = 6356752.314245   # Semi-minor axis, in meters
WGS84_F = 1 / 298.257223563  # Flattening

def geodesic(point1, point2):
    # Unpack coordinates
    lat1, lon1 = map(math.radians, point1)
    lat2, lon2 = map(math.radians, point2)

    # Differences in coordinates
    delta_lon = lon2 - lon1

    # Auxiliary values
    U1 = math.atan((1 - WGS84_F) * math.tan(lat1))
    U2 = math.atan((1 - WGS84_F) * math.tan(lat2))
    sin_U1, cos_U1 = math.sin(U1), math.cos(U1)
    sin_U2, cos_U2 = math.sin(U2), math.cos(U2)

    # Iterative solution using Vincenty's formula
    lamb = delta_lon
    for _ in range(1000):  # Iterate up to 1000 times to converge
        sin_lamb, cos_lamb = math.sin(lamb), math.cos(lamb)
        sin_sigma = math.sqrt((cos_U2 * sin_lamb)**2 + (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lamb)**2)
        cos_sigma = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lamb
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cos_U1 * cos_U2 * sin_lamb / sin_sigma
        cos2_alpha = 1 - sin_alpha**2
        cos2_sigma_m = cos_sigma - 2 * sin_U1 * sin_U2 / cos2_alpha if cos2_alpha != 0 else 0
        C = WGS84_F / 16 * cos2_alpha * (4 + WGS84_F * (4 - 3 * cos2_alpha))
        lamb_prev = lamb
        lamb = delta_lon + (1 - C) * WGS84_F * sin_alpha * (
            sigma + C * sin_sigma * (cos2_sigma_m + C * cos_sigma * (-1 + 2 * cos2_sigma_m**2))
        )
        if abs(lamb - lamb_prev) < 1e-12:  # Convergence check
            break

    u_squared = cos2_alpha * (WGS84_A**2 - WGS84_B**2) / (WGS84_B**2)
    A = 1 + u_squared / 16384 * (4096 + u_squared * (-768 + u_squared * (320 - 175 * u_squared)))
    B = u_squared / 1024 * (256 + u_squared * (-128 + u_squared * (74 - 47 * u_squared)))
    delta_sigma = B * sin_sigma * (cos2_sigma_m + B / 4 * (
        cos_sigma * (-1 + 2 * cos2_sigma_m**2) -
        B / 6 * cos2_sigma_m * (-3 + 4 * sin_sigma**2) * (-3 + 4 * cos2_sigma_m**2)
    ))

    # Distance in meters
    distance = WGS84_B * A * (sigma - delta_sigma)
    return distance / 1000  # Return distance in kilometers

# Example usage
point1 = (41.49008, -71.312796)  # Newport, RI
point2 = (41.499498, -81.695391)  # Cleveland, OH

distance = geodesic(point1, point2)
print(f"Distance: {distance:.2f} km")
```

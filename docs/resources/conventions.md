# Conventions

This document details the key conventions used in DeepMIMO.

## Cartesian Coordinate System

The coordinate system follows:

- Origin (0,0,0) is typically at the center of the ground plane
- X and Y axes lie in the ground plane
- Z axis points upward
- Ground plane is at z=0
- Buildings and structures have positive z coordinates
- Ground plane is modeled as infinitesimally thin

## Angle Conventions 

Angles follow the spherical system conventions, see [Spherical Coordinate System](https://en.wikipedia.org/wiki/Spherical_coordinate_system). 

Additionally, all angles are in degrees (not radians)

- Azimuth angle (φ):
  - 0° points along positive x-axis
  - Increases counter-clockwise (looking from above, right-hand rule) in x-y plane
  - Range: [0°, 360°)

- Elevation angle (θ):
  - 0° is in x-y plane (Horizon)
  - 90° points along positive z-axis (Zenith)
  - -90° points along negative z-axis (Nadir)
  - Range: [-90°, 90°]

Note: This differs from the standard physics convention (where 0° is the zenith). DeepMIMO uses elevation from the horizon.

## Array Elements and Channel Indexing

When specifying array dimensions, elements are indexed and grow as follows:

- For 2D arrays, elements grow first along X (horizontal) then Y (vertical)
- Channel matrices use a flattened representation of the array elements
- For an MxN array, element (i,j) maps to index i*N + j in the flattened channel

![Array element indexing convention](../assets/images/array_indexing.png)



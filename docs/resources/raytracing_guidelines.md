# Ray Tracing Guidelines

*(under construction)*
---
General: 

Most raytracing simulations can be static. A car driving in the city or a drone flying in the air can be modeled as a set of BSs on the ground and a set of points along the trajectory of the car/drone, which are raytraced in parallel. Instead of a trajectory, it is more general to consider a grid of points with sufficiently small spacing between points, which will contain a much higher number of possible trajectories.

Some simulation guidelines:
- always choose isotropic antennas, since a specific pattern can be applied in post-processing
- 2 pol > 1 pol (VH > V)
- ideally, number of reflections >= 4, with diffraction and diffuse scattering ON.
- 


In Wireless Insite:

- The only 1 required outputs: paths
- For robustness: Use a single study area, antenna and waveform.
- Define as many <points> and <grids> as needed. 
- When defining a set of <points>, define a single location per set. 
- Ensure the txrx elements in your workspace have growing ids ("1,2,3.." and not "1 3 2")
- Keep the default VSWR of 1 (perfect antenna matching).
- scattering properties: do not enable "use reflection coefficient" - it will use 
  a method completely different than the other ray tracers and it's not advised


Left to test:
- When the reference of a grid is set to terrain, do the points heights change
  according to the terray underneath them? Or only according to the terrain
  on the corner of the grid? Hypothesis: the 2nd. 
  Limitation: our code considers the 2nd option and won't locate points properly.
- 


In Sionna:
- Be aware when using the isotropic antenna. The UE tilt must be 0. 
  Why? Because Sionna applies phase rotations even with the isotropic antenna. 
  It won't match the other raytracers for non-zero tilts and can give strange results.
- Ensure the compute_parameters were extracted.
- Compute a list of Paths, but ensure all txs and rxs are included in the scene when exporting.
- Current TX/RX position assumptions:
  - Multiple TX positions are supported (many-to-many scenarios)
  - All TX positions must be present in each paths dictionary (e.g., if using 3 TXs, 
    each paths dictionary must contain those same 3 TXs)
  - RX positions can vary across path dictionaries and are processed sequentially
    (e.g., first batch has RXs 0-9, second has 10-22, etc.)


In AODT:
- set the antenna pattern to ISOTROPIC

Aligning AODT and Sionna:
* RU position - AODT has an extra 1.25m. Need to add this to Sionna or remove from AODT
* UE orientations - default is 45º tilt (and sometimes not updated in DB)
* antenna - if ISO, make RU and UE orientations match (vertical). With halfwave dipole, no issue.
* tx power - not included in Sionna, but included in AODT
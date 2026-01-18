# Comparing Ray Tracers

!!! warning "Under Construction"
    This document is currently under active development. Content may be incomplete or subject to change as we continue to refine the comparison and gather additional information.

This document provides a comprehensive comparison of three prominent ray tracing tools: **Wireless Insite**, **Sionna**, and **AODT (Aerial Omniverse Digital Twin)**. The focus is purely on the ray tracing component rather than end-to-end wireless system simulation capabilities.

---


---

## Table of Contents

1. [Overview](#overview)
2. [Scope of Comparison](#scope-of-comparison)
3. [Logical Distribution of Functions](#logical-distribution-of-functions)
4. [Ray Tracing Capabilities](#ray-tracing-capabilities)
5. [Antenna Support](#antenna-support)
6. [Waveform Support](#waveform-support)
7. [Materials & Scattering](#materials-scattering)
8. [Software & Platform Details](#software-platform-details)
9. [Parameter Equivalence](#parameter-equivalence)
10. [Ray Tracer Caveats](#ray-tracer-caveats)
11. [Performance Comparison](#performance-comparison)

---

## Overview

### What This Comparison Covers

This comparison examines the similarities and differences across three ray tracers:

- **Wireless Insite (X3D)** - Commercial solution from Remcom
- **Sionna RT** - Open-source ray tracer from NVIDIA
- **AODT** - NVIDIA's Aerial Omniverse Digital Twin

### Key Philosophy

While all three tools have evolved to become more than simple ray tracers, this comparison focuses **purely on the ray tracing component**. Interface with wireless system simulation tools is beyond the scope.

---

## Scope of Comparison

### Ray Tracing Only, Not Wireless System Simulation

**Corollary 1**: Beamforming, standard compliance, hardware imperfections modeling, scheduler types, and other system-level capabilities don't matter beyond the possibility of performing all simulation in a single platform.

**Corollary 2**: Antenna patterns aren't that relevant as they can be applied in post-processing. It's always more general to do ray tracing with the isotropic element. However, antenna arrays matter for element-to-element ray tracing and whether the response is synthetic or ground truth per element.

**Corollary 3**: Waveforms don't matter. In all systems, the signal bandwidth is used purely for system simulations, not for ray tracing. The specific waveform matters only in the context of sensing and communications, not for finding rays and computing EM fields.

### Simulation Method vs Scale

| Method | Use Case | Simulation Scale | Complexity | Time for 1-meter scene |
|--------|----------|------------------|------------|------------------------|
| Maxwell | Antenna and Circuit Design | < 10 cm | Very high | 27 centuries |
| Optics | Light Propagation | 10 cm - 10 m | High | 1 day |
| Geometric | Radio Propagation | > 1 m | Medium | 5 seconds |

---

## Logical Distribution of Functions

Some decisions can't be made with just two ray tracers. Having three ray tracers helped bring clarity regarding the distribution of roles in the environment:

### 1. **txrx (antenna panel)**
- Mainly `bool(polarization / tx / rx)` and antenna specifics
- Assumes isotropic radiation pattern

### 2. **paths**
- rx/tx locations, angles of arrival/departure, powers/phases/delays
- Interaction types & locations
- Assumes electromagnetic propagation law of geometric raytracing
- Assumes orientations in post

### 3. **scene**
- Geometry, objects, movement → stuff that *has* a material

### 4. **materials**
- EM properties of materials

---

## Ray Tracing Capabilities

### Supported Propagation Mechanisms

| Ray Tracer | LoS | Reflection (max) | Diffraction (wedge/edge/max) | Diffuse Scattering (last bounce/any/keep all) | Refraction/Transmission (max) | Max No. of Paths | Max Depth | Algorithm | Angular Ray Launching Limits | Frequencies |
|------------|-----|------------------|------------------------------|-----------------------------------------------|-------------------------------|------------------|-----------|-----------|------------------------------|-------------|
| **Wireless Insite (X3D)** | ✅ | ✅ (30) | ✅ / ✅ / 3 | ✅ / ✅ / ❌ | ✅ (8) | ~100? | ~30 | SBR* | ✅ | 100 MHz - 100 GHz |
| **Sionna** | ✅ | ✅ (any) | ✅ / ✅✅ / 1* | ✅ / ❌ / ✅ | ❌ (0) | ~50? | any? | SBR & Exhaustive | ❌ | 100 MHz - 100 GHz |
| **AODT** | ✅ | ✅ (5) | ✅ / ✅ / 1 | ✅ / ❌ / ✅ | ✅ (5*) | ~256,000 | 10* | SBR | ❌ | 100 MHz - 100 GHz |

**Legend:**
- ✅✅ = configurable
- *SBR = Shooting and Bouncing Rays

**Note**: Frequencies are mostly bound by ITU materials

### Interaction Capabilities

Additional supported features:

| Feature | Wireless Insite | Sionna | AODT |
|---------|----------------|--------|------|
| `max_reflections` | 5 | `max_reflections` | `reflection` (= max depth if no scat, else = depth-1) |
| `max_diffractions` | 1 | `max_diffractions` | `int(diffraction)` |
| `max_scattering` | 1 | `max_scattering` | `int(scattering)` |
| `max_transmissions` | 0 | `max_transmissions` | - |
| `edge_diffraction` | - | - | `edge_diffraction` |
| `diffuse_reflections` | - | `diffuse_reflections` | `(depth - 1)` |
| `diffuse_diffractions` | - | `diffuse_diffractions` | `(False)` |
| `diffuse_transmissions` | - | `diffuse_transmissions` | - |
| `diffuse_final_interaction_only` | - | `diffuse_final_interaction_only` | `(True)` |
| `diffuse_random_phases` | - | `diffuse_random_phases` | `scat_random_phases` |
| `terrain_reflection` | TRUE | `terrain_reflection` | `(True)` |
| `terrain_diffraction` | TRUE | `terrain_diffraction` | `(False)` |
| `terrain_scattering` | FALSE | `terrain_scattering` | `(True)` |
| `scat_keep_prob` | - | - | `0.001` |
| `synthetic_array` | TRUE | - | `synthetic_array` |

---

## Antenna Support

### Antenna Types

| Ray Tracer | Isotropic | Omnidirectional | Dipole (Halfwave/infinitesimal/Config.) | Monopole | microstrip patch | tr38901 | Horn | Parabolic | Other | Array (Rectangular/Flexible) | Mutual coupling | Max size (single pol.) | Per element orientation |
|------------|-----------|-----------------|----------------------------------------|----------|------------------|---------|------|-----------|-------|------------------------------|-----------------|------------------------|------------------------|
| **Wireless Insite** | ✅ | ✅ | ✅ / ✅* / ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ / ✅ | ❌ | - | ❌ |
| **Sionna** | ✅* | ❌ | ✅ / ✅ / ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ / ✅ | ❌ | - | ❌ |
| **AODT** | ✅ | ❌ | ✅ / ✅ / ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ / ❌ | ✅* | 64 RU / 8 UE | ❌ |

### Additional Antenna Features

| Feature | Wireless Insite | Sionna | AODT |
|---------|----------------|--------|------|
| Custom Element Patterns | ✅ | ✅ | ✅ |
| Antenna Design Helper | ✅ | ❌ | ❌ |
| Antenna Array Visualizer | ✅ | ✅ | ✅ |
| Multi antenna elements | ✅ | ✅ | ✅ |
| Multiple txrx pairs per sim. (>2 diff panels) | ✅ | ❌ | ✅ |
| Different panels for UL/DL | ✅ | ❌ | ❌ |
| Synthetic array mode | ❌ | ✅ | ❌ |

### Polarization Support

| Polarization Type | Wireless Insite | Sionna | AODT |
|-------------------|----------------|--------|------|
| Single (V/H) | ✅ | ✅ | ✅ |
| Single (flexible) | ✅ | ✅ | ✅ |
| Dual (V/H) | ✅ | ✅ | ✅ |
| Dual (cross) | ✅ | ✅ | ✅ |

**Notes:**
- In Wireless Insite, the infinitesimal dipole can be obtained by configuring the length of the dipole to be very small
- In Sionna, the isotropic pattern is not exactly the idealized pattern as in literature - it still accounts for orientation and causes changes in vertical polarization
- In AODT, mutual coupling is only supported for halfwave dipoles
- Sionna has 2 polarization models: one only accounts for slant/tilt (model 2, simpler), the other accounts for both tilt and azimuth (model 1, more precise). Both follow 3GPP

---

## Waveform Support

| Ray Tracer | Sinusoid | Sinc | Chirp | Raised Cosine | Root RC | Tukey | Hanning | Hamming | Blackman | Gaussian | Custom | OFDM waveform & grid | Response per subcarrier | Use of Bandwidth in RT |
|------------|----------|------|-------|---------------|---------|-------|---------|---------|----------|----------|--------|----------------------|------------------------|------------------------|
| **Wireless Insite** | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Sionna** | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **AODT** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ? | ❌ |

**Note**: The waveforms don't really matter for ray tracing. We only care about:
- The ray tracing / output
- Whether bandwidth matters
- Whether any better computation is done by considering a resource grid (like adjustment of the responses at each subcarrier's frequency)

---

## Materials & Scattering

### Material Properties Support

Both ray tracers support ITU-R P.2040 materials with varying parameter naming conventions (see [Parameter Equivalence](#parameter-equivalence) section).

### Decoupling Ray Tracing and EM Phenomena

The issue is that some paths depend on the material, namely for scattering and refraction/transmission. This is good for research but not for maximum realism.

### Diffuse Scattering: Key Differences

The maximum number of scattering events supported by any ray tracer is **1**. All parties agree that no further depth than 1 is necessary. However, where they differ is at what stage in a path this scattering event can happen:

- **Sionna**: Only allows scattering in the last bounce after reflections
- **AODT**: Only allows one scattering, nothing else on the path profile
- **Wireless Insite**: Has an option for putting scattering in the last bounce only, or not. Plus, lets us configure how many of each event can happen until scattering. The number of possible combinations becomes *really* large when the scattering event is not enforced on the last bounce.

**Conclusion**: Wireless Insite is the most realistic and flexible in this regard. However, in THz, 2 scattering events may become relevant since scattering may become the main form of propagation.

---

## Software & Platform Details

### Platform Compatibility

| Feature | Wireless Insite | Sionna RT | AODT |
|---------|----------------|-----------|------|
| **OS Support** |
| Windows | ✅ | ✅* | ✅* |
| MacOS | ❌ | ✅ | ❌ |
| Linux | ✅* | ✅ | ✅* |
| **Hardware** |
| GPU supported | ✅ | ❌* | ✅ |
| Headless mode | ✅ | ✅ | ❌ |
| Min. Num. GPU | 1 | 1 | 2 |
| Min. GPU RAM | 2 GB | N/A | 12GB & 48GB |
| **Development** |
| Version | 3.3? | 0.19.1 | 1.2 |
| Open-source | ❌ | ✅ | ✅* |
| Language | ❔ | Python | C++ / Python |
| Active Development | ❔ | ✅ | ✅ |
| Release Cycle | ❔ | 4 / year | 3 / year |
| Forum | ❌ | ✅ | ✅ |
| First release | 2002 | 2023 | 2024 |
| Support latency | ~days | ~days | <12h |
| **Documentation** |
| Ray tracing algorithm | ✅ | ❌ | ❌ |
| EM principles | ✅ | ✅ | ❌ |
| Tips | ✅ | ❌ | ❌ |

**Notes on Compatibility:**
- Wireless Insite UI only works on Windows
- Sionna works on Windows via WSL2
- **AODT**: UI runs on Windows and Linux; Backend runs only on Linux; For single-GPU capabilities, the machine must use Linux currently
- **AODT open-source**: Yes, after getting accepted in NVIDIA's developer program (primarily a mechanism to understand who is working with the software; all developers from universities and industry get accepted)

### Language Details

**AODT:**
- UI: Python using the Omniverse Framework
- Backend: One backend in C++ and another in Python
- EM Solver (performance critical): C++

**Sionna:**
- Python leveraging GPU-accelerated frameworks like TensorFlow and DrJit
- Doesn't have a UI per se, but supports interactive ray tracing visualizations in Jupyter notebooks
- Requires knowledge of Python to use

**Wireless Insite:**
- Has comprehensive GUI tools including antenna design helpers and array visualizers
- Includes tips and heuristic guidelines on parameter setting, e.g.:
  - "0.2° ray spacing for areas smaller than 500m x 500m"
  - "6 reflections are usually sufficient, except in narrow streets or areas with highly reflective buildings"

---

## Parameter Equivalence

### Material Properties

| Category | Parameter | Typical Value | Wireless Insite | Sionna | AODT |
|----------|-----------|---------------|-----------------|--------|------|
| **Material** | name | ITU ... | `name` | `name` | `label` |
| | thickness | 0.03 | `thickness` | - | - |
| | permittivity | | `permittivity` | `relative_permittivity` | - |
| | conductivity | | `conductivity` | `conductivity` | - |
| | a,b,c,d | | - | - | `itu_r_p2040_{a/b/c/d}` |

### Scattering Model Properties

| Parameter | Wireless Insite | Sionna | AODT |
|-----------|-----------------|--------|------|
| scattering model | `diffuse_scattering_model` | `scattering_pattern` | (UI) |
| roughness | `roughness` (std = RMS) | - | `rms_roughness` |
| scattering coefficient | `fields_diffusively_scattered` | `scattering_coefficient` | `scattering_coeff` |
| cross polarization factor | `cross_polarized_power` | `xpd_coefficient` | `scattering_xpd` |
| alpha_i | `directive_alpha` | `alpha_i` | `exponent_alpha_i` |
| alpha_r | `directive_beta` | `alpha_r` | `exponent_alpha_r` |

---

## Ray Tracer Caveats

### Sionna Caveats

1. **Diffraction**: Only first-order diffraction is available. **And**, when accounting for diffractions, no other interaction along the path is possible (i.e., single-interaction path only)
2. **Scattering**: Only happens in the last bounce, and previous interactions can only be reflections
3. **Ray Tracing Methods**: Sionna has 2 types:
   - Uniform SBR in a Fibonacci sphere
   - Exhaustive (not compatible with scattering) but finds "all possible combinations of primitives" → might be possible to use it as a *ground truth!*

### AODT Caveats

1. Only specular reflection interactions along a ray with transmissions
2. Only direct diffuse scattering (single bounce and nothing else on the path - like Sionna with diffractions)
3. **Unbelievable scale**: Maximum number of paths limited to: 500 × num_ru_ele × num_ue_ele
   - If RU and UE have max elements, 64 × 8 = 512
   - 500 × 512 = **256,000 paths per user**
   - This assumes 1M rays emitted per RU and 10k users

### Wireless Insite Caveats

1. **Algorithm**: X3D uses SBR with "exact path calculation algorithm"
2. **Adjacent Path Generation (APG)**: Uses ray interpolation to determine exact paths to receiver based on rays that land close to it
   - This method relies on interpolation, so it can result in accuracy errors in some cases
   - For the same computational load, may help result in more accurate results
   - Reduces total computation load necessary to find a path that intersects the receiver sphere
   - Enablement is optional
3. **APG Parameters**:
   - *Adjacency distance*: Max distance between true receiver position and where coarse ray landed
   - *Number of paths*: Number of paths required in approximation (higher = more accurate interpolation)
4. **Path Validation**: Algorithm validates angles and blocking situations, discarding non-compliant paths
   - This *can* be a place for possible errors if diffuse scattering is ignored

### General Ray Tracing Characteristics

- **SBR requires intersecting rays with a receiver sphere**: Bigger spheres collect rays faster but may be less accurate

---

## Performance Comparison

### Key Observations

1. **Wireless Insite**: Tends to put numbers to maximum parameters (doesn't necessarily mean less capable - may be purely a development practice)
2. **AODT shines on**: Massive resolution per user (i.e., many, many paths)
3. **Wireless Insite shines on**: Massive depth (many diffractions, scattering, reflections)
4. **In terms of realism**: Depth beyond a certain point won't lead to more resolution, so AODT's approach may be better, although still lacking a bit on the basics

---

## Sources of Error: Digital Twin vs Real World

Even if a digital twin is PERFECT, there will be differences from the real world:

### Scene (Geometry)
- Geometry (e.g., size, shape, exact location of objects)
- LOD errors (not capturing all the right details and angles)

### Materials (EM Properties)
- Materials (conductivity, permittivity, roughness, thickness)

### Paths (EM Propagation)
- Imperfect ray tracing
- Imperfect propagation modeling

### Hardware
- Time offset
- Frequency offset
- Distortion in amplifiers
- Quantization in ADC
- IQ imbalance
- Filter imperfections

### Hard-to-Model Factors

**Human:**
- User device position and orientation error (and mobility)
- Non-modeling of blockages by other users
- Non-modeling of self-user blockage by hand, head, or body

**Channel Measurements:**
- Imperfect measurement (includes unknown noise)
- Channel asymmetry (most devices have more receive antennas than transmit antennas)
- Antenna channel mapping problem
- Frequency mapping problem

**Note**: The necessary realism of a digital twin should always be discussed with its purpose/utility in mind - what wireless task is the DT intended to help?

---

## Future Work & Open Questions

### Papers to Review

1. [Toward Real-Time Digital Twins of EM Environments: Computational Benchmark for Ray Launching Software](https://arxiv.org/pdf/2406.05042)
2. WiThRay: A Versatile Ray-Tracing Simulator for Smart Wireless Environments
3. The Design and Applications of High-Performance Ray-Tracing Simulation Platform for 5G and Beyond Wireless Communications: A Tutorial
4. A Survey of 5G Channel Measurements and Models
5. A Comprehensive Review of Efficient Ray-Tracing Techniques for Wireless Communication

### Topics for Further Analysis

**Performance (runtime) across:**
- Number of users in scene
- Number of triangles in scene

**Accuracy (comparison between ray tracers):**
- Per-path comparisons for different path types:
  - Phase diff, Amplitude diff, delay diff
- Number of paths found
- Resultant Channel MSE
- How does MSE change as we increase simulation parameters? (randomness off)
  - Should tell us which one is "closer" to ground truth
  - Check Sionna's "exhaustive" method
  - Check if Wireless Insite without APG is more precise

**Features analysis:**
- RT Method
- Paths discarded
- Open-source? GIS included? RAN included?

**Other considerations:**
- List other ray tracers and speculate why they are or aren't widely used
- Ask Sionna, AODT, Wireless Insite maintainers to review for accuracy
- List stuff not modelled by any ray tracer:
  - A) Because it's not included in ITU (e.g., magnetic losses)
  - B) Just because they didn't (e.g., frequency response along the band for a signal with given bandwidth)

---

## Disclaimers

*This comparison was created to help researchers and practitioners understand the trade-offs between different ray tracing solutions. It should be noted that this is a pragmatic modeling choice by all ray tracing engines. For transparency: information herein is accurate for the ray tracer versions considered at the time of writing.*

*Version information: Wireless Insite 3.3(?), Sionna 0.19.1, AODT 1.2*

---

**Last Updated**: 2026-01
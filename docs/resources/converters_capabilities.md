## Converter capability comparison

| Feature | AODT | Sionna RT | Wireless InSite |
|---|---|---|---|
| Multiple BS (TX points) | ✓ | ✓ [a] | ✓ |
| Multiple RX sets | ✗ [b] | ✗ [b] | ✓ |
| TX multi-antenna arrays | ✓ | ✓ [a] | ✗ [l] |
| RX multi-antenna arrays | ✓ | ✓ [a] | ✗ [l] |
| Dual polarization | ✓ [q] | ✓ [q] | ✓ [q] |
| Per-element positions (arrays) | ✓ | ✓ | ✗ [l] |
| Antenna pattern support | Isotropic-only [c] | N/A | Isotropic (via polarization) |
| BS–BS paths | ✗ | ✓ [k], [s] | ✗ |
| Reflections | ✓ | ✓ | ✓ |
| Diffractions | ✓ (≤1) | ✓ (≤1) | ✓ |
| Diffuse scattering | ✓ | ✓ (final only) | ✓ |
| Transmissions | ✓ | ✗ [e] | ✓ |
| Interaction positions export | ✓ | ✓ | ✓ [n] |
| Angles export (AoA/AoD) | ✓ [r] | ✓ | ✓ |
| Delay export | ✓ | ✓ | ✓ |
| Power export | ✓ | ✓ | ✓ |
| Phase export | ✓ | ✓ | ✓ |
| Path sorting by amplitude | ✗ | ✓ [t] | ✗ |
| Inactive RX detection | ✗ [g] | ✓ [g] | ✓ [g] |
| Dynamics/time-varying scenes | ✓ [f] | ✓ [f] | ✓ (API) [f] |
| Scene export | ✓ (disabled by default) [h] | ✓ | ✓ |
| Materials richness | ✓ (ITU-R P.2040) [i] | ✓ (scat. patterns) [i] | ✓ (incl. foliage) [i] |
| GPS bounding box | ✗ [j] | ✓ (when present) [j] | ✓ [j] |
| Terrain interaction flags | ✓ [o] | ✓ [o] | ✓ [o] |
| Uniform ray casting | ✓ | ✓ (mapped) | ✓ |
| Copy RT source files | ✓ [p] | ✓ [p] | ✓ [p] |
| Max path/interactions handled | ✓ | ✓ | ✓ |

### Notes
- [a] Sionna supports multiple TX points or multi-antenna, but not both simultaneously in one run; same constraint for RX. Single TX and RX sets in metadata.
- [b] AODT aggregates all UEs into one RX set; Sionna uses one RX set; InSite supports multiple RX sets.
- [c] AODT validates patterns: isotropic required; warns on halfwave dipole, errors on non-isotropic/custom.
- [e] Sionna does not support transmissions; AODT and InSite map transmission-like events.
- [f] AODT picks a time index for paths and RX positions from routes; Sionna exports one time slice of amplitudes; InSite runs are static per export but converter supports multi-scene foldering.
- [g] Inactive RXs: Sionna counts zero-amplitude paths; InSite uses 250 dB path-loss sentinel; AODT does not update active counts.
- [h] AODT scene reader exists (USD-like) but is disabled in the main converter flow; Sionna and InSite export scenes.
- [i] AODT computes permittivity/conductivity via ITU-R P.2040; Sionna maps RT scattering patterns; InSite includes EM, roughness/thickness, and foliage attenuation.
- [j] GPS bbox: Sionna includes when min/max lat/lon provided; InSite computes from study area + origin; AODT sets (0,0,0,0).
- [k] Sionna detects sources==targets and exports BS–BS paths.
- [l] InSite infers element count from polarization (Vertical/Horizontal=1, Both=2) but doesn’t export per-element geometry.
- [n] InSite interaction positions parsed from .p2m; Sionna from vertices; AODT from interaction point lists.
- [o] Terrain flags: AODT reflection/scattering true, diffraction false; Sionna per-RT flags; InSite per setup.
- [p] Copy sources: AODT (.parquet), Sionna (.pkl), InSite (.setup/.txrx/.city/.ter/.veg/.kmz).
- [q] Dual-pol: AODT via panel flag; Sionna via num_ant vs size; InSite via “Both” polarization.
- [r] AODT computes angles from geometry; Sionna provides angles; InSite from file fields.
- [t] Sionna sorts by path amplitude before truncation to MAX_PATHS. 


Note:
- AODT doesn't have scene.
- Only Sionna has proper multi-antenna support. Others support it via sep-TX.
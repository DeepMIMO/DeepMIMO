# Scenario studio

Generate furnished indoor scenes, ray trace them, and explore the result in 3D —
from a browser.

```bash
uv run scripts/scenario_dashboard/server.py     # http://127.0.0.1:8000
```

Two environments are needed because Infinigen pins `numpy<2` and Sionna needs
`numpy>=2`: tracing runs on a sibling `.venv-rt` (or the interpreter you
launched with, if it has Sionna), generating on a sibling `.venv-infinigen`.
Both are auto-detected — pass `--rt-python` or `--gen-python` only if yours live
elsewhere.

Each is probed at startup for the modules it needs, and the reason an
interpreter is unusable is shown on the run button rather than surfacing as a
traceback once a run is under way. The two are independent: with no Infinigen
you can still trace anything that already exists, and only *Generate a new
scene* is blocked.

## What it does

**Generate → trace → convert**, streamed as one run with a real progress bar.
Every parameter is exposed:

| panel | parameters |
|---|---|
| Source | generate, reuse a built scene, or rebuild an existing scenario |
| Generate scene | name, seed, room type, furniture density, solver presets |
| Geometry budget | per-class triangle budgets, minimum object size, open doorways |
| Ray tracing | carrier frequency, reflections, receiver spacing and height, transmitter count and height, rays per source, diffraction, explicit placement |

## Tracing without generating

Generation and tracing are independent. A scene is built once and is usually
traced many times — a frequency sweep, a different transmitter, more reflections
— so **Source** decides which stages run at all:

| source | stages | use it for |
|---|---|---|
| Generate a new scene | generate → export → trace | a scene that does not exist yet |
| A scene already built | trace | new RT parameters on the same geometry |
| …with *Re-export geometry* | export → trace | new triangle budgets or open doorways |
| A DeepMIMO scenario | rebuild → trace | re-tracing a scenario you already have |

**Source follows the scenario on screen.** Picking a scenario to look at also
picks it to trace, preferring the scene it was built from when that still
exists and falling back to rebuilding it. Generating is the expensive stage, so
it is never what a run does by default — it has to be chosen. The skipped panels
are greyed and labelled, the run button names the stages it will run, and the
line under it says whether anything will be generated. Trace-only runs start
their progress bar at zero rather than at the weight of the stages they skipped.

Any folder holding a `scene.xml` works, including one Blender or Sionna wrote:
pick it from the list, type its path, or start the server with
`--scene-root DIR` to list a whole folder of them.

**Rebuilding a scenario.** A converted scenario keeps its geometry and its
material constants, so it can be turned back into a Mitsuba scene without the
original Wireless InSite, Sionna or Blender source — that is what makes
re-tracing at a new frequency possible. The **Materials** control decides how
they are carried over:

- *ITU names where known* writes an `itu-radio-material`, so Sionna re-derives
  the constants at the new frequency. Only materials whose name matches an ITU
  class qualify.
- *Exactly as stored* keeps the scenario's own permittivity and conductivity.
  Those were measured at one frequency and are not re-derived, so a trace
  elsewhere reuses them; the run log says so, and names the materials affected.

The same thing from the command line:

```bash
uv run scripts/pipelines/infinigen_pipeline_runner.py from-scenario \
    city_37_seoul_3p5 /tmp/seoul_scene
uv run scripts/pipelines/infinigen_pipeline_runner.py trace \
    /tmp/seoul_scene --scenario seoul_28g --frequency 28e9
```

Progress is measured, not guessed. Infinigen announces how many greedy solver
stages a scene needs and logs each as it completes, so the bar is a true
fraction and the estimate comes from how long the finished stages took. This
matters more than it sounds: a full-fidelity run once sat on a nine-hour
trajectory with nothing in the output to reveal it.

**Runtime, measured on an M-series laptop.** `singleroom.gin` (`--single-room`)
is the dominant lever, not `fast_solve`: a furnished multi-room dwelling takes
about 5 minutes with it and hours without.

## Scene types

Infinigen ships one constraint script and it describes a dwelling, so what a
preset varies is the building's shape rather than its room vocabulary.

| preset | what changes |
|---|---|
| `home` | Infinigen's own dwelling — 3.0 m storeys |
| `tall_space` | 5.5 m storeys, for hall-like volumes |
| `compact` / `elongated` | near-square, or long and corridor-dominated |

Room kinds can be added to the mix but not swapped out, and adding one rarely
changes the layout — `scripts/pipelines/INFINIGEN.md` has the measurements and
what a genuine office building would take.

## Exploring

| input | action |
|---|---|
| drag | orbit |
| shift + drag | pan |
| wheel | dolly |
| click | trace the rays reaching that point |
| `F` | frame all |
| `1` / `3` / `7` | front / right / top |

**Look** switches between a white studio and a dark theme. Both are drawn with a
shadow map from the key light, hemispheric ambient, screen-space ambient
occlusion and contact edges; flat shading made rooms unreadable, because every
wall met every other wall at the same brightness. Turn the shading off for a
flat-colour view when comparing materials.

**Cut at height** clips above a plane — a converted interior is a closed box
from outside, so the section is what lets you see in.

**Placing devices.** The receiver grid defaults to the whole footprint and is
previewed live in 3D, so the spacing and height are visible before any time is
spent tracing. *Draw footprint* sets it by clicking two corners; *Whole
footprint* restores the default. Transmitters can be typed as `x,y,z ; x,y,z`
or placed with *Pick in 3D*, and both appear as markers in the preview.

Rays are drawn as ribbons expanded in screen space rather than as GL lines,
because `lineWidth` is capped at 1 pixel on essentially every platform — a
one-pixel path is invisible next to the geometry. **Ray thickness** sets their
width in pixels, and they are coloured violet through crimson to orange by
received power: a ramp through yellow looks hotter but disappears against the
studio background.

**Clicking** traces the propagation paths to the nearest receiver: transmitter →
interaction points → receiver, coloured by received power, with the strongest
path's power, delay and bounce count. Seeing which surfaces a path bounces off
is how you tell a plausible channel from a wrong one.

The legend lists materials with triangle counts; click to hide. Materials are
identified from object names where available and from permittivity and
conductivity otherwise, because converters disagree on naming — the Sionna path
numbers them `material_0` upward.

`?scenario=NAME&cut=1.4&rx=900` deep-links a view.

## Relationship to `scripts/deepmimo/dashboard`

That dashboard edits an Infinigen scene *before* tracing — load a `.blend`, move
objects, approve, trace. This one runs the whole pipeline and inspects the
converted scenario, for any producer, with no GLB bake. They compose.

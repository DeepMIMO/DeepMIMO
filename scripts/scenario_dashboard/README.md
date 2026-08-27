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
| Generate scene | name, seed, building preset, storey height, which rooms to furnish, furniture density, solver presets |
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

**Building** picks the preset; **Storey height** overrides it. **Furnish only**
is a different thing entirely — it restricts which rooms get furniture, and
leaves the rest of the building bare.

Room kinds can be added to the mix but not swapped out, and adding one rarely
changes the layout — `scripts/pipelines/INFINIGEN.md` has the measurements and
what a genuine office building would take.

## Stopping a run

The progress panel carries a **Stop** button while a job is running. Generation
can take hours, and until now the only way out was finding the process in a
terminal and killing it.

Stopping signals the whole process group, not just the child: the runner spawns
Blender, which spawns more, and signalling the child alone leaves those running.
The group gets SIGTERM, then SIGKILL five seconds later — Blender ignores a
polite request while it is inside a solver step. A stopped job ends as
*cancelled* rather than *failed*, so it reports no error.

Stopping during the trace stage keeps whatever was already written: if the
traced paths reached disk, **Convert traced paths** still appears.

## When a run dies

Tracing writes its Sionna output to the scene folder *before* converting, so a
conversion that fails costs the conversion and not the hour of tracing. A failed
run that left `sionna_paths.pkl` behind offers **Convert traced paths**, which
resumes from those files — three minutes against the eight it took to trace them.
From the command line that is `infinigen_pipeline_runner.py convert <scene folder>
--scenario NAME`.

Failures also name their cause rather than a number. A negative exit status is a
signal, and `exit code -15` says nothing — least of all that nothing in the
pipeline asked for it. It now reads *"was killed by SIGTERM — something outside
the pipeline stopped it: the machine under memory pressure, or the server being
restarted"*.

Conversion is the memory-hungry stage: it loads every traced path at once, and a
dense grid over a large building can produce hundreds of megabytes of them. If
runs keep dying there, widen **RX spacing** or draw a smaller receiver footprint
before reaching for anything else.

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

**Automatic placement** decides where transmitters go when you do not name
positions. *Coverage-aware* keeps only candidates that are genuinely indoors —
an upward ray must reach a ceiling at least 2.2 m above the floor, so the
underside of a table does not count — and clear of anything they would be
embedded in, hangs each below its **own** local ceiling, then picks the set
seeing the most receivers by greedy maximum coverage.

Candidates are then ranked by an estimated path loss rather than by whether they
can see a receiver — a room off a corridor is served through its doorway — and
each transmitter is the one that most improves the 90th percentile, the tail
where coverage holes live. `scripts/pipelines/INFINIGEN.md` has the model and how
it was validated against traced ground truth.

*Evenly spaced* is the old behaviour, kept for comparison. It divides the
bounding box, which is not the floor plate: a position can land inside a wall,
and asking for one more moves them all rather than adding one — which on one
apartment made three transmitters cover less than two.

**Ray colour** switches what a traced path's colour means. By *received power* a
single ramp runs violet through red to gold, strongest last. By *interaction
type* each segment takes the colour of what happened at the end of it, so a ray
visibly changes colour where it diffracts or passes through a wall — reflection,
diffraction and transmission each have their own hue, checked for colour-vision
separation against both the studio and dark backgrounds, and named in a legend
because the hues sit below 3:1 against the scene.

The material legend lists material **classes**, not instances. Sionna needs a
separate material instance per group — it merges shapes that share one — so a
41-object scene carries 41 materials of about five classes, and listing them
individually said "concrete" eight times to no purpose. Each row shows how many
instances it stands for and hides all of them at once.

The coverage panel is resizable: drag its lower-left corner.

**Coverage map** switches what the bottom-right panel shows, all for the same
receivers:

| tab | what it maps |
|---|---|
| power | received power per receiver, summed over paths in the linear domain |
| loss | path loss |
| LOS | line of sight, obstructed, or no path at all |
| delay | RMS delay spread — the second moment of the power delay profile |
| paths | how many paths reach each receiver |

A regular receiver grid is drawn as a field rather than as dots, so the rooms
are readable. Continuous maps clip their colour to the middle 98% — a handful of
deep-shadow receivers otherwise span half the scale on their own — and the title
reports the true extremes. **Coverage transmitter** picks one transmitter or the
best server, chosen per receiver on received power.

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
